import numpy as np
import torch
from torch_geometric.loader import DataLoader
from utils.dataset_loader import ogb_metric

avail_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_crit = {
    "ogbg-molhiv": torch.nn.BCEWithLogitsLoss(reduction="none"),
    "ogbg-molbbbp": torch.nn.BCEWithLogitsLoss(reduction="none"),
    "ogbg-molbace": torch.nn.BCEWithLogitsLoss(reduction="none"),
    "ogbg-molclintox": torch.nn.BCEWithLogitsLoss(reduction="none"),
    "ogbg-moltox21": torch.nn.BCEWithLogitsLoss(reduction="none"),
    "ogbg-moltoxcast": torch.nn.BCEWithLogitsLoss(reduction="none"),
    "ogbg-molsider": torch.nn.BCEWithLogitsLoss(reduction="none"),
    "ogbg-molmuv": torch.nn.BCEWithLogitsLoss(reduction="none"),
    "ogbg-molpcba": torch.nn.BCEWithLogitsLoss(reduction="none"),
    "ogbg-ppa": torch.nn.CrossEntropyLoss(reduction="mean"),
}  # All these datasets use single splits...


@torch.no_grad()
def test(model, loader, evaluator, metric, dataset_name=None):
    if metric != "F1":
        model.eval()
        predictions = []
        correct = []
        for data in loader:
            y_pred = model(data)
            if dataset_name == "ogbg-ppa":
                predictions.append(
                    torch.argmax(y_pred, dim=1).view(-1, 1).cpu().numpy()
                )
                correct.append(data.y.view(-1, 1).cpu().numpy())
            else:
                predictions.append(y_pred.cpu().numpy())
                correct.append(data.y.cpu().numpy())

        if dataset_name == "ogbg-ppa":
            correct = np.concatenate(correct)
            predictions = np.concatenate(predictions)

        correct = np.concatenate(correct, axis=0)
        predictions = np.concatenate(predictions, axis=0)
        out_metric = evaluator.eval({"y_true": correct, "y_pred": predictions})[metric]
        return out_metric
    else:
        raise NotImplementedError()


def train(model, loader, loss_fun, optimizer, device=avail_device):
    model.train()
    loss_all = 0

    for data in loader:  # Just like the standard GC
        data = data.to(device)
        optimizer.zero_grad()

        if type(loss_fun) == torch.nn.modules.loss.BCEWithLogitsLoss:
            prediction = model(data).to(torch.float64)
            y = data.y.view(prediction.shape).to(
                torch.float64
            )  # There could be nans in here, so clean them up

            # Whether y is non-null or not.
            mask = torch.logical_not(torch.isnan(y))
            y = y[mask]  # Don't feed the nan's, avoid them if possible
            prediction = prediction[mask]
            # Loss matrix
            loss_mat = loss_fun(prediction, y)
            # loss matrix after removing null target
            loss = torch.mean(loss_mat)
            loss_all += data.num_graphs * loss.item()
            loss.backward()

            # New, gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        elif type(loss_fun) == torch.nn.modules.loss.CrossEntropyLoss:
            loss = loss_fun(model(data).to(torch.float32), data.y.view(-1,))
            loss.backward()
            loss_all += loss.item()
            optimizer.step()
        else:
            raise ValueError("Invalid loss for OGB.")

    return loss_all / len(loader)


def val(model, loader, loss_fun, device=avail_device):
    model.eval()
    loss_all = 0
    for data in loader:
        data = data.to(device)
        if type(loss_fun) == torch.nn.modules.loss.BCEWithLogitsLoss:
            prediction = model(data).to(
                torch.float64
            )  # We need the careful loss computation here too
            y = data.y.view(prediction.shape).to(
                torch.float64
            )  # There could be nans in here, so clean them up

            # Whether y is non-null or not.
            mask = torch.logical_not(torch.isnan(y))
            y = y[mask]  # Don't feed the nan's, avoid them if possible
            prediction = prediction[mask]
            # Loss matrix
            loss_mat = loss_fun(prediction, y)
            # loss matrix after removing null target
            loss = torch.sum(loss_mat) / torch.sum(1 * mask)
            loss_all += data.num_graphs * loss.item()
        elif type(loss_fun) == torch.nn.modules.loss.CrossEntropyLoss:
            loss = loss_fun(model(data).to(torch.float32), data.y.view(-1,))
            loss_all += loss.item()
        else:
            raise ValueError("Invalid loss for OGB.")

    return loss_all / len(loader)


def run_model_gc_ogb(
    model,
    dataset,
    dataset_name,
    batch_size,
    evaluator,
    nb_reruns=10,
    lr=0.0001,
    epochs=300,
    neptune_client=None,
    device=avail_device,
):
    test_results = []
    for i in range(nb_reruns):
        model.reset_parameters()
        split_idx = dataset.get_idx_split()
        loss_fun = loss_crit[dataset_name]  # Determine loss function
        metric = ogb_metric[dataset_name]
        train_loader = DataLoader(
            dataset[split_idx["train"]], batch_size=batch_size, shuffle=True
        )
        valid_loader = DataLoader(
            dataset[split_idx["valid"]], batch_size=batch_size, shuffle=False
        )
        test_loader = DataLoader(
            dataset[split_idx["test"]], batch_size=batch_size, shuffle=False
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        best_val_metric = -1000000
        results = {
            "rerun_" + str(i) + "/" + "final_train": 0,
            "rerun_" + str(i) + "/" + "final_valid": 0,
            "rerun_" + str(i) + "/" + "final_test": 0,
        }

        for epoch in range(1, epochs + 1):
            epoch_loss = train(model, train_loader, loss_fun, optimizer, device=device)
            validation_loss = val(model, valid_loader, loss_fun, device=device)

            train_metric = test(
                model, train_loader, evaluator, metric
            )  # Log these in an online fashion.
            valid_metric = test(model, valid_loader, evaluator, metric)
            if valid_metric > best_val_metric:  # Validation is by metric
                results["rerun_" + str(i) + "/" + "final_valid"] = valid_metric
                results["rerun_" + str(i) + "/" + "final_train"] = train_metric
                test_metric = test(
                    model, test_loader, evaluator, metric, dataset_name=dataset_name
                )  # Only compute when best is reached (cleaner)
                results["rerun_" + str(i) + "/final_test"] = test_metric
                best_val_metric = valid_metric

            if neptune_client:
                neptune_client[
                    dataset_name + "_rerun_" + str(i) + "/" "/train_loss"
                ].log(epoch_loss)
                neptune_client[
                    dataset_name + "_rerun_" + str(i) + "/" "/train_" + str(metric)
                ].log(train_metric)
                neptune_client[
                    dataset_name + "_rerun_" + str(i) + "/" "/validation_" + str(metric)
                ].log(valid_metric)
                neptune_client[
                    dataset_name + "_rerun_" + str(i) + "/" "/validation_loss"
                ].log(validation_loss)
                neptune_client[
                    dataset_name + "_rerun_" + str(i) + "/" "/test_" + str(metric)
                ].log(results["rerun_" + str(i) + "/final_test"])

                model.log_hop_weights(neptune_client, "model")

        test_results.append(results["rerun_" + str(i) + "/" + "final_test"])
        results_as_np = np.array(test_results)
        mean = np.mean(results_as_np)
        st_dev = np.std(results_as_np)
        neptune_client[dataset_name + "/test_" + str(metric) + "_mean"].log(mean)
        neptune_client[dataset_name + "/test_" + str(metric) + "_std"].log(st_dev)
    if neptune_client:
        for key, v in results.items():
            neptune_client[key].log(v)
