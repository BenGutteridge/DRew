import torch
from torch_geometric.loader import DataLoader
import time


avail_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, loader, optimizer, loss_fun):
    model.train()
    loss_all = 0

    for data in loader:
        data = data.to(avail_device)
        optimizer.zero_grad()
        loss = loss_fun(model(data), data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()

    return loss_all / len(loader.dataset)


def val(model, loader, loss_fun):
    model.eval()
    loss_all = 0

    for data in loader:
        data = data.to(avail_device)
        loss_all += data.num_graphs * loss_fun(model(data), data.y).item()

    return loss_all / len(loader.dataset)


def test(model, loader):
    model.eval()
    correct = 0

    for data in loader:
        data = data.to(avail_device)
        pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()

    return correct / len(loader.dataset)


def run_model_gc(
    model, dataset, splits, batch_size=32, lr=0.0001, epochs=300, neptune_client=None
):
    loss_fun = torch.nn.CrossEntropyLoss()
    acc = []
    acc_val = []
    for i, splits_dict in enumerate(splits):
        print("---------------- Split {} ----------------".format(i))
        split_str = "split_" + str(i)
        avg_test_acc = 0
        avg_val_acc = 0
        for rerun in range(3):
            model.reset_parameters()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=50, gamma=0.5
            )

            test_idxs = torch.tensor(splits_dict["test"], dtype=torch.long)
            train_idxs = torch.tensor(
                splits_dict["model_selection"][0]["train"], dtype=torch.long
            )
            val_idxs = torch.tensor(
                splits_dict["model_selection"][0]["validation"], dtype=torch.long
            )

            test_dataset = dataset[test_idxs]
            train_dataset = dataset[train_idxs]
            val_dataset = dataset[val_idxs]

            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )  # Shuffling is good here

            rerun_str = split_str + "/" + "rerun_" + str(rerun)
            print("---------------- Re-run {} ----------------".format(rerun))
            best_val_loss, test_acc = 100, 0
            best_val_acc = 0
            for epoch in range(1, epochs + 1):
                start_t = time.time()
                lr = scheduler.optimizer.param_groups[0]["lr"]
                train_loss = train(model, train_loader, optimizer, loss_fun)
                val_loss = val(model, val_loader, loss_fun)
                scheduler.step(val_loss)
                if best_val_loss >= val_loss:
                    test_acc = test(model, test_loader)
                    best_val_acc = test(model, val_loader)
                    best_val_loss = val_loss

                if neptune_client is not None:
                    neptune_client[rerun_str + "/params/lr"].log(lr)
                    neptune_client[rerun_str + "/train/loss"].log(train_loss)
                    neptune_client[rerun_str + "/train/accuracy"].log(
                        test(model, train_loader)
                    )
                    neptune_client[rerun_str + "/validation/loss"].log(val_loss)
                    neptune_client[rerun_str + "/validation/accuracy"].log(
                        test(model, val_loader)
                    )
                    neptune_client[rerun_str + "/test/accuracy"].log(test_acc)

                    model.log_hop_weights(neptune_client, rerun_str)

                print(
                    "Epoch: {:03d}, LR: {:7f}, Train Loss: {:.7f}, "
                    "Val Loss: {:.7f}, Test Acc: {:.7f}, Time: {:.1f}".format(
                        epoch, lr, train_loss, val_loss, test_acc, time.time() - start_t
                    )
                )

            avg_test_acc += test_acc
            avg_val_acc += best_val_acc

        test_acc = avg_test_acc / 3
        best_val_acc = avg_val_acc / 3

        acc.append(test_acc)
        acc_val.append(best_val_acc)

        if neptune_client is not None:
            neptune_client[split_str + "/test_acc"].log(test_acc)
            neptune_client[split_str + "/val_acc"].log(best_val_acc)

            neptune_client["results_mean"].log(torch.tensor(acc).mean())
            neptune_client["val_results_mean"].log(torch.tensor(acc_val).mean())

            if i > 0:
                neptune_client["results_std"].log(torch.tensor(acc).std())
                neptune_client["val_results_std"].log(torch.tensor(acc_val).std())

            torch.save(model, "../model.pt")
            neptune_client[split_str + "/model"].upload("model.pt")

    acc = torch.tensor(acc)
    acc_val = torch.tensor(acc_val)
    print("---------------- Final Result ----------------")
    print("Test -- Mean: {:7f}, Std: {:7f}".format(acc.mean(), acc.std()))
    print("Validation -- Mean: {:7f}, Std: {:7f}".format(acc_val.mean(), acc_val.std()))
