import torch
from torch_geometric.loader import DataLoader
import numpy as np
import time
import os
from os import path as osp
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

# This file is essentially tailor-made for QM9
TASKS = [
    "mu",
    "alpha",
    "HOMO",
    "LUMO",
    "gap",
    "R2",
    "ZPVE",
    "U0",
    "U",
    "H",
    "G",
    "Cv",
    "Omega",
]
# QM9 y values were normalized, so we need to de-normalize.
CHEMICAL_ACC_NORMALISING_FACTORS = [
    0.066513725,
    0.012235489,
    0.071939046,
    0.033730778,
    0.033486113,
    0.004278493,
    0.001330901,
    0.004165489,
    0.004128926,
    0.00409976,
    0.004527465,
    0.012292586,
    0.037467458,
]
# Dropout is not used. MSE training, MAE for valid and test, with the above de-normalizing factors.


def train(model, loader, optimizer, loss_fun, device="cpu", y_idx=0):
    model.train()
    loss_all = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = loss_fun(model(data), data.y[:, y_idx : y_idx + 1]).to(device)  #
        loss.backward()
        loss_all += loss.item()
        optimizer.step()

    return loss_all / len(loader.dataset)


def val(model, loader, loss_fun, device="cpu", y_idx=0):
    model.eval()
    loss_all = 0

    for data in loader:
        data = data.to(device)
        loss_all += loss_fun(model(data), data.y[:, y_idx : y_idx + 1]).item()

    return loss_all / len(loader.dataset)


def test(model, loader, device="cpu", y_idx=0):
    model.eval()
    total_err = 0

    for data in loader:
        data = data.to(device)
        pred = torch.sum(torch.abs(model(data) - data.y[:, y_idx : y_idx + 1])).item()
        total_err += pred

    return total_err / (
        len(loader.dataset) * CHEMICAL_ACC_NORMALISING_FACTORS[y_idx]
    )  # Introduce norm factors

def get_num_graphs(loader):
    num = 0
    for l in loader:
        num += l.y.shape[0]
    return num

# Treat every target separately. So you're effectively training 13 times. <- their note not mine


def run_model_gr(
    model,
    dataset_tr,
    dataset_val,
    dataset_tst,
    batch_size=32,
    lr=0.0001,
    epochs=300,
    neptune_client=None,
    device="cpu",
    nb_reruns=5, # number of repeats
    specific_task=-1,
    run_name=time.strftime("%Y-%m-%d_%H%M"),
    args=None,
):
    run_id, seed = args.run_id, args.seed
    start_time = time.strftime("%m-%d_%H%M")
    loss_fun = torch.nn.MSELoss(
        reduction="sum"
    )  # Mean-Squared Loss is used for regression
    print("---------------- Training on provided split (QM9) ----------------")
    for y_idx, targ in enumerate(TASKS):  # Solve each one at a time...
        if 0 <= specific_task != y_idx:
            continue
        print("----------------- Predicting " + str(targ) + " -----------------")
        all_test_mae = np.zeros(nb_reruns,)
        all_val_mae = np.zeros(nb_reruns,)

        for rerun in range(nb_reruns):  # 5 Reruns for GR
            if run_id is not None and run_id != 'None':
                print('Using run id: %s' % run_id)
                # id = '%s-run_id' % run_id
                id = run_id
            else:
                print('Not using given run id: run_id = %s' % str(run_id))
                id = start_time
            seed_run = '%02d-%d' % (seed, rerun)
            logdir = osp.join('runs', run_name, str(targ), id, seed_run)
            writer = SummaryWriter(
                        log_dir=logdir)
            model.reset_parameters()
            writer.add_scalar('num_params', sum(p.numel() for p in model.parameters() if p.requires_grad), 0)
            writer.flush()
            with open(osp.join(logdir, "config.txt"), "w") as file:
                file.write(str(args).replace(",", ",\n" ) + '\ndevice: %s' % device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)  # Made static

            val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(dataset_tst, batch_size=batch_size, shuffle=False)
            train_loader = DataLoader(
                dataset_tr, batch_size=batch_size, shuffle=True
            )  # Shuffling is good here

            rerun_str = "QM9/" + str(targ) + "/rerun_" + str(rerun)
            print(
                "---------------- "
                + str(targ)
                + ": Re-run {} ----------------".format(rerun)
            )
            print('Saving runs to %s' % logdir)

            best_val_mse = 100000
            test_mae = 100
            best_val_mae = 100000

            # load from saved model if there is one present
            dictfiles = []
            for file in os.listdir(logdir):
                if file.endswith('.pt'):
                    dictfiles.append(file)
            if len(dictfiles) > 0:
                dictfile = sorted(dictfiles)[-1]
                start_epoch = int(dictfile[-6:-3])
                model.load_state_dict(torch.load(osp.join(logdir, dictfile)))
                print('model statedict file found at %s\nTraining from epoch %03d' % (logdir, start_epoch))
            else:
                start_epoch = 0

            for epoch in range(start_epoch, epochs + 1):
                if epoch % 50 == 0 and epoch != 0:
                    filepath = osp.join(logdir, 'model_e=%03d.pt' % epoch)
                    print('Saving model statedict file at %s' % filepath)
                    torch.save(model.state_dict(), filepath)
                start_t = time.time()
                # lr = scheduler.optimizer.param_groups[0]['lr']  # Same as GC
                train_mse = train(
                    model, train_loader, optimizer, loss_fun, device=device, y_idx=y_idx
                )
                if epoch % 10 == 0 or epoch == 1:
                    val_mse = val(model, val_loader, loss_fun, device=device, y_idx=y_idx)
                    # scheduler.step(val_mse_sum)
                    if best_val_mse >= val_mse:  # Improvement in validation loss
                        test_mae = test(model, test_loader, device=device, y_idx=y_idx)
                        best_val_mae = test(model, val_loader, device=device, y_idx=y_idx)
                        best_val_mse = val_mse

                    writer.add_scalar('train_mae', test(model, train_loader, device=device, y_idx=y_idx), epoch)
                    writer.add_scalar('val_mae', test(model, val_loader, device=device, y_idx=y_idx), epoch)
                    writer.add_scalar('test_mae', test_mae, epoch)
                    writer.add_scalar('train_loss', train_mse, epoch)
                    writer.flush()

                    if neptune_client is not None:
                        neptune_client[rerun_str + "/params/lr"].log(lr)
                        neptune_client[rerun_str + "/train/loss"].log(train_mse)
                        train_mae = test(model, train_loader, device=device, y_idx=y_idx)
                        neptune_client[rerun_str + "/train/MAE"].log(train_mae)
                        neptune_client[rerun_str + "/validation/loss"].log(val_mse)

                        val_mae = test(model, val_loader, device=device, y_idx=y_idx)
                        neptune_client[rerun_str + "/validation/MAE"].log(val_mae)
                        neptune_client[rerun_str + "/test/MAE"].log(test_mae)

                        model.log_hop_weights(neptune_client, rerun_str)

                    print(
                        "Epoch: {:03d}, LR: {:7f}, Train Loss: {:.7f}, "
                        "Val Loss: {:.7f}, Test MAE: {:.7f}, Time: {:.1f}".format(
                            epoch, lr, train_mse, val_mse, test_mae, time.time() - start_t
                        )
                    , flush=True)
                else:
                    writer.add_scalar('train_loss', train_mse, epoch)
                    writer.flush()
                    print(
                        "Epoch: {:03d}, LR: {:7f}, Train Loss: {:.7f}, Time: {:.1f}".format(
                            epoch, lr, train_mse, time.time() - start_t
                        )
                    , flush=True)

            all_test_mae[rerun] = test_mae
            all_val_mae[rerun] = best_val_mae

        avg_test_mae = all_test_mae.mean()
        avg_val_mae = all_val_mae.mean()

        std_test_mae = np.std(all_test_mae)
        std_val_mae = np.std(all_val_mae)
        # No need for averaging. This is 1 split anyway.

        if neptune_client is not None:
            neptune_client["QM9/" + str(targ) + "/validation/MAE"].log(avg_val_mae)
            neptune_client["QM9/" + str(targ) + "/test/MAE"].log(avg_test_mae)
            neptune_client["QM9/" + str(targ) + "/test/MAE_std"].log(std_test_mae)
            neptune_client["QM9/" + str(targ) + "/validation/MAE_std"].log(std_val_mae)

            torch.save(model, "../model.pt")
            neptune_client["QM9/" + str(targ) + "/model"].upload("model.pt")

        print("---------------- Final Result ----------------")
        print("Test -- Mean: " + str(avg_test_mae) + ", Std: " + str(std_test_mae))
        print("Validation -- Mean: " + str(avg_val_mae) + ", Std: " + str(std_val_mae))
        writer = SummaryWriter(
                        log_dir='runs/'+run_name+'/'+str(targ)+'/'+start_time+'/'+'agg')
        writer.add_scalar('avg_val_mae', avg_val_mae, epoch)
        writer.add_scalar('std_val_mae', std_val_mae, epoch)
        writer.add_scalar('avg_test_mae', avg_test_mae, epoch)
        writer.add_scalar('std_test_mae', std_test_mae, epoch)
        writer.flush()
