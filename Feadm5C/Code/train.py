import data_load
import model
import torch
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
import pandas as pd

from visualization import tsne_visualization


def train(net, loaders, learning_rate, n_epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    train_acc_all = []
    val_acc_all = []
    test_acc_all = []
    train_loss_all = []
    val_loss_all = []
    test_loss_all = []
    train_mean_loss_all = []
    val_mean_loss_all = []
    test_mean_loss_all = []
    train_mcc_all = []
    test_mcc_all = []
    train_sn_all = []
    test_sn_all = []
    train_sp_all = []
    test_sp_all = []
    train_auc_all = []
    test_auc_all = []
    for epoch in range(n_epochs):
        print("-----Epoch %s-----" % (epoch + 1))
        for j in range(10):
            train_x = []
            train_y = []
            test_x = []
            test_y = []
            [train_load, val_load, test_load] = loaders[j]
            net.train()
            loss_fn = torch.nn.BCELoss()
            opti = torch.optim.Adam(net.parameters(), lr=learning_rate)
            train_loss = 0
            train_all_label = []
            train_all_temp_label = []
            train_all_pred = []
            for i, data in enumerate(train_load, 0):
                inputs, labels = data
                inputs1 = inputs[:, 0, :, :, :].reshape(-1, 41, 11 * 17)
                inputs2 = inputs[:, 1, :, :, :].reshape(-1, 41, 11, 17)
                # inputs1 = inputs[:, 0, :, :, :].reshape(-1, 41, 11, 17)
                # inputs2 = inputs[:, 1, :, :, :].reshape(-1, 41, 11, 17)
                inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)
                # inputs = inputs.reshape(-1, 41, 11 * 17)
                labels = torch.tensor(labels, dtype=torch.float)
                _, outputs = net(inputs1, inputs2)
                train_x.extend(_.tolist())
                train_y.extend(labels.int().tolist())
                train_all_label.extend(labels.tolist())
                train_all_pred.extend(outputs.tolist())
                train_all_temp_label.extend([1 if i > 0.5 else 0 for i in outputs.tolist()])
                loss = loss_fn(outputs, labels)
                train_loss += loss.data
                opti.zero_grad()
                loss.backward()
                opti.step()

            train_tn, train_fp, train_fn, train_tp = confusion_matrix(train_all_label, train_all_temp_label).ravel()
            train_auc = metrics.roc_auc_score(train_all_label, train_all_pred)
            train_mcc = metrics.matthews_corrcoef(train_all_label, train_all_temp_label)
            train_acc = metrics.accuracy_score(train_all_label, train_all_temp_label)
            train_sn = train_tp / (train_tp + train_fn)
            train_sp = train_tn / (train_tn + train_fp)
            train_acc_all.append(train_acc)
            train_loss_all.append(train_loss.item())
            train_sn_all.append(train_sn)
            train_sp_all.append(train_sp)
            train_auc_all.append(train_auc)
            train_mcc_all.append(train_mcc)
            train_mean_loss = train_loss / len(train_load)
            train_mean_loss_all.append(train_mean_loss.item())
            print("-----Fold  %s-----" % (j + 1))
            print(
                "Train:    mean loss: {6:.4f}    acc: {1:.4f}    "
                "sn: {2:.4f}    sp: {3:.4f}    mcc: {4:.4f}    auc: {5:.4f}    loss: {0:.4f}"
                .format(train_loss.item(), train_acc,
                        train_sn, train_sp,
                        train_mcc, train_auc, train_mean_loss))

            val_loss = 0
            val_all_label = []
            val_all_pred = []
            val_all_temp_label = []
            net.eval()
            with torch.no_grad():
                for i, data in enumerate(val_load, 0):
                    inputs, labels = data
                    inputs1 = inputs[:, 0, :, :, :].reshape(-1, 41, 11 * 17)
                    inputs2 = inputs[:, 1, :, :, :].reshape(-1, 41, 11, 17)
                    # inputs1 = inputs[:, 0, :, :, :].reshape(-1, 41, 11, 17)
                    # inputs2 = inputs[:, 1, :, :, :].reshape(-1, 41, 11, 17)
                    inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)
                    # inputs, labels = inputs.to(device), labels.to(device)
                    labels = torch.tensor(labels, dtype=torch.float)
                    _, outputs = net(inputs1, inputs2)
                    val_all_label.extend(labels.tolist())
                    val_all_temp_label.extend([1 if i > 0.5 else 0 for i in outputs.tolist()])
                    val_all_pred.extend(outputs.tolist())
                    loss = loss_fn(outputs, labels)
                    val_loss += loss.data
                val_tn, val_fp, val_fn, val_tp = confusion_matrix(val_all_label, val_all_temp_label).ravel()
                val_auc = metrics.roc_auc_score(val_all_label, val_all_pred)
                val_mcc = metrics.matthews_corrcoef(val_all_label, val_all_temp_label)
                val_acc = (val_tn + val_tp) / (val_tn + val_tp + val_fn + val_fp)
                val_acc_all.append(val_acc)
                val_loss_all.append(val_loss.item())
                val_mean_loss = val_loss / len(val_load)
                val_mean_loss_all.append(val_mean_loss.item())
                print(
                    "Val:      mean loss: {6:.4f}    acc: {1:.4f}    "
                    "sn: {2:.4f}    sp: {3:.4f}    mcc: {4:.4f}    auc: {5:.4f}    loss: {0:.4f}"
                    .format(val_loss.item(), val_acc,
                            val_tp / (val_tp + val_fn), val_tn / (val_tn + val_fp),
                            val_mcc, val_auc, val_mean_loss))

            test_loss = 0
            test_all_label = []
            test_all_pred = []
            test_all_temp_label = []
            net.eval()
            with torch.no_grad():
                for i, data in enumerate(test_load, 0):
                    inputs, labels = data
                    inputs1 = inputs[:, 0, :, :, :].reshape(-1, 41, 11 * 17)
                    inputs2 = inputs[:, 1, :, :, :].reshape(-1, 41, 11, 17)
                    # inputs1 = inputs[:, 0, :, :, :].reshape(-1, 41, 11, 17)
                    # inputs2 = inputs[:, 1, :, :, :].reshape(-1, 41, 11, 17)
                    inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)
                    # inputs, labels = inputs.to(device), labels.to(device)
                    labels = torch.tensor(labels, dtype=torch.float)
                    _, outputs = net(inputs1, inputs2)
                    test_x.extend(_.tolist())
                    test_y.extend(labels.int().tolist())
                    test_all_label.extend(labels.tolist())
                    test_all_temp_label.extend([1 if i > 0.5 else 0 for i in outputs.tolist()])
                    test_all_pred.extend(outputs.tolist())
                    loss = loss_fn(outputs, labels)
                    test_loss += loss.data
                test_tn, test_fp, test_fn, test_tp = confusion_matrix(test_all_label, test_all_temp_label).ravel()
                test_auc = metrics.roc_auc_score(test_all_label, test_all_pred)
                test_mcc = metrics.matthews_corrcoef(test_all_label, test_all_temp_label)
                test_acc = (test_tn + test_tp) / (test_tn + test_tp + test_fn + test_fp)
                test_acc_all.append(test_acc)
                test_sn = test_tp / (test_tp + test_fn)
                test_sp = test_tn / (test_tn + test_fp)
                test_loss_all.append(test_loss.item())
                test_mean_loss = test_loss / len(test_load)
                test_mean_loss_all.append(test_mean_loss.item())
                test_sn_all.append(test_sn)
                test_sp_all.append(test_sp)
                test_auc_all.append(test_auc)
                test_mcc_all.append(test_mcc)
                print("Test:     mean loss: {6:.4f}    acc: {1:.4f}    "
                      "sn: {2:.4f}    sp: {3:.4f}    mcc: {4:.4f}    auc: {5:.4f}    loss: {0:.4f}"
                      .format(test_loss.item(), test_acc,
                              test_sn, test_sp,
                              test_mcc, test_auc, test_mean_loss))
                if test_acc >= 0.74 and train_acc >= 0.74:
                    tsne_visualization(train_x, train_y, test_x, test_y)
        print('=' * 100, "\n\n")

    df = pd.DataFrame({"train loss": train_loss_all,
                       "val loss": val_loss_all,
                       "test loss": test_loss_all,
                       "train acc": train_acc_all,
                       "val acc": val_acc_all,
                       "test acc": test_acc_all,
                       "train mean loss": train_mean_loss_all,
                       "val mean loss": val_mean_loss_all,
                       "test_mean_loss": test_mean_loss_all,
                       "train auc": train_auc_all,
                       "train mcc": train_mcc_all,
                       "train sn": train_sn_all,
                       "train sp": train_sp_all,
                       "test auc": test_auc_all,
                       "test mcc": test_mcc_all,
                       "test sn": test_sn_all,
                       "test sp": test_sp_all})
    file_name = "XXX" + str(learning_rate) + ".xlsx"
    df.to_excel(file_name)


if __name__ == "__main__":
    # res_net = model.ResNet(model.BasicBlock, [2])
    # cnn_model = model.CNN(32, 32)
    bilstm = model.BiLSTM(11 * 17, 256, 2)
    # kmer_bilstm = model.kmerBiLSTM(11*17, 64, 2)
    # resnetbilstm = model.ResNetBiLSTM(model.BasicBlock, [2], 1, 256, 2)
    # mlp = model.MLP()
    loader = data_load.data_loader()
    # train(mlp, loader, 0.005, 10)
    # train(resnetbilstm, loader, 0.0005, 10)
    # train(cnn_model, loader, 0.0001, 10)
    train(bilstm, loader, 0.001, 10)
    # train(kmer_bilstm, loader, 0.005, 10)
    # train(res_net, loader, 0.00001, 10)
