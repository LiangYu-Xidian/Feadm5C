import pandas as pd
import numpy as np

import torch.utils.data
from sklearn.model_selection import KFold
import feature


def read_fasta(fasta_file):
    try:
        fp = open(fasta_file)
    except IOError:
        print('cannot open ' + fasta_file + ', check if it exists!')
        exit()
    else:
        fp = open(fasta_file)
        lines = fp.readlines()

        fasta_dict = {}  # record seq for one id
        idlist = []  # record id list sorted
        gene_id = ""
        idx = 1
        for line in lines:
            line = line.replace('\r', '')
            if line[0] == '>':
                if gene_id != "":
                    fasta_dict[gene_id] = seq.upper()
                    idlist.append(gene_id)
                seq = ""
                gene_id = line.strip('\n') + str(idx)  # line.split('|')[1] all in > need to be id
                idx += 1
            else:
                seq += line.strip('\n')

        fasta_dict[gene_id] = seq.upper()  # last seq need to be record
        idlist.append(gene_id)

    return fasta_dict, idlist


def get_sequence_odd_fixed(fasta_dict, idlist, focus_amino='C', window=20, label=1):
    seq_list_2d = []
    id_list = []
    pos_list = []
    focus_list = []
    for id in idlist:  # for sort
        seq = fasta_dict[id]
        final_seq_list = [label] + [AA for AA in seq]

        id_list.append(id)
        pos_list.append(window)
        focus_list.append(focus_amino)
        seq_list_2d.append(final_seq_list)

    df = pd.DataFrame(seq_list_2d)
    df2 = pd.DataFrame(id_list)
    df3 = pd.DataFrame(pos_list)
    df4 = pd.DataFrame(focus_list)

    return df, df2, df3, df4


def analyseFixedPredict(fasta_file, focus_amino='C', window=20, label=1):
    fasta_dict, idlist = read_fasta(fasta_file)

    sequence, ids, poses, focus = get_sequence_odd_fixed(fasta_dict, idlist, focus_amino, window, label)

    return sequence, ids, poses, focus


class m5CDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)


def data_loader(root="../data/A.thaliana/A.thaliana5289_pos.fasta"):
    fasta_dict_train_pos, id_list_pos = read_fasta(root)
    fasta_dict_train_neg, id_list_neg = read_fasta("../data/A.thaliana/A.thaliana5289_neg.fasta")
    fasta_dict_train = dict(fasta_dict_train_pos, **fasta_dict_train_neg)
    fasta_dict_test_pos, id_list_pos = read_fasta("../data/A.thaliana/A.thaliana1000_pos.fasta")
    fasta_dict_test_neg, id_list_neg = read_fasta("../data/A.thaliana/A.thaliana1000_neg.fasta")
    fasta_dict_test = dict(fasta_dict_test_pos, **fasta_dict_test_neg)

    features_train = []
    labels_train = []
    for key, value in fasta_dict_train.items():

        temp = np.zeros((41, 11, 17))
        features_kmer = feature.kmer(value)
        for m in range(64):
            temp[0][m // 17][m % 17] = features_kmer[m]
        features = [feature.convert_to_graph(value), temp]

        features_train.append(features)
        if "+" in key:
            labels_train.append(1)
        else:
            labels_train.append(0)

    features_test = []
    labels_test = []
    for key, value in fasta_dict_test.items():

        temp = np.zeros((41, 11, 17))
        features_kmer = feature.kmer(value)
        for m in range(64):
            temp[0][m // 17][m % 17] = features_kmer[m]
        features = [feature.convert_to_graph(value), temp]

        features_test.append(features)

        if "+" in key:
            labels_test.append(1)
        else:
            labels_test.append(0)
    skf = KFold(n_splits=10, shuffle=True, random_state=1226)
    loader = []

    for train_index, val_index in skf.split(features_train, labels_train):
        features_train_fold = np.array(features_train)[train_index]
        features_val = np.array(features_train)[val_index]
        labels_train_fold = np.array(labels_train)[train_index]
        labels_val = np.array(labels_train)[val_index]
        features_train_fold = torch.tensor(features_train_fold, dtype=torch.float32)
        features_val = torch.tensor(features_val, dtype=torch.float32)

        train_loader = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(torch.Tensor(features_train_fold),
                                                   torch.Tensor(labels_train_fold)),
            batch_size=128,
            shuffle=True,
            drop_last=True)
        val_loader = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(torch.Tensor(features_val),
                                                   torch.Tensor(labels_val)),
            batch_size=128,
            shuffle=True,
            drop_last=True)
        features_test = torch.tensor(features_test, dtype=torch.float32)
        test_loader = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(torch.Tensor(features_test),
                                                   torch.Tensor(labels_test)),
            batch_size=128,
            shuffle=True,
            drop_last=True)
        loader.append([train_loader, val_loader, test_loader])

    return loader


if __name__ == "__main__":
    loader1 = data_loader()
    [train_load, val_load, test_load] = loader1[0]
    for i, data in enumerate(train_load, 0):
        inputs, labels = data
        print(inputs[0][0].shape, inputs[0][1].shape, sep='\n')
