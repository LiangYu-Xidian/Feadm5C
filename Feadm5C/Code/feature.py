import numpy as np
from rdkit import Chem

import warnings

warnings.filterwarnings("ignore")


def get_SMILES():
    base_dict = {}
    """
    SMILES strings
    """
    with open('../data/basesSMILES/basesSMILES.dict', 'r') as f:
        for line in f:
            line = line.strip().split(',')
            base_dict[line[0]] = line[1]

    return base_dict


def atom_feature(atom):
    """
    atom feature
    """
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(), ['C', 'N', 'O']) +
                    one_of_k_encoding(atom.GetDegree(), [1, 2, 3]) +
                    one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3]) +
                    one_of_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3]) +
                    [atom.GetIsAromatic()] +
                    get_ring_info(atom))


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))

    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]

    return list(map(lambda s: x == s, allowable_set))


def get_ring_info(atom):
    ring_info_feature = []
    for i in range(5, 7):
        if atom.IsInRingSize(i):
            ring_info_feature.append(1)
        else:
            ring_info_feature.append(0)

    return ring_info_feature


def norm_adj(adjacency):
    """
    adjacency matrix normalization
    """
    I = np.array(np.eye(adjacency.shape[0]))
    adj_hat = adjacency + I

    # D^(-1/2) * (A + I) * D^(-1/2)
    D_hat = np.diag(np.power(np.array(adj_hat.sum(1)), -0.5).flatten(), 0)
    adj_norm = adj_hat.dot(D_hat).transpose().dot(D_hat)

    return adj_norm


def norm_fea(features):
    """
    node matrix normalization
    """
    fea_norm = features / features.sum(1).reshape(-1, 1)

    return fea_norm


def convert_to_graph(seq):
    base_dict = get_SMILES()
    max_atoms_num = 11  # G has 11 atoms
    graph_features_one_seq = []
    seq_SMILES = [base_dict[b] for b in seq]
    for base_SMILES in seq_SMILES:
        dna_mol = Chem.MolFromSmiles(base_SMILES)

        # Adjacency matrix
        adj_tmp = Chem.GetAdjacencyMatrix(dna_mol)
        adj_norm = norm_adj(adj_tmp)

        # Node feature matrix
        if adj_norm.shape[0] <= max_atoms_num:
            graph_feature = np.zeros((max_atoms_num, 17))
            node_feature_tmp = []
            for atom in dna_mol.GetAtoms():
                node_feature_tmp.append(atom_feature(atom))
            node_feature_norm = norm_fea(np.asarray(node_feature_tmp))

            graph_feature[0:len(node_feature_tmp), 0:17] = np.dot(adj_norm.T, node_feature_norm)

            graph_features_one_seq.append(graph_feature)

    graph_features_one_seq = np.asarray(graph_features_one_seq, dtype=np.float32)

    return graph_features_one_seq  # # (41, 11, 17)


def kmer(seq):
    kmer_1 = [0] * 4
    kmer_3 = [0] * 64
    base_dict_1 = {'A': 1, 'G': 2, 'C': 3, 'U': 4}
    base_dict_3 = {'AAA': 1, 'AAG': 2, 'AAC': 3, 'AAU': 4, 'AGA': 5, 'AGG': 6, 'AGC': 7, 'AGU': 8,
                   'ACA': 9, 'ACG': 10, 'ACC': 11, 'ACU': 12, 'AUA': 13, 'AUG': 14, 'AUC': 15, 'AUU': 16,
                   'GAA': 17, 'GAG': 18, 'GAC': 19, 'GAU': 20, 'GGA': 21, 'GGG': 22, 'GGC': 23, 'GGU': 24,
                   'GCA': 25, 'GCG': 26, 'GCC': 27, 'GCU': 28, 'GUA': 29, 'GUG': 30, 'GUC': 31, 'GUU': 32,
                   'CAA': 33, 'CAG': 34, 'CAC': 35, 'CAU': 36, 'CGA': 37, 'CGG': 38, 'CGC': 39, 'CGU': 40,
                   'CCA': 41, 'CCG': 42, 'CCC': 43, 'CCU': 44, 'CUA': 45, 'CUG': 46, 'CUC': 47, 'CUU': 48,
                   'UAA': 49, 'UAG': 50, 'UAC': 51, 'UAU': 52, 'UGA': 53, 'UGG': 54, 'UGC': 55, 'UGU': 56,
                   'UCA': 57, 'UCG': 58, 'UCC': 59, 'UCU': 60, 'UUA': 61, 'UUG': 62, 'UUC': 63, 'UUU': 64}

    for i in range(len(seq)):
        kmer_1[int(base_dict_1[seq[i]]) - 1] += 1
    for index in range(len(seq) - 2):
        kmer_3[int(base_dict_3[seq[index:index + 3]]) - 1] += 1

    kmer_1 = [i / sum(kmer_1) for i in kmer_1]
    kmer_3 = [i / sum(kmer_3) for i in kmer_3]
    kmer_feature = kmer_3
    return np.asarray(kmer_feature, dtype=np.float32)  # # (68,1)


def mgf_kmer(seq):
    mgf_feature = convert_to_graph(seq)
    kmer_feature = kmer(seq)
    mgf_feature = mgf_feature.reshape(17, 1, 451)
    kmer_feature = kmer_feature.reshape(17, 4, 1)
    mgf_kmer_feature = mgf_feature + kmer_feature
    mgf_kmer_feature = mgf_kmer_feature.reshape(41, 44, 17)
    return mgf_kmer_feature


if __name__ == "__main__":
    sequence = "GUCCGAUGCUCCGUCGUCUUCCCCGGAUGCCACGGCGUCGC"
    sequence2 = "CCGGAUCAAAACGAAGAUGGCUCGGUGAUUCUCGCGACGUG"
    sequences = [sequence]

    a = kmer(sequence)
