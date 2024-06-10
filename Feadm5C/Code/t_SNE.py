from data_load import read_fasta
import feature
from visualization import tsne_visualization

fasta_dict_train_pos, id_list_pos = read_fasta("../data/A.thaliana/A.thaliana5289_pos.fasta")
fasta_dict_train_neg, id_list_neg = read_fasta("../data/A.thaliana/A.thaliana5289_neg.fasta")
fasta_dict_train = dict(fasta_dict_train_pos, **fasta_dict_train_neg)
fasta_dict_test_pos, id_list_pos = read_fasta("../data/A.thaliana/A.thaliana1000_pos.fasta")
fasta_dict_test_neg, id_list_neg = read_fasta("../data/A.thaliana/A.thaliana1000_neg.fasta")
fasta_dict_test = dict(fasta_dict_test_pos, **fasta_dict_test_neg)
X1 = []
y1 = []
X2 = []
y2 = []
for key, value in fasta_dict_train.items():
    features = feature.convert_to_graph(value)
    features = features.flatten()
    X1.append(features)
    if "+" in key:
        y1.append(1)
    else:
        y1.append(0)
for key, value in fasta_dict_test.items():
    features = feature.convert_to_graph(value)
    features = features.flatten()
    X2.append(features)
    if "+" in key:
        y2.append(1)
    else:
        y2.append(0)
tsne_visualization(X1, y1, X2, y2)
