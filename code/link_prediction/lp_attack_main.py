import sys
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
import numpy as np
import os
import inspect
import networkx as nx

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from utils.clustering_utils import DataLoader

from USER_model.USER_main import use_ESGC

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import scipy.sparse as sp
from sklearn.preprocessing import normalize

def get_roc_score(edges_pos, edges_neg, embeddings, adj_sparse):
    "from https://github.com/tkipf/gae"

    score_matrix = np.dot(embeddings, embeddings.T)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Store positive edge predictions, actual values
    preds_pos = []
    pos = []
    for edge in edges_pos:
        preds_pos.append(sigmoid(score_matrix[edge[0], edge[1]]))  # predicted score
        pos.append(adj_sparse[edge[0], edge[1]])  # actual value (1 for positive)

    # Store negative edge predictions, actual values
    preds_neg = []
    neg = []
    for edge in edges_neg:
        preds_neg.append(sigmoid(score_matrix[edge[0], edge[1]]))  # predicted score
        neg.append(adj_sparse[edge[0], edge[1]])  # actual value (0 for negative)

    # Calculate scores
    preds_all = np.hstack([preds_pos, preds_neg])
    labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])

    # print(preds_all, labels_all )

    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    return roc_score, ap_score


def ESGC(data, data_name):
    _, y_out, _,embs = use_ESGC(data, data_name)
    return normalize(embs.cpu().detach().numpy())

def use_model(data_name, adj_attack, test_edges, test_edges_false, model_name='USER'):
    data = DataLoader(data_name)
    adj_sparse = data.adj

    adjnew = adj_sparse - sp.dia_matrix((adj_sparse.diagonal()[np.newaxis, :], [0]), shape=adj_sparse.shape)
    adjnew.eliminate_zeros()
    adj_sparse = adjnew

    data.adj = adj_attack
    adj = data.adj
    label_true = data.labels
    features = data.features
    clu_num = data.clu_num

    embs = ESGC(data, data_name)

    roc, ap = get_roc_score(test_edges, test_edges_false, embs,
                                            adj_sparse)
    return roc, ap


def try_model(data_name, adj_attack, test_edges, test_edges_false, rounds=20, model_name='kmeans'):
    print(data_name)
    roc_list = []
    ap_list = []
    for ite in range(rounds):
        roc, ap = use_model(data_name, adj_attack, test_edges, test_edges_false, model_name)
        roc_list.append(roc)
        ap_list.append(ap)
        print(roc, ap)
    roc_mean = np.mean(roc_list)
    ap_mean = np.mean(ap_list)
    roc_std = np.std(roc_list)
    ap_std = np.std(ap_list)
    return roc_mean, ap_mean, roc_std, ap_std


if __name__ == '__main__':
    data_l = ['citeseer','wiki']
    model_name_l = [
        'USER',
    ]
    rounds = 10
    for data_name in data_l:
        for model_name in model_name_l:
            for rate in [0, 0.1,0.2,0.3,0.4,0.5]:
                text_result = ''
                flip_list = np.load(str(data_name) + str(rate) + '_flip.npy', allow_pickle=True)
                test_edges = np.load(str(data_name) + 'test_edges_add.npy', allow_pickle=True)
                test_edges_false = np.load(str(data_name) + 'test_edges_false_add.npy', allow_pickle=True)
                print('-' * 20 + ' ' + model_name + ' ' + '-' * 20)
                text_result += '-' * 20 + ' ' + model_name + ' ' + '-' * 20
                text_result += '\n'
                roc_mean_l = []
                ap_mean_l = []
                for adj_attack in flip_list:
                    roc_mean, ap_mean, roc_std, ap_std = try_model(data_name, adj_attack, test_edges[0], test_edges_false[0], rounds,
                                                                                      model_name)
                    print(roc_mean, ap_mean, roc_std, ap_std)
                    roc_mean_l += [roc_mean]
                    ap_mean_l += [ap_mean]
                    text_result += 'sc_roc_mean: ' + str(roc_mean_l) + 'roc_std: ' + str(roc_std)
                    text_result += '\n'
                    text_result += 'ap_mean: ' + str(ap_mean_l) + 'ap_std: ' + str(ap_std)
                    text_result += '\n'
                text_result += 'roc_mean_l: ' + str(np.mean(roc_mean_l))
                text_result += '\n'
                text_result += 'ap_mean_l: ' + str(np.mean(ap_mean_l))
                text_result += '\n'
                f = open(str(data_name) + str(rate) + str(model_name) + '_flip' + '_attack_baseline_result.txt', 'w')
                f.write(text_result)
                f.close()
