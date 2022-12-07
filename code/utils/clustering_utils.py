import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn import metrics
import scipy.sparse as sp
import sklearn.preprocessing as preprocess
import networkx as nx
import pickle as pkl
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


class ClusterMetrics:
    def __init__(self, y_true, y_pred):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        # do best map
        assert len(y_pred) == len(y_true)
        D = max(np.max(y_pred), np.max(y_true)) + 1
        w = np.zeros((D, D), dtype=np.int64)

        for i in range(len(y_pred)):
            w[y_pred[i], y_true[i]] += 1

        ind, tid = linear_assignment(np.max(w) - w)
        y_aim = y_pred
        for i in range(len(y_aim)):
            y_aim[i] = tid[y_aim[i]]
        self.y_aim = y_aim

    def get_acc(self):
        acc_count = 0.0
        for i in range(len(self.y_true)):
            if self.y_true[i] == self.y_aim[i]:
                acc_count += 1.0
        return acc_count / (len(self.y_aim))

    def get_nmi(self):
        nmi = metrics.normalized_mutual_info_score(self.y_true, self.y_aim)
        return nmi

    def get_f1(self):
        f1score = metrics.f1_score(self.y_true, self.y_aim, average='macro')
        return f1score


class DataLoader:
    def __init__(self, dataname):
        self.dataname = dataname
        self.adj, self.features, self.labels = self.load_data(self.dataname)

        if self.dataname == 'cora':
            self.clu_num = 7

        elif self.dataname == 'citeseer':
            self.clu_num = 6

        elif self.dataname == 'wiki':
            self.clu_num = 17

        if self.dataname == 'cora':
            self.labels = np.argmax(self.labels, 1)
        elif self.dataname == 'citeseer':
            self.labels = np.argmax(self.labels, 1)
        self.clu_num = len(set(self.labels))

    def parse_index_file(self, filename):
        index = []
        for line in open(filename):
            index.append(int(line.strip()))
        return index

    def load_data(self, dataset_str):
        """Load data."""
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []

        if self.dataname == 'wiki':
            adj, features, labels = self.load_wiki()
            return adj, features, labels

        for i in range(len(names)):
            with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = self.parse_index_file("data/ind.{}.test.index".format(dataset_str))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset_str == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range - min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + 500)

        return adj, features, labels

    def load_wiki(self):
        f = open('data/graph.txt', 'r')
        adj, xind, yind = [], [], []
        for line in f.readlines():
            line = line.split()

            xind.append(int(line[0]))
            yind.append(int(line[1]))
            adj.append([int(line[0]), int(line[1])])
        f.close()
        ##print(len(adj))

        f = open('data/group.txt', 'r')
        label = []
        for line in f.readlines():
            line = line.split()
            label.append(int(line[1]))
        f.close()

        f = open('data/tfidf.txt', 'r')
        fea_idx = []
        fea = []
        adj = np.array(adj)
        adj = np.vstack((adj, adj[:, [1, 0]]))
        adj = np.unique(adj, axis=0)

        labelset = np.unique(label)
        labeldict = dict(zip(labelset, range(len(labelset))))
        label = np.array([labeldict[x] for x in label])
        adj = sp.csr_matrix((np.ones(len(adj)), (adj[:, 0], adj[:, 1])), shape=(len(label), len(label)))

        for line in f.readlines():
            line = line.split()
            fea_idx.append([int(line[0]), int(line[1])])
            fea.append(float(line[2]))
        f.close()

        fea_idx = np.array(fea_idx)
        features = sp.csr_matrix((fea, (fea_idx[:, 0], fea_idx[:, 1])), shape=(len(label), 4973)).toarray()
        scaler = preprocess.MinMaxScaler()
        # features = preprocess.normalize(features, norm='l2')
        features = scaler.fit_transform(features)
        #features = torch.FloatTensor(features)

        return adj, features, label



if __name__ == '__main__':
    # download raw dataset from https://github.com/kimiyoung/planetoid/tree/master/data
    data = DataLoader('wiki')
