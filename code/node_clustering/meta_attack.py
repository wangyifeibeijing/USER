#from deeprobust.graph.global_attack import Random
from deeprobust.graph.global_attack import Metattack
from deeprobust.graph.defense import GCN
import scipy.sparse as sp
import numpy as np
import os
import sys
import inspect
import torch
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from utils.clustering_utils import DataLoader

np.random.seed(1)


def metanoise(dataname):
    data = DataLoader(dataname)
    adj = data.adj
    features = data.features
    labels = data.labels
    n,_=adj.shape
    idx_train=np.array(list(range(n)))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1, nhid=16,
                    with_relu=False, device=device)
    surrogate = surrogate.to(device)
    surrogate.fit(features, adj, labels, idx_train)
    idx_unlabeled =idx_train
    model = Metattack(model=surrogate, nnodes=n, feature_shape=features.shape, device=device)
    model = model.to(device)
    for rate in [0, 0.05, 0.1, 0.15, 0.2]:
        addlist = []
        perturbations = int(adj.getnnz() / 2 * rate)
        total_attack=5
        for num in range(total_attack):
            model.attack(features, adj, labels, idx_train, idx_unlabeled, perturbations, ll_constraint=False)
            if torch.cuda.is_available():
                modified_adj = model.modified_adj.cpu().numpy()
            else:
                modified_adj = model.modified_adj.numpy()
            addlist.append(modified_adj)
        addnumpy = np.array(addlist)
        np.save('m_att/'+str(dataname) + str(rate) + '_meta.npy', addnumpy)




if __name__ == '__main__':
    data_name_l = ['cora','citeseer','wiki']
    for data_name in data_name_l:
        metanoise(data_name)
