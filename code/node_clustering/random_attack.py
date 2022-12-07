from deeprobust.graph.global_attack import Random
import scipy.sparse as sp
import numpy as np
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from utils.clustering_utils import DataLoader

np.random.seed(1)




def flipnoise(dataname, rate_l):
    data = DataLoader(dataname)
    model = Random()
    adj = data.adj
    for rate in rate_l:
        fliplist = []
        n_perturbations = int(adj.getnnz() / 2 * rate)
        for num in range(5):
            model.attack(adj, n_perturbations, type='flip')
            modified_adj = model.modified_adj
            fliplist.append(modified_adj)
        flipnumpy = np.array(fliplist)
        np.save('f_att/' + str(dataname) + str(rate) + '_flip.npy', flipnumpy)


if __name__ == '__main__':
    data_name_l = ['cora','citeseer','wiki']
    rate_l = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    for data_name in data_name_l:
        flipnoise(data_name, rate_l)
