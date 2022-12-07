import torch
import numpy as np


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def torchnormalization(A, withoutdia=True, device='cuda'):
    if withoutdia == True:
        A = A.fill_diagonal_(0)
    [n, _] = A.shape
    Iden = torch.eye(n).to(device)
    A = A + Iden
    d = A.sum(1)
    D = torch.diag(torch.pow(d, -0.5))
    return torch.mm(D, (torch.mm(A, D)))

def sparse_dense_mul(s, d):
    i = s._indices()
    v = s._values()
    dv = d[i[0, :], i[1, :]]  # get values from relevant entries of dense matrix
    return torch.sparse.FloatTensor(i, v * dv, s.size())
