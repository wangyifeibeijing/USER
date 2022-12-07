import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import USER_model.USER_functions as USER_functions
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import normalize
import sys


# 2-layer GCN
class Net(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden):
        super(Net, self).__init__()
        self.conv1 = nn.Linear(num_node_features, hidden, bias=True)
        self.conv2 = nn.Linear(hidden, num_classes, bias=True)
        self.norm1 = nn.BatchNorm1d(num_node_features)
        self.norm2 = nn.BatchNorm1d(hidden)

    def forward(self, x, a):
        x2 = self.norm1(x)
        x2 = self.conv1(a.mm(x2))
        x2 = F.relu(x2)
        x2 = self.norm2(x2)
        x1 = F.dropout(x2, training=self.training)
        x1 = self.conv2(a.mm(x1))
        x1 = F.dropout(x1, training=self.training)
        return x1, x2


# softmax
class Clu(torch.nn.Module):
    def __init__(self):
        super(Clu, self).__init__()

    def forward(self, x):
        return F.softmax(x, dim=1)


# USER_structure--entropy supervised GCN (ESGC)
class ESGC(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, ADJ, hidden):
        super(ESGC, self).__init__()
        self.num_node_features = num_node_features
        self.num_classes = num_classes
        self.clu_encoder = Net(self.num_node_features, self.num_classes, hidden)
        self.clu_decoder = Clu()
        self.A_adaptive = torch.nn.Parameter(torch.randn(ADJ.size()[0], ADJ.size()[0]), requires_grad=True)

    def forward(self, x, adj):
        a = self.A_adaptive
        a = F.leaky_relu(a)
        a = a.mm(a.t())
        ad = 0.5 * adj + 0.5 * USER_functions.torchnormalization(a)
        ad = USER_functions.torchnormalization(ad)
        x1, x2 = self.clu_encoder(x, ad)
        c = self.clu_decoder(x1)
        return a, c, x2


# residual entropy loss
class ETR_loss_trace(torch.nn.Module):
    def __init__(self, IsumC, IsumCDC, Amask):  # Isum: matrix made up by ones
        super(ETR_loss_trace, self).__init__()
        self.IsumC = IsumC
        # IsumC : torch.ones(1, n).to(device)
        self.IsumCDC = IsumCDC
        # IsumCDC : torch.ones(k, 1).to(device)
        self.Amask = Amask
        # Amask : (torch.ones(n, n) - torch.eye(n)).to(device)

    def forward(self, A_t, C, B_t, X, rate_a, rate_b,x2):  # A: input weight; C: indicator
        A = (A_t).mul(self.Amask)  # selfloop
        Deno_sumA = 1 / (torch.sum(A))
        Rate_p = (C.t().mm(A.mm(C))) * Deno_sumA
        enco_p = (self.IsumCDC.mm(self.IsumC.mm(A.mm(C)))) * Deno_sumA
        encolen = torch.log2(enco_p + 1e-20)
        total1 = torch.trace(Rate_p.mul(encolen))

        B = (B_t).mul(self.Amask)
        Deno_sumB = 1 / (torch.sum(B))
        Rate_pB = (C.t().mm(B.mm(C))) * Deno_sumB
        enco_pB = (self.IsumCDC.mm(self.IsumC.mm(B.mm(C)))) * Deno_sumB
        encolenB = torch.log2(enco_pB + 1e-20)
        total2 = torch.trace(Rate_pB.mul(encolenB))

        An, _ = A.size()
        cons = torch.norm((A_t.mul(B_t) - B_t).mul(self.Amask))
        fea_loss = my_DBI(C, X)
        return total1 + total2 + rate_b * fea_loss + rate_a * cons, total1, total2, cons, fea_loss


def my_Smooth(A, X):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A = A.float().clone().to(device)
    A = USER_functions.torchnormalization(A)
    d = A.sum(1)
    D = torch.diag(d)
    L = D - A
    return torch.trace(torch.matmul(X.t(), torch.matmul(L, X)))


def my_DBI(C, X):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    C = C.float().clone().to(device)
    X = X.float().clone().to(device)
    _, d = X.size()
    n, k = C.size()
    In1 = torch.ones(1, n).to(device)
    DI1 = torch.ones(d, 1).to(device)
    C_3D = C.unsqueeze_(-1).expand(n, k, d).transpose(1, 0)
    X_3D = X.unsqueeze_(-1).expand(n, d, k).transpose(2, 0).transpose(2, 1)

    XC3D = torch.mul(C_3D, X_3D)
    Cen = torch.matmul(In1, XC3D)
    C_n = torch.pow(torch.matmul(In1, C_3D), -1)
    Cen = torch.mul(Cen, C_n)
    Cen_nd = Cen.expand(-1, n, -1)
    Cen_n2 = torch.mul(C_3D, torch.pow(Cen_nd, 2))
    XC3Dn2 = torch.pow(XC3D, 2)
    S = torch.matmul(
        torch.mul(torch.matmul(In1, (Cen_n2 - 2 * torch.mul(C_3D, torch.mul(Cen_nd, XC3D)) + XC3Dn2)), C_n), DI1)
    S = torch.pow(S, 0.5)

    Cen_n1 = (torch.matmul(torch.pow(Cen, 2), DI1)).expand(-1, -1, k)
    M1 = Cen_n1 + Cen_n1.transpose(0, 2)
    M2 = 2 * torch.matmul(Cen.transpose(0, 1), Cen.transpose(1, 2)).transpose(1, 2)
    M = M1 - M2
    M11 = (M.transpose(1, 0)[0]).fill_diagonal_(1)
    M11 = torch.pow(M11, -0.5)
    S = S.expand(-1, -1, k)
    S11 = ((S + S.transpose(0, 2)).transpose(1, 0)[0]).fill_diagonal_(0)
    R = torch.mul(S11, M11)
    ma = torch.max(R, 0).values
    result = torch.sum(ma) * (1 / k)
    return result


def use_ESGC(data, data_name, ite=400, rate_a=0.1, rate_b=0.03):
    # cora: 0.5 0.005 256 0.1
    # 20 rounds
    # 0.5 1.9124649281278763 0.697821270310192 0.5555555407514049 0.6590881170662796

    # wiki: 0.05 4.2 256 0.001
    # 20 rounds
    # 0.05 1.4004294192834754 0.5057172557172558 0.4813966392856986 0.413315524280521

    # citeseer: 0.22 0.19 512 0.003
    # 20 rounds
    # 0.22 1.572250858244522 0.6205139765554554 0.36456307324782067 0.587173808441246

    if (data_name == 'cora'):
        args_rate_a = 0.5
        args_rate_b = 0.005
        args_hidden = 256
        args_lr = 0.1
    elif (data_name == 'wiki'):
        args_rate_a = 0.05
        args_rate_b = 4.2
        args_hidden = 256
        args_lr = 0.001
    elif (data_name == 'citeseer'):
        args_rate_a = 0.22
        args_rate_b = 0.19
        args_hidden = 512
        args_lr = 0.003
    elif (data_name == 'Polblogs'):
        args_rate_a = 0.4
        args_rate_b = 0
        args_hidden = 256
        args_lr = 0.1
    num_node = len(data.labels)
    if data.dataname == 'wiki':
        data.features = data.features.numpy()
        data.features = preprocessing.normalize(data.features, axis=1)
    elif data.dataname == 'Polblogs':
        data.features = data.features.numpy()
    else:
        data.features = np.array(data.features.todense())
    data.features = data.features.astype(np.float32)
    num_node_features = len(data.features[0])

    num_classes = data.clu_num
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data.features = torch.from_numpy(data.features)
    data.features = data.features.to(device)
    try:
        data.adj = np.array(data.adj.todense())
    except:
        data.adj = np.array(data.adj)
        
    data.adj = data.adj.astype(np.float32)
    np.fill_diagonal(data.adj, 0)
    adj_sy = 0.5 * (data.adj + data.adj.T)
    adj_sy[adj_sy != 0] = 1
    B_tem = torch.from_numpy(adj_sy)
    B_U = torch.from_numpy(adj_sy).to(device)
    B_U = USER_functions.torchnormalization(B_U)
    model = ESGC(num_node_features, num_classes, B_U, args_hidden).to(device)

    B_U1 = B_tem.to(device)

    optimizer = torch.optim.Adam(model.parameters(), args_lr, weight_decay=5e-4)
    IsumC = torch.ones(1, num_node).to(device)
    IsumCDC = torch.ones(num_classes, 1).to(device)
    Amask = (torch.ones(num_node, num_node) - torch.eye(num_node)).to(device)
    criterion = ETR_loss_trace(IsumC, IsumCDC, Amask).to(device)
    lr_optim = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3)

    model.train()
    loss_list = []
    loss_x = data.features

    for epoch in range(ite):
        optimizer.zero_grad()
        a, c, x2 = model(data.features, B_U)
        x2 = x2.cpu().detach()
        if np.isnan(x2).any() == 1:
            print("nan")
            sys.exit()
        loss, _, _, _, _ = criterion(a, c, B_U1, loss_x, args_rate_a, args_rate_b,x2)
        loss.backward()
        optimizer.step()
        lr_optim.step(loss)
        loss_list.append(loss.item())
    model.eval()
    a, c, x2 = model(data.features, B_U)
    _, y_out = c.max(1)
    ad = 0.5 * B_U + 0.5 * USER_functions.torchnormalization(a)
    ad = USER_functions.torchnormalization(ad)
    return a, y_out, loss_list, torch.mm(ad,x2)