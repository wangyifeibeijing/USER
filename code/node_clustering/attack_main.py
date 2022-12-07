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
from utils.clustering_utils import ClusterMetrics
from USER_model.USER_main import use_ESGC








def ESGC(data, data_name):
    _, y_out, _,z = use_ESGC(data, data_name)
#     np.save("USER_emb.npy", z.detach().cpu().numpy())
    labels = y_out.cpu()
    labels = labels.tolist()
    return labels

def use_model(data_name, adj_attack, model_name='USER'):
    data = DataLoader(data_name)
    data.adj = adj_attack
    adj = data.adj
    label_true = data.labels
    features = data.features
    clu_num = data.clu_num
    label_pred = ESGC(data, data_name)

    cluster_metric = ClusterMetrics(label_true, label_pred)
    acc = cluster_metric.get_acc()
    nmi = cluster_metric.get_nmi()
    f1 = cluster_metric.get_f1()
    return acc, nmi, f1


def try_model(data_name, adj_attack, rounds=20, model_name='kmeans'):
    print(data_name)
    acc_list = []
    nmi_list = []
    f1_list = []
    for ite in range(rounds):
        acc, nmi, f1 = use_model(data_name, adj_attack, model_name)
        acc_list.append(acc)
        nmi_list.append(nmi)
        f1_list.append(f1)
        print(acc, nmi, f1)
    return acc_list, nmi_list, f1_list

def run_all():
    data_name_l = [
        'cora',
        'citeseer',
        'wiki',
    ]
    model_name_l = [
        'USER',
    ]
    att_l = [
        'm_att',  # mete-attack
        'f_att',  # random noises
    ]
    rounds = 10
    for model_name in model_name_l:
        for data_name in data_name_l:
            for att_type in att_l:
                if (att_type == 'm_att'):
                    rate_list_use = [0.05, 0.1, 0.15, 0.2]
                    for rate in rate_list_use:
                        ################### meta_attack ###################
                        text_result = ''
                        meta_list = np.load('m_att/' + str(data_name) + str(rate) + '_meta.npy', allow_pickle=True)
                        print('-' * 20 + ' ' + model_name + ' ' + data_name + ' ' + str(rate) + ' ' + '-' * 20)
                        text_result += '-' * 20 + ' ' + model_name + ' ' + '-' * 20
                        text_result += '\n'
                        acc_mean_l = []
                        nmi_mean_l = []
                        f1_mean_l = []
                        for adj_attack in meta_list:
                            acc_list, nmi_list, f1_list = try_model(data_name, adj_attack, rounds, model_name)
                            acc_mean_l = acc_mean_l + acc_list
                            nmi_mean_l = nmi_mean_l + nmi_list
                            f1_mean_l = f1_mean_l + f1_list
                        text_result += 'acc_mean_l: ' + str(acc_mean_l)
                        text_result += '\n'
                        text_result += 'nmi_mean_l: ' + str(nmi_mean_l)
                        text_result += '\n'
                        text_result += 'f1_mean_l: ' + str(f1_mean_l)
                        text_result += '\n'
                        text_result += 'acc_list_mean: ' + str(np.mean(acc_mean_l))
                        text_result += '\n'
                        text_result += 'acc_list_std: ' + str(np.std(acc_mean_l))
                        text_result += '\n'
                        text_result += 'nmi_list_mean: ' + str(np.mean(nmi_mean_l))
                        text_result += '\n'
                        text_result += 'nmi_list_std: ' + str(np.std(nmi_mean_l))
                        text_result += '\n'
                        text_result += 'f1_list_mean: ' + str(np.mean(f1_mean_l))
                        text_result += '\n'
                        text_result += 'f1_list_std: ' + str(np.std(f1_mean_l))
                        text_result += '\n'
                        f = open(
                            'm_result/' + str(data_name) + '_' + str(rate) + '_' + str(model_name) + '_meta_result.txt',
                            'w')
                        f.write(text_result)
                        f.close()
                else:
                    rate_list_use = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
                    for rate in rate_list_use:
                        ################### random-noises ###################
                        text_result = ''
                        flip_list = np.load('f_att/' + str(data_name) + str(rate) + '_flip.npy', allow_pickle=True)
                        print('-' * 20 + ' ' + model_name + ' ' + data_name + ' ' + str(rate) + ' ' + '-' * 20)
                        text_result += '-' * 20 + ' ' + model_name + ' ' + '-' * 20
                        text_result += '\n'
                        acc_mean_l = []
                        nmi_mean_l = []
                        f1_mean_l = []
                        for adj_attack in flip_list:
                            acc_list, nmi_list, f1_list = try_model(data_name, adj_attack, rounds,
                                                                    model_name)
                            acc_mean_l = acc_mean_l + acc_list
                            nmi_mean_l = nmi_mean_l + nmi_list
                            f1_mean_l = f1_mean_l + f1_list
                        text_result += 'acc_mean_l: ' + str(acc_mean_l)
                        text_result += '\n'
                        text_result += 'nmi_mean_l: ' + str(nmi_mean_l)
                        text_result += '\n'
                        text_result += 'f1_mean_l: ' + str(f1_mean_l)
                        text_result += '\n'
                        text_result += 'acc_list_mean: ' + str(np.mean(acc_mean_l))
                        text_result += '\n'
                        text_result += 'acc_list_std: ' + str(np.std(acc_mean_l))
                        text_result += '\n'
                        text_result += 'nmi_list_mean: ' + str(np.mean(nmi_mean_l))
                        text_result += '\n'
                        text_result += 'nmi_list_std: ' + str(np.std(nmi_mean_l))
                        text_result += '\n'
                        text_result += 'f1_list_mean: ' + str(np.mean(f1_mean_l))
                        text_result += '\n'
                        text_result += 'f1_list_std: ' + str(np.std(f1_mean_l))
                        text_result += '\n'
                        f = open('f_result/' + str(data_name) + '_' + str(rate) + '_' + str(
                            model_name) + '_random_result.txt', 'w')
                        f.write(text_result)
                        f.close()


if __name__ == '__main__':
    run_all()