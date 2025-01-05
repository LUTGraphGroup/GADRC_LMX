import numpy as np
import random
import torch
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os
from sklearn.utils import resample

def move_to_device(obj, device):
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(list(obj), device))
    elif isinstance(obj, torch.Tensor):
        return obj.to(device)
    else:
        return obj


def k_fold(data, args):
    k = args.k_fold

    # k折交叉验证
    skf = StratifiedKFold(n_splits=k, random_state=None, shuffle=False)
    X = data['all_drdi']
    Y = data['all_label']

    # 确定总的正样本数量
    total_positives = np.sum(Y == 1)

    X_train_all, X_test_all, Y_train_all, Y_test_all = [], [], [], []

    for train_index, test_index in skf.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        X_train_pos = X_train[Y_train == 1]
        X_train_neg = X_train[Y_train == 0]
        Y_train_pos = Y_train[Y_train == 1]
        Y_train_neg = Y_train[Y_train == 0]
        X_train_neg = resample(X_train_neg, replace=False, n_samples=len(X_train_pos) * 10, random_state=42)
        Y_train_neg = resample(Y_train_neg, replace=False, n_samples=len(X_train_pos) * 10, random_state=42)

        # 合并正负样本
        X_train = np.concatenate([X_train_pos, X_train_neg])
        Y_train = np.concatenate([Y_train_pos, Y_train_neg])

        # 在测试集中，使负样本数量与正样本数量相等
        X_test_pos = X_test[Y_test == 1]
        X_test_neg = X_test[Y_test == 0]
        Y_test_pos = Y_test[Y_test == 1]
        Y_test_neg = Y_test[Y_test == 0]
        X_test_neg = resample(X_test_neg, replace=False, n_samples=len(X_test_pos), random_state=42)
        Y_test_neg = resample(Y_test_neg, replace=False, n_samples=len(X_test_pos), random_state=42)

        # 合并正负样本
        X_test = np.concatenate([X_test_pos, X_test_neg])
        Y_test = np.concatenate([Y_test_pos, Y_test_neg])

        # 扩展维度并转换类型
        Y_train = np.expand_dims(Y_train, axis=1).astype('float64')
        Y_test = np.expand_dims(Y_test, axis=1).astype('float64')

        # 添加到列表中
        X_train_all.append(X_train)
        X_test_all.append(X_test)
        Y_train_all.append(Y_train)
        Y_test_all.append(Y_test)


    for i in range(k):
        # 创建目录
        fold_dir = os.path.join(args.data_dir, 'fold', str(i))
        os.makedirs(fold_dir, exist_ok=True)

        X_train1 = pd.DataFrame(data=np.concatenate((X_train_all[i], Y_train_all[i]), axis=1),
                                columns=['drug', 'disease', 'label'])
        X_train1.to_csv(os.path.join(fold_dir, 'data_train.csv'))

        X_test1 = pd.DataFrame(data=np.concatenate((X_test_all[i], Y_test_all[i]), axis=1),
                               columns=['drug', 'disease', 'label'])
        X_test1.to_csv(os.path.join(fold_dir, 'data_test.csv'))

    data['X_train'] = X_train_all  # 训练集索引
    data['X_test'] = X_test_all    # 测试集索引
    data['Y_train'] = Y_train_all  # 训练集标签
    data['Y_test'] = Y_test_all    # 测试集标签
    return data


def get_adj(edges, size):
    edges_tensor = torch.LongTensor(edges).t()  # 转置
    values = torch.ones(len(edges))

    adj = torch.sparse.LongTensor(edges_tensor, values, size).to_dense().long()
    return adj


def data_processing(data, args):
    drdi_matrix = get_adj(data['drdi'], (args.drug_number, args.disease_number))
    one_index = []
    zero_index = []
    for i in range(drdi_matrix.shape[0]):
        for j in range(drdi_matrix.shape[1]):
            if drdi_matrix[i][j] >= 1:
                one_index.append([i, j])
            else:
                zero_index.append([i, j])
    random.seed(args.random_seed)
    random.shuffle(one_index)
    random.shuffle(zero_index)

    index = np.array(one_index + zero_index, dtype=int)

    label = np.array([1] * len(one_index) + [0] * len(zero_index), dtype=int)

    samples = np.concatenate((index, np.expand_dims(label, axis=1)), axis=1)
    label_p = np.array([1] * len(one_index), dtype=int)

    drdi_p = samples[samples[:, 2] == 1, :]
    drdi_n = samples[samples[:, 2] == 0, :]

    drs_mean = (data['drf'] + data['drg']) / 2
    dis_mean = (data['dip'] + data['dig']) / 2

    drs = np.where(data['drf'] == 0, data['drg'], drs_mean)
    dis = np.where(data['dip'] == 0, data['dig'], dis_mean)



    drs_non_zero_positions = np.nonzero(drs)

    drs_edge_index = torch.tensor(drs_non_zero_positions, dtype=torch.long)

    drs_edge_index_np = drs_edge_index.numpy().transpose()

    drs_edge_weights = drs[drs_edge_index_np[:, 0], drs_edge_index_np[:, 1]]

    drs_edge_weights_tensor = torch.from_numpy(drs_edge_weights)
    data['drug_edge_index'] = drs_edge_index
    data['drug_edge_weight'] = drs_edge_weights_tensor


    dis_non_zero_positions = np.nonzero(dis)

    dis_edge_index = torch.tensor(dis_non_zero_positions, dtype=torch.long)

    dis_edge_index_np = dis_edge_index.numpy().transpose()

    dis_edge_weights = dis[dis_edge_index_np[:, 0], dis_edge_index_np[:, 1]]

    dis_edge_weights_tensor = torch.from_numpy(dis_edge_weights)
    data['dis_edge_index'] = dis_edge_index
    data['dis_edge_weight'] = dis_edge_weights_tensor



    data['drug_disease_matrix'] = drdi_matrix
    data['drs'] = drs
    data['dis'] = dis
    data['all_samples'] = samples
    data['all_drdi'] = samples[:, :2]
    data['all_drdi_p'] = drdi_p
    data['all_drdi_n'] = drdi_n
    data['all_label'] = label
    data['all_label_p'] = label_p

    return data

def get_data(args):
    data = dict()

    drf = pd.read_csv(args.data_dir + 'DrugFingerprint.csv').iloc[:, 1:].to_numpy()
    drg = pd.read_csv(args.data_dir + 'DrugGIP.csv').iloc[:, 1:].to_numpy()

    dip = pd.read_csv(args.data_dir + 'DiseasePS.csv').iloc[:, 1:].to_numpy()
    dig = pd.read_csv(args.data_dir + 'DiseaseGIP.csv').iloc[:, 1:].to_numpy()

    data['drug_number'] = int(drf.shape[0])
    data['disease_number'] = int(dip.shape[0])

    data['drf'] = drf
    data['drg'] = drg
    data['dip'] = dip
    data['dig'] = dig

    data['drdi'] = pd.read_csv(args.data_dir + 'DrugDiseaseAssociation.csv', usecols=[0, 2], dtype=int, skiprows=0).to_numpy()

    return data


def print_metrics(name, values):
    mean_value = np.mean(values)
    std_value = np.std(values)
    print(f'{name}: {values}')
    print(f'Mean {name}: {mean_value} ({std_value})')


