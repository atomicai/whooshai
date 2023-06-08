import os
import pickle
import sys
import unittest
from pathlib import Path

import graph_tool.generation as g_gen
import graph_tool.topology as g_topo
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.optim as optim
import yaml

# from models_spagat import SpaGAT
from whooshai.modeling import SpaGAT
# from utils import gen_pathm
from whooshai.tooling import gen_pathm
from whooshai.training.module import Trainer


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum + 9e-15, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def parse_index_file(filename):
    """Parse index file."""
    index = []
    with open(str(Path(filename))) as f:
        for line in f:
            index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask)
    # return np.array(mask, dtype=np.bool)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    _adj = sp.coo_matrix(adj)  # координатный формат матрицы смежности.
    # См.https://scipy-lectures.org/advanced/scipy_sparse/coo_matrix.html

    # Например,
    # 22 -> 22, 7307, 18734
    # 23 -> 10601
    # 24 -> 16184, 7733, 8129, 750
    # _adj.row[165:168]
    # (22, 22) | idx == 165 | (_adj.row[165], _adj.col[165])
    # (22, 7307) | idx == 166 | (_adj.row[166], _adj.col[166])
    # (22, 18734) | idx == 167 | (_adj.row[167], _adj.col[167])
    # _adj.col[165:168]
    rowsum = np.array(_adj.sum(1))  # исходящая степень вершины
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()  # 1 / sqrt(deg(v_i))
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0  # Вдруг есть изолированные вершины. В этих вершинах нужно явно занулить
    d_mat_inv_sqrt = sp.diags(
        d_inv_sqrt
    )  # Делаем диагональную матрицу, размером NxN, где на диагоналях будут нормализованные значения
    # N = 5
    # (1, 2)
    # (1, 3)
    # (1, 4)
    # (1, 5)
    # (2, 3)
    # (3, 4)
    # [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 1, 1, 1, 0], [1, 0, 1, 1, 0], [1, 0, 0, 0, 1]]
    # DEG(1) => 5
    # DEG(2) => 3
    # DEG(3) => 4
    # DEG(4) => 3
    # DEG(5) => 2
    # Сделаем в матрице смежности вместо (1/0) нормализованное значение, которое обратно
    # пропорционально степени вершины (куда ведет ребро)
    # Это как-то помогает сходимости (видимо?)
    # Здесь дважды делаем эту операцию, чтобы сделать граф симметричным (i-j ребро => есть и j-i ребро)
    return _adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized


def load_data():
    names = ['x', 'y', 'tx', 'ty', 'graph']
    objects = []
    for i in range(len(names)):
        filepath = str(Path(os.getcwd()) / "data/ind.whoosh.{}".format(names[i]))
        with open(filepath, 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pickle.load(f, encoding='latin1'))
            else:
                objects.append(pickle.load(f))

    x, y, tx, ty, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.whoosh.test.index")
    test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((x, tx)).tolil()
    # Эта строчка  ниже меняет местами строки (вектор-фичи) тестовой под-выборки
    # чтобы они шли по-возрастанию. Т.е., условно говоря, имея индексы [5, 9, 4, 3],
    # мы меняем местами строки так, чтобы они шли [3, 4, 5, 9]
    features[test_idx_reorder, :] = features[test_idx_range, :]
    # И теперь создаем матрицу смежности из всего этого.
    # Переменная `graph` - это обычная матрица смежности, где показываются ссылки на другие статьи
    # e.g. graph[0] = [14442, 1378, 1544, 6092, 7636]

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    # Повторяем процедуру с метками.
    labels = np.vstack((y, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    # return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
    features = normalize(features)
    adj = preprocess_adj(adj)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    print('node number:', features.shape[0])
    print('training sample:', sum(train_mask))
    print('validation sample:', sum(val_mask))
    print('testing sample:', sum(test_mask))
    idx_train = torch.LongTensor(np.where(train_mask)[0])
    idx_val = torch.LongTensor(np.where(val_mask)[0])
    idx_test = torch.LongTensor(np.where(test_mask)[0])

    return adj, features, labels, idx_train, idx_val, idx_test


def train():
    with open(str(Path(os.getcwd()) / "whooshai" / "recoiling" / "config.yaml")) as f:
        config = yaml.safe_load(f)

    adj, features, labels, idx_train, idx_val, idx_test = load_data()
    model = SpaGAT(
        nfeat=features.shape[1],
        nhid=config["hidden"],
        nclass=labels.max().item() + 1,
        dropout=config["dropout"],
        alpha=config["alpha"],
        nheads=config["nheads"],
    )

    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    trainer = Trainer(model, adj=adj, features=features, optimizer=optimizer, epochs=config["epochs"], device="cpu", labels=labels)

    trainer.train(labels=labels, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)

    print("Convergence is completed")


def inference():
    # TODO:
    with open(str(Path(os.getcwd()) / "whooshai" / "recoiling" / "config.yaml")) as f:
        config = yaml.safe_load(f)
    adj, features, labels, idx_train, idx_val, idx_test = load_data()
    model = SpaGAT(
        nfeat=features.shape[1],
        nhid=config["hidden"],
        nclass=labels.max().item() + 1,
        dropout=config["dropout"],
        alpha=config["alpha"],
        nheads=config["nheads"],
    )
    model.load_state_dict(torch.load(str(Path(os.getcwd()) / "weights" / "zaeleillaep.pkl")))
    
    genPath = str(Path(os.getcwd()) / "weights" / "att")
    Nratio, Ndeg = 1.0, 0.5
    response = gen_pathm([8], matpath=genPath, Nratio=Nratio, Ndeg=Ndeg)

    return response


__all__ = ["train", "inference"]
