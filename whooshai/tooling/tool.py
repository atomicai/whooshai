import os
import pickle
import sys
import time

import graph_tool.generation as g_gen
import graph_tool.topology as g_topo
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch


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
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized


def single_gen_path(pathRes, shape, pathLen=3, Nratio=0.6, Ndeg=2):
    cols = []
    rows = []
    pathValues = []

    for row in pathRes.keys():
        degree = 0
        tmppath = []
        tmplen = []
        tmpcols = []
        tmprows = []
        dists = pathRes[row][0]
        paths = pathRes[row][1]
        for col in paths.keys():
            path = paths[col]
            if len(path) == pathLen:
                tmpcols.append(col)
                tmprows.append(row)
                tmplen.append(dists[col])
                tmppath.append(path)
            if len(path) == 2:
                degree += 1
        if pathLen == 2:
            ratio = 1.0
            maxRange = int(ratio * len(tmplen))
        else:
            ratio = Nratio
            maxRange = int(min([degree * Ndeg, int(ratio * len(tmplen))]))

        topk = sorted(range(len(tmplen)), key=lambda i: tmplen[i])[0:maxRange]

        pathValues = pathValues + [tmppath[i] for i in topk]

        cols = cols + [tmpcols[i] for i in topk]
        rows = rows + [tmprows[i] for i in topk]

    cols = cols + [i for i in range(shape[0])]
    rows = rows + [i for i in range(shape[0])]
    pathValues = pathValues + [[j] * pathLen for j in range(shape[0])]

    cols = np.array(cols).reshape(1, -1)
    rows = np.array(rows).reshape(1, -1)
    pathValues = np.array(pathValues)
    i = np.concatenate((rows, cols), axis=0)
    i = torch.LongTensor(i)

    pathV = torch.FloatTensor(pathValues)

    pathM = torch.sparse.FloatTensor(i, pathV, torch.Size([shape[0], shape[1], pathLen]))

    return pathM, 0


def load_pathm(dataset_str='whoosh', matpath=''):
    with open(os.path.join(matpath, 'pathDict.pickle'), 'rb') as pathfile:
        pathM = pickle.load(pathfile)
    for layer in pathM.keys():
        for head in pathM[layer].keys():
            for pathlen in pathM[layer][head].keys():
                pathM[layer][head][pathlen]['indices'] = pathM[layer][head][pathlen]['indices'].long().cpu()
                pathM[layer][head][pathlen]['values'] = pathM[layer][head][pathlen]['values'].long().cpu()

    return pathM


def graph_tool_apsp(spmatrix, cutoff=3):
    nodeN = spmatrix.shape[0]  # кол-во вершин
    edgeN = spmatrix.data.shape[0]  # кол-во ребер
    weightM = spmatrix.data  # Массив весов графа. в разреженной форме
    print(nodeN, edgeN)

    t = time.time()
    g = g_gen.Graph()
    g.add_vertex(nodeN)
    row = spmatrix.row.reshape(-1, 1)
    col = spmatrix.col.reshape(-1, 1)
    edge_list = np.hstack((row, col)).tolist()
    g.add_edge_list(edge_list)

    weights = g.new_edge_property("double")
    print()
    for i in range(edgeN):
        # weights[row[i][0], col[i][0]] = weightM[i]
        weights.fa[i] = weightM[i]
        # weights[g.vertex(row[i]), g.vertex(col[i])] = weightM[i]
    print("construct time: {:.4f}".format(time.time() - t))

    pathRes = {}
    for centerNode in range(nodeN):
        # print('\rnode:{:07d}'.format(centerNode))
        pathOneRes = {}
        distDict = {}
        pathDict = {}

        dist, pred = g_topo.shortest_distance(g, source=g.vertex(centerNode), weights=weights, pred_map=True)
        dist = dist.a  # to list
        pred = pred.a  # to list
        pathDict[centerNode] = [centerNode]
        pathkeys = set(pathDict.keys())
        for i in range(1, cutoff):  # generate path with length at most cutoff
            newpathkeys = []
            for col in range(nodeN):
                if col != centerNode and pred[col] in pathkeys:
                    pathDict[col] = pathDict[pred[col]] + [col]
                    distDict[col] = dist[col]
                    newpathkeys.append(col)
            pathkeys = set(newpathkeys)

        pathOneRes[0] = distDict
        pathOneRes[1] = pathDict
        pathRes[centerNode] = pathOneRes
    return pathRes


def gen_pathm(nheads=[8], matpath=None, Nratio=0.6, Ndeg=2):  # nheads=[8]: does not generate for the last layer
    # generate path matrix for each attention head
    # pathM is a dict of dict of dict, each element is a pytorch sparse matrix
    #   the first key is which 'layer' (start from 0)
    #   the second key is which 'head'  (start from 0)
    #   the third key is which 'path length' (start from 2)
    pathM = {}
    layerPathM = {}
    for layer, nhead in enumerate(nheads):
        # if layer>0:
        #    continue
        headPathM = {}
        headMatrix = None
        for head in range(nhead):
            if not matpath:
                spmatrix = sp.load_npz('models/exp1/attmat_{:d}_{:d}.npz'.format(layer + 1, head))
            else:
                spmatrix = sp.load_npz(matpath + '/attmat_{:d}_{:d}.npz'.format(layer + 1, head))
            if headMatrix is None:
                headMatrix = spmatrix
            else:
                headMatrix.data += spmatrix.data
        headMatrix.data = headMatrix.data / nhead

        headMatrix.data = -headMatrix.data  # Развернем граф (ij -> ji) чтобы удобнее считать пути до текущей
        headMatrix.setdiag(0)  # not include self attention weight

        attMin = min(headMatrix.data)
        attMax = max(headMatrix.data)
        headMatrix.data = (headMatrix.data - attMin) / (attMax - attMin)

        print(min(headMatrix.data), max(headMatrix.data))
        # TODO:
        nodeN = headMatrix.shape[0]  # кол-во вершин
        edgeN = headMatrix.data.shape[0]  # кол-во ребер
        weightM = headMatrix.data  # Массив весов графа. в разреженной форме
        print(nodeN, edgeN)
        # Код ниже - это просто ресерч путей и далее. Отсюда можно не читать, все веса обновленные.
        # G = nx.from_scipy_sparse_matrix(headMatrix)
        t = time.time()
        # pathRes = dict(nx.all_pairs_dijkstra(G, cutoff=3))
        pathRes = graph_tool_apsp(headMatrix, cutoff=3)
        print('shortest path time:', time.time() - t)

        # single_path: sparse matrix N*N*pathLen
        t = time.time()
        lenPathM = {}
        for pathLen in range(2, 4):  # generate path 2,3,...
            indexValue = {}
            single_path, _ = single_gen_path(pathRes, headMatrix.shape, pathLen=pathLen, Nratio=Nratio, Ndeg=Ndeg)
            indexValue['indices'] = single_path._indices()
            indexValue['values'] = single_path._values()
            print("pathlen:{:d}, #{:d}".format(pathLen, indexValue['indices'].shape[1]))
            lenPathM[pathLen] = indexValue
        headPathM[0] = lenPathM
        print('generate path time:', time.time() - t)

        layerPathM[layer] = headPathM
    pathM = layerPathM
    # pytorch doesn't support sparse matrix save, so we need to save it as indices and values
    with open(os.path.join(matpath, 'pathDict.pickle'), 'wb') as pathfile:
        pickle.dump(pathM, pathfile)
        print('pathM saved')
    return pathM


__all__ = ["gen_pathm"]
