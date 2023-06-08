import math
import os
import time
from pathlib import Path

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch_geometric.utils import convert
from torch_scatter import scatter_add, scatter_max
from torch_sparse import spmm


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True, layerN='', device="cpu"):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.layerN = layerN

        self.device = device

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))  # проектор, самый обычный линейный
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))  # x2 тк вектора h_i и h_j сначала конкатенируются
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.bias = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_normal_(self.bias.data, gain=1.414)

        # attention param for path
        self.pathW = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.pathW.data, gain=1.414)

        self.pathbias = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_normal_(self.pathbias.data, gain=1.414)

        self.patha = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.patha.data, gain=1.414)

        self.patha_2 = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.patha_2.data, gain=1.414)

        self.patha_3 = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.patha_3.data, gain=1.414)

        self.pathMerge = nn.Parameter(torch.zeros(size=(2 * out_features, out_features)))
        nn.init.xavier_normal_(self.pathMerge.data, gain=1.414)

        self.lenAtt = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.lenAtt.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def gat_layer(self, input, adj, genPath=False, eluF=True):
        # input.shape == num_vertices (N) x hf (hidden_feature)
        N = input.size()[0]  # num of vertices
        edge = adj._indices()  # Возвращает двухмерный тензор, где
        # указаны ребра графа (в виде индексов), ведущие в вершины
        # edge[0][s] == FROM
        # edge[1][s] == TO
        # adj[FROM][TO]  != 0
        h = torch.mm(input, self.W)  # N x projected (=8)
        h = h + self.bias  # h: N x out
        # Содержит для каждой вершины фичу (вектор) из projected (8) размерностей

        # Self-attention on the nodes - Shared attention mechanism
        # Как и в статье. Сначала конкатенируем вектора (фичи)
        # Обращаем внимание, что мы берем только те вершины, откуда есть ребро
        # (edge[0][i], edge[1][i]) -> индексы вершин для i-го ребра
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()  # edge_h: 2*D x (num_edges)
        # Затем умножаем их на shared_attention вектор
        edge_att = self.a.mm(edge_h).squeeze()  # Скалярное произведение. shape == [108365]
        # pos = 22
        # edge_att[pos] = alpha_ij между вершинами adj[edge[0][pos]][edge[1][pos]]
        # Применяя по дороге relu
        edge_e_a = self.leakyrelu(edge_att)  # edge_e_a: E   attetion score for each edge
        # pos = 22
        # edge_e_a[pos] -> вес ребра между вершинами
        # FROM = adj[edge[0][pos]]
        # TO = adj[edge[1][pos]]
        # adj[edge]
        if genPath:
            with torch.no_grad():
                edge_weight = edge_e_a
                # Обрати внимание `scatter_max`. Здесь вторым аргументом идут
                # индексы всех вершин, из которых исходят ребра (в порядке возрастания).
                #
                # (1) out𝑖=max(out𝑖,max𝑗(src𝑗))
                # см. https://pytorch-scatter.readthedocs.io/en/1.3.0/functions/max.html
                # edge[0, :]
                # Поэтому мы выбираем максимум среди всех выходящих ребер вершины.
                # Для этого рассматриваем максимум, спроецированный на индексы edge[0, :]
                #
                # В самом конце, когда
                p_a_e = edge_weight - scatter_max(edge_weight, edge[0, :], dim=0, dim_size=N)[0][edge[0, :]]
                p_a_e = p_a_e.exp()
                p_a_e = p_a_e / (
                    scatter_add(p_a_e, edge[0, :], dim=0, dim_size=N)[edge[0, :]] + torch.Tensor([9e-15]).to(self.device)
                )

                scisp = convert.to_scipy_sparse_matrix(edge, p_a_e, N)
                # TODO: В процессе самого forward_pass - пересохраняем веса всего графа в сжатом виде.
                scipy.sparse.save_npz(str(Path(os.getcwd()) / "weights" / "att" / f'attmat_{self.layerN}'), scisp)

        edge_e = torch.exp(edge_e_a - torch.max(edge_e_a))  # edge_e: E
        e_rowsum = spmm(index=edge, value=edge_e, m=N, n=N, matrix=torch.ones(size=(N, 1)).to(self.device))  # e_rowsum: N x 1
        # Строчка выше делает простую вещь: она умножает матрицу смежности в которой a_i_j - вес ребра между v_i и v_j
        # умножает ее на вектор строчку из единиц. Че получается по итогу? Правильно, сумма весов, исходящих из i-=ой вершины
        #
        edge_e = self.dropout(edge_e)  # add dropout improve from 82.4 to 83.8
        # edge_e: E
        # Ниже происходит супер простая вещь.
        # У нас уже есть веса, которые проставились через вектор-фичи вершин (скалярное произведение `self.a` и конката двух векторов вершин)
        # Теперь надо сделать для каждой вершины обновление эмбеддинга.
        # (как в уравнении  (3))
        # Сумму всех выходящих весов, возведенных в exp - мы уже сделали. Осталось вычислить числитель
        # Основываясь на сгенерированных весах ребер - умножаем их
        # Строчка ниже делает сумму всех соседних эмбеддингов, умноженных на соответствующий вес
        h_prime = spmm(index=edge, value=edge_e, m=N, n=N, matrix=h)  # Все еще N x (out_features)
        h_prime = h_prime.div(e_rowsum + torch.Tensor([9e-15]).to(self.device))  # h_prime: N x out
        print()
        if self.concat and eluF:
            return F.elu(h_prime)
        else:
            return h_prime

    def pathat_layer(self, input, pathM, pathlens, eluF=True):
        N = input.size()[0]
        pathh = torch.mm(input, self.pathW)
        pathh = pathh + self.pathbias  # h: N x out

        if not self.concat:  # if the last layer
            pathlens = [2]

        pathfeat_all = None
        for pathlen_iter in pathlens:
            i = pathM[pathlen_iter]['indices']
            v = pathM[pathlen_iter]['values']
            featlen = pathh.shape[1]
            pathlen = v.shape[1]
            pathfeat = tuple((pathh[v[:, i], :] for i in range(1, pathlen)))
            pathfeat = torch.cat(pathfeat, dim=1)
            pathfeat = pathfeat.view(-1, pathlen - 1, featlen)
            pathfeat, _ = torch.max(pathfeat, dim=1)  # seems max is better?
            # pathfeat = torch.mean(pathfeat, dim=1)     #
            att_feat = torch.cat((pathfeat, pathh[i[0, :], :]), dim=1).t()
            if pathlen_iter == 2:
                path_att = self.leakyrelu(self.patha_2.mm(att_feat).squeeze())
            else:
                path_att = self.leakyrelu(self.patha_3.mm(att_feat).squeeze())
            # softmax of p_a -> p_a_e
            path_att = path_att - scatter_max(path_att, i[0, :], dim=0, dim_size=N)[0][i[0, :]]
            path_att = path_att.exp()
            path_att = path_att / (scatter_add(path_att, i[0, :], dim=0, dim_size=N)[i[0, :]] + torch.Tensor([9e-15]).cuda())
            path_att = path_att.view(-1, 1)
            path_att = self.dropout(path_att)  # add dropout here of p_a_e
            w_pathfeat = torch.mul(pathfeat, path_att)
            h_path_prime = scatter_add(w_pathfeat, i[0, :], dim=0)
            # h_path_prime is the feature embedded from paths  N*feat

            if pathfeat_all is None:
                pathfeat_all = h_path_prime
            else:
                pathfeat_all = torch.cat((pathfeat_all, h_path_prime), dim=0)

        if len(pathlens) == 2:
            leni = torch.tensor(np.array(list(range(N)) + list(range(N)))).cuda()

            att_feat = torch.cat((pathfeat_all, pathh[leni, :]), dim=1).t()
            path_att = self.leakyrelu(self.lenAtt.mm(att_feat).squeeze())
            # softmax of p_a -> p_a_e
            path_att = path_att - scatter_max(path_att, leni, dim=0, dim_size=N)[0][leni]
            path_att = path_att.exp()
            path_att = path_att / (scatter_add(path_att, leni, dim=0, dim_size=N)[leni] + torch.Tensor([9e-15]).cuda())
            path_att = path_att.view(-1, 1)
            # path_att = self.dropout(path_att)         # add dropout here of p_a_e
            w_pathfeat = torch.mul(pathfeat_all, path_att)
            h_path_prime = scatter_add(w_pathfeat, leni, dim=0)

        if self.concat and eluF:
            return F.elu(h_path_prime)
        else:
            return h_path_prime

    def forward(self, input, adj, pathM, pathlens=[2], genPath=False, mode='GAT'):
        if not self.concat:  # if the last layer
            pathM = {}
            pathM[2] = {}
            pathM[2]['indices'] = adj._indices()
            pathM[2]['values'] = adj._indices().transpose(1, 0)

        return self.gat_layer(input, adj, genPath=genPath)


__all__ = ["SpGraphAttentionLayer"]
