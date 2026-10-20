# -*- coding: utf-8 -*-
"""
models.py — graph neural network baselines for the AtMAD benchmark.

Baselines
---------
MLP : node-feature-only multilayer perceptron (no graph structure)
GCN : 2-layer GCN on the homogeneous projection of the graph
GAT : 2-layer GAT on the homogeneous projection
RGCN: relational GCN (heterogeneous relations preserved)
HGT : heterogeneous graph transformer (relation-aware attention)

Link score: inner product of the gene and phenotype node embeddings.
encode(data, x_dict, edge_index_dict=None) supports edge masking (ablation).
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, RGCNConv, HGTConv


def type_offsets(data):
    """Start offset of each node type in the flattened node order."""
    offs, off = {}, 0
    for nt in data.node_types:
        offs[nt] = off
        off += data[nt].num_nodes
    return offs


def flatten_edges(edge_index_dict, data):
    """Flatten a HeteroData edge dict into one edge_index + edge_type vector."""
    offs = type_offsets(data)
    eis, ets = [], []
    types = list(edge_index_dict.keys())
    for t, (src, rel, dst) in enumerate(types):
        ei = edge_index_dict[(src, rel, dst)]
        eis.append(torch.stack([ei[0] + offs[src], ei[1] + offs[dst]]))
        ets.append(torch.full((ei.shape[1],), t, dtype=torch.long))
    return torch.cat(eis, dim=1), torch.cat(ets)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=0.3):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden)
        self.lin2 = nn.Linear(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index=None):
        x = F.dropout(F.relu(self.lin1(x)), p=self.dropout, training=self.training)
        return self.lin2(x)


class HomoGNN(nn.Module):
    """GCN, GAT or GraphSAGE on the homogeneous collapsed graph."""
    def __init__(self, in_dim, hidden, out_dim, kind="gcn", num_heads=2,
                 dropout=0.3, num_layers=2):
        super().__init__()
        self.kind = kind
        self.num_layers = num_layers
        if kind == "gcn":
            conv = GCNConv
        elif kind == "gat":
            conv = GATConv
        elif kind == "sage":
            conv = SAGEConv
        else:
            raise ValueError(kind)
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            d_in = in_dim if i == 0 else hidden
            d_out = hidden if i < num_layers - 1 else out_dim
            if kind == "gat":
                self.convs.append(conv(d_in, d_out, heads=num_heads,
                                       concat=False, dropout=dropout))
            else:
                self.convs.append(conv(d_in, d_out))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.dropout(F.relu(x), p=self.dropout, training=self.training)
        return x


class RGCN(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, num_relations, num_bases=8,
                 dropout=0.3, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            d_in = in_dim if i == 0 else hidden
            d_out = hidden if i < num_layers - 1 else out_dim
            self.convs.append(RGCNConv(d_in, d_out, num_relations,
                                       num_bases=num_bases))
        self.dropout = dropout

    def forward(self, x, edge_index, edge_type):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if i < len(self.convs) - 1:
                x = F.dropout(F.relu(x), p=self.dropout, training=self.training)
        return x


class HGTModel(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, metadata, num_heads=4, dropout=0.3):
        super().__init__()
        self.lin_in = nn.Linear(in_dim, hidden)
        self.conv1 = HGTConv(hidden, hidden, metadata, heads=num_heads)
        self.conv2 = HGTConv(hidden, hidden, metadata, heads=num_heads)
        self.lin_out = nn.Linear(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x_dict, edge_index_dict):
        x_dict = {k: F.relu(self.lin_in(v)) for k, v in x_dict.items()}
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.dropout(F.relu(v), p=self.dropout, training=self.training)
                  for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return {k: self.lin_out(v) for k, v in x_dict.items()}


class DeepWalkN2V(nn.Module):
    """DeepWalk-style skip-gram with negative sampling (node2vec with p=q=1),
    implemented with sparse embeddings and vectorised random walks so that the
    full 118k-node graph remains tractable on CPU. Used as the non-GNN
    shallow graph-embedding baseline of the benchmark."""
    def __init__(self, n_nodes, dim, indptr, indices, walk_length=12,
                 window=2, walks_per_node=2, num_neg=5):
        super().__init__()
        self.n_nodes = n_nodes
        self.dim = dim
        self.walk_length = walk_length
        self.window = window
        self.walks_per_node = walks_per_node
        self.num_neg = num_neg
        self.register_buffer("indptr", torch.from_numpy(indptr))
        self.register_buffer("indices", torch.from_numpy(indices))
        self.emb = nn.Embedding(n_nodes, dim, sparse=True)
        nn.init.xavier_uniform_(self.emb.weight)

    @torch.no_grad()
    def sample_walks(self, rng):
        n_w = self.n_nodes * self.walks_per_node
        indptr = self.indptr.numpy(); indices = self.indices.numpy()
        cur = np.repeat(np.arange(self.n_nodes), self.walks_per_node)
        walks = np.zeros((n_w, self.walk_length), dtype=np.int64)
        walks[:, 0] = cur
        for t in range(1, self.walk_length):
            d = indptr[cur + 1] - indptr[cur]
            d = np.maximum(d, 1)
            off = indptr[cur] + (rng.random(n_w) * d).astype(np.int64)
            off = np.minimum(off, len(indices) - 1)
            cur = indices[off]
            walks[:, t] = cur
        return walks

    @torch.no_grad()
    def pairs_from_walks(self, walks):
        """centre-context pairs (skip-gram, window w) from sampled walks."""
        flat = walks.reshape(-1)
        n = flat.shape[0]
        centers, contexts = [], []
        for w in range(1, self.window + 1):
            centers.append(flat[w:]); contexts.append(flat[:-w])
            centers.append(flat[:-w]); contexts.append(flat[w:])
        return np.concatenate(centers).astype(np.int64), \
               np.concatenate(contexts).astype(np.int64)

    def loss_batch(self, centers, contexts, negatives):
        u = self.emb(centers)
        v = self.emb(contexts)
        nv = self.emb(negatives)                     # [B, neg, dim]
        pos = (u * v).sum(1)
        neg = torch.bmm(nv, u.unsqueeze(2)).squeeze(2)   # [B, neg]
        lp = F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
        ln = F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg))
        return lp + ln


class LinkPredictor(nn.Module):
    """Per-type encoders + GNN + dot-product link scorer."""
    def __init__(self, feats, hidden, out_dim, model_name="rgcn", metadata=None,
                 num_relations=None, num_heads=2, dropout=0.3, num_layers=2):
        super().__init__()
        self.model_name = model_name
        self.encoders = nn.ModuleDict({nt: nn.Linear(d, hidden) for nt, d in feats.items()})
        if model_name == "mlp":
            self.gnn = MLP(hidden, hidden, out_dim, dropout=dropout)
        elif model_name in ("gcn", "gat", "sage"):
            self.gnn = HomoGNN(hidden, hidden, out_dim, kind=model_name,
                               num_heads=num_heads, dropout=dropout,
                               num_layers=num_layers)
        elif model_name == "rgcn":
            self.gnn = RGCN(hidden, hidden, out_dim, num_relations=num_relations,
                            dropout=dropout, num_layers=num_layers)
        elif model_name == "hgt":
            self.gnn = HGTModel(hidden, hidden, out_dim, metadata,
                                num_heads=num_heads, dropout=dropout)
        self.dropout = dropout

    def _proj_all(self, x_dict, data):
        xs, offsets = [], {}
        off = 0
        for nt in data.node_types:
            offsets[nt] = off
            xs.append(self.encoders[nt](x_dict[nt]))
            off += data[nt].num_nodes
        return torch.cat(xs, dim=0), offsets

    def encode(self, data, x_dict, edge_index_dict=None):
        eid = edge_index_dict if edge_index_dict is not None \
            else {et: v.edge_index for et, v in data.edge_items()}
        if self.model_name == "mlp":
            x, _ = self._proj_all(x_dict, data)
            h = self.gnn(x)
            offsets, _ = self._offsets(data)
            return {nt: h[offsets[nt]:offsets[nt] + data[nt].num_nodes]
                    for nt in data.node_types}
        if self.model_name == "hgt":
            x_in = {nt: self.encoders[nt](x_dict[nt]) for nt in x_dict}
            return self.gnn(x_in, eid)
        x, offsets = self._proj_all(x_dict, data)
        if self.model_name == "rgcn":
            ei, et = flatten_edges(eid, data)
            h = self.gnn(x, ei, et)
        else:  # gcn / gat / sage
            offs = type_offsets(data)
            eis = []
            for (src, rel, dst), ei in eid.items():
                eis.append(torch.stack([ei[0] + offs[src], ei[1] + offs[dst]]))
            h = self.gnn(x, torch.cat(eis, dim=1))
        return {nt: h[offsets[nt]:offsets[nt] + data[nt].num_nodes]
                for nt in data.node_types}

    def _offsets(self, data):
        offsets, off = {}, 0
        for nt in data.node_types:
            offsets[nt] = off
            off += data[nt].num_nodes
        return offsets, off

    def score_links(self, gene_emb, pheno_emb, gene_idx, pheno_idx):
        g = gene_emb[gene_idx]
        p = pheno_emb[pheno_idx]
        return (g * p).sum(dim=1)
