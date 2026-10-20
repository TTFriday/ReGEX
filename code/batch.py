# -*- coding: utf-8 -*-
"""
batch.py — explain many queries at once on a disjoint union of subgraphs.

Subgraphs of different queries share no nodes, so a single forward pass on
their disjoint union yields exactly the per-query predictions, and the
gradient of the summed per-query losses w.r.t. one query's mask equals the
gradient of that query's own loss. Batching therefore changes nothing but
throughput (it lets the GPU process tens of small subgraphs in parallel).

Flat layout: masks of all queries are concatenated query-major, each query
in its own ``q.etypes`` order (the same layout ``Query.split`` expects).
"""
import numpy as np
import torch
from torch_geometric.data import Batch
from masked_forward import masked_encode


class QueryBatch:
    def __init__(self, queries):
        self.qs = queries
        q0 = queries[0]
        self.model, self.data, self.pair_types = q0.model, q0.data, q0.pair_types
        self.dev = q0.dev
        self.B = len(queries)
        self.big = Batch.from_data_list([q.sub for q in queries]).to(self.dev)
        self.edge_types = list(self.data.edge_types)
        ta, tb = self.pair_types
        # node offsets of every query inside the union graph
        a_idx, b_idx = [], []
        for i, q in enumerate(queries):
            a_idx.append(int(self.big[ta].ptr[i]) + q.a)
            b_idx.append(int(self.big[tb].ptr[i]) + q.b)
        self.a_idx = torch.tensor(a_idx, device=self.dev)
        self.b_idx = torch.tensor(b_idx, device=self.dev)
        # flat (query-major) -> per-edge-type layout of the union graph
        self.n_per_q = [q.n_edges for q in queries]
        self.start = np.concatenate([[0], np.cumsum(self.n_per_q)[:-1]]).astype(int)
        self.N = int(sum(self.n_per_q))
        perm = {et: [] for et in self.edge_types}
        for i, q in enumerate(queries):
            off = self.start[i]
            for et, n in zip(q.etypes, q.sizes):
                perm[et].append(np.arange(off, off + n))
                off += n
        self.perm = {et: torch.tensor(np.concatenate(v), dtype=torch.long, device=self.dev)
                     for et, v in perm.items() if v}
        # segment ids
        self.seg_q = torch.repeat_interleave(
            torch.arange(self.B, device=self.dev), torch.tensor(self.n_per_q, device=self.dev))
        qt, n_qt, base = [], 0, []
        for q in queries:
            base.append(n_qt)
            qt.append(np.repeat(np.arange(len(q.sizes)) + n_qt, q.sizes))
            n_qt += len(q.sizes)
        self.seg_qt = torch.tensor(np.concatenate(qt), dtype=torch.long, device=self.dev)
        self.qt_owner = torch.tensor(np.concatenate(
            [np.full(len(q.sizes), i) for i, q in enumerate(queries)]), device=self.dev)
        self.q_rel = torch.tensor(np.concatenate([q.q_rel for q in queries]),
                                  dtype=torch.float32, device=self.dev)
        self.n_qt = n_qt
        self.dist = torch.tensor(np.concatenate([q.dist / max(q.dist.max(), 1.0) for q in queries]),
                                 dtype=torch.float32, device=self.dev)
        self.raw_dist = np.concatenate([q.dist for q in queries])
        self.y_hat = torch.tensor([q.y_hat for q in queries], dtype=torch.float32, device=self.dev)
        self.counts = torch.tensor(self.n_per_q, dtype=torch.float32, device=self.dev)

    # ------------------------------------------------------------------
    def masks(self, flat):
        return {et: flat[idx] for et, idx in self.perm.items()}

    def logits(self, flat=None):
        masks = None if flat is None else self.masks(flat)
        emb = masked_encode(self.model, self.big, self.data, masks)
        ta, tb = self.pair_types
        return (emb[ta][self.a_idx] * emb[tb][self.b_idx]).sum(-1)

    def qmean(self, v):
        """per-query mean of a flat vector -> [B]"""
        return torch.zeros(self.B, device=self.dev).index_add(0, self.seg_q, v) / self.counts

    def split(self, flat):
        """flat tensor/array -> list of per-query numpy weight dicts"""
        x = flat.detach().cpu().numpy() if torch.is_tensor(flat) else np.asarray(flat)
        return [q.to_numpy_dict(x[s:s + n]) for q, s, n in zip(self.qs, self.start, self.n_per_q)]

    def join(self, per_query):
        """list of per-query weight dicts -> flat numpy vector"""
        return np.concatenate([np.concatenate([np.asarray(w[et], dtype=np.float64) for et in q.etypes])
                               for q, w in zip(self.qs, per_query)])


def chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]
