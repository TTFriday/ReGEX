# -*- coding: utf-8 -*-
"""
explainers.py — ReGEX and baseline edge explainers for heterogeneous GNN
link predictors.

All methods return, for one query pair, a weight for every edge of the same
bounded 2-hop subgraph ({edge_type: np.ndarray}); weights are never
re-normalised per relation type, so the global top-K selection used for
evaluation treats every method identically.

Methods
-------
ReGEX          Gumbel-sigmoid edge mask optimised through a differentiable
               masked forward pass (masked_forward.py) under
                 L = BCE(s(m), y)                                  sufficiency
                   + l_n * BCE(s(1 - m), 1 - y)                    necessity
                   + l_s * mean(m) + l_e * mean H(m)               sparsity
                   + l_g * mean(m * d_e / d_max)                   geometry
                   + l_r * KL(q_hat(m) || q)                       relation
               Final weights are the deterministic sigmoid(logits).
GNNExplainer   sigmoid edge mask, fidelity + size + entropy only.
PGExplainer    amortised MLP over [h_u, h_v, h_a, h_b] edge embeddings,
               trained once per encoder on training pairs.
Saliency       |d logit / d m_e| at m = 1.
IG             integrated gradients of the logit w.r.t. the edge mask along
               m = alpha * 1, alpha in [0, 1] (signed toward predicted class).
Distance       -(graph distance of the edge to the query pair) + tie noise.
Random         i.i.d. uniform weights.
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
sys.path.insert(0, C.P2_CODE_DIR)          # models.py lives in Paper-2 code/
from models import LinkPredictor
from torch_geometric.data import HeteroData
from masked_forward import masked_encode, masked_logit


# --------------------------------------------------------------------------
# subgraph construction
# --------------------------------------------------------------------------
_ADJ_CACHE = {}


def _adjacency(data):
    key = id(data)
    if key not in _ADJ_CACHE:
        adj = {}
        for (s, r, d), v in data.edge_items():
            if r.endswith("_rev"):
                continue  # reverse edges are added explicitly below
            ei = v.edge_index
            for a, b in zip(ei[0].tolist(), ei[1].tolist()):
                adj.setdefault((s, a), []).append((r, (d, b)))
                adj.setdefault((d, b), []).append((r + "_rev", (s, a)))
        _ADJ_CACHE[key] = adj
    return _ADJ_CACHE[key]


def build_subgraph(data, a_idx, b_idx, pair_types=("gene", "phenotype"),
                   max_edges=C.MAX_SUBGRAPH_EDGES, hop=C.TOP_HOP, max_deg=50,
                   drop_query_edge=True):
    """Bounded 2-hop subgraph around a query pair (a, b).

    Neighbourhood growth is degree-capped (max_deg sampled edges per node and
    hop, fixed RNG seed) so that subgraphs stay tractable. Any edge directly
    linking the two query nodes is removed so that a prediction is never
    explained by the association it predicts.
    Returns (sub, keep, local, edges) with local[nt][global_idx] -> local idx.
    """
    adj = _adjacency(data)
    ta, tb = pair_types
    qa, qb = (ta, int(a_idx)), (tb, int(b_idx))
    rng = np.random.default_rng(0)
    keep = {qa, qb}
    frontier = [qa, qb]
    seen_e = set()
    for _ in range(hop):
        nxt = []
        for u in frontier:
            nbrs = adj.get(u, [])
            if len(nbrs) > max_deg:
                pick = rng.choice(len(nbrs), max_deg, replace=False)
                nbrs = [nbrs[i] for i in pick]
            for r, w in nbrs:
                if drop_query_edge and {u, w} == {qa, qb}:
                    continue
                if (u, r, w) in seen_e:
                    continue
                seen_e.add((u, r, w))
                if w not in keep:
                    keep.add(w)
                    nxt.append(w)
        frontier = nxt
        if not frontier:
            break
    keep = sorted(keep)
    edges = sorted(seen_e)
    if len(edges) > max_edges:
        rng2 = np.random.default_rng(0)
        edges = [edges[i] for i in sorted(rng2.choice(len(edges), max_edges, replace=False))]

    local = {nt: {} for nt in data.node_types}
    for nt in data.node_types:
        ids = [idx for (t, idx) in keep if t == nt]
        local[nt] = {idx: k for k, idx in enumerate(ids)}
    sub = HeteroData()
    for nt in data.node_types:
        ids = [idx for (t, idx) in keep if t == nt]
        sub[nt].x = data[nt].x[torch.tensor(ids, dtype=torch.long)] if ids else \
            torch.zeros((0, data[nt].x.shape[1]))
        sub[nt].num_nodes = len(ids)
    by_type = {}
    for ((t_a, a), r, (t_b, b)) in edges:
        by_type.setdefault((t_a, r, t_b), []).append((local[t_a][a], local[t_b][b]))
    for et in data.edge_types:
        if et in by_type:
            sub[et].edge_index = torch.tensor(by_type[et], dtype=torch.long).t().contiguous()
        else:  # keep every edge type present so that subgraphs can be batched
            sub[et].edge_index = torch.zeros((2, 0), dtype=torch.long)
    return sub, keep, local, edges


# --------------------------------------------------------------------------
# model restore
# --------------------------------------------------------------------------
def restore_model(model_name, data, ckpt, device="cpu"):
    feats = {nt: data[nt].x.shape[1] for nt in data.node_types}
    metadata = (list(data.node_types), list(data.edge_types))
    model = LinkPredictor(feats, C.HIDDEN, C.HIDDEN, model_name=model_name,
                          metadata=metadata, num_relations=len(data.edge_types),
                          num_heads=2, dropout=C.DROPOUT, num_layers=C.GNN_LAYERS)
    model.load_state_dict(ckpt["state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device)


# --------------------------------------------------------------------------
# per-query context
# --------------------------------------------------------------------------
class Query:
    """Everything an explainer needs for one (a, b) query on one encoder."""

    def __init__(self, model, data, sub, a_local, b_local,
                 pair_types=("gene", "phenotype")):
        self.model, self.data, self.sub = model, data, sub
        self.a, self.b, self.pair_types = a_local, b_local, pair_types
        self.dev = next(model.parameters()).device
        self.etypes = [et for et in sub.edge_types if sub[et].edge_index.shape[1] > 0]
        self.sizes = [sub[et].edge_index.shape[1] for et in self.etypes]
        self.n_edges = int(sum(self.sizes))
        if self.n_edges == 0:
            raise ValueError("Edge explanation undefined for a query subgraph with no edges")
        with torch.no_grad():
            self.logit_full = float(self.logit(None))
        self.y_hat = 1.0 if self.logit_full >= 0 else 0.0
        self.dist = self._edge_distance()            # np array, concat order
        self.q_rel = np.array(self.sizes, dtype=np.float64) / max(self.n_edges, 1)

    # ---- masks <-> flat vectors -------------------------------------------
    def split(self, flat):
        out, off = {}, 0
        for et, n in zip(self.etypes, self.sizes):
            out[et] = flat[off:off + n]
            off += n
        return out

    def to_numpy_dict(self, flat):
        flat = flat.detach().cpu().numpy() if torch.is_tensor(flat) else np.asarray(flat)
        return {et: v.astype(np.float64) for et, v in self.split(flat).items()}

    def logit(self, flat_mask):
        masks = None if flat_mask is None else self.split(flat_mask)
        return masked_logit(self.model, self.sub, self.data, self.a, self.b,
                            masks, self.pair_types)

    def fidelity_loss(self, flat_mask):
        s = self.logit(flat_mask)
        return F.binary_cross_entropy_with_logits(
            s.view(1), torch.tensor([self.y_hat], device=self.dev)), s

    # ---- geometry -----------------------------------------------------------
    def _edge_distance(self):
        """min hop distance of each edge's endpoints to either query node,
        computed on the (undirected) subgraph with type-aware node ids."""
        offs, off = {}, 0
        for nt in self.data.node_types:
            offs[nt] = off
            off += self.sub[nt].num_nodes
        rows, cols = [], []
        for et in self.etypes:
            ei = self.sub[et].edge_index.cpu().numpy()
            rows.append(ei[0] + offs[et[0]]); cols.append(ei[1] + offs[et[2]])
        if not rows:
            return np.zeros(0)
        r = np.concatenate(rows); c = np.concatenate(cols)
        A = coo_matrix((np.ones(len(r)), (r, c)), shape=(off, off)).tocsr()
        A = A + A.T
        seeds = [offs[self.pair_types[0]] + self.a, offs[self.pair_types[1]] + self.b]
        D = shortest_path(A, unweighted=True, directed=False, indices=seeds).min(0)
        D[~np.isfinite(D)] = 3.0
        return np.minimum(D[r], D[c])


# --------------------------------------------------------------------------
# mask optimisation (ReGEX, its ablations and GNNExplainer) — batched
# --------------------------------------------------------------------------
def gumbel_sigmoid(logits, tau):
    u = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
    return torch.sigmoid((logits + torch.log(u) - torch.log1p(-u)) / tau)


def _entropy(m):
    m = m.clamp(1e-6, 1 - 1e-6)
    return -(m * torch.log(m) + (1 - m) * torch.log(1 - m))


def _relation_kl(qb, m):
    """per-query KL( mask-weighted relation distribution || subgraph one )"""
    mass = torch.zeros(qb.n_qt, device=qb.dev).index_add(0, qb.seg_qt, m)
    tot = torch.zeros(qb.B, device=qb.dev).index_add(0, qb.qt_owner, mass)
    q_hat = mass / tot[qb.qt_owner].clamp_min(1e-12)
    kl = q_hat * (torch.log(q_hat + 1e-12) - torch.log(qb.q_rel + 1e-12))
    return torch.zeros(qb.B, device=qb.dev).index_add(0, qb.qt_owner, kl)


def optimise_masks(qb, lam_size=None, lam_ent=None, lam_geom=None, lam_rel=None,
                   lam_nec=None, steps=None, lr=None, tau=None, gumbel=True, seed=None,
                   trace=False):
    """Optimise one edge mask per query of batch ``qb``.

    Returns (list of per-query weight dicts, trace rows). Each query's loss
    only depends on its own mask, so summing losses over the batch is
    equivalent to optimising every query separately.
    """
    lam_size = C.LAMBDA_SIZE if lam_size is None else lam_size
    lam_ent = C.LAMBDA_ENT if lam_ent is None else lam_ent
    lam_geom = C.LAMBDA_GEOM if lam_geom is None else lam_geom
    lam_rel = C.LAMBDA_REL if lam_rel is None else lam_rel
    lam_nec = C.LAMBDA_NEC if lam_nec is None else lam_nec
    steps = C.MASK_STEPS if steps is None else steps
    lr = C.MASK_LR if lr is None else lr
    tau = C.GUMBEL_TAU if tau is None else tau
    if seed is not None:
        torch.manual_seed(seed)
    logits = (0.1 * torch.randn(qb.N, device=qb.dev)).requires_grad_(True)
    opt = torch.optim.Adam([logits], lr=lr)
    rows = []
    for step in range(steps):
        opt.zero_grad()
        m = gumbel_sigmoid(logits, tau) if gumbel else torch.sigmoid(logits)
        s = qb.logits(m)
        l_fid = F.binary_cross_entropy_with_logits(s, qb.y_hat, reduction="none")
        if lam_nec > 0:
            l_nec = F.binary_cross_entropy_with_logits(qb.logits(1.0 - m), 1.0 - qb.y_hat,
                                                       reduction="none")
        else:
            l_nec = torch.zeros_like(l_fid)
        l_size = qb.qmean(m)
        l_ent = qb.qmean(_entropy(m))
        l_geom = qb.qmean(m * qb.dist)
        l_rel = _relation_kl(qb, m)
        per_q = l_fid + lam_nec * l_nec + lam_size * l_size + lam_ent * l_ent + lam_geom * l_geom + lam_rel * l_rel
        per_q.sum().backward()
        opt.step()
        if trace:
            with torch.no_grad():
                p = torch.sigmoid(s)
                p_y = torch.where(qb.y_hat > 0.5, p, 1 - p)
                n_half = torch.zeros(qb.B, device=qb.dev).index_add(
                    0, qb.seg_q, (torch.sigmoid(logits) > 0.5).float())
            for i in range(qb.B):
                rows.append({"query": i, "step": step, "loss": float(per_q[i]),
                             "l_fid": float(l_fid[i]), "l_nec": float(l_nec[i]),
                             "l_size": float(l_size[i]),
                             "l_ent": float(l_ent[i]), "l_geom": float(l_geom[i]),
                             "l_rel": float(l_rel[i]), "masked_p_y": float(p_y[i]),
                             "frac_above_half": float(n_half[i] / qb.counts[i])})
    return qb.split(torch.sigmoid(logits.detach())), rows


def regex_explain(qb, seed=None, **kw):
    return optimise_masks(qb, seed=seed, **kw)[0]


def gnnexplainer_explain(qb, seed=None):
    """GNNExplainer edge mask (Ying et al., 2019): plain sigmoid mask with
    fidelity, size and entropy terms only."""
    return optimise_masks(qb, lam_size=C.GNNEXP_LAMBDA_SIZE, lam_geom=0.0, lam_rel=0.0,
                          lam_nec=0.0, gumbel=False, seed=seed)[0]


# --------------------------------------------------------------------------
# gradient baselines on the edge mask
# --------------------------------------------------------------------------
def saliency_explain(qb):
    m = torch.ones(qb.N, device=qb.dev, requires_grad=True)
    (g,) = torch.autograd.grad(qb.logits(m).sum(), m)
    return qb.split(g.abs())


def ig_explain(qb, steps=256, integration="log"):
    """Edge-mask IG with an all-zero baseline, signed to the predicted
    class. Nonuniform midpoint quadrature resolves the near-zero clamp
    transitions (1 / in-degree); ordinary uniform quadrature is retained
    solely for the integration-convergence check."""
    sign = (2 * qb.y_hat - 1)[qb.seg_q]
    acc = torch.zeros(qb.N, device=qb.dev)
    if integration == "uniform":
        bounds = np.linspace(0.0, 1.0, steps + 1)
    else:
        bounds = np.r_[0.0, np.geomspace(1e-6, 1.0, steps)]
    for left, right in zip(bounds[:-1], bounds[1:]):
        alpha = (left + right) / 2
        m = torch.full((qb.N,), float(alpha), device=qb.dev, requires_grad=True)
        (g,) = torch.autograd.grad(qb.logits(m).sum(), m)
        acc += (right - left) * g
    return qb.split(sign * acc)


def distance_explain(qb, rng):
    return qb.split(-qb.raw_dist + 1e-3 * rng.random(qb.N))


def random_explain(qb, rng):
    return qb.split(rng.random(qb.N))


# --------------------------------------------------------------------------
# PGExplainer (amortised, Luo et al., 2020)
# --------------------------------------------------------------------------
class PGExplainerMLP(nn.Module):
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4 * dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def pg_inputs(qb):
    """[h_u, h_v, h_a, h_b] for every edge, in flat query-major layout."""
    with torch.no_grad():
        emb = masked_encode(qb.model, qb.big, qb.data, None)
        ta, tb = qb.pair_types
        ha, hb = emb[ta][qb.a_idx], emb[tb][qb.b_idx]
        X = torch.zeros(qb.N, 4 * ha.size(1), device=qb.dev)
        for et, idx in qb.perm.items():
            ei = qb.big[et].edge_index
            owner = qb.big[et[0]].batch[ei[0]]
            X[idx] = torch.cat([emb[et[0]][ei[0]], emb[et[2]][ei[1]], ha[owner], hb[owner]], 1)
    return X


def train_pgexplainer(batches, epochs=None, lr=3e-3, seed=0):
    """Fit one PGExplainer per encoder on batches of training-pair queries."""
    epochs = C.PG_EPOCHS if epochs is None else epochs
    torch.manual_seed(seed)
    pg = PGExplainerMLP(C.HIDDEN).to(batches[0].dev)
    opt = torch.optim.Adam(pg.parameters(), lr=lr)
    inputs = [pg_inputs(qb) for qb in batches]
    for ep in range(epochs):
        tau = 5.0 * (1.0 / 5.0) ** (ep / max(epochs - 1, 1))      # 5 -> 1
        for i in np.random.default_rng(seed + ep).permutation(len(batches)):
            qb, X = batches[i], inputs[i]
            opt.zero_grad()
            m = gumbel_sigmoid(pg(X), tau)
            s = qb.logits(m)
            loss = (F.binary_cross_entropy_with_logits(s, qb.y_hat, reduction="none")
                    + C.PG_LAMBDA_SIZE * qb.qmean(m) + C.LAMBDA_ENT * qb.qmean(_entropy(m))).mean()
            loss.backward()
            opt.step()
    pg.eval()
    return pg


def pgexplainer_explain(pg, qb):
    with torch.no_grad():
        return qb.split(torch.sigmoid(pg(pg_inputs(qb))))


# --------------------------------------------------------------------------
# PaGE-Link (Zhang et al., WWW 2023) — path-based explainer for
# heterogeneous link prediction, re-implemented on the shared masked forward
# --------------------------------------------------------------------------
def _pagelink_graph(q, k_core):
    """k-core pruned computation graph: per flat edge, kept flag, head-node
    degree and (u, v) flat node ids."""
    import networkx as nx
    offs, off = {}, 0
    for nt in q.data.node_types:
        offs[nt] = off
        off += q.sub[nt].num_nodes
    U, V = [], []
    for et in q.etypes:
        ei = q.sub[et].edge_index.cpu().numpy()
        U.append(ei[0] + offs[et[0]]); V.append(ei[1] + offs[et[2]])
    U, V = np.concatenate(U), np.concatenate(V)
    G = nx.Graph(); G.add_edges_from(zip(U.tolist(), V.tolist()))
    G.remove_edges_from(nx.selfloop_edges(G))
    src, dst = offs[q.pair_types[0]] + q.a, offs[q.pair_types[1]] + q.b
    core = set(nx.k_core(G, k_core).nodes()) | {src, dst}
    kept = np.array([u in core and v in core for u, v in zip(U, V)])
    deg = {}
    for u, v in zip(U[kept], V[kept]):
        deg[u] = deg.get(u, 0) + 1; deg[v] = deg.get(v, 0) + 1
    Dv = np.array([max(deg.get(v, 1), 1) for v in V], dtype=np.float64)
    return U, V, kept, Dv, src, dst


def _top_path_edges(U, V, kept, dist, src, dst, n_paths):
    import networkx as nx
    G = nx.DiGraph()
    for j in np.where(kept)[0]:
        u, v, w = int(U[j]), int(V[j]), float(dist[j])
        if not G.has_edge(u, v) or G[u][v]["w"] > w:
            G.add_edge(u, v, w=w, j=j)
    on = np.zeros(len(U), dtype=bool)
    if src not in G or dst not in G:
        return on
    try:
        for i, path in enumerate(nx.shortest_simple_paths(G, src, dst, weight="w")):
            for a, b in zip(path[:-1], path[1:]):
                on[G[a][b]["j"]] = True
            if i + 1 >= n_paths:
                break
    except nx.NetworkXNoPath:
        pass
    return on


def pagelink_explain(qb, alpha=None, beta=None, k_core=2, n_paths=5, steps=None, lr=None, seed=0):
    """Mask logits M on the k-core pruned subgraph, loss
         -log P(y | G (.) sigma(M))  -  (alpha * sum_{e in paths} M_e - beta * sum_{e not in paths} M_e)
    with 'paths' the top-5 shortest s-t paths under distance -Score(e),
    Score(e) = log sigma(M_e) - log D_v, recomputed every iteration.
    Edges are ranked by Score(e); pruned edges rank last."""
    alpha = C.PAGELINK_ALPHA if alpha is None else alpha
    beta = C.PAGELINK_BETA if beta is None else beta
    steps = C.MASK_STEPS if steps is None else steps
    lr = C.MASK_LR if lr is None else lr
    torch.manual_seed(seed)
    info = [_pagelink_graph(q, k_core) for q in qb.qs]
    kept = torch.tensor(np.concatenate([i[2] for i in info]), device=qb.dev)
    logD = torch.tensor(np.log(np.concatenate([i[3] for i in info])), dtype=torch.float32, device=qb.dev)
    M = (0.1 * torch.randn(qb.N, device=qb.dev)).requires_grad_(True)
    opt = torch.optim.Adam([M], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        m = torch.sigmoid(M) * kept
        s = qb.logits(m)
        l_pred = F.binary_cross_entropy_with_logits(s, qb.y_hat, reduction="none")
        with torch.no_grad():
            score = (F.logsigmoid(M) - logD).cpu().numpy()
        on = np.concatenate([_top_path_edges(U, V, kp, -score[st:st + n], src, dst, n_paths)
                             for (U, V, kp, _, src, dst), st, n in zip(info, qb.start, qb.n_per_q)])
        on = torch.tensor(on, device=qb.dev)
        Mk = M * kept
        l_path = -(alpha * torch.zeros(qb.B, device=qb.dev).index_add(0, qb.seg_q, Mk * on)
                   - beta * torch.zeros(qb.B, device=qb.dev).index_add(0, qb.seg_q, Mk * (~on) * kept))
        (l_pred + l_path).sum().backward()
        opt.step()
    with torch.no_grad():
        score = F.logsigmoid(M) - logD
        score = torch.where(kept, score, torch.full_like(score, -1e6))
    return qb.split(score)
