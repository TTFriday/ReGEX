# -*- coding: utf-8 -*-
"""
evaluate.py — evaluation of edge explanations at matched sparsity budgets.

For a budget fraction f, every method keeps its globally top-K edges
(K = round(f * |E_sub|), ties broken at random) as the explanation S.
Fidelity follows the probability-space definitions of GraphFramEx
(Amara et al., 2022), with y the class predicted on the full subgraph:

  Fid+ (necessity, higher is better)   p_y(G) - p_y(G \\ S)
  Fid- (sufficiency, lower is better)  p_y(G) - p_y(S)

and the characterisation score is the harmonic mean of Fid+ and 1 - Fid-.

Gene-level evidence consistency (EGC) correlates, over the genes of the
subgraph, the fraction of each gene's incident edges that are selected with
an independent per-gene omics evidence count (Spearman). Normalising by the
gene's subgraph degree removes the trivial association between degree and
evidence that any edge selector inherits; the un-normalised variant is also
returned for reference.
"""
import numpy as np
import torch
from scipy.stats import spearmanr
from masked_forward import hard_logit


def topk_keep(q, weights, frac, rng=None):
    """Boolean keep-mask dict for the global top-K edges of ``weights``."""
    flat = np.concatenate([np.asarray(weights[et], dtype=np.float64) for et in q.etypes])
    K = int(round(frac * len(flat)))
    K = max(1, min(K, len(flat)))
    rng = rng or np.random.default_rng(0)
    tie = rng.random(len(flat)) * 1e-9 * (np.abs(flat).max() + 1e-12)
    order = np.argsort(-(flat + tie), kind="stable")
    sel = np.zeros(len(flat), dtype=bool)
    sel[order[:K]] = True
    return q.split(sel), K


def _p_y(logit, y):
    p = 1.0 / (1.0 + np.exp(-logit))
    return p if y == 1.0 else 1.0 - p


def fidelity(q, keep):
    """Probability-space Fid+ / Fid- for one keep-mask."""
    data, sub = q.data, q.sub
    drop = {et: ~keep[et] for et in keep}
    s_keep = hard_logit(q.model, sub, data, q.a, q.b, keep, q.pair_types)
    s_drop = hard_logit(q.model, sub, data, q.a, q.b, drop, q.pair_types)
    p_full = _p_y(q.logit_full, q.y_hat)
    return {"fid_plus": p_full - _p_y(s_drop, q.y_hat),
            "fid_minus": p_full - _p_y(s_keep, q.y_hat),
            "p_full": p_full, "p_keep": _p_y(s_keep, q.y_hat),
            "p_drop": _p_y(s_drop, q.y_hat)}


def charact(fid_plus, fid_minus):
    a, b = np.clip(fid_plus, 0, 1), np.clip(1 - fid_minus, 0, 1)
    return 0.0 if a + b == 0 else 2 * a * b / (a + b + 1e-12)


def egc(q, keep, evidence, local, min_genes=10):
    """(EGC_norm, EGC_raw) for one keep-mask; NaN when undefined."""
    n_gene = q.sub["gene"].num_nodes if "gene" in q.sub.node_types else 0
    if n_gene == 0 or evidence is None:
        return np.nan, np.nan
    deg = np.zeros(n_gene); mass = np.zeros(n_gene)
    for et in q.etypes:
        ei = q.sub[et].edge_index.cpu().numpy()
        k = keep[et]
        for side, nt in ((0, et[0]), (1, et[2])):
            if nt == "gene":
                np.add.at(deg, ei[side], 1.0)
                np.add.at(mass, ei[side], k.astype(float))
    inv = {v: g for g, v in local["gene"].items()}
    ev = np.array([evidence.get(inv[i], 0.0) for i in range(n_gene)])

    def _rho(x, y):
        if len(x) < min_genes or np.std(x) == 0 or np.std(y) == 0:
            return np.nan
        return float(spearmanr(x, y).statistic)

    has = deg > 0
    norm = _rho(mass[has] / deg[has], ev[has])
    sel = mass > 0
    raw = _rho(mass[sel], ev[sel])
    return norm, raw


def evaluate(q, weights, fracs, evidence=None, local=None, rng=None):
    """Rows of metrics for one explanation across sparsity budgets."""
    rows = []
    for f in fracs:
        keep, K = topk_keep(q, weights, f, rng)
        r = fidelity(q, keep)
        e_norm, e_raw = egc(q, keep, evidence, local)
        r.update({"frac": f, "K": K, "n_edges": q.n_edges,
                  "charact": charact(r["fid_plus"], r["fid_minus"]),
                  "egc": e_norm, "egc_raw": e_raw})
        rows.append(r)
    return rows


def evaluate_batch(qb, weights, fracs, evidence=None, locals_=None, seed=0):
    """Batched ``evaluate``: ``weights`` is a list (one dict per query).
    Masked predictions use the exact (binary) masked forward, i.e. hard
    deletion of the non-selected / selected edges, for all queries at once.
    Returns a list (per query) of lists (per budget) of metric dicts."""
    out = [[] for _ in qb.qs]
    p_full = np.array([_p_y(q.logit_full, q.y_hat) for q in qb.qs])
    for f in fracs:
        keeps, Ks = [], []
        for i, (q, w) in enumerate(zip(qb.qs, weights)):
            k, K = topk_keep(q, w, f, np.random.default_rng(seed + i))
            keeps.append(k); Ks.append(K)
        flat = torch.tensor(qb.join(keeps), dtype=torch.float32, device=qb.dev)
        with torch.no_grad():
            s_keep = qb.logits(flat).cpu().numpy()
            s_drop = qb.logits(1.0 - flat).cpu().numpy()
        for i, q in enumerate(qb.qs):
            pk, pd_ = _p_y(s_keep[i], q.y_hat), _p_y(s_drop[i], q.y_hat)
            fp, fm = p_full[i] - pd_, p_full[i] - pk
            e_norm, e_raw = egc(q, keeps[i], evidence,
                                locals_[i] if locals_ is not None else None)
            out[i].append({"frac": f, "K": Ks[i], "n_edges": q.n_edges,
                           "fid_plus": fp, "fid_minus": fm, "charact": charact(fp, fm),
                           "p_full": p_full[i], "p_keep": pk, "p_drop": pd_,
                           "egc": e_norm, "egc_raw": e_raw})
    return out


def jaccard(q, w1, w2, frac=0.10):
    k1, _ = topk_keep(q, w1, frac)
    k2, _ = topk_keep(q, w2, frac)
    a = np.concatenate([k1[et] for et in q.etypes])
    b = np.concatenate([k2[et] for et in q.etypes])
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


def relation_composition(q, weights, frac=0.10):
    keep, K = topk_keep(q, weights, frac)
    return {et[1]: int(keep[et].sum()) for et in q.etypes}, \
           {et[1]: int(len(keep[et])) for et in q.etypes}
