# -*- coding: utf-8 -*-
"""
paths.py — explanation paths and their biological read-out.

The explanation subgraph of a method is its top-K edge set (same budget for
every method). On that subgraph we report
  * connectivity : whether the two query nodes are linked by a path of at most
                   MAX_PATH_LEN explanation edges (the direct query edge is
                   never part of the subgraph), and the shortest such length;
  * beam paths   : top-k paths from the query source node, scored by the
                   geometric mean of rank-normalised edge weights, preferring
                   paths that reach the query target or a curated gene;
  * curated hit  : fraction of beam paths passing through a curated gene other
                   than the query gene itself (AtMAD flowering module).
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path

import config as C


def _flat_offsets(q):
    offs, off = {}, 0
    for nt in q.data.node_types:
        offs[nt] = off
        off += q.sub[nt].num_nodes
    return offs, off


def explanation_paths(q, keep, weights, local, curated_global=None,
                      max_len=None, beam=None, top_k=None):
    max_len = C.MAX_PATH_LEN if max_len is None else max_len
    beam = C.BEAM_WIDTH if beam is None else beam
    top_k = C.TOP_PATHS if top_k is None else top_k
    offs, n = _flat_offsets(q)
    flat_w = np.concatenate([np.asarray(weights[et], dtype=np.float64) for et in q.etypes])
    pct = np.empty(len(flat_w)); pct[np.argsort(flat_w, kind="stable")] = (np.arange(len(flat_w)) + 1) / len(flat_w)
    ta, tb = q.pair_types
    src, dst = offs[ta] + q.a, offs[tb] + q.b
    adj, rows, cols, off = {}, [], [], 0
    for et in q.etypes:
        ei = q.sub[et].edge_index.cpu().numpy()
        k = keep[et]
        for j in np.where(k)[0]:
            u, v = int(ei[0, j]) + offs[et[0]], int(ei[1, j]) + offs[et[2]]
            wt = pct[off + j]
            rel = et[1][:-4] if et[1].endswith("_rev") else et[1]
            adj.setdefault(u, []).append((rel, v, wt))
            adj.setdefault(v, []).append((rel, u, wt))
            rows.append(u); cols.append(v)
        off += len(k)
    # connectivity (exact BFS on the explanation subgraph)
    if rows:
        A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
        d = shortest_path(A, unweighted=True, directed=False, indices=[src])[0, dst]
    else:
        d = np.inf
    connected = bool(np.isfinite(d) and d <= max_len)
    # curated targets (flat ids), excluding the query gene itself
    cur = set()
    if curated_global is not None and ta == "gene":
        for g in curated_global:
            if int(g) in local["gene"] and local["gene"][int(g)] != q.a:
                cur.add(offs["gene"] + local["gene"][int(g)])
    targets = cur | {dst}
    # beam search
    beams, finished = [(src, [src], [], 1.0, False)], []
    for _ in range(max_len):
        nxt = []
        for node, nodes, rels, sc, hit in beams:
            for rel, nb, wt in adj.get(node, []):
                if nb in nodes:
                    continue
                nxt.append((nb, nodes + [nb], rels + [rel], sc * wt, hit or nb in targets))
        nxt.sort(key=lambda x: (-x[4], -x[3] ** (1.0 / len(x[2]))))
        beams = nxt[:beam]
        finished += [b for b in beams if b[4]]
        if not beams:
            break
    finished.sort(key=lambda x: -x[3] ** (1.0 / len(x[2])))
    paths = finished[:top_k] if finished else beams[:top_k]
    n_cur = sum(any(v in cur for v in p[1][1:]) for p in paths)
    rel_hist = {}
    for p in paths:
        for r in p[2]:
            rel_hist[r] = rel_hist.get(r, 0) + 1
    return {"connected": connected, "conn_len": float(d) if np.isfinite(d) else np.nan,
            "n_paths": len(paths),
            "curated_hit": (n_cur / len(paths)) if (paths and cur) else np.nan,
            "n_curated_in_sub": len(cur), "rel_hist": rel_hist,
            "paths": [(p[1], p[2], p[3] ** (1.0 / max(len(p[2]), 1))) for p in paths]}
