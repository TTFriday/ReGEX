# -*- coding: utf-8 -*-
"""
run_main.py — main benchmark: eight explainers x three encoders x all query
cohorts, evaluated at four matched sparsity budgets.

Outputs (results/tables/<task>/):
  main_<model>.csv      one row per (pair, method, budget)
  paths_<model>.csv     path read-out at the 10% budget, one row per (pair, method)
  weights_<model>.npz   edge weights of every explanation (for re-analysis)

Usage:
  python run_main.py --task atmad [--models sage rgcn hgt] [--smoke]
"""
import os, sys, time, json, argparse
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from tasks import load_task, all_pairs, apply_selected
from explainers import (build_subgraph, restore_model, Query, regex_explain,
                        gnnexplainer_explain, train_pgexplainer, pgexplainer_explain, pagelink_explain,
                        saliency_explain, ig_explain, distance_explain, random_explain)
from batch import QueryBatch, chunks
from evaluate import evaluate_batch, topk_keep
from paths import explanation_paths

METHODS = ["regex", "gnnexplainer", "pgexplainer", "pagelink", "saliency", "ig", "distance", "random"]


def build_queries(task, model, pairs, subs, dev):
    qs = []
    for k in range(pairs.shape[1]):
        sub, keep, local = subs[k]
        a, b = int(pairs[0, k]), int(pairs[1, k])
        ta, tb = task.pair_types
        q = Query(model, task.data, sub.to(dev), local[ta][a], local[tb][b], task.pair_types)
        q.local = local
        q.pair_id = k
        q.global_pair = (a, b)
        qs.append(q)
    return qs


def make_subgraphs(task, pairs):
    out = []
    for k in range(pairs.shape[1]):
        sub, keep, local, _ = build_subgraph(task.data, int(pairs[0, k]), int(pairs[1, k]),
                                             task.pair_types)
        out.append((sub, keep, local))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="atmad")
    ap.add_argument("--models", nargs="*", default=["sage", "rgcn", "hgt"])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--pg-batch", type=int, default=30,
                    help="PG training batch must match validation tuning (30)")
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dev = torch.device(C.DEVICE)
    task = load_task(args.task)
    sel = apply_selected(task.name)
    print("hyper-parameters:", sel, flush=True)
    out_dir = os.path.join(C.TABLES_DIR, task.name + ("_smoke" if args.smoke else ""))
    os.makedirs(out_dir, exist_ok=True)

    pairs, kinds = all_pairs(task)
    if args.smoke:
        sel = np.r_[0:12, max(0, pairs.shape[1] - 6):pairs.shape[1]]
        pairs, kinds = pairs[:, sel], [kinds[i] for i in sel]
    t0 = time.time()
    subs = make_subgraphs(task, pairs)
    tr_pairs = task.train_pairs[:, :24] if args.smoke else task.train_pairs
    tr_subs = make_subgraphs(task, tr_pairs)
    print(f"[{task.name}] {pairs.shape[1]} query pairs, {tr_pairs.shape[1]} PG training pairs, "
          f"subgraphs built in {time.time()-t0:.0f}s on {dev}", flush=True)

    for mname in args.models:
        model = restore_model(mname, task.data, torch.load(task.ckpt(mname), map_location="cpu",
                                                           weights_only=False), dev)
        t0 = time.time()
        tr_q = build_queries(task, model, tr_pairs, tr_subs, dev)
        pg = train_pgexplainer([QueryBatch(c) for c in chunks(tr_q, args.pg_batch)],
                               epochs=2 if args.smoke else None)
        print(f"  [{mname}] PGExplainer trained in {time.time()-t0:.0f}s", flush=True)
        qs = build_queries(task, model, pairs, subs, dev)
        rows, prow, wsave = [], [], {}
        rng = np.random.default_rng(2024)
        for bi, idx in enumerate(chunks(list(range(len(qs))), args.batch)):
            qb = QueryBatch([qs[i] for i in idx])
            W, T = {}, {}
            for meth in args.methods:
                ts = time.time()
                if meth == "regex":
                    W[meth] = regex_explain(qb, seed=0)
                elif meth == "gnnexplainer":
                    W[meth] = gnnexplainer_explain(qb, seed=0)
                elif meth == "pgexplainer":
                    W[meth] = pgexplainer_explain(pg, qb)
                elif meth == "pagelink":
                    W[meth] = pagelink_explain(qb)
                elif meth == "saliency":
                    W[meth] = saliency_explain(qb)
                elif meth == "ig":
                    W[meth] = ig_explain(qb)
                elif meth == "distance":
                    W[meth] = distance_explain(qb, rng)
                elif meth == "random":
                    W[meth] = random_explain(qb, rng)
                T[meth] = (time.time() - ts) / len(idx)
            locs = [subs[i][2] for i in idx]
            for meth in args.methods:
                res = evaluate_batch(qb, W[meth], C.FRACS, task.evidence, locs)
                for j, i in enumerate(idx):
                    for r in res[j]:
                        rows.append({"model": mname, "pair_id": i, "cohort": kinds[i],
                                     "a_idx": int(pairs[0, i]), "b_idx": int(pairs[1, i]),
                                     "method": meth, "sec_per_query": T[meth], **r})
                    q = qs[i]
                    keep, _ = topk_keep(q, W[meth][j], 0.10, np.random.default_rng(j))
                    pr = explanation_paths(q, keep, W[meth][j], subs[i][2], task.curated)
                    prow.append({"model": mname, "pair_id": i, "cohort": kinds[i], "method": meth,
                                 "connected": pr["connected"], "conn_len": pr["conn_len"],
                                 "n_paths": pr["n_paths"], "curated_hit": pr["curated_hit"],
                                 "n_curated_in_sub": pr["n_curated_in_sub"],
                                 "rel_hist": json.dumps(pr["rel_hist"])})
                    wsave[f"{i}|{meth}"] = np.concatenate(
                        [W[meth][j][et] for et in q.etypes]).astype(np.float32)
            print(f"  [{mname}] {min((bi+1)*args.batch, len(qs))}/{len(qs)} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        for stem, new_rows in [("main", rows), ("paths", prow)]:
            path = os.path.join(out_dir, f"{stem}_{mname}.csv")
            frame = pd.DataFrame(new_rows)
            if os.path.exists(path) and set(args.methods) != set(METHODS):
                old = pd.read_csv(path)
                frame = pd.concat([old[~old.method.isin(args.methods)], frame], ignore_index=True)
            frame.to_csv(path, index=False)
        path = os.path.join(out_dir, f"weights_{mname}.npz")
        if os.path.exists(path) and set(args.methods) != set(METHODS):
            with np.load(path) as old:
                merged = {k: old[k].copy() for k in old.files
                          if k.split("|", 1)[1] not in args.methods}
            merged.update(wsave)
            wsave = merged
        np.savez_compressed(path, **wsave)
        print(f"  [{mname}] done in {time.time()-t0:.0f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
