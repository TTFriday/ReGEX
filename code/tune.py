# -*- coding: utf-8 -*-
"""
tune.py — hyper-parameter selection on *validation* pairs only.

Every tunable explainer gets the same budget of validation queries and the
same selection criterion: the characterisation score averaged over the four
sparsity budgets and the three encoders. Test pairs are never used here.

  GNNExplainer : lambda_size
  PGExplainer  : lambda_size
  ReGEX        : stage 1  lambda_size x lambda_nec      (no priors)
                 stage 2  lambda_geom x lambda_rel      (at the stage-1 optimum)

Writes results/tables/<task>/tuning.csv and tuning_selected.json.
"""
import os, sys, json, time, argparse, itertools
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from tasks import load_task
from explainers import (restore_model, optimise_masks, train_pgexplainer, pgexplainer_explain,
                        pagelink_explain)
from batch import QueryBatch, chunks
from evaluate import evaluate_batch
from run_main import make_subgraphs, build_queries

# one grid for both benchmarks and all tunable explainers
SIZES = [0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0]
NECS = [0.0, 0.5, 1.0, 2.0]
GEOMS = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0]
RELS = [0.0, 0.05, 0.1, 0.5, 1.0, 2.0]


def score(batches, weights_per_batch):
    vals = []
    for qb, W in zip(batches, weights_per_batch):
        for per_q in evaluate_batch(qb, W, C.FRACS):
            vals.append(np.mean([r["charact"] for r in per_q]))
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="atmad")
    ap.add_argument("--models", nargs="*", default=["sage", "rgcn", "hgt"])
    ap.add_argument("--batch", type=int, default=30)
    ap.add_argument("--methods", nargs="*", default=["gnnexplainer", "pgexplainer", "regex", "pagelink"])
    ap.add_argument("--grid", choices=["full", "coarse"], default="full")
    ap.add_argument("--n_val", type=int, default=60)
    ap.add_argument("--extend_s2", nargs=2, metavar=("GEOMS", "RELS"), default=None,
                    help="comma-separated extra stage-2 values; re-uses the stored stage-1 optimum")
    args = ap.parse_args()
    global SIZES, NECS, GEOMS, RELS
    if args.grid == "coarse":     # larger subgraphs (DTINet): 3 values per axis
        SIZES, NECS, GEOMS, RELS = [0.01, 0.1, 1.0], [0.0, 1.0, 2.0], [0.0, 0.5, 2.0], [0.0, 0.5, 2.0]
    dev = torch.device(C.DEVICE)
    task = load_task(args.task)
    task.val_pairs = task.val_pairs[:, :args.n_val]
    out_dir = os.path.join(C.TABLES_DIR, task.name); os.makedirs(out_dir, exist_ok=True)
    vsubs = make_subgraphs(task, task.val_pairs)
    tsubs = make_subgraphs(task, task.train_pairs)
    models = {m: restore_model(m, task.data, torch.load(task.ckpt(m), map_location="cpu",
                                                        weights_only=False), dev) for m in args.models}
    VB = {m: [QueryBatch(c) for c in chunks(build_queries(task, mod, task.val_pairs, vsubs, dev), args.batch)]
          for m, mod in models.items()}
    TB = {m: [QueryBatch(c) for c in chunks(build_queries(task, mod, task.train_pairs, tsubs, dev), args.batch)]
          for m, mod in models.items()}
    rows, t0 = [], time.time()

    def run(method, cfg, fn):
        sc = np.mean([score(VB[m], [fn(m, qb) for qb in VB[m]]) for m in args.models])
        rows.append({"method": method, **cfg, "val_charact": sc})
        print(f"  {method:12s} {cfg} -> {sc:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        return sc

    sel_path = os.path.join(out_dir, "tuning_selected.json")
    sel = json.load(open(sel_path)) if os.path.exists(sel_path) else {}
    old_rows = pd.read_csv(os.path.join(out_dir, "tuning.csv")) if os.path.exists(
        os.path.join(out_dir, "tuning.csv")) else pd.DataFrame()

    if args.extend_s2 is not None:
        old = pd.read_csv(os.path.join(out_dir, "tuning.csv"))
        b1 = old[old.method == "regex_s1"].sort_values("val_charact").iloc[-1]
        gs = [float(v) for v in args.extend_s2[0].split(",")]
        rs = [float(v) for v in args.extend_s2[1].split(",")]
        for g, r in itertools.product(gs, rs):
            run("regex_s2", {"lam_size": b1.lam_size, "lam_nec": b1.lam_nec, "lam_geom": g, "lam_rel": r},
                lambda m, qb, g=g, r=r: optimise_masks(qb, lam_size=b1.lam_size, lam_nec=b1.lam_nec,
                                                       lam_geom=g, lam_rel=r, seed=0)[0])
        df = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
        df.to_csv(os.path.join(out_dir, "tuning.csv"), index=False)
        b = df[df.method == "regex_s2"].sort_values("val_charact").iloc[-1]
        sel["regex"] = {k: float(b[k]) for k in ["lam_size", "lam_nec", "lam_geom", "lam_rel"]}
        json.dump(sel, open(sel_path, "w"), indent=2); print(json.dumps(sel, indent=2))
        return

    # PaGE-Link
    if "pagelink" in args.methods:
        grid_pl = itertools.product([0.01, 0.1, 1.0], [0.001, 0.01, 0.1]) if args.grid == "full"             else [(0.01, 0.001), (0.1, 0.001), (0.1, 0.01)]
        for a, b in grid_pl:
            run("pagelink", {"alpha": a, "beta": b},
                lambda m, qb, a=a, b=b: pagelink_explain(qb, alpha=a, beta=b))
        d = pd.DataFrame(rows); bp = d[d.method == "pagelink"].sort_values("val_charact").iloc[-1]
        sel["pagelink"] = {"alpha": float(bp.alpha), "beta": float(bp.beta)}
    if not set(args.methods) & {"gnnexplainer", "pgexplainer", "regex"}:
        df = pd.concat([old_rows[old_rows.get("method", pd.Series(dtype=str)) != "pagelink"] if len(old_rows) else old_rows,
                        pd.DataFrame(rows)])
        df.to_csv(os.path.join(out_dir, "tuning.csv"), index=False)
        json.dump(sel, open(sel_path, "w"), indent=2); print(json.dumps(sel, indent=2))
        return

    # GNNExplainer
    for s in SIZES:
        run("gnnexplainer", {"lam_size": s}, lambda m, qb, s=s: optimise_masks(
            qb, lam_size=s, lam_geom=0, lam_rel=0, lam_nec=0, gumbel=False, seed=0)[0])
    # PGExplainer
    for s in SIZES:
        old = C.PG_LAMBDA_SIZE; C.PG_LAMBDA_SIZE = s
        pgs = {m: train_pgexplainer(TB[m]) for m in args.models}
        C.PG_LAMBDA_SIZE = old
        run("pgexplainer", {"lam_size": s}, lambda m, qb: pgexplainer_explain(pgs[m], qb))
    # ReGEX stage 1
    for s, n in itertools.product(SIZES, NECS):
        run("regex_s1", {"lam_size": s, "lam_nec": n}, lambda m, qb, s=s, n=n: optimise_masks(
            qb, lam_size=s, lam_nec=n, lam_geom=0, lam_rel=0, seed=0)[0])
    df = pd.DataFrame(rows)
    b1 = df[df.method == "regex_s1"].sort_values("val_charact").iloc[-1]
    # ReGEX stage 2
    for g, r in itertools.product(GEOMS, RELS):
        run("regex_s2", {"lam_size": b1.lam_size, "lam_nec": b1.lam_nec, "lam_geom": g, "lam_rel": r},
            lambda m, qb, g=g, r=r: optimise_masks(qb, lam_size=b1.lam_size, lam_nec=b1.lam_nec,
                                                   lam_geom=g, lam_rel=r, seed=0)[0])
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "tuning.csv"), index=False)
    best = lambda meth: df[df.method == meth].sort_values("val_charact").iloc[-1]
    sel.update({"gnnexplainer": {"lam_size": float(best("gnnexplainer").lam_size)},
                "pgexplainer": {"lam_size": float(best("pgexplainer").lam_size)},
                "regex": {k: float(best("regex_s2")[k]) for k in ["lam_size", "lam_nec", "lam_geom", "lam_rel"]}})
    json.dump(sel, open(sel_path, "w"), indent=2)
    print(json.dumps(sel, indent=2))


if __name__ == "__main__":
    main()
