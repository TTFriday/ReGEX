# -*- coding: utf-8 -*-
"""
compare_relaxation.py — validation-only comparison of the two soft-mask
relaxations of mean aggregation / attention (masked_forward.MASK_MODE):
  ratio : sum m x / sum m              (scale-invariant)
  clamp : sum m x / max(sum m, 1)      (default; attention x min(sum m, 1))
Both are exact for binary masks. Reported: validation characterization score
(mean over budgets) of every explainer that uses the soft mask.
"""
import os, sys, argparse, json
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import masked_forward as MF
from tasks import load_task, apply_selected
from explainers import (restore_model, regex_explain, gnnexplainer_explain, saliency_explain,
                        ig_explain)
from batch import QueryBatch, chunks
from evaluate import evaluate_batch
from run_main import make_subgraphs, build_queries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="atmad")
    ap.add_argument("--n_val", type=int, default=60)
    ap.add_argument("--batch", type=int, default=30)
    ap.add_argument("--sel_dir", default=None,
                    help="directory holding tuning_selected.json (default: results/tables/<task>)")
    args = ap.parse_args()
    dev = torch.device(C.DEVICE)
    task = load_task(args.task)
    if args.sel_dir:        # hyper-parameters tuned under the ratio relaxation (conservative)
        sel = json.load(open(os.path.join(args.sel_dir, "tuning_selected.json"))); r = sel["regex"]
        C.LAMBDA_SIZE, C.LAMBDA_NEC, C.LAMBDA_GEOM, C.LAMBDA_REL = r["lam_size"], r["lam_nec"], r["lam_geom"], r["lam_rel"]
        C.GNNEXP_LAMBDA_SIZE = sel["gnnexplainer"]["lam_size"]
    else:
        apply_selected(task.name)
    pairs = task.val_pairs[:, :args.n_val]
    subs = make_subgraphs(task, pairs)
    rows = []
    for mode in ["ratio", "clamp"]:
        MF.MASK_MODE = mode
        for m in ["sage", "rgcn", "hgt"]:
            model = restore_model(m, task.data, torch.load(task.ckpt(m), map_location="cpu",
                                                           weights_only=False), dev)
            for qb in [QueryBatch(c) for c in chunks(build_queries(task, model, pairs, subs, dev), args.batch)]:
                for meth, fn in [("regex", lambda: regex_explain(qb, seed=0)),
                                 ("gnnexplainer", lambda: gnnexplainer_explain(qb, seed=0)),
                                 ("saliency", lambda: saliency_explain(qb)), ("ig", lambda: ig_explain(qb))]:
                    for per in evaluate_batch(qb, fn(), C.FRACS):
                        rows.append({"mode": mode, "model": m, "method": meth,
                                     "charact": np.mean([r["charact"] for r in per])})
        print(f"  {mode} done", flush=True)
    d = pd.DataFrame(rows)
    os.makedirs(os.path.join(C.TABLES_DIR, task.name), exist_ok=True)
    out = os.path.join(C.TABLES_DIR, task.name, "relaxation_validation.csv")
    d.to_csv(out, index=False)
    print(d.pivot_table(index=["method", "model"], columns="mode", values="charact").round(4))
    print(d.pivot_table(index="method", columns="mode", values="charact").round(4))


if __name__ == "__main__":
    main()
