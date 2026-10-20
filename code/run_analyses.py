# -*- coding: utf-8 -*-
"""
run_analyses.py — ablation, sensitivity, stability, robustness, convergence
and model-randomisation sanity check for ReGEX (and baselines where
meaningful), all on test-cohort query pairs.

  python run_analyses.py --task atmad --which ablation sensitivity stability \
                         robustness convergence sanity

Each analysis writes results/tables/<task>/<analysis>.csv.
"""
import os, sys, time, copy, argparse
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from tasks import load_task, apply_selected
from explainers import (restore_model, optimise_masks, regex_explain, gnnexplainer_explain,
                        saliency_explain, ig_explain, Query)
from batch import QueryBatch, chunks
from evaluate import evaluate_batch, jaccard
from run_main import make_subgraphs, build_queries

N_PAIRS = 100


def test_pairs(task, n):
    return task.cohorts["test"][:, :n]


def batches(task, model, pairs, subs, dev, bs):
    return [QueryBatch(c) for c in chunks(build_queries(task, model, pairs, subs, dev), bs)]


def flat_corr(q, w1, w2):
    a = np.concatenate([w1[et] for et in q.etypes]); b = np.concatenate([w2[et] for et in q.etypes])
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(spearmanr(a, b).statistic)


# ---------------------------------------------------------------------------
def ablation(task, models, B, out):
    full = dict(lam_size=C.LAMBDA_SIZE, lam_nec=C.LAMBDA_NEC, lam_geom=C.LAMBDA_GEOM, lam_rel=C.LAMBDA_REL)
    variants = {"full": {}, "no_necessity": {"lam_nec": 0.0}, "no_geometry": {"lam_geom": 0.0},
                "no_relation": {"lam_rel": 0.0}, "no_priors": {"lam_geom": 0.0, "lam_rel": 0.0},
                "no_gumbel": {"gumbel": False}}
    rows = []
    for m, bl in B.items():
        for name, ov in variants.items():
            kw = {**full, **ov}
            pid = 0
            for qb in bl:
                W = optimise_masks(qb, seed=0, **kw)[0]
                locals_ = [q.local for q in qb.qs]
                for i, per in enumerate(evaluate_batch(qb, W, C.FRACS, task.evidence, locals_)):
                    for r in per:
                        rows.append({"model": m, "variant": name, "pair_id": pid + i, **r})
                pid += qb.B
        print(f"  ablation [{m}] done", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "ablation.csv"), index=False)


def sensitivity(task, models, B, out):
    base = dict(lam_size=C.LAMBDA_SIZE, lam_nec=C.LAMBDA_NEC, lam_geom=C.LAMBDA_GEOM,
                lam_rel=C.LAMBDA_REL, tau=C.GUMBEL_TAU, steps=C.MASK_STEPS)
    grid = {"lam_size": [0.05, 0.1, 0.5, 1.0, 2.0], "lam_nec": [0.0, 0.25, 0.5, 1.0, 2.0],
            "lam_geom": [0.0, 0.1, 0.5, 1.0, 2.0], "lam_rel": [0.0, 0.1, 0.5, 1.0, 2.0],
            "tau": [0.1, 0.25, 0.5, 1.0], "steps": [25, 50, 100, 200]}
    rows = []
    for m, bl in B.items():
        for p, vals in grid.items():
            for v in vals:
                kw = {**base, p: v}
                for qb in bl:
                    W = optimise_masks(qb, seed=0, **kw)[0]
                    for per in evaluate_batch(qb, W, C.FRACS):
                        rows.append({"model": m, "param": p, "value": v,
                                     "charact_mean": np.mean([r["charact"] for r in per]),
                                     **{f"{k}_10": per[1][k] for k in ["fid_plus", "fid_minus", "charact"]}})
        print(f"  sensitivity [{m}] done", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "sensitivity.csv"), index=False)


def sensitivity_selected(task, models, B, out):
    """Additional local grid points that include the selected small DTI
    penalties and the larger AtMAD geometry weight; no re-selection occurs."""
    grid = ({"lam_geom": [4.0, 8.0]} if task.name == "atmad" else
            {"lam_size": [0.01, 0.02], "lam_geom": [0.01, 0.05], "lam_rel": [0.05]})
    base = dict(lam_size=C.LAMBDA_SIZE, lam_nec=C.LAMBDA_NEC,
                lam_geom=C.LAMBDA_GEOM, lam_rel=C.LAMBDA_REL,
                tau=C.GUMBEL_TAU, steps=C.MASK_STEPS)
    rows = []
    for model, bl in B.items():
        for parameter, values in grid.items():
            for value in values:
                for qb in bl:
                    weights = optimise_masks(qb, seed=0, **{**base, parameter: value})[0]
                    for result in evaluate_batch(qb, weights, C.FRACS):
                        rows.append({"model": model, "param": parameter, "value": value,
                                     "charact_mean": np.mean([r["charact"] for r in result]),
                                     **{f"{k}_10": result[1][k] for k in ["fid_plus", "fid_minus", "charact"]}})
    pd.DataFrame(rows).to_csv(os.path.join(out, "sensitivity_selected.csv"), index=False)


def stability(task, models, B, out, n_seeds=10):
    rows = []
    for m, bl in B.items():
        for meth, fn in [("regex", regex_explain), ("gnnexplainer", gnnexplainer_explain)]:
            for qb in bl:
                Ws = [fn(qb, seed=1000 + s) for s in range(n_seeds)]
                for i, q in enumerate(qb.qs):
                    J = [jaccard(q, Ws[a][i], Ws[b][i]) for a in range(n_seeds) for b in range(a + 1, n_seeds)]
                    R = [flat_corr(q, Ws[a][i], Ws[b][i]) for a in range(n_seeds) for b in range(a + 1, n_seeds)]
                    rows.append({"model": m, "method": meth, "n_edges": q.n_edges,
                                 "jaccard": np.mean(J), "spearman": np.nanmean(R),
                                 "chance_jaccard": 0.1 / 1.9})
                # fidelity spread across seeds (10% budget)
                ch = np.array([[per[0]["charact"] for per in evaluate_batch(qb, W, [0.10])] for W in Ws])
                for i in range(qb.B):
                    rows[-qb.B + i]["charact_sd"] = float(ch[:, i].std())
                    rows[-qb.B + i]["charact_mean"] = float(ch[:, i].mean())
        print(f"  stability [{m}] done", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "stability.csv"), index=False)


def robustness(task, models, B, out, sigmas=(0.01, 0.05, 0.1, 0.2, 0.5)):
    """Gaussian feature noise scaled by each feature column's s.d.; the
    explanation of the noisy input is compared with the clean one (same seed)."""
    rows = []
    for m, bl in B.items():
        for qb in bl:
            clean = {"regex": regex_explain(qb, seed=0), "gnnexplainer": gnnexplainer_explain(qb, seed=0),
                     "saliency": saliency_explain(qb), "ig": ig_explain(qb)}
            for sg in sigmas:
                qs2 = []
                for k, q in enumerate(qb.qs):
                    sub2 = q.sub.clone()
                    g = torch.Generator(device="cpu").manual_seed(k)
                    for nt in q.data.node_types:
                        x = sub2[nt].x
                        if x.shape[0] > 1:
                            sd = x.std(0, keepdim=True).clamp_min(1e-6)
                            sub2[nt].x = x + sg * sd * torch.randn(x.shape, generator=g).to(x.device)
                    qs2.append(Query(q.model, q.data, sub2, q.a, q.b, q.pair_types))
                qb2 = QueryBatch(qs2)
                noisy = {"regex": regex_explain(qb2, seed=0), "gnnexplainer": gnnexplainer_explain(qb2, seed=0),
                         "saliency": saliency_explain(qb2), "ig": ig_explain(qb2)}
                for meth in clean:
                    for i, q in enumerate(qb.qs):
                        rows.append({"model": m, "method": meth, "sigma": sg,
                                     "jaccard": jaccard(q, clean[meth][i], noisy[meth][i]),
                                     "spearman": flat_corr(q, clean[meth][i], noisy[meth][i]),
                                     "pred_shift": abs(qs2[i].logit_full - q.logit_full)})
        print(f"  robustness [{m}] done", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "robustness.csv"), index=False)


def convergence(task, models, B, out, steps=200):
    rows = []
    for m, bl in B.items():
        for bi, qb in enumerate(bl[:2]):
            _, tr = optimise_masks(qb, seed=0, steps=steps, trace=True)
            for r in tr:
                rows.append({"model": m, "batch": bi, **r})
        print(f"  convergence [{m}] done", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "convergence.csv"), index=False)


def _randomise(model, level, seed=0):
    """Cascading parameter randomisation (Adebayo et al., 2018): re-initialise
    the output layer ('top'), all GNN layers ('gnn') or every parameter ('all')."""
    rm = copy.deepcopy(model)
    g = torch.Generator(device="cpu").manual_seed(seed)
    if level == "all":
        mods = [rm]
    elif model.model_name == "hgt":
        mods = [rm.gnn.lin_out] if level == "top" else [rm.gnn]
    else:
        mods = [rm.gnn.convs[-1]] if level == "top" else [rm.gnn]
    for mod in mods:
        for p in mod.parameters():
            # Population standard deviation is defined for singleton HGT
            # skip parameters; the sample estimate was NaN for those.
            scale = p.detach().float().std(correction=0).cpu() + 1e-3
            p.data.copy_((torch.randn(p.shape, generator=g) * scale).to(p.device))
    assert all(torch.isfinite(p).all() for p in rm.parameters()), "Nonfinite randomized parameter"
    return rm


def sanity(task, models, B, out, subs, pairs, dev, bs):
    rows = []
    for m, bl in B.items():
        clean = [{"regex": regex_explain(qb, seed=0), "gnnexplainer": gnnexplainer_explain(qb, seed=0),
                  "saliency": saliency_explain(qb), "ig": ig_explain(qb)} for qb in bl]
        for level in ["top", "gnn", "all"]:
            rmod = _randomise(models[m], level)
            randomized_queries = build_queries(task, rmod, pairs, subs, dev)
            original_queries = [q for qb in bl for q in qb.qs]
            for old, new in zip(original_queries, randomized_queries):
                new.y_hat = old.y_hat  # hold attribution target class fixed
            bl2 = [QueryBatch(c) for c in chunks(randomized_queries, bs)]
            for qb, qb2, cw in zip(bl, bl2, clean):
                rw = {"regex": regex_explain(qb2, seed=0), "gnnexplainer": gnnexplainer_explain(qb2, seed=0),
                      "saliency": saliency_explain(qb2), "ig": ig_explain(qb2)}
                for meth in cw:
                    for i, q in enumerate(qb.qs):
                        rows.append({"model": m, "level": level, "method": meth,
                                     "spearman": flat_corr(q, cw[meth][i], rw[meth][i]),
                                     "jaccard": jaccard(q, cw[meth][i], rw[meth][i])})
        print(f"  sanity [{m}] done", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "sanity.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="atmad")
    ap.add_argument("--models", nargs="*", default=["sage", "rgcn", "hgt"])
    ap.add_argument("--which", nargs="*", default=["ablation", "sensitivity", "stability",
                                                    "robustness", "convergence", "sanity"])
    ap.add_argument("--n", type=int, default=N_PAIRS)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--out-dir", default=None,
                    help="separate output for validation refresh; never overwrite a concurrent run")
    args = ap.parse_args()
    dev = torch.device(C.DEVICE)
    task = load_task(args.task)
    print("hyper-parameters:", apply_selected(task.name), flush=True)
    out = args.out_dir or os.path.join(C.TABLES_DIR, task.name)
    os.makedirs(out, exist_ok=True)
    pairs = test_pairs(task, args.n)
    subs = make_subgraphs(task, pairs)
    models = {m: restore_model(m, task.data, torch.load(task.ckpt(m), map_location="cpu",
                                                        weights_only=False), dev) for m in args.models}
    B = {m: batches(task, mod, pairs, subs, dev, args.batch) for m, mod in models.items()}
    for w in args.which:
        t0 = time.time()
        if w == "sanity":
            sanity(task, models, B, out, subs, pairs, dev, args.batch)
        else:
            globals()[w](task, models, B, out)
        print(f"[{task.name}] {w} finished in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
