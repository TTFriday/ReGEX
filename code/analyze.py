# -*- coding: utf-8 -*-
"""
analyze.py — summary statistics and LaTeX tables from the raw result CSVs.

Every number quoted in the manuscript is produced here (results/tables/
<task>/summary_*.csv and results/tables/latex/*.tex), so the text can be
checked against the data.

Statistics: paired Wilcoxon signed-rank tests over (encoder, query) units,
ReGEX vs each competitor, Holm correction within each table; effect size =
median paired difference and win rate.
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

METHOD_ORDER = ["regex", "gnnexplainer", "pgexplainer", "pagelink", "ig", "saliency", "distance", "random"]
LABEL = {"regex": "ReGEX", "gnnexplainer": "GNNExplainer", "pgexplainer": "PGExplainer", "pagelink": "PaGE-Link",
         "ig": "Integrated Gradients", "saliency": "Saliency", "distance": "Distance",
         "random": "Random"}
ENC = {"sage": "GraphSAGE", "rgcn": "RGCN", "hgt": "HGT"}
LATEX_DIR = os.path.join(C.TABLES_DIR, "latex")


def holm(p):
    p = np.asarray(p, dtype=float); n = len(p); order = np.argsort(p)
    adj = np.empty(n); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (n - r) * p[i])); adj[i] = run
    return adj


def fmt_p(p):
    if np.isnan(p):
        return "--"
    return r"$<$10$^{-4}$" if p < 1e-4 else f"{p:.3f}" if p >= 1e-3 else f"{p:.1e}"


def iqr(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) == 0:
        return "--"
    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    return f"{q2:.3f} [{q1:.3f}, {q3:.3f}]"


def paired(df, metric, a, b, keys):
    # For pooled comparisons keys=['pair_id']: average the three encoder
    # results of a query before testing. They are repeated measurements,
    # not three independent observations. Per-encoder comparisons use the
    # same keys on an already encoder-filtered frame.
    pa = df[df.method == a].groupby(keys)[metric].mean()
    pb = df[df.method == b].groupby(keys)[metric].mean()
    j = pa.to_frame("a").join(pb.to_frame("b"), how="inner").dropna()
    if len(j) < 5:
        return np.nan, np.nan, np.nan, len(j)
    delta = j.a - j.b
    p = 1.0 if np.allclose(delta, 0) else wilcoxon(delta).pvalue
    return p, float(np.median(delta)), float((delta > 0).mean()), len(j)


def load_main(task):
    d = os.path.join(C.TABLES_DIR, task)
    df = pd.concat([pd.read_csv(os.path.join(d, f"main_{m}.csv")) for m in ENC
                    if os.path.exists(os.path.join(d, f"main_{m}.csv"))])
    pt = pd.concat([pd.read_csv(os.path.join(d, f"paths_{m}.csv")) for m in ENC
                    if os.path.exists(os.path.join(d, f"paths_{m}.csv"))])
    return df, pt


def write_tex(name, body):
    os.makedirs(LATEX_DIR, exist_ok=True)
    open(os.path.join(LATEX_DIR, name), "w", encoding="utf-8").write(body)


def main_table(task, df, cohort="test", frac=0.10):
    d = df[(df.cohort == cohort) & (df.frac == frac)]
    keys = ["pair_id"]
    rows, ps = [], []
    for m in METHOD_ORDER:
        s = d[d.method == m]
        if s.empty:
            continue
        p, dm, win, n = paired(d, "charact", "regex", m, keys) if m != "regex" else (np.nan,) * 4
        per_query = s.groupby("pair_id").charact.mean().to_numpy()
        rng = np.random.default_rng(42)
        means = per_query[rng.integers(len(per_query), size=(2000, len(per_query)))].mean(1)
        lo, hi = np.percentile(means, [2.5, 97.5])
        rows.append({"method": m, "fid_plus": iqr(s.fid_plus), "fid_minus": iqr(s.fid_minus),
                     "charact_mean": s.charact.mean(), "charact_sd": s.charact.std(),
                     "fid_plus_mean": s.fid_plus.mean(), "fid_minus_mean": s.fid_minus.mean(),
                     "charact_ci_low": lo, "charact_ci_high": hi,
                     "charact_of_means": 2 * np.clip(s.fid_plus.mean(), 0, 1) * np.clip(1 - s.fid_minus.mean(), 0, 1)
                     / (np.clip(s.fid_plus.mean(), 0, 1) + np.clip(1 - s.fid_minus.mean(), 0, 1) + 1e-12),
                     "egc": s.egc.median(), "egc_raw": s.egc_raw.median(),
                     "sec": s.sec_per_query.mean(), "p_raw": p, "delta": dm, "win": win, "n": n})
    t = pd.DataFrame(rows)
    mask = ~t.p_raw.isna()
    t.loc[mask, "p_holm"] = holm(t.loc[mask, "p_raw"])
    return t


def budget_table(df, cohort="test"):
    d = df[df.cohort == cohort]
    return d.groupby(["method", "frac"]).agg(charact=("charact", "mean"), fid_plus=("fid_plus", "mean"),
                                             fid_minus=("fid_minus", "mean")).reset_index()


def encoder_table(df, cohort="test", frac=0.10):
    d = df[(df.cohort == cohort) & (df.frac == frac)]
    out = []
    for enc in ENC:
        e = d[d.model == enc]
        if e.empty:
            continue
        for m in METHOD_ORDER:
            s = e[e.method == m]
            p = paired(e, "charact", "regex", m, ["pair_id"])[0] if m != "regex" else np.nan
            out.append({"model": enc, "method": m, "charact": s.charact.mean(),
                        "fid_plus": s.fid_plus.mean(), "fid_minus": s.fid_minus.mean(), "p_raw": p})
    t = pd.DataFrame(out)
    for enc in ENC:
        mk = (t.model == enc) & ~t.p_raw.isna()
        if mk.any():
            t.loc[mk, "p_holm"] = holm(t.loc[mk, "p_raw"])
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="atmad")
    args = ap.parse_args()
    out = os.path.join(C.TABLES_DIR, args.task)
    df, pt = load_main(args.task)
    T = main_table(args.task, df); T.to_csv(os.path.join(out, "summary_main.csv"), index=False)
    print(T.round(4).to_string())
    budget_table(df).to_csv(os.path.join(out, "summary_budget.csv"), index=False)
    E = encoder_table(df); E.to_csv(os.path.join(out, "summary_encoder.csv"), index=False)
    print(E.round(4).to_string())
    P = pt.groupby(["cohort", "method"]).agg(connected=("connected", "mean"),
                                              curated_hit=("curated_hit", "mean"),
                                              n=("connected", "size")).reset_index()
    P.to_csv(os.path.join(out, "summary_paths.csv"), index=False); print(P.round(3).to_string())
    for name in ["ablation", "sensitivity", "stability", "robustness", "sanity", "convergence"]:
        f = os.path.join(out, f"{name}.csv")
        if not os.path.exists(f):
            continue
        a = pd.read_csv(f)
        extra = os.path.join(out, "sensitivity_selected.csv")
        if name == "sensitivity" and os.path.exists(extra):
            a = pd.concat([a, pd.read_csv(extra)], ignore_index=True)
        if name == "ablation":
            s = a[a.frac == 0.10].groupby("variant").agg(charact=("charact", "mean"),
                                                         fid_plus=("fid_plus", "mean"),
                                                         fid_minus=("fid_minus", "mean")).reset_index()
            ref = a[(a.frac == 0.10) & (a.variant == "full")].groupby("pair_id").charact.mean()
            ps = []
            for v in s.variant:
                x = a[(a.frac == 0.10) & (a.variant == v)].groupby("pair_id").charact.mean()
                j = ref.to_frame("f").join(x.to_frame("v"), how="inner").dropna()
                ps.append(np.nan if v == "full" else 1.0 if np.allclose(j.f, j.v) else wilcoxon(j.f, j.v).pvalue)
            s["p_raw"] = ps
            mk = ~s.p_raw.isna(); s.loc[mk, "p_holm"] = holm(s.loc[mk, "p_raw"])
            avg = a.groupby(["variant", "model", "pair_id"]).charact.mean().groupby("variant").mean()
            s["charact_allbudgets"] = s.variant.map(avg)
        elif name == "sensitivity":
            s = a.groupby(["param", "value"]).agg(charact_10=("charact_10", "mean"),
                                                  charact_mean=("charact_mean", "mean")).reset_index()
        elif name == "stability":
            s = a.groupby("method").agg(jaccard=("jaccard", "mean"), jaccard_sd=("jaccard", "std"),
                                        spearman=("spearman", "mean"), charact_sd=("charact_sd", "mean"),
                                        chance=("chance_jaccard", "mean")).reset_index()
        elif name == "robustness":
            s = a.groupby(["method", "sigma"]).agg(jaccard=("jaccard", "mean"), spearman=("spearman", "mean"),
                                                   pred_shift=("pred_shift", "median")).reset_index()
        elif name == "sanity":
            s = a.groupby(["method", "level"]).agg(spearman=("spearman", "mean"),
                                                   n_valid=("spearman", "count"),
                                                   n_total=("jaccard", "size"),
                                                   jaccard=("jaccard", "mean")).reset_index()
        else:
            s = a.groupby("step")[["loss", "l_fid", "l_nec", "masked_p_y", "frac_above_half"]].mean().reset_index()
        s.to_csv(os.path.join(out, f"summary_{name}.csv"), index=False)
        print(f"\n== {name}\n", s.round(4).to_string() if name != "convergence" else s.iloc[[0, 24, 49, 99, -1]].round(4).to_string())


if __name__ == "__main__":
    main()
