# -*- coding: utf-8 -*-
"""
make_figures.py — result figures (Figs. 2-6 and supplementary) from the raw
result CSVs. Fig. 1 (schematic) is produced by make_fig1.py.
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.titlesize": 10,
                     "axes.labelsize": 9.5, "legend.fontsize": 8.5, "xtick.labelsize": 8.5,
                     "ytick.labelsize": 8.5, "axes.spines.top": False, "axes.spines.right": False})
ORDER = ["regex", "gnnexplainer", "pgexplainer", "pagelink", "ig", "saliency", "distance", "random"]
LAB = {"regex": "ReGEX", "gnnexplainer": "GNNExpl.", "pgexplainer": "PGExpl.", "pagelink": "PaGE-Link",
       "ig": "IG", "saliency": "Saliency", "distance": "Distance", "random": "Random"}
COL = {"regex": "#C0392B", "gnnexplainer": "#E67E22", "pgexplainer": "#F1C40F", "pagelink": "#8E44AD",
       "ig": "#2980B9", "saliency": "#16A085", "distance": "#7F8C8D", "random": "#BDC3C7"}
TASKS = {"atmad": "AtMAD (gene–phenotype)", "dtinet": "DTINet (drug–target)"}
ENC = ["sage", "rgcn", "hgt"]


def load(task, name):
    if name == "main":
        fs = [os.path.join(C.TABLES_DIR, task, f"main_{m}.csv") for m in ENC]
        fs = [f for f in fs if os.path.exists(f)]
        return pd.concat([pd.read_csv(f) for f in fs]) if fs else None
    p = os.path.join(C.TABLES_DIR, task, f"{name}.csv")
    frame = pd.read_csv(p) if os.path.exists(p) else None
    extra = os.path.join(C.TABLES_DIR, task, "sensitivity_selected.csv")
    if name == "sensitivity" and frame is not None and os.path.exists(extra):
        frame = pd.concat([frame, pd.read_csv(extra)], ignore_index=True)
    return frame


def save(fig, name):
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(C.FIGURES_DIR, f"{name}.{ext}"), dpi=400, bbox_inches="tight")
    plt.close(fig); print("wrote", name)


def ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def fig_budget():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=False)
    for ax, (t, tn) in zip(axes, TASKS.items()):
        d = load(t, "main")
        if d is None:
            ax.axis("off"); continue
        d = d[d.cohort == "test"]
        d = d.groupby(["method", "frac", "pair_id"], as_index=False).charact.mean()
        for m in ORDER:
            s = d[d.method == m].groupby("frac").charact
            mu, e = s.mean(), s.apply(ci)
            ax.errorbar(mu.index * 100, mu.values, yerr=e.values, color=COL[m], marker="o", ms=3,
                        lw=1.8 if m == "regex" else 1.0, capsize=1.5, label=LAB[m], zorder=3 if m == "regex" else 2)
        ax.set_xscale("log"); ax.set_xticks([5, 10, 20, 50]); ax.set_xticklabels(["5", "10", "20", "50"])
        ax.set_xlabel("budget (% of subgraph edges kept)"); ax.set_ylabel("characterization score")
        ax.set_title(tn, loc="left")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout()
    save(fig, "Fig2_budget")


def fig_plane():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for ax, (t, tn) in zip(axes, TASKS.items()):
        d = load(t, "main")
        if d is None:
            ax.axis("off"); continue
        d = d[(d.cohort == "test") & (d.frac == 0.10)]
        for m in ORDER:
            s = d[d.method == m]
            for enc, mk in zip(ENC, ["o", "s", "^"]):
                e = s[s.model == enc]
                ax.scatter(e.fid_minus.mean(), e.fid_plus.mean(), color=COL[m], marker=mk, s=26,
                           edgecolor="k", lw=0.3, zorder=3)
            ax.scatter(s.fid_minus.mean(), s.fid_plus.mean(), color=COL[m], marker="*", s=90,
                       edgecolor="k", lw=0.4, zorder=4, label=LAB[m])
        ax.axvline(0, color="#999", lw=0.6, ls=":"); ax.axhline(0, color="#999", lw=0.6, ls=":")
        ax.set_xlabel(r"Fid$^-$ (sufficiency loss, lower is better)")
        ax.set_ylabel(r"Fid$^+$ (necessity, higher is better)"); ax.set_title(tn + ", 10% budget", loc="left")
    h = [plt.Line2D([], [], marker=mk, ls="", color="#555", label=l) for mk, l in
         zip(["o", "s", "^", "*"], ["GraphSAGE", "RGCN", "HGT", "pooled"])]
    hm, lm = axes[0].get_legend_handles_labels()
    fig.legend(hm + h, lm + [x.get_label() for x in h], frameon=False, ncol=6, loc="upper center",
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout()
    save(fig, "Fig3_plane")


def fig_ablation_sensitivity():
    fig = plt.figure(figsize=(7.0, 4.6))
    gs = fig.add_gridspec(2, 4, hspace=0.55, wspace=0.45)
    VAR = ["full", "no_necessity", "no_geometry", "no_relation", "no_priors", "no_gumbel"]
    VL = ["full", "−nec.", "−geom.", "−rel.", "−priors", "−Gumbel"]
    for j, (t, tn) in enumerate(TASKS.items()):
        ax = fig.add_subplot(gs[0, 2 * j:2 * j + 2])
        a = load(t, "ablation")
        if a is None:
            ax.axis("off"); continue
        a = a[a.frac == 0.10]
        mu = [a[a.variant == v].charact.mean() for v in VAR]
        e = [ci(a[a.variant == v].charact) for v in VAR]
        ax.bar(range(len(VAR)), mu, yerr=e, color=["#C0392B"] + ["#E59866"] * 5, capsize=2, width=0.65)
        ax.set_xticks(range(len(VAR))); ax.set_xticklabels(VL, rotation=0)
        ax.set_ylabel("char. score @10%"); ax.set_title(f"({'ab'[j]}) ablation, {tn}", loc="left")
        lo = min(mu) - 0.05; ax.set_ylim(max(0, lo), max(mu) + 0.05)
    PAR = [("lam_nec", r"$\lambda_n$"), ("lam_geom", r"$\lambda_g$"), ("lam_rel", r"$\lambda_r$"), ("lam_size", r"$\lambda_s$")]
    for k, (p, pl) in enumerate(PAR):
        ax = fig.add_subplot(gs[1, k])
        for t, ls in zip(TASKS, ["-", "--"]):
            s = load(t, "sensitivity")
            if s is None:
                continue
            s = s[s.param == p].groupby("value").charact_mean.mean()
            x = np.array(s.index, float); xs = np.where(x == 0, x[x > 0].min() / 3 if (x > 0).any() else 1e-3, x)
            ax.plot(xs, s.values, ls, marker="o", ms=3, color="#C0392B", label={"atmad": "AtMAD", "dtinet": "DTINet"}[t])
            sel = json.load(open(os.path.join(C.TABLES_DIR, t, "tuning_selected.json")))["regex"][p]
            ax.axvline(sel if sel > 0 else xs.min(), color="#555", lw=0.6, ls=ls)
        ax.set_xscale("log"); ax.set_xlabel(pl + " (0 at left end)")
        if k == 0:
            ax.set_ylabel("char. score (mean of budgets)")
            ax.legend(frameon=False)
        ax.set_title(f"({'cdef'[k]}) {pl}", loc="left")
    save(fig, "Fig4_ablation_sensitivity")


def fig_reliability():
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.3))
    MM = ["regex", "gnnexplainer", "ig", "saliency"]
    for i, (t, tn) in enumerate(TASKS.items()):
        st, ro, sa = load(t, "stability"), load(t, "robustness"), load(t, "sanity")
        ax = axes[i, 0]
        if st is not None:
            v = [st[st.method == m].jaccard.values for m in ["regex", "gnnexplainer"]]
            bp = ax.boxplot(v, widths=0.5, patch_artist=True, showfliers=False)
            for b, m in zip(bp["boxes"], ["regex", "gnnexplainer"]):
                b.set_facecolor(COL[m]); b.set_alpha(0.75)
            ax.axhline(0.1 / 1.9, color="k", ls=":", lw=0.8); ax.text(2.45, 0.1 / 1.9, "chance", fontsize=6, va="bottom", ha="right")
            ax.set_xticks([1, 2]); ax.set_xticklabels(["ReGEX", "GNNExpl."]); ax.set_ylim(0, 1)
        ax.set_ylabel("Jaccard (top-10%), 10 seeds"); ax.set_title(f"({'ad'[i]}) seed stability, {t.replace('atmad','AtMAD').replace('dtinet','DTINet')}", loc="left")
        ax = axes[i, 1]
        if ro is not None:
            for m in MM:
                s = ro[ro.method == m].groupby("sigma").jaccard.mean()
                ax.plot(s.index, s.values, marker="o", ms=3, color=COL[m], label=LAB[m])
            ax.set_xscale("log"); ax.set_ylim(0, 1.02)
        ax.set_xlabel(r"feature noise $\sigma$ (× column s.d.)"); ax.set_ylabel("Jaccard vs. clean")
        ax.set_title(f"({'be'[i]}) robustness", loc="left")
        if i == 0:
            ax.legend(frameon=False, fontsize=6)
        ax = axes[i, 2]
        if sa is not None:
            lv = ["top", "gnn", "all"]
            for k, m in enumerate(MM):
                y = [sa[(sa.method == m) & (sa.level == l)].spearman.mean() for l in lv]
                ax.bar(np.arange(3) + (k - 1.5) * 0.2, y, width=0.2, color=COL[m], label=LAB[m])
            ax.set_xticks(range(3)); ax.set_xticklabels(["output layer", "GNN layers", "all params"])
            ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel("Spearman (trained vs. random)"); ax.set_title(f"({'cf'[i]}) model randomization", loc="left")
    fig.tight_layout()
    save(fig, "Fig5_reliability")


def fig_relation():
    """relation composition of top-10% explanations vs. subgraph composition"""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from tasks import load_task, all_pairs
    from explainers import build_subgraph
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw={"width_ratios": [1.3, 1]})
    for ax, (t, tn) in zip(axes, TASKS.items()):
        wf = os.path.join(C.TABLES_DIR, t, "weights_hgt.npz")
        if not os.path.exists(wf):
            ax.axis("off"); continue
        task = load_task(t); pairs, kinds = all_pairs(task)
        W = np.load(wf)
        comp = {m: {} for m in ["base"] + ORDER}
        test_ids = [i for i, k in enumerate(kinds) if k == "test"]
        for i in test_ids:
            sub, keep, local, _ = build_subgraph(task.data, int(pairs[0, i]), int(pairs[1, i]), task.pair_types)
            ets = [et for et in sub.edge_types if sub[et].edge_index.shape[1] > 0]
            sizes = [sub[et].edge_index.shape[1] for et in ets]
            rel = np.concatenate([[et[1].replace("_rev", "")] * n for et, n in zip(ets, sizes)])
            for r in set(rel):
                comp["base"][r] = comp["base"].get(r, 0) + (rel == r).sum()
            for m in ORDER:
                w = W[f"{i}|{m}"]
                K = max(1, int(round(0.1 * len(w))))
                top = np.argsort(-(w + 1e-9 * np.random.default_rng(i).random(len(w))))[:K]
                for r in rel[top]:
                    comp[m][r] = comp[m].get(r, 0) + 1
        rels = sorted(comp["base"], key=lambda r: -comp["base"][r])
        cmap = plt.get_cmap("tab10")
        rows = ["base"] + ORDER
        for k, row in enumerate(rows):
            tot = sum(comp[row].values()); left = 0
            for j, r in enumerate(rels):
                v = comp[row].get(r, 0) / tot
                ax.barh(k, v, left=left, color=cmap(j % 10), edgecolor="white", lw=0.3,
                        label=r if k == 0 else None)
                left += v
        ax.set_yticks(range(len(rows))); ax.set_yticklabels(["subgraph"] + [LAB[m] for m in ORDER])
        ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel("fraction of top-10% edges (HGT)")
        ax.set_title(tn, loc="left")
        ax.legend(frameon=False, fontsize=5.8, ncol=3, bbox_to_anchor=(0.5, -0.22), loc="upper center")
    fig.tight_layout()
    save(fig, "Fig6_relation")


def fig_convergence():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
    for ax, (t, tn) in zip(axes, TASKS.items()):
        c = load(t, "convergence")
        if c is None:
            ax.axis("off"); continue
        g = c.groupby("step")
        for col, lab, cl in [("l_fid", "sufficiency BCE", "#C0392B"), ("l_nec", "necessity BCE", "#2980B9"),
                             ("frac_above_half", "fraction of $m_e>0.5$", "#7F8C8D")]:
            ax.plot(g[col].mean().index, g[col].mean().values, color=cl, label=lab)
        ax.axvline(C.MASK_STEPS, color="k", ls=":", lw=0.7)
        ax.set_xlabel("optimization step"); ax.set_title(tn, loc="left")
    axes[0].legend(frameon=False)
    save(fig, "FigS1_convergence")


if __name__ == "__main__":
    which = sys.argv[1:] or ["budget", "plane", "ablation_sensitivity", "reliability", "relation", "convergence"]
    for w in which:
        globals()["fig_" + w]()
