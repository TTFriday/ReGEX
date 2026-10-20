# -*- coding: utf-8 -*-
"""make_fig1.py — schematic overview of ReGEX (Fig. 1)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
import config as C

NODE = {"gene": "#C0392B", "phenotype": "#E59866", "snp": "#5DADE2", "cpg": "#48C9B0",
        "pathway": "#AF7AC5"}
REL = {"eqtl": "#2E86C1", "gwas": "#8E44AD", "emqtl": "#17A589", "ewas": "#28B463",
       "twas": "#D35400", "epathway": "#7D3C98", "pheno_path": "#7F8C8D"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})


def node(ax, xy, nt, label, r=0.38):
    ax.add_patch(Circle(xy, r, color=NODE[nt], zorder=3, lw=0))
    ax.text(*xy, label, ha="center", va="center", color="white", fontsize=6 if len(label) > 2 else 7,
            fontweight="bold", zorder=4)


def edge(ax, a, b, rel, w=1.6, alpha=1.0, ls="-", label=True):
    ax.plot([a[0], b[0]], [a[1], b[1]], color=REL.get(rel, "#555"), lw=w, alpha=alpha, ls=ls, zorder=2)
    if label:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, rel, fontsize=5.5, color=REL.get(rel, "#555"),
                ha="center", va="center", alpha=max(alpha, 0.6),
                bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.8), zorder=3)


def box(ax, x, y, w, h, text, fc="#F4F6F7", fs=7.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.03",
                                fc=fc, ec="#AAB7B8", lw=0.8, zorder=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=2)


def arrow(fig, x0, x1, y=0.52):
    fig.patches.append(matplotlib.patches.FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure,
                                                          arrowstyle="-|>", mutation_scale=12,
                                                          color="#34495E", lw=1.2))


G = {"g": (0.6, 2.6), "p": (4.7, 2.6), "s1": (2.0, 4.25), "s2": (1.7, 1.15), "c1": (2.65, 3.25),
     "c2": (3.95, 4.25), "w": (4.75, 0.75), "g2": (3.05, 0.75)}
E = [("g", "s1", "eqtl"), ("s1", "p", "gwas"), ("g", "c1", "emqtl"), ("c1", "p", "ewas"),
     ("g", "s2", "eqtl"), ("c2", "p", "ewas"), ("w", "p", "pheno_path"),
     ("s2", "g2", "eqtl"), ("g2", "p", "twas")]
TYPE = {"g": "gene", "p": "phenotype", "s1": "snp", "s2": "snp", "c1": "cpg", "c2": "cpg",
        "w": "pathway", "g2": "gene"}
LAB = {"g": "g", "p": "p", "s1": "SNP", "s2": "SNP", "c1": "CpG", "c2": "CpG", "w": "PW", "g2": "g'"}
MASK = {("g", "s1"): 0.92, ("s1", "p"): 0.88, ("g", "c1"): 0.81, ("c1", "p"): 0.77,
        ("g", "s2"): 0.35, ("c2", "p"): 0.30, ("w", "p"): 0.12,
        ("s2", "g2"): 0.05, ("g2", "p"): 0.40}
SHOW = {("g", "s1"), ("s1", "p"), ("g", "c1"), ("c1", "p"), ("c2", "p"), ("g2", "p")}


def graph(ax, weighted=False, keep=None):
    for a, b, r in E:
        if keep is not None and (a, b) not in keep:
            edge(ax, G[a], G[b], r, w=0.8, alpha=0.18, label=False)
            continue
        m = MASK[(a, b)]
        edge(ax, G[a], G[b], r, w=0.6 + 3.2 * m if weighted else 1.4,
             alpha=0.25 + 0.75 * m if weighted else 1.0,
             label=((a, b) in SHOW) and (not weighted or m > 0.5) and keep is None)
    for k, xy in G.items():
        node(ax, xy, TYPE[k], LAB[k])
    ax.set_xlim(0, 5.3); ax.set_ylim(-0.6, 4.8); ax.set_aspect("equal"); ax.axis("off")


def main():
    fig = plt.figure(figsize=(5.0, 4.7))
    global TITLES
    TITLES = []
    positions = [(0.015, 0.53, 0.46, 0.41), (0.52, 0.53, 0.46, 0.41),
                 (0.015, 0.03, 0.46, 0.41), (0.52, 0.03, 0.46, 0.41)]
    axs = [fig.add_axes(position) for position in positions]
    # (a) query subgraph
    graph(axs[0])
    axs[0].plot([G["g"][0] + 0.36, G["p"][0] - 0.36], [G["g"][1], G["p"][1]], ls=(0, (2.5, 2)),
                color="#D35400", lw=1.1, zorder=1)
    axs[0].text(2.65, 2.25, "query edge (g, p) removed", fontsize=5.8, color="#D35400",
                ha="center", va="top", bbox=dict(fc="white", ec="none", pad=0.5))
    TITLES.append("(a) Query subgraph")
    # (b) masked message passing
    graph(axs[1], weighted=True)
    axs[1].text(2.65, -0.2, "weights $m_e\\in(0,1)$ enter every\naggregation / attention step",
                fontsize=6, ha="center", va="bottom")
    TITLES.append("(b) Masked GNN")
    # (c) objective
    ax = axs[2]; ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    items = [(r"sufficiency  $\mathrm{BCE}(s(\mathbf{m}),\hat y)$", "#FDEBD0"),
             (r"necessity  $\lambda_n\,\mathrm{BCE}(s(\mathbf{1}-\mathbf{m}),1-\hat y)$", "#FDEBD0"),
             (r"sparsity  $\lambda_s\,\bar m+\lambda_e\,\bar H(\mathbf{m})$", "#EAF2F8"),
             (r"geometry  $\lambda_g\,\overline{m_e d_e}/d_{\max}$", "#E8F8F5"),
             (r"relation  $\lambda_r\,\mathrm{KL}(\hat{\mathbf{q}}(\mathbf{m})\,\|\,\mathbf{q})$", "#F5EEF8")]
    for i, (t, fc) in enumerate(items):
        box(ax, 0.03, 0.82 - 0.152 * i, 0.94, 0.118, t, fc=fc, fs=7.2)
    ax.text(0.5, 0.06, "Gumbel-sigmoid relaxation, Adam;" + chr(10) + "final weights = sigmoid(logits)", fontsize=7,
            ha="center", va="bottom")
    TITLES.append("(c) Objective")
    # (d) evaluation
    keep = {k for k, v in MASK.items() if v > 0.5}
    graph(axs[3], keep=keep)
    axs[3].text(2.65, -0.2, "top-$K$ edges at matched budgets:" + chr(10) +
                "Fid$^+$, Fid$^-$, characterization, paths", fontsize=6, ha="center", va="bottom")
    TITLES.append("(d) Evaluation")
    for (x, y, width, height), title in zip(positions, TITLES):
        fig.text(x + 0.005, y + height + 0.015, title, fontsize=9,
                 fontweight="bold", ha="left", va="bottom")
    os.makedirs(C.FIGURES_DIR, exist_ok=True)
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(C.FIGURES_DIR, f"Fig1_overview.{ext}"), dpi=400, bbox_inches="tight")
    print("wrote Fig1_overview")


if __name__ == "__main__":
    main()
