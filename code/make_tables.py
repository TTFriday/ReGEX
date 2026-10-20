# -*- coding: utf-8 -*-
"""
make_tables.py — LaTeX tables for the manuscript, generated from the summary
CSVs written by analyze.py (no number is typed by hand).
Output: paper/tables/*.tex
"""
import os, sys, json, re
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from analyze import METHOD_ORDER, LABEL, ENC, fmt_p

OUT = os.path.join(C.PAPER_DIR, "tables")
TASKS = {"atmad": "AtMAD", "dtinet": "DTINet"}
SHORT = {"regex": "ReGEX", "gnnexplainer": "GNNExpl.", "pgexplainer": "PGExpl.", "pagelink": "PaGE-Link",
         "ig": "IG", "saliency": "Saliency", "distance": "Distance", "random": "Random"}


def rd(t, f):
    p = os.path.join(C.TABLES_DIR, t, f)
    return pd.read_csv(p) if os.path.exists(p) else None


def save(name, body):
    # ws-jcbc's \tbl puts \caption inside a vbox group. A label after
    # \tbl therefore records the enclosing section rather than the table.
    # Keep table labels inside the first caption argument of \tbl.
    label = re.search(r"\\label\{[^}]+\}", body)
    if label and r"\tbl{" in body:
        command = label.group(0)
        body = body[:label.start()] + body[label.end():]
        start = body.index(r"\tbl{") + len(r"\tbl{")
        depth, end = 1, start
        while depth:
            if body[end] == "{" and body[end - 1] != "\\":
                depth += 1
            elif body[end] == "}" and body[end - 1] != "\\":
                depth -= 1
            end += 1
        body = body[:end - 1] + command + body[end - 1:]
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, name), "w", encoding="utf-8").write(body)
    print("wrote", name)


def iqr_cell(s):
    # "0.169 [0.064, 0.244]" -> "0.169 (0.064, 0.244)"
    return s.replace("[", "(").replace("]", ")") if isinstance(s, str) else "--"


def table_main():
    rows = []
    for t, tn in TASKS.items():
        d = rd(t, "summary_main.csv")
        if d is None:
            continue
        best = d.charact_mean.max()
        for i, r in d.iterrows():
            ch = f"{r.charact_mean:.3f}"
            if np.isclose(r.charact_mean, best):
                ch = r"\textbf{" + ch + "}"
            p = "--" if r.method == "regex" else fmt_p(r.p_holm)
            ch += f" ({r.charact_ci_low:.3f}, {r.charact_ci_high:.3f})"
            rows.append((tn if i == 0 else "", SHORT[r.method], f"{r.fid_plus_mean:.3f}",
                         f"{r.fid_minus_mean:.3f}", ch, p))
        rows.append(None)
    body = [r"\begin{table}[t]",
            r"\tbl{Explanation quality at the 10\% budget on the test cohorts. "
            r"Fidelity columns and Char.\ are means over queries and encoders; parentheses give "
            r"query-level 95\% bootstrap confidence intervals for Char.\ ($n=300$ AtMAD; $n=192$ DTINet). "
            r"$p$ is the Holm-adjusted two-sided Wilcoxon comparison with ReGEX, after averaging "
            r"the repeated encoder scores within each query. Higher Fid$^+$ and Char., and lower Fid$^-$, "
            r"are preferable. PaGE-Link is our common-forward adaptation.}",
            r"{\tablefont\setlength{\tabcolsep}{3pt}\begin{tabular}{@{}llcccr@{}}\toprule",
            r"Graph & Method & $\mathrm{Fid}^{+}$ & $\mathrm{Fid}^{-}$ & Char.\ (95\% CI) & $p$\\\colrule"]
    for r in rows:
        if r is None:
            body.append(r"\colrule")
            continue
        body.append(" & ".join(r) + r"\\")
    if body[-1] == r"\colrule":
        body = body[:-1]
    body += [r"\botrule\end{tabular}}", r"\label{tab:main}", r"\end{table}"]
    save("tab_main.tex", "\n".join(body))


def table_encoder():
    cols, data = [], {}
    for t, tn in TASKS.items():
        d = rd(t, "summary_encoder.csv")
        if d is None:
            continue
        for enc in ENC:
            e = d[d.model == enc]
            if e.empty:
                continue
            cols.append((tn, enc))
            best = e.charact.max()
            for _, r in e.iterrows():
                star = "" if r.method == "regex" or np.isnan(r.p_holm) or r.p_holm >= 0.05 else r"$^{*}$"
                v = f"{r.charact:.3f}"
                v = (r"\textbf{" + v + "}") if np.isclose(r.charact, best) else v
                data[(r.method, tn, enc)] = v + star
    head1 = " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{tn}}}" for tn in TASKS.values()
                               if any(c[0] == tn for c in cols)) + r"\\"
    head2 = "Method & " + " & ".join({"sage": "SAGE", "rgcn": "RGCN", "hgt": "HGT"}[c[1]] for c in cols) + r"\\"
    body = [r"\begin{table}[t]",
            r"\tbl{Mean characterization score at the 10\% budget per encoder (test cohorts). Bold: best per "
            r"column; $^{*}$: differs from ReGEX (Holm-adjusted Wilcoxon $p<0.05$).}",
            r"{\tablefont\setlength{\tabcolsep}{4pt}\begin{tabular}{@{}l" + "c" * len(cols) + r"@{}}\toprule",
            head1, head2 + r"\colrule"]
    for m in METHOD_ORDER:
        body.append(SHORT[m] + " & " + " & ".join(data.get((m, c[0], c[1]), "--") for c in cols) + r"\\")
    body += [r"\botrule\end{tabular}}", r"\label{tab:encoder}", r"\end{table}"]
    save("tab_encoder.tex", "\n".join(body))


def table_tuning():
    from tune import SIZES, NECS, GEOMS, RELS
    sel = {t: json.load(open(os.path.join(C.TABLES_DIR, t, "tuning_selected.json")))
           for t in TASKS if os.path.exists(os.path.join(C.TABLES_DIR, t, "tuning_selected.json"))}
    g = lambda v: "\\{" + ", ".join(f"{x:g}" for x in v) + "\\}"
    rowdefs = [("GNNExplainer", r"$\lambda_s$", g(SIZES), lambda s: f"{s['gnnexplainer']['lam_size']:g}"),
               ("PGExplainer", r"$\lambda_s$", g(SIZES), lambda s: f"{s['pgexplainer']['lam_size']:g}"),
               ("PaGE-Link", r"$(\alpha,\beta)$", r"$\{0.01,0.1,1\}\times\{0.001,0.01,0.1\}$",
                lambda s: f"({s['pagelink']['alpha']:g}, {s['pagelink']['beta']:g})"),
               ("ReGEX (1)", r"$\lambda_s$", g(SIZES), lambda s: f"{s['regex']['lam_size']:g}"),
               ("", r"$\lambda_n$", g(NECS), lambda s: f"{s['regex']['lam_nec']:g}"),
               ("ReGEX (2)", r"$\lambda_g$", g(GEOMS), lambda s: f"{s['regex']['lam_geom']:g}"),
               ("", r"$\lambda_r$", g(RELS), lambda s: f"{s['regex']['lam_rel']:g}")]
    body = [r"\begin{table}[t]",
            r"\tbl{Hyper-parameter grids and values selected on validation pairs (criterion: characterization "
            r"score averaged over budgets and encoders). ReGEX is tuned in two stages; stage~2 searches the "
            r"priors at the stage-1 optimum.}",
            r"{\tablefont\setlength{\tabcolsep}{3pt}\begin{tabular}{@{}llp{5.2cm}cc@{}}\toprule",
            r"Method & Param. & Grid & AtMAD & DTINet\\\colrule"]
    for name, par, grid, f in rowdefs:
        vals = [f(sel[t]) if t in sel else "--" for t in TASKS]
        body.append(f"{name} & {par} & {grid} & " + " & ".join(vals) + r"\\")
    body += [r"\botrule\end{tabular}}", r"\label{tab:tuning}", r"\end{table}"]
    save("tab_tuning.tex", "\n".join(body))


VAR = {"full": "ReGEX (full)", "no_necessity": r"$-$ necessity", "no_geometry": r"$-$ geometric prior",
       "no_relation": r"$-$ relation prior", "no_priors": r"$-$ both priors", "no_gumbel": r"$-$ Gumbel relaxation"}


def table_ablation():
    body = [r"\begin{table}[t]",
            r"\tbl{Ablation of ReGEX on 100 test queries per encoder. Char.@10\%: mean characterization score at "
            r"the 10\% budget; Char.: mean over the four budgets; $p$: Holm-adjusted Wilcoxon test against the "
            r"full model at 10\%.}",
            r"{\tablefont\setlength{\tabcolsep}{3.5pt}\begin{tabular}{@{}lcccccc@{}}\toprule",
            r" & \multicolumn{3}{c}{AtMAD} & \multicolumn{3}{c}{DTINet}\\",
            r"Variant & Char.@10\% & Char. & $p$ & Char.@10\% & Char. & $p$\\\colrule"]
    A = {t: rd(t, "summary_ablation.csv") for t in TASKS}
    for v, lab in VAR.items():
        cells = []
        for t in TASKS:
            a = A[t]
            if a is None or v not in set(a.variant):
                cells += ["--"] * 3; continue
            r = a[a.variant == v].iloc[0]
            cells += [f"{r.charact:.3f}", f"{r.charact_allbudgets:.3f}", "--" if v == "full" else fmt_p(r.p_holm)]
        body.append(lab + " & " + " & ".join(cells) + r"\\")
    body += [r"\botrule\end{tabular}}", r"\label{tab:ablation}", r"\end{table}"]
    save("tab_ablation.tex", "\n".join(body))


def table_reliability():
    body = [r"\begin{table}[t]",
            r"\tbl{Reliability of the explanations (100 test queries per encoder, pooled). Seed stability: mean "
            r"pairwise Jaccard index of the top-10\% edge sets over 10 seeds (chance 0.053). Robustness: Jaccard "
            r"index between clean and noisy-feature explanations at $\sigma=0.1$. Randomization: Spearman "
            r"correlation between explanations of the trained model and of a model with all parameters "
            r"re-initialized (lower is better).}",
            r"{\tablefont\setlength{\tabcolsep}{3.5pt}\begin{tabular}{@{}lcccccc@{}}\toprule",
            r" & \multicolumn{3}{c}{AtMAD} & \multicolumn{3}{c}{DTINet}\\",
            r"Method & Seed & Noise & Random. & Seed & Noise & Random.\\\colrule"]
    for m in ["regex", "gnnexplainer", "ig", "saliency"]:
        cells = []
        for t in TASKS:
            st, ro, sa = rd(t, "summary_stability.csv"), rd(t, "summary_robustness.csv"), rd(t, "summary_sanity.csv")
            c1 = (f"{st[st.method == m].jaccard.iloc[0]:.3f}" if st is not None and m in set(st.method)
                  else "1.000$^{\\dagger}$" if m in ["ig", "saliency"] else "--")
            c2 = (f"{ro[(ro.method == m) & np.isclose(ro.sigma, 0.1)].jaccard.iloc[0]:.3f}" if ro is not None else "--")
            c3 = (f"{sa[(sa.method == m) & (sa.level == 'all')].spearman.iloc[0]:.3f}" if sa is not None else "--")
            cells += [c1, c2, c3]
        body.append(SHORT[m] + " & " + " & ".join(cells) + r"\\")
    body += [r"\botrule\end{tabular}}",
             r"\begin{tabnote}$^{\dagger}$Deterministic method.\end{tabnote}",
             r"\label{tab:reliability}", r"\end{table}"]
    save("tab_reliability.tex", "\n".join(body))


def table_paths():
    d = rd("atmad", "summary_paths.csv"); e = rd("dtinet", "summary_paths.csv")
    if d is None:
        return
    body = [r"\begin{table}[t]",
            r"\tbl{Path read-out of the 10\% explanations. Conn.: fraction of queries whose two endpoints are "
            r"linked by at most four explanation edges. Flow.: fraction of beam-search paths from the query gene "
            r"that pass through another literature flowering-time gene (flowering cohort).}",
            r"{\tablefont\setlength{\tabcolsep}{4pt}\begin{tabular}{@{}lcccc@{}}\toprule",
            r" & \multicolumn{3}{c}{AtMAD} & DTINet\\",
            r"Method & Conn.\ (test) & Conn.\ (flow.) & Flow. & Conn.\ (test)\\\colrule"]
    for m in METHOD_ORDER:
        g = lambda df, c, col: df[(df.cohort == c) & (df.method == m)][col]
        a1, a2, a3 = g(d, "test", "connected"), g(d, "flower", "connected"), g(d, "flower", "curated_hit")
        b1 = g(e, "test", "connected") if e is not None else pd.Series(dtype=float)
        f = lambda s: f"{s.iloc[0]:.3f}" if len(s) and not np.isnan(s.iloc[0]) else "--"
        body.append(f"{SHORT[m]} & {f(a1)} & {f(a2)} & {f(a3)} & {f(b1)}" + r"\\")
    body += [r"\botrule\end{tabular}}", r"\label{tab:paths}", r"\end{table}"]
    save("tab_paths.tex", "\n".join(body))


def table_context():
    body = [r"\begin{table}[t]",
            r"\tbl{Context shift on known positive test queries. Agreement is the fraction for which "
            r"full-graph and degree-capped-subgraph class predictions coincide. On this concordant subset "
            r"only, the last columns compare mean characterization at 10\%; this diagnostic does not "
            r"replace the full local benchmark or eliminate extraction bias.}",
            r"{\tablefont\setlength{\tabcolsep}{4pt}\begin{tabular}{@{}llrrrr@{}}\toprule",
            r"Graph & Encoder & Agreement & $n$ & ReGEX & IG\\\colrule"]
    for task, title in TASKS.items():
        shift = rd(task, "context_shift.csv")
        scores = rd(task, "summary_context_agreement.csv")
        if shift is None or scores is None:
            continue
        for model, encoder in ENC.items():
            fraction = shift[(shift.model == model) & (shift.cohort == "test")].label_agreement.mean()
            selected = scores[scores.model == model].set_index("method")
            n = int(selected.n.iloc[0])
            body.append(f"{title} & {encoder} & {fraction:.3f} & {n} & "
                        f"{selected.loc['regex','charact']:.3f} & {selected.loc['ig','charact']:.3f}" + r"\\")
    body += [r"\botrule\end{tabular}}", r"\label{tab:context}", r"\end{table}"]
    save("tab_context.tex", "\n".join(body))


def table_auxiliary():
    body = []
    for task, title in TASKS.items():
        frame = rd(task, "summary_main.csv")
        if frame is None:
            continue
        body += [r"\begin{table}[t]",
                 r"\tbl{" + title + r": median (interquartile range) of probability fidelities at 10\%. "
                 r"Time is observed amortized GPU seconds per query, excluding PGExplainer training; "
                 r"concurrent processes and CPU path searches limit direct latency comparisons.}",
                 r"{\tablefont\setlength{\tabcolsep}{4pt}\begin{tabular}{@{}lccr@{}}\toprule",
                 r"Method & Fid$^+$ & Fid$^-$ & Time (s)\\\colrule"]
        for _, r in frame.iterrows():
            body.append(f"{SHORT[r.method]} & {iqr_cell(r.fid_plus)} & {iqr_cell(r.fid_minus)} & {r.sec:.4f}" + r"\\")
        body += [r"\botrule\end{tabular}}", r"\end{table}"]
    save("tab_auxiliary.tex", "\n".join(body))

    frame = rd("atmad", "summary_main.csv")
    if frame is not None:
        body = [r"\begin{table}[t]",
                r"\tbl{Auxiliary AtMAD consistency at 10\%: median Spearman correlation. "
                r"Degree-normalized EGC uses selected incident-edge fraction; raw EGC uses selected "
                r"incident-edge count. Association counts originate from the same database as the graph, "
                r"so neither quantity constitutes external biological validation.}",
                r"{\tablefont\begin{tabular}{@{}lrr@{}}\toprule",
                r"Method & Normalized EGC & Raw EGC\\\colrule"]
        for _, r in frame.iterrows():
            body.append(f"{SHORT[r.method]} & {r.egc:.3f} & {r.egc_raw:.3f}" + r"\\")
        body += [r"\botrule\end{tabular}}", r"\label{tab:egc}", r"\end{table}"]
        save("tab_egc.tex", "\n".join(body))


if __name__ == "__main__":
    table_tuning(); table_main(); table_encoder(); table_ablation(); table_reliability(); table_paths(); table_auxiliary(); table_context()
