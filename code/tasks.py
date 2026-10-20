# -*- coding: utf-8 -*-
"""
tasks.py — the two explanation benchmarks used in the paper.

atmad  : Arabidopsis multi-omics graph (gene, SNP, CpG, pathway, phenotype);
         query = (gene, phenotype) TWAS association; encoders from Paper 2.
dtinet : drug–target heterogeneous network of Luo et al. (2017)
         (drug, protein, disease, side effect); query = (drug, protein)
         interaction; encoders trained by train_dtinet.py.

A Task bundles the message-passing graph (held-out query edges removed),
query cohorts, training pairs (for PGExplainer), validation pairs (for
hyper-parameter selection), optional gene-level evidence, and checkpoints.
"""
import os, re, json
import numpy as np
import pandas as pd
import torch

import config as C


class Task:
    pass


def _atmad():
    t = Task()
    t.name = "atmad"
    t.pair_types = ("gene", "phenotype")
    t.data = C.load_data()
    sp = C.load_splits()
    nn = json.load(open(os.path.join(C.P2_DATA_DIR, "node_names.json"), encoding="utf-8"))
    t.names = nn
    gene_list, pheno_list = nn["gene"], nn["phenotype"]
    # independent per-gene omics evidence (supported-association counts)
    m = pd.read_csv(os.path.join(C.P2_CURATED_DIR, "15_gene_evidence_matrix.csv"))
    m["Gene"] = m["Gene"].map(lambda s: str(s).strip().upper())
    cols = [c for c in m.columns if c.startswith(("n_cis", "n_trans", "n_emqtl", "n_twas", "n_gwas"))]
    ev = m.set_index("Gene")[cols].sum(axis=1)
    t.evidence = {i: float(ev.get(g, 0.0)) for i, g in enumerate(gene_list)}
    # literature flowering-time genes (independent of AtMAD; build_flowering_set.py)
    lit = pd.read_csv(os.path.join(C.PROJECT_DIR, "data", "flowering_literature.csv"))
    gidx = {g: i for i, g in enumerate(gene_list)}
    t.curated = np.array([gidx[l] for l in lit["locus"].str.upper() if l in gidx], dtype=np.int64)
    t.curated_symbols = {gidx[l]: s for s, l in zip(lit["symbol"], lit["locus"].str.upper()) if l in gidx}
    flow_re = re.compile(C.FLOWER_PHENO_RE, re.I)
    flow_pheno = [i for i, p in enumerate(pheno_list) if flow_re.search(p)]
    # cohorts (same sampling as the original protocol)
    rng = np.random.default_rng(7)
    test_pos, cold = sp["test_pos"], set(sp["cold_phenotypes"])
    idx = [i for i in range(test_pos.shape[1]) if test_pos[1, i] not in cold]
    test = test_pos[:, rng.choice(idx, min(C.N_SAMPLES, len(idx)), replace=False)]
    # flowering cohort: literature genes that take part in message passing x five DTF phenotypes
    deg = np.zeros(len(gene_list))
    for (s_, r_, d_) in t.data.edge_types:
        ei = t.data[s_, r_, d_].edge_index
        if s_ == "gene":
            np.add.at(deg, ei[0].numpy(), 1)
    fg = [int(i) for i in t.curated if deg[i] > 0]
    t.flower_genes = fg
    flower = np.array([(g, p) for g in fg for p in flow_pheno[:5]], dtype=np.int64).T
    t.cohorts = {"test": np.asarray(test, dtype=np.int64), "flower": flower}
    r2 = np.random.default_rng(13)
    t.train_pairs = sp["train_pos"][:, r2.choice(sp["train_pos"].shape[1], C.PG_TRAIN_PAIRS, replace=False)]
    t.val_pairs = sp["val_pos"][:, r2.choice(sp["val_pos"].shape[1], 60, replace=False)]
    t.ckpt = lambda m: os.path.join(C.P2_MODELS_DIR, f"{m}_s{C.SEED}.pt")
    return t


def _dtinet():
    t = Task()
    t.name = "dtinet"
    t.pair_types = ("drug", "protein")
    pack = torch.load(os.path.join(C.DTI_DIR, "dtinet_hetero.pt"), map_location="cpu", weights_only=False)
    t.data, sp = pack["data"], pack["splits"]
    t.names = pack["names"]
    t.evidence, t.curated = None, None
    rng = np.random.default_rng(7)
    test = sp["test_pos"]
    t.cohorts = {"test": test[:, rng.permutation(test.shape[1])[:C.N_SAMPLES]]}
    r2 = np.random.default_rng(13)
    tr = sp["train_pos"]
    t.train_pairs = tr[:, r2.choice(tr.shape[1], min(C.PG_TRAIN_PAIRS, tr.shape[1]), replace=False)]
    va = sp["val_pos"]
    t.val_pairs = va[:, r2.choice(va.shape[1], min(60, va.shape[1]), replace=False)]
    t.ckpt = lambda m: os.path.join(C.DTI_MODELS_DIR, f"{m}_s{C.SEED}.pt")
    return t


def load_task(name):
    return {"atmad": _atmad, "dtinet": _dtinet}[name]()


def apply_selected(source="atmad"):
    """Set explainer hyper-parameters to those selected on the validation
    pairs of benchmark ``source`` by tune.py (every method is tuned on the
    validation split of each benchmark; test pairs are never used)."""
    path = os.path.join(C.TABLES_DIR, source, "tuning_selected.json")
    if not os.path.exists(path):
        print(f"[warn] {path} missing - using config defaults", flush=True)
        return None
    sel = json.load(open(path))
    r = sel["regex"]
    C.LAMBDA_SIZE, C.LAMBDA_NEC = r["lam_size"], r["lam_nec"]
    C.LAMBDA_GEOM, C.LAMBDA_REL = r["lam_geom"], r["lam_rel"]
    C.GNNEXP_LAMBDA_SIZE = sel["gnnexplainer"]["lam_size"]
    C.PG_LAMBDA_SIZE = sel["pgexplainer"]["lam_size"]
    if "pagelink" in sel:
        C.PAGELINK_ALPHA, C.PAGELINK_BETA = sel["pagelink"]["alpha"], sel["pagelink"]["beta"]
    return sel


def all_pairs(task):
    """(pairs [2, n], cohort labels) over all cohorts."""
    ps, kinds = [], []
    for k, v in task.cohorts.items():
        ps.append(v); kinds += [k] * v.shape[1]
    return np.concatenate(ps, axis=1), kinds
