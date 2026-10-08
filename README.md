# ReGEX: Relation-aware Geometric Explainer for Multi-omics Gene–Phenotype Associations

This repository contains the full experimental pipeline behind the paper:

> **ReGEX: Relation-aware Geometric Explainer for Interpretable Genotype–Phenotype Association in Multi-omics Heterogeneous Graphs**

ReGEX learns a sparse, geometry- and relation-regularised Gumbel edge mask on the 2-hop subgraph of a target (gene, phenotype) pair, and extracts layer-resolved explanation paths. It is evaluated against Mask-Only (GNNExplainer-style), input-gradient saliency, integrated gradients, and random attribution under a unified Fidelity–Sparsity–Consistency (FSC) framework on the AtMAD *Arabidopsis thaliana* multi-omics graph, across three base link predictors (GraphSAGE, RGCN, HGT).

## Repository layout

```
.
├── code/                 # ReGEX explainer and evaluation code
│   ├── config.py         #   shared configuration (paths, hyper-parameters)
│   ├── explainers.py     #   ReGEX + MaskOnly + Saliency + IG + Random
│   ├── evaluate.py       #   Fidelity± / EGC / rank-IG metrics
│   ├── paths.py          #   beam-search explanation-path extraction
│   ├── run_experiments.py#   main experiment entry point
│   ├── make_summary.py   #   aggregates per-pair results into summary tables
│   ├── stats_subgraph.py #   explanation-subgraph statistics (Table 1)
│   └── make_figures.py, make_fig1.py   # figure code (Figs. 1–4)
├── paper2_deps/          # dependency pipeline (AtMAD-Graph benchmark)
│   ├── config.py         #   benchmark configuration
│   ├── models.py         #   LinkPredictor (GraphSAGE / RGCN / HGT)
│   ├── build_graph.py    #   builds the heterogeneous graph from AtMAD tables
│   └── train_link.py     #   trains the base link predictors
├── data/
│   └── curated/          # curated gene-level evidence matrix and
│                         # flowering-time candidate gene set
├── results/
│   ├── tables/           # summary tables reported in the paper
│   └── figures/          # figures reported in the paper (PNG + PDF)
└── requirements.txt
```

## Environment

- Python ≥ 3.9
- PyTorch ≥ 2.0, PyTorch Geometric ≥ 2.4
- pandas, numpy, scipy, scikit-learn, matplotlib

```bash
pip install -r requirements.txt
```

## Data

The heterogeneous graph is built from the eight association tables of the
**AtMAD** database (*Arabidopsis thaliana* Multi-omics Association Database,
<http://www.megabionet.org/atmad>). The original tables are redistributed by
AtMAD and are not re-hosted here.

1. Download the eight AtMAD association tables (Table 1 in the paper of
   Lan et al., *Nucleic Acids Research* 2021, D1445–D1451) into a local
   directory, e.g. `paper2_deps/rawdata/`:
   ```bash
   mkdir -p paper2_deps/rawdata
   # download the eight AtMAD tables into paper2_deps/rawdata/
   ```
2. Build the heterogeneous graph (nodes: 21,613 genes, 86,949 SNPs, 15,735
   CpGs, 267 pathways, 280 phenotypes; eight relation families):
   ```bash
   cd paper2_deps
   python build_graph.py
   cd ..
   ```
   This writes `paper2_deps/data/atmad_hetero.pt`, `splits.pt`,
   `node_names.json` and `graph_stats.json`.

If your raw tables live elsewhere, set `ATMAD_RAW_DIR` to that directory
before running.

## Base models

The paper uses three base link predictors (GraphSAGE, RGCN, HGT, 2 layers,
32 hidden units, seed 42). Train them with:

```bash
cd paper2_deps
python train_link.py      # trains all models and seeds
cd ..
```

The full run covers seven architectures and five seeds; for this paper only
the `*_s42.pt` checkpoints of `sage`, `rgcn` and `hgt` are needed, which are
written to `paper2_deps/results/models/`.

## Reproducing the experiments

Smoke test (20 pairs per method, fast):

```bash
python code/run_experiments.py --smoke
```

Full run (3 models × 5 methods × 450 pairs × 4 sparsity budgets;
a few minutes per model on CPU):

```bash
python code/run_experiments.py --models sage rgcn hgt
```

Run the three models in parallel (recommended):

```bash
python code/run_experiments.py --models sage --tag full &
python code/run_experiments.py --models rgcn --tag full &
python code/run_experiments.py --models hgt  --tag full &
wait
```

Per-pair outputs are written to `results/tables/explanations_{tag}_{model}.csv`,
`paths_{tag}_{model}.csv` and `path_biology_{tag}_{model}.csv`.

Aggregate the results and regenerate the tables and figures:

```bash
python code/make_summary.py
python code/make_figures.py
python code/make_fig1.py
python code/stats_subgraph.py
```

## Results

`results/tables/` contains the six summary tables reported in the paper
(median ± MAD over encoders and pairs at the 10% sparsity budget):

| File | Paper table |
|---|---|
| `summary_main_frac10.csv` | Table 2 (FSC metrics, pooled) |
| `summary_method_frac.csv` | Table 3 (Fidelity+ across budgets) |
| `summary_per_model_frac10.csv` | Table 4 (per-encoder Fidelity+ / EGC) |
| `summary_path_biology.csv` | Table 5 (path coverage, seed-/pass-fraction) |
| `summary_rel_hist.csv` | Table 6 (molecular-layer histogram) |

`results/figures/` contains the four paper figures in PNG and PDF.

## License

The code in this repository is released under a permissive licence for
academic use. The original AtMAD association tables are the property of the
AtMAD database and are referenced, not redistributed.

## Citation

If you use this code or the results, please cite the paper (reference list
entry 26 in the manuscript) together with the AtMAD database and the
AtMAD-Graph benchmark paper.
# ReGEX
