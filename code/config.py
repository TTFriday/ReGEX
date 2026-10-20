# -*- coding: utf-8 -*-
"""
config.py — shared configuration for Paper 3 (ReGEX explainer).

Reuses the Paper-2 benchmark graph, trained model weights and curated
gene sets from atmad_paper2; all Paper-3 outputs land under this project.
"""
import os

# AtMAD benchmark assets can be placed next to this project or supplied
# through ATMAD_BENCHMARK_DIR when running the reproduction package.
_DEFAULT_P2 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "atmad_paper2")
if not os.path.exists(_DEFAULT_P2):
    _DEFAULT_P2 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "atmad_paper2")
if not os.path.exists(_DEFAULT_P2):
    _DEFAULT_P2 = r"D:\2026科研\科研投稿\atmad_paper2"
P2_DIR = os.environ.get("ATMAD_BENCHMARK_DIR", _DEFAULT_P2)
RAW_DATA_DIR = os.environ.get("ATMAD_RAW_DATA_DIR", r"D:\qodertu\rawdata")
P2_CODE_DIR = os.path.join(P2_DIR, "code")
P2_DATA_DIR = os.path.join(P2_DIR, "data")
P2_CURATED_DIR = os.path.join(P2_DATA_DIR, "curated")
P2_MODELS_DIR = os.path.join(P2_DIR, "results", "models")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(PROJECT_DIR, "code")
DTI_DIR = os.path.join(PROJECT_DIR, "data", "dtinet")          # DTINet benchmark
DTI_MODELS_DIR = os.path.join(PROJECT_DIR, "results", "models_dtinet")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
PAPER_DIR = os.path.join(PROJECT_DIR, "paper")

for d in [RESULTS_DIR, TABLES_DIR, FIGURES_DIR, PAPER_DIR]:
    os.makedirs(d, exist_ok=True)

# --- Paper-2 hyper-parameters reused for base models ---
HIDDEN = 32
GNN_LAYERS = 2
DROPOUT = 0.3
SEED = 42            # base models loaded from seed 42

# --- explanation sampling ---
N_SAMPLES = 300                 # test (gene, phenotype) pairs to explain
TOP_HOP = 2                     # explanation subgraph = 2-hop neighbourhood
MASK_STEPS = 100                # mask optimisation steps (Adam)
MASK_LR = 0.05
LAMBDA_SIZE = 0.5               # mask-size sparsity weight   (mean m)
LAMBDA_ENT = 0.1                # mask-entropy weight         (mean H(m))
LAMBDA_GEOM = 0.5               # geometric (graph-distance) prior weight
LAMBDA_REL = 0.1                # relation-prior KL weight
LAMBDA_NEC = 1.0                # necessity (complement-mask) weight
GNNEXP_LAMBDA_SIZE = 0.5        # GNNExplainer size weight (tuned separately)
PG_LAMBDA_SIZE = 0.5            # PGExplainer size weight (tuned separately)
PAGELINK_ALPHA = 0.1            # PaGE-Link path-loss weights (tuned separately)
PAGELINK_BETA = 0.01
GUMBEL_TAU = 0.5
PG_EPOCHS = 10                  # PGExplainer training epochs
PG_TRAIN_PAIRS = 200            # training pairs used to fit PGExplainer
MASK_THRESHOLD = 0.5            # binarisation threshold for explanation edges
MAX_SUBGRAPH_EDGES = 12000      # cap on 2-hop subgraph size before sampling
FRACS = [0.05, 0.10, 0.20, 0.50]  # sparsity budgets (fraction of edges kept)

try:
    import torch as _torch_dev
    DEVICE = "cuda" if _torch_dev.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

# --- path extraction ---
BEAM_WIDTH = 5                  # beam width for path enumeration
MAX_PATH_LEN = 4                # max number of edges in an explanation path
TOP_PATHS = 8                   # top-K paths reported per explanation

# --- evaluation ---
TOP_K_RECALL = 50
FLOWER_PHENO_RE = r"(flower|bolt|vernal|dtf|days.?to)"

try:
    import torch as _torch
except ImportError:
    _torch = None  # paper-rendering only; torch required solely for load_data/load_splits

def load_data(mask_eval_edges=True):
    """AtMAD heterogeneous graph as used for message passing at train time.

    Validation / test / cold-start TWAS edges are removed (identical to
    Paper-2 ``train_link.mask_twas``) so that explanations are computed on the
    same graph the link predictors were trained and evaluated on, and the
    query association itself is never part of its own explanation subgraph.
    """
    data = _torch.load(os.path.join(P2_DATA_DIR, "atmad_hetero.pt"),
                       map_location="cpu", weights_only=False)
    if mask_eval_edges:
        tr = _torch.as_tensor(load_splits()["train_pos"], dtype=_torch.long)
        data["gene", "twas", "phenotype"].edge_index = tr
        data["phenotype", "twas_rev", "gene"].edge_index = tr.flip(0)
    return data

def load_splits():
    return _torch.load(os.path.join(P2_DATA_DIR, "splits.pt"),
                       map_location="cpu", weights_only=False)
