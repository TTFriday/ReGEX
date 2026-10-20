# -*- coding: utf-8 -*-
"""
train_dtinet.py — train the three encoders (GraphSAGE, RGCN, HGT) as
drug–protein link predictors on the DTINet graph, with the same
architecture, optimiser and early-stopping protocol as the AtMAD encoders
(Paper-2 train_link.py): full-batch training, Adam (lr 1e-3), 1:1 resampled
negatives, validation AUC every 5 epochs, patience 8 evaluations.
"""
import os, sys, time, json
import numpy as np
import torch
from scipy.stats import rankdata


def roc_auc_score(y, scores):
    """Binary ROC AUC from average ranks, including tied scores."""
    y, scores = np.asarray(y, dtype=bool), np.asarray(scores)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if not n_pos or not n_neg:
        raise ValueError("AUC requires both classes")
    ranks = rankdata(scores, method="average")
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision_score(y, scores):
    """Non-interpolated binary AP, grouping equal-score thresholds."""
    y, scores = np.asarray(y, dtype=bool), np.asarray(scores)
    if not y.any():
        return 0.0
    order = np.argsort(-scores, kind="stable")
    labels, ranked = y[order], scores[order]
    endpoints = np.r_[np.flatnonzero(np.diff(ranked)), len(ranked) - 1]
    tp = np.cumsum(labels)[endpoints]
    precision = tp / (endpoints + 1)
    recall = tp / y.sum()
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
sys.path.insert(0, C.P2_CODE_DIR)
from models import LinkPredictor

MAX_EPOCHS, LR, PATIENCE = 400, 1e-3, 8


def evaluate(model, data, pos, neg):
    model.eval()
    with torch.no_grad():
        emb = model.encode(data, {nt: data[nt].x for nt in data.node_types})
        s = lambda e: (emb["drug"][e[0]] * emb["protein"][e[1]]).sum(-1)
        y = np.r_[np.ones(pos.shape[1]), np.zeros(neg.shape[1])]
        p = torch.cat([s(pos), s(neg)]).cpu().numpy()
    return roc_auc_score(y, p), average_precision_score(y, p)


def main():
    dev = torch.device(C.DEVICE)
    pack = torch.load(os.path.join(C.DTI_DIR, "dtinet_hetero.pt"), map_location="cpu", weights_only=False)
    data, sp = pack["data"].to(dev), pack["splits"]
    os.makedirs(C.DTI_MODELS_DIR, exist_ok=True)
    T = lambda a: torch.as_tensor(a, dtype=torch.long, device=dev)
    tr, va, te, va_n, te_n = (T(sp[k]) for k in ["train_pos", "val_pos", "test_pos", "val_neg", "test_neg"])
    pos_set = set(map(tuple, np.concatenate([sp["train_pos"], sp["val_pos"], sp["test_pos"]], 1).T.tolist()))
    nd, npr = data["drug"].num_nodes, data["protein"].num_nodes
    summary = {}
    for mname in ["sage", "rgcn", "hgt"]:
        torch.manual_seed(C.SEED); rng = np.random.default_rng(C.SEED)
        feats = {nt: data[nt].x.shape[1] for nt in data.node_types}
        model = LinkPredictor(feats, C.HIDDEN, C.HIDDEN, model_name=mname,
                              metadata=(list(data.node_types), list(data.edge_types)),
                              num_relations=len(data.edge_types), num_heads=2,
                              dropout=C.DROPOUT, num_layers=C.GNN_LAYERS).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        lossf = torch.nn.BCEWithLogitsLoss()
        best, best_ep, state, pat, t0 = 0, 0, None, 0, time.time()
        for ep in range(1, MAX_EPOCHS + 1):
            model.train(); opt.zero_grad()
            emb = model.encode(data, {nt: data[nt].x for nt in data.node_types})
            neg = []
            while len(neg) < tr.shape[1]:
                a, b = int(rng.integers(nd)), int(rng.integers(npr))
                if (a, b) not in pos_set:
                    neg.append((a, b))
            neg = T(np.array(neg).T)
            s = lambda e: (emb["drug"][e[0]] * emb["protein"][e[1]]).sum(-1)
            lp, ln = s(tr), s(neg)
            loss = lossf(torch.cat([lp, ln]), torch.cat([torch.ones_like(lp), torch.zeros_like(ln)]))
            loss.backward(); opt.step()
            if ep % 5 == 0:
                auc, _ = evaluate(model, data, va, va_n)
                if auc > best:
                    best, best_ep, pat = auc, ep, 0
                    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    pat += 1
                if pat >= PATIENCE:
                    break
        model.load_state_dict(state)
        te_auc, te_ap = evaluate(model, data, te, te_n)
        print(f"{mname}: best val AUC {best:.4f} @ep{best_ep} | test AUC {te_auc:.4f} AP {te_ap:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        torch.save({"state": state, "val_auc": best, "test_auc": te_auc, "test_ap": te_ap,
                    "best_epoch": best_ep}, os.path.join(C.DTI_MODELS_DIR, f"{mname}_s{C.SEED}.pt"))
        summary[mname] = {"val_auc": best, "test_auc": te_auc, "test_ap": te_ap, "best_epoch": best_ep}
    json.dump(summary, open(os.path.join(C.DTI_MODELS_DIR, "summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
