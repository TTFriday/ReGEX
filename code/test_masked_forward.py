# -*- coding: utf-8 -*-
"""
test_masked_forward.py — numerical check that the differentiable masked
forward equals hard edge deletion for binary masks, and that gradients reach
the mask, for GraphSAGE, RGCN and HGT.
"""
import os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from explainers import build_subgraph, restore_model
from masked_forward import masked_logit, hard_logit


def main(n_pairs=8, n_masks=4):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = C.load_data(); splits = C.load_splits()
    tp = splits["test_pos"]
    rng = np.random.default_rng(11)
    picks = rng.choice(tp.shape[1], n_pairs, replace=False)
    subs = []
    for k in picks:
        g, p = int(tp[0, k]), int(tp[1, k])
        sub, keep, local, _ = build_subgraph(data, g, p)
        subs.append((sub.to(dev), local["gene"][g], local["phenotype"][p]))
    for mname in ["sage", "rgcn", "hgt"]:
        ck = torch.load(os.path.join(C.P2_MODELS_DIR, f"{mname}_s{C.SEED}.pt"),
                        map_location="cpu", weights_only=False)
        model = restore_model(mname, data, ck).to(dev)
        max_err, grad_ok, t0 = 0.0, True, time.time()
        for sub, gl, pl in subs:
            for t in range(n_masks + 1):
                keep = {et: (rng.random(sub[et].edge_index.shape[1]) < (1.0 if t == 0 else 0.3))
                        for et in sub.edge_types}
                masks = {et: torch.tensor(keep[et], dtype=torch.float32, device=dev)
                         for et in sub.edge_types}
                with torch.no_grad():
                    s_soft = float(masked_logit(model, sub, data, gl, pl, masks))
                s_hard = hard_logit(model, sub, data, gl, pl, keep)
                max_err = max(max_err, abs(s_soft - s_hard) / (abs(s_hard) + 1e-6))
            # gradient reaches the mask
            ms = {et: torch.full((sub[et].edge_index.shape[1],), 0.5, device=dev,
                                 requires_grad=True) for et in sub.edge_types}
            masked_logit(model, sub, data, gl, pl, ms).backward()
            gn = sum(float(m.grad.abs().sum()) for m in ms.values() if m.grad is not None)
            grad_ok &= gn > 0
        print(f"{mname:5s} max rel. error soft-vs-hard = {max_err:.2e} | "
              f"grad reaches mask: {grad_ok} | {time.time()-t0:.1f}s on {dev}", flush=True)


if __name__ == "__main__":
    main()
