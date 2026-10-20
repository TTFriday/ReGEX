# -*- coding: utf-8 -*-
"""
build_dtinet.py — heterogeneous graph for the DTINet benchmark
(Luo et al., Nat. Commun. 8:573, 2017; https://github.com/luoyunan/DTINet).

Node types : drug (708), protein (1,512), disease (5,603), side_effect (4,192)
Edge types : drug-protein interaction (query relation), drug-drug interaction,
             protein-protein interaction, drug-disease, protein-disease,
             drug-side-effect; every type has an explicit reverse type.
Features   : drug  = chemical-structure similarity profile (708-d);
             protein = sequence similarity profile (1,512-d) when available;
             disease / side effect (and protein otherwise) = fixed Gaussian
             random features (64-d), i.e. a featureless setting.
Split      : drug-protein interactions split 80/10/10 (seed 42); validation and
             test interactions are removed from the message-passing graph;
             1:1 fixed negatives for validation / test.
"""
import os, sys
import numpy as np
import torch
from torch_geometric.data import HeteroData

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

D = C.DTI_DIR


def mat(name):
    return np.loadtxt(os.path.join(D, name))


def names(name):
    return [l.strip() for l in open(os.path.join(D, name), encoding="utf-8") if l.strip()]


def edges_of(M, upper=False):
    if upper:
        M = np.triu(M, 1)
    r, c = np.nonzero(M)
    return torch.tensor(np.vstack([r, c]), dtype=torch.long)


def main():
    rng = np.random.default_rng(42)
    dp = mat("mat_drug_protein.txt")
    dd, pp = mat("mat_drug_drug.txt"), mat("mat_protein_protein.txt")
    ddis, pdis, dse = mat("mat_drug_disease.txt"), mat("mat_protein_disease.txt"), mat("mat_drug_se.txt")
    nd, npr, ndis, nse = dp.shape[0], dp.shape[1], ddis.shape[1], dse.shape[1]
    print(f"drugs {nd}, proteins {npr}, diseases {ndis}, side effects {nse}")

    # --- split drug-protein interactions
    pos = np.vstack(np.nonzero(dp))
    perm = rng.permutation(pos.shape[1])
    n_te = n_va = int(round(0.1 * pos.shape[1]))
    te, va, tr = pos[:, perm[:n_te]], pos[:, perm[n_te:n_te + n_va]], pos[:, perm[n_te + n_va:]]
    pos_set = set(map(tuple, pos.T.tolist()))

    def negatives(n):
        out = set()
        while len(out) < n:
            a, b = int(rng.integers(nd)), int(rng.integers(npr))
            if (a, b) not in pos_set:
                out.add((a, b))
        return np.array(sorted(out)).T

    neg_va, neg_te = negatives(va.shape[1]), negatives(te.shape[1])

    # --- features
    g = np.random.default_rng(0)
    data = HeteroData()
    data["drug"].x = torch.tensor(mat("Similarity_Matrix_Drugs.txt"), dtype=torch.float32)
    psim = os.path.join(D, "Similarity_Matrix_Proteins.txt")
    if os.path.getsize(psim) > 1_000_000:
        data["protein"].x = torch.tensor(np.loadtxt(psim) / 100.0, dtype=torch.float32)
        pfeat = "sequence similarity"
    else:
        data["protein"].x = torch.tensor(g.standard_normal((npr, 64)), dtype=torch.float32)
        pfeat = "random"
    data["disease"].x = torch.tensor(g.standard_normal((ndis, 64)), dtype=torch.float32)
    data["side_effect"].x = torch.tensor(g.standard_normal((nse, 64)), dtype=torch.float32)
    print("protein features:", pfeat)

    rels = {("drug", "dti", "protein"): torch.tensor(tr, dtype=torch.long),
            ("drug", "ddi", "drug"): edges_of(dd, upper=True),
            ("protein", "ppi", "protein"): edges_of(pp, upper=True),
            ("drug", "drug_disease", "disease"): edges_of(ddis),
            ("protein", "protein_disease", "disease"): edges_of(pdis),
            ("drug", "drug_se", "side_effect"): edges_of(dse)}
    for (s, r, d), ei in rels.items():
        data[s, r, d].edge_index = ei
        data[d, r + "_rev", s].edge_index = ei.flip(0)
        print(f"  {s}-{r}-{d}: {ei.shape[1]}")

    splits = {"train_pos": tr, "val_pos": va, "test_pos": te,
              "val_neg": neg_va, "test_neg": neg_te}
    nm = {"drug": names("drug.txt"), "protein": names("protein.txt"),
          "disease": names("disease.txt"), "side_effect": names("se.txt"),
          "protein_features": pfeat}
    torch.save({"data": data, "splits": splits, "names": nm},
               os.path.join(D, "dtinet_hetero.pt"))
    print(f"split train/val/test = {tr.shape[1]}/{va.shape[1]}/{te.shape[1]}")


if __name__ == "__main__":
    main()
