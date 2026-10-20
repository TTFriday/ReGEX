# -*- coding: utf-8 -*-
"""
masked_forward.py — differentiable edge-masked forward passes for the three
Paper-2 encoders (GraphSAGE, RGCN, HGT).

Each function re-uses the trained parameters of the PyG layers but exposes a
continuous per-edge weight m_e in [0, 1]:

  SAGE : weighted mean aggregation   sum_j m_ij x_j / sum_j m_ij
  RGCN : weighted mean per relation  (same, within each relation)
  HGT  : weighted attention softmax  m_ij exp(e_ij) / sum_k m_ik exp(e_ik)

For binary masks these are *exactly* the forward passes on the subgraph with
the m = 0 edges deleted (verified in test_masked_forward.py), so gradients of
the masked prediction w.r.t. the mask are well defined and the hard top-K
evaluation is consistent with the quantity being optimised.

Relation ids follow the global ``data.edge_types`` order used at training
time, regardless of which edge types happen to occur in a subgraph.
"""
import math
import torch
import torch.nn.functional as F
from torch_geometric.utils import scatter
from torch_geometric.utils.hetero import construct_bipartite_edge_index


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def full_edge_dict(sub, edge_types, device=None):
    """Edge-index dict over *all* global edge types (empty where absent)."""
    out = {}
    for et in edge_types:
        if et in sub.edge_types:
            ei = sub[et].edge_index
        else:
            ei = torch.zeros((2, 0), dtype=torch.long)
        out[et] = ei.to(device) if device is not None else ei
    return out


# "clamp" (default): mean aggregation with denominator max(sum m, 1) and HGT
# attention scaled by min(sum m, 1). Both equal hard edge deletion for binary
# masks, but - unlike the plain weighted mean ("ratio") - they are not
# invariant to a common rescaling of a node's incoming masks, so a soft mask
# can express the removal of all evidence around a node and the masked logit is
# continuous at m = 0 (as required, e.g., by integrated gradients).
MASK_MODE = "clamp"


def _den(d):
    return d.clamp_min(1.0) if MASK_MODE == "clamp" else d.clamp_min(1e-12)


def _weighted_mean(x_src, src, dst, w, n_dst):
    num = scatter(x_src[src] * w.unsqueeze(-1), dst, dim=0, dim_size=n_dst, reduce="sum")
    den = scatter(w, dst, dim=0, dim_size=n_dst, reduce="sum")
    return num / _den(den).unsqueeze(-1)


def _flat(sub, node_types, eid, masks, device):
    """Flatten typed edges to homogeneous ids; returns src, dst, w, rel."""
    offs, off = {}, 0
    for nt in node_types:
        offs[nt] = off
        off += sub[nt].num_nodes
    srcs, dsts, ws, rels = [], [], [], []
    for r, (s, rel, d) in enumerate(eid.keys()):
        ei = eid[(s, rel, d)]
        if ei.shape[1] == 0:
            continue
        srcs.append(ei[0] + offs[s]); dsts.append(ei[1] + offs[d])
        m = masks.get((s, rel, d)) if masks is not None else None
        ws.append(m if m is not None else torch.ones(ei.shape[1], device=device))
        rels.append(torch.full((ei.shape[1],), r, dtype=torch.long, device=device))
    if not srcs:
        z = torch.zeros(0, dtype=torch.long, device=device)
        return z, z, torch.zeros(0, device=device), z, offs, off
    return (torch.cat(srcs), torch.cat(dsts), torch.cat(ws), torch.cat(rels), offs, off)


def _project(model, sub, node_types):
    return torch.cat([model.encoders[nt](sub[nt].x) for nt in node_types], dim=0)


# --------------------------------------------------------------------------
# SAGE / RGCN
# --------------------------------------------------------------------------
def _sage(model, x, src, dst, w):
    convs = model.gnn.convs
    for i, conv in enumerate(convs):
        agg = _weighted_mean(x, src, dst, w, x.size(0))
        out = conv.lin_l(agg)
        if conv.root_weight:
            out = out + conv.lin_r(x)
        if conv.normalize:
            out = F.normalize(out, p=2.0, dim=-1)
        x = out
        if i < len(convs) - 1:
            x = F.relu(x)
    return x


def _rgcn(model, x, src, dst, w, rel):
    """RGCN with per-relation weighted mean, vectorised over all edges:
    out_i = x_i W_root + b + sum_r sum_{j in N_r(i)} m_ij x_j W_r / sum_{j in N_r(i)} m_ij."""
    convs = model.gnn.convs
    n = x.size(0)
    for i, conv in enumerate(convs):
        weight = conv.weight
        if conv.num_bases is not None:
            weight = (conv.comp @ weight.view(conv.num_bases, -1)).view(
                conv.num_relations, conv.in_channels_l, conv.out_channels)
        R = conv.num_relations
        msg = torch.bmm(x[src].unsqueeze(1), weight[rel]).squeeze(1)          # [E, out]
        key = dst * R + rel
        den = scatter(w, key, dim=0, dim_size=n * R, reduce="sum")
        coef = w / _den(den)[key]
        out = scatter(msg * coef.unsqueeze(-1), dst, dim=0, dim_size=n, reduce="sum")
        if conv.root is not None:
            out = out + x @ conv.root
        if conv.bias is not None:
            out = out + conv.bias
        x = out
        if i < len(convs) - 1:
            x = F.relu(x)
    return x


# --------------------------------------------------------------------------
# HGT
# --------------------------------------------------------------------------
def _hgt_conv(conv, x_dict, eid, masks):
    Fo, H = conv.out_channels, conv.heads
    D = Fo // H
    dev = next(iter(x_dict.values())).device
    kqv = conv.kqv_lin(x_dict)
    k_d, q_d, v_d = {}, {}, {}
    for key, val in kqv.items():
        k, q, v = torch.tensor_split(val, 3, dim=1)
        k_d[key], q_d[key], v_d[key] = k.view(-1, H, D), q.view(-1, H, D), v.view(-1, H, D)
    q, dst_offset = conv._cat(q_d)

    # source features with relation-specific transforms (as in HGTConv)
    n_et = len(conv.edge_types)
    ks, vs, tl, src_offset, cum = [], [], [], {}, 0
    for et in eid.keys():
        s = et[0]
        N = k_d[s].size(0)
        src_offset[et] = cum
        cum += N
        tv = torch.arange(H, dtype=torch.long, device=dev).view(-1, 1).repeat(1, N) * n_et \
            + conv.edge_types_map[et]
        tl.append(tv); ks.append(k_d[s]); vs.append(v_d[s])
    ks = torch.cat(ks, 0).transpose(0, 1).reshape(-1, D)
    vs = torch.cat(vs, 0).transpose(0, 1).reshape(-1, D)
    tv = torch.cat(tl, 1).flatten()
    k = conv.k_rel(ks, tv).view(H, -1, D).transpose(0, 1)
    v = conv.v_rel(vs, tv).view(H, -1, D).transpose(0, 1)

    edge_index, edge_attr = construct_bipartite_edge_index(
        eid, src_offset, dst_offset, edge_attr_dict=conv.p_rel, num_nodes=k.size(0))
    w = torch.cat([masks[et] if (masks is not None and masks.get(et) is not None)
                   else torch.ones(eid[et].shape[1], device=dev) for et in eid.keys()])
    src, dst = edge_index[0], edge_index[1]
    n_dst = q.size(0)
    alpha = (q[dst] * k[src]).sum(-1) * edge_attr / math.sqrt(D)          # [E, H]
    amax = scatter(alpha.detach(), dst, dim=0, dim_size=n_dst, reduce="max")
    ex = torch.exp(alpha - amax[dst]) * w.unsqueeze(-1)
    den = scatter(ex, dst, dim=0, dim_size=n_dst, reduce="sum")
    a = ex / (den[dst] + 1e-16)
    if MASK_MODE == "clamp":
        mass = scatter(w, dst, dim=0, dim_size=n_dst, reduce="sum").clamp_max(1.0)
        a = a * mass[dst].unsqueeze(-1)
    out = scatter(v[src] * a.unsqueeze(-1), dst, dim=0, dim_size=n_dst, reduce="sum")
    out = out.view(-1, Fo)

    out_dict = {}
    for nt, st in dst_offset.items():
        if nt in conv.dst_node_types:
            out_dict[nt] = out[st:st + q_d[nt].size(0)]
    a_dict = conv.out_lin({kk: F.gelu(vv) for kk, vv in out_dict.items()})
    for nt in list(out_dict.keys()):
        o = a_dict[nt]
        if o.size(-1) == x_dict[nt].size(-1):
            al = conv.skip[nt].sigmoid()
            o = al * o + (1 - al) * x_dict[nt]
        out_dict[nt] = o
    return out_dict


def _hgt(model, sub, node_types, eid, masks):
    g = model.gnn
    x = {nt: F.relu(g.lin_in(model.encoders[nt](sub[nt].x))) for nt in node_types}
    x = _hgt_conv(g.conv1, x, eid, masks)
    x = {k: F.relu(v) for k, v in x.items()}
    x = _hgt_conv(g.conv2, x, eid, masks)
    return {k: g.lin_out(v) for k, v in x.items()}


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
def masked_encode(model, sub, data, masks=None):
    """Node embeddings on ``sub`` with continuous edge weights ``masks``
    ({edge_type: tensor[n_e]}; missing types / None -> weight 1)."""
    node_types = list(data.node_types)
    edge_types = list(data.edge_types)
    dev = next(model.parameters()).device
    eid = full_edge_dict(sub, edge_types, dev)
    if model.model_name == "hgt":
        return _hgt(model, sub, node_types, eid, masks)
    x = _project(model, sub, node_types)
    src, dst, w, rel, offs, _ = _flat(sub, node_types, eid, masks, dev)
    h = _sage(model, x, src, dst, w) if model.model_name == "sage" \
        else _rgcn(model, x, src, dst, w, rel)
    return {nt: h[offs[nt]:offs[nt] + sub[nt].num_nodes] for nt in node_types}


def masked_logit(model, sub, data, g_local, p_local, masks=None,
                 pair_types=("gene", "phenotype")):
    """Link logit of the query pair under edge weights ``masks``."""
    emb = masked_encode(model, sub, data, masks)
    a, b = pair_types
    return (emb[a][g_local] * emb[b][p_local]).sum()


def hard_logit(model, sub, data, g_local, p_local, keep=None,
               pair_types=("gene", "phenotype")):
    """Reference logit from the original ``model.encode`` on the subgraph with
    edges deleted according to boolean ``keep`` ({edge_type: bool array}).
    Uses all global edge types in training order (fixes relation ids)."""
    dev = next(model.parameters()).device
    eid = full_edge_dict(sub, list(data.edge_types), dev)
    if keep is not None:
        for et in eid:
            if et in keep:
                k = torch.as_tensor(keep[et], dtype=torch.bool, device=dev)
                eid[et] = eid[et][:, k]
    xd = {nt: sub[nt].x for nt in data.node_types}
    with torch.no_grad():
        emb = model.encode(sub, xd, eid)
    a, b = pair_types
    return float((emb[a][g_local] * emb[b][p_local]).sum())
