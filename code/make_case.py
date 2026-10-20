# -*- coding: utf-8 -*-
"""Illustrative first DTINet test query, selected by index before examining
its explanation score. Draw up to six selected edges incident on each query
endpoint; this display subset is not the full top-10% explanation.
"""
import json
import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import config as C
from tasks import load_task
from explainers import build_subgraph


def main():
    task = load_task('dtinet')
    a, b = map(int, task.cohorts['test'][:, 0])
    sub, _, local, _ = build_subgraph(task.data, a, b, task.pair_types)
    etypes = [et for et in sub.edge_types if sub[et].edge_index.shape[1] > 0]
    with np.load(os.path.join(C.TABLES_DIR, 'dtinet', 'weights_sage.npz')) as archive:
        weights = archive['0|regex'].astype(float)
    k = max(1, int(round(.1 * len(weights))))
    selected = set(np.argsort(-(weights + np.random.default_rng(0).random(len(weights))*1e-9))[:k])
    roots = [('drug', local['drug'][a]), ('protein', local['protein'][b])]
    incident = [[], []]
    offset = 0
    for et in etypes:
        ei = sub[et].edge_index.numpy()
        for j in range(ei.shape[1]):
            if offset+j not in selected:
                continue
            u, v = (et[0], int(ei[0,j])), (et[2], int(ei[1,j]))
            rel = et[1].removesuffix('_rev')
            for side, root in enumerate(roots):
                if root in (u, v):
                    other = v if root == u else u
                    if other not in roots:
                        incident[side].append((root, other, rel, float(weights[offset+j])))
        offset += ei.shape[1]
    shown = []
    for edges in incident:
        seen = set()
        for edge in sorted(edges, key=lambda v: -v[3]):
            if edge[1] not in seen:
                seen.add(edge[1]); shown.append(edge)
            if len(seen) >= 6:
                break
    inv = {nt: {v:g for g,v in mapping.items()} for nt,mapping in local.items()}
    drug_id, protein_id = task.names['drug'][a], task.names['protein'][b]
    with zipfile.ZipFile(os.path.join(C.DTI_DIR,'dtinet_repo.zip')) as archive:
        dictionary = {}
        for kind in ['drug', 'protein']:
            text = archive.read(f'DTINet-master/data/{kind}_dict_map.txt').decode()
            dictionary[kind] = dict(line.split(':',1) for line in text.splitlines() if ':' in line)
    root_labels = [dictionary['drug'].get(drug_id, drug_id), dictionary['protein'].get(protein_id, protein_id)]
    pos = {roots[0]: (-.85,0), roots[1]: (.85,0)}
    owner = {}
    for root, node, rel, w in shown:
        owner.setdefault(node, set()).add(roots.index(root))
    for sides in [{0}, {1}, {0,1}]:
        nodes = [node for node, ids in owner.items() if ids == sides]
        x = -2.25 if sides == {0} else 2.25 if sides == {1} else 0.0
        for node, y in zip(nodes, np.linspace(-1.05,1.05,len(nodes)) if len(nodes)>1 else [0.8]):
            pos[node] = (x,y)
    fig, axes = plt.subplots(1,2,figsize=(7.0,2.9),gridspec_kw={'width_ratios':[2.5,1]})
    ax = axes[0]
    colours = {'drug':'#2980B9','protein':'#C0392B','disease':'#7D3C98','side_effect':'#17A589'}
    for u,v,rel,w in shown:
        ax.plot([pos[u][0],pos[v][0]],[pos[u][1],pos[v][1]],color=colours[v[0]],lw=1.5,alpha=.65,zorder=1)
    for node,(x,y) in pos.items():
        isroot = node in roots
        ax.scatter([x],[y],s=140 if isroot else 35,color=colours[node[0]],zorder=2)
        if isroot:
            label = root_labels[roots.index(node)]
            ax.text(x,y-.16,label,ha='center',va='top',fontsize=8,fontweight='bold')
        else:
            global_idx = inv[node[0]][node[1]]
            label = str(task.names[node[0]][global_idx])
            if node[0] in dictionary:
                label = dictionary[node[0]].get(label, label)
            if len(label)>27:
                label=label[:24]+'...'
            ax.text(x + (.12 if x>0 else -.12),y,label,
                    ha='left' if x>0 else 'right',va='center',fontsize=5.8)
    ax.set_xlim(-4.3,4.3);ax.set_ylim(-1.3,1.3);ax.axis('off')
    ax.set_title('(a) Selected query-incident edges (display subset)',fontsize=8,loc='left')
    for kind in ['drug','protein','disease','side_effect']:
        ax.scatter([],[],s=30,color=colours[kind],label=kind.replace('_',' '))
    ax.legend(frameon=False,ncol=4,fontsize=6,loc='upper center',bbox_to_anchor=(.5,-.05))
    frame = pd.read_csv(os.path.join(C.TABLES_DIR,'dtinet','main_sage.csv'))
    r = frame[(frame.pair_id==0)&(frame.method=='regex')&(frame.frac==.1)].iloc[0]
    ax=axes[1]
    values=[r.p_full,r.p_keep,r.p_drop]
    ax.bar(range(3),values,color=['#555555','#C0392B','#BDC3C7'],width=.65)
    ax.set_xticks(range(3));ax.set_xticklabels(['full','keep','remove'],fontsize=7)
    ax.set_ylim(0,1.15);ax.set_ylabel('predicted-class probability',fontsize=7)
    for x,v in enumerate(values):
        ax.text(x,v+.025,f'{v:.3f}',ha='center',fontsize=7)
    ax.set_title('(b) Edge intervention',fontsize=8,loc='left')
    ax.spines[['top','right']].set_visible(False)
    fig.suptitle(f'{root_labels[0]} ({drug_id}) – {root_labels[1]} ({protein_id})',fontsize=9)
    fig.tight_layout()
    for ext in ['pdf','png']:
        fig.savefig(os.path.join(C.FIGURES_DIR,'Fig7_case.'+ext),dpi=400,bbox_inches='tight')
    record = dict(pair_id=0,drug=drug_id,protein=protein_id,drug_name=root_labels[0],target_name=root_labels[1],
                  p_full=float(r.p_full),p_keep=float(r.p_keep),p_drop=float(r.p_drop),
                  K=int(r.K),n_edges=int(r.n_edges),display_edges=len(shown),
                  selection='First saved DTINet test query; displayed incident edges selected only after fixing the query.')
    with open(os.path.join(C.TABLES_DIR,'dtinet','case.json'),'w') as f:
        json.dump(record,f,indent=2)
    print(record)


if __name__ == '__main__':
    main()
