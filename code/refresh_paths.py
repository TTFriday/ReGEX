# -*- coding: utf-8 -*-
"""Recompute path read-outs from saved weights using the identical tie
seed as final fidelity evaluation; also report full subgraph connectivity.
No explainer is refitted and no predictive metric is changed.
"""
import argparse
import json
import os
from types import SimpleNamespace
import numpy as np
import pandas as pd
import config as C
from tasks import load_task, all_pairs
from run_main import make_subgraphs
from evaluate import topk_keep
from paths import explanation_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True)
    ap.add_argument('--batch', type=int, required=True)
    args = ap.parse_args()
    task = load_task(args.task)
    pairs, kinds = all_pairs(task)
    subs = make_subgraphs(task, pairs)
    out = os.path.join(C.TABLES_DIR, task.name)
    base = []
    for model in ['sage', 'rgcn', 'hgt']:
        rows = []
        with np.load(os.path.join(out, f'weights_{model}.npz')) as ws:
            for i, (sub, _, local) in enumerate(subs):
                a, b = pairs[:, i]
                ta, tb = task.pair_types
                ets = [et for et in sub.edge_types if sub[et].edge_index.shape[1] > 0]
                sizes = [sub[et].edge_index.shape[1] for et in ets]
                q = SimpleNamespace(data=task.data, sub=sub, a=local[ta][int(a)], b=local[tb][int(b)],
                                    pair_types=task.pair_types, etypes=ets, sizes=sizes, n_edges=sum(sizes))

                def split(v):
                    pieces = np.split(v, np.cumsum(sizes)[:-1])
                    return dict(zip(ets, pieces))

                q.split = split
                if model == 'sage':
                    full = {et: np.ones(n, dtype=bool) for et, n in zip(ets, sizes)}
                    r = explanation_paths(q, full, full, local, task.curated)
                    base.append(dict(pair_id=i, cohort=kinds[i], n_edges=sum(sizes),
                                     connected=r['connected'], conn_len=r['conn_len'],
                                     n_curated_in_sub=r['n_curated_in_sub']))
                for method in ['regex', 'gnnexplainer', 'pgexplainer', 'pagelink',
                               'saliency', 'ig', 'distance', 'random']:
                    weights = split(ws[f'{i}|{method}'])
                    keep, _ = topk_keep(q, weights, .10, np.random.default_rng(i % args.batch))
                    r = explanation_paths(q, keep, weights, local, task.curated)
                    rows.append(dict(model=model, pair_id=i, cohort=kinds[i], method=method,
                                     connected=r['connected'], conn_len=r['conn_len'],
                                     n_paths=r['n_paths'], curated_hit=r['curated_hit'],
                                     n_curated_in_sub=r['n_curated_in_sub'], rel_hist=json.dumps(r['rel_hist'])))
        pd.DataFrame(rows).to_csv(os.path.join(out, f'paths_{model}.csv'), index=False)
    pd.DataFrame(base).to_csv(os.path.join(out, 'subgraph_readout.csv'), index=False)
    print(task.name, 'full subgraph connectivity by cohort:')
    print(pd.DataFrame(base).groupby('cohort')[['connected', 'n_curated_in_sub', 'n_edges']].mean())


if __name__ == '__main__':
    main()
