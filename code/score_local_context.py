# -*- coding: utf-8 -*-
"""Record each query's local-subgraph logit and predicted label separately
from the fidelity tables, so predicted-class probabilities are unambiguous.
"""
import os
import pandas as pd
import torch
import config as C
from tasks import load_task, all_pairs
from explainers import restore_model
from run_main import make_subgraphs, build_queries


def main():
    for name in ['atmad', 'dtinet']:
        task = load_task(name)
        pairs, cohorts = all_pairs(task)
        subs = make_subgraphs(task, pairs)
        rows = []
        for model_name in ['sage', 'rgcn', 'hgt']:
            model = restore_model(model_name, task.data,
                                  torch.load(task.ckpt(model_name), map_location='cpu', weights_only=False), C.DEVICE)
            queries = build_queries(task, model, pairs, subs, C.DEVICE)
            for i,q in enumerate(queries):
                rows.append(dict(model=model_name, pair_id=i, cohort=cohorts[i],
                                 a_idx=int(pairs[0,i]), b_idx=int(pairs[1,i]),
                                 local_logit=q.logit_full, predicted_label=int(q.y_hat), n_edges=q.n_edges))
        d = pd.DataFrame(rows)
        d.to_csv(os.path.join(C.TABLES_DIR,name,'query_context.csv'),index=False)
        print(name, 'local positive prediction fraction:', d.groupby(['model','cohort']).predicted_label.mean())


if __name__ == '__main__':
    main()
