# -*- coding: utf-8 -*-
"""Check IG completeness using validation queries, never tune on test scores."""
import os
import numpy as np
import pandas as pd
import torch
import config as C
from tasks import load_task
from run_main import make_subgraphs, build_queries
from explainers import restore_model, ig_explain
from batch import QueryBatch


def main():
    rows = []
    dev = torch.device(C.DEVICE)
    for name in ['atmad', 'dtinet']:
        task = load_task(name)
        pairs = task.val_pairs[:, :12]
        subs = make_subgraphs(task, pairs)
        for model_name in ['sage', 'rgcn', 'hgt']:
            model = restore_model(model_name, task.data,
                                  torch.load(task.ckpt(model_name), map_location='cpu', weights_only=False), dev)
            qb = QueryBatch(build_queries(task, model, pairs, subs, dev))
            with torch.no_grad():
                reference = ((2 * qb.y_hat - 1) *
                             (qb.logits() - qb.logits(torch.zeros(qb.N, device=dev)))).cpu().numpy()
            for scheme, steps in [('uniform', 32), ('log', 128), ('log', 256)]:
                weights = ig_explain(qb, steps=steps, integration=scheme)
                for i, w in enumerate(weights):
                    total = sum(v.sum() for v in w.values())
                    rows.append(dict(task=name, model=model_name, pair_id=i, scheme=scheme, steps=steps,
                                     reference=float(reference[i]), attr_sum=float(total),
                                     abs_error=float(abs(total-reference[i])),
                                     rel_error=float(abs(total-reference[i])/max(abs(reference[i]), 1e-3))))
                print(name, model_name, scheme, steps, flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(os.path.join(C.TABLES_DIR, 'ig_completeness.csv'), index=False)
    print(frame.groupby(['task', 'scheme', 'steps'])[['abs_error', 'rel_error']].median())


if __name__ == '__main__':
    main()
