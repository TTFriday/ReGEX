# -*- coding: utf-8 -*-
"""Measure prediction changes caused by bounded-subgraph extraction.
AtMAD uses saved full-graph embeddings from the original trained checkpoint;
DTINet recomputes original-encoder embeddings on the complete training graph.
This diagnostic never selects hyperparameters or alters test explanations.
"""
import os
import pandas as pd
import torch
import config as C
from tasks import load_task
from explainers import restore_model


def main():
    for name in ['atmad','dtinet']:
        task=load_task(name)
        path=os.path.join(C.TABLES_DIR,name,'query_context.csv')
        context=pd.read_csv(path)
        out=[]
        for model_name in ['sage','rgcn','hgt']:
            checkpoint=torch.load(task.ckpt(model_name),map_location='cpu',weights_only=False)
            if name=='atmad':
                emb_a,emb_b=checkpoint['gene_emb'],checkpoint['pheno_emb']
            else:
                model=restore_model(model_name,task.data,checkpoint,'cpu')
                with torch.no_grad():
                    embeddings=model.encode(task.data,{nt:task.data[nt].x for nt in task.data.node_types})
                    emb_a,emb_b=embeddings['drug'],embeddings['protein']
            frame=context[context.model==model_name].copy()
            a=torch.tensor(frame.a_idx.to_numpy(),dtype=torch.long)
            b=torch.tensor(frame.b_idx.to_numpy(),dtype=torch.long)
            frame['full_logit']=(emb_a[a]*emb_b[b]).sum(-1).numpy()
            frame['full_label']=(frame.full_logit>=0).astype(int)
            frame['label_agreement']=frame.full_label==frame.predicted_label
            out.append(frame)
        result=pd.concat(out,ignore_index=True)
        result.to_csv(os.path.join(C.TABLES_DIR,name,'context_shift.csv'),index=False)
        print(name,flush=True)
        print(result.groupby(['model','cohort'])[['full_label','predicted_label','label_agreement']].mean(),flush=True)


if __name__=='__main__':
    main()
