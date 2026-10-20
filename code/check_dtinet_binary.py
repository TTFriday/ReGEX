# -*- coding: utf-8 -*-
"""Additional DTINet binary-mask checks including all-zero, all-one and
random retained-edge masks; finite agreement is asserted, not presumed.
"""
import numpy as np
import torch
import config as C
from tasks import load_task
from run_main import make_subgraphs, build_queries
from explainers import restore_model
from masked_forward import hard_logit
from batch import QueryBatch


def main():
    task=load_task('dtinet')
    pairs=task.val_pairs[:,:4]
    subs=make_subgraphs(task,pairs)
    rng=np.random.default_rng(25)
    for name in ['sage','rgcn','hgt']:
        model=restore_model(name,task.data,torch.load(task.ckpt(name),map_location='cpu',weights_only=False),C.DEVICE)
        qs=build_queries(task,model,pairs,subs,C.DEVICE)
        qb=QueryBatch(qs)
        worst=0.0
        for fraction in [0.0,1.0,0.1,0.5]:
            flat=(rng.random(qb.N)<fraction).astype(float)
            with torch.no_grad():
                batch=qb.logits(torch.tensor(flat,dtype=torch.float32,device=qb.dev)).cpu().numpy()
            weights=qb.split(flat)
            refs=np.array([hard_logit(model,q.sub,task.data,q.a,q.b,
                                     {et:values.astype(bool) for et,values in w.items()},task.pair_types)
                           for q,w in zip(qs,weights)])
            assert np.isfinite(batch).all()
            assert np.allclose(batch,refs,atol=2e-5,rtol=2e-5),(name,batch,refs)
            worst=max(worst,float(np.max(abs(batch-refs))))
        print(name,'DTINet maximum absolute discrepancy:',worst,flush=True)


if __name__=='__main__':
    main()
