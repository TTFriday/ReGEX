# -*- coding: utf-8 -*-
"""Sensitivity analysis restricted to queries whose original full-graph
and bounded-subgraph class predictions agree. All queries remain in the
primary descriptive table; this subset cannot erase a context-shift problem.
"""
import os
import pandas as pd
import config as C
from analyze import load_main, main_table


def main():
    for task in ['atmad','dtinet']:
        frame,_=load_main(task)
        shift=pd.read_csv(os.path.join(C.TABLES_DIR,task,'context_shift.csv'))
        merged=frame.merge(shift[['model','pair_id','cohort','full_label','predicted_label','label_agreement']],
                           on=['model','pair_id','cohort'],validate='many_to_one')
        selected=merged[merged.label_agreement & merged.cohort.eq('test')].copy()
        out=selected[selected.frac.eq(.1)].groupby(['model','method']).agg(
            charact=('charact','mean'),n=('pair_id','nunique')).reset_index()
        out.to_csv(os.path.join(C.TABLES_DIR,task,'summary_context_agreement.csv'),index=False)
        print(task,flush=True)
        print(out.pivot(index='method',columns='model',values='charact').round(3),flush=True)
        print('concordant query counts:',out.groupby('model').n.first().to_dict(),flush=True)


if __name__=='__main__':
    main()
