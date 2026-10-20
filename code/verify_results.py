# -*- coding: utf-8 -*-
"""Validate final result cardinality, finite values and provenance.

Run after all final analyses and corrected baselines are complete. JSON
provenance records hashes of current source, outputs and checkpoints;
archived debugging runs are not treated as final experiments.
"""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch_geometric
import config as C


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def finite(frame, columns):
    assert np.isfinite(frame[columns].to_numpy(dtype=float)).all(), f'Nonfinite values in {columns}'


def main():
    checks = []
    hashes = {}
    for task, n in [('atmad', 515), ('dtinet', 192)]:
        d = Path(C.TABLES_DIR) / task
        for model in ['sage', 'rgcn', 'hgt']:
            mainf = pd.read_csv(d / f'main_{model}.csv')
            assert len(mainf) == n * 8 * 4
            assert not mainf.duplicated(['pair_id', 'method', 'frac']).any()
            assert mainf.groupby(['pair_id', 'frac']).method.nunique().eq(8).all()
            finite(mainf, ['fid_plus', 'fid_minus', 'charact', 'p_full', 'p_keep', 'p_drop'])
            assert mainf.charact.between(0, 1).all()
            p = pd.read_csv(d / f'paths_{model}.csv')
            assert len(p) == n * 8
            assert not p.duplicated(['pair_id', 'method']).any()
            with np.load(d / f'weights_{model}.npz') as w:
                assert len(w.files) == n * 8
                for key in w.files:
                    assert np.isfinite(w[key]).all(), key
            checks.append(f'{task}/{model}: complete main and path results, finite masks')
        expected = {'ablation': 7200, 'sensitivity': 8400, 'stability': 600,
                    'robustness': 6000, 'convergence': 30000, 'sanity': 3600}
        for analysis, count in expected.items():
            f = pd.read_csv(d / f'{analysis}.csv')
            assert len(f) == count, (task, analysis, len(f), count)
            if analysis == 'ablation':
                finite(f, ['charact', 'fid_plus', 'fid_minus'])
                assert not f.duplicated(['model', 'pair_id', 'variant', 'frac']).any()
            elif analysis == 'sanity':
                # A constant attribution can have an undefined Spearman
                # correlation. Jaccard must nevertheless be well-defined.
                finite(f, ['jaccard'])
                hgt = f[f.model.eq('hgt')]
                assert hgt.spearman.notna().any(), 'All HGT sanity correlations undefined'
            elif analysis in ['robustness', 'stability']:
                finite(f, ['jaccard'])
            elif analysis == 'convergence':
                finite(f, ['loss', 'l_fid', 'l_nec', 'masked_p_y'])
            else:
                finite(f, ['charact_mean', 'charact_10'])
            checks.append(f'{task}/{analysis}: {count} rows verified')
        for path in d.glob('*'):
            if path.suffix in ['.csv', '.json', '.npz']:
                hashes[str(path.relative_to(Path(C.PROJECT_DIR)))] = sha(path)
    for path in Path(C.CODE_DIR).glob('*.py'):
        hashes[str(path.relative_to(Path(C.PROJECT_DIR)))] = sha(path)
    for name in ['sage', 'rgcn', 'hgt']:
        for prefix in [C.P2_MODELS_DIR, C.DTI_MODELS_DIR]:
            path = Path(prefix) / f'{name}_s42.pt'
            hashes[str(path)] = sha(path)
    for name in ['atmad_hetero.pt', 'splits.pt', 'node_names.json']:
        path = Path(C.P2_DATA_DIR) / name
        hashes[str(path)] = sha(path)
    for path in (Path(C.PROJECT_DIR) / 'data').rglob('*'):
        if path.is_file() and path.suffix in ['.txt', '.pt', '.csv']:
            hashes[str(path.relative_to(Path(C.PROJECT_DIR)))] = sha(path)
    for path in [Path(C.TABLES_DIR) / 'ig_completeness.csv', Path(C.CODE_DIR) / 'requirements.txt']:
        hashes[str(path.relative_to(Path(C.PROJECT_DIR)))] = sha(path)
    record = {'torch': torch.__version__, 'pyg': torch_geometric.__version__,
              'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(0),
              'checks': checks, 'sha256': hashes,
              'notes': ['Final masks use clamp relaxation; ratio exploratory runs are archived.',
                        'PG training batch is 30; evaluation batches differ by benchmark.',
                        'IG uses 256 nonuniform midpoints; validation completeness is separately saved.',
                        'Robustness and sanity were refreshed following precision and singleton fixes.',
                        'Source hashes identify final reproducible implementation, not historical run-time snapshots.']}
    target = Path(C.RESULTS_DIR) / 'verification_manifest.json'
    target.write_text(json.dumps(record, indent=2), encoding='utf-8')
    print('\n'.join(checks))
    print('Verification manifest:', target)


if __name__ == '__main__':
    main()
