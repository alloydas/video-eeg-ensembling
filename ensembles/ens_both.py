import os
"""Seed / architecture ensembling for BOTH modalities, from predictions already on disk.

Video : output/v3_vidseeds/<arch>_<task>_s<seed>/val_preds.npz        (5,289 clips)
EEG   : output/v3_bestcfg/<arch>_<task>_s<seed>/val_clip_preds.npz    (5,319 clips)

Three things this has to get right, each of which silently corrupts the result:
  * Row order differs between runs -> align on the stored `path`. Averaging by row
    index mixes probability vectors across clips and INFLATES the score.
  * Some runs store softmax(softmax(.)) -> detect and invert (preds_io.unsquash).
  * Runs with no `path` are admitted only if their raw label sequence matches the
    reference's raw sequence, i.e. they are already in reference order.
Metrics match the repo's tables: macro P/R/F1, binary or OvR-macro AUROC, multiclass MCC.
"""
import glob, os, re, sys
import numpy as np
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, matthews_corrcoef)

import os
_DEFAULT_ROOT = os.environ.get('EEG_ROOT', '/work/mech-ai-scratch/alloy/EEG')
ROOT = os.environ.get('EEG_ROOT', _DEFAULT_ROOT)  # override for another machine
SPECS = {
    'VIDEO': dict(root=f'{ROOT}/output/v3_vidseeds', fname='val_preds.npz'),
    'EEG':   dict(root=f'{ROOT}/output/v3_bestcfg',  fname='val_clip_preds.npz'),
}
_TOL = 1e-3

def is_double_softmax(P):
    k = P.shape[1]; e = np.e
    return abs(P.min() - 1.0/(k-1+e)) < _TOL and abs(P.max() - e/(k-1+e)) < _TOL

def unsquash(P):
    k = P.shape[1]
    lp = np.log(np.clip(P, 1e-12, None))
    q = np.clip(lp + (1.0 - lp.sum(1, keepdims=True))/k, 0.0, None)
    return q / q.sum(1, keepdims=True)

def metrics(y, P):
    pred = P.argmax(1)
    auc = (roc_auc_score(y, P[:, 1]) if P.shape[1] == 2
           else roc_auc_score(y, P, multi_class='ovr', average='macro'))
    return dict(P=precision_score(y, pred, average='macro', zero_division=0),
                R=recall_score(y, pred, average='macro', zero_division=0),
                F1=f1_score(y, pred, average='macro', zero_division=0),
                AUC=auc, MCC=matthews_corrcoef(y, pred))

def load(spec, task):
    raw = {}
    for d in sorted(glob.glob(os.path.join(spec['root'], '*_%s_s*' % task))):
        f = os.path.join(d, spec['fname'])
        if not os.path.exists(f):
            continue
        z = np.load(f, allow_pickle=True)
        P = z['probs'].astype(np.float64)
        sq = is_double_softmax(P)
        raw[os.path.basename(d)] = dict(y=z['y'].astype(int), P=unsquash(P) if sq else P,
                                        path=z['path'].astype(str) if 'path' in z.files else None,
                                        sq=sq)
    if not raw:
        return {}, [], None, 0
    ref = next((n for n in sorted(raw) if raw[n]['path'] is not None), None)
    if ref is None:
        return {}, [], None, 0
    order = np.argsort(raw[ref]['path'])
    rpath, ry = raw[ref]['path'][order], raw[ref]['y'][order]
    rrawy = raw[ref]['y']
    out, dropped, nsq = {}, [], 0
    for n, r in raw.items():
        nsq += int(r['sq'])
        if r['path'] is not None:
            o = np.argsort(r['path'])
            if r['path'][o].shape != rpath.shape or not np.array_equal(r['path'][o], rpath) \
               or not np.array_equal(r['y'][o], ry):
                dropped.append((n, 'different clip list')); continue
            out[n] = r['P'][o]
        elif r['y'].shape == rrawy.shape and np.array_equal(r['y'], rrawy):
            out[n] = r['P'][order]
        else:
            dropped.append((n, 'no path and label order differs'))
    return out, dropped, ry, nsq

for mod, spec in SPECS.items():
    print('\n' + '=' * 96)
    print('### %s   %s' % (mod, spec['root'].replace(ROOT + '/', '')))
    for task in ('bin', 'g3', 'g5'):
        runs, dropped, y, nsq = load(spec, task)
        if not runs:
            print('\n-- task=%s : no usable runs' % task); continue
        names = sorted(runs)
        print('\n-- task=%-4s members=%-3d clips=%-5d classes=%d  dropped=%d  double-softmax repaired=%d'
              % (task, len(names), y.size, int(y.max()) + 1, len(dropped), nsq))
        for n, why in dropped:
            print('     DROPPED %-22s %s' % (n, why))
        singles = {n: metrics(y, runs[n]) for n in names}
        f1 = np.array([singles[n]['F1'] for n in names])
        best = max(names, key=lambda n: singles[n]['F1'])
        ens = metrics(y, np.mean([runs[n] for n in names], axis=0))
        print('     single runs  F1 mean %.4f sd %.4f  min %.4f  max %.4f  (%s)'
              % (f1.mean(), f1.std(ddof=0), f1.min(), f1.max(), best))
        print('     BEST SINGLE  P %.4f  R %.4f  F1 %.4f  AUC %.4f  MCC %.4f'
              % tuple(singles[best][k] for k in ('P', 'R', 'F1', 'AUC', 'MCC')))
        print('     ENSEMBLE     P %.4f  R %.4f  F1 %.4f  AUC %.4f  MCC %.4f   (F1 %+.4f vs best, %+.4f vs mean)'
              % (ens['P'], ens['R'], ens['F1'], ens['AUC'], ens['MCC'],
                 ens['F1'] - f1.max(), ens['F1'] - f1.mean()))
        arch = {}
        for n in names:
            arch.setdefault(re.sub(r'_%s_s\d+$' % task, '', n), []).append(n)
        print('     %-12s %3s %10s %10s %10s %9s' % ('backbone', 'n', 'seed-mean', 'seed-best', 'seed-ens', 'gain'))
        rows = []
        for a, ms in arch.items():
            fa = metrics(y, np.mean([runs[m] for m in ms], axis=0))['F1']
            sm = np.array([singles[m]['F1'] for m in ms])
            rows.append((a, len(ms), sm.mean(), sm.max(), fa, fa - sm.max()))
        for a, n, mn, mx, fa, g in sorted(rows, key=lambda r: -r[4]):
            print('     %-12s %3d %10.4f %10.4f %10.4f %+9.4f' % (a, n, mn, mx, fa, g))
