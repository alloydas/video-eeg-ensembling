import os
import json, sys, numpy as np
sys.path.insert(0, './fuse')
from fuse_lib import *

B = 4000
rng = np.random.default_rng(20260922)
D = {t: load(t) for t in ('bin', 'g3', 'g5')}
Ebin = D['bin']['E'].mean(0)
OUT = {}

def build(task, d):
    Vens = d['V'].mean(0); Eens = d['E'].mean(0); y = d['y']; K = d['K']
    esing = {n: metrics(y, d['E'][i], K)['F1'] for i, n in enumerate(d['enames'])}
    ebest = max(esing, key=esing.get)
    v = {}
    v['VIDEO ensemble'] = Vens
    v['EEG ensemble'] = Eens
    v['EEG best-single*'] = d['E'][list(d['enames']).index(ebest)]
    v['avg 0.5/0.5'] = 0.5 * Vens + 0.5 * Eens
    v['geometric mean'] = geometric(Vens, Eens)
    v['rank average'] = rank_avg(Vens, Eens)
    v['max-confidence gate'] = maxconf(Vens, Eens)
    Xcat = np.hstack([Vens, Eens])
    v['stack LOAO LR'] = stack_loao(Xcat, y, d['subj'], K)
    v['stack LOAO LR (bal)'] = stack_loao(Xcat, y, d['subj'], K, balanced=True)
    v['stack LOAO LR video-only'] = stack_loao(Vens, y, d['subj'], K)
    ws = np.round(np.arange(0, 1.0001, 0.05), 2)
    bw = max(ws, key=lambda w: metrics(y, w * Vens + (1 - w) * Eens, K)['F1'])
    v['weighted w=%.2f (oracle)' % bw] = bw * Vens + (1 - bw) * Eens
    if K > 2:
        p0 = Ebin[:, 0]
        cond = Vens[:, 1:] / np.clip(Vens[:, 1:].sum(1, keepdims=True), 1e-12, None)
        v['EEG gate + video severity'] = np.hstack([p0[:, None], (1 - p0)[:, None] * cond])
    return v

for task in ('bin', 'g3', 'g5'):
    d = D[task]; y = d['y']; K = d['K']
    v = build(task, d)
    preds = {n: S.argmax(1) for n, S in v.items()}
    sev = SEVERE[task]
    animals = np.unique(d['subj'])
    idx_by = {a: np.where(d['subj'] == a)[0] for a in animals}
    names = list(v)
    # observed
    obs = {}
    for n in names:
        f1, rec, sup = fast_f1_rec(y, preds[n], K)
        sr = float(sum(rec[c] * sup[c] for c in sev) / max(1e-9, sum(sup[c] for c in sev)))
        obs[n] = dict(F1=f1, sev=sr)
    # bootstrap
    bF1 = {n: np.empty(B) for n in names}
    bSV = {n: np.empty(B) for n in names}
    nan_sev = 0
    for b in range(B):
        pick = rng.choice(animals, size=animals.size, replace=True)
        ii = np.concatenate([idx_by[a] for a in pick])
        yb = y[ii]
        nsev = sum((yb == c).sum() for c in sev)
        if nsev == 0:
            nan_sev += 1
        for n in names:
            f1, rec, sup = fast_f1_rec(yb, preds[n][ii], K)
            bF1[n][b] = f1
            bSV[n][b] = (sum(rec[c] * sup[c] for c in sev) / nsev) if nsev else np.nan
    base = 'VIDEO ensemble'
    rows = []
    for n in names:
        dF = bF1[n] - bF1[base]
        dS = bSV[n] - bSV[base]
        dS = dS[~np.isnan(dS)]
        rows.append(dict(variant=n, F1=obs[n]['F1'], sev=obs[n]['sev'],
                         dF1=obs[n]['F1'] - obs[base]['F1'],
                         dF1_lo=float(np.percentile(dF, 2.5)), dF1_hi=float(np.percentile(dF, 97.5)),
                         dF1_pgt=float((dF > 0).mean()),
                         dsev=obs[n]['sev'] - obs[base]['sev'],
                         dsev_lo=float(np.percentile(dS, 2.5)), dsev_hi=float(np.percentile(dS, 97.5)),
                         dsev_pgt=float((dS > 0).mean())))
    OUT[task] = dict(rows=rows, B=B, n_animals=int(animals.size), nan_sev=nan_sev)
    print('\n' + '=' * 118)
    print('TASK %s  animal-clustered bootstrap B=%d  animals=%d   baseline = VIDEO ensemble'
          % (task, B, animals.size))
    print('%-30s %7s %8s %-20s %5s | %7s %8s %-20s %5s'
          % ('variant', 'F1', 'dF1', '95% CI (animal)', 'P>0', 'sevRec', 'dSev', '95% CI (animal)', 'P>0'))
    for r in rows:
        print('%-30s %7.4f %+8.4f [%+.4f,%+.4f] %5.3f | %7.4f %+8.4f [%+.4f,%+.4f] %5.3f'
              % (r['variant'], r['F1'], r['dF1'], r['dF1_lo'], r['dF1_hi'], r['dF1_pgt'],
                 r['sev'], r['dsev'], r['dsev_lo'], r['dsev_hi'], r['dsev_pgt']))

json.dump(OUT, open('./fuse/boot.json', 'w'), indent=1)
