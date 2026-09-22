import os
import json, sys, numpy as np
sys.path.insert(0, './fuse')
from fuse_lib import *

OUT = {}
D = {t: load(t) for t in ('bin', 'g3', 'g5')}
Ebin = D['bin']['E'].mean(0)          # EEG binary ensemble, reused as the detection gate

for task in ('bin', 'g3', 'g5'):
    d = D[task]; y = d['y']; K = d['K']
    Vens = d['V'].mean(0)
    Eens = d['E'].mean(0)
    # --- baselines ---------------------------------------------------------
    vsing = {n: metrics(y, d['V'][i], K)['F1'] for i, n in enumerate(d['vnames'])}
    esing = {n: metrics(y, d['E'][i], K)['F1'] for i, n in enumerate(d['enames'])}
    vbest = max(vsing, key=vsing.get); ebest = max(esing, key=esing.get)
    Vbest = d['V'][list(d['vnames']).index(vbest)]
    Ebest = d['E'][list(d['enames']).index(ebest)]

    var = {}
    var['VIDEO ensemble (40)'] = Vens
    var['EEG ensemble (35)'] = Eens
    var['EEG best-single*'] = Ebest
    var['VIDEO best-single*'] = Vbest
    var['avg 0.5/0.5'] = 0.5 * Vens + 0.5 * Eens
    var['geometric mean'] = geometric(Vens, Eens)
    var['rank average'] = rank_avg(Vens, Eens)
    var['max-confidence gate'] = maxconf(Vens, Eens)
    # weighted sweep
    ws = np.round(np.arange(0, 1.0001, 0.05), 2)
    sweep = []
    for w in ws:
        S = w * Vens + (1 - w) * Eens
        m = metrics(y, S, K)
        sev = float(np.sum([m['cm'][c][c] for c in SEVERE[task]]) /
                    max(1, sum(m['sup'][c] for c in SEVERE[task])))
        sweep.append(dict(w=float(w), F1=m['F1'], P=m['P'], R=m['R'], AUC=m['AUC'],
                          MCC=m['MCC'], sev_rec=sev, rec=m['rec']))
    bw = max(sweep, key=lambda r: r['F1'])['w']
    var['weighted w=%.2f (oracle)' % bw] = bw * Vens + (1 - bw) * Eens
    # stacking
    Xcat = np.hstack([Vens, Eens])
    var['stack LOAO LR'] = stack_loao(Xcat, y, d['subj'], K, balanced=False)
    var['stack LOAO LR (bal)'] = stack_loao(Xcat, y, d['subj'], K, balanced=True)
    var['stack LOAO LR video-only'] = stack_loao(Vens, y, d['subj'], K, balanced=False)
    # hierarchical EEG-detection-gate (the project's standing position)
    if K > 2:
        p0 = Ebin[:, 0]
        cond = Vens[:, 1:] / np.clip(Vens[:, 1:].sum(1, keepdims=True), 1e-12, None)
        G = np.hstack([p0[:, None], (1 - p0)[:, None] * cond])
        var['EEG gate + video severity'] = G

    res = {}
    for nm, S in var.items():
        m = metrics(y, S, K)
        m['sev_rec'] = float(np.sum([m['cm'][c][c] for c in SEVERE[task]]) /
                             max(1, sum(m['sup'][c] for c in SEVERE[task])))
        res[nm] = m
    OUT[task] = dict(K=K, names=NAMES[task], counts=np.bincount(y, minlength=K).tolist(),
                     n=int(y.size), sweep=sweep, best_w=float(bw), res=res,
                     vbest=vbest, ebest=ebest,
                     vsing_F1=dict(sorted(vsing.items(), key=lambda kv: -kv[1])[:5]),
                     esing_F1=dict(sorted(esing.items(), key=lambda kv: -kv[1])[:5]),
                     vsing_sd=float(np.std(list(vsing.values()))),
                     esing_sd=float(np.std(list(esing.values()))))
    print('== %s done' % task, flush=True)

json.dump(OUT, open('./fuse/main.json', 'w'), indent=1)

for task in ('bin', 'g3', 'g5'):
    o = OUT[task]
    print('\n' + '=' * 110)
    print('TASK %s  N=%d  counts=%s  classes=%s' % (task, o['n'], o['counts'], o['names']))
    print('%-30s %6s %6s %6s %6s %6s %7s   %s' % ('variant', 'P', 'R', 'F1', 'AUC', 'MCC', 'sevRec', 'per-class recall'))
    for nm, m in o['res'].items():
        print('%-30s %6.4f %6.4f %6.4f %6.4f %6.4f %7.4f   %s'
              % (nm, m['P'], m['R'], m['F1'], m['AUC'], m['MCC'], m['sev_rec'],
                 ' '.join('%.3f' % r for r in m['rec'])))
    print('  best single video=%s  eeg=%s   (oracle-selected on this set)' % (o['vbest'], o['ebest']))
