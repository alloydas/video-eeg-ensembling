import os
"""Controls: is the severe-recall gain from ADDING EEG, or just from reweighting the prior?
Plus within-session severity discrimination and gate diagnostics."""
import json, sys, itertools, numpy as np
sys.path.insert(0, './fuse')
from fuse_lib import *

D = {t: load(t) for t in ('bin', 'g3', 'g5')}
Ebin = D['bin']['E'].mean(0)
rng = np.random.default_rng(7)
B = 4000
OUT = {}

def within_session_auc(score, y, sess, pos_mask, neg_mask):
    """P(score(pos) > score(neg)) over pairs drawn from the SAME session only."""
    num = den = 0.0; nsess = 0; npair = 0
    for s in np.unique(sess):
        m = sess == s
        p = score[m & pos_mask]; n = score[m & neg_mask]
        if p.size == 0 or n.size == 0:
            continue
        nsess += 1; npair += p.size * n.size
        d = p[:, None] - n[None, :]
        num += (d > 0).sum() + 0.5 * (d == 0).sum(); den += d.size
    return (num / den if den else float('nan')), nsess, npair

def pooled_auc(score, y, pos_mask, neg_mask):
    p = score[pos_mask]; n = score[neg_mask]
    d = p[:, None] - n[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)

for task in ('bin', 'g3', 'g5'):
    d = D[task]; y = d['y']; K = d['K']; sess = d['sess']; subj = d['subj']
    Vens = d['V'].mean(0); Eens = d['E'].mean(0)
    Xcat = np.hstack([Vens, Eens])

    # ---- prior-reweighting controls, video ONLY (no EEG anywhere) --------
    ctrl = {}
    ctrl['VIDEO ensemble'] = Vens
    ctrl['VIDEO rank (cols)'] = colrank(Vens)                     # prior-free video
    ctrl['VIDEO / prior'] = Vens / np.bincount(y, minlength=K)     # prior-divided video
    ctrl['VIDEO stack LOAO bal'] = stack_loao(Vens, y, subj, K, balanced=True)
    ctrl['FUSE rank average'] = rank_avg(Vens, Eens)
    ctrl['FUSE stack LOAO bal'] = stack_loao(Xcat, y, subj, K, balanced=True)
    ctrl['FUSE avg 0.5/0.5'] = 0.5 * Vens + 0.5 * Eens
    ctrl['FUSE maxconf gate'] = maxconf(Vens, Eens)
    print('\n' + '=' * 104)
    print('TASK %s  --- prior-reweighting controls (does EEG add, or is it just reweighting?)' % task)
    print('%-24s %7s %7s %7s %7s  %s' % ('variant', 'F1', 'MCC', 'AUC', 'sevRec', 'per-class recall'))
    crows = []
    for n, S in ctrl.items():
        m = metrics(y, S, K)
        sv = float(sum(m['cm'][c][c] for c in SEVERE[task]) / sum(m['sup'][c] for c in SEVERE[task]))
        crows.append(dict(variant=n, **{k: m[k] for k in ('P', 'R', 'F1', 'AUC', 'MCC', 'rec', 'sup')}, sev=sv))
        print('%-24s %7.4f %7.4f %7.4f %7.4f  %s' % (n, m['F1'], m['MCC'], m['AUC'], sv,
              ' '.join('%.3f' % r for r in m['rec'])))

    # ---- within-session vs pooled severity discrimination ---------------
    wrows = []
    if K > 2:
        ict = y > 0
        sevm = np.isin(y, SEVERE[task]) & ict
        mildm = ict & ~sevm
        def sevscore(S):
            return S[:, SEVERE[task]].sum(1) if K > 3 else S[:, SEVERE[task][0]]
        cand = dict(VIDEO=Vens, EEG=Eens, AVG=0.5 * Vens + 0.5 * Eens,
                    GEO=geometric(Vens, Eens), RANK=rank_avg(Vens, Eens),
                    MAXCONF=maxconf(Vens, Eens),
                    STACK=stack_loao(Xcat, y, subj, K),
                    STACK_V=stack_loao(Vens, y, subj, K))
        print('  severity discrimination: severe(%d) vs mild(%d) ictal clips' % (sevm.sum(), mildm.sum()))
        print('  %-10s %9s %9s %9s' % ('scorer', 'pooled', 'within-s', 'delta'))
        for n, S in cand.items():
            sc = sevscore(S)
            pa = pooled_auc(sc, y, sevm, mildm)
            wa, ns, npair = within_session_auc(sc, y, sess, sevm, mildm)
            # animal bootstrap on the within-session AUC
            bs = []
            an = np.unique(subj)
            for b in range(600):
                pick = rng.choice(an, an.size, replace=True)
                ii = np.concatenate([np.where(subj == a)[0] for a in pick])
                # rebuild session ids unique per draw copy to avoid cross-copy pairing
                tag = np.concatenate([np.full((subj == a).sum(), '%s#%d' % (a, j))
                                      for j, a in enumerate(pick)])
                ss = np.char.add(np.char.add(sess[ii], '@'), tag)
                v, _, _ = within_session_auc(sc[ii], y[ii], ss, sevm[ii], mildm[ii])
                if not np.isnan(v): bs.append(v)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            wrows.append(dict(scorer=n, pooled=pa, within=wa, lo=float(lo), hi=float(hi),
                              n_sess=ns, n_pairs=npair))
            print('  %-10s %9.4f %9.4f %+9.4f   [%.3f,%.3f] sessions=%d pairs=%d'
                  % (n, pa, wa, wa - pa, lo, hi, ns, npair))
    else:
        # detection: seizure vs non-seizure, within session
        cand = dict(VIDEO=Vens, EEG=Eens, AVG=0.5 * Vens + 0.5 * Eens,
                    MAXCONF=maxconf(Vens, Eens))
        print('  detection AUROC, seizure(%d) vs non(%d)' % ((y == 1).sum(), (y == 0).sum()))
        for n, S in cand.items():
            sc = S[:, 1]
            pa = pooled_auc(sc, y, y == 1, y == 0)
            wa, ns, npair = within_session_auc(sc, y, sess, y == 1, y == 0)
            wrows.append(dict(scorer=n, pooled=pa, within=wa, n_sess=ns, n_pairs=npair))
            print('  %-10s pooled %.4f  within-session %.4f  (sessions=%d pairs=%d)' % (n, pa, wa, ns, npair))

    # ---- gate diagnostics ------------------------------------------------
    pickV = Vens.max(1) >= Eens.max(1)
    agree = (Vens.argmax(1) == Eens.argmax(1))
    OUT[task] = dict(controls=crows, within=wrows,
                     gate_pick_video_frac=float(pickV.mean()),
                     gate_pick_video_n=int(pickV.sum()),
                     modality_agree_frac=float(agree.mean()),
                     v_conf_mean=float(Vens.max(1).mean()), e_conf_mean=float(Eens.max(1).mean()))
    print('  max-conf gate picks VIDEO on %d/%d clips (%.1f%%); modalities agree on argmax %.1f%%; '
          'mean max-prob video %.3f eeg %.3f'
          % (pickV.sum(), y.size, 100 * pickV.mean(), 100 * agree.mean(),
             Vens.max(1).mean(), Eens.max(1).mean()))

json.dump(OUT, open('./fuse/ctrl.json', 'w'), indent=1)
