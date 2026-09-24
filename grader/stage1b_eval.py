"""Stage 1b evaluation, exactly as pre-registered in grader/stage1b_prereg.md.

Arms (fixed X3D, dual heads, seeds 4-6): R0 control@12, R1 augment@12, R2 logit_bound 1.0 @12,
R3 control read at the pre-registered epoch 3. Primary: 3-seed mean within-session severe-vs-mild
AUROC of the g3 head vs R0@12. Adoption: primary gain >= +0.02 AND g3 macro-F1 not lower than
R0@12 by more than 0.01. Also: per-class recall with counts, g5 S5 recall, and the effect of
swapping each arm (3 seeds, same epoch rule) into the stored top-3x5 ensemble.

Usage (from the video-eeg-ensembling repo, any cwd; EEG_ROOT env var, default
/work/mech-ai-scratch/alloy/EEG; DHLIB_DIR env var, default grader/, where dhlib.py is vendored):
  python grader/stage1b_eval.py [--out <json>]
The script chdirs to EEG_ROOT: the run directories (output/ttg_stage1b, output/v3_vidseeds) and a
relative --out are resolved against EEG_ROOT, and --out must resolve under
$EEG_ROOT/output/ttg_*. The default --out is the stored result
output/ttg_stage1b/stage1b_results.json, which a run overwrites; pass --out elsewhere to
re-derive it without touching it.
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EEG_ROOT = os.path.realpath(os.environ.get('EEG_ROOT', '/work/mech-ai-scratch/alloy/EEG'))
os.environ.setdefault('EEG_ROOT', EEG_ROOT)          # dhlib reads EEG_ROOT at import
DH_DIR = os.path.realpath(os.environ.get('DHLIB_DIR', HERE))
if DH_DIR not in sys.path:
    sys.path.insert(0, DH_DIR)
import dhlib as D

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--out', default='output/ttg_stage1b/stage1b_results.json',
                help='result JSON (relative: to EEG_ROOT)')
A = ap.parse_args()
os.chdir(EEG_ROOT)                                    # every relative path below is EEG_ROOT's
A.out = os.path.realpath(A.out)                       # relative: EEG_ROOT's (after the chdir)
if not A.out.startswith(os.path.join(EEG_ROOT, 'output', 'ttg_')):
    raise SystemExit(f'--out must resolve under {EEG_ROOT}/output/ttg_*, got {A.out}')
os.makedirs(os.path.dirname(A.out), exist_ok=True)

B = 'output/ttg_stage1b'
ARMS = {'R0 control @12': ('ctrl', 12), 'R1 augment @12': ('aug', 12),
        'R2 bounded logits @12': ('bound', 12), 'R3 control @3 (pre-registered)': ('ctrl', 3)}
REPS = 2000

def load(arm, ep, task, keys):
    pos = {k: i for i, k in enumerate(keys)}; P = []
    for s in (4, 5, 6):
        z = np.load(f'{B}/x3dfix_{arm}_s{s}/val_ep{ep:02d}.npz', allow_pickle=True)
        k = np.array([D.key_of(p) for p in z['path']]); p = z[f'probs_{task}'].astype(np.float64)
        a = np.empty_like(p); a[np.array([pos[x] for x in k])] = p; P.append(a)
    return np.array(P)

out = {}
for task in ('g3', 'g5'):
    keys, y, y5 = D.reference_val(task)
    an, se = D.animal_session(keys); ug = np.unique(an); W = D.boot_weights(len(ug), REPS, seed=0)
    is_sev = np.isin(y, D.SEV[task]); is_mild = np.isin(y, D.MILD[task])
    ref_raw_y = np.load(f'{D.EEG_ROOT}/output/v3_vidseeds/mvit_{task}_s1/val_preds.npz')['y']
    dirs = [f'output/v3_vidseeds/{b}_{task}_s{s}' for b in ('mvit', 'slowfast') for s in (1, 2, 3, 5)] + [f'vid_{b}_{task}' for b in ('mvit', 'slowfast')]
    Pms, _, _ = D.load_members(dirs, keys, ref_raw_y, y)
    dirx = [f'output/v3_vidseeds/x3d_{task}_s{s}' for s in (1, 2, 3, 5)] + [f'vid_x3d_{task}']
    Px, _, lg = D.load_members(dirx, keys, ref_raw_y, y)
    base = np.concatenate([Pms, Px]).mean(0)
    base_rep = D.full_report(y, base.argmax(1), task)
    Cb, _ = D.per_group_cm(y, base.argmax(1), an, D.K[task], ug); bb = D.boot_cm_metrics(Cb, W, task)
    res = {'top3x5 stored': {'macro_f1': base_rep['macro_f1']}}
    for name, (arm, ep) in ARMS.items():
        P = load(arm, ep, task, keys)
        singles = []
        for p in P:
            rep = D.full_report(y, p.argmax(1), task)
            s = p[:, D.SEV[task]].sum(1)
            rep['within'] = float(D.within_session_auc(s, is_sev, is_mild, se)[0])
            singles.append(rep)
        E = P.mean(0); erep = D.full_report(y, E.argmax(1), task)
        se_ = E[:, D.SEV[task]].sum(1); erep['within'] = float(D.within_session_auc(se_, is_sev, is_mild, se)[0])
        sw = np.concatenate([Pms, P]).mean(0); swrep = D.full_report(y, sw.argmax(1), task)
        Cs, _ = D.per_group_cm(y, sw.argmax(1), an, D.K[task], ug); bs = D.boot_cm_metrics(Cs, W, task)
        d = bs['macro_f1'] - bb['macro_f1']
        res[name] = dict(
            single_macro_f1=[r['macro_f1'] for r in singles], single_within=[r['within'] for r in singles],
            mean_macro_f1=float(np.mean([r['macro_f1'] for r in singles])),
            mean_within=float(np.mean([r['within'] for r in singles])),
            mean_recall={c: float(np.mean([r['per_class'][c]['recall'] for r in singles])) for c in singles[0]['per_class']},
            counts={c: singles[0]['per_class'][c]['n'] for c in singles[0]['per_class']},
            ens3=dict(macro_f1=erep['macro_f1'], within=erep['within'], recall={c: v['recall'] for c, v in erep['per_class'].items()}),
            swap_into_top3=dict(macro_f1=swrep['macro_f1'], d=float(swrep['macro_f1'] - base_rep['macro_f1']),
                                ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                                recall={c: v['recall'] for c, v in swrep['per_class'].items()}))
    r0 = res['R0 control @12']
    for name in ARMS:
        if name == 'R0 control @12': continue
        a = res[name]
        a['primary_gain_within'] = a['mean_within'] - r0['mean_within']
        a['g3_f1_change'] = a['mean_macro_f1'] - r0['mean_macro_f1']
        a['adopt'] = bool(task == 'g3' and a['primary_gain_within'] >= 0.02 and a['g3_f1_change'] >= -0.01) if task == 'g3' else None
    out[task] = res
    print(f'\n== {task}  (stored top3x5 {base_rep["macro_f1"]:.4f})')
    for name, a in res.items():
        if name == 'top3x5 stored': continue
        rec = ' '.join(f'{v:.3f}/{a["counts"][c]}' for c, v in a['mean_recall'].items())
        extra = f"  gain_within {a.get('primary_gain_within', 0):+.4f} dF1 {a.get('g3_f1_change', 0):+.4f} adopt={a.get('adopt')}" if name != 'R0 control @12' else ''
        print(f"{name:32s} mF1 {a['mean_macro_f1']:.4f} {['%.3f'%x for x in a['single_macro_f1']]}  within {a['mean_within']:.4f} {['%.3f'%x for x in a['single_within']]}  recall {rec}{extra}")
        print(f"{'':32s} 3-seed ens mF1 {a['ens3']['macro_f1']:.4f} within {a['ens3']['within']:.4f} | swap into top3: {a['swap_into_top3']['macro_f1']:.4f} d {a['swap_into_top3']['d']:+.4f} [{a['swap_into_top3']['ci'][0]:+.4f},{a['swap_into_top3']['ci'][1]:+.4f}]")
json.dump(out, open(A.out, 'w'), indent=1)
print('wrote', A.out)
