"""Stage 1 follow-up: what the fixed X3D does to the top-3 x 5 ensemble (mvit, slowfast, x3d).

Swaps the 5 stored (bugged) X3D members for the 3 fixed-X3D Stage-1 runs, or adds them, per task.
Stored members are best-epoch checkpoints chosen on this val set, so every ensemble here is an
upper bound; new members are scored both at their best epoch (like-for-like) and at the
pre-registered last epoch. Deltas vs the stored top-3 x 5 use a paired animal-clustered bootstrap.

Usage (from the video-eeg-ensembling repo, any cwd; EEG_ROOT env var, default
/work/mech-ai-scratch/alloy/EEG; DHLIB_DIR env var, default grader/, where dhlib.py is vendored):
  python grader/stage1_ensemble.py [--out <json>]
The script chdirs to EEG_ROOT: the run directories (output/ttg_stage1, output/v3_vidseeds) and a
relative --out are resolved against EEG_ROOT, and --out must resolve under
$EEG_ROOT/output/ttg_*. The default --out is the stored result
output/ttg_stage1/ens/stage1_ensemble.json, which a run overwrites; pass --out elsewhere to
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
ap.add_argument('--out', default='output/ttg_stage1/ens/stage1_ensemble.json',
                help='result JSON (relative: to EEG_ROOT)')
A = ap.parse_args()
os.chdir(EEG_ROOT)                                    # every relative path below is EEG_ROOT's
A.out = os.path.realpath(A.out)                       # relative: EEG_ROOT's (after the chdir)
if not A.out.startswith(os.path.join(EEG_ROOT, 'output', 'ttg_')):
    raise SystemExit(f'--out must resolve under {EEG_ROOT}/output/ttg_*, got {A.out}')
os.makedirs(os.path.dirname(A.out), exist_ok=True)

REPS = 2000
S1 = 'output/ttg_stage1'

def new_members(task, arm, rule):
    P = []
    for s in (1, 2, 3):
        d = f'{S1}/x3dfix_{arm}_s{s}'
        if rule == 'last':
            e = 12
        else:
            h = json.load(open(f'{d}/history.json'))
            rows = h['epochs'] if isinstance(h, dict) and 'epochs' in h else h
            vals = [(r['val'][task]['macro_f1'] if 'val' in r else r[f'val_{task}_macro_f1'], r['epoch']) for r in rows]
            e = max(vals)[1]
        z = np.load(f'{d}/val_ep{e:02d}.npz', allow_pickle=True)
        P.append((np.array([D.key_of(p) for p in z['path']]), z[f'probs_{task}'].astype(np.float64)))
    return P

def align(pairs, keys):
    pos = {k: i for i, k in enumerate(keys)}
    out = []
    for k, p in pairs:
        assert set(k) == set(keys)
        a = np.empty_like(p); a[np.array([pos[x] for x in k])] = p
        out.append(a)
    return np.array(out)

def metrics(E, y, task, an, se, W, ug):
    pred = E.argmax(1)
    rep = D.full_report(y, pred, task)
    s = E[:, D.SEV[task]].sum(1)
    is_sev = np.isin(y, D.SEV[task]); is_mild = np.isin(y, D.MILD[task])
    rep['within'] = float(D.within_session_auc(s, is_sev, is_mild, se)[0])
    Cg, _ = D.per_group_cm(y, pred, an, D.K[task], ug)
    bm = D.boot_cm_metrics(Cg, W, task)
    return rep, bm

res = {}
for task in ('g3', 'g5'):
    keys, y, y5 = D.reference_val(task)
    an, se = D.animal_session(keys)
    ug = np.unique(an); W = D.boot_weights(len(ug), REPS, seed=0)
    ref_raw_y = np.load(os.path.join(D.EEG_ROOT, f'output/v3_vidseeds/mvit_{task}_s1/val_preds.npz'))['y']
    def stored(bbs):
        dirs = [f'output/v3_vidseeds/{b}_{task}_s{s}' for b in bbs for s in (1, 2, 3, 5)] + [f'vid_{b}_{task}' for b in bbs]
        P, names, log = D.load_members(dirs, keys, ref_raw_y, y)
        return P, log
    Pms, lg1 = stored(['mvit', 'slowfast'])
    Px, lg2 = stored(['x3d'])
    arms = {'dual': 'dual', 'dedicated': task}
    ens = {'top3x5 stored (bugged X3D)': np.concatenate([Pms, Px]),
           'mvit+slowfast only (10)': Pms}
    for an_name, arm in arms.items():
        for rule in ('best', 'last'):
            Pn = align(new_members(task, arm, rule), keys)
            ens[f'swap: fixed X3D {an_name}, {rule} epoch (13)'] = np.concatenate([Pms, Pn])
            ens[f'add: fixed X3D {an_name}, {rule} epoch (18)'] = np.concatenate([Pms, Px, Pn])
    base_rep, base_bm = metrics(ens['top3x5 stored (bugged X3D)'].mean(0), y, task, an, se, W, ug)
    res[task] = {'load_log': {'double_softmax_repaired': lg1['double_softmax_repaired'] + lg2['double_softmax_repaired'],
                              'dropped': lg1['dropped'] + lg2['dropped'], 'pathless_admitted': lg1['pathless_admitted'] + lg2['pathless_admitted']}}
    for name, P in ens.items():
        rep, bm = metrics(P.mean(0), y, task, an, se, W, ug)
        d = bm['macro_f1'] - base_bm['macro_f1']
        rep['d_macro_f1_vs_stored'] = float(rep['macro_f1'] - base_rep['macro_f1'])
        rep['d_macro_f1_ci'] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
        rep['p_d_gt0'] = float((d > 0).mean())
        rep['n_members'] = int(P.shape[0])
        res[task][name] = rep
        rc = ' '.join(f"{v['recall']:.3f}/{v['n']}" for v in rep['per_class'].values())
        print(f"{task} {name:44s} mF1 {rep['macro_f1']:.4f}  d {rep['d_macro_f1_vs_stored']:+.4f} [{rep['d_macro_f1_ci'][0]:+.4f},{rep['d_macro_f1_ci'][1]:+.4f}] P>0 {rep['p_d_gt0']:.3f}  within {rep['within']:.4f}  recall {rc}")
json.dump(res, open(A.out, 'w'), indent=1, default=str)
print('wrote', A.out)
