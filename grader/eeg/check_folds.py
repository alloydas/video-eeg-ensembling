#!/usr/bin/env python3
"""
Do the EEG and video subject-disjoint folds (split_subjects, split seed 49, 5 folds) hold
the SAME animals? (EGRG deliverable 5; imported by joint_gate.py for protocol B.)

PURPOSE
  The EGRG gate is fitted leave-one-animal-out on out-of-fold (OOF) posteriors that come
  from two separately trained modalities. If EEG fold k and video fold k held different
  animals, some OOF predictions of one modality would come from a model that trained on
  the scored animal, and the gate would be fitted on leaked scores. The two trainers use
  different item lists (EEG: the segment cache's clip_sess, 601 sessions / 24,452 clips;
  video: 604 sessions / 24,497 clips), so the equality is checked, not assumed:

    EEG      train_pooled_eeg.split_subjects(clip_sess, 49, k)          (train_eeg_det.py,
             v3_subject_cv/eeg_*_fold k)
    video    dhlib.split_subjects(items from cache_frames/f32s224/index.json, 49, k)
             (the stored R(2+1)D subject_cv/vid_*_fold k, microway item order) and
             train_pooled.split_subjects(sorted items, 49, k) (grader/train_grader.py
             --split subject, which sorts items by path first)
  plus, per fold: every EEG clip lies in the video fold (0 EEG clips outside) and the
  number of video clips with no EEG (these are the 45 clips dropped in OOF scoring), and,
  when present, the animals actually stored in output/v3_subject_cv/eeg_bin_fold k.

USAGE (from the video-eeg-ensembling repo, cwd anywhere; EEG_ROOT env var, default
       /work/mech-ai-scratch/alloy/EEG, holds the segment cache, the frame index and the imported
       train_pooled / train_pooled_eeg; DHLIB_DIR env var, default grader/ (this file's parent
       directory), the directory holding the vendored dhlib.py)
  python grader/eeg/check_folds.py [--json /work/mech-ai-scratch/alloy/EEG/output/ttg_eeg_gate/folds.json]
  Exit 0 if every fold matches on every route, 1 otherwise.
"""
import argparse
import json
import os
import sys

sys.dont_write_bytecode = True
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EEG_ROOT = os.path.realpath(os.environ.get("EEG_ROOT", "/work/mech-ai-scratch/alloy/EEG"))
# dhlib.py (the decision-headroom loaders) is vendored into grader/; DHLIB_DIR overrides.
DH_DIR = os.path.realpath(os.environ.get("DHLIB_DIR", os.path.dirname(HERE)))
if not os.path.isfile(os.path.join(DH_DIR, "dhlib.py")):
    raise SystemExit(f"dhlib.py not found in {DH_DIR}: check_folds.py needs the validated decision-headroom "
                     f"loaders (items_from_cache_index, split_subjects). Set DHLIB_DIR to the directory "
                     f"that holds dhlib.py.")
for p in (EEG_ROOT, DH_DIR, os.path.dirname(HERE)):         # base trainers, dhlib, grader/
    if p not in sys.path:
        sys.path.insert(0, p)
CACHE = os.path.join(EEG_ROOT, "cache_bestcfg", "seg_w6.0_s3.0_d8.npz")
SEED, N_FOLDS = 49, 5


def _key(p):
    p = str(p)
    return p[: -len("/video.mp4")] if p.endswith("/video.mp4") else p.rstrip("/")


def expected_folds(seed=SEED, n_folds=N_FOLDS, cache=CACHE):
    """{fold: dict(eeg=set(animals), video_index=set, video_grader=set, eeg_keys=set,
    video_keys=set)} -- the held-out animals of every route, and the clip keys."""
    import dhlib as D
    import train_pooled_eeg as tpe                 # split_subjects(csess, ...)
    import train_pooled as tp                      # split_subjects(items, ...) (train_grader route)
    z = np.load(cache, allow_pickle=True)
    csess, cpath = z["clip_sess"], z["clip_path"]
    items = D.items_from_cache_index()
    items_sorted = sorted(items, key=lambda it: it[0])
    out = {}
    for k in range(n_folds):
        vs, valsub = tpe.split_subjects(csess, seed, k, n_folds)
        _, va_i, val_i, _ = D.split_subjects(items, seed, fold=k, n_folds=n_folds)
        _, va_g, val_g, _ = tp.split_subjects(items_sorted, seed, k, n_folds)
        eeg_keys = {str(p) for p, s in zip(cpath, csess) if s in vs}
        out[k] = dict(eeg=set(valsub), video_index=set(val_i), video_grader=set(val_g),
                      eeg_keys=eeg_keys, video_keys={_key(i[0]) for i in va_i},
                      video_keys_grader={_key(i[0]) for i in va_g})
    return out


def fold_table(seed=SEED, n_folds=N_FOLDS, cache=CACHE, stored=True, verbose=True):
    """Run every check; returns (ok, rows)."""
    F = expected_folds(seed, n_folds, cache)
    rows, ok = [], True
    for k, f in F.items():
        same = f["eeg"] == f["video_index"] == f["video_grader"]
        outside = len(f["eeg_keys"] - f["video_keys"])
        same_clips = f["video_keys"] == f["video_keys_grader"]
        r = dict(fold=k, eeg_animals=sorted(f["eeg"]), video_animals=sorted(f["video_index"]),
                 video_grader_animals=sorted(f["video_grader"]), same_animals=same,
                 n_eeg_clips=len(f["eeg_keys"]), n_video_clips=len(f["video_keys"]),
                 n_video_clips_without_eeg=len(f["video_keys"] - f["eeg_keys"]),
                 n_eeg_clips_outside_video_fold=outside,
                 video_index_and_grader_routes_same_clips=same_clips)
        if stored:
            p = os.path.join(EEG_ROOT, f"output/v3_subject_cv/eeg_bin_fold{k}/val_clip_preds.npz")
            if os.path.exists(p):
                s = set(str(x) for x in np.load(p, allow_pickle=True)["sub"])
                r["stored_eeg_bin_fold_animals"] = sorted(s)
                r["stored_eeg_matches"] = s == f["eeg"]
                same = same and s == f["eeg"]
        r["ok"] = bool(same and outside == 0 and same_clips)
        ok = ok and r["ok"]
        rows.append(r)
        if verbose:
            print(f"fold {k}: EEG {sorted(f['eeg'])}  video {sorted(f['video_index'])}  "
                  f"grader-route {sorted(f['video_grader'])}  same={same}  | clips EEG "
                  f"{r['n_eeg_clips']} video {r['n_video_clips']} (video w/o EEG "
                  f"{r['n_video_clips_without_eeg']}, EEG outside video fold {outside})"
                  + (f"  stored eeg_bin_fold{k} same={r.get('stored_eeg_matches')}" if 'stored_eeg_matches' in r else "")
                  + f"  [{'PASS' if r['ok'] else 'FAIL'}]", flush=True)
    if verbose:
        n_wo = sum(r["n_video_clips_without_eeg"] for r in rows)
        print(f"{'ALL FOLDS MATCH' if ok else 'FOLD MISMATCH'}: {sum(r['n_eeg_clips'] for r in rows)} EEG "
              f"clips, {sum(r['n_video_clips'] for r in rows)} video clips, {n_wo} video clips have no EEG")
    return ok, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n_folds", type=int, default=N_FOLDS)
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--json", default=None, help="write the table here (under output/ttg_eeg*/)")
    a = ap.parse_args()
    ok, rows = fold_table(a.seed, a.n_folds, a.cache)
    if a.json:
        rp = os.path.realpath(a.json)
        if not rp.startswith(os.path.join(EEG_ROOT, "output", "ttg_eeg")):
            raise SystemExit("--json must live under output/ttg_eeg*/")
        os.makedirs(os.path.dirname(rp), exist_ok=True)
        with open(rp, "w") as f:
            json.dump(dict(ok=ok, seed=a.seed, n_folds=a.n_folds, folds=rows), f, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
