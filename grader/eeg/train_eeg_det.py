#!/usr/bin/env python3
"""
EEG seizure DETECTOR for the EEG-Gated Racine Grader (EGRG): the parent repo's TCN, binary,
reported at a FIXED LAST EPOCH, with per-epoch clip AND window posterior dumps.

PURPOSE
  EGRG decides seizure-vs-not from video and EEG together (a 3-parameter logistic gate,
  grader/eeg/joint_gate.py) and lets video alone grade the seizure. Its EEG half is exactly
  train_pooled_eeg.py's TCN run with --group2 (non-seizure vs seizure). What changes here is
  the reporting contract, not the model:

  * NO checkpoint selection on the scored set. train_pooled_eeg.py writes best.pt and
    val_clip_preds.npz only at the epoch with the best val clip macro-F1 (an upper bound).
    Here every epoch is dumped and results.json reports the LAST epoch (--epochs, default
    30). The best epoch is recorded only as 'best_epoch_upper_bound' for comparison.
  * per epoch: val_clip_ep{e:02d}.npz (epochs numbered from 1) holding
        path          clip key (the clip DIRECTORY, 'data/Data_RNxxx_cropped/<sess>/<clip>'),
                      the key joint_gate.py aligns on (video keys are <key>/video.mp4)
        y, y5         binary label and the 5-class stage (0 non-seizure, 1..4 = S2..S5),
                      so g3 / g5 can be derived for the joint model
        p_logmean     clip P(seizure) under logmean pooling (geometric mean of window
                      posteriors, renormalised) = the trainer's usual --agg logmean score,
                      as v3_bestcfg / v3_eegalign. PRIMARY gate input.
        p_mean        clip P(seizure) under mean pooling
        p_top25       mean of the top 25% window P(seizure) of the clip (MIL top-k; the
                      prototype's jlib.topk_pool). SECONDARY: it cost S5 recall at g5 OOF.
        probs_logmean / probs_mean   the full (N, 2) pooled vectors
        win_probs, win_offsets       window posteriors (Nw, 2) packed per clip: the windows
                      of clip i, in time order, are win_probs[win_offsets[i]:win_offsets[i+1]]
        clip_id, sub, sess, epoch, seed, split, fold, session_universe, arch
    history.json (train loss, clip metrics under logmean and mean pooling, AUROC, MCC,
    per-class recall with counts), last.pt (atomic: model, optimizer, scheduler, python /
    numpy / torch / cuda RNG states, the epoch's shuffle seed, step, history), and at the end
    final.pt (last-epoch weights) and results.json (LAST epoch = the reported number).
  * resumable on the preemptible scavenger partition (GraceTime 0, KillWait 30 s): a
    SIGTERM / SIGUSR1 / SIGINT handler checkpoints at the next step boundary (a step is a few
    ms on a GPU) or inside validation, and exits 3. A resumed run continues with the same
    shuffle, the same dropout RNG stream and the same optimizer state; on CPU the result is
    BIT-identical to an uninterrupted run (grader/eeg/verify_eeg_det.py resume). On a GPU, cuDNN's
    non-deterministic kernels and a requeue onto another GPU type change only float rounding.

FAITHFULNESS TO train_pooled_eeg.py (imported, not copied: TCN / build_model,
split_sessions, split_subjects, video_val_sessions, aggregate_clip, GROUP2)
  identical: cache, per-window z-score, --group2 window and clip labels, the three split
  modes, the aligned-subset proof, inverse-frequency class weights normalised to mean 1,
  AdamW(lr 1e-3, wd 1e-4), CosineAnnealingLR(T_max=epochs) stepped per epoch, batch 256
  with shuffle and drop_last, fp32, val batch 512, seeding (random / numpy / torch), and
  the ORDER of every global-RNG draw (model init, the dummy forward, the DataLoader
  iterator's base seed, the RandomSampler seed, dropout, the val iterator's base seed).
  The training loop is replicated (about 30 lines) because the original has no resume;
  its shuffle is reproduced exactly: per epoch one int64 is drawn for the iterator base
  seed and one for the sampler seed, and the permutation is randperm(n) under the latter.
  verify_eeg_det.py proves it: same seed -> the same first-batch losses as the ORIGINAL
  main() run in-process, and a stored v3_eegalign best.pt evaluated through this file's
  split / eval / pooling path reproduces its stored window and clip posteriors.

USAGE (from the video-eeg-ensembling repo, any cwd; PYTHONDONTWRITEBYTECODE=1, the eeg conda env)
  EEG_ROOT (env var, default /work/mech-ai-scratch/alloy/EEG) holds the segment cache, the video
  frame index, the imported train_pooled_eeg and every run output. The script puts EEG_ROOT on
  sys.path and chdirs there, so relative --cache / --video_index are resolved against EEG_ROOT.
  aligned 5,279-clip split (protocol A; EEG val = strict subset of video's 5,289):
    python grader/eeg/train_eeg_det.py --split session --session_universe video --seed 1 \
        --output /work/mech-ai-scratch/alloy/EEG/output/ttg_eeg/aligned/tcn_bin_s1
  subject-disjoint fold (protocol B; --session_universe video is refused here):
    python grader/eeg/train_eeg_det.py --split subject --fold 0 --seed 1 \
        --output /work/mech-ai-scratch/alloy/EEG/output/ttg_eeg/subject/tcn_bin_fold0_s1
  CPU smoke tests (results meaningless):
    --dry_run --allow_cpu              split + proof + class weights + one forward; writes nothing
    --allow_cpu --limit_train 4096 --limit_val 40 --epochs 2 --threads 4 --output ...
  SLURM: grader/eeg/sbatch_eeg_det.sh with grader/eeg/eeg_aligned.tsv or eeg_subject.tsv
         (submit with grader/eeg/submit_eeg.sh; it dry-runs unless DRY_RUN=0).

EXIT CODES  0 finished (results.json written, or already complete) | 3 stopped by a signal
            after a resumable checkpoint | 4 GPU architecture not supported by this torch build
            | 1 configuration / preflight error.
"""
import argparse
import json
import os
import random
import signal
import sys
import time

sys.dont_write_bytecode = True
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                 # grader/ (ttg_common)
import ttg_common as C                                    # noqa: E402  (puts EEG_ROOT on sys.path)

import torch                                              # noqa: E402
import torch.nn as nn                                     # noqa: E402
from torch.utils.data import DataLoader, TensorDataset    # noqa: E402

import train_pooled_eeg as tpe                            # noqa: E402

EEG_ROOT = C.EEG_ROOT
DEFAULT_CACHE = "cache_bestcfg/seg_w6.0_s3.0_d8.npz"
OUT_PREFIX = os.path.join(EEG_ROOT, "output", "ttg_eeg")
EXIT_STOPPED, EXIT_BAD_GPU = 3, 4
NC = 2                                                    # binary detector (--group2)
NAMES_BY_NC = {2: tpe.NAMES2, 3: tpe.NAMES3, 5: tpe.NAMES5}
GROUP_BY_NC = {2: tpe.GROUP2, 3: tpe.GROUP3, 5: np.arange(5)}
STOP = {"flag": False, "sig": None}


def _on_signal(sig, _frm):
    if not STOP["flag"]:
        print(f"[{time.strftime('%F %T')}] signal {signal.Signals(sig).name}: checkpoint at the "
              f"next step boundary, then exit {EXIT_STOPPED}", flush=True)
    STOP["flag"], STOP["sig"] = True, int(sig)


# ============================================================================ data / split

def load_cache(path):
    """The segment cache, z-scored per window exactly as train_pooled_eeg.main()."""
    d = np.load(path, allow_pickle=True)
    segs = d["segs"]
    out = dict(wlab5=d["wlab"], cid=d["clip_id"], clab5=d["clip_lab"], csess=d["clip_sess"],
               cpath=d["clip_path"] if "clip_path" in d.files else None)
    m = segs.mean(1, keepdims=True); sd = segs.std(1, keepdims=True); sd[sd == 0] = 1.0
    out["segs"] = ((segs - m) / sd).astype(np.float32)
    return out


def make_split(split, universe, fold, n_folds, split_seed, csess, clab, cpath, nc,
               video_index=tpe.VIDEO_INDEX, allow_orphans=False):
    """Per-clip val mask + info dict. Mirrors the split block of train_pooled_eeg.main()
    (including its module-global NC, which split_sessions reads) for task width `nc`."""
    tpe.NC, tpe.NAMES = nc, NAMES_BY_NC[nc]          # what main()'s `global NAMES, NC` does
    info = dict(split=split, session_universe=universe, split_seed=split_seed,
                fold=fold if split == "subject" else None,
                n_folds=n_folds if split == "subject" else None)
    vid_val_clips = None
    if split == "subject":
        if universe == "video":
            raise SystemExit("--session_universe video applies to --split session only: the "
                             "subject-disjoint folds hold out whole animals and are already "
                             "identical for EEG and video (check_folds.py). Drop the flag.")
        val_sess, valsub = tpe.split_subjects(csess, split_seed, fold, n_folds)
        info.update(used_seed=split_seed, val_subjects=sorted(valsub))
        print(f"*** SUBJECT-DISJOINT EEG: fold {fold}/{n_folds}, held-out={sorted(valsub)} ***")
    elif universe == "video":
        if cpath is None:
            raise SystemExit("--session_universe video needs a cache carrying clip_path")
        val_sess, used, vid_val_clips, vid_sess, n_vid = tpe.video_val_sessions(split_seed, video_index)
        orphan = set(csess.tolist()) - vid_sess
        if orphan and not allow_orphans:
            raise SystemExit(f"{len(orphan)} EEG session(s) are absent from the video universe "
                             f"({sorted(orphan)[:5]}); pass --allow_orphan_sessions if intended")
        info.update(used_seed=int(used), video_index=video_index, n_video_items=int(n_vid),
                    n_video_sessions=len(vid_sess), n_video_val_sessions=len(val_sess),
                    n_video_val_clips=len(vid_val_clips), n_orphan_sessions=len(orphan))
        print(f"*** SESSION UNIVERSE = VIDEO: {n_vid} video clips / {len(vid_sess)} sessions -> "
              f"val {len(val_sess)} sessions / {len(vid_val_clips)} video clips (split seed {used}) ***")
    elif universe == "eeg":
        val_sess, used = tpe.split_sessions(list(csess), [int(x) for x in clab], split_seed)
        info.update(used_seed=int(used))
        print("*** SESSION UNIVERSE = EEG (train_pooled_eeg default): this val set is NOT the "
              "video val set (5,319 vs 5,289 clips, overlap 2,830); never pair it with video ***")
    else:
        raise SystemExit("--split session needs --session_universe {eeg,video} chosen explicitly "
                         "('video' = the aligned 5,279-clip split that can be paired with video)")
    clip_is_val = np.array([s in val_sess for s in csess])
    if vid_val_clips is not None:                    # PROVE the subset, as the original does
        vp = set(np.asarray(cpath)[clip_is_val].tolist())
        extra = vp - vid_val_clips
        if extra:
            raise SystemExit(f"ALIGNMENT FAILED: {len(extra)} EEG val clips are not in the video "
                             f"val set, e.g. {sorted(extra)[:3]}")
        lab = np.asarray(clab)
        ctr = np.bincount(lab[~clip_is_val], minlength=nc)
        cva = np.bincount(lab[clip_is_val], minlength=nc)
        if (ctr[:nc] == 0).any() or (cva[:nc] == 0).any():
            raise SystemExit(f"aligned split leaves a class empty (train={ctr.tolist()}, "
                             f"val={cva.tolist()}) -- refusing to train")
        info.update(n_eeg_val_clips=len(vp), n_video_val_clips_without_eeg=len(vid_val_clips) - len(vp),
                    n_val_sessions_with_eeg=len(set(csess[clip_is_val].tolist())),
                    subset_of_video_val=True)
        print(f"*** ALIGNED val: {len(vp)} EEG clips, a strict SUBSET of the {len(vid_val_clips)} "
              f"video val clips ***")
    info["n_val_sessions"] = len(val_sess)
    info["val_sessions"] = sorted(val_sess)
    return clip_is_val, info


def class_weights(ytr, nc):
    """Inverse-frequency weights normalised to mean 1 (train_pooled_eeg.main, verbatim)."""
    freq = np.array([max(int((ytr == i).sum()), 1) for i in range(nc)], float)
    w = 1.0 / freq
    return w / w.mean()


def clip_offsets(cid_va, val_clips):
    """Per-clip [start, end) into the val window arrays. The cache stores each clip's windows
    contiguously and in time order (clip_id is non-decreasing); refuse anything else."""
    if len(cid_va) and np.any(np.diff(cid_va) < 0):
        raise SystemExit("val windows are not grouped by clip; cannot pack window posteriors")
    st = np.searchsorted(cid_va, val_clips, "left")
    en = np.searchsorted(cid_va, val_clips, "right")
    if not (np.array_equal(st[1:], en[:-1]) and st[0] == 0 and en[-1] == len(cid_va)):
        raise SystemExit("val windows are not contiguous per clip")
    return np.r_[st, en[-1]].astype(np.int64)


def top_frac(p_w, off, frac=0.25):
    """MIL top-k clip score: mean of the top max(1, round(frac * n)) window scores of each
    clip (same rule as the joint-gated prototype's jlib.topk_pool)."""
    out = np.empty(len(off) - 1)
    for i in range(len(off) - 1):
        v = np.sort(p_w[off[i]:off[i + 1]])
        k = max(1, int(round(frac * len(v))))
        out[i] = v[-k:].mean()
    return out


def predict_windows(model, dva, dev):
    """Window softmax posteriors, as the validation block of train_pooled_eeg.main().
    Returns None if a stop signal arrives mid-way."""
    model.eval()
    P = []
    with torch.no_grad():
        for x, _ in dva:
            if STOP["flag"]:
                return None
            P.append(torch.softmax(model(x.to(dev)), 1).cpu().numpy())
    return np.concatenate(P)


def clip_metrics(y, prob):
    """Binary clip metrics: macro P/R/F1, accuracy, MCC, AUROC, per-class recall WITH counts."""
    from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                                 precision_recall_fscore_support, roc_auc_score)
    pred = prob.argmax(1)
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, labels=range(NC), zero_division=0)
    try:
        auc = float(roc_auc_score(y, prob[:, 1]))
    except ValueError:
        auc = float("nan")
    return dict(macro_f1=float(f1_score(y, pred, average="macro", zero_division=0)),
                macro_precision=float(pr.mean()), macro_recall=float(rc.mean()),
                accuracy=float(accuracy_score(y, pred)), mcc=float(matthews_corrcoef(y, pred)),
                auroc=auc,
                per_class={tpe.NAMES2[i]: dict(recall=float(rc[i]), n_correct=int(((y == i) & (pred == i)).sum()),
                                               n=int((y == i).sum()), precision=float(pr[i]))
                           for i in range(NC)})


# ============================================================================ state io

def rng_state():
    st = dict(py=random.getstate(), np=np.random.get_state(), torch=torch.get_rng_state())
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st):
    random.setstate(st["py"]); np.random.set_state(st["np"]); torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def atomic_torch_save(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


_LOCK_FH = []


def hold_run_lock(out):
    import fcntl
    fh = open(os.path.join(out, ".run.lock"), "a+")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another process is training into {out} (.run.lock held); refusing")
    except OSError as e:
        print(f"warning: could not lock {out}/.run.lock ({e}); continuing", flush=True)
    fh.seek(0); fh.truncate()
    fh.write(f"{os.uname().nodename} pid {os.getpid()} job {os.environ.get('SLURM_JOB_ID')}\n")
    fh.flush()
    _LOCK_FH.append(fh)


def check_output(path):
    if not path or not os.path.isabs(path):
        raise SystemExit(f"--output must be an absolute path under {OUT_PREFIX}*/, got {path!r}")
    rp = os.path.realpath(path)
    if not rp.startswith(OUT_PREFIX):
        raise SystemExit(f"--output must live under {OUT_PREFIX}*/, got {rp}")
    return rp


# ============================================================================ main

def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="See the module docstring (python -c 'import ...') or the "
                                        "file header for the output contract.")
    ap.add_argument("--split", choices=["session", "subject"], default="session")
    ap.add_argument("--session_universe", choices=["eeg", "video"], default=None,
                    help="session split only, REQUIRED there: 'video' = aligned 5,279-clip EEG val "
                         "set (strict subset of video's 5,289); 'eeg' = train_pooled_eeg default "
                         "(5,319 clips, NOT pairable with video). Refused with --split subject.")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--split_seed", type=int, default=49)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4, help="train_pooled_eeg hard-codes 1e-4")
    ap.add_argument("--arch", default="tcn", choices=["tcn", "gru", "lstm", "eegnet", "conformer"],
                    help="EGRG uses tcn (5 blocks x 64 ch, k=7, dilations 1..16)")
    ap.add_argument("--hidden", type=int, default=128, help="gru/lstm only")
    ap.add_argument("--agg", default="logmean", choices=["logmean", "mean"],
                    help="pooling behind the printed / history clip metrics and results.json "
                         "(both poolings are always dumped)")
    ap.add_argument("--topk_frac", type=float, default=0.25, help="p_top25 fraction")
    ap.add_argument("--cache", default=DEFAULT_CACHE, help="segment cache, relative to EEG_ROOT or absolute")
    ap.add_argument("--video_index", default=tpe.VIDEO_INDEX,
                    help="video frame index defining the video universe, relative to EEG_ROOT or absolute")
    ap.add_argument("--allow_orphan_sessions", action="store_true")
    ap.add_argument("--output", default=None,
                    help=f"absolute run dir under {OUT_PREFIX}*/ (not needed with --dry_run)")
    ap.add_argument("--threads", type=int, default=0, help="torch.set_num_threads (0 = torch default)")
    ap.add_argument("--require_cuda", action="store_true", help="hard-fail without CUDA")
    ap.add_argument("--allow_cpu", action="store_true", help="smoke tests only")
    ap.add_argument("--dry_run", action="store_true",
                    help="split + proof + class weights + one forward pass; writes nothing")
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke tests: shorthand for --limit_train N --limit_val N")
    ap.add_argument("--limit_train", type=int, default=0,
                    help="smoke tests: seeded subsample of N TRAIN WINDOWS")
    ap.add_argument("--limit_val", type=int, default=0,
                    help="smoke tests: seeded subsample of N VAL CLIPS (all their windows)")
    ap.add_argument("--debug_max_steps", type=int, default=0,
                    help="tests: stop every epoch after this many steps")
    ap.add_argument("--debug_sigterm_at", default=None,
                    help="tests: 'E:S' raise SIGTERM in-process after step S of epoch E")
    ap.add_argument("--debug_sigterm_in_val", type=int, default=0,
                    help="tests: raise SIGTERM in-process before validating epoch E")
    a = ap.parse_args(argv)
    if a.limit:
        a.limit_train = a.limit_train or a.limit
        a.limit_val = a.limit_val or a.limit
    return a


def main(argv=None):
    a = parse(argv)
    C.enter_eeg_root(need_videos=False)      # the cache and the video index are relative paths
    if a.split == "subject" and a.session_universe == "video":
        raise SystemExit("--session_universe video applies to --split session only (the subject "
                         "folds are identical for EEG and video already; see check_folds.py)")
    if a.split == "subject" and not (a.n_folds >= 2 and 0 <= a.fold < a.n_folds):
        raise SystemExit(f"--fold {a.fold} / --n_folds {a.n_folds}: need n_folds >= 2 and 0 <= fold < n_folds "
                         f"(split_subjects would silently wrap the fold and train the wrong animals "
                         f"under this output name)")
    if a.split == "session" and a.session_universe is None:
        raise SystemExit("--split session needs --session_universe {eeg,video} chosen explicitly")
    out = None if a.dry_run else check_output(a.output)
    if out and os.path.exists(os.path.join(out, "results.json")):
        print(f"{out}/results.json exists: run already complete, nothing to do")
        return 0
    if a.threads:
        torch.set_num_threads(a.threads)
    if not a.dry_run:                   # before the ~15 s cache load: a signal there exits 3 below
        for s_ in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
            signal.signal(s_, _on_signal)

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        ok, msg = C.check_cuda_arch()
        print(msg, flush=True)
        if not ok:
            print("FATAL: this torch build has no kernels for this GPU; exclude the node "
                  "(--constraint='a100|h200|l40s')", flush=True)
            sys.exit(EXIT_BAD_GPU)
        dev = torch.device("cuda")
        (torch.ones(8, device=dev) * 2).sum().item()        # CUDA really initialises
    elif a.require_cuda:
        raise SystemExit("FATAL: CUDA is not available (--require_cuda)")
    elif a.allow_cpu or a.dry_run:
        dev = torch.device("cpu")
        print("running on CPU (smoke test)", flush=True)
    else:
        raise SystemExit("CUDA is not available -- refusing to fall back to CPU (pass --allow_cpu "
                         "only for smoke tests)")

    t0 = time.time()
    D = load_cache(a.cache)
    segs, cid, csess, cpath = D["segs"], D["cid"], D["csess"], D["cpath"]
    if cpath is None:
        raise SystemExit(f"{a.cache} has no clip_path: the clip key every dump is aligned on")
    wlab = tpe.GROUP2[D["wlab5"]]                 # --group2 window / clip labels, verbatim
    clab = tpe.GROUP2[D["clab5"]]
    clab5 = D["clab5"]
    print(f"*** pooled EEG 2-class *** cache {a.cache}: {len(segs)} windows / {len(clab)} clips "
          f"({time.time() - t0:.0f}s)", flush=True)

    clip_is_val, split_info = make_split(a.split, a.session_universe, a.fold, a.n_folds,
                                         a.split_seed, csess, clab, cpath, NC, a.video_index,
                                         a.allow_orphan_sessions)
    win_is_val = clip_is_val[cid]
    tr_idx = np.flatnonzero(~win_is_val)
    va_win = np.flatnonzero(win_is_val)
    if a.limit_train or a.limit_val:
        rng = np.random.default_rng(12345)       # never the global RNG
        if a.limit_train:
            tr_idx = np.sort(rng.choice(tr_idx, min(a.limit_train, len(tr_idx)), replace=False))
        if a.limit_val:
            vc = np.unique(cid[va_win])
            keep = np.sort(rng.choice(vc, min(a.limit_val, len(vc)), replace=False))
            va_win = va_win[np.isin(cid[va_win], keep)]
        print(f"*** SMOKE SUBSAMPLE: {len(tr_idx)} train windows, {len(np.unique(cid[va_win]))} "
              f"val clips -- results are meaningless, do not report them ***", flush=True)
    Xtr, ytr = segs[tr_idx], wlab[tr_idx]
    Xva, yva, cid_va = segs[va_win], wlab[va_win], cid[va_win]
    T = segs.shape[1]
    del segs, D
    val_clips = np.unique(cid_va)
    off = clip_offsets(cid_va, val_clips)
    v_path = np.asarray(cpath)[val_clips]
    v_y, v_y5 = clab[val_clips].astype(np.int64), clab5[val_clips].astype(np.int64)
    v_sess = np.asarray(csess)[val_clips]
    v_sub = np.array([s.split("/")[0] for s in v_sess])
    n_tr_clips = len(np.unique(cid[tr_idx]))
    print(f"subjects: {len(set(s.split('/')[0] for s in csess))}  clips: train={n_tr_clips} "
          f"val={len(val_clips)}  val_sessions={split_info['n_val_sessions']}  "
          f"seed={split_info['used_seed']}")
    print(f"windows: train={len(Xtr)} val={len(Xva)}")
    print(f"  train window labels: {np.bincount(ytr, minlength=NC).tolist()}")
    print(f"  val clip labels: binary {np.bincount(v_y, minlength=NC).tolist()}  "
          f"5-class {np.bincount(v_y5, minlength=5).tolist()}")
    w = class_weights(ytr, NC)
    print("class weights: " + "  ".join(f"{tpe.NAMES2[i]}={w[i]:.2f}" for i in range(NC)), flush=True)
    if len(Xtr) < a.batch_size:
        raise SystemExit(f"train set ({len(Xtr)} windows) smaller than one batch ({a.batch_size})")

    model = tpe.build_model(a.arch, NC, a.hidden).to(dev)
    with torch.no_grad():                         # as the original (materialises LazyLinear)
        model(torch.zeros(2, T, 1, device=dev))
    n_par = sum(p.numel() for p in model.parameters())
    print(f"arch: {a.arch}  params={n_par / 1e3:.1f}k", flush=True)
    if a.dry_run:
        with torch.no_grad():
            o = model(torch.from_numpy(Xtr[:4]).unsqueeze(-1).to(dev))
        print(f"DRY RUN: forward on 4 train windows -> logits {tuple(o.shape)}; "
              f"val fingerprint {C.fingerprint([str(p) for p in v_path])}; nothing written.")
        return 0

    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    Xtr_t, ytr_t = torch.from_numpy(Xtr).unsqueeze(-1), torch.from_numpy(ytr)
    dva = DataLoader(TensorDataset(torch.from_numpy(Xva).unsqueeze(-1), torch.from_numpy(yva)),
                     batch_size=512, shuffle=False)
    n_tr, bs = len(Xtr), a.batch_size
    n_batches = n_tr // bs                        # drop_last=True

    os.makedirs(out, exist_ok=True)
    hold_run_lock(out)
    run_key = dict(arch=a.arch, hidden=a.hidden, nc=NC, seed=a.seed, split=a.split,
                   session_universe=a.session_universe, fold=a.fold, n_folds=a.n_folds,
                   split_seed=a.split_seed, epochs=a.epochs, batch_size=bs, lr=a.lr,
                   weight_decay=a.weight_decay, agg=a.agg, topk_frac=a.topk_frac,
                   cache=os.path.realpath(a.cache), limit_train=a.limit_train,
                   limit_val=a.limit_val, debug_max_steps=a.debug_max_steps,
                   train_fp=C.fingerprint([str(i) for i in tr_idx.tolist()]),
                   val_fp=C.fingerprint([str(p) for p in v_path]))
    last_path = os.path.join(out, "last.pt")
    state = dict(next_epoch=1, next_step=0, train_done=False, perm_seed=None,
                 accum=dict(tot=0.0, n=0, steps=0), history=[], first_losses=[],
                 first_batch=None, resumes=[], rng_train_end=None)
    if os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        if ck["run_key"] != run_key:
            diff = {k: (ck["run_key"].get(k), v) for k, v in run_key.items() if ck["run_key"].get(k) != v}
            raise SystemExit(f"{last_path} belongs to a different configuration: {diff}")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); state = ck["state"]; set_rng_state(ck["rng"])
        state["resumes"].append(dict(at=time.strftime("%F %T"), epoch=state["next_epoch"],
                                     step=state["next_step"], train_done=state["train_done"],
                                     job=os.environ.get("SLURM_JOB_ID")))
        print(f"RESUME from {last_path}: epoch {state['next_epoch']} step {state['next_step']} "
              f"train_done={state['train_done']}", flush=True)
    else:
        if split_info.get("session_universe") == "video" and a.split == "session":
            C.atomic_json(os.path.join(out, "align_info.json"), split_info)
        C.atomic_json(os.path.join(out, "config.json"), dict(
            run_key=run_key, argv=sys.argv if argv is None else argv, started=time.strftime("%F %T"),
            split=split_info, n_train_windows=n_tr, n_val_windows=len(Xva), n_train_clips=n_tr_clips,
            n_val_clips=len(val_clips), train_window_labels=np.bincount(ytr, minlength=NC).tolist(),
            val_clip_labels_bin=np.bincount(v_y, minlength=NC).tolist(),
            val_clip_labels_5=np.bincount(v_y5, minlength=5).tolist(),
            class_weights=w.tolist(), params=n_par, host=os.uname().nodename,
            job=os.environ.get("SLURM_JOB_ID"), torch=torch.__version__))

    def save_last(rng=None):
        atomic_torch_save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                               state=state, rng=rng or rng_state(), run_key=run_key,
                               saved_at=time.strftime("%F %T")), last_path)

    def stop_now(where, rng=None):
        save_last(rng)
        print(f"[{time.strftime('%F %T')}] checkpointed ({where}: epoch {state['next_epoch']} step "
              f"{state['next_step']} train_done={state['train_done']}); exit {EXIT_STOPPED}", flush=True)
        sys.exit(EXIT_STOPPED)

    if STOP["flag"]:                    # signalled while loading: last.pt (if any) is still valid
        print(f"stopped by a signal before training started; exit {EXIT_STOPPED}", flush=True)
        sys.exit(EXIT_STOPPED)
    dbg = tuple(int(v) for v in a.debug_sigterm_at.split(":")) if a.debug_sigterm_at else None

    for ep in range(state["next_epoch"], a.epochs + 1):
        te = time.time()
        acc = state["accum"]
        if not state["train_done"]:
            model.train()
            lr_ep = float(opt.param_groups[0]["lr"])
            if state["perm_seed"] is None:
                # == iter(DataLoader(..., shuffle=True)): _BaseDataLoaderIter draws its base
                # seed, then RandomSampler draws the permutation seed, both from the global RNG
                torch.empty((), dtype=torch.int64).random_()
                state["perm_seed"] = int(torch.empty((), dtype=torch.int64).random_().item())
            g = torch.Generator(); g.manual_seed(state["perm_seed"])
            perm = torch.randperm(n_tr, generator=g)
            stop_at = min(n_batches, a.debug_max_steps) if a.debug_max_steps else n_batches
            for b in range(state["next_step"], stop_at):
                if STOP["flag"]:
                    state["next_step"] = b
                    stop_now("mid-epoch")
                idx = perm[b * bs:(b + 1) * bs]
                x, y = Xtr_t[idx].to(dev), ytr_t[idx].to(dev)
                opt.zero_grad(set_to_none=True)
                loss = crit(model(x), y); loss.backward(); opt.step()
                lv = loss.item()
                acc["tot"] += lv * len(y); acc["n"] += len(y); acc["steps"] += 1
                if ep == 1 and b < 5:              # reproducibility record (verify_eeg_det.py)
                    rb = dict(step=b, loss=lv, y_sum=int(y.sum()), idx_sum=int(idx.sum()))
                    state["first_losses"].append(rb)
                    if b == 0:
                        state["first_batch"] = rb
                        print(f"first batch: loss={lv!r} idx_sum={rb['idx_sum']} y_sum={rb['y_sum']}",
                              flush=True)
                if not np.isfinite(lv):
                    raise SystemExit(f"non-finite loss at epoch {ep} step {b}")
                if dbg and dbg == (ep, b + 1):
                    os.kill(os.getpid(), signal.SIGTERM)
            sched.step()
            state.update(train_done=True, next_step=0, lr_epoch=lr_ep)
            state["rng_train_end"] = rng_state()
        if a.debug_sigterm_in_val == ep:
            os.kill(os.getpid(), signal.SIGTERM)
        if STOP["flag"]:
            stop_now("before validation", state["rng_train_end"])
        P = predict_windows(model, dva, dev)
        if P is None:
            stop_now("during validation", state["rng_train_end"])
        cp_log = tpe.aggregate_clip(P, cid_va, val_clips, "logmean")
        cp_mean = tpe.aggregate_clip(P, cid_va, val_clips, "mean")
        p_top = top_frac(1.0 - P[:, 0].astype(np.float64), off, a.topk_frac)
        cp = cp_log if a.agg == "logmean" else cp_mean
        m_log, m_mean = clip_metrics(v_y, cp_log), clip_metrics(v_y, cp_mean)
        from sklearn.metrics import roc_auc_score
        try:
            auc_top = float(roc_auc_score(v_y, p_top))
        except ValueError:
            auc_top = float("nan")
        C.atomic_npz(os.path.join(out, f"val_clip_ep{ep:02d}.npz"),
                     path=v_path.astype(str), clip_id=val_clips.astype(np.int64), y=v_y, y5=v_y5,
                     sub=v_sub, sess=v_sess.astype(str),
                     p_logmean=cp_log[:, 1].astype(np.float64), p_mean=cp_mean[:, 1].astype(np.float64),
                     p_top25=p_top, probs_logmean=cp_log.astype(np.float32),
                     probs_mean=cp_mean.astype(np.float32), win_probs=P.astype(np.float32),
                     win_offsets=off, topk_frac=np.float64(a.topk_frac), epoch=np.int64(ep),
                     seed=np.int64(a.seed), arch=np.array(a.arch), split=np.array(a.split),
                     fold=np.int64(a.fold if a.split == "subject" else -1),
                     session_universe=np.array(str(a.session_universe)),
                     last_epoch=np.bool_(ep == a.epochs))
        m = m_log if a.agg == "logmean" else m_mean
        n_ = max(acc["n"], 1)
        h = dict(epoch=ep, lr=state.get("lr_epoch"), train_loss=acc["tot"] / n_, n_train=acc["n"],
                 steps=acc["steps"], clip_acc=m["accuracy"], clip_macro_f1=m["macro_f1"],
                 val_logmean=m_log, val_mean=m_mean, auroc_top25=auc_top,
                 secs=round(time.time() - te, 1))
        state["history"] = [r for r in state["history"] if r["epoch"] != ep] + [h]
        print(f"ep{ep:>2}  loss={h['train_loss']:.4f}  clip_acc={m['accuracy']:.4f}  "
              f"clip_macroF1={m['macro_f1']:.4f}  AUROC {a.agg}={m['auroc']:.4f} top25={auc_top:.4f}  "
              f"recall " + "/".join(f"{v['n_correct']}of{v['n']}" for v in m["per_class"].values())
              + f"  ({h['secs']}s)", flush=True)
        C.atomic_json(os.path.join(out, "history.json"), state["history"])
        state.update(next_epoch=ep + 1, next_step=0, train_done=False, perm_seed=None,
                     accum=dict(tot=0.0, n=0, steps=0), rng_train_end=None)
        save_last()
        if STOP["flag"]:
            stop_now("after epoch end")

    # ---- final summary: the LAST epoch is the reported number (no selection)
    last = os.path.join(out, f"val_clip_ep{a.epochs:02d}.npz")
    z = np.load(last, allow_pickle=True)
    prob = z["probs_logmean"] if a.agg == "logmean" else z["probs_mean"]
    try:
        rep = tpe.report(z["y"], prob.argmax(1), prob,
                         f"EEG detector {a.arch} seed {a.seed} -- LAST epoch {a.epochs} ({a.agg})")
        rep_src = "train_pooled_eeg.report"
    except Exception as e:                        # never lose results.json
        rep, rep_src = None, f"report failed: {type(e).__name__}: {e}"
    H = state["history"]
    key = "clip_macro_f1"
    best = max(H, key=lambda r: r[key])
    res = dict(run_key=run_key, complete=True, finished=time.strftime("%F %T"),
               reported_epoch=a.epochs, reported_file=os.path.basename(last),
               selection="none: fixed last epoch (pre-registered)",
               last_epoch={"agg": a.agg, "metrics_logmean": clip_metrics(z["y"], z["probs_logmean"]),
                           "metrics_mean": clip_metrics(z["y"], z["probs_mean"]),
                           "auroc_top25": H[-1]["auroc_top25"], "report": rep, "report_source": rep_src},
               best_epoch_upper_bound=dict(epoch=best["epoch"], clip_macro_f1=best[key],
                                           note="selected on the scored val set: an UPPER BOUND, "
                                                "never the reported number"),
               split=split_info, n_val_clips=int(len(z["y"])),
               val_clip_labels_5=np.bincount(z["y5"], minlength=5).tolist(),
               first_batch=state["first_batch"], first_losses=state["first_losses"],
               resumes=state["resumes"])
    atomic_torch_save(dict(model=model.state_dict(), epoch=a.epochs, run_key=run_key,
                           note="last-epoch weights; no selection"), os.path.join(out, "final.pt"))
    C.atomic_json(os.path.join(out, "results.json"), res)
    print(f"\nwrote {out}/results.json (LAST epoch {a.epochs})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
