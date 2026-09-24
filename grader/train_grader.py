#!/usr/bin/env python3
"""
Stage-1 (and later) video grader trainer for the Two-Timescale Grader (TT-X3D) programme.

PURPOSE
  Train X3D-M (or SlowFast-R50) with the pytorchvideo head-Softmax bug optionally fixed,
  with one shared backbone and separate Linear heads for g3 (non-seizure / mild S2-3 /
  severe S4-5) and g5 (non-seizure / S2 / S3 / S4 / S5), under a training contract that
  makes every stored number reproducible and every run resumable on the preemptible
  scavenger partition:

  * reuse, not copy: discover / split_sessions / split_subjects / FrameCache / norm_batch /
    _autocast / _SlowFastWrap / report are imported from train_pooled.py;
  * X3D fix (--fix_x3d): pytorchvideo's x3d_m head applies nn.Softmax in TRAIN mode too, so
    train_pooled's CE was taken on probabilities (loss floor ~0.59 g3 / ~1.03 g5, S5 recall
    0/34 in 5/5 seeds). The backbone here always has head.proj = Identity and
    head.activation = None (it is a feature extractor); the bug is reproduced for the
    control arm by applying softmax to each head's output before CE (numerically the
    historical head: pool -> dropout -> Linear -> Softmax -> 1x1x1 average);
  * heads: dual = shared features -> Linear(g3) and Linear(g5); loss = w_g3*CE_g3 + w_g5*CE_g5,
    each CE with inverse-frequency class weights normalised to mean 1 over that task's
    TRAIN counts (as train_pooled). g3 / g5 = one head only;
  * no silent input corruption: every train/val clip must be in --cache_dir and not marked
    unreadable, otherwise the run refuses to start (train_pooled.Clips would decode the
    zero-byte data/ placeholder and fill it with the Kinetics mean colour);
  * item order is canonical (sorted by path) so a requeued job on another node gets the
    identical train list even if NFS glob order differs; split membership is unchanged
    (split_sessions/split_subjects depend only on the session/subject sets);
  * per epoch: val_ep{e:02d}.npz, epochs numbered from 1 (val_ep01 ... val_ep12; there is
    no val_ep00) (path, y5, probs_g3/probs_g5 for the heads present; probs
    are SINGLE-softmax posteriors for fixed and bugged runs alike, flag head_softmax_bug),
    history.json (train loss per head, val macro-F1, MCC, per-class recall with counts,
    severe-vs-mild AUROC pooled and within-session), last.pt (atomic tmp+rename: model,
    optimizer, scheduler, GradScaler, epoch/step, python/numpy/torch/cuda RNG states,
    history), best_g3.pt / best_g5.pt by val macro-F1 (the historical rule; best-epoch
    numbers are an UPPER BOUND because they are selected on the scored set). Nothing is
    ever deleted;
  * resume: if last.pt exists the run continues from the next epoch -- or from the next
    STEP inside an epoch: the train order is a deterministic permutation per (seed, epoch)
    and augmentation draws come from a per-(seed, epoch, clip) generator, so the remaining
    batches are identical. A SIGTERM/SIGUSR1 handler saves a mid-epoch checkpoint at the
    next step boundary (checkpoint first, then DataLoader shutdown) and exits with code 3
    (scavenger: GraceTime 0, KillWait 30 s);
    DataLoader workers ignore SIGTERM so the in-flight step completes cleanly. A periodic
    mid-epoch checkpoint (--ckpt_every_min) bounds the loss if SIGKILL comes first;
  * mixed precision exactly as train_pooled (fp16 autocast + GradScaler on CUDA);
  * the realised train loss is logged per head every epoch next to the bugged floor, and
    the first training batch's output range / row sums are printed, so the fix can be
    verified from the log (fixed: unbounded logits, row sums != 1);
  * the final last-epoch summary cannot crash the run: train_pooled.report is called with
    integer support (sklearn returns float support when no clip is predicted correctly)
    and falls back to ttg_common.cls_report, so results.json is always written.

USAGE (from the video-eeg-ensembling repo, any cwd; PYTHONDONTWRITEBYTECODE=1, eeg conda env)
  EEG_ROOT (env var, default /work/mech-ai-scratch/alloy/EEG) holds data/, data_full/, the
  frame caches, the imported train_pooled / train_classifier and every run output. The script
  puts EEG_ROOT on sys.path and chdirs there, so relative paths (--cache_dir, --output) are
  resolved against EEG_ROOT.
  python grader/train_grader.py --arch x3d --fix_x3d --heads dual --seed 1 \
      --split session --output /work/mech-ai-scratch/alloy/EEG/output/ttg_stage1/x3dfix_dual_s1
  CPU smoke tests (results meaningless):
    --dry_run --allow_cpu --only_cached --cache_dir output/ttg_tmp/<test cache>
        one forward+backward on 2 train clips and a forward on 2 val clips; writes nothing
    --allow_cpu --only_cached --limit_train 8 --limit_val 4 --epochs 2 --batch_size 2
  SLURM: grader/sbatch_grader.sh with a config table (grader/stage1.tsv).

EXIT CODES  0 finished (results.json written) | 3 stopped by signal after a resumable
            checkpoint | 4 GPU architecture not supported by this torch build.
"""
import argparse
import json
import os
import random
import re
import signal
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ttg_common as C                                               # noqa: E402  (puts EEG_ROOT on sys.path)

os.environ.setdefault("TORCH_HOME", "/work/mech-ai-scratch/alloy/.cache/torch")
import torch                                                          # noqa: E402
import torch.nn as nn                                                 # noqa: E402
from torch.utils.data import DataLoader, Dataset, Sampler             # noqa: E402

import train_pooled as tp                                             # noqa: E402
from train_classifier import KINETICS_MEAN, KINETICS_STD              # noqa: E402

ARCH_GEOM = {"x3d": (16, 224), "slowfast": (32, 224)}
DEFAULT_CACHE = {"x3d": "cache_frames/f16s224", "slowfast": "cache_frames/f32s224"}
BUGGED_FLOOR = {"g3": 0.59, "g5": 1.03}       # epoch-12 train loss of the stored bugged X3D
EXIT_STOPPED, EXIT_BAD_GPU = 3, 4
STOP = {"flag": False, "sig": None}


def _on_signal(sig, _frm):
    if not STOP["flag"]:
        print(f"[{time.strftime('%F %T')}] signal {signal.Signals(sig).name}: checkpoint at the "
              f"next step boundary, then exit {EXIT_STOPPED}", flush=True)
    STOP["flag"], STOP["sig"] = True, int(sig)


def _worker_init(_wid):
    # SLURM signals every process of the job. A worker dying on SIGTERM makes the
    # DataLoader raise in the main thread at an arbitrary point (possibly mid-optimizer
    # step); ignoring it lets the main process finish the step and checkpoint cleanly.
    for s in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
        signal.signal(s, signal.SIG_IGN)


# ----------------------------------------------------------------------------- model

class GraderNet(nn.Module):
    """Pretrained Kinetics backbone -> pooled features -> one Linear per task."""

    def __init__(s, arch, tasks, fix_x3d, frames, size, pretrained=True, logit_bound=0.0):
        super().__init__()
        s.logit_bound = float(logit_bound)   # >0: logits = B*tanh(z/B) (Stage 1b bounded head)
        if arch == "x3d":
            from pytorchvideo.models.hub import x3d_m
            net = x3d_m(pretrained=pretrained)
            head = net.blocks[-1]
            if not isinstance(getattr(head, "activation", None), nn.Softmax):
                raise SystemExit("pytorchvideo x3d_m head no longer has activation=Softmax; "
                                 "re-verify the bug before training")
            dim = head.proj.in_features
            head.proj = nn.Identity()
            head.activation = None          # the feature path never sees a softmax
            if (frames, size) != (16, 224):
                # the Kinetics head pools with a fixed AvgPool3d(16,7,7): exact global
                # average at 16x224, a crash (or partial pool) at any other geometry
                head.pool.pool = nn.AdaptiveAvgPool3d(1)
            s.backbone = net
            s.head_softmax_bug = not fix_x3d
        elif arch == "slowfast":
            from pytorchvideo.models.hub import slowfast_r50
            net = slowfast_r50(pretrained=pretrained)
            head = net.blocks[-1]
            if getattr(head, "activation", None) is not None:
                raise SystemExit("slowfast head unexpectedly has an activation")
            dim = head.proj.in_features
            head.proj = nn.Identity()
            s.backbone = tp._SlowFastWrap(net)
            s.head_softmax_bug = False
        else:
            raise SystemExit(f"unknown arch {arch}")
        s.feat_dim = dim
        s.heads = nn.ModuleDict({t: nn.Linear(dim, C.NCLS[t]) for t in tasks})

    def forward(s, x):
        f = s.backbone(x)
        if f.dim() != 2:
            f = f.flatten(1)
        out = {}
        for t, h in s.heads.items():
            o = h(f)
            if s.logit_bound > 0:
                o = s.logit_bound * torch.tanh(o / s.logit_bound)
            out[t] = torch.softmax(o, dim=1) if s.head_softmax_bug else o
        return out


# ----------------------------------------------------------------------------- data

def augment_clip(a, rng, S):
    """Clip-consistent augmentation on uint8 (C,T,H,W): horizontal flip p=0.5, square
    random resized crop with area fraction U(0.8,1.0) at a uniform position (bilinear back
    to SxS), and brightness / contrast factors U(0.9,1.1) returned for the GPU."""
    flip = rng.random() < 0.5
    area = rng.uniform(0.8, 1.0)
    side = int(round(S * np.sqrt(area)))
    y0 = int(rng.integers(0, S - side + 1))
    x0 = int(rng.integers(0, S - side + 1))
    jit = np.array([rng.uniform(0.9, 1.1), rng.uniform(0.9, 1.1)], np.float32)
    v = a.transpose(1, 2, 3, 0)[:, y0:y0 + side, x0:x0 + side]
    if flip:
        v = v[:, :, ::-1]
    if side != S:
        import cv2
        v = np.stack([cv2.resize(np.ascontiguousarray(f), (S, S), interpolation=cv2.INTER_LINEAR)
                      for f in v])
    return np.ascontiguousarray(v.transpose(3, 0, 1, 2)), jit


class ClipSet(Dataset):
    """Keys are (row, epoch); epoch 0 = evaluation (never augmented)."""

    def __init__(s, items, cache_dir, frames, size, augment, seed):
        s.items, s.cache_dir, s.frames, s.size = items, cache_dir, frames, size
        s.augment, s.seed = augment, seed

    def __len__(s):
        return len(s.items)

    def __getitem__(s, key):
        i, epoch = key
        p, y5 = s.items[i][0], s.items[i][1]
        a = tp.FrameCache.get(s.cache_dir, s.frames, s.size).fetch(p)
        if a is None:
            raise RuntimeError(f"{p} is not (readably) in {s.cache_dir}; refusing to fill")
        jit = np.ones(2, np.float32)
        if s.augment and epoch > 0:
            a, jit = augment_clip(a, np.random.default_rng([s.seed, epoch, i]), s.size)
        return torch.from_numpy(np.ascontiguousarray(a)), int(y5), int(i), torch.from_numpy(jit)


class EpochBatches(Sampler):
    """Deterministic per-(seed, epoch) permutation, drop_last, restartable at a step."""

    def __init__(s, n, bs, seed):
        s.n, s.bs, s.seed, s.epoch, s.start = n, bs, seed, 1, 0

    def order(s, epoch):
        g = torch.Generator()
        g.manual_seed(int(s.seed) * 1_000_003 + int(epoch))
        return torch.randperm(s.n, generator=g).tolist()

    def n_batches(s):
        return s.n // s.bs

    def __iter__(s):
        perm = s.order(s.epoch)
        for b in range(s.start, s.n_batches()):
            yield [(perm[k], s.epoch) for k in range(b * s.bs, (b + 1) * s.bs)]

    def __len__(s):
        return max(s.n_batches() - s.start, 0)


class ValBatches(Sampler):
    def __init__(s, n, bs):
        s.n, s.bs = n, bs

    def __iter__(s):
        for b in range(0, s.n, s.bs):
            yield [(k, 0) for k in range(b, min(b + s.bs, s.n))]

    def __len__(s):
        return (s.n + s.bs - 1) // s.bs


_AUGN = {}


def to_input(x, jit, dev, augment):
    """uint8 (B,C,T,H,W) -> normalised float. Without --augment this IS tp.norm_batch
    (bit-identical to train_pooled); with it, brightness then contrast (around the clip's
    mean intensity) are applied in [0,1] before the Kinetics normalisation."""
    x = x.to(dev, non_blocking=True)
    if not augment:
        return tp.norm_batch(x, dev)
    if dev not in _AUGN:
        _AUGN[dev] = (torch.as_tensor(KINETICS_MEAN, device=dev).view(1, 3, 1, 1, 1),
                      torch.as_tensor(KINETICS_STD, device=dev).view(1, 3, 1, 1, 1))
    mean, std = _AUGN[dev]
    jit = jit.to(dev, non_blocking=True)
    x = x.float().div_(255.0)
    x.mul_(jit[:, 0].view(-1, 1, 1, 1, 1))
    m = x.mean(dim=(1, 2, 3, 4), keepdim=True)
    x.sub_(m).mul_(jit[:, 1].view(-1, 1, 1, 1, 1)).add_(m).clamp_(0.0, 1.0)
    return x.sub_(mean).div_(std)


# ----------------------------------------------------------------------------- state io

def rng_state():
    st = dict(py=random.getstate(), np=np.random.get_state(), torch=torch.get_rng_state())
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st):
    random.setstate(st["py"])
    np.random.set_state(st["np"])
    torch.set_rng_state(st["torch"])
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


def _hold_run_lock(out):
    """Exclusive flock on <out>/.run.lock for the life of the process, so two launches of
    the same run (e.g. an array submitted twice) can never train into one directory."""
    import fcntl
    fh = open(os.path.join(out, ".run.lock"), "a+")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another process is training into {out} (.run.lock held); refusing")
    except OSError as e:                       # filesystem without lock support: warn only
        print(f"warning: could not lock {out}/.run.lock ({e}); continuing", flush=True)
    fh.seek(0)
    fh.truncate()
    fh.write(f"{os.uname().nodename} pid {os.getpid()} job {os.environ.get('SLURM_JOB_ID')}\n")
    fh.flush()
    _LOCK_FH.append(fh)


def pick_device(a):
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        ok, msg = C.check_cuda_arch()
        print(msg, flush=True)
        if not ok:
            print("FATAL: this torch build has no kernels for this GPU; exclude the node "
                  "(--constraint='a100|h200|l40s')", flush=True)
            sys.exit(EXIT_BAD_GPU)
        (torch.ones(8, device=dev) * 2).sum().item()          # CUDA really initialises
        return dev
    if not (a.allow_cpu or a.dry_run):
        raise SystemExit("CUDA is not available -- refusing to fall back to CPU "
                         "(pass --allow_cpu only for smoke tests)")
    return torch.device("cpu")


# ----------------------------------------------------------------------------- metrics

def head_probs(model, out_t):
    o = out_t.float()
    if model.head_softmax_bug:            # already a softmax; renormalise fp rounding only
        return o / o.sum(1, keepdim=True)
    return torch.softmax(o, 1)


def final_report(task, y, pred, prob, tag):
    """train_pooled.report() for one head, made total so results.json is always written.

    train_pooled.report prints the per-class support with a 'd' format. When NO val clip is
    predicted correctly, sklearn 1.5.2 precision_recall_fscore_support takes its
    'pathological case' branch and returns the support as float64, and the print raises
    ValueError. That is easy to hit in --limit smoke runs and a resumed run would then crash
    again on every restart. Here the support is cast back to int64 (it is an exact count)
    for the duration of the call, and should report() still raise, the numbers come from
    ttg_common.cls_report instead. tp.NAMES / tp.NC / the sklearn symbol are restored
    afterwards. Returns (report dict, source string)."""
    saved = (tp.NAMES, tp.NC, tp.precision_recall_fscore_support)
    prfs = saved[2]

    def prfs_int_support(*x, **k):
        pr, rc, f1, sup = prfs(*x, **k)
        return pr, rc, f1, (None if sup is None else np.asarray(sup).astype(np.int64))

    tp.NAMES, tp.NC = (tp.NAMES3, 3) if task == "g3" else (tp.NAMES5, 5)
    tp.precision_recall_fscore_support = prfs_int_support
    try:
        return tp.report(y, pred, prob, tag), "train_pooled.report"
    except Exception as e:                                   # never lose results.json
        print(f"   train_pooled.report raised {type(e).__name__}: {e}; "
              f"falling back to ttg_common.cls_report", flush=True)
        return C.cls_report(y, pred, task), f"ttg_common.cls_report ({type(e).__name__} in report)"
    finally:
        tp.NAMES, tp.NC, tp.precision_recall_fscore_support = saved


def val_metrics(y5, P, task, sess, animals):
    y = y5 if task == "g5" else C.GROUP3[y5]
    rep = C.cls_report(y, P.argmax(1), task)
    sz = np.isin(y, C.SEV[task] + C.MILD[task])
    sc = C.severity_logit(P, task)
    is_sev = np.isin(y, C.SEV[task])
    pooled = C.auroc(sc[sz], is_sev[sz]) if sz.any() else float("nan")
    within = C.within_auroc(sc[sz], is_sev[sz], sess[sz], animals[sz]) if sz.any() else float("nan")
    return dict(macro_f1=rep["macro_f1"], mcc=rep["mcc"],
                recall={k: [round(v["recall"], 4), v["n_correct"], v["n"]]
                        for k, v in rep["per_class"].items()},
                sev_vs_mild_auroc_pooled=pooled, sev_vs_mild_auroc_within_session=within)


# ----------------------------------------------------------------------------- main

def parse():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", choices=["x3d", "slowfast"], default="x3d")
    ap.add_argument("--fix_x3d", action="store_true",
                    help="remove the X3D head softmax so CE sees real logits")
    ap.add_argument("--heads", choices=["dual", "g3", "g5"], default="dual")
    ap.add_argument("--w_g3", type=float, default=1.0, help="dual: weight of CE_g3")
    ap.add_argument("--w_g5", type=float, default=1.0, help="dual: weight of CE_g5")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--split", choices=["session", "subject"], default="session")
    ap.add_argument("--split_seed", type=int, default=49)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4, help="v3_vidseeds X3D value")
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--cache_dir", default=None,
                    help="frame cache, relative to EEG_ROOT or absolute (default "
                         "cache_frames/f16s224 for x3d, f32s224 for slowfast)")
    ap.add_argument("--logit_bound", type=float, default=0.0,
                    help="Stage 1b: bound head logits as B*tanh(z/B); 0 = off (default)")
    ap.add_argument("--augment", action="store_true",
                    help="flip / random-resized-crop 0.8-1.0 / brightness-contrast +-0.1 (OFF by default)")
    ap.add_argument("--output", default=None,
                    help="absolute run dir under /work/mech-ai-scratch/alloy/EEG/output/ttg_*")
    ap.add_argument("--ckpt_every_min", type=float, default=20.0,
                    help="also write a mid-epoch last.pt this often (bounds loss on SIGKILL)")
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--deterministic", action="store_true",
                    help="cudnn.deterministic + deterministic algorithms (slower; opt-in as in train_pooled)")
    ap.add_argument("--items", choices=["discover", "f32index"], default="discover",
                    help="item source: train_pooled.discover() (default) or the f32s224 index "
                         "(same set, ~100x faster to list; smoke tests)")
    ap.add_argument("--allow_cpu", action="store_true", help="smoke tests only")
    ap.add_argument("--dry_run", action="store_true",
                    help="one forward+backward on 2 train clips, one eval forward on 2 val "
                         "clips; prints output range / losses; writes nothing")
    ap.add_argument("--only_cached", action="store_true",
                    help="smoke tests: restrict train/val to clips present in --cache_dir")
    ap.add_argument("--limit_train", type=int, default=0, help="smoke tests: seeded subsample")
    ap.add_argument("--limit_val", type=int, default=0, help="smoke tests: seeded subsample")
    ap.add_argument("--allow_val_mismatch", action="store_true")
    ap.add_argument("--no_pretrained", action="store_true", help="tests only")
    ap.add_argument("--debug_sigterm_at", default=None,
                    help="tests: 'E:S' raise SIGTERM in-process after step S of epoch E")
    ap.add_argument("--debug_sigterm_in_val", type=int, default=0,
                    help="tests: raise SIGTERM in-process before validating epoch E")
    ap.add_argument("--debug_stop_after_epoch", type=int, default=0,
                    help="tests: exit 3 right after epoch E is fully checkpointed")
    return ap.parse_args()


def main():
    a = parse()
    C.enter_eeg_root()                                   # relative paths below are EEG_ROOT's
    tasks = ["g3", "g5"] if a.heads == "dual" else [a.heads]
    tw = {"g3": a.w_g3 if a.heads == "dual" else 1.0, "g5": a.w_g5 if a.heads == "dual" else 1.0}
    frames, size = ARCH_GEOM[a.arch]
    cache_dir = a.cache_dir or DEFAULT_CACHE[a.arch]
    if a.arch != "x3d" and a.fix_x3d:
        print("note: --fix_x3d is a no-op for slowfast (its head has no activation)")
    out = None if a.dry_run else C.check_output_dir(a.output or "")

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    if a.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    dev = pick_device(a)

    # ---- items and split (5-class labels; split membership as train_pooled)
    t0 = time.time()
    items = tp.discover() if a.items == "discover" else C.items_from_f32_index()
    items = sorted(items, key=lambda it: it[0])          # canonical order (see docstring)
    print(f"{len(items)} clips from {a.items} in {time.time() - t0:.0f}s", flush=True)
    assert tp.NC == 5, "train_pooled.NC must be the 5-class default when splitting"
    if a.split == "subject":
        tr, va, valgroups, sseed = tp.split_subjects(items, a.split_seed, a.fold, a.n_folds)
    else:
        tr, va, valgroups, sseed = tp.split_sessions(items, a.split_seed)
    print(f"split={a.split} seed={sseed} fold={a.fold}: train {len(tr)}  val {len(va)}  "
          f"({len(valgroups)} val groups)", flush=True)
    if a.split == "session" and a.split_seed == 49 and len(va) != 5289 and not a.allow_val_mismatch:
        raise SystemExit(f"expected the 5,289-clip seed-49 val set, got {len(va)}")

    meta = json.load(open(os.path.join(cache_dir, "index.json")))
    if meta["frames"] != frames or meta["size"] != size:
        raise SystemExit(f"{cache_dir} is {meta['frames']}x{meta['size']}, {a.arch} needs {frames}x{size}")
    cached = set(meta["paths"]) - set(meta.get("unreadable", []))
    if a.only_cached:
        tr = [i for i in tr if i[0] in cached]
        va = [i for i in va if i[0] in cached]
        print(f"*** --only_cached: train {len(tr)}  val {len(va)} (smoke test) ***")
    if a.limit_train or a.limit_val:
        rng = random.Random(12345)
        if a.limit_train:
            tr = sorted(rng.sample(tr, min(a.limit_train, len(tr))))
        if a.limit_val:
            va = sorted(rng.sample(va, min(a.limit_val, len(va))))
        print(f"*** SMOKE SUBSAMPLE: train {len(tr)}  val {len(va)} -- results are "
              f"meaningless, do not report them ***", flush=True)
    miss = [i[0] for i in tr + va if i[0] not in cached]
    if miss:
        raise SystemExit(f"{len(miss)} train/val clips are not readable in {cache_dir} "
                         f"(e.g. {miss[0]}); refusing to decode placeholders or fill. "
                         f"Build the cache with grader/build_f16_cache.py.")
    if len(tr) < a.batch_size:
        raise SystemExit(f"train set ({len(tr)}) smaller than one batch ({a.batch_size})")

    y5tr = np.array([i[1] for i in tr])
    cw = {}
    for t in tasks:
        yt = y5tr if t == "g5" else C.GROUP3[y5tr]
        cnt = np.bincount(yt, minlength=C.NCLS[t])
        w = 1.0 / np.maximum(cnt, 1)
        cw[t] = (w / w.mean()).astype(np.float32)
        yv = np.array([i[1] for i in va])
        yv = yv if t == "g5" else C.GROUP3[yv]
        cv = np.bincount(yv, minlength=C.NCLS[t])
        print(f"[{t}] " + "  ".join(f"{C.NAMES[t][k]}: train {cnt[k]} val {cv[k]} w={cw[t][k]:.2f}"
                                    for k in range(C.NCLS[t])), flush=True)

    ds_tr = ClipSet(tr, cache_dir, frames, size, a.augment, a.seed)
    ds_va = ClipSet(va, cache_dir, frames, size, False, a.seed)
    va_paths = np.array([i[0] for i in va])
    va_y5 = np.array([i[1] for i in va])
    va_sess = np.array([i[2] for i in va])
    va_anim = np.array([i[3] for i in va])

    model = GraderNet(a.arch, tasks, a.fix_x3d, frames, size, not a.no_pretrained,
                      logit_bound=a.logit_bound).to(dev)
    print(f"arch={a.arch} frames={frames} size={size} heads={tasks} feat_dim={model.feat_dim} "
          f"head_softmax_bug={model.head_softmax_bug} params="
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev.type == "cuda"))
    crit = {t: nn.CrossEntropyLoss(weight=torch.tensor(cw[t], device=dev)) for t in tasks}
    lut = torch.tensor(C.GROUP3, device=dev)

    def step_fn(x, y5, jit):
        x = to_input(x, jit, dev, a.augment)
        y5 = y5.to(dev, non_blocking=True)
        ys = {"g5": y5, "g3": lut[y5]}
        opt.zero_grad(set_to_none=True)
        with tp._autocast(dev):
            o = model(x)
            ls = {t: crit[t](o[t], ys[t]) for t in tasks}
            loss = sum(tw[t] * ls[t] for t in tasks)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        return {t: float(ls[t].item()) for t in tasks}, float(loss.item()), o

    def diag(o):
        d = {}
        for t in tasks:
            v = o[t].detach().float()
            d[t] = dict(min=round(float(v.min()), 4), max=round(float(v.max()), 4),
                        row_sum_mean=round(float(v.sum(1).mean()), 4),
                        bounded_like_probs=bool(v.min() >= 0 and v.max() <= 1 and
                                                torch.allclose(v.sum(1), torch.ones(len(v), device=v.device), atol=1e-3)))
        return d

    if a.dry_run:
        nb = min(2, len(tr))
        xb = torch.stack([ds_tr[(k, 1)][0] for k in range(nb)])
        yb = torch.tensor([ds_tr[(k, 1)][1] for k in range(nb)])
        jb = torch.stack([ds_tr[(k, 1)][3] for k in range(nb)])
        model.train()
        t1 = time.time()
        ls, tot, o = step_fn(xb, yb, jb)
        gn = sum(float(p.grad.norm() ** 2) for p in model.parameters() if p.grad is not None) ** 0.5
        print(f"DRY RUN train step on {nb} clips {tuple(xb.shape)} in {time.time() - t1:.1f}s: "
              f"loss {ls} total {tot:.4f} grad-norm {gn:.3f}", flush=True)
        print(f"   train-mode head outputs: {diag(o)}", flush=True)
        if model.head_softmax_bug:
            print("   (bugged control: outputs are probabilities -- CE on them has a floor)")
        else:
            print("   (no head softmax: outputs are unbounded logits)")
        model.eval()
        nv = min(2, len(va))
        with torch.no_grad(), tp._autocast(dev):
            ov = model(tp.norm_batch(torch.stack([ds_va[(k, 0)][0] for k in range(nv)]).to(dev), dev))
        print(f"   eval probs: { {t: head_probs(model, ov[t]).cpu().numpy().round(3).tolist() for t in tasks} }")
        print("DRY RUN complete; nothing written.")
        return

    # ---- run directory, config, resume
    os.makedirs(out, exist_ok=True)
    _hold_run_lock(out)
    run_key = dict(arch=a.arch, fix_x3d=bool(a.fix_x3d), heads=a.heads, w_g3=a.w_g3, w_g5=a.w_g5,
                   seed=a.seed, split=a.split, split_seed=a.split_seed, fold=a.fold,
                   n_folds=a.n_folds, lr=a.lr, weight_decay=a.weight_decay,
                   batch_size=a.batch_size, epochs=a.epochs, augment=bool(a.augment),
                   cache_dir=os.path.realpath(cache_dir), only_cached=bool(a.only_cached),
                   limit_train=a.limit_train, limit_val=a.limit_val,
                   pretrained=not a.no_pretrained,
                   train_fp=C.fingerprint([i[0] for i in tr]),
                   val_fp=C.fingerprint([i[0] for i in va]))
    last_path = os.path.join(out, "last.pt")
    state = dict(next_epoch=1, next_step=0, train_done=False,
                 accum=dict(n=0, steps=0, loss={t: 0.0 for t in tasks}, total=0.0),
                 history=[], best={t: dict(macro_f1=-1.0, epoch=None) for t in tasks},
                 first_batch_diag=None, resumes=[])
    if os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        if ck["run_key"] != run_key:
            diff = {k: (ck["run_key"].get(k), v) for k, v in run_key.items() if ck["run_key"].get(k) != v}
            raise SystemExit(f"{last_path} belongs to a different configuration: {diff}")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        state = ck["state"]
        set_rng_state(ck["rng"])
        state["resumes"].append(dict(at=time.strftime("%F %T"), epoch=state["next_epoch"],
                                     step=state["next_step"], train_done=state["train_done"],
                                     job=os.environ.get("SLURM_JOB_ID")))
        print(f"RESUME from {last_path}: epoch {state['next_epoch']} step {state['next_step']} "
              f"train_done={state['train_done']}", flush=True)
    else:
        C.atomic_json(os.path.join(out, "config.json"), dict(
            run_key=run_key, argv=sys.argv, started=time.strftime("%F %T"),
            n_train=len(tr), n_val=len(va), class_weights={t: cw[t].tolist() for t in tasks},
            val_groups=sorted(valgroups), feat_dim=model.feat_dim,
            head_softmax_bug=model.head_softmax_bug,
            host=os.uname().nodename, job=os.environ.get("SLURM_JOB_ID")))

    def save_last():
        atomic_torch_save(dict(model=model.state_dict(), opt=opt.state_dict(),
                               sched=sched.state_dict(), scaler=scaler.state_dict(),
                               state=state, rng=rng_state(), run_key=run_key,
                               saved_at=time.strftime("%F %T")), last_path)

    def stop_now(where, saved=False):
        if not saved:
            save_last()
        print(f"[{time.strftime('%F %T')}] checkpointed ({where}: epoch {state['next_epoch']} "
              f"step {state['next_step']} train_done={state['train_done']}); exit {EXIT_STOPPED}",
              flush=True)
        sys.exit(EXIT_STOPPED)

    for s in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
        signal.signal(s, _on_signal)
    dbg = tuple(int(v) for v in a.debug_sigterm_at.split(":")) if a.debug_sigterm_at else None

    sampler = EpochBatches(len(tr), a.batch_size, a.seed)
    gen = torch.Generator()
    gen.manual_seed(0)
    dl_kw = dict(num_workers=a.workers, pin_memory=(dev.type == "cuda"),
                 worker_init_fn=_worker_init, generator=gen)
    if a.workers > 0:
        dl_kw["prefetch_factor"] = 2
    dl_tr = DataLoader(ds_tr, batch_sampler=sampler, **dl_kw)
    dl_va = DataLoader(ds_va, batch_sampler=ValBatches(len(va), a.batch_size), **dl_kw)

    if state["next_epoch"] > a.epochs:
        print("all epochs already done")
    for ep in range(state["next_epoch"], a.epochs + 1):
        te = time.time()
        acc = state["accum"]
        if not state["train_done"]:
            sampler.epoch, sampler.start = ep, state["next_step"]
            lr_ep = opt.param_groups[0]["lr"]
            model.train()
            it = iter(dl_tr)
            step = state["next_step"]
            last_ck = time.time()
            while True:
                if STOP["flag"]:
                    state["next_step"] = step
                    save_last()             # checkpoint FIRST: KillWait is 30 s, and the
                    print(f"[{time.strftime('%F %T')}] last.pt saved (epoch {ep} step {step}); "
                          f"stopping loader workers", flush=True)
                    del it                  # worker shutdown below can take seconds
                    stop_now("mid-epoch", saved=True)
                try:
                    x, y5, _, jit = next(it)
                except StopIteration:
                    break
                ls, tot, o = step_fn(x, y5, jit)
                if state["first_batch_diag"] is None:
                    state["first_batch_diag"] = diag(o)
                    print(f"first batch head outputs: {state['first_batch_diag']}", flush=True)
                if not np.isfinite(tot):
                    raise SystemExit(f"non-finite loss at epoch {ep} step {step}")
                step += 1
                b = len(y5)
                acc["n"] += b
                acc["steps"] += 1
                acc["total"] += tot * b
                for t in tasks:
                    acc["loss"][t] += ls[t] * b
                if a.log_every and step % a.log_every == 0:
                    print(f"  ep{ep} step {step}/{sampler.n_batches()}  " +
                          "  ".join(f"{t}={acc['loss'][t] / acc['n']:.4f}" for t in tasks) +
                          f"  {(time.time() - te) / max(acc['steps'], 1):.3f}s/step", flush=True)
                if dbg and dbg == (ep, step):
                    os.kill(os.getpid(), signal.SIGTERM)
                if time.time() - last_ck > 60 * a.ckpt_every_min:
                    state["next_step"] = step
                    save_last()
                    last_ck = time.time()
            del it
            sched.step()
            state["train_done"], state["next_step"] = True, 0
            state["lr_epoch"] = lr_ep
        # ---- validation (deterministic: eval mode, fixed order, no augmentation)
        model.eval()
        Y, I, PP = [], [], {t: [] for t in tasks}
        if a.debug_sigterm_in_val == ep:
            os.kill(os.getpid(), signal.SIGTERM)
        with torch.no_grad():
            for x, y5, i, _ in dl_va:
                if STOP["flag"]:
                    stop_now("during validation")
                x = tp.norm_batch(x.to(dev, non_blocking=True), dev)
                with tp._autocast(dev):
                    o = model(x)
                for t in tasks:
                    PP[t].append(head_probs(model, o[t]).cpu().numpy())
                Y.append(y5.numpy())
                I.append(i.numpy())
        I = np.concatenate(I)
        Y = np.concatenate(Y)
        assert np.array_equal(I, np.arange(len(va))) and np.array_equal(Y, va_y5)
        P = {t: np.concatenate(PP[t]).astype(np.float32) for t in tasks}
        dump = dict(path=va_paths, y5=va_y5, epoch=np.int64(ep), seed=np.int64(a.seed),
                    arch=np.array(a.arch), heads=np.array(a.heads), split=np.array(a.split),
                    fold=np.int64(a.fold), fix_x3d=np.bool_(a.fix_x3d),
                    head_softmax_bug=np.bool_(model.head_softmax_bug))
        for t in tasks:
            dump[f"probs_{t}"] = P[t]
        C.atomic_npz(os.path.join(out, f"val_ep{ep:02d}.npz"), **dump)
        n = max(acc["n"], 1)
        h = dict(epoch=ep, lr=state.get("lr_epoch"), n_train=acc["n"], steps=acc["steps"],
                 train_loss={t: acc["loss"][t] / n for t in tasks}, train_loss_total=acc["total"] / n,
                 val={t: val_metrics(va_y5, P[t], t, va_sess, va_anim) for t in tasks},
                 secs=round(time.time() - te, 1))
        state["history"] = [r for r in state["history"] if r["epoch"] != ep] + [h]
        print(f"ep{ep:>2} train loss " +
              "  ".join(f"{t}={h['train_loss'][t]:.4f} (bugged floor ~{BUGGED_FLOOR[t]})" for t in tasks) +
              " | val " + "  ".join(
                  f"{t}: macroF1={h['val'][t]['macro_f1']:.4f} within-sess sevAUROC="
                  f"{h['val'][t]['sev_vs_mild_auroc_within_session']:.4f} recall="
                  + "/".join(f"{v[1]}of{v[2]}" for v in h['val'][t]['recall'].values())
                  for t in tasks) + f" | {h['secs']}s", flush=True)
        C.atomic_json(os.path.join(out, "history.json"), state["history"])
        for t in tasks:
            mf1 = h["val"][t]["macro_f1"]
            if mf1 > state["best"][t]["macro_f1"]:     # historical rule: strict improvement
                state["best"][t] = dict(macro_f1=mf1, epoch=ep)
                atomic_torch_save(dict(model=model.state_dict(), epoch=ep, head=t, macro_f1=mf1,
                                       run_key=run_key, note="best val macro-F1 on the scored "
                                       "val set: an UPPER-BOUND selection"),
                                  os.path.join(out, f"best_{t}.pt"))
                print(f"   saved best_{t}.pt (epoch {ep}, macro-F1 {mf1:.4f})", flush=True)
        state.update(next_epoch=ep + 1, next_step=0, train_done=False,
                     accum=dict(n=0, steps=0, loss={t: 0.0 for t in tasks}, total=0.0))
        save_last()
        if STOP["flag"]:
            stop_now("after epoch end")
        if a.debug_stop_after_epoch and ep == a.debug_stop_after_epoch and ep < a.epochs:
            print(f"--debug_stop_after_epoch {ep}: exiting as if preempted", flush=True)
            sys.exit(EXIT_STOPPED)

    # ---- final summary (last epoch = the pre-registered number)
    res = dict(run_key=run_key, complete=True, finished=time.strftime("%F %T"),
               epochs=a.epochs, resumes=state["resumes"], first_batch_diag=state["first_batch_diag"],
               heads={})
    zl = np.load(os.path.join(out, f"val_ep{a.epochs:02d}.npz"))
    last_h = [r for r in state["history"] if r["epoch"] == a.epochs][0]
    for t in tasks:
        Pt = zl[f"probs_{t}"]
        yt = zl["y5"] if t == "g5" else C.GROUP3[zl["y5"]]
        rep, rep_src = final_report(t, yt, Pt.argmax(1), Pt, f"{a.arch} fix={a.fix_x3d} head {t} "
                                    f"-- LAST epoch {a.epochs} (pre-registered)")
        floor = BUGGED_FLOOR[t]
        res["heads"][t] = dict(
            last_epoch=dict(report=rep, report_source=rep_src, **last_h["val"][t]),
            best_epoch_upper_bound=state["best"][t],
            train_loss_last=last_h["train_loss"][t],
            train_loss_below_half_bugged_floor=bool(last_h["train_loss"][t] < 0.5 * floor),
            bugged_floor=floor)
    C.atomic_json(os.path.join(out, "results.json"), res)
    print(f"\nwrote {out}/results.json", flush=True)


if __name__ == "__main__":
    main()
