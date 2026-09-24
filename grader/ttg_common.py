"""
Shared helpers for the Two-Timescale Grader (TTG) scripts in video-eeg-ensembling/grader/.

PURPOSE
  One place for the hazards every TTG script has to get right, so each deliverable
  handles them identically:

  * EEG_ROOT, imports and cwd. The code lives in this repo (grader/); the data, caches,
    run outputs and the imported base trainers (train_pooled, train_classifier,
    train_pooled_eeg, preds_io, frame_cache, ...) live in the parent EEG repo, EEG_ROOT
    (env var EEG_ROOT, default /work/mech-ai-scratch/alloy/EEG). EEG_ROOT is put on
    sys.path explicitly, so `import train_pooled` never depends on the cwd.
    train_pooled.discover() and every cache key still walk RELATIVE data/... paths, so
    enter_eeg_root() chdirs to EEG_ROOT (and checks data/ and data_full/ are there):
    every relative path a script uses or is given is therefore resolved against EEG_ROOT,
    exactly as when the scripts had to be started from there. The scripts can be started
    from any directory.
  * data/ mp4s are ZERO-BYTE placeholders. Real videos live at data_full/<same rel path>.
    to_full() maps a key to its source; keys stored anywhere stay 'data/...'.
  * decode_linspace() is a line-for-line replica of build_frame_cache._one /
    train_classifier.load_clip (header CAP_PROP_FRAME_COUNT -> sample_frame_indices ->
    sequential grab/retrieve -> BGR->RGB -> INTER_AREA), except that it RAISES
    DecodeError instead of returning None / filling with a mean colour, and it refuses
    nearest-frame substitution unless explicitly allowed.
  * dense-snippet geometry (snippet_starts) and frame-exact seek decoding
    (decode_seek_snippets) for probe_extract.py.
  * numpy/scipy metrics: AUROC, within-session AUROC decomposed per animal, and the
    animal-cluster bootstrap used by probe_analysis.py and stage1_eval.py.
  * atomic writes (tmp file in the same directory + os.replace).

USAGE
  Imported by the other grader scripts; not meant to be run. `python grader/ttg_common.py`
  prints the resolved EEG_ROOT, the grader directory and the dhlib directory, and exits.
"""
import json
import os
import re
import sys
import tempfile
import zlib

import numpy as np

EEG_ROOT = os.path.realpath(os.environ.get("EEG_ROOT", "/work/mech-ai-scratch/alloy/EEG"))
if EEG_ROOT not in sys.path:
    sys.path.insert(0, EEG_ROOT)                 # the parent repo's base trainers

GRADER_DIR = os.path.dirname(os.path.abspath(__file__))   # this repo's grader/ (tables, dhlib)
TTG_OUTPUT_PREFIX = os.path.join(EEG_ROOT, "output", "ttg_")
F32_INDEX = os.path.join(EEG_ROOT, "cache_frames", "f32s224", "index.json")
# dhlib.py (decision-headroom loaders) is vendored into grader/; DHLIB_DIR overrides.
DH_DIR = os.path.realpath(os.environ.get("DHLIB_DIR", GRADER_DIR))

STAGE = {"Stage_2": 1, "Stage_3": 2, "Stage_4": 3, "Stage_5": 4}
GROUP3 = np.array([0, 1, 1, 2, 2])          # g5 label -> g3 label
NAMES = {"g3": ["non-seizure", "mild(S2-3)", "severe(S4-5)"],
         "g5": ["non-seizure", "Stage2", "Stage3", "Stage4", "Stage5"]}
NCLS = {"g3": 3, "g5": 5}
SEV = {"g3": [2], "g5": [3, 4]}
MILD = {"g3": [1], "g5": [1, 2]}


class DecodeError(RuntimeError):
    """Raised for any clip that cannot be decoded exactly. Never swallowed into a fill."""


# ----------------------------------------------------------------------------- paths

def enter_eeg_root(need_videos=True):
    """chdir to EEG_ROOT: train_pooled.discover(), train_pooled_eeg.VIDEO_INDEX and every
    cache key use paths relative to EEG_ROOT, and so do the scripts' relative defaults and
    arguments. need_videos: also require data/ (keys) and data_full/ (real videos)."""
    if not os.path.isdir(EEG_ROOT):
        raise SystemExit(f"EEG_ROOT {EEG_ROOT} is not a directory (set the EEG_ROOT env var)")
    os.chdir(EEG_ROOT)
    if need_videos and (not os.path.isdir("data") or not os.path.isdir("data_full")):
        raise SystemExit(f"EEG_ROOT ({EEG_ROOT}) must contain data/ (keys) and data_full/ (real videos)")


def to_full(p):
    """'data/<rel>' key -> 'data_full/<rel>' source (relative to EEG_ROOT)."""
    p = str(p)
    if not p.startswith("data/"):
        raise ValueError(f"expected a data/... key, got {p!r}")
    return "data_full/" + p[len("data/"):]


def key_of(p):
    p = str(p)
    return p[: -len("/video.mp4")] if p.endswith("/video.mp4") else p


def animal_of(p):
    return re.search(r"Data_(RN\d+)_", str(p)).group(1)


def session_of(p):
    """'RN197/<session dir>' exactly as train_pooled.discover() builds item[2]."""
    k = key_of(p)
    parts = [c for c in k.split("/") if c]
    return f"{animal_of(p)}/{parts[-2]}"


def y5_of(p):
    b = os.path.basename(key_of(p))
    if b.startswith("seizure_"):
        m = re.search(r"Stage_[0-9]+", b)
        if not m or m.group() not in STAGE:
            return None
        return STAGE[m.group()]
    return 0


def items_from_f32_index():
    """discover()-equivalent item list rebuilt from cache_frames/f32s224/index.json
    (original microway order). Same SET as discover() on Nova (verified: 24,497 paths,
    identical set, different order), 100x faster than globbing NFS. Used only where the
    order does not matter (split membership) or for smoke tests."""
    meta = json.load(open(F32_INDEX))
    out = []
    for mp4 in meta["paths"]:
        y = y5_of(mp4)
        if y is None:
            continue
        out.append((mp4, y, session_of(mp4), animal_of(mp4)))
    return out


def fingerprint(strings):
    h = zlib.crc32(b"")
    for s in strings:
        h = zlib.crc32(s.encode() + b"\n", h)
    return f"{h:08x}:{len(strings)}"


def check_output_dir(path):
    """Outputs must be absolute and under EEG/output/ttg_*."""
    if not os.path.isabs(path):
        raise SystemExit(f"--output must be an absolute path, got {path!r}")
    rp = os.path.realpath(path)
    if not rp.startswith(TTG_OUTPUT_PREFIX):
        raise SystemExit(f"--output must live under {TTG_OUTPUT_PREFIX}*, got {rp}")
    return rp


def check_cuda_arch():
    """(ok, message). torch 2.4.1+cu121 ships sm_50..sm_90 kernels; a device whose major
    version has no kernel with minor <= its own (e.g. sm_120 rtx_pro_6000) cannot run."""
    import torch
    maj, mn = torch.cuda.get_device_capability(0)
    archs = []
    for s in torch.cuda.get_arch_list():
        m = re.match(r"sm_(\d+?)(\d)[a-z]?$", s)
        if m:
            archs.append((int(m.group(1)), int(m.group(2))))
    ok = any(M == maj and m <= mn for M, m in archs)
    name = torch.cuda.get_device_name(0)
    return ok, f"GPU {name} sm_{maj}{mn}; torch kernels {torch.cuda.get_arch_list()}"


# ----------------------------------------------------------------------------- atomic io

def _umask_mode():
    m = os.umask(0)
    os.umask(m)
    return 0o666 & ~m


def atomic_write_bytes(path, data):
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=d)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, _umask_mode())          # mkstemp is 0600; match a normal open()
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def atomic_json(path, obj):
    atomic_write_bytes(path, json.dumps(obj, indent=1, default=_json_default).encode())


def atomic_npz(path, /, **arrays):
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".npz", dir=d)
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, _umask_mode())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


# ----------------------------------------------------------------------------- decoding

def _cv2():
    import cv2
    cv2.setNumThreads(1)          # decoding workers are parallel processes already
    return cv2


def sample_frame_indices(n_total, n_sample):
    """Imported, not copied: the historical rule lives in train_classifier."""
    from train_classifier import sample_frame_indices as sfi
    return sfi(n_total, n_sample)


def decode_linspace(full, T, S, allow_substitution=False):
    """(T, S, S, 3) uint8 RGB frames at the historical linspace indices.

    Exact replica of build_frame_cache._one: n_total = header CAP_PROP_FRAME_COUNT,
    idxs = sample_frame_indices(n_total, T), one sequential grab() pass retrieving the
    target indices, BGR->RGB, INTER_AREA resize to SxS. Differences, all fail-loud:
      * zero-byte / unopenable / header count <= 0 / nothing decoded -> DecodeError;
      * a target index that the sequential pass did not reach (header count larger than
        the real stream) is filled from the nearest decoded frame ONLY when
        allow_substitution=True (the historical behaviour); otherwise DecodeError.
    Returns (frames, info) with info = {n_header, n_substituted}.
    """
    cv2 = _cv2()
    if not os.path.exists(full):
        raise DecodeError(f"missing source {full}")
    if os.path.getsize(full) == 0:
        raise DecodeError(f"zero-byte source {full}")
    cap = cv2.VideoCapture(full)
    if not cap.isOpened():
        raise DecodeError(f"cv2 cannot open {full}")
    try:
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = sample_frame_indices(n_total, T)
        if idxs is None:
            raise DecodeError(f"header frame count {n_total} for {full}")
        target = set(int(x) for x in idxs)
        grabbed, i = {}, 0
        while i < n_total and len(grabbed) < len(target):
            if not cap.grab():
                break
            if i in target:
                ok, fr = cap.retrieve()
                if ok:
                    grabbed[i] = fr
            i += 1
    finally:
        cap.release()
    if not grabbed:
        raise DecodeError(f"no frames decoded from {full}")
    keys = sorted(grabbed)
    out, n_sub = [], 0
    for ix in idxs:
        f = grabbed.get(int(ix))
        if f is None:
            n_sub += 1
            if not allow_substitution:
                raise DecodeError(f"frame {int(ix)} of {full} not decodable (header says "
                                  f"{n_total}, stream ended at {i}); refusing substitution")
            f = grabbed[min(keys, key=lambda k: abs(k - int(ix)))]
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        out.append(cv2.resize(f, (S, S), interpolation=cv2.INTER_AREA))
    return np.stack(out, axis=0), dict(n_header=n_total, n_substituted=n_sub)


def header_frame_count(full):
    cv2 = _cv2()
    if not os.path.exists(full) or os.path.getsize(full) == 0:
        raise DecodeError(f"missing or zero-byte source {full}")
    cap = cv2.VideoCapture(full)
    if not cap.isOpened():
        raise DecodeError(f"cv2 cannot open {full}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n <= 0:
        raise DecodeError(f"header frame count {n} for {full}")
    return n


def snippet_starts(n, k=8, T=16, stride=2, buffer=150):
    """First-frame index of each of k dense snippets (T frames at `stride`).

    Window W = [buffer, n - buffer) (10 s buffers at 15 fps). Snippet j is centred on the
    centre of the j-th of k equal strata of W: c_j = buffer + (j + 0.5) * |W| / k. A
    snippet spans span = (T-1)*stride + 1 native frames (31 = 2.07 s of frame-to-frame
    extent, 16 frames x 0.133 s = 2.13 s of exposure) and its middle frame sits on
    round(c_j). If |W| < 2*T frames (32) one snippet centred in the clip is used for all
    k slots. Starts are clamped to [0, n - span]. Returns (starts[k], single_flag).

    Snippets are CENTRED on their strata, not clipped to W, so near a short window the first
    or last snippet can reach up to half a span (15 frames) into a 10-s buffer. On the
    14,835 clips of temporal-sampling/features.pkl this affects 236 clips (1.6%); 4 clips use
    the single-snippet fallback. Clips shorter than one span (n <= 31) would fail the seek
    decode loudly; the shortest real clip has 276 frames.
    """
    span = (T - 1) * stride + 1
    half = span // 2
    L = n - 2 * buffer
    if L < 2 * T:
        c = np.full(k, (n - 1) / 2.0)
        single = True
    else:
        c = buffer + (np.arange(k) + 0.5) * L / k
        single = False
    st = np.round(c).astype(int) - half
    st = np.clip(st, 0, max(n - span, 0))
    return st, single


def decode_seek_snippets(full, starts, T=16, stride=2, S=224):
    """(k, T, S, S, 3) uint8 RGB: for each start, seek with CAP_PROP_POS_FRAMES, read
    span = (T-1)*stride+1 consecutive frames, keep every `stride`-th. Identical starts are
    decoded once. Any failed read raises DecodeError (no fill, no substitution)."""
    cv2 = _cv2()
    if not os.path.exists(full) or os.path.getsize(full) == 0:
        raise DecodeError(f"missing or zero-byte source {full}")
    span = (T - 1) * stride + 1
    cap = cv2.VideoCapture(full)
    if not cap.isOpened():
        raise DecodeError(f"cv2 cannot open {full}")
    cache = {}
    try:
        for s in sorted(set(int(x) for x in starts)):
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, s):
                raise DecodeError(f"seek to {s} failed in {full}")
            fr = []
            for j in range(span):
                ok, f = cap.read()
                if not ok:
                    raise DecodeError(f"read failed at frame {s + j} of {full}")
                if j % stride == 0:
                    f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                    fr.append(cv2.resize(f, (S, S), interpolation=cv2.INTER_AREA))
            cache[s] = np.stack(fr, 0)
    finally:
        cap.release()
    return np.stack([cache[int(s)] for s in starts], 0)


def decode_sequential_indices(full, idxs, S=224):
    """Reference decoder for seek verification: one sequential read() pass from frame 0,
    keeping the requested indices. Slow; used only by --verify_seek."""
    cv2 = _cv2()
    want = set(int(i) for i in idxs)
    cap = cv2.VideoCapture(full)
    got, i = {}, 0
    last = max(want)
    while i <= last:
        ok, f = cap.read()
        if not ok:
            break
        if i in want:
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            got[i] = cv2.resize(f, (S, S), interpolation=cv2.INTER_AREA)
        i += 1
    cap.release()
    missing = want - set(got)
    if missing:
        raise DecodeError(f"sequential pass missed {sorted(missing)[:5]} in {full}")
    return np.stack([got[int(i)] for i in idxs], 0)


def clip_seed(path):
    """Fixed per-clip seed (stable across runs/machines): crc32 of the data/ key."""
    return zlib.crc32(str(path).encode())


# ----------------------------------------------------------------------------- metrics

def auroc(score, pos):
    """Mann-Whitney AUROC of `score` for pos (bool) vs ~pos; ties count one half."""
    from scipy.stats import rankdata
    score = np.asarray(score, float)
    pos = np.asarray(pos, bool)
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ws_parts(score, pos, sess, animals, ua):
    """Per-animal (U, N) of within-session pairs: only (pos, neg) pairs that share a
    session count, and sessions nest in animals, so a resample's within-session AUROC is
    sum_a c_a U_a / sum_a c_a N_a."""
    from scipy.stats import rankdata
    ai = {a: i for i, a in enumerate(ua)}
    U = np.zeros(len(ua))
    N = np.zeros(len(ua))
    score = np.asarray(score, float)
    pos = np.asarray(pos, bool)
    for q in np.unique(sess):
        m = sess == q
        a, b = score[m & pos], score[m & ~pos]
        if len(a) == 0 or len(b) == 0:
            continue
        r = rankdata(np.concatenate([a, b]))
        k = ai[animals[m][0]]
        U[k] += r[:len(a)].sum() - len(a) * (len(a) + 1) / 2
        N[k] += len(a) * len(b)
    return U, N


def within_auroc(score, pos, sess, animals):
    ua = np.unique(animals)
    U, N = ws_parts(score, pos, sess, animals, ua)
    return float(U.sum() / N.sum()) if N.sum() else float("nan")


def animal_picks(animals, reps=2000, seed=0):
    ua = np.unique(animals)
    rng = np.random.default_rng(seed)
    return ua, rng.integers(0, len(ua), (reps, len(ua)))


def boot_auc(score, pos, sess, animals, ua, picks, pooled=True):
    """Animal-cluster bootstrap draws (reps, 2): [pooled AUROC, within-session AUROC].
    `ua`/`picks` must come from animal_picks() on the SAME clip set so that two scores
    bootstrapped with the same picks give paired draws."""
    U, N = ws_parts(score, pos, sess, animals, ua)
    C = np.stack([np.bincount(p, minlength=len(ua)) for p in picks]).astype(float)
    CN = C @ N
    within = np.where(CN > 0, (C @ U) / np.where(CN > 0, CN, 1), np.nan)
    if not pooled:
        return np.c_[np.full(len(picks), np.nan), within]
    groups = [np.where(animals == a)[0] for a in ua]
    pl = np.empty(len(picks))
    for r, p in enumerate(picks):
        ix = np.concatenate([groups[k] for k in p])
        pl[r] = auroc(score[ix], pos[ix])
    return np.c_[pl, within]


def ci95(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return [float("nan"), float("nan")]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def confusion(y, pred, k):
    return np.bincount(np.asarray(y) * k + np.asarray(pred), minlength=k * k).reshape(k, k)


def cls_report(y, pred, task):
    """Macro P/R/F1, MCC and per-class recall WITH counts (the CLAUDE.md convention)."""
    k = NCLS[task]
    C = confusion(y, pred, k).astype(float)
    tp = np.diag(C)
    rows, cols = C.sum(1), C.sum(0)
    rec = np.where(rows > 0, tp / np.where(rows > 0, rows, 1), 0.0)
    prec = np.where(cols > 0, tp / np.where(cols > 0, cols, 1), 0.0)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.where(prec + rec > 0, prec + rec, 1), 0.0)
    present = (rows > 0) | (cols > 0)            # sklearn macro over labels in y or pred
    s, c = C.sum(), tp.sum()
    num = c * s - (cols * rows).sum()
    den = np.sqrt((s ** 2 - (cols ** 2).sum()) * (s ** 2 - (rows ** 2).sum()))
    return dict(
        macro_f1=float(f1[present].mean()), macro_precision=float(prec[present].mean()),
        macro_recall=float(rec[present].mean()), mcc=float(num / den) if den > 0 else 0.0,
        per_class={NAMES[task][i]: dict(recall=float(rec[i]), precision=float(prec[i]),
                                        f1=float(f1[i]), n=int(rows[i]),
                                        n_correct=int(tp[i])) for i in range(k)},
        confusion=C.astype(int).tolist())


def severity_logit(P, task):
    """log(P(severe) / P(mild)) -- the score behind the 0.831 / 0.811 SlowFast
    within-session references (verified: dhlib, 5 runs, log-ratio reproduces them)."""
    ps = P[:, SEV[task]].sum(1)
    pm = P[:, MILD[task]].sum(1)
    return np.log(np.clip(ps, 1e-12, None)) - np.log(np.clip(pm, 1e-12, None))


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    print("EEG_ROOT =", EEG_ROOT)
    print("GRADER_DIR =", GRADER_DIR)
    print("DH_DIR =", DH_DIR)
