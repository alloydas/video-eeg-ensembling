#!/usr/bin/env python3
"""
Rebuild cache_frames/f16s224 (16 frames x 224 x 224, the historical X3D / MViT / S3D input)
by decoding data_full/ while keeping the data/... keys that train_pooled.FrameCache reads.

WHY
  Only cache_frames/f32s224 survives on Nova, and running build_frame_cache.py as written
  would decode the ZERO-BYTE data/ placeholders and mark every clip unreadable (or, through
  train_pooled.Clips, silently train on the Kinetics-mean colour). This builder:
    * takes the item list from train_pooled.discover() (same order),
    * decodes data_full/<rel> for each data/<rel> key,
    * uses the historical indices exactly: n = header CAP_PROP_FRAME_COUNT,
      train_classifier.sample_frame_indices(n, 16), one sequential grab()/retrieve() pass,
      BGR->RGB, INTER_AREA to 224x224 (ttg_common.decode_linspace),
    * never fills: a clip that cannot be decoded exactly (after --decode_retries re-tries,
      default 2, against transient NFS errors) is recorded as a failure; the build ABORTS
      before any index is written if any source is zero-byte/missing or more than 0.5% of
      clips fail. Up to 0.5% unreadable clips are listed in index.json 'unreadable', and
      train_grader.py refuses to start if ANY train/val clip is unreadable, so
      grader/sbatch_cache.sh exits non-zero in that case (an afterok dependency then holds),
    * writes frames.u8 under a .partial name and index.json via tmp+rename only after the
      whole build and the verification passed, so FrameCache can never see a half cache,
    * is resumable: rows already written are recorded in progress.npz, so a killed job
      continues where it stopped,
    * verifies that the path SET equals cache_frames/f32s224/index.json (full builds), and
      re-decodes --verify clips with the historical train_classifier.load_clip() on the
      data_full source and requires bit-identical frames.

  data_full decodes are NOT bit-identical to the old microway cache for some clips (a
  review found 7/12 differing by 0.1-0.4 grey levels on average); that is expected and is
  reported, not fixed. Frame ALIGNMENT is checked against f32s224 on the two indices both
  geometries always share (0 and n-1) as an informational cross-check.

LAYOUT (identical to build_frame_cache.py, readable by train_pooled.FrameCache)
  <out>/frames.u8    raw uint8, shape (N, 16, 224, 224, 3), C-order
  <out>/index.json   {"paths", "frames", "size", "n", "unreadable", + provenance keys}

USAGE (from the video-eeg-ensembling repo, any cwd; PYTHONDONTWRITEBYTECODE=1; EEG_ROOT env
       var, default /work/mech-ai-scratch/alloy/EEG. The script chdirs to EEG_ROOT: --out and
       --paths_file, when relative, are resolved against EEG_ROOT.)
  full build (CPU job, see grader/sbatch_cache.sh):
    python grader/build_f16_cache.py --out cache_frames/f16s224 --workers 30
  smoke test on 20 random clips into a temp dir:
    python grader/build_f16_cache.py --out output/ttg_tmp/f16s224_test20 --limit 20 \
        --sample_seed 0 --workers 4 --verify 20
  --dry_run lists what would be built (counts, size, first sources) and writes nothing.
"""
import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ttg_common as C                                        # noqa: E402  (puts EEG_ROOT on sys.path)

_G = {}


def _init(path, rowbytes, T, S, allow_sub, retries=2):
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _G["fd"] = os.open(path, os.O_WRONLY)
    _G.update(rowbytes=rowbytes, T=T, S=S, allow_sub=allow_sub, retries=retries)


def _one(job):
    """Decode and write one row. A failure is re-tried --decode_retries times (1 s, 2 s
    pauses) so a transient NFS / open error does not mark the clip unreadable; a clip that
    still fails is recorded, never filled."""
    row, key = job
    err = None
    for attempt in range(_G["retries"] + 1):
        try:
            arr, info = C.decode_linspace(C.to_full(key), _G["T"], _G["S"],
                                          allow_substitution=_G["allow_sub"])
            b = np.ascontiguousarray(arr).tobytes()
            assert len(b) == _G["rowbytes"]
            n = os.pwrite(_G["fd"], b, row * _G["rowbytes"])
            if n != len(b):
                raise OSError(f"short write {n}/{len(b)} at row {row}")
            info["attempts"] = attempt + 1
            return row, None, info
        except Exception as e:                   # recorded as a failure, never filled
            err = f"{type(e).__name__}: {e}"
            if attempt < _G["retries"]:
                time.sleep(1.0 + attempt)
    return row, err, None


def _paths(a):
    import train_pooled as tp
    t = time.time()
    items = tp.discover()
    print(f"discover(): {len(items)} clips in {time.time() - t:.0f}s", flush=True)
    paths = [it[0] for it in items]
    if len(set(paths)) != len(paths):
        raise SystemExit("discover() returned duplicate paths")
    f32 = set(json.load(open(C.F32_INDEX))["paths"])
    if set(paths) != f32:
        msg = (f"discover() path set != f32s224 index path set "
               f"({len(set(paths) - f32)} only in discover, {len(f32 - set(paths))} only in f32)")
        if not a.allow_pathset_mismatch:
            raise SystemExit("ABORT: " + msg)
        print("WARNING: " + msg, flush=True)
    else:
        print("path set == cache_frames/f32s224/index.json path set  (OK)", flush=True)
    if a.paths_file:
        want = [l.strip() for l in open(a.paths_file) if l.strip() and not l.startswith("#")]
        pos = {p: i for i, p in enumerate(paths)}
        unknown = [p for p in want if p not in pos]
        if unknown:
            raise SystemExit(f"{len(unknown)} --paths_file entries are not discover() clips, "
                             f"e.g. {unknown[0]}")
        paths = [paths[r] for r in sorted(pos[p] for p in set(want))]
        print(f"*** --paths_file: {len(paths)} clips (discover() order); NOT a full "
              f"training cache ***", flush=True)
    elif a.limit:
        if a.sample_seed is not None:
            rows = sorted(random.Random(a.sample_seed).sample(range(len(paths)), a.limit))
        else:
            rows = list(range(a.limit))
        paths = [paths[r] for r in rows]           # keeps discover() order
        print(f"*** --limit {a.limit}: subset of discover() order (sample_seed="
              f"{a.sample_seed}); NOT a usable training cache ***", flush=True)
    return paths


def _preflight(paths):
    bad = []
    for p in paths:
        if not p.startswith("data/"):
            bad.append((p, "key does not start with data/"))
            continue
        f = C.to_full(p)
        if not os.path.exists(f):
            bad.append((p, "data_full source missing"))
        elif os.path.getsize(f) == 0:
            bad.append((p, "data_full source is zero-byte"))
    return bad


def _f32_crosscheck(key, arr16, f32meta, f32mm):
    """Frames 0 and n-1 are sampled by both linspace(0,n-1,16) and linspace(0,n-1,32).
    Returns mean |diff| at those shared frames and to the adjacent f32 frame (alignment
    is exact when the same-index diff is much smaller than the neighbour diff)."""
    r = f32meta.get(key)
    if r is None:
        return None
    old = f32mm[r]
    same = [np.abs(arr16[0].astype(int) - old[0]).mean(),
            np.abs(arr16[-1].astype(int) - old[-1]).mean()]
    nb = [np.abs(arr16[0].astype(int) - old[1]).mean(),
          np.abs(arr16[-1].astype(int) - old[-2]).mean()]
    return dict(same_index_mad=float(np.mean(same)), neighbour_mad=float(np.mean(nb)),
                bit_identical_shared=bool(np.array_equal(arr16[0], old[0]) and
                                          np.array_equal(arr16[-1], old[-1])))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output dir, relative to EEG_ROOT or absolute; "
                    "must resolve under EEG_ROOT/cache_frames/ or EEG_ROOT/output/ttg_*")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: build only N clips")
    ap.add_argument("--sample_seed", type=int, default=None,
                    help="with --limit: random (seeded) subset instead of the first N")
    ap.add_argument("--paths_file", default=None,
                    help="build only these data/ keys (one per line; must be discover() "
                         "clips), e.g. smoke-test subsets")
    ap.add_argument("--verify", type=int, default=50,
                    help="re-decode this many random built clips with "
                         "train_classifier.load_clip and require bit-identity")
    ap.add_argument("--max_fail_frac", type=float, default=0.005)
    ap.add_argument("--decode_retries", type=int, default=2,
                    help="re-try a failed clip decode this many times before recording it")
    ap.add_argument("--allow_substitution", action="store_true",
                    help="historical nearest-frame fill when the header count exceeds the "
                         "stream (default: that clip fails). Measured 0 such clips in "
                         "14,835 checked.")
    ap.add_argument("--allow_pathset_mismatch", action="store_true")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing COMPLETE cache in --out")
    ap.add_argument("--dry_run", action="store_true", help="plan only; write nothing")
    a = ap.parse_args()

    C.enter_eeg_root()                         # relative paths below are EEG_ROOT's
    T, S = a.frames, a.size
    out = a.out
    _rp = os.path.realpath(out)                # a cache is tens of GiB: never into this repo
    if not any(_rp.startswith(os.path.join(C.EEG_ROOT, d)) for d in ("cache_frames" + os.sep, "output/ttg_")):
        raise SystemExit(f"--out must resolve under {C.EEG_ROOT}/cache_frames/ or "
                         f"{C.EEG_ROOT}/output/ttg_*, got {_rp}")
    idx_path = os.path.join(out, "index.json")
    bin_final = os.path.join(out, "frames.u8")
    bin_part = os.path.join(out, "frames.u8.partial")
    prog_path = os.path.join(out, "progress.npz")
    if os.path.exists(idx_path) and not a.overwrite:
        raise SystemExit(f"{idx_path} exists (complete cache); pass --overwrite to rebuild")

    paths = _paths(a)
    n = len(paths)
    rowbytes = T * S * S * 3
    fp = C.fingerprint(paths) + f":{T}x{S}"
    print(f"{n} clips -> ({n}, {T}, {S}, {S}, 3) uint8 = {n * rowbytes / 2**30:.1f} GiB at {out}",
          flush=True)

    bad = _preflight(paths)
    if bad:
        for p, why in bad[:10]:
            print(f"   {why}: {p}")
        raise SystemExit(f"ABORT: {len(bad)} unusable sources (zero-byte/missing/bad key); "
                         f"no cache written")
    print("preflight: every data_full source exists and is non-empty  (OK)", flush=True)
    if a.dry_run:
        print("--dry_run: first sources:")
        for p in paths[:3]:
            print("   ", p, "->", C.to_full(p))
        print("--dry_run: nothing written")
        return

    os.makedirs(out, exist_ok=True)
    done = np.zeros(n, bool)
    if os.path.exists(bin_part) and os.path.exists(prog_path):
        z = np.load(prog_path)
        if str(z["fingerprint"]) == fp and os.path.getsize(bin_part) == n * rowbytes:
            done = z["done"].astype(bool)
            print(f"RESUME: {done.sum()}/{n} rows already written", flush=True)
        else:
            print("stale partial build (different item list or size): starting over", flush=True)
            done[:] = False
    if not done.any():
        with open(bin_part, "wb") as f:
            f.truncate(n * rowbytes)

    def save_progress():
        fd = os.open(bin_part, os.O_RDONLY)
        try:
            os.fsync(fd)                 # rows reported done must be on disk first
        finally:
            os.close(fd)
        C.atomic_npz(prog_path, done=done, fingerprint=np.array(fp))

    todo = [(r, paths[r]) for r in range(n) if not done[r]]
    fails, subs, retried = {}, {}, {}
    t0 = time.time()
    last_save = time.time()
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init,
                             initargs=(bin_part, rowbytes, T, S, a.allow_substitution,
                                       a.decode_retries)) as ex:
        for k, (row, err, info) in enumerate(ex.map(_one, todo, chunksize=4)):
            if err is None:
                done[row] = True
                if info["n_substituted"]:
                    subs[paths[row]] = info["n_substituted"]
                if info["attempts"] > 1:
                    retried[paths[row]] = info["attempts"]
                    print(f"   decoded on attempt {info['attempts']}: {paths[row]}", flush=True)
            else:
                fails[paths[row]] = err
            if (k + 1) % 500 == 0 or k + 1 == len(todo):
                el = time.time() - t0
                print(f"  {k + 1}/{len(todo)}  {(k + 1) / el:.1f} clips/s  "
                      f"failed {len(fails)}  substituted {len(subs)}", flush=True)
            if time.time() - last_save > 120:
                save_progress()
                last_save = time.time()
            if len(fails) > max(a.max_fail_frac * n, 0) and len(fails) > 20:
                print("failure budget exceeded: cancelling remaining work", flush=True)
                ex.shutdown(wait=True, cancel_futures=True)
                break                               # hopeless: stop early, abort below
    save_progress()

    frac = len(fails) / n
    if fails:
        for p, e in list(fails.items())[:10]:
            print(f"   FAIL {p}: {e}")
    if frac > a.max_fail_frac:
        raise SystemExit(f"ABORT: {len(fails)}/{n} clips failed ({100 * frac:.2f}% > "
                         f"{100 * a.max_fail_frac:.2f}%). No index written; partial kept at "
                         f"{bin_part} for inspection.")
    missing = int((~done).sum()) - len(fails)
    if missing:
        raise SystemExit(f"ABORT: {missing} rows neither written nor failed (interrupted?)")

    # ---- verification against the historical decoder on the data_full source
    from train_classifier import load_clip
    mm = np.memmap(bin_part, dtype=np.uint8, mode="r", shape=(n, T, S, S, 3))
    ok_rows = [r for r in range(n) if done[r]]
    vr = sorted(random.Random(12345).sample(ok_rows, min(a.verify, len(ok_rows))))
    f32 = json.load(open(C.F32_INDEX))
    f32row = {p: i for i, p in enumerate(f32["paths"])}
    f32mm = np.memmap(os.path.join(os.path.dirname(C.F32_INDEX), "frames.u8"), dtype=np.uint8,
                      mode="r", shape=(f32["n"], f32["frames"], f32["size"], f32["size"], 3)) \
        if f32["size"] == S else None
    mism, xchk = [], []
    for r in vr:
        ref = load_clip(C.to_full(paths[r]), T, S, raw=True)
        if ref is None or not np.array_equal(ref.transpose(1, 2, 3, 0), mm[r]):
            mism.append(paths[r])
        if f32mm is not None:
            x = _f32_crosscheck(paths[r], np.asarray(mm[r]), f32row, f32mm)
            if x:
                xchk.append(x)
    print(f"verify: {len(vr) - len(mism)}/{len(vr)} clips bit-identical to "
          f"train_classifier.load_clip(data_full/..., {T}, {S})", flush=True)
    if mism:
        raise SystemExit(f"ABORT: {len(mism)} verified clips differ from a direct decode, "
                         f"e.g. {mism[0]}. No index written.")
    xsum = None
    if xchk:
        xsum = dict(n=len(xchk),
                    n_bit_identical_shared=int(sum(x["bit_identical_shared"] for x in xchk)),
                    same_index_mad_mean=float(np.mean([x["same_index_mad"] for x in xchk])),
                    same_index_mad_max=float(np.max([x["same_index_mad"] for x in xchk])),
                    neighbour_mad_mean=float(np.mean([x["neighbour_mad"] for x in xchk])),
                    aligned_all=bool(all(x["same_index_mad"] < x["neighbour_mad"] for x in xchk)))
        print(f"f32s224 cross-check on shared frames (0, n-1): {xsum}", flush=True)
    del mm

    os.replace(bin_part, bin_final)
    meta = {"paths": paths, "frames": T, "size": S, "n": n, "unreadable": sorted(fails),
            "unreadable_reasons": fails, "substituted": subs, "decoded_after_retry": retried,
            "source": "data_full/<rel> decoded, keys kept as data/<rel>",
            "index_rule": "sample_frame_indices(header CAP_PROP_FRAME_COUNT, T); sequential "
                          "grab/retrieve; BGR->RGB; INTER_AREA",
            "built_by": "video-eeg-ensembling/grader/build_f16_cache.py", "built_at": time.strftime("%F %T"),
            "limit": a.limit, "sample_seed": a.sample_seed, "paths_file": a.paths_file,
            "verify": dict(n=len(vr), bit_identical=len(vr) - len(mism)),
            "f32s224_crosscheck": xsum, "fingerprint": fp}
    C.atomic_json(idx_path, meta)
    if os.path.exists(prog_path):
        os.replace(prog_path, os.path.join(out, "progress.done.npz"))
    print(f"done: {n - len(fails)} cached, {len(fails)} unreadable, {len(subs)} with "
          f"substituted frames; {time.time() - t0:.0f}s. wrote {idx_path}", flush=True)


if __name__ == "__main__":
    main()
