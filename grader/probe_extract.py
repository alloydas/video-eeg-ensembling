#!/usr/bin/env python3
"""
Stage 2 of TT-X3D: frozen Kinetics X3D-M features for every discover() clip under three
inputs, for the pre-registered linear-probe kill test of the dense (native-rate) pathway.

PURPOSE
  Before any dense-pathway training is funded, ask whether 2.1-s native-rate snippets carry
  severity information that (i) the historical sparse input does not and (ii) is MOTION,
  not appearance. For each clip (all 24,497, sorted by path) this extracts the pooled
  2048-d X3D-M embedding (pretrained on Kinetics-400, eval mode; head.proj -> Identity AND
  head.activation -> None, otherwise the 2048 features would be softmaxed; the dimension is
  asserted with a dummy forward) for:
    (a) sparse16  - the historical 16 linspace frames at 224, read from the f16s224 cache
                    (decoded with the identical rule if the cache or the clip is absent);
    (b) dense     - 8 snippets of 16 frames at stride 2 (every other native frame: 0.133 s
                    spacing, 2.13 s of exposure at 15 fps). Snippet j is centred on the
                    centre of the j-th of 8 equal strata of W = [150, n-150) frames (10-s
                    buffers; n = header frame count); if |W| < 32 frames one snippet centred
                    in the clip fills all 8 slots. Near a short window a snippet may extend
                    up to 15 frames past W, because it is centred, not clipped, on its
                    stratum. Frames come from data_full with cv2 CAP_PROP_POS_FRAMES seeks
                    (frame-exact: --verify_seek re-decodes sequentially and requires bit
                    identity), BGR->RGB, INTER_AREA to 224x224, then train_pooled.norm_batch;
    (c) dense_shuffled - the SAME frames with the temporal order permuted inside each
                    snippet by a fixed per-clip seed (crc32 of the data/ key): appearance is
                    kept, motion is destroyed.
  GOP refresh frames (index 0 mod 32, a quality step) are left untouched in every input.

OUTPUT  <out_dir>/features.npz
  path [N] (data/... keys), y5 [N], sparse [N,D] f32, dense_mean [N,D] f32,
  dense_snip [N,8,D] f16, shuffled_mean [N,D] f32, starts [N,8], n_header [N],
  single_snippet [N], sparse_src [N] ('cache'|'decode'), failed [F] (paths),
  plus provenance scalars. A clip that cannot be decoded exactly has NaN features and is
  listed in `failed`; the merge ABORTS if more than 0.5% of clips fail. Each decode is
  re-tried --decode_retries times (default 2) inside the worker before it counts as failed,
  and --retry_failed re-extracts the failed rows of already-finished shards (rewriting those
  shards atomically), so a transient NFS error never has to become a permanent drop.
  Work is written in shards (<out_dir>/shards/shard_00000.npz, ...) atomically, so a
  preempted job resumes where it stopped; features.npz is assembled only when every shard
  exists and matches the item fingerprint.

USAGE (from the video-eeg-ensembling repo, any cwd; PYTHONDONTWRITEBYTECODE=1; EEG_ROOT env
       var, default /work/mech-ai-scratch/alloy/EEG. The script chdirs to EEG_ROOT, so a
       relative --cache_dir is resolved against EEG_ROOT; --out_dir must be absolute.)
  GPU (see grader/sbatch_probe.sh):
    python grader/probe_extract.py --out_dir /work/mech-ai-scratch/alloy/EEG/output/ttg_probe \
        --workers 14 --batch_clips 4 --verify_seek 20
  CPU dry run on 10 clips (fp32, writes only to --out_dir):
    python grader/probe_extract.py --dry_run --limit 10 --workers 2 \
        --out_dir /work/mech-ai-scratch/alloy/EEG/output/ttg_tmp/probe_dryrun
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ttg_common as C                                               # noqa: E402  (puts EEG_ROOT on sys.path)

os.environ.setdefault("TORCH_HOME", "/work/mech-ai-scratch/alloy/.cache/torch")
import torch                                                          # noqa: E402
import torch.nn as nn                                                 # noqa: E402
from torch.utils.data import DataLoader, Dataset                      # noqa: E402

import train_pooled as tp                                             # noqa: E402

K, T, STRIDE, S, BUFFER = 8, 16, 2, 224, 150


def build_feature_net():
    from pytorchvideo.models.hub import x3d_m
    m = x3d_m(pretrained=True)
    head = m.blocks[-1]
    if not isinstance(getattr(head, "activation", None), nn.Softmax):
        raise SystemExit("unexpected x3d_m head (activation is not Softmax); re-verify")
    head.proj = nn.Identity()
    head.activation = None        # else Softmax(dim=1) would run over the 2048 features
    return m.eval()


class ProbeClips(Dataset):
    """size: decode resolution (default S = 224, the X3D input; step0_vjepa_extract.py passes
    256 with cache_dir=None). Frame indices, snippet geometry and the shuffle seed do not
    depend on it."""
    def __init__(s, items, rows, cache_dir, retries=2, size=S):
        s.items, s.rows, s.cache_dir, s.retries, s.size = items, rows, cache_dir, retries, size

    def __len__(s):
        return len(s.rows)

    def __getitem__(s, j):
        """Decode one clip; a failure is retried `retries` times (1 s, 2 s pauses) so a
        transient NFS / open error does not become a permanent NaN row."""
        for attempt in range(s.retries + 1):
            d = s._load(s.rows[j])
            if d["ok"] or attempt == s.retries:
                if attempt:
                    d["attempts"] = attempt + 1
                return d
            time.sleep(1.0 + attempt)

    def _load(s, r):
        p, y5 = s.items[r][0], s.items[r][1]
        try:
            full = C.to_full(p)
            a, src = None, "decode"
            if s.cache_dir:
                a = tp.FrameCache.get(s.cache_dir, T, s.size).fetch(p)
                src = "cache" if a is not None else "decode"
            if a is None:
                arr, _ = C.decode_linspace(full, T, s.size)
                a = arr.transpose(3, 0, 1, 2)
            n = C.header_frame_count(full)
            starts, single = C.snippet_starts(n, K, T, STRIDE, BUFFER)
            dense = C.decode_seek_snippets(full, starts, T, STRIDE, s.size).transpose(0, 4, 1, 2, 3)
            rng = np.random.default_rng(C.clip_seed(p))
            perm = np.stack([rng.permutation(T) for _ in range(K)])
            return dict(ok=True, row=r, y5=int(y5), sparse=torch.from_numpy(np.ascontiguousarray(a)),
                        dense=torch.from_numpy(np.ascontiguousarray(dense)),
                        perm=torch.from_numpy(perm), starts=torch.from_numpy(starts.astype(np.int64)),
                        n_header=n, single=bool(single), src=src)
        except Exception as e:                       # recorded, features left NaN; never filled
            return dict(ok=False, row=r, err=f"{type(e).__name__}: {e}")


def collate(batch):
    good = [b for b in batch if b["ok"]]
    bad = [(b["row"], b["err"]) for b in batch if not b["ok"]]
    out = dict(bad=bad, n=len(good), retried=[b["row"] for b in good if b.get("attempts")])
    if good:
        for k in ("sparse", "dense", "perm", "starts"):
            out[k] = torch.stack([b[k] for b in good])
        for k in ("row", "y5", "n_header", "single", "src"):
            out[k] = [b[k] for b in good]
    return out


def verify_seek(items, rows, size=S):
    """Seek-decoded snippet frames must equal a sequential decode bit-for-bit (at `size`)."""
    for r in rows:
        full = C.to_full(items[r][0])
        n = C.header_frame_count(full)
        st, _ = C.snippet_starts(n, K, T, STRIDE, BUFFER)
        a = C.decode_seek_snippets(full, st, T, STRIDE, size)
        idx = (st[:, None] + STRIDE * np.arange(T)[None]).ravel()
        b = C.decode_sequential_indices(full, idx, size).reshape(a.shape)
        if not np.array_equal(a, b):
            raise SystemExit(f"ABORT: seek decode is not frame-exact for {items[r][0]}")
    print(f"verify_seek: {len(rows)} clips, seek-decoded snippets bit-identical to a "
          f"sequential decode", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default=None,
                    help="default /work/.../EEG/output/ttg_probe (ttg_tmp/probe_dryrun with --dry_run)")
    ap.add_argument("--cache_dir", default="cache_frames/f16s224",
                    help="sparse16 source, relative to EEG_ROOT or absolute; clips absent from "
                         "it (or no cache) are decoded")
    ap.add_argument("--items", choices=["discover", "f32index"], default="discover")
    ap.add_argument("--shard_size", type=int, default=512)
    ap.add_argument("--batch_clips", type=int, default=4, help="clips per batch (x17 views)")
    ap.add_argument("--chunk_views", type=int, default=68, help="views per forward pass")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--verify_seek", type=int, default=0,
                    help="first check N random clips: seek decode == sequential decode")
    ap.add_argument("--limit", type=int, default=0, help="only N clips (tests)")
    ap.add_argument("--sample_seed", type=int, default=None, help="with --limit: random subset")
    ap.add_argument("--max_fail_frac", type=float, default=0.005)
    ap.add_argument("--decode_retries", type=int, default=2,
                    help="re-try a failed clip decode this many times inside the worker")
    ap.add_argument("--retry_failed", action="store_true",
                    help="re-extract the failed (NaN) rows of already-finished shards and "
                         "rewrite those shards atomically (e.g. after a transient NFS error)")
    ap.add_argument("--dry_run", action="store_true",
                    help="CPU, fp32, --limit 10 unless given; output only to --out_dir")
    a = ap.parse_args()

    C.enter_eeg_root()                                   # relative paths below are EEG_ROOT's
    if a.dry_run:
        a.limit = a.limit or 10
        a.out_dir = a.out_dir or os.path.join(C.EEG_ROOT, "output", "ttg_tmp", "probe_dryrun")
    out = C.check_output_dir(a.out_dir or os.path.join(C.EEG_ROOT, "output", "ttg_probe"))
    if torch.cuda.is_available() and not a.dry_run:
        dev = torch.device("cuda")
        ok, msg = C.check_cuda_arch()
        print(msg, flush=True)
        if not ok:
            print("FATAL: unsupported GPU for this torch build", flush=True)
            sys.exit(4)
    elif a.dry_run:
        dev = torch.device("cpu")
    else:
        raise SystemExit("CUDA not available (use --dry_run for a CPU test)")
    cache_dir = a.cache_dir if a.cache_dir and os.path.exists(os.path.join(a.cache_dir, "index.json")) else None
    print(f"sparse16 source: {cache_dir or 'decode from data_full (no f16s224 cache)'}", flush=True)

    items = tp.discover() if a.items == "discover" else C.items_from_f32_index()
    items = sorted(items, key=lambda it: it[0])
    if a.limit:
        if a.sample_seed is not None:
            keep = sorted(random.Random(a.sample_seed).sample(range(len(items)), a.limit))
            items = [items[r] for r in keep]
        else:
            items = items[:a.limit]
        print(f"*** --limit {a.limit}: NOT the full feature set ***", flush=True)
    N = len(items)
    fp = C.fingerprint([i[0] for i in items])
    print(f"{N} clips, fingerprint {fp}", flush=True)

    if a.verify_seek:
        verify_seek(items, sorted(random.Random(0).sample(range(N), min(a.verify_seek, N))))

    model = build_feature_net().to(dev)
    with torch.no_grad():
        D = model(torch.zeros(1, 3, T, S, S, device=dev)).shape
    if len(D) != 2 or D[1] != 2048:
        raise SystemExit(f"unexpected feature shape {tuple(D)}")
    D = int(D[1])
    print(f"X3D-M feature dim verified: {D}", flush=True)
    amp = torch.autocast("cuda", dtype=torch.float16) if dev.type == "cuda" \
        else torch.autocast("cpu", enabled=False)

    sd = os.path.join(out, "shards")
    os.makedirs(sd, exist_ok=True)
    n_sh = (N + a.shard_size - 1) // a.shard_size
    shard_rows = [list(range(k * a.shard_size, min((k + 1) * a.shard_size, N))) for k in range(n_sh)]

    def shard_path(k):
        return os.path.join(sd, f"shard_{k:05d}.npz")

    def new_row():
        return dict(sparse=np.full(D, np.nan, np.float32),
                    dense_snip=np.full((K, D), np.nan, np.float16),
                    dense_mean=np.full(D, np.nan, np.float32),
                    shuffled_mean=np.full(D, np.nan, np.float32),
                    starts=np.full(K, -1, np.int64), n_header=-1, single=False, src="",
                    err="")

    done, buf, retry_rows, kept_failed = set(), {}, [], 0
    for k in range(n_sh):
        f = shard_path(k)
        if os.path.exists(f):
            z = np.load(f)
            if str(z["fingerprint"]) != fp or z["rows"].tolist() != shard_rows[k]:
                raise SystemExit(f"{f} was written for a different item list; use a fresh --out_dir")
            bad = np.flatnonzero(z["err"] != "")
            if len(bad) and a.retry_failed:
                z = {key: z[key] for key in z.files}            # read each array once
                if z["sparse"].shape[1] != D:
                    raise SystemExit(f"{f} has feature dim {z['sparse'].shape[1]} != {D}")
                for j, r in enumerate(shard_rows[k]):           # keep the good rows as they are
                    if z["err"][j] == "":
                        buf[r] = dict(sparse=z["sparse"][j], dense_snip=z["dense_snip"][j],
                                      dense_mean=z["dense_mean"][j], shuffled_mean=z["shuffled_mean"][j],
                                      starts=z["starts"][j], n_header=int(z["n_header"][j]),
                                      single=bool(z["single_snippet"][j]), src=str(z["sparse_src"][j]),
                                      err="")
                    else:
                        retry_rows.append(r)
            else:
                done.add(k)
                kept_failed += len(bad)
    todo = sorted([r for k in range(n_sh) if k not in done for r in shard_rows[k] if r not in buf])
    print(f"{len(done)}/{n_sh} shards already done; {len(todo)} clips to extract"
          + (f" (incl. {len(retry_rows)} failed rows re-tried from finished shards)" if retry_rows else ""),
          flush=True)
    if kept_failed:
        print(f"note: {kept_failed} failed (NaN) rows in finished shards are kept; pass "
              f"--retry_failed to re-extract them", flush=True)

    if todo:
        dl = DataLoader(ProbeClips(items, todo, cache_dir, a.decode_retries), batch_size=a.batch_clips,
                        shuffle=False, num_workers=a.workers, collate_fn=collate,
                        pin_memory=(dev.type == "cuda"),
                        prefetch_factor=2 if a.workers > 0 else None)
        t0, n_seen, last_print = time.time(), 0, time.time()

        def flush_ready():
            for k in sorted({r // a.shard_size for r in buf}):
                rows = shard_rows[k]
                if all(r in buf for r in rows):
                    R = [buf.pop(r) for r in rows]
                    C.atomic_npz(
                        shard_path(k), rows=np.array(rows), fingerprint=np.array(fp),
                        path=np.array([items[r][0] for r in rows]),
                        y5=np.array([items[r][1] for r in rows]),
                        sparse=np.stack([x["sparse"] for x in R]),
                        dense_snip=np.stack([x["dense_snip"] for x in R]),
                        dense_mean=np.stack([x["dense_mean"] for x in R]),
                        shuffled_mean=np.stack([x["shuffled_mean"] for x in R]),
                        starts=np.stack([x["starts"] for x in R]),
                        n_header=np.array([x["n_header"] for x in R]),
                        single_snippet=np.array([x["single"] for x in R]),
                        sparse_src=np.array([x["src"] for x in R]),
                        err=np.array([x["err"] for x in R]))
                    print(f"  wrote shard {k + 1}/{n_sh}", flush=True)

        with torch.no_grad():
            for b in dl:
                for r in b["retried"]:
                    print(f"  decoded after retry: {items[r][0]}", flush=True)
                for r, err in b["bad"]:
                    x = new_row()
                    x["err"] = err
                    buf[r] = x
                    print(f"  FAIL {items[r][0]}: {err}", flush=True)
                if b["n"]:
                    B = b["n"]
                    xs = b["sparse"].to(dev, non_blocking=True)
                    xd = b["dense"].to(dev, non_blocking=True).flatten(0, 1)       # (B*K,3,T,H,W)
                    pm = b["perm"].to(dev).flatten(0, 1)                             # (B*K,T)
                    xsh = torch.stack([xd[i][:, pm[i]] for i in range(B * K)])
                    views = torch.cat([xs, xd, xsh], 0)
                    F = []
                    for c in range(0, len(views), a.chunk_views):
                        with amp:
                            F.append(model(tp.norm_batch(views[c:c + a.chunk_views], dev)).float())
                    F = torch.cat(F).cpu().numpy()
                    fs, fd, fh = F[:B], F[B:B + B * K].reshape(B, K, D), F[B + B * K:].reshape(B, K, D)
                    for j in range(B):
                        x = new_row()
                        x.update(sparse=fs[j], dense_snip=fd[j].astype(np.float16),
                                 dense_mean=fd[j].mean(0), shuffled_mean=fh[j].mean(0),
                                 starts=b["starts"][j].numpy(), n_header=b["n_header"][j],
                                 single=b["single"][j], src=b["src"][j])
                        buf[b["row"][j]] = x
                n_seen += b["n"] + len(b["bad"])
                if time.time() - last_print > 60 or n_seen == len(todo):
                    last_print = time.time()
                    print(f"  {n_seen}/{len(todo)} clips  {n_seen / (time.time() - t0):.2f} clips/s",
                          flush=True)
                flush_ready()
        flush_ready()
        if buf:
            raise SystemExit(f"internal error: {len(buf)} rows never flushed")

    # ---- merge
    Z = [np.load(shard_path(k)) for k in range(n_sh)]
    cat = {k: np.concatenate([z[k] for z in Z]) for k in
           ("rows", "path", "y5", "sparse", "dense_snip", "dense_mean", "shuffled_mean",
            "starts", "n_header", "single_snippet", "sparse_src", "err")}
    assert cat["rows"].tolist() == list(range(N))
    failed = cat["path"][cat["err"] != ""]
    print(f"merge: {N} clips, {len(failed)} failed, sparse from cache "
          f"{int((cat['sparse_src'] == 'cache').sum())} / decoded "
          f"{int((cat['sparse_src'] == 'decode').sum())}, single-snippet clips "
          f"{int(cat['single_snippet'].sum())}", flush=True)
    if len(failed) > a.max_fail_frac * N:
        raise SystemExit(f"ABORT: {len(failed)}/{N} clips failed (> {100 * a.max_fail_frac:.1f}%); "
                         f"shards kept, features.npz NOT written (re-run with --retry_failed "
                         f"once the cause is fixed)")
    del cat["rows"]
    C.atomic_npz(os.path.join(out, "features.npz"), **cat, failed=failed,
                 fingerprint=np.array(fp), feature_dim=np.int64(D),
                 geometry=np.array(json.dumps(dict(K=K, T=T, stride=STRIDE, size=S, buffer=BUFFER,
                                                   model="x3d_m Kinetics-400, proj=Identity, activation=None",
                                                   precision="fp16 autocast" if dev.type == "cuda" else "fp32 (cpu)",
                                                   items=a.items, limit=a.limit))))
    print(f"wrote {out}/features.npz", flush=True)


if __name__ == "__main__":
    main()
