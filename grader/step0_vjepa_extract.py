#!/usr/bin/env python3
"""
Step 0 of the JEPA plan: frozen V-JEPA 2 ViT-L features for every discover() clip under the
Stage-2 inputs (sparse16 / dense / dense_shuffled) at 256x256, for the pre-registered severity
probe of grader/step0_vjepa_prereg.md. This script implements that document's sections 1, 2 and 8
(extraction, budget, dry run). The probe is the analysis script's job.

PURPOSE
  Does a frozen, pretrained video JEPA carry severe-vs-mild information on unseen animals beyond
  frozen Kinetics X3D-M, and is it order-dependent (motion)? This script only extracts. For each
  clip (24,497, discover() sorted by path; fingerprint 75e31ebd:24497, equal to Stage 2's) it runs
  17 views through facebook/vjepa2-vitl-fpc64-256 (revision b3c1679b..., VJEPA2Model, encoder only,
  fp32 weights, sdpa, bf16 autocast on the GPU, fp32 on CPU):
    sparse16        C.decode_linspace(full, 16, 256): Stage 2's historical linspace frames;
    dense           8 snippets x 16 frames at stride 2: C.snippet_starts(n, 8, 16, 2, 150) and
                    C.decode_seek_snippets(full, starts, 16, 2, 256), frame-exact seeks;
    dense_shuffled  the same frames, reordered inside each snippet with probe_extract.py's code:
                    rng = np.random.default_rng(C.clip_seed(p)) with p the FULL item string
                    ('data/.../video.mp4'), perm = [rng.permutation(16) for _ in range(8)].
  The frames come from data_full, BGR->RGB, full-frame INTER_AREA squash to 256x256: the decoding,
  the frame indices and the shuffle are probe_extract.ProbeClips (size=256, no frame cache), so
  they are Stage 2's indices exactly; per clip, starts / n_header / single_snippet are asserted
  equal to Stage 2's features.npz. Normalisation is done here, never with train_pooled.norm_batch
  (Kinetics constants): x = (u8/255 - mean_c)/std_c with the ImageNet constants read from and
  asserted against the pinned snapshot's video_preprocessor_config.json, then (B,C,T,H,W) is
  permuted to the model's (B,T,C,H,W). A 16-frame 256x256 view gives 8 x 16 x 16 = 2,048 tokens,
  token index 256 t + 16 h + w.

OUTPUT  <out_dir> (default $EEG_ROOT/output/ttg_vjepa; tests under $EEG_ROOT/output/ttg_tmp/vjepa_*;
        nothing else is accepted)
  items.npz     the sorted discover() item list (path, y5, session, animal), fingerprint-checked;
                written once (the GPU segments reuse it instead of re-globbing NFS for 2.5 min).
  run.json      the run config (items fingerprint, geometry, model revision, precision, readout
                layers, shard plan, this script's sha256, package versions) and its key; every
                later segment, shard and the merge must match it, else they refuse.
  shards/shard_KKKKK.npz   512 clips each, written atomically.
  budget/seg_*.json, budget/job_*.txt, budget.json   GPU-time accounting (below).
  features.npz  (--merge) rows in sorted-path order:
     path, y5, starts [N,8], n_header, single_snippet, perm [N,8,16] int8 (applied permutations),
     sparse, dense_mean, shuffled_mean [N,1024] f32  mean of the 2,048 last-layer tokens
                                  (dense / shuffled: mean over the 8 snippets of snippet means),
     dense_tt, shuffled_tt [N,8,8,1024] f16  snippet j, temporal index t: mean of tokens 256t..256t+255,
     dense_tv, shuffled_tv [N,1024] f32  mean over j and t=0..6 of |u[j,t+1]-u[j,t]|, computed in
                                  fp32 on the device from the fp32 temporal means (not from *_tt),
     sparse_ml, dense_ml, shuffled_ml [N,3,1024] f32  blocks 17, 19, 21 of model.encoder.layer:
                                  forward hook on output[0] -> fp32 -> model.encoder.layernorm ->
                                  mean over tokens (dense / shuffled: mean over snippets),
     err, failed (paths), not_extracted (paths), fingerprint, feature_dim, ml_blocks, geometry,
     provenance (JSON: config, versions, the GPU precision checks, first-batch checks of every
     segment, budget, code commit and sha256s).
  A clip that cannot be decoded exactly gets NaN rows and is listed in `failed`, never filled.

WORK ORDER, SHARDS, SPLITTING ACROSS GPUs
  Pass 1 = the 12,140 seizure rows (y5 >= 1) in sorted order, shards 0..23; pass 2 = the 12,357
  non-seizure rows, shards 24..48 (512 rows per shard; the numbers are printed by --prepare).
  --shard_start / --shard_stop restrict a job to shards [start, stop), so several GPU jobs can
  split the run (ranges must not overlap; give pass-1 ranges first). GPU jobs never merge:
  --merge (CPU) reassembles sorted-path order, re-checks every row and refuses on any fingerprint,
  config, row, label or consistency mismatch. --allow_pass2_incomplete lets the merge write
  features.npz when every pass-1 shard exists but some pass-2 shards do not (their rows are NaN
  and listed in `not_extracted`, not in `failed`), as section 8 of the pre-registration says.
  Resume: finished shards are skipped, a partial shard is lost; --retry_failed re-extracts the
  failed rows of finished shards and rewrites them atomically; each decode is retried
  --decode_retries times first. The merge aborts if more than 0.5% of the attempted clips failed.

BUDGET (section 8: 8 GPU-hours over every job of the run, preempted segments included)
  Each segment (one start of one job) writes budget/seg_<job>_r<restart>_<host>_<pid>.json with
  its start (the driver's start time, VJ_SEGMENT_T0) and last-write time, and rewrites the
  aggregate budget.json, at every shard flush, every 5 minutes from a thread, and from the
  SIGTERM handler. The running total is max(sum of segment elapsed times, sacct elapsed x GPUs
  over every recorded job id incl. requeued runs), logged with its source at the segment start and
  re-queried (at most every 5 min) before each shard. A shard starts only if
      total + (live segments) x (one shard's measured duration)  [+ unfinished pass-1 shards x
      one shard's duration, when the shard is a pass-2 shard]  <=  cap,
  so parallel jobs cannot overrun the cap together and pass 2 never eats pass 1's budget. When the
  cap stops a segment it exits 5 (the driver does not requeue) and prints what is done.

CHECKS (every segment, first batch; the dry run is the pre-registered CPU test)
  token shape (V, 2048, 1024); the normalisation constants against the snapshot; the tensor
  actually fed to the model (pre-hook on the embeddings) at fixed (view, t, h, w) and every
  channel equals (u8/255 - mean_c)/std_c within 1e-6, for a sparse, a dense and a shuffled view;
  every fed shuffled frame t equals the fed dense frame perm[j][t] exactly; the first sorted clip's
  perm[0] equals the pre-registered anchor; the block-23 hook path (LayerNorm then mean) equals the
  mean of last_hidden_state, and a direct `model(x, skip_predictor=True)` call on the first chunk
  gives the same token means, within 1e-5 relative on CPU (both are exactly 0 in the dry run; a GPU
  segment records the value and aborts above 1e-3); on the GPU every segment's first batch is
  run again in fp32 (TF32 off) and a bf16-vs-fp32 token-mean cosine below 0.99 aborts. Recorded
  with it, never gating: the bf16 error of u, *_mean, *_tv and *_ml, and `bf16_over_gap`, that
  error divided by the fp32 dense-vs-shuffled gap of the same clips (for *_mean, the decision's
  readout, it should be << 1: 0.003 in a CPU bf16 simulation; for *_tv, the non-gating TT
  readout, that simulation gave 0.13, because first differences amplify token noise). At each
  shard flush and again at the merge: finite values in every non-failed row,
  |mean_{j,t} *_tt - *_mean| and |v(f16 *_tt) - *_tv| <= 1e-3 (1 + max_{j,t} |*_tt|), perm equal to
  the regenerated permutations, starts / n_header / single_snippet equal to Stage 2's.
  Code and weights: --prepare hashes the content of model.safetensors (not only its blob name)
  against the pinned sha256. Every segment records the sha256 of this script, probe_extract.py and
  ttg_common.py (decoding, snippet geometry, shuffle seed) and refuses to start if they differ
  from run.json's; the merge refuses rows from a segment that ran other code. A dry run refuses
  a directory that already holds features.npz (it would extract nothing).

USAGE (the V-JEPA 2 venv, never the shared env; PYTHONDONTWRITEBYTECODE=1; any cwd, the script
       chdirs to EEG_ROOT; HF_HOME defaults to /work/mech-ai-scratch/alloy/hf_cache and the model
       is read with local_files_only)
  PY=/work/mech-ai-scratch/alloy/.venvs/vjepa2/bin/python
  # 1. CPU: item list, --verify_seek at 256, run.json (grader/sbatch_vjepa.sh prepare)
  $PY grader/step0_vjepa_extract.py --prepare --verify_seek 20 --workers 8
  # 2. GPU (grader/sbatch_vjepa.sh [start stop]); several jobs may split the shards
  $PY grader/step0_vjepa_extract.py --workers 14 --batch_clips 4 --retry_failed [--shard_start 0 --shard_stop 24]
  # 3. CPU: assemble features.npz (grader/sbatch_vjepa.sh merge)
  $PY grader/step0_vjepa_extract.py --merge [--allow_pass2_incomplete]
  $PY grader/step0_vjepa_extract.py --status
  # the pre-registered CPU dry run (first 3 sorted clips; under srun -c 8, never the login node)
  $PY grader/step0_vjepa_extract.py --dry_run --limit 3 --verify_seek 3 --workers 2 \\
      --out_dir /work/mech-ai-scratch/alloy/EEG/output/ttg_tmp/vjepa_dryrun
"""
import argparse
import contextlib
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time

os.environ.setdefault("HF_HOME", "/work/mech-ai-scratch/alloy/hf_cache")   # before any HF import

import numpy as np                                                    # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ttg_common as C                                                # noqa: E402  (EEG_ROOT on sys.path)
import probe_extract as PE                                            # noqa: E402  (decoding, shuffle, verify_seek)
import torch                                                          # noqa: E402
from torch.utils.data import DataLoader                               # noqa: E402

K, T, STRIDE, BUFFER = PE.K, PE.T, PE.STRIDE, PE.BUFFER               # 8 snippets x 16 frames, stride 2, 150
S = 256                                                               # V-JEPA 2 crop; H = W = 256 exactly
REPO = "facebook/vjepa2-vitl-fpc64-256"
REV = "b3c1679b7c34d3255ef3547f27c7b226aefab26f"
MODEL_SHA256 = "25466aef85727d16546c6cf8c99f12fcfad9cbca8225d45f23685e2e025b786b"   # model.safetensors
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
D, NTOK, TT, TOK_PER_T = 1024, 2048, 8, 256          # width, tokens per view, temporal tokens, tokens each
ML_BLOCKS = (17, 19, 21)
LAST_BLOCK = 23
EXPECTED_FP = "75e31ebd:24497"                       # Stage 2's discover() fingerprint
ANCHOR_PATH = ("data/Data_RN197_cropped/10-12-2023 (2)/"
               "clip_01_vs_seizure_02_Stage_3_20231012_162605/video.mp4")
ANCHOR_SEED = 3765462539
ANCHOR_PERM0 = [11, 14, 5, 8, 6, 1, 10, 13, 15, 7, 12, 3, 4, 0, 2, 9]
WRONG_KEY_PERM0 = [10, 13, 8, 14, 4, 5, 7, 6, 3, 0, 12, 2, 1, 11, 15, 9]   # the C.key_of(p) seed rule
STAGE2 = os.path.join(C.EEG_ROOT, "output", "ttg_probe", "features.npz")
REAL_OUT = os.path.join(C.EEG_ROOT, "output", "ttg_vjepa")
TEST_PREFIX = os.path.join(C.EEG_ROOT, "output", "ttg_tmp", "vjepa_")
EXIT_SIGTERM, EXIT_GPU, EXIT_BUDGET = 3, 4, 5
FLOAT_KEYS = ("sparse", "dense_mean", "shuffled_mean", "dense_tt", "shuffled_tt", "dense_tv",
              "shuffled_tv", "sparse_ml", "dense_ml", "shuffled_ml")
SHAPES = dict(sparse=((D,), np.float32), dense_mean=((D,), np.float32), shuffled_mean=((D,), np.float32),
              dense_tt=((K, TT, D), np.float16), shuffled_tt=((K, TT, D), np.float16),
              dense_tv=((D,), np.float32), shuffled_tv=((D,), np.float32),
              sparse_ml=((len(ML_BLOCKS), D), np.float32), dense_ml=((len(ML_BLOCKS), D), np.float32),
              shuffled_ml=((len(ML_BLOCKS), D), np.float32))


def log(msg):
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- small helpers

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def check_out_dir(path):
    """Absolute, under $EEG_ROOT/output/ttg_* (the grader contract), and more narrowly the real
    run's output/ttg_vjepa or a test dir output/ttg_tmp/vjepa_<name>."""
    rp = C.check_output_dir(path)
    if not (rp == REAL_OUT or rp.startswith(REAL_OUT + os.sep) or rp.startswith(TEST_PREFIX)):
        raise SystemExit(f"--out_dir must be {REAL_OUT} or under {TEST_PREFIX}<name>, got {rp}")
    return rp


def versions():
    import cv2
    import huggingface_hub
    import tokenizers
    import transformers
    return dict(python=sys.version.split()[0], torch=torch.__version__, cuda=torch.version.cuda,
                transformers=transformers.__version__, tokenizers=tokenizers.__version__,
                huggingface_hub=huggingface_hub.__version__, numpy=np.__version__, cv2=cv2.__version__,
                venv=sys.prefix)


CODE_FILES = ("step0_vjepa_extract.py", "probe_extract.py", "ttg_common.py")


def code_sha256():
    """sha256 of the code that decides the features (this script, and probe_extract.py /
    ttg_common.py, which hold the decoding, the snippet geometry and the shuffle seed)."""
    return {f: sha256_file(os.path.join(C.GRADER_DIR, f)) for f in CODE_FILES}


def code_identity():
    """code_sha256(), plus the git commit that last touched this script and whether the checkout
    differs from it."""
    g = C.GRADER_DIR
    files = code_sha256()
    out = dict(sha256=files)
    try:
        run = lambda *c: subprocess.run(["git", "-C", g, *c], capture_output=True, text=True,  # noqa: E731
                                        timeout=30).stdout.strip()
        out["head"] = run("rev-parse", "HEAD")
        out["script_commit"] = run("log", "-1", "--format=%H", "--", "step0_vjepa_extract.py") or "untracked"
        out["dirty"] = run("status", "--porcelain", "--", *files) or ""
    except Exception as e:                                   # provenance only; never fatal
        out["git_error"] = f"{type(e).__name__}: {e}"
    return out


def snapshot_dir():
    """The pinned local snapshot, with its config and preprocessing constants asserted."""
    from huggingface_hub import snapshot_download
    snap = snapshot_download(REPO, revision=REV, local_files_only=True)
    if os.path.basename(os.path.normpath(snap)) != REV:
        raise SystemExit(f"snapshot {snap} is not revision {REV}")
    blob = os.path.basename(os.path.realpath(os.path.join(snap, "model.safetensors")))
    if blob != MODEL_SHA256:
        raise SystemExit(f"model.safetensors blob {blob} != pinned sha256 {MODEL_SHA256}")
    cfg = json.load(open(os.path.join(snap, "config.json")))
    want = dict(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, patch_size=16,
                tubelet_size=2, crop_size=256, image_size=256, layer_norm_eps=1e-6, mlp_ratio=4)
    bad = {k: (cfg.get(k), v) for k, v in want.items() if cfg.get(k) != v}
    if bad:
        raise SystemExit(f"config.json differs from the pre-registered facts: {bad}")
    pp = json.load(open(os.path.join(snap, "video_preprocessor_config.json")))
    if (list(pp["image_mean"]) != list(MEAN) or list(pp["image_std"]) != list(STD)
            or abs(pp["rescale_factor"] - 1 / 255) > 1e-15 or not pp["do_rescale"] or not pp["do_normalize"]):
        raise SystemExit(f"normalisation constants in the snapshot differ: mean {pp['image_mean']} "
                         f"std {pp['image_std']} rescale {pp['rescale_factor']}")
    return snap, dict(snapshot=snap, safetensors_sha256=blob, image_mean=pp["image_mean"],
                      image_std=pp["image_std"], rescale_factor=pp["rescale_factor"])


def perms_for(path):
    rng = np.random.default_rng(C.clip_seed(path))
    return np.stack([rng.permutation(T) for _ in range(K)])


def check_anchor():
    if C.clip_seed(ANCHOR_PATH) != ANCHOR_SEED or perms_for(ANCHOR_PATH)[0].tolist() != ANCHOR_PERM0:
        raise SystemExit("shuffle seed rule does not reproduce the pre-registered anchor permutation")
    wrong = np.random.default_rng(C.clip_seed(C.key_of(ANCHOR_PATH))).permutation(T).tolist()
    if wrong != WRONG_KEY_PERM0:
        raise SystemExit("the documented wrong (key_of) permutation does not reproduce; re-verify")


# ----------------------------------------------------------------------------- items, plan, stage 2

def load_items(out, items_npz=None):
    """The full sorted discover() list. Cached in <out>/items.npz (NFS globbing takes minutes);
    the cache is accepted only if its fingerprint is Stage 2's and every label re-derives."""
    f = os.path.join(out, "items.npz")
    src = f if os.path.exists(f) else items_npz
    if src:
        z = np.load(src)
        items = [(str(p), int(y), str(s), str(a)) for p, y, s, a in
                 zip(z["path"], z["y5"], z["session"], z["animal"])]
        how = f"cached {src}"
    else:
        import train_pooled as tp
        t0 = time.time()
        items = sorted(tp.discover(), key=lambda it: it[0])
        items = [(str(p), int(y), str(s), str(a)) for p, y, s, a in items]
        how = f"train_pooled.discover() ({time.time() - t0:.0f} s)"
    paths = [i[0] for i in items]
    fp = C.fingerprint(paths)
    if fp != EXPECTED_FP:
        raise SystemExit(f"item fingerprint {fp} != Stage 2's {EXPECTED_FP} ({how})")
    if paths != sorted(paths) or paths[0] != ANCHOR_PATH:
        raise SystemExit("item list is not sorted by path, or its first clip is not the anchor clip")
    bad = [p for p, y, s, a in items if C.y5_of(p) != y or C.animal_of(p) != a or C.session_of(p) != s]
    if bad:
        raise SystemExit(f"{len(bad)} items whose label/session/animal do not re-derive, e.g. {bad[0]}")
    if src != f:
        C.atomic_npz(f, path=np.array(paths), y5=np.array([i[1] for i in items]),
                     session=np.array([i[2] for i in items]), animal=np.array([i[3] for i in items]),
                     fingerprint=np.array(fp), source=np.array(how))
    log(f"items: {len(items)} clips, fingerprint {fp} ({how})")
    return items


def select(items, limit, rows):
    if rows:
        idx = sorted(set(int(x) for x in rows.split(",")))
        print(f"*** --rows {idx}: NOT the full feature set ***", flush=True)
        return [items[i] for i in idx]
    if limit:
        print(f"*** --limit {limit}: NOT the full feature set ***", flush=True)
        return items[:limit]
    return items


def shard_plan(items, shard_size):
    """[(pass, [rows...]), ...]: pass-1 (seizure) shards first, then pass-2, each in sorted order."""
    plan = []
    for pas, keep in ((1, lambda y: y >= 1), (2, lambda y: y == 0)):
        rows = [r for r, it in enumerate(items) if keep(it[1])]
        plan += [(pas, rows[i:i + shard_size]) for i in range(0, len(rows), shard_size)]
    return plan


def load_stage2():
    z = np.load(STAGE2)
    if str(z["fingerprint"]) != EXPECTED_FP:
        raise SystemExit(f"{STAGE2} fingerprint {z['fingerprint']} != {EXPECTED_FP}")
    g = json.loads(str(z["geometry"]))
    if (g["K"], g["T"], g["stride"], g["buffer"]) != (K, T, STRIDE, BUFFER):
        raise SystemExit(f"Stage 2 geometry {g} differs from probe_extract's constants")
    st, n, sg = z["starts"], z["n_header"], z["single_snippet"]
    return {str(p): (st[i], int(n[i]), bool(sg[i])) for i, p in enumerate(z["path"])}


def stage2_mismatch(s2, path, starts, n_header, single):
    ref = s2.get(path)
    if ref is None:
        return "not in Stage 2's features.npz"
    if not np.array_equal(np.asarray(starts, np.int64), ref[0]) or int(n_header) != ref[1] \
            or bool(single) != ref[2]:
        return (f"starts {list(starts)} / n_header {n_header} / single {single} != Stage 2's "
                f"{list(ref[0])} / {ref[1]} / {ref[2]}")
    return None


# ----------------------------------------------------------------------------- run config

def make_config(a, items, plan, dev_kind, full_fp):
    n1 = sum(1 for p, _ in plan if p == 1)
    import transformers
    return dict(
        items=dict(fingerprint=C.fingerprint([i[0] for i in items]), n=len(items), full_fingerprint=full_fp,
                   source="train_pooled.discover() sorted by path", limit=a.limit, rows=a.rows or ""),
        geometry=dict(K=K, T=T, stride=STRIDE, size=S, buffer=BUFFER, decode="data_full, BGR->RGB, "
                      "full-frame cv2.INTER_AREA to 256x256", sparse="C.decode_linspace(full, 16, 256)",
                      dense="C.snippet_starts(n, 8, 16, 2, 150); C.decode_seek_snippets(full, starts, 16, 2, 256)"),
        shuffle="np.random.default_rng(C.clip_seed(full item path)); [rng.permutation(16) for _ in range(8)]",
        model=dict(repo=REPO, revision=REV, safetensors_sha256=MODEL_SHA256, cls="VJEPA2Model",
                   readout="model.encoder(pixel_values_videos=x).last_hidden_state", attn="sdpa",
                   weights="fp32"),
        precision="bf16 autocast (cuda)" if dev_kind == "cuda" else "fp32 (cpu)",
        norm=dict(mean=MEAN, std=STD, rescale="1/255", layout="(B,C,T,H,W) normalised, then (B,T,C,H,W)"),
        ml_blocks=list(ML_BLOCKS), ml_readout="encoder.layernorm(block output[0].float()).mean(tokens)",
        shard_size=a.shard_size, n_shards=len(plan), pass1_shards=[0, n1], pass2_shards=[n1, len(plan)],
        script_sha256=sha256_file(os.path.abspath(__file__)),
        transformers=transformers.__version__, torch=torch.__version__)


def config_key(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------- the model

class VJEPA:
    """Encoder-only V-JEPA 2 with the pre-registered readouts. `run` returns, per view, the fp32
    token mean, the fp32 temporal-token means u [V,8,1024], the LayerNorm-then-mean of blocks 17 /
    19 / 21 and of block 23 (the identity check)."""

    def __init__(s, snap, dev):
        from transformers import VJEPA2Model
        m = VJEPA2Model.from_pretrained(snap, attn_implementation="sdpa", torch_dtype=torch.float32,
                                        local_files_only=True)
        if m.config._attn_implementation != "sdpa":
            raise SystemExit(f"attention implementation is {m.config._attn_implementation}, not sdpa")
        if len(m.encoder.layer) != 24:
            raise SystemExit("expected 24 encoder blocks")
        s.m = m.eval().requires_grad_(False).to(dev)
        s.dev = dev
        s.mean = torch.tensor(MEAN, dtype=torch.float32, device=dev).view(1, 3, 1, 1, 1)
        s.std = torch.tensor(STD, dtype=torch.float32, device=dev).view(1, 3, 1, 1, 1)
        s.store, s.capture = {}, None
        for i in ML_BLOCKS + (LAST_BLOCK,):
            s.m.encoder.layer[i].register_forward_hook(s._hook(i))
        s.m.encoder.embeddings.register_forward_pre_hook(s._pre)
        s.amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if dev.type == "cuda" \
            else contextlib.nullcontext
        s.noamp = (lambda: torch.autocast("cuda", enabled=False)) if dev.type == "cuda" \
            else contextlib.nullcontext

    def _hook(s, i):
        def hook(mod, inp, out):
            s.store[i] = s.m.encoder.layernorm(out[0].float()).mean(1)
        return hook

    def _pre(s, mod, args):
        if s.capture is not None:
            s.capture.append(args[0].detach().clone())

    def normalise(s, u8):
        """(V,C,T,H,W) uint8 -> (V,T,C,H,W) fp32: (u8/255 - mean_c)/std_c, then the permute."""
        x = u8.float().div_(255.0)
        return ((x - s.mean) / s.std).permute(0, 2, 1, 3, 4)

    def run(s, views, chunk, fp32=False):
        out = dict(mean=[], u=[], ml=[], l23=[])
        for c in range(0, len(views), chunk):
            x = s.normalise(views[c:c + chunk])
            with (s.noamp() if fp32 else s.amp()):
                h = s.m.encoder(pixel_values_videos=x).last_hidden_state
            if tuple(h.shape[1:]) != (NTOK, D):
                raise SystemExit(f"unexpected token shape {tuple(h.shape)}; expected (V, {NTOK}, {D})")
            h = h.float()
            out["mean"].append(h.mean(1))
            out["u"].append(h.view(-1, TT, TOK_PER_T, D).mean(2))
            out["ml"].append(torch.stack([s.store[i] for i in ML_BLOCKS], 1))
            out["l23"].append(s.store[LAST_BLOCK])
            s.store.clear()
        return {k: torch.cat(v) for k, v in out.items()}


def aggregate(r, B):
    """Per-clip readouts from the 17B views [sparse B | dense B*K | shuffled B*K], on the device."""
    ds, hs = slice(B, B + B * K), slice(B + B * K, B + 2 * B * K)
    o = dict(sparse=r["mean"][:B], sparse_ml=r["ml"][:B])
    for name, sl in (("dense", ds), ("shuffled", hs)):
        u = r["u"][sl].view(B, K, TT, D)
        o[f"{name}_mean"] = r["mean"][sl].view(B, K, D).mean(1)
        o[f"{name}_tt"] = u
        o[f"{name}_tv"] = (u[:, :, 1:] - u[:, :, :-1]).abs().mean((1, 2))
        o[f"{name}_ml"] = r["ml"][sl].view(B, K, len(ML_BLOCKS), D).mean(1)
    return {k: v.cpu().numpy() for k, v in o.items()}


def rel_err(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-12))


def first_batch_checks(vj, b, views, fed, r, chunk, items):
    """The pre-registered assertions on what the model was actually given (section 8).
    The two identity checks (block-23 hook path, direct model call) must hold within 1e-5
    relative in the fp32 CPU dry run, as registered; both are exact there (0.0). On the GPU both
    sides run the same kernels on the same shapes, so they are expected to be exact too, but a GPU
    segment aborts only above 1e-3 (a wrong block or readout is off by O(1)); the measured value
    is recorded either way."""
    B = b["n"]
    info = {}
    tol = 1e-5 if vj.dev.type == "cpu" else 1e-3
    if fed.shape != (len(views), T, 3, S, S):
        raise SystemExit(f"fed tensor shape {tuple(fed.shape)} != ({len(views)}, {T}, 3, {S}, {S})")
    # (1) normalisation at fixed points, from the CPU uint8 batch (independent of the device path)
    perm = b["perm"].numpy()
    pts = [(5, 100, 37), (0, 0, 0), (T - 1, S - 1, S - 1), (9, 200, 131)]
    worst = 0.0
    for vname, v, src in (("sparse", 0, lambda t: b["sparse"][0][:, t]),
                          ("dense j=3", B + 3, lambda t: b["dense"][0, 3][:, t]),
                          ("shuffled j=5", B + B * K + 5, lambda t: b["dense"][0, 5][:, int(perm[0, 5, t])])):
        for (t, h, w) in pts:
            u8 = src(t)[:, h, w].numpy().astype(np.float64)
            want = (u8 / 255.0 - np.array(MEAN)) / np.array(STD)
            got = fed[v, t, :, h, w].double().cpu().numpy()
            err = float(np.abs(got - want).max())
            worst = max(worst, err)
            if err > 1e-6:
                raise SystemExit(f"fed tensor check failed ({vname}, t={t}, h={h}, w={w}): {got} vs {want}")
    info["fed_norm_max_abs_err"] = worst
    # (2) shuffle: fed shuffled frame t == fed dense frame perm[j][t], exactly
    fd = fed[B:B + B * K].view(B, K, T, 3, S, S)
    fh = fed[B + B * K:].view(B, K, T, 3, S, S)
    pm = b["perm"].to(fed.device)
    for i in range(B):
        for j in range(K):
            if not torch.equal(fh[i, j], fd[i, j][pm[i, j]]):
                raise SystemExit(f"shuffled view of clip {b['row'][i]} snippet {j} is not the dense "
                                 f"snippet permuted by perm[{j}]")
    info["shuffle_views_checked"] = B * K
    info["perm_identity_snippets"] = int(sum((perm[i, j] == np.arange(T)).all()
                                             for i in range(B) for j in range(K)))
    # (3) anchor permutation, when the anchor clip is in the batch
    for i, row in enumerate(b["row"]):
        if items[row][0] == ANCHOR_PATH:
            if perm[i, 0].tolist() != ANCHOR_PERM0:
                raise SystemExit(f"anchor clip perm[0] {perm[i, 0].tolist()} != {ANCHOR_PERM0}")
            info["anchor_perm0_checked"] = True
    # (4) ML hook path on block 23 reproduces the mean of last_hidden_state
    e23 = rel_err(r["l23"], r["mean"])
    info["block23_hook_vs_last_hidden_rel_err"] = e23
    if e23 > tol:
        raise SystemExit(f"block-23 LayerNorm-then-mean differs from mean(last_hidden_state): rel {e23:.3g}")
    # (5) a direct transformers call on the first chunk's fed tensor
    n = min(chunk, len(views))
    x = fed[:n]
    with vj.amp():
        h = vj.m(pixel_values_videos=x, skip_predictor=True).last_hidden_state
    vj.store.clear()
    ed = rel_err(h.float().mean(1), r["mean"][:n])
    info["direct_model_call_rel_err"] = ed
    info["identity_tolerance"] = tol
    if ed > tol:
        raise SystemExit(f"model(x, skip_predictor=True) token means differ from the extractor's: rel {ed:.3g}")
    info["token_shape"] = [len(views), NTOK, D]
    return info


def _rows_rel_l2(a, b):
    """Per-row ||a - b|| / ||b|| (float64) of two [n, ...] arrays."""
    a, b = np.asarray(a, np.float64).reshape(len(a), -1), np.asarray(b, np.float64).reshape(len(b), -1)
    return np.linalg.norm(a - b, axis=1) / np.maximum(np.linalg.norm(b, axis=1), 1e-300)


def _rows_one_minus_cos(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return 1.0 - (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-300)


def precision_check(vj, views, r, B, chunk):
    """GPU: rerun the first batch in fp32 (TF32 off) and compare (section 8). The registered gate is
    the per-view token-mean cosine: `min_cos` below 0.99 aborts (the caller). Recorded as well,
    never gating: the error of the temporal-token means u and of the clip-level readouts the probes
    use (*_mean, *_tv, *_ml), and, as the scale that error must stay far below, the fp32 gap between
    the dense and shuffled readouts of the same clips. `bf16_over_gap` = (largest bf16-vs-fp32 error)
    / (smallest fp32 dense-vs-shuffled gap), once for 1 - cos of *_mean and once for the relative L2
    of *_tv. For *_mean (the decision's readout) it should be << 1; a value near 1 would mean bf16
    error as large as the M signal. A CPU bf16 simulation on one seizure clip gave 0.003 (*_mean)
    and 0.13 (*_tv, the non-gating TT readout)."""
    tf = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        r32 = vj.run(views, chunk, fp32=True)
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf
    cos = torch.nn.functional.cosine_similarity(r["mean"], r32["mean"], dim=1).cpu().numpy()
    out = {}
    for name, sl in (("sparse", slice(0, B)), ("dense", slice(B, B + B * K)),
                     ("shuffled", slice(B + B * K, B + 2 * B * K))):
        c = cos[sl].astype(float)
        out[name] = dict(n=len(c), min=float(c.min()), median=float(np.median(c)), max=float(c.max()))
    out["ml_block_rel_err_max"] = rel_err(r["ml"], r32["ml"])
    out["min_cos"] = float(cos.min())
    out["one_minus_min_cos"] = float(1.0 - cos.min())
    # --- record-only (added after review): u, *_tv, clip-level readouts, and the dense-vs-shuffled scale
    out["u_rel_err_max"] = rel_err(r["u"], r32["u"])
    out["u_rel_l2_max"] = float(_rows_rel_l2(r["u"].cpu().numpy(), r32["u"].cpu().numpy()).max())
    a16, a32 = aggregate(r, B), aggregate(r32, B)
    clip = {}
    for k in ("sparse", "dense_mean", "shuffled_mean", "dense_tv", "shuffled_tv", "dense_ml", "shuffled_ml"):
        clip[k] = dict(rel_l2_max=float(_rows_rel_l2(a16[k], a32[k]).max()))
        if k in ("sparse", "dense_mean", "shuffled_mean"):
            clip[k]["one_minus_cos_max"] = float(_rows_one_minus_cos(a16[k], a32[k]).max())
    out["clip_level_bf16_vs_fp32"] = clip
    gap_mean = _rows_one_minus_cos(a32["dense_mean"], a32["shuffled_mean"])
    gap_tv = _rows_rel_l2(a32["dense_tv"], a32["shuffled_tv"])
    err_mean = max(clip["dense_mean"]["one_minus_cos_max"], clip["shuffled_mean"]["one_minus_cos_max"])
    err_tv = max(clip["dense_tv"]["rel_l2_max"], clip["shuffled_tv"]["rel_l2_max"])
    out["fp32_dense_vs_shuffled_gap"] = dict(
        mean_one_minus_cos=dict(min=float(gap_mean.min()), median=float(np.median(gap_mean))),
        tv_rel_l2=dict(min=float(gap_tv.min()), median=float(np.median(gap_tv))))
    out["bf16_over_gap"] = dict(mean=float(err_mean / max(gap_mean.min(), 1e-300)),
                                tv=float(err_tv / max(gap_tv.min(), 1e-300)))
    return out


# ----------------------------------------------------------------------------- budget

class Budget:
    """Per-segment GPU-time records under <out>/budget, the aggregate budget.json, and sacct."""

    def __init__(s, out, cap_hours, gpu_name, pass_of_shard, shard_path):
        s.out, s.dir = out, os.path.join(out, "budget")
        os.makedirs(s.dir, exist_ok=True)
        s.cap = cap_hours * 3600.0
        job = os.environ.get("SLURM_JOB_ID", "")
        restart = int(os.environ.get("SLURM_RESTART_COUNT", "0") or 0)
        s.seg_id = f"{job or 'local'}_r{restart}_{socket.gethostname()}_{os.getpid()}"
        t0 = os.environ.get("VJ_SEGMENT_T0")
        s.rec = dict(seg_id=s.seg_id, job_id=job, restart=restart, host=socket.gethostname(),
                     pid=os.getpid(), gpu=gpu_name, t_start=float(t0) if t0 else time.time(),
                     t_start_source="VJ_SEGMENT_T0 (driver start)" if t0 else "process start",
                     t_last=time.time(), state="running", shards=[], clips=0,
                     precision_check=None, first_batch_checks=None, notes=[])
        s.lock = threading.RLock()
        s._sacct = (None, 0.0, None)            # (seconds, query time, detail)
        s.pass_of_shard, s.shard_path = pass_of_shard, shard_path
        s.path = os.path.join(s.dir, f"seg_{s.seg_id}.json")

    def segments(s):
        out = []
        for f in sorted(os.listdir(s.dir)):
            if f.startswith("seg_") and f.endswith(".json"):
                try:
                    out.append(json.load(open(os.path.join(s.dir, f))))
                except Exception:
                    pass                                          # a torn read; the next write fixes it
        return out

    def job_ids(s, segs):
        """Job ids from the segment records and from the driver's budget/job_<id>_r<n>.txt files
        (written before Python starts, so a job killed during start-up is still counted)."""
        ids = {g["job_id"] for g in segs if g.get("job_id")}
        for f in os.listdir(s.dir):
            m = re.match(r"job_(\d+)_r\d+\.txt$", f)
            if m:
                ids.add(m.group(1))
        return sorted(ids)

    def sacct_seconds(s, segs, max_age=300.0):
        """sacct -D (every run of a requeued job) elapsed x allocated GPUs; None if unavailable."""
        val, t, _ = s._sacct
        if val is not None and time.time() - t < max_age:
            return s._sacct
        ids = s.job_ids(segs)
        if not ids:
            s._sacct = (None, time.time(), "no job ids")
            return s._sacct
        try:
            p = subprocess.run(["sacct", "-D", "-X", "-n", "-P", "-j", ",".join(ids), "-o",
                                "JobIDRaw,ElapsedRaw,AllocTRES,State"],
                               capture_output=True, text=True, timeout=60)
            if p.returncode != 0:
                raise RuntimeError(p.stderr.strip()[:200])
            tot, rows = 0.0, []
            for ln in p.stdout.strip().splitlines():
                jid, el, tres, state = ln.split("|")[:4]
                m = re.search(r"gres/gpu=(\d+)", tres)
                g = int(m.group(1)) if m else 1                      # conservative when unlisted
                tot += float(el or 0) * g
                rows.append([jid, int(el or 0), g, state])
            s._sacct = (tot, time.time(), rows)
        except Exception as e:
            s._sacct = (None, time.time(), f"sacct unavailable: {type(e).__name__}: {e}")
        return s._sacct

    def totals(s, now=None):
        now = now or time.time()
        segs = s.segments()
        ours = [g for g in segs if g["seg_id"] != s.seg_id] + [s.rec]
        el, live = 0.0, 0
        for g in ours:
            alive = g["seg_id"] == s.seg_id or (g.get("state") == "running" and now - g["t_last"] < 900)
            el += (now if alive else g["t_last"]) - g["t_start"]
            live += alive
        sa, _, detail = s.sacct_seconds(ours)
        if sa is not None and sa > el:
            return sa, "sacct", live, el, sa
        return el, "budget records" + ("" if sa is not None else f" ({detail})"), live, el, sa

    def write(s, state=None):
        with s.lock:
            s.rec["t_last"] = time.time()
            if state:
                s.rec["state"] = state
            C.atomic_json(s.path, s.rec)
            segs = s.segments()
            tot = sum(g["t_last"] - g["t_start"] for g in segs)
            sa, tq, detail = s._sacct
            C.atomic_json(os.path.join(s.out, "budget.json"), dict(
                cap_gpu_hours=s.cap / 3600, total_gpu_hours_budget_records=tot / 3600,
                sacct_gpu_hours=None if sa is None else sa / 3600, sacct_detail=detail,
                sacct_queried=tq, updated=time.time(), updated_by=s.seg_id, segments=segs))

    def may_start(s, k, est):
        """Section 8's rule, extended for parallel jobs (see the module docstring)."""
        tot, src, live, el, sa = s.totals()
        reserve = 0
        if s.pass_of_shard(k) == 2:
            reserve = sum(1 for j in range(s.n_shards) if s.pass_of_shard(j) == 1
                          and not os.path.exists(s.shard_path(j)))
        proj = tot + (live + reserve) * est
        return proj <= s.cap, dict(total_h=tot / 3600, source=src, live_segments=live,
                                   est_shard_s=est, pass1_reserve_shards=reserve, projected_h=proj / 3600)


# ----------------------------------------------------------------------------- shards

def new_row():
    r = {k: np.full(shape, np.nan, dt) for k, (shape, dt) in SHAPES.items()}
    r.update(perm=np.full((K, T), -1, np.int8), starts=np.full(K, -1, np.int64), n_header=-1,
             single=False, err="", seg="")
    return r


def check_rows(z, s2, where):
    """Consistency of one shard's arrays (section 2); returns the good-row mask."""
    err = np.asarray(z["err"])
    good = err == ""
    for k in FLOAT_KEYS:
        a = np.asarray(z[k])
        fin = np.isfinite(a.reshape(len(a), -1)).all(1)
        if not fin[good].all():
            raise SystemExit(f"{where}: non-finite {k} in {int((~fin[good]).sum())} non-failed rows")
        if np.isfinite(a[~good]).any():
            raise SystemExit(f"{where}: a failed row has finite {k}")
    for name in ("dense", "shuffled"):
        tt = np.asarray(z[f"{name}_tt"])[good].astype(np.float64)          # [n,K,TT,D]
        tol = 1e-3 * (1 + np.abs(tt).max(axis=(1, 2)))
        m = np.asarray(z[f"{name}_mean"])[good].astype(np.float64)
        if (np.abs(tt.mean(axis=(1, 2)) - m) > tol).any():
            raise SystemExit(f"{where}: {name}_tt does not average to {name}_mean")
        v = np.abs(np.diff(tt, axis=2)).mean(axis=(1, 2))
        tv = np.asarray(z[f"{name}_tv"])[good].astype(np.float64)
        if (np.abs(v - tv) > tol).any():
            raise SystemExit(f"{where}: v recomputed from f16 {name}_tt disagrees with {name}_tv")
    paths = np.asarray(z["path"])
    for i in np.flatnonzero(good):
        p = str(paths[i])
        if not np.array_equal(np.asarray(z["perm"][i]), perms_for(p).astype(np.int8)):
            raise SystemExit(f"{where}: stored perm of {p} is not the seed rule's")
        mm = stage2_mismatch(s2, p, z["starts"][i], z["n_header"][i], z["single_snippet"][i])
        if mm:
            raise SystemExit(f"{where}: {p}: {mm}")
    return good


# ----------------------------------------------------------------------------- prepare / extract / merge

def prepare(a, out, items_full, items, plan, dev_kind):
    cfg = make_config(a, items, plan, dev_kind, C.fingerprint([i[0] for i in items_full]))
    s2 = load_stage2()
    miss = [i[0] for i in items if i[0] not in s2]
    if miss:
        raise SystemExit(f"{len(miss)} items are not in Stage 2's features.npz, e.g. {miss[0]}")
    snap, snapinfo = snapshot_dir()
    t0 = time.time()                                  # the blob name is the etag; hash the content too
    content = sha256_file(os.path.join(snap, "model.safetensors"))
    if content != MODEL_SHA256:
        raise SystemExit(f"model.safetensors content sha256 {content} != pinned {MODEL_SHA256}")
    snapinfo.update(safetensors_sha256_content=content, safetensors_hash_seconds=round(time.time() - t0, 1))
    log(f"model.safetensors content sha256 verified ({snapinfo['safetensors_hash_seconds']} s)")
    vs = None
    if a.verify_seek:
        rows = sorted(np.random.default_rng(0).choice(len(items), min(a.verify_seek, len(items)),
                                                      replace=False).tolist())
        t0 = time.time()
        PE.verify_seek(items, rows, size=S)
        vs = dict(n=len(rows), rows=rows, size=S, ok=True, seconds=round(time.time() - t0, 1))
    n1, n_sh = cfg["pass1_shards"][1], len(plan)
    doc = dict(config=cfg, config_key=config_key(cfg),
               prepared=dict(time=time.strftime("%F %T"), host=socket.gethostname(), verify_seek=vs,
                             snapshot=snapinfo, versions=versions(), code=code_identity(),
                             stage2=dict(path=STAGE2, items_found=len(items)),
                             pass_rows={"1": sum(len(r) for p, r in plan if p == 1),
                                        "2": sum(len(r) for p, r in plan if p == 2)}))
    C.atomic_json(os.path.join(out, "run.json"), doc)
    rng = lambda lo, hi: f"shards {lo}..{hi - 1}" if hi > lo else "no shards"      # noqa: E731
    log(f"prepared {out}/run.json: config {doc['config_key']}, {n_sh} shards (pass 1 = {rng(0, n1)}, "
        f"{doc['prepared']['pass_rows']['1']} rows; pass 2 = {rng(n1, n_sh)}, "
        f"{doc['prepared']['pass_rows']['2']} rows); verify_seek {vs}")
    return doc


def extract(a, out, items, plan, doc, dev, s2):
    key = doc["config_key"]
    fp = doc["config"]["items"]["fingerprint"]
    n_sh = len(plan)
    shard_of = {r: k for k, (_, rows) in enumerate(plan) for r in rows}
    sd = os.path.join(out, "shards")
    os.makedirs(sd, exist_ok=True)
    shard_path = lambda k: os.path.join(sd, f"shard_{k:05d}.npz")          # noqa: E731
    lo, hi = a.shard_start, (n_sh if a.shard_stop is None else min(a.shard_stop, n_sh))
    if not (0 <= lo < hi <= n_sh):
        raise SystemExit(f"--shard_start/--shard_stop [{lo}, {hi}) outside [0, {n_sh})")
    gpu = torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"
    budget = Budget(out, a.budget_hours, gpu, lambda k: plan[k][0], shard_path)
    budget.n_shards = n_sh
    code_now = code_sha256()                  # recorded per segment; the merge checks they all agree
    budget.rec["code"] = code_now
    code_prep = (doc["prepared"].get("code") or {}).get("sha256") or {}
    main_pid, stopping = os.getpid(), [False]

    def on_term(signum, frame):
        if os.getpid() != main_pid:                    # a DataLoader worker: die the default way
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
            return
        if stopping[0]:
            return
        stopping[0] = True
        log(f"signal {signum}: recording budget and stopping (finished shards are kept)")
        budget.write(state="sigterm")
        raise SystemExit(EXIT_SIGTERM)

    signal.signal(signal.SIGTERM, on_term)
    budget.write()
    if code_now != code_prep:
        changed = sorted(f for f in CODE_FILES if code_now.get(f) != code_prep.get(f))
        budget.write(state="abort")
        raise SystemExit(f"REFUSING: {changed} changed since --prepare (run.json); every shard of one run must "
                         f"come from one code state. Restore the file(s), or prepare a fresh --out_dir")
    tot, src, live, el, sa = budget.totals()
    log(f"segment {budget.seg_id} on {gpu}: shards [{lo}, {hi}); GPU time so far {tot / 3600:.3f} h "
        f"(source: {src}; budget records {el / 3600:.3f} h, sacct "
        f"{'n/a' if sa is None else f'{sa / 3600:.3f} h'}) of the {a.budget_hours:g} h cap")

    def keeper():
        while True:
            time.sleep(300)
            try:
                budget.write()
            except Exception as e:                     # never kill the run over a budget write
                print(f"budget write failed: {e}", flush=True)

    threading.Thread(target=keeper, daemon=True).start()

    done, buf, retry, kept_failed = set(), {}, [], 0
    for k in range(lo, hi):
        f = shard_path(k)
        if not os.path.exists(f):
            continue
        z = np.load(f)
        if str(z["fingerprint"]) != fp or str(z["config_key"]) != key or z["rows"].tolist() != plan[k][1]:
            raise SystemExit(f"{f} was written for a different item list or config; use a fresh --out_dir")
        bad = np.flatnonzero(z["err"] != "")
        if len(bad) and a.retry_failed:
            z = {kk: z[kk] for kk in z.files}
            for j, r in enumerate(plan[k][1]):
                if z["err"][j] == "":
                    x = {kk: z[kk][j] for kk in FLOAT_KEYS}
                    x.update(perm=z["perm"][j], starts=z["starts"][j], n_header=int(z["n_header"][j]),
                             single=bool(z["single_snippet"][j]), err="", seg=str(z["seg_id"][j]))
                    buf[r] = x
                else:
                    retry.append(r)
        else:
            done.add(k)
            kept_failed += len(bad)
    todo_shards = [k for k in range(lo, hi) if k not in done]
    todo = {k: [r for r in plan[k][1] if r not in buf] for k in todo_shards}
    n_todo = sum(len(v) for v in todo.values())
    log(f"{len(done)}/{hi - lo} shards of the range already done; {n_todo} clips to extract"
        + (f" (incl. {len(retry)} failed rows re-tried)" if retry else ""))
    if kept_failed:
        log(f"note: {kept_failed} failed (NaN) rows in finished shards are kept; pass --retry_failed")
    if not n_todo:
        budget.write(state="done")
        return 0

    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    snap, _ = snapshot_dir()
    vj = VJEPA(snap, dev)
    log(f"V-JEPA 2 loaded ({REPO}@{REV[:8]}, sdpa, fp32 weights, "
        f"{'bf16 autocast' if dev.type == 'cuda' else 'fp32'})")

    flat, batches = [], []
    for k in todo_shards:
        rows = todo[k]
        for i in range(0, len(rows), a.batch_clips):
            batches.append((k, list(range(len(flat) + i, len(flat) + min(i + a.batch_clips, len(rows))))))
        flat += rows
    ds = PE.ProbeClips(items, flat, None, a.decode_retries, size=S)
    dl = DataLoader(ds, batch_sampler=[b for _, b in batches], num_workers=a.workers,
                    collate_fn=PE.collate, pin_memory=(dev.type == "cuda"),
                    prefetch_factor=2 if a.workers > 0 else None)
    shard_t0, rate = {}, None
    rec_rates = [g["seconds"] / g["rows"] for seg in budget.segments() for g in seg.get("shards", [])
                 if g.get("rows", 0) >= 64]
    if rec_rates:
        rate = max(rec_rates)

    def flush_ready():
        nonlocal rate
        for k in sorted({shard_of[r] for r in buf}):
            rows = plan[k][1]
            if not all(r in buf for r in rows):
                continue
            R = [buf.pop(r) for r in rows]
            arr = dict(rows=np.array(rows), fingerprint=np.array(fp), config_key=np.array(key),
                       pass_id=np.int64(plan[k][0]), path=np.array([items[r][0] for r in rows]),
                       y5=np.array([items[r][1] for r in rows]),
                       perm=np.stack([x["perm"] for x in R]).astype(np.int8),
                       starts=np.stack([x["starts"] for x in R]).astype(np.int64),
                       n_header=np.array([x["n_header"] for x in R], np.int64),
                       single_snippet=np.array([x["single"] for x in R], bool),
                       err=np.array([x["err"] for x in R]), seg_id=np.array([x["seg"] for x in R]),
                       segment=np.array(json.dumps(budget.rec)))
            for kk in FLOAT_KEYS:
                arr[kk] = np.stack([x[kk] for x in R]).astype(SHAPES[kk][1])
            check_rows(arr, s2, f"shard {k}")
            C.atomic_npz(shard_path(k), **arr)
            n_new = sum(1 for r in rows if r in set(todo.get(k, [])))
            dt = time.time() - shard_t0.get(k, time.time())
            budget.rec["shards"].append(dict(shard=k, pass_id=plan[k][0], rows=n_new, seconds=round(dt, 1),
                                             failed=int((arr["err"] != "").sum()), t=time.time()))
            budget.rec["clips"] += n_new
            if n_new >= 64:
                rate = dt / n_new
            budget.write()
            log(f"  wrote shard {k} (pass {plan[k][0]}, {len(rows)} rows, {n_new} extracted, "
                f"{int((arr['err'] != '').sum())} failed) in {dt:.0f} s")

    t0, n_seen, last_print, first, cur = time.time(), 0, time.time(), True, None
    stop = None
    with torch.inference_mode():
        for (k, _), b in zip(batches, dl):
            if k != cur:
                cur = k
                est = (rate or a.default_shard_seconds / a.shard_size) * len(plan[k][1])
                ok, info = budget.may_start(k, est)
                if not ok:
                    stop = dict(shard=k, **info)
                    break
                shard_t0[k] = time.time()
            for r in b["retried"]:
                log(f"  decoded after retry: {items[r][0]}")
            for r, err in b["bad"]:
                x = new_row()
                x.update(err=err, seg=budget.seg_id)
                buf[r] = x
                log(f"  FAIL {items[r][0]}: {err}")
            if b["n"]:
                B = b["n"]
                xs = b["sparse"].to(dev, non_blocking=True)
                xd = b["dense"].to(dev, non_blocking=True).flatten(0, 1)             # (B*K,3,T,H,W)
                pm = b["perm"].to(dev).flatten(0, 1)                                 # (B*K,T)
                xsh = torch.stack([xd[i][:, pm[i]] for i in range(B * K)])
                views = torch.cat([xs, xd, xsh], 0)
                if first:
                    vj.capture = []
                r = vj.run(views, a.chunk_views)
                if first:
                    fed = torch.cat(vj.capture)
                    vj.capture = None
                    chk = first_batch_checks(vj, b, views, fed, r, a.chunk_views, items)
                    del fed
                    if dev.type == "cuda":
                        pc = precision_check(vj, views, r, B, a.chunk_views)
                        chk["precision"] = pc
                        budget.rec["precision_check"] = pc
                        log(f"precision check (bf16 vs fp32 token-mean cosine): "
                            f"{json.dumps({k2: v for k2, v in pc.items()})}")
                        log(f"precision vs the M scale (record only): 1 - min cos {pc['one_minus_min_cos']:.3g}; "
                            f"bf16 error / fp32 dense-vs-shuffled gap = {pc['bf16_over_gap']['mean']:.3g} for "
                            f"*_mean (1 - cos; the decision's readout, should be << 1), "
                            f"{pc['bf16_over_gap']['tv']:.3g} for *_tv (rel L2; the non-gating TT readout, "
                            f"0.13 in a CPU bf16 simulation)")
                        if pc["min_cos"] < 0.99:
                            budget.write(state="abort")
                            raise SystemExit(f"ABORT: bf16-vs-fp32 cosine {pc['min_cos']:.5f} < 0.99")
                    budget.rec["first_batch_checks"] = chk
                    budget.write()
                    log(f"first-batch checks passed: {json.dumps(chk)}")
                    first = False
                o = aggregate(r, B)
                for j in range(B):
                    row = b["row"][j]
                    mm = stage2_mismatch(s2, items[row][0], b["starts"][j].numpy(), b["n_header"][j],
                                         b["single"][j])
                    if mm:
                        raise SystemExit(f"ABORT: {items[row][0]}: {mm}")
                    x = new_row()
                    for kk in FLOAT_KEYS:
                        x[kk] = o[kk][j].astype(SHAPES[kk][1])
                    x.update(perm=b["perm"][j].numpy().astype(np.int8), starts=b["starts"][j].numpy(),
                             n_header=int(b["n_header"][j]), single=bool(b["single"][j]), err="",
                             seg=budget.seg_id)
                    buf[row] = x
            n_seen += b["n"] + len(b["bad"])
            if time.time() - last_print > 60 or n_seen == n_todo:
                last_print = time.time()
                log(f"  {n_seen}/{n_todo} clips  {n_seen / (time.time() - t0):.2f} clips/s")
            flush_ready()
    del dl
    flush_ready()
    if stop:
        n1 = doc["config"]["pass1_shards"][1]
        have = [k for k in range(n_sh) if os.path.exists(shard_path(k))]
        p1 = sum(len(plan[k][1]) for k in have if k < n1)
        p2 = sum(len(plan[k][1]) for k in have if k >= n1)
        budget.rec["notes"].append(dict(budget_stop=stop))
        budget.write(state="budget_stop")
        log(f"BUDGET CAP: shard {stop['shard']} not started: {json.dumps(stop)}")
        log(f"done so far: pass 1 {p1}/{sum(len(r) for p, r in plan if p == 1)} clips, pass 2 "
            f"{p2}/{sum(len(r) for p, r in plan if p == 2)} clips. "
            + ("Pass 1 is INCOMPLETE: no Step-0 decision on a partial set (section 8)."
               if p1 < sum(len(r) for p, r in plan if p == 1)
               else "Pass 1 is complete: merge with --allow_pass2_incomplete."))
        return EXIT_BUDGET
    if buf:
        raise SystemExit(f"internal error: {len(buf)} rows never flushed")
    budget.write(state="done")
    log(f"range [{lo}, {hi}) complete: {budget.rec['clips']} clips extracted in this segment")
    return 0


def merge(a, out, items, plan, doc, s2):
    fp, key = doc["config"]["items"]["fingerprint"], doc["config_key"]
    N, n_sh = len(items), len(plan)
    n1 = doc["config"]["pass1_shards"][1]
    f_out = os.path.join(out, "features.npz")
    if os.path.exists(f_out) and not a.overwrite:
        raise SystemExit(f"{f_out} exists; pass --overwrite to rebuild it")
    sp = lambda k: os.path.join(out, "shards", f"shard_{k:05d}.npz")          # noqa: E731
    missing = [k for k in range(n_sh) if not os.path.exists(sp(k))]
    m1 = [k for k in missing if k < n1]
    if m1:
        raise SystemExit(f"REFUSING to merge: pass-1 shards missing {m1}; no decision on a partial "
                         f"seizure set (section 8)")
    if missing and not a.allow_pass2_incomplete:
        raise SystemExit(f"REFUSING to merge: pass-2 shards missing {missing}; pass "
                         f"--allow_pass2_incomplete only if the budget stopped pass 2 (section 8)")
    res = {k: np.full((N,) + shape, np.nan, dt) for k, (shape, dt) in SHAPES.items()}
    perm = np.full((N, K, T), -1, np.int8)
    starts = np.full((N, K), -1, np.int64)
    n_header = np.full(N, -1, np.int64)
    single = np.zeros(N, bool)
    err = np.array([""] * N, dtype=object)
    seg_ids, seg_docs = set(), {}
    for k in range(n_sh):
        if k in missing:
            continue
        z = np.load(sp(k))
        rows = plan[k][1]
        if str(z["fingerprint"]) != fp:
            raise SystemExit(f"REFUSING: shard {k} fingerprint {z['fingerprint']} != {fp}")
        if str(z["config_key"]) != key:
            raise SystemExit(f"REFUSING: shard {k} config {z['config_key']} != run.json's {key}")
        if z["rows"].tolist() != rows or int(z["pass_id"]) != plan[k][0]:
            raise SystemExit(f"REFUSING: shard {k} rows / pass differ from the plan")
        if z["path"].tolist() != [items[r][0] for r in rows] or z["y5"].tolist() != [items[r][1] for r in rows]:
            raise SystemExit(f"REFUSING: shard {k} paths or labels differ from the item list")
        zz = {kk: z[kk] for kk in z.files}
        check_rows(zz, s2, f"shard {k}")
        ix = np.array(rows)
        for kk in FLOAT_KEYS:
            res[kk][ix] = zz[kk]
        perm[ix], starts[ix], n_header[ix], single[ix] = zz["perm"], zz["starts"], zz["n_header"], zz["single_snippet"]
        err[ix] = zz["err"]
        seg_ids |= set(zz["seg_id"].tolist()) - {""}
        sd = json.loads(str(zz["segment"]))
        seg_docs[sd["seg_id"]] = sd
    not_ext = np.array(sorted(r for k in missing for r in plan[k][1]), np.int64)
    paths = np.array([i[0] for i in items])
    err = err.astype(str)
    failed = paths[err != ""]
    attempted = N - len(not_ext)
    log(f"merge: {N} clips, {attempted} attempted, {len(failed)} failed, {len(not_ext)} not extracted "
        f"(pass-2 shards {missing}), single-snippet clips {int(single.sum())} (among the "
        f"{attempted - len(failed)} extracted rows)")
    if len(failed) > a.max_fail_frac * attempted:
        raise SystemExit(f"ABORT: {len(failed)}/{attempted} clips failed (> {100 * a.max_fail_frac:.1f}%); "
                         f"shards kept, features.npz NOT written (re-run with --retry_failed)")
    bdir = os.path.join(out, "budget")
    segs = []
    if os.path.isdir(bdir):
        for f in sorted(os.listdir(bdir)):
            if f.startswith("seg_") and f.endswith(".json"):
                segs.append(json.load(open(os.path.join(bdir, f))))
    for g in segs:                                         # the fuller record of each segment
        seg_docs[g["seg_id"]] = g
    # every segment that wrote a row must have run the code --prepare recorded (it refuses to start
    # otherwise; this is the backstop)
    code_prep = (doc["prepared"].get("code") or {}).get("sha256") or {}
    bad_code = {s: (seg_docs.get(s) or {}).get("code") for s in sorted(seg_ids)
                if (seg_docs.get(s) or {}).get("code") != code_prep}
    if bad_code:
        raise SystemExit(f"REFUSING to merge: segment(s) {sorted(bad_code)} have no code record or ran other "
                         f"code than run.json's --prepare recorded: {json.dumps(bad_code)[:800]}")
    b = Budget.__new__(Budget)
    b.dir, b._sacct = bdir, (None, 0.0, None)
    sacct = b.sacct_seconds(segs) if os.path.isdir(bdir) else (None, 0, "no budget dir")
    code_now = code_identity()
    shas = {g.get("seg_id"): g for g in seg_docs.values()}
    prov = dict(
        created=time.strftime("%F %T"), host=socket.gethostname(), config=doc["config"], config_key=key,
        prepared=doc["prepared"], merge_versions=versions(), merge_code=code_now,
        merge_script_matches_run=code_now["sha256"]["step0_vjepa_extract.py"] == doc["config"]["script_sha256"],
        merge_code_matches_prepare=code_now["sha256"] == code_prep,
        segment_code_check=f"all {len(seg_ids)} segments with rows ran the prepared code {code_prep}",
        segments=[seg_docs[s] for s in sorted(seg_docs)], segments_with_rows=sorted(seg_ids),
        precision_checks={s: g.get("precision_check") for s, g in shas.items()},
        budget=dict(cap_gpu_hours=a.budget_hours,
                    budget_records_gpu_hours=sum(g["t_last"] - g["t_start"] for g in segs) / 3600,
                    sacct_gpu_hours=None if sacct[0] is None else sacct[0] / 3600, sacct_detail=sacct[2]),
        counts=dict(N=N, attempted=attempted, failed=len(failed), not_extracted=len(not_ext),
                    single_snippet=int(single.sum()),
                    single_snippet_scope="among extracted rows (failed and not-extracted rows count 0)",
                    pass1_rows=sum(len(r) for p, r in plan if p == 1),
                    pass2_rows=sum(len(r) for p, r in plan if p == 2)),
        stage2_check="starts / n_header / single_snippet equal Stage 2's for every extracted row",
        consistency="*_tt vs *_mean, f16 v vs *_tv, perm vs seed rule, finiteness: all rows passed")
    geometry = dict(K=K, T=T, stride=STRIDE, size=S, buffer=BUFFER, model=f"{REPO}@{REV}",
                    precision=doc["config"]["precision"], items="discover", limit=doc["config"]["items"]["limit"],
                    rows=doc["config"]["items"]["rows"])
    C.atomic_npz(f_out, path=paths, y5=np.array([i[1] for i in items]), **res, perm=perm, starts=starts,
                 n_header=n_header, single_snippet=single, err=err, failed=failed,
                 not_extracted=paths[not_ext] if len(not_ext) else np.array([], dtype=paths.dtype),
                 fingerprint=np.array(fp), feature_dim=np.int64(D), ml_blocks=np.array(ML_BLOCKS),
                 geometry=np.array(json.dumps(geometry)), provenance=np.array(json.dumps(prov, default=str)))
    log(f"wrote {f_out}  (GPU time: budget records {prov['budget']['budget_records_gpu_hours']:.3f} h, "
        f"sacct {'n/a' if sacct[0] is None else f'{sacct[0] / 3600:.3f} h'})")
    return 0


def status(out, plan, doc):
    n1 = doc["config"]["pass1_shards"][1]
    have = [k for k in range(len(plan)) if os.path.exists(os.path.join(out, "shards", f"shard_{k:05d}.npz"))]
    for p, rng in ((1, range(0, n1)), (2, range(n1, len(plan)))):
        d = [k for k in rng if k in have]
        print(f"pass {p}: {len(d)}/{len(rng)} shards, {sum(len(plan[k][1]) for k in d)}/"
              f"{sum(len(plan[k][1]) for k in rng)} clips; missing {[k for k in rng if k not in have]}")
    bj = os.path.join(out, "budget.json")
    if os.path.exists(bj):
        j = json.load(open(bj))
        print(f"budget.json: {j['total_gpu_hours_budget_records']:.3f} GPU-h from records, sacct "
              f"{j['sacct_gpu_hours']} (cap {j['cap_gpu_hours']}); segments "
              f"{[(g['seg_id'], g['state']) for g in j['segments']]}")
    print(f"features.npz: {'present' if os.path.exists(os.path.join(out, 'features.npz')) else 'absent'}")


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default=None, help=f"default {REAL_OUT} (ttg_tmp/vjepa_dryrun with --dry_run)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true",
                      help="CPU: item list, --verify_seek, run.json; no model forward")
    mode.add_argument("--merge", action="store_true", help="CPU: assemble features.npz from the shards")
    mode.add_argument("--status", action="store_true", help="print shard and budget status")
    ap.add_argument("--dry_run", action="store_true",
                    help="CPU fp32 test: prepare + extract + merge on --limit (default 3) clips")
    ap.add_argument("--limit", type=int, default=0, help="first N sorted clips only (tests)")
    ap.add_argument("--rows", default="", help="comma-separated sorted-list indices only (tests)")
    ap.add_argument("--items_npz", default=None,
                    help="reuse another run's items.npz instead of globbing (fingerprint-checked)")
    ap.add_argument("--shard_size", type=int, default=512)
    ap.add_argument("--shard_start", type=int, default=0)
    ap.add_argument("--shard_stop", type=int, default=None, help="exclusive; default all shards")
    ap.add_argument("--batch_clips", type=int, default=4, help="clips per batch (x17 views)")
    ap.add_argument("--chunk_views", type=int, default=None, help="views per forward (default 68 GPU, 4 CPU)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = torch default)")
    ap.add_argument("--verify_seek", type=int, default=None,
                    help="with --prepare / --dry_run (or when run.json is missing): check N random "
                         "clips, seek decode == sequential decode at 256 (default 20; dry run: all)")
    ap.add_argument("--decode_retries", type=int, default=2)
    ap.add_argument("--retry_failed", action="store_true")
    ap.add_argument("--max_fail_frac", type=float, default=0.005)
    ap.add_argument("--budget_hours", type=float, default=8.0, help="the pre-registered cap (8)")
    ap.add_argument("--default_shard_seconds", type=float, default=300.0,
                    help="shard-duration estimate before any shard has been measured")
    ap.add_argument("--allow_pass2_incomplete", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="--merge: rebuild an existing features.npz")
    a = ap.parse_args()

    C.enter_eeg_root()
    if a.threads:
        torch.set_num_threads(a.threads)
    if a.dry_run:
        a.limit = a.limit or (0 if a.rows else 3)
        a.out_dir = a.out_dir or os.path.join(C.EEG_ROOT, "output", "ttg_tmp", "vjepa_dryrun")
    out = check_out_dir(a.out_dir or REAL_OUT)
    if a.dry_run and os.path.exists(os.path.join(out, "features.npz")):
        # a dry run over a finished one would re-run verify_seek, extract nothing and exit 0: a stale
        # pass that looks like a fresh dry run
        raise SystemExit(f"{out}/features.npz exists: a dry run into it would extract nothing. For a fresh dry "
                         f"run remove the directory first (rm -rf {out})")
    if (a.merge or a.status) and not a.dry_run and not os.path.exists(os.path.join(out, "run.json")):
        raise SystemExit(f"{out}/run.json missing: nothing was prepared or extracted there")
    os.makedirs(out, exist_ok=True)
    check_anchor()

    if a.dry_run:
        dev = torch.device("cpu")
    elif a.prepare or a.merge or a.status:
        dev = None
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
        ok, msg = C.check_cuda_arch()
        log(msg)
        if not ok:
            log("FATAL: unsupported GPU for this torch build")
            sys.exit(EXIT_GPU)
    else:
        raise SystemExit("CUDA not available (use --dry_run for a CPU test, --prepare / --merge on CPU)")
    dev_kind = "cpu" if a.dry_run else "cuda"
    a.chunk_views = a.chunk_views or (4 if dev_kind == "cpu" else 68)

    items_full = load_items(out, a.items_npz)
    items = select(items_full, a.limit, a.rows)
    plan = shard_plan(items, a.shard_size)
    cfg = make_config(a, items, plan, dev_kind, EXPECTED_FP)
    rj = os.path.join(out, "run.json")
    if a.prepare or a.dry_run or not os.path.exists(rj):
        if not (a.prepare or a.dry_run):
            log("run.json missing: preparing inside this GPU segment (prefer --prepare on CPU first)")
        if os.path.exists(rj):
            old = json.load(open(rj))
            if old["config_key"] != config_key(cfg):
                raise SystemExit(f"{rj} holds a different config; use a fresh --out_dir")
            if old["prepared"].get("verify_seek") and a.verify_seek is None:
                a.verify_seek = 0                           # already verified for this config
        if a.verify_seek is None:
            a.verify_seek = len(items) if a.dry_run else 20
        doc = prepare(a, out, items_full, items, plan, dev_kind)
        if a.prepare:
            return 0
    doc = json.load(open(rj))
    if doc["config_key"] != config_key(cfg):
        diff = {k: (doc["config"].get(k), cfg.get(k)) for k in cfg if doc["config"].get(k) != cfg.get(k)}
        if a.merge and set(diff) == {"script_sha256"}:
            log("note: the script changed since extraction; merging with run.json's config (recorded)")
        else:
            raise SystemExit(f"run.json config differs from this invocation: {json.dumps(diff)[:1500]}")
    if a.status:
        status(out, plan, doc)
        return 0
    s2 = load_stage2()
    if a.merge:
        return merge(a, out, items, plan, doc, s2)
    if os.path.exists(os.path.join(out, "features.npz")):
        log(f"{out}/features.npz exists: nothing to extract")
        return 0
    rc = extract(a, out, items, plan, doc, dev, s2)
    whole = a.shard_stop is None and a.shard_start == 0
    if rc == 0 and a.dry_run and whole:
        return merge(a, out, items, plan, doc, s2)
    if rc == 0 and whole:
        log("all shards of the run are extracted: assemble with --merge (CPU)")
    elif rc == 0:
        log("range done; when every range is done run --merge (CPU)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
