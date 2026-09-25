#!/usr/bin/env python3
"""
Step 0 evaluation (CPU): frozen V-JEPA 2 severity probes on UNSEEN animals, exactly as registered in
grader/step0_vjepa_prereg.md (sections 3 to 7 and 10). No GPU, no torch in the fits.

PURPOSE
  Does a frozen V-JEPA 2 ViT-L representation carry severe-vs-mild information on unseen animals
  (B) beyond frozen Kinetics X3D-M, and (M) as frame ORDER rather than appearance? Every feature
  arm gets the same probe: probe_analysis.standardise + probe_analysis.fit_logreg (class-balanced
  L2 logistic, L-BFGS-B, maxiter 2000), C chosen from the 15-value grid by 4-fold inner CV over
  whole TRAINING animals (probe_analysis.animal_folds), criterion = unweighted mean over the inner
  folds of the within-session AUROC (pooled-AUROC fallback, recorded), ties -> smaller C; cold
  refit; each outer fold of protocol B (split_subjects seed 49, 5 folds x 4 animals) is scored
  once and the out-of-fold (OOF) scores are pooled. Primary metric WS = within-session AUROC of
  the pooled OOF scores (ttg_common.within_auroc); CIs are the animal bootstrap
  (ttg_common.animal_picks seed 11, 2,000 reps; the SAME picks for every arm of a contrast, so
  differences are paired draw by draw). Everything that gates is unrounded float64.

  Decision (section 6, primary contrast severe vs mild, computed mechanically):
      d_M = WS(VJ-dense) - WS(VJ-shuffled)      M = CI_M[0] > 0
      d_B = WS(VJ-dense) - WS(X3D-dense)        B = d_B >= 0.02 AND CI_B[0] > 0
      M and B -> GO | B only -> GO-ENCODER | M only -> WEAK | neither -> KILL

  Arms: VJ-dense / VJ-shuffled / VJ-sparse (V-JEPA 2 dense_mean / shuffled_mean / sparse),
  X3D-dense / X3D-sparse / X3D-shuffled (Stage 2 features), TT-dense / TT-shuffled ([mean, tv]),
  ML-dense / ML-shuffled ([ml blocks 17, 19, 21, mean]; severe vs mild only). 10 jobs for severe
  vs mild and 8 for S4 vs S3 = 18 (contrast, arm) jobs. Reported, never gating: the fine-tuned
  fixed-X3D grader (per seed and the probability-averaged 3-seed ensemble, val_ep12, aligned on
  the clip path) and the bugged X3D as context, VJ-dense vs the grader, LOAO stacks, TT, ML, shuffle
  diagnostics (cosine), drop-one-animal robustness and per-animal WS.

  Clips: the 12,140 seizure clips; a clip enters every arm only if it is finite in every stored
  V-JEPA 2 array, finite in the three X3D arrays and has a val_ep12 posterior from all 3 fixed
  seeds (the common set). Drops are counted by reason; more than 60 drops (0.5%), or any seizure
  row not extracted (pass 1 incomplete), STOPS the analysis with no decision.

STAGES (all deterministic; BLAS is pinned before numpy loads: OMP/OPENBLAS_NUM_THREADS=4 and
        OPENBLAS_CORETYPE=Haswell, so a fit gives the same bits on every node type)
  prepare   items, folds (dhlib and train_pooled routes, asserted equal and equal to the
            registered animals), both feature files (keys, labels, finiteness, the extraction's
            consistency checks: *_tt mean vs *_mean, v from f16 *_tt vs *_tv, perm vs the seed
            rule and its anchor, starts/n_header/single_snippet vs Stage 2), every grader dump
            (path set == split_subjects fold, y5, its own epoch / seed / fold / variant metadata,
            double-softmax), the common set, sha256 of both feature files. On the registered run
            the V-JEPA file must also BE the registered extraction (D 1024, blocks 17/19/21,
            geometry at 256, revision b3c1679b, bf16 on a GPU with precision records), else abort;
            a test run records the differences. Writes <out>/step0_analysis/prepare.{json,npz}.
  jobs      one or more (contrast, arm) jobs. Each outer fold is cached atomically in
            <out>/step0_analysis/jobs/<contrast>__<arm>/fold<k>.npz, so a preempted job resumes.
            A cached fold made from other inputs or other code is refused, never mixed in.
  assemble  every registered job must be complete; bootstraps, differences, the decision, the
            reported-only analyses; writes <out>/step0_results.json and <out>/step0_results.txt.
            The decision is printed with a calibration note (a post-registration null study: the
            CIs are too narrow) and the extraction's precision / identity / code records; the
            Deviations list every implementation choice beyond the pre-registration's letter.
  all       prepare + every job serially + assemble (tests; the real run uses the array driver
            grader/sbatch_step0_analysis.sh).

OUTPUT (the real run; pre-registration section 10)
  $EEG_ROOT/output/ttg_vjepa/step0_results.json, step0_results.txt, and the working files in
  $EEG_ROOT/output/ttg_vjepa/step0_analysis/. Tests go to $EEG_ROOT/output/ttg_tmp/vjepa_<name>/.
  Writing into output/ttg_vjepa/ is refused unless every input and setting is the registered one.

USAGE (the eeg env python; PYTHONDONTWRITEBYTECODE=1; EEG_ROOT env var, default
       /work/mech-ai-scratch/alloy/EEG; relative --vjepa / --x3d / --vsubj are EEG_ROOT's, --out
       must be absolute)
  PY=/work/mech-ai-scratch/alloy/.conda/envs/eeg/bin/python
  $PY grader/step0_vjepa_analysis.py --stage prepare
  $PY grader/step0_vjepa_analysis.py --stage jobs --job_index $SLURM_ARRAY_TASK_ID   # 0..17
  $PY grader/step0_vjepa_analysis.py --stage assemble
  $PY grader/step0_vjepa_analysis.py --list_jobs
  tests (numbers and decision meaningless, labelled so):
  $PY grader/step0_vjepa_synth.py --kind standin --out_dir $EEG_ROOT/output/ttg_tmp/vjepa_standin
  $PY grader/step0_vjepa_analysis.py --vjepa output/ttg_tmp/vjepa_standin/features.npz \
      --out $EEG_ROOT/output/ttg_tmp/vjepa_standin --stage all
  --dry_run: 200 reps, grid {1e-5, 1e-3, 1e-1}, 1,500 training clips per outer fold (seeded).
"""
import os
import sys

# ---- BLAS determinism, fixed BEFORE numpy is imported (pre-registration section 8: fixed
#      OMP_NUM_THREADS=4). OpenBLAS (DYNAMIC_ARCH) picks its kernels by CPU and its summation
#      order by thread count; both are pinned so every node gives bit-identical fits.
BLAS_ENV = {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
            "OPENBLAS_CORETYPE": "Haswell"}
BLAS_ENV_BEFORE = {k: os.environ.get(k) for k in BLAS_ENV}
os.environ.update(BLAS_ENV)
sys.dont_write_bytecode = True

import argparse                                                      # noqa: E402
import hashlib                                                       # noqa: E402
import json                                                          # noqa: E402
import re                                                            # noqa: E402
import subprocess                                                    # noqa: E402
import time                                                          # noqa: E402
import zipfile                                                       # noqa: E402

import numpy as np                                                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ttg_common as C                                               # noqa: E402  (EEG_ROOT on sys.path)
import probe_analysis as PA                                          # noqa: E402  (fit_logreg, standardise, animal_folds, loao)
import dhlib as D                                                    # noqa: E402  (split_subjects, is_double_softmax, unsquash)

# ============================================================================ registered constants

REAL_VJEPA = os.path.join(C.EEG_ROOT, "output", "ttg_vjepa", "features.npz")
REAL_X3D = os.path.join(C.EEG_ROOT, "output", "ttg_probe", "features.npz")
REAL_VSUBJ = os.path.join(C.EEG_ROOT, "output", "ttg_vsubj")
REAL_OUT = os.path.join(C.EEG_ROOT, "output", "ttg_vjepa")
TMP_PREFIX = os.path.join(C.EEG_ROOT, "output", "ttg_tmp", "vjepa_")
PREREG = os.path.join(HERE, "step0_vjepa_prereg.md")
EXTRACT_SCRIPT = os.path.join(HERE, "step0_vjepa_extract.py")
WORK = "step0_analysis"

FINGERPRINT = "75e31ebd:24497"                  # Stage 2's sorted discover() item list
SPLIT_SEED, N_FOLDS = 49, 5
PREREG_FOLDS = {0: ["RN213", "RN223", "RN235", "RN237"], 1: ["RN197", "RN208", "RN215", "RN238"],
                2: ["RN216", "RN229", "RN242", "RN245"], 3: ["RN199", "RN210", "RN219", "RN224"],
                4: ["RN204", "RN222", "RN227", "RN244"]}
# section 3 table (all 12,140 seizure clips): seizure, severe, mild, sessions with both, pairs
PREREG_FOLD_TABLE = {0: (1922, 56, 1866, 27, 371), 1: (2098, 179, 1919, 45, 7227),
                     2: (2813, 470, 2343, 61, 11770), 3: (3741, 370, 3371, 64, 8647),
                     4: (1566, 187, 1379, 41, 2500)}
PREREG_STAGE_COUNTS = {1: 1459, 2: 9419, 3: 1068, 4: 194}
PREREG_INNER_MIN = {"severe_vs_mild": (1746, 3), "S4_vs_S3": (1517, 3)}
GRID = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0]
DRY_GRID = [1e-5, 1e-3, 1e-1]
DRY_LIMIT = 1500
INNER_K = 4
MAXITER = 2000
REPS, PICK_SEED = 2000, 11
DROP_LIMIT = 60                                  # "more than 0.5% of the 12,140 clips (over 60)"
B_MARGIN = 0.02
PAIR_HEAVY = ["RN197", "RN229", "RN242", "RN210", "RN224"]
PERM_ANCHOR = [11, 14, 5, 8, 6, 1, 10, 13, 15, 7, 12, 3, 4, 0, 2, 9]
PERM_ANCHOR_SEED = 3765462539
GRADER_EPOCH = 12
# section 5 bars on all 12,140 seizure clips, severe vs mild (4-decimal regression check)
PREREG_BARS = {"x3dfix": dict(ws=[0.7438, 0.7415, 0.7396, 0.7559], pooled=[0.7846, 0.7940, 0.7643, 0.7966]),
               "x3dbug": dict(ws=[0.7496, 0.7592, 0.7564, 0.7797], pooled=[0.7299, 0.7561, 0.7578, 0.7814])}

VJ_ARRAYS = ["sparse", "dense_mean", "shuffled_mean", "dense_tt", "shuffled_tt", "dense_tv", "shuffled_tv",
             "sparse_ml", "dense_ml", "shuffled_ml"]
# the registered extraction (pre-registration sections 1 and 2); asserted on the registered run only
REG_REV = "b3c1679b7c34d3255ef3547f27c7b226aefab26f"
REG_DIM = 1024
REG_ML_BLOCKS = [17, 19, 21]
REG_GEOMETRY = dict(K=8, T=16, stride=2, size=256, buffer=150, model=f"facebook/vjepa2-vitl-fpc64-256@{REG_REV}",
                    precision="bf16 autocast (cuda)", items="discover", limit=0, rows="")
REG_SHAPES = {k: ([REG_DIM], "float32") for k in ("sparse", "dense_mean", "shuffled_mean", "dense_tv", "shuffled_tv")}
REG_SHAPES.update({k: ([8, 8, REG_DIM], "float16") for k in ("dense_tt", "shuffled_tt")})
REG_SHAPES.update({k: ([3, REG_DIM], "float32") for k in ("sparse_ml", "dense_ml", "shuffled_ml")})
GPU_IDENTITY_TOL_REGISTERED = 1e-5              # section 8 (the extractor aborts on the GPU only above 1e-3)
VJ_META = ["path", "y5", "starts", "n_header", "single_snippet", "failed", "not_extracted", "perm"]
X3D_ARRAYS = ["sparse", "dense_mean", "shuffled_mean"]

CONTRASTS = {
    "severe_vs_mild": dict(pos=[3, 4], neg=[1, 2], label="severe (S4+S5) vs mild (S2+S3)", primary=True),
    "S4_vs_S3": dict(pos=[3], neg=[2], label="S4 vs S3 (S2 and S5 excluded)", primary=False),
}
# arm -> (feature file, [array or array:block], contrasts)
ARMS = {
    "VJ-dense": ("vjepa", ["dense_mean"], None),
    "VJ-shuffled": ("vjepa", ["shuffled_mean"], None),
    "VJ-sparse": ("vjepa", ["sparse"], None),
    "X3D-dense": ("x3d", ["dense_mean"], None),
    "X3D-sparse": ("x3d", ["sparse"], None),
    "X3D-shuffled": ("x3d", ["shuffled_mean"], None),
    "TT-dense": ("vjepa", ["dense_mean", "dense_tv"], None),
    "TT-shuffled": ("vjepa", ["shuffled_mean", "shuffled_tv"], None),
    "ML-dense": ("vjepa", ["dense_ml:0", "dense_ml:1", "dense_ml:2", "dense_mean"], ["severe_vs_mild"]),
    "ML-shuffled": ("vjepa", ["shuffled_ml:0", "shuffled_ml:1", "shuffled_ml:2", "shuffled_mean"],
                    ["severe_vs_mild"]),
}
JOBS = [(c, a) for c in CONTRASTS for a, (_, _, cs) in ARMS.items() if cs is None or c in cs]   # 18
GRADERS = [f"{v}_s{s}" for v in ("x3dfix", "x3dbug") for s in (1, 2, 3)] + ["x3dfix_ens", "x3dbug_ens"]
# section 7.2 / 7.5 / 7.6 differences (a - b), computed in every contrast holding both arms
DIFFS = [("VJ-dense", "VJ-shuffled", "M quantity (motion)"),
         ("VJ-dense", "X3D-dense", "B quantity (better start)"),
         ("VJ-dense", "VJ-sparse", "7.2"),
         ("VJ-sparse", "X3D-sparse", "7.2"),
         ("X3D-dense", "X3D-shuffled", "7.2: Stage 2's kill test re-read on unseen animals"),
         ("TT-dense", "TT-shuffled", "7.5: order information in the token speed"),
         ("TT-dense", "VJ-dense", "7.5: what v adds to the mean"),
         ("ML-dense", "X3D-dense", "7.6: B's quantity under the ML readout"),
         ("ML-dense", "ML-shuffled", "7.6: M's quantity under the ML readout")]
MOTION_CAVEAT = (
    "What M means for V-JEPA 2 (pre-registration section 9): V-JEPA 2 was pretrained to predict latents of "
    "temporally coherent video, so a within-snippet shuffle is out of distribution for it and can disturb its "
    "appearance encoding as well as remove order. VJ-dense - VJ-shuffled > 0 can therefore occur without any "
    "severity-relevant motion (X3D showed no such degradation in Stage 2). Shuffling within a 2.13-s snippet "
    "keeps the set of frames, so M tests ORDER dependence, not motion versus no motion. Read M with the shuffle "
    "diagnostics (section 7.7) printed beside it.")
KILL_CAVEAT = ("A KILL means 'not linearly available in mean-pooled frozen last-layer tokens' (readout asymmetry, "
               "pre-registration section 9); the ML readout measures part of that risk and does not change the "
               "decision.")
CALIBRATION_NOTE = (
    "Calibration (added after the post-registration code review; reported, never gating): the rule is implemented "
    "as registered, but on an artificial exchangeable null (46 replicates through this script's own job and "
    "summary code: VJ-dense, VJ-shuffled and X3D-dense each the real X3D dense_mean PCA-reduced plus independent "
    "noise) M fired in 2/46 runs and B in 5/46 (11%, despite the 0.02 margin); CI_M excluded 0 in 6/46 (13%, "
    "nominal 5%) and CI_B in 10/46 (22%). The animal bootstrap holds the probe fits and the chosen C fixed "
    "(section 9), and C moved between arms by orders of magnitude under the null, so the CIs are too narrow. The "
    "null is artificial, so the rates are indicative; read a GO, GO-ENCODER or WEAK whose CI bound is near 0, or "
    "whose d_B is near 0.02, with this in mind.")
S4S3_GRADER_NOTE = ("grader score here = the g3 severe-vs-mild logit log P2 - log P1 (the pre-registration does not "
                    "specify an S4-vs-S3 grader score; reported only)")
# implementation choices beyond the letter of the pre-registration, listed in every run's Deviations
IMPLEMENTATION_DEVIATIONS = [
    dict(what="analysis: BLAS pinned before numpy loads (OMP/OPENBLAS/MKL threads 4, OPENBLAS_CORETYPE=Haswell) so "
              "every node type gives bit-identical fits; section 8 registers only OMP_NUM_THREADS=4",
         could_change_decision="no (determinism only; no registered choice changes)"),
    dict(what="analysis: for S4 vs S3 the grader bars and the VJ-dense-vs-grader differences use the g3 "
              "severe-vs-mild logit (log P2 - log P1), not a g5 S4-vs-S3 logit; the LOAO stacks, drop-one-animal "
              "robustness and shuffle diagnostics are computed on severe vs mild only",
         could_change_decision="no (reported only)"),
    dict(what="extraction (c): on the GPU the block-23 hook identity and the direct model(x, skip_predictor=True) "
              "identity abort only above 1e-3 relative (the measured values are recorded and summarised under "
              "'Extraction'); the registered 1e-5 applies to the CPU dry run, where both measured exactly 0",
         could_change_decision="no (a wrong block or readout is off by O(1); the GPU values are listed)"),
    dict(what="extraction (d): the section 8 budget rule is extended for parallel GPU jobs: a shard starts only if "
              "total + (live segments + unfinished pass-1 shards, the latter counted for a pass-2 shard) x one "
              "shard's measured duration stays within 8 h; sacct is re-queried at most every 5 min and budget.json "
              "is written every 5 min, at every flush and on SIGTERM",
         could_change_decision="no (it can only stop a shard earlier than the registered rule; a pass-1 stop means "
                               "no decision, as registered)"),
    dict(what="extraction (a): the bf16-vs-fp32 precision check (abort below cosine 0.99) runs on the first batch of "
              "every GPU segment, not only the first segment; it also records the bf16 error of u, *_mean, *_tv and "
              "*_ml and its ratio to the fp32 dense-vs-shuffled gap (bf16_over_gap), never gating",
         could_change_decision="no (stricter than registered)"),
    dict(what="extraction (b): GPU segments read the discover() list cached in items.npz by --prepare "
              "(fingerprint 75e31ebd:24497 re-checked, every label, session and animal re-derived from the path) "
              "instead of re-globbing NFS in every segment",
         could_change_decision="no (the same item list)"),
]
CELLS = {(True, True): ("GO","JEPA domain pretraining on the raw video is funded (temporal prediction is justified)."),
         (False, True): ("GO-ENCODER", "fine-tune V-JEPA 2 as the video grader directly; do NOT claim temporal "
                                       "prediction helps; domain pretraining optional."),
         (True, False): ("WEAK", "motion present but no better start; report, do not fund pretraining yet."),
         (False, False): ("KILL", "KILL the JEPA-for-severity plan (the EEG-gated grader stays).")}


# ============================================================================ small helpers

def die(msg):
    raise SystemExit(f"step0_vjepa_analysis: {msg}")


def resolve(p):
    return os.path.realpath(p if os.path.isabs(p) else os.path.join(C.EEG_ROOT, p))


def sha256_file(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def identity(path):
    st = os.stat(path)
    return dict(path=path, size=int(st.st_size), mtime_ns=int(st.st_mtime_ns))


def sig_of(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def code_hashes():
    return {os.path.basename(f): sha256_file(f) for f in
            (os.path.abspath(__file__), os.path.join(HERE, "probe_analysis.py"),
             os.path.join(HERE, "ttg_common.py"), os.path.join(HERE, "dhlib.py"))}


def git_info(path):
    """Last commit touching `path` in its repository, and whether it has uncommitted changes."""
    out = dict(file=path, exists=os.path.exists(path))
    if not out["exists"]:
        return out
    out["sha256"] = sha256_file(path)
    d = os.path.dirname(path)
    try:
        c = subprocess.run(["git", "-C", d, "log", "-1", "--format=%H", "--", path], capture_output=True,
                           text=True, timeout=60).stdout.strip()
        s = subprocess.run(["git", "-C", d, "status", "--porcelain", "--", path], capture_output=True,
                           text=True, timeout=60).stdout.strip()
        out.update(last_commit=c or None, uncommitted_changes=bool(s), git_status=s or "clean")
    except Exception as e:                                         # noqa: BLE001
        out.update(last_commit=None, git_error=repr(e))
    return out


def check_out_dir(out):
    if not os.path.isabs(out):
        die(f"--out must be an absolute path, got {out!r}")
    rp = os.path.realpath(out)
    ok_real = rp == REAL_OUT
    ok_tmp = (os.path.dirname(rp) == os.path.dirname(TMP_PREFIX)
              and os.path.basename(rp).startswith(os.path.basename(TMP_PREFIX))
              and len(os.path.basename(rp)) > len(os.path.basename(TMP_PREFIX)))
    if not (ok_real or ok_tmp):
        die(f"--out must be {REAL_OUT} (the real run) or {TMP_PREFIX}<name> (tests), got {rp}")
    return rp


def atomic_text(path, text):
    C.atomic_write_bytes(path, text.encode())


def pair_stats(pos, sess):
    """(sessions holding both classes, within-session pairs)."""
    if len(sess) == 0:
        return 0, 0
    us, inv = np.unique(sess, return_inverse=True)
    npos = np.bincount(inv, weights=pos.astype(float), minlength=len(us))
    nneg = np.bincount(inv, weights=(~pos).astype(float), minlength=len(us))
    return int(((npos > 0) & (nneg > 0)).sum()), int((npos * nneg).sum())


# ---- npz streaming (the *_tt arrays are GBs; never load them whole)

def _npz_open(zf, key):
    fh = zf.open(key + ".npy")
    ver = np.lib.format.read_magic(fh)
    if ver == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(fh)
    else:
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(fh)
    if dtype.hasobject:
        die(f"{key}: object arrays are refused")
    if fortran and len(shape) > 1:
        die(f"{key}: Fortran-ordered array")
    return fh, tuple(shape), dtype


def npz_shape(path, key):
    with zipfile.ZipFile(path) as zf:
        fh, shape, dtype = _npz_open(zf, key)
        fh.close()
    return shape, dtype


def npz_row_chunks(path, key, rows=256):
    with zipfile.ZipFile(path) as zf:
        fh, shape, dtype = _npz_open(zf, key)
        per = int(np.prod(shape[1:], dtype=np.int64)) * dtype.itemsize
        with fh:
            for s in range(0, shape[0], rows):
                n = min(rows, shape[0] - s)
                want, parts = n * per, []
                while want:
                    b = fh.read(want)
                    if not b:
                        die(f"{path}:{key} ended early")
                    parts.append(b)
                    want -= len(b)
                yield s, np.frombuffer(b"".join(parts), dtype=dtype).reshape((n,) + shape[1:])


def row_finite(arr):
    return np.isfinite(arr).reshape(len(arr), -1).all(1)


# ============================================================================ prepare

def build_items(route):
    if route == "discover":
        C.enter_eeg_root()
        import train_pooled as tp
        items = tp.discover()
    else:
        items = C.items_from_f32_index()
    items = sorted(items, key=lambda it: it[0])
    fp = C.fingerprint([i[0] for i in items])
    if fp != FINGERPRINT:
        die(f"item fingerprint {fp} != registered {FINGERPRINT}")
    return items, fp


def build_folds(items):
    """Protocol-B folds by the joint_gate / check_folds routes: dhlib.split_subjects on the
    cache-index items and train_pooled.split_subjects on the sorted items (the train_grader route).
    Both must give the same clips and the registered animals."""
    import train_pooled as tp
    items_ci = D.items_from_cache_index()
    folds, rows = {}, []
    for k in range(N_FOLDS):
        _, va_d, val_d, _ = D.split_subjects(items_ci, SPLIT_SEED, fold=k, n_folds=N_FOLDS)
        _, va_t, val_t, _ = tp.split_subjects(items, SPLIT_SEED, k, N_FOLDS)
        _, va_s, val_s, _ = D.split_subjects(items, SPLIT_SEED, fold=k, n_folds=N_FOLDS)
        if not (set(val_d) == set(val_t) == set(val_s) == set(PREREG_FOLDS[k])):
            die(f"fold {k}: animals dhlib {sorted(val_d)} / train_pooled {sorted(val_t)} / sorted "
                f"{sorted(val_s)} / registered {PREREG_FOLDS[k]} differ")
        pd_, pt_ = {i[0] for i in va_d}, {i[0] for i in va_t}
        if pd_ != pt_ or pd_ != {i[0] for i in va_s}:
            die(f"fold {k}: the dhlib and train_pooled routes hold different clips")
        folds[k] = dict(animals=sorted(val_d), paths=pd_)
        rows.append(dict(fold=k, animals=sorted(val_d), n_clips=len(pd_), routes_agree=True))
    return folds, rows


def tt_check_chunk(u16, mean, tv):
    """The merge's per-clip checks on one chunk of f16 temporal-token means (section 2):
    |mean_{j,t} tt - *_mean| and |v(f16 tt) - *_tv| must be <= 1e-3 * (1 + max_{j,t} |tt|) per element."""
    u = u16.astype(np.float64)                                          # (n, 8, 8, D)
    tol = 1e-3 * (1.0 + np.abs(u).max(axis=(1, 2)))
    em = np.abs(u.mean(axis=(1, 2)) - mean.astype(np.float64)) / tol
    ev = np.abs(np.abs(np.diff(u, axis=2)).mean(axis=(1, 2)) - tv.astype(np.float64)) / tol
    return int((em > 1).sum()), float(em.max()), int((ev > 1).sum()), float(ev.max())


def load_feature_file(path, arrays, what, items_paths, meta_keys, tt_check=False):
    """Keys, labels and row finiteness of one feature file. The *_tt arrays are streamed once; with
    tt_check the merge's *_tt / *_mean / *_tv consistency checks run in the same pass, on every row
    whose three arrays are finite (a violation is a bug and aborts)."""
    with zipfile.ZipFile(path) as zf:
        have = {n[:-4] for n in zf.namelist() if n.endswith(".npy")}
    missing = [k for k in list(arrays) + list(meta_keys) if k not in have]
    if missing:
        die(f"{what} features {path} lacks keys {missing} (has {sorted(have)})")
    Z = np.load(path, allow_pickle=False)
    paths = Z["path"].astype(str)
    if not np.array_equal(paths, items_paths):
        die(f"{what}: `path` is not the sorted fingerprinted item list")
    y5 = Z["y5"].astype(int)
    lab = np.array([C.y5_of(p) for p in paths])
    if not np.array_equal(y5, lab):
        die(f"{what}: y5 disagrees with the path labels at {int((y5 != lab).sum())} rows")
    fin, dims, shapes, ttrep = {}, {}, {}, {}
    for k in sorted(arrays, key=lambda k: k.endswith("_tt")):          # *_tt last: needs *_mean, *_tv
        shp, dt = npz_shape(path, k)
        shapes[k] = [list(shp), str(dt)]
        dims[k] = int(shp[-1])
        if shp[0] != len(paths):
            die(f"{what}:{k} has {shp[0]} rows, expected {len(paths)}")
        if not k.endswith("_tt"):
            fin[k] = row_finite(Z[k])
            continue
        view = k[:-3]
        mean, tv = (Z[f"{view}_mean"], Z[f"{view}_tv"]) if tt_check else (None, None)
        f = np.zeros(len(paths), bool)
        acc = [0, 0.0, 0, 0.0, 0]
        for s, ch in npz_row_chunks(path, k, rows=128):
            r = np.arange(s, s + len(ch))
            f[r] = row_finite(ch)
            if tt_check:
                m = f[r] & fin[f"{view}_mean"][r] & fin[f"{view}_tv"][r]
                if m.any():
                    bm, wm, bv, wv = tt_check_chunk(ch[m], mean[r[m]], tv[r[m]])
                    acc = [acc[0] + bm, max(acc[1], wm), acc[2] + bv, max(acc[3], wv), acc[4] + int(m.sum())]
        fin[k] = f
        if tt_check:
            ttrep[view] = dict(rows_checked=acc[4], tt_mean_violations=acc[0], tt_mean_worst_ratio_to_tol=acc[1],
                               tv_violations=acc[2], tv_worst_ratio_to_tol=acc[3])
            if acc[0] or acc[2]:
                die(f"{view}: *_tt / *_mean / *_tv consistency fails ({ttrep[view]}); not a checked merge")
    prov = {}
    for k in sorted(have - set(arrays) - {"path", "y5", "perm", "starts", "n_header", "single_snippet",
                                           "failed", "not_extracted", "dense_snip", "sparse_src", "err"}):
        v = Z[k]
        if v.ndim == 0 or v.size <= 64:
            prov[k] = v.tolist()
    return Z, paths, y5, fin, dims, shapes, prov, ttrep


def check_vjepa_consistency(Z, ok_rows, x3d, ttrep):
    """The extraction merge's remaining checks (pre-registration section 2) on the rows the analysis
    uses: perm against the seed rule and its anchor, geometry against Stage 2 (the *_tt checks ran
    while streaming, in load_feature_file). Any violation is a bug and aborts."""
    rep = dict(tt=ttrep)
    # shuffle permutations: the seed rule on the FULL item string, and the registered anchor
    perm = Z["perm"]
    if tuple(perm.shape[1:]) != (8, 16):
        die(f"perm has shape {perm.shape}, expected [N,8,16]")
    paths = Z["path"].astype(str)
    if C.clip_seed(paths[0]) != PERM_ANCHOR_SEED:
        die("anchor clip_seed mismatch")
    a0 = np.random.default_rng(C.clip_seed(paths[0])).permutation(16).tolist()
    if a0 != PERM_ANCHOR:
        die(f"anchor permutation {a0} != registered {PERM_ANCHOR}")
    bad = 0
    for i in np.flatnonzero(ok_rows):
        rng = np.random.default_rng(C.clip_seed(paths[i]))
        if not np.array_equal(perm[i], np.stack([rng.permutation(16) for _ in range(8)])):
            bad += 1
    if bad:
        die(f"perm differs from the seed rule at {bad} rows")
    rep["perm_rows_checked"] = int(ok_rows.sum())
    rep["perm_anchor_ok"] = True
    rep["file_perm0_is_anchor"] = bool(ok_rows[0] and perm[0, 0].tolist() == PERM_ANCHOR)
    # geometry against Stage 2
    geo = {}
    for k in ("starts", "n_header", "single_snippet"):
        a, b = Z[k][ok_rows], x3d[k][ok_rows]
        n = int((a.reshape(len(a), -1) != b.reshape(len(b), -1)).any(1).sum())
        geo[k] = n
    if any(geo.values()):
        die(f"starts / n_header / single_snippet differ from Stage 2's features.npz: {geo}")
    rep["geometry_mismatches_vs_stage2"] = geo
    return rep


def extraction_summary(prov_json):
    """The parts of the extractor's `provenance` (a JSON string) the results report: model, precision,
    script sha256, and per segment the first-batch identity values, the precision check and the code
    it ran. None if the file carries no provenance (stand-in and synthetic files)."""
    if not isinstance(prov_json, str):
        return None
    try:
        p = json.loads(prov_json)
    except ValueError as e:
        return dict(error=f"provenance is not JSON: {e}")
    cfg = p.get("config") or {}
    segs = []
    for g in p.get("segments") or []:
        fb = g.get("first_batch_checks") or {}
        pc = g.get("precision_check") or {}
        segs.append(dict(seg_id=g.get("seg_id"), job_id=g.get("job_id"), gpu=g.get("gpu"), state=g.get("state"),
                         clips=g.get("clips"), block23_rel_err=fb.get("block23_hook_vs_last_hidden_rel_err"),
                         direct_call_rel_err=fb.get("direct_model_call_rel_err"),
                         identity_tolerance=fb.get("identity_tolerance"), precision_min_cos=pc.get("min_cos"),
                         precision_bf16_over_gap=pc.get("bf16_over_gap"), precision_check=pc or None,
                         code=g.get("code")))
    return dict(config_key=p.get("config_key"), model=cfg.get("model"), precision=cfg.get("precision"),
                script_sha256=cfg.get("script_sha256"), items=cfg.get("items"),
                prepared_code=((p.get("prepared") or {}).get("code") or {}).get("sha256"),
                merge_code=(p.get("merge_code") or {}).get("sha256"), counts=p.get("counts"),
                budget=p.get("budget"), segments=segs, segments_with_rows=p.get("segments_with_rows"))


def vjepa_identity(prov, dims, shapes):
    """The V-JEPA file's geometry and extraction provenance, parsed, and every way it differs from the
    registered extraction (sections 1 and 2). The problems abort the registered run (stage_prepare);
    a test run only records them."""
    problems = []
    geo = None
    if isinstance(prov.get("geometry"), str):
        try:
            geo = json.loads(prov["geometry"])
        except ValueError:
            problems.append("geometry is not JSON")
    if geo is None:
        problems.append("no geometry record")
    else:
        for k, v in REG_GEOMETRY.items():
            if geo.get(k) != v:
                problems.append(f"geometry {k} = {geo.get(k)!r}, registered {v!r}")
    for k, v in (("feature_dim", REG_DIM), ("ml_blocks", REG_ML_BLOCKS), ("fingerprint", FINGERPRINT)):
        if prov.get(k) != v:
            problems.append(f"{k} = {prov.get(k)!r}, registered {v!r}")
    for k, (shp, dt) in REG_SHAPES.items():
        got = shapes.get(k)
        if got is None or list(got[0][1:]) != shp or got[1] != dt:
            problems.append(f"{k} has shape/dtype {got}, registered [N]+{shp} {dt}")
    ex = extraction_summary(prov.get("provenance"))
    if ex is None or "error" in ex:
        problems.append("no readable extraction provenance" + (f" ({ex['error']})" if ex else ""))
    else:
        m = ex.get("model") or {}
        if m.get("revision") != REG_REV:
            problems.append(f"model revision {m.get('revision')!r}, registered {REG_REV}")
        if ex.get("precision") != REG_GEOMETRY["precision"]:
            problems.append(f"precision {ex.get('precision')!r}, registered {REG_GEOMETRY['precision']!r}")
        rows_segs = set(ex.get("segments_with_rows") or [])
        wrote = [g for g in ex["segments"] if g.get("seg_id") in rows_segs]
        if not wrote or any(g.get("gpu") in (None, "cpu") for g in wrote):
            problems.append("a segment that wrote rows did not run on a GPU (or no segment record)")
        if not any(g.get("precision_min_cos") is not None for g in wrote):
            problems.append("no segment that wrote rows carries a bf16-vs-fp32 precision record")
    return dict(geometry=geo, dims=dims, problems=problems, extraction=ex)


def grader_scores(P):
    P = P.astype(np.float64)
    return np.log(np.clip(P[:, 2], 1e-300, None)) - np.log(np.clip(P[:, 1], 1e-300, None))


def load_dumps(vsubj, items, folds, row_of):
    """val_ep12 dumps of both variants, aligned on the clip path. The path set of every dump must
    equal its split_subjects fold, and its y5 the item labels (else abort)."""
    lab = {i[0]: i[1] for i in items}
    N = len(items)
    probs, log = {}, dict(loaded=[], missing=[], double_softmax_repaired=[], checked_fold_sets=0,
                          checked_metadata=0)
    for v in ("x3dfix", "x3dbug"):
        for s in (1, 2, 3):
            P = np.full((N, 3), np.nan)
            for k in range(N_FOLDS):
                d = os.path.join(vsubj, f"{v}_dual_s{s}_fold{k}")
                fn = os.path.join(d, f"val_ep{GRADER_EPOCH:02d}.npz")
                if not os.path.exists(fn):
                    log["missing"].append(os.path.relpath(fn, C.EEG_ROOT))
                    continue
                z = np.load(fn, allow_pickle=False)
                if "path" not in z.files:
                    die(f"{fn}: no path (cannot align)")
                pth = z["path"].astype(str)
                if set(pth.tolist()) != folds[k]["paths"] or len(pth) != len(folds[k]["paths"]):
                    die(f"{fn}: the dump's path set is not split_subjects fold {k}")
                if not np.array_equal(z["y5"].astype(int), np.array([lab[p] for p in pth])):
                    die(f"{fn}: y5 disagrees with the item labels")
                # the dump's own metadata must match its directory name (variant, seed, fold) and the
                # registered epoch; the directory name alone is never trusted
                want = dict(epoch=GRADER_EPOCH, seed=s, fold=k, split="subject", arch="x3d", heads="dual",
                            fix_x3d=(v == "x3dfix"), head_softmax_bug=(v == "x3dbug"))
                miss = [m for m in want if m not in z.files]
                if miss:
                    die(f"{fn}: metadata {miss} missing (cannot confirm variant / seed / fold / epoch)")
                got = {m: z[m].item() for m in want}
                bad = {m: (got[m], w) for m, w in want.items() if got[m] != w}
                if bad:
                    die(f"{fn}: metadata (got, expected) disagrees with the registered run: {bad}")
                log["checked_metadata"] += 1
                p = z["probs_g3"].astype(np.float64)
                if D.is_double_softmax(p):
                    p = D.unsquash(p)
                    log["double_softmax_repaired"].append(os.path.relpath(fn, C.EEG_ROOT))
                P[[row_of[x] for x in pth]] = p
                log["loaded"].append(os.path.relpath(fn, C.EEG_ROOT))
                log["checked_fold_sets"] += 1
            probs[f"{v}_s{s}"] = P
    scores = {n: grader_scores(P) for n, P in probs.items()}
    for v in ("x3dfix", "x3dbug"):
        Pm = [probs[f"{v}_s{s}"] for s in (1, 2, 3)]
        scores[f"{v}_ens"] = grader_scores(np.mean(Pm, 0))       # probability-averaged (joint_gate.ensemble)
    return scores, log


def stage_prepare(a, out, work):
    t0 = time.time()
    items, fp = build_items(a.items)
    ipaths = np.array([i[0] for i in items])
    row_of = {p: r for r, p in enumerate(ipaths)}
    folds, fold_rows = build_folds(items)
    print(f"[prepare] {len(items)} items, fingerprint {fp}; folds hold the registered animals (3 routes agree)",
          flush=True)

    X3, _, x3y5, x3fin, x3dims, x3shapes, x3prov, _ = load_feature_file(a.x3d, X3D_ARRAYS, "x3d", ipaths,
                                                                      ["path", "y5", "starts", "n_header",
                                                                       "single_snippet"])
    VJ, _, vjy5, vjfin, vjdims, vjshapes, vjprov, ttrep = load_feature_file(a.vjepa, VJ_ARRAYS, "vjepa", ipaths,
                                                                             VJ_META, tt_check=True)
    print(f"[prepare] feature files read ({time.time() - t0:.0f}s); V-JEPA dims {vjdims}", flush=True)
    dmean = vjdims["dense_mean"]
    for k in VJ_ARRAYS:
        if vjdims[k] != dmean:
            die(f"vjepa:{k} has dim {vjdims[k]}, dense_mean has {dmean}")
    ml_shape = vjshapes["dense_ml"][0]
    if len(ml_shape) != 3 or ml_shape[1] != 3:
        die(f"dense_ml has shape {ml_shape}, expected [N,3,D]")
    ident = vjepa_identity(vjprov, vjdims, vjshapes)
    if a.registered and ident["problems"]:
        die("the V-JEPA file is not the registered extraction: " + "; ".join(ident["problems"]))
    print(f"[prepare] V-JEPA identity vs the registered extraction: "
          f"{'OK' if not ident['problems'] else str(len(ident['problems'])) + ' differences (test run: recorded)'}",
          flush=True)

    y5 = np.array([i[1] for i in items])
    an = np.array([i[3] for i in items])
    se = np.array([C.session_of(p) for p in ipaths])
    if not (np.array_equal(y5, vjy5) and np.array_equal(y5, x3y5)):
        die("labels disagree between the items and a feature file")
    fold = np.full(len(items), -1)
    for k, f in folds.items():
        fold[np.isin(an, f["animals"])] = k
    if (fold < 0).any():
        die("an animal is in no fold")

    vj_failed = set(VJ["failed"].astype(str).tolist())
    vj_notext = set(VJ["not_extracted"].astype(str).tolist())
    if not (vj_failed | vj_notext) <= set(ipaths.tolist()):
        die("failed / not_extracted list paths that are not items")
    vj_ok = np.all([vjfin[k] for k in VJ_ARRAYS], axis=0)
    x3_ok = np.all([x3fin[k] for k in X3D_ARRAYS], axis=0)
    print(f"[prepare] V-JEPA finite rows {int(vj_ok.sum())}/{len(vj_ok)}; X3D finite rows {int(x3_ok.sum())}",
          flush=True)
    consistency = check_vjepa_consistency(VJ, vj_ok, X3, ttrep)
    print(f"[prepare] V-JEPA consistency checks pass ({time.time() - t0:.0f}s)", flush=True)

    scores, dump_log = load_dumps(a.vsubj, items, folds, row_of)

    sz = y5 >= 1
    in_failed = np.array([p in vj_failed for p in ipaths])
    in_notext = np.array([p in vj_notext for p in ipaths])
    fix_ok = np.all([np.isfinite(scores[f"x3dfix_s{s}"]) for s in (1, 2, 3)], axis=0)
    reasons = {
        "vjepa_failed (listed in failed)": sz & in_failed,
        "vjepa_not_extracted (listed in not_extracted)": sz & in_notext,
        "vjepa_nonfinite_not_listed": sz & ~vj_ok & ~in_failed & ~in_notext,
        "x3d_nonfinite": sz & ~x3_ok,
        "grader_posterior_missing (a fixed seed has no val_ep12 row)": sz & ~fix_ok,
    }
    per_array_nonfinite = {k: int((sz & ~vjfin[k]).sum()) for k in VJ_ARRAYS}
    per_array_nonfinite.update({f"x3d:{k}": int((sz & ~x3fin[k]).sum()) for k in X3D_ARRAYS})
    # a row listed in failed / not_extracted is dropped even if it were finite, so n_dropped and
    # by_reason always agree (the extractor never writes such a row: its merge refuses one)
    common = sz & vj_ok & x3_ok & fix_ok & ~in_failed & ~in_notext
    n_drop = int(sz.sum() - common.sum())
    bug_ok = np.all([np.isfinite(scores[f"x3dbug_s{s}"]) for s in (1, 2, 3)], axis=0)
    status, stop_reason = "OK", None
    if (sz & in_notext).any():
        status, stop_reason = "STOPPED", (f"{int((sz & in_notext).sum())} seizure rows were not extracted "
                                          "(pass 1 incomplete): no decision is made on a partial set (section 8)")
    elif n_drop > DROP_LIMIT:
        status, stop_reason = "STOPPED", (f"{n_drop} of {int(sz.sum())} seizure clips dropped (> {DROP_LIMIT}, "
                                          "0.5%): stop and report (section 3)")
    drops = dict(n_seizure=int(sz.sum()), n_common=int(common.sum()), n_dropped=n_drop,
                 by_reason={k: int(v.sum()) for k, v in reasons.items()},
                 per_array_nonfinite_seizure_rows=per_array_nonfinite,
                 dropped_paths=ipaths[sz & ~common].tolist(),
                 failed_listed_total=len(vj_failed), not_extracted_listed_total=len(vj_notext),
                 not_extracted_nonseizure=int((~sz & in_notext).sum()),
                 bugged_grader_covers_common=bool(bug_ok[common].all()),
                 x3d_failed_listed=int(len(X3["failed"])) if "failed" in X3.files else None)

    if a.skip_sha or a.dry_run:
        shas = dict(vjepa="skipped (dry run / --skip_sha)", x3d="skipped (dry run / --skip_sha)")
    else:
        shas = dict(vjepa=sha256_file(a.vjepa), x3d=sha256_file(a.x3d))
    m = sz
    C.atomic_npz(os.path.join(work, "prepare.npz"),
                 row=np.flatnonzero(m), path=ipaths[m], y5=y5[m], animal=an[m], session=se[m], fold=fold[m],
                 common=common[m], single_snippet=VJ["single_snippet"][m].astype(bool),
                 **{f"g_{n}": scores[n][m] for n in GRADERS})
    prep = dict(created=time.strftime("%F %T"), argv=sys.argv, status=status, stop_reason=stop_reason,
                items=dict(route=a.items, n=len(items), fingerprint=fp), folds=fold_rows,
                vjepa=dict(identity(a.vjepa), sha256=shas["vjepa"], dims=vjdims, shapes=vjshapes, provenance=vjprov,
                           identity_check=ident),
                x3d=dict(identity(a.x3d), sha256=shas["x3d"], dims=x3dims, shapes=x3shapes, provenance=x3prov),
                vsubj=a.vsubj, consistency=consistency, drops=drops, dumps=dump_log,
                code=code_hashes(), seconds=round(time.time() - t0, 1))
    prep["signature"] = sig_of(dict(vjepa=identity(a.vjepa), x3d=identity(a.x3d), vsubj=a.vsubj,
                                    common=C.fingerprint(ipaths[common].tolist()), code=prep["code"]))
    C.atomic_json(os.path.join(work, "prepare.json"), prep)
    print(f"[prepare] {int(sz.sum())} seizure clips, common set {int(common.sum())}, dropped {n_drop} "
          f"{drops['by_reason']}; status {status} ({time.time() - t0:.0f}s)", flush=True)
    if status != "OK":
        print(f"[prepare] STOPPED: {stop_reason}", flush=True)
    return prep


def load_prepare(work, a):
    pj = os.path.join(work, "prepare.json")
    if not os.path.exists(pj):
        die(f"{pj} missing: run --stage prepare first")
    prep = json.load(open(pj))
    for what, p in (("vjepa", a.vjepa), ("x3d", a.x3d)):
        now = identity(p)
        was = {k: prep[what][k] for k in ("path", "size", "mtime_ns")}
        if now != was:
            die(f"{what} file changed since prepare ({was} -> {now}); re-run --stage prepare")
    if prep["code"] != code_hashes():
        die("the analysis code changed since prepare; re-run --stage prepare (and every job) into a fresh --out")
    Pz = dict(np.load(os.path.join(work, "prepare.npz"), allow_pickle=False))
    return prep, Pz


# ============================================================================ jobs

def arm_matrix(a, arm, rows):
    src, parts, _ = ARMS[arm]
    Z = np.load(a.vjepa if src == "vjepa" else a.x3d, allow_pickle=False)
    cols = []
    for p in parts:
        k, _, blk = p.partition(":")
        v = Z[k][rows]
        cols.append(v[:, int(blk)] if blk else v)
    X = np.concatenate([c.reshape(len(rows), -1) for c in cols], 1).astype(np.float64)
    if not np.isfinite(X).all():
        die(f"{arm}: non-finite features inside the common set")
    return X


def job_settings(a):
    return dict(grid=DRY_GRID if a.dry_run else GRID, maxiter=MAXITER, inner_k=INNER_K,
                limit=(a.limit or DRY_LIMIT) if a.dry_run else a.limit, dry_run=a.dry_run)


def run_job(a, work, prep, Pz, cname, arm):
    st = job_settings(a)
    grid = st["grid"]
    c = CONTRASTS[cname]
    m = Pz["common"] & np.isin(Pz["y5"], c["pos"] + c["neg"])
    idx = np.flatnonzero(m)
    y = np.isin(Pz["y5"][idx], c["pos"]).astype(float)
    an, se, fo, paths = Pz["animal"][idx], Pz["session"][idx], Pz["fold"][idx], Pz["path"][idx]
    jd = os.path.join(work, "jobs", f"{cname}__{arm}")
    os.makedirs(jd, exist_ok=True)
    sig = sig_of(dict(prepare=prep["signature"], contrast=cname, arm=arm, settings=st))
    todo = []
    for k in range(N_FOLDS):
        fn = os.path.join(jd, f"fold{k}.npz")
        if os.path.exists(fn):
            if str(np.load(fn)["sig"]) != sig:
                die(f"{fn} was computed from other inputs, settings or code; remove {jd} (or use a fresh --out)")
            continue
        todo.append(k)
    if not todo:
        print(f"[job {cname}:{arm}] all folds cached", flush=True)
        return
    t0 = time.time()
    X = arm_matrix(a, arm, Pz["row"][idx])
    print(f"[job {cname}:{arm}] X {X.shape} ({time.time() - t0:.0f}s); folds to run {todo}", flush=True)
    for k in todo:
        t1 = time.time()
        tr, te = np.flatnonzero(fo != k), np.flatnonzero(fo == k)
        if st["limit"] and len(tr) > st["limit"]:
            tr = np.sort(np.random.default_rng(1000 + k).choice(tr, st["limit"], replace=False))
        Xtr, ytr, atr, str_ = X[tr], y[tr], an[tr], se[tr]
        inner = PA.animal_folds(atr, k=INNER_K)
        crit_parts = np.full((INNER_K, len(grid)), np.nan)
        inner_rec, fits = [], []
        for f in range(INNER_K):
            itr, ite = inner != f, inner == f
            pos_te = ytr[ite] > 0
            n_s, n_p = pair_stats(pos_te, str_[ite])
            rec = dict(inner_fold=f, animals=sorted(set(atr[ite].tolist())), n_clips=int(ite.sum()),
                       n_pos=int(pos_te.sum()), n_neg=int((~pos_te).sum()), sessions_with_both=n_s,
                       within_pairs=n_p)
            if pos_te.all() or (~pos_te).all() or ytr[itr].min() == ytr[itr].max():
                rec["mode"] = "skipped (a class is absent)"
                inner_rec.append(rec)
                continue
            rec["mode"] = "within_session" if n_p > 0 else "pooled_fallback"
            A, B = PA.standardise(Xtr[itr], Xtr[ite])
            w = None
            for j, Cr in enumerate(grid):                       # increasing C, warm-started
                w, info = PA.fit_logreg(A, ytr[itr], Cr, w0=w, maxiter=MAXITER, return_info=True)
                s = B @ w[:-1] + w[-1]
                crit_parts[f, j] = (C.within_auroc(s, pos_te, str_[ite], atr[ite]) if n_p > 0
                                    else C.auroc(s, pos_te))
                fits.append([k, f, Cr, info["nit"], float(info["success"]), info["grad_inf"]])
            inner_rec.append(rec)
        used = ~np.isnan(crit_parts).all(1)
        crit = crit_parts[used].mean(0) if used.any() else np.full(len(grid), np.nan)
        key = [(round(float(v), 6) if np.isfinite(v) else -np.inf, -Cr) for v, Cr in zip(crit, grid)]
        jbest = max(range(len(grid)), key=lambda j: key[j])
        Cb = grid[jbest]
        A, B = PA.standardise(Xtr, X[te])
        w, info = PA.fit_logreg(A, ytr, Cb, w0=None, maxiter=MAXITER, return_info=True)   # cold refit
        fits.append([k, -1, Cb, info["nit"], float(info["success"]), info["grad_inf"]])
        score = B @ w[:-1] + w[-1]
        C.atomic_npz(os.path.join(jd, f"fold{k}.npz"), sig=np.array(sig), path=paths[te], score=score,
                     C_best=np.float64(Cb), crit=crit, crit_parts=crit_parts, grid=np.array(grid),
                     fits=np.array(fits, dtype=np.float64), inner=np.array(json.dumps(inner_rec)),
                     n_train=np.int64(len(tr)), n_test=np.int64(len(te)), seconds=np.float64(time.time() - t1),
                     blas=np.array(json.dumps(BLAS_ENV)))
        print(f"[job {cname}:{arm}] fold {k}: C={Cb:g}{' (grid boundary)' if jbest in (0, len(grid) - 1) else ''} "
              f"crit {crit[jbest]:.4f}; train {len(tr)} test {len(te)}; {len(fits)} fits "
              f"({time.time() - t1:.0f}s)", flush=True)


# ============================================================================ assemble

def summarise(score, pos, sess, an, ua, picks, fold, pooled_boot=True):
    b = C.boot_auc(score, pos, sess, an, ua, picks, pooled=pooled_boot)
    r = dict(ws=C.within_auroc(score, pos, sess, an), ws_ci=C.ci95(b[:, 1]),
             ws_nonfinite_draws=int((~np.isfinite(b[:, 1])).sum()),
             pooled=C.auroc(score, pos), pooled_ci=C.ci95(b[:, 0]) if pooled_boot else None,
             pooled_nonfinite_draws=int((~np.isfinite(b[:, 0])).sum()) if pooled_boot else None)
    if fold is not None:
        r["per_fold"] = {}
        for k in range(N_FOLDS):
            m = fold == k
            r["per_fold"][str(k)] = dict(ws=C.within_auroc(score[m], pos[m], sess[m], an[m]),
                                         pooled=C.auroc(score[m], pos[m]))
    return r, b


def diff(name_a, name_b, S, Bt, note=""):
    ra, rb, ba, bb = S[name_a], S[name_b], Bt[name_a], Bt[name_b]
    Dw, Dp = ba[:, 1] - bb[:, 1], ba[:, 0] - bb[:, 0]
    return dict(a=name_a, b=name_b, note=note, ws_a=ra["ws"], ws_b=rb["ws"], d_ws=ra["ws"] - rb["ws"],
                ci_ws=C.ci95(Dw), nonfinite_ws_draws=int((~np.isfinite(Dw)).sum()),
                d_pooled=ra["pooled"] - rb["pooled"], ci_pooled=C.ci95(Dp),
                nonfinite_pooled_draws=int((~np.isfinite(Dp)).sum()))


def load_job(work, prep, a, cname, arm, paths_expected):
    st = job_settings(a)
    sig = sig_of(dict(prepare=prep["signature"], contrast=cname, arm=arm, settings=st))
    jd = os.path.join(work, "jobs", f"{cname}__{arm}")
    folds = []
    for k in range(N_FOLDS):
        fn = os.path.join(jd, f"fold{k}.npz")
        if not os.path.exists(fn):
            return None
        z = np.load(fn, allow_pickle=False)
        if str(z["sig"]) != sig:
            die(f"{fn}: signature mismatch (other inputs, settings or code)")
        folds.append(z)
    pos_of = {p: i for i, p in enumerate(paths_expected)}
    score = np.full(len(paths_expected), np.nan)
    seen = np.zeros(len(paths_expected), int)
    Cs, bnd, fits, inner, crit, secs = [], [], [], [], [], 0.0
    for k, z in enumerate(folds):
        ix = np.array([pos_of[p] for p in z["path"].astype(str)])
        score[ix] = z["score"]
        seen[ix] += 1
        g = z["grid"].tolist()
        Cb = float(z["C_best"])
        Cs.append(Cb)
        bnd.append(Cb in (g[0], g[-1]))
        fits.append(z["fits"])
        inner.append(json.loads(str(z["inner"])))
        crit.append(dict(zip([f"{c:g}" for c in g], z["crit"].tolist())))
        secs += float(z["seconds"])
    if not (seen == 1).all():
        die(f"{cname}:{arm}: {int((seen != 1).sum())} clips not scored exactly once")
    F = np.concatenate(fits)
    conv = dict(n_fits=int(len(F)), n_reached_maxiter=int((F[:, 3] >= MAXITER).sum()),
                n_success_false=int((F[:, 4] == 0).sum()), max_grad_inf=float(F[:, 5].max()))
    modes = [r["mode"] for fl in inner for r in fl]
    return score, dict(C_per_fold=Cs, boundary_hits=int(sum(bnd)), boundary_folds=[k for k, b in enumerate(bnd) if b],
                       crit_per_fold=crit, inner_cv=inner,
                       inner_modes={mm: modes.count(mm) for mm in sorted(set(modes))},
                       convergence=conv, fits_columns=["outer_fold", "inner_fold(-1=refit)", "C", "nit", "success",
                                                       "grad_inf"],
                       fits=F.tolist(), cpu_seconds=round(secs, 1))


def counts_block(y5, pos, an, se, fo):
    n_s, n_p = pair_stats(pos, se)
    by_a = {}
    for q in np.unique(an):
        m = an == q
        by_a[q] = pair_stats(pos[m], se[m])[1]
    tot = sum(by_a.values())
    per_fold = []
    for k in range(N_FOLDS):
        m = fo == k
        s_k, p_k = pair_stats(pos[m], se[m])
        per_fold.append(dict(fold=k, animals=sorted(set(an[m].tolist())), n=int(m.sum()),
                             n_pos=int(pos[m].sum()), n_neg=int((~pos[m]).sum()),
                             stages={f"S{v + 1}": int((y5[m] == v).sum()) for v in (1, 2, 3, 4)},
                             sessions_with_both=s_k, within_pairs=p_k))
    return dict(n=int(len(pos)), n_pos=int(pos.sum()), n_neg=int((~pos).sum()),
                stages={f"S{v + 1}": int((y5 == v).sum()) for v in (1, 2, 3, 4)},
                animals=int(len(np.unique(an))), sessions=int(len(np.unique(se))),
                sessions_with_both=n_s, within_pairs=n_p,
                animals_with_pairs=int(sum(v > 0 for v in by_a.values())),
                pair_share_by_animal={q: v / tot for q, v in sorted(by_a.items(), key=lambda t: -t[1]) if v > 0}
                if tot else {},
                per_fold=per_fold)


def inner_pair_minima(pos, an, se, fo, limit_note):
    out = []
    for k in range(N_FOLDS):
        tr = fo != k
        inner = PA.animal_folds(an[tr], k=INNER_K)
        pr = [pair_stats(pos[tr][inner == f], se[tr][inner == f])[1] for f in range(INNER_K)]
        out.append(dict(outer_fold=k, inner_pairs=pr))
    mn = min((min(r["inner_pairs"]), r["outer_fold"]) for r in out)
    return dict(per_outer_fold=out, min_pairs=mn[0], min_in_outer_fold=mn[1], note=limit_note)


def cosine_rows(Xa, Xb):
    num = (Xa * Xb).sum(1)
    den = np.linalg.norm(Xa, axis=1) * np.linalg.norm(Xb, axis=1)
    return num / den


def q3(v):
    return dict(median=float(np.median(v)), q25=float(np.percentile(v, 25)), q75=float(np.percentile(v, 75)),
                n=int(len(v)))


def budget_report(vjepa_path):
    p = os.path.join(os.path.dirname(vjepa_path), "budget.json")
    if not os.path.exists(p):
        return dict(status=f"no budget.json next to {vjepa_path}")
    try:
        b = json.load(open(p))
    except Exception as e:                                         # noqa: BLE001
        return dict(status=f"budget.json unreadable: {e!r}")
    ids = set()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if "job" in k.lower() and "id" in k.lower():
                    for x in (v if isinstance(v, list) else [v]):
                        if isinstance(x, (str, int)) and str(x).strip():
                            ids.add(str(x).split("_")[0].split(".")[0])
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(b)
    # jobs the driver recorded before Python started (budget/job_<id>_r<n>.txt), e.g. killed at start-up
    bdir = os.path.join(os.path.dirname(vjepa_path), "budget")
    from_files = set()
    if os.path.isdir(bdir):
        for f in os.listdir(bdir):
            mm = re.match(r"job_(\d+)_r\d+\.txt$", f)
            if mm:
                from_files.add(mm.group(1))
    ids |= from_files
    rep = dict(budget_json=b, job_ids=sorted(ids), job_ids_from_driver_files=sorted(from_files))
    if ids:
        try:
            r = subprocess.run(["sacct", "-n", "-P", "-X", "-D", "-j", ",".join(sorted(ids)),
                                "--format=JobIDRaw,ElapsedRaw,AllocTRES,State"], capture_output=True,
                               text=True, timeout=120)
            tot, rows = 0.0, []
            for line in r.stdout.strip().splitlines():
                jid, el, tres, state = (line.split("|") + ["", "", "", ""])[:4]
                ng, assumed = 1, True                     # unlisted: 1 GPU, conservative, as the extractor
                for t in tres.split(","):
                    if t.startswith("gres/gpu="):
                        ng, assumed = int(t.split("=")[1]), False
                rows.append(dict(job=jid, elapsed_s=int(el or 0), gpus=ng, gpus_assumed=assumed, state=state))
                tot += int(el or 0) * ng / 3600.0
            rep.update(sacct_rows=rows, sacct_gpu_hours=tot, sacct_rc=r.returncode,
                       over_cap=bool(tot > 8.0))
        except Exception as e:                                     # noqa: BLE001
            rep["sacct_error"] = repr(e)
    return rep


def stage_assemble(a, out, work, test_reasons):
    t0 = time.time()
    prep, Pz = load_prepare(work, a)
    reps = a.reps
    res = dict(created=time.strftime("%F %T"), argv=sys.argv, status=prep["status"],
               prereg="grader/step0_vjepa_prereg.md", test_run=bool(test_reasons), test_reasons=test_reasons,
               blas=dict(set=BLAS_ENV, before=BLAS_ENV_BEFORE), reps=reps, pick_seed=PICK_SEED,
               settings=job_settings(a), inputs=dict(vjepa=prep["vjepa"], x3d=prep["x3d"], vsubj=prep["vsubj"]),
               items=prep["items"], folds=prep["folds"], drops=prep["drops"], dumps=prep["dumps"],
               consistency=prep["consistency"])
    deviations = []
    res["provenance"] = dict(prereg=git_info(PREREG), analysis_script=git_info(os.path.abspath(__file__)),
                             probe_analysis=git_info(os.path.join(HERE, "probe_analysis.py")),
                             ttg_common=git_info(os.path.join(HERE, "ttg_common.py")),
                             dhlib=git_info(os.path.join(HERE, "dhlib.py")),
                             extract_script=git_info(a.extract_script),
                             features_sha256=dict(vjepa=prep["vjepa"]["sha256"], x3d=prep["x3d"]["sha256"]),
                             vjepa_file_provenance=prep["vjepa"]["provenance"])
    for k, g in res["provenance"].items():
        if isinstance(g, dict) and g.get("exists") and (g.get("uncommitted_changes") or not g.get("last_commit")):
            deviations.append(dict(what=f"{k} ({os.path.relpath(g['file'], os.path.dirname(HERE))}) is uncommitted "
                                        f"or has uncommitted changes at run time",
                                   could_change_decision="unknown: the frozen code is not what ran"
                                   if k != "prereg" else "no (the pre-registration text only)"))
    res["budget"] = budget_report(a.vjepa)
    if res["budget"].get("over_cap"):
        deviations.append(dict(what=f"GPU budget overage: sacct {res['budget']['sacct_gpu_hours']:.2f} h > 8 h",
                               could_change_decision="no (the features are complete; section 8 overage rule)"))
    if prep["dumps"]["missing"]:
        deviations.append(dict(what=f"{len(prep['dumps']['missing'])} grader dumps missing: "
                                    f"{prep['dumps']['missing'][:6]}", could_change_decision="no (reported only)"))
    rep = prep["dumps"]["double_softmax_repaired"]
    if rep:
        deviations.append(dict(what=f"{len(rep)} grader dumps were double-softmaxed and repaired with dhlib.unsquash on "
                                    f"float32 posteriors, which loses near-zero probabilities; bars from them are not "
                                    f"trustworthy: {rep[:6]}", could_change_decision="no (the grader is not a gate)"))
    ident = prep["vjepa"].get("identity_check") or {}
    res["extraction"] = ident.get("extraction")
    res["vjepa_identity_problems"] = ident.get("problems")
    ex = ident.get("extraction") or {}
    if ex and "error" not in ex:
        vals = [g[k] for g in ex["segments"] if g.get("gpu") not in (None, "cpu")
                for k in ("block23_rel_err", "direct_call_rel_err") if g.get(k) is not None]
        if vals and max(vals) > GPU_IDENTITY_TOL_REGISTERED:
            deviations.append(dict(what=f"GPU identity checks (block-23 hook, direct model call) measured up to "
                                        f"{max(vals):.3g} relative, above the registered 1e-5 (see extraction (c))",
                                   could_change_decision="no (a wrong block or readout is off by O(1))"))
        cur = res["provenance"]["extract_script"].get("sha256")
        if ex.get("script_sha256") and cur and ex["script_sha256"] != cur:
            deviations.append(dict(what=f"the extraction script changed after the features were extracted (sha256 "
                                        f"{ex['script_sha256'][:16]} at extraction, {cur[:16]} now)",
                                   could_change_decision="unknown: compare the two versions"))
    deviations += [dict(x) for x in IMPLEMENTATION_DEVIATIONS]
    if prep["status"] != "OK":
        res.update(decision=None, stop_reason=prep["stop_reason"], deviations=deviations)
        C.atomic_json(os.path.join(out, "step0_results.json"), res)
        atomic_text(os.path.join(out, "step0_results.txt"), render(res))
        print(f"STOPPED: {prep['stop_reason']}")
        return res

    y5, an, se, fo, cm = Pz["y5"], Pz["animal"], Pz["session"], Pz["fold"], Pz["common"]
    # ---- section 3 counts, on all seizure clips and on the common set
    sz_all = counts_block(y5, y5 >= 3, an, se, fo)
    reg = {}
    for r in sz_all["per_fold"]:
        got = (r["n"], r["n_pos"], r["n_neg"], r["sessions_with_both"], r["within_pairs"])
        reg[str(r["fold"])] = dict(got=got, registered=PREREG_FOLD_TABLE[r["fold"]],
                                   ok=got == PREREG_FOLD_TABLE[r["fold"]])
    stage_ok = {f"S{v + 1}": int((y5 == v).sum()) == n for v, n in PREREG_STAGE_COUNTS.items()}
    res["counts"] = dict(all_seizure=dict(n=int(len(y5)), sessions=int(len(np.unique(se))),
                                          animals=int(len(np.unique(an))),
                                          stages={f"S{v + 1}": int((y5 == v).sum()) for v in (1, 2, 3, 4)},
                                          severe_vs_mild=sz_all),
                         registered_table_check=dict(per_fold=reg, stages=stage_ok,
                                                     all_ok=all(v["ok"] for v in reg.values()) and all(stage_ok.values())),
                         common=dict(n=int(cm.sum()), sessions=int(len(np.unique(se[cm]))),
                                     animals=int(len(np.unique(an[cm]))),
                                     stages={f"S{v + 1}": int((y5[cm] == v).sum()) for v in (1, 2, 3, 4)}))
    if not res["counts"]["registered_table_check"]["all_ok"]:
        deviations.append(dict(what="the section 3 counts on all seizure clips do not reproduce the registered table",
                               could_change_decision="yes if the clip set differs; see counts.registered_table_check"))

    arms_run = a.arms or list(ARMS)
    res["contrasts"] = {}
    boots = {}
    for cname, c in CONTRASTS.items():
        m = cm & np.isin(y5, c["pos"] + c["neg"])
        idx = np.flatnonzero(m)
        pos = np.isin(y5[idx], c["pos"])
        can, cse, cfo, cpaths = an[idx], se[idx], fo[idx], Pz["path"][idx]
        ua, picks = C.animal_picks(can, reps, seed=PICK_SEED)
        blk = dict(label=c["label"], counts=counts_block(y5[idx], pos, can, cse, cfo),
                   grader_score_note=None if c["primary"] else S4S3_GRADER_NOTE,
                   inner_fold_pairs=inner_pair_minima(pos, can, cse, cfo, "full training sets (as registered)"),
                   single_snippet_clips=int(Pz["single_snippet"][idx].sum()),
                   bootstrap=dict(animals=ua.tolist(), reps=reps, seed=PICK_SEED), arms={}, graders={},
                   differences=[], missing_jobs=[])
        mn = PREREG_INNER_MIN[cname]
        blk["inner_fold_pairs"]["registered_min"] = dict(pairs=mn[0], outer_fold=mn[1])
        S, Bt, OOF = {}, {}, {}
        for arm, (_, _, cs) in ARMS.items():
            if cs is not None and cname not in cs:
                continue
            if arm not in arms_run:
                blk["missing_jobs"].append(arm)
                continue
            got = load_job(work, prep, a, cname, arm, cpaths)
            if got is None:
                if not a.allow_partial:
                    die(f"job {cname}:{arm} is incomplete; run it (or pass --allow_partial for a test)")
                blk["missing_jobs"].append(arm)
                continue
            score, info = got
            S[arm], Bt[arm] = summarise(score, pos, cse, can, ua, picks, cfo)
            S[arm].update(info)
            S[arm]["dim"] = int(arm_dim(prep, arm))
            OOF[arm] = score
        for g in GRADERS:
            sc = Pz[f"g_{g}"][idx]
            if not np.isfinite(sc).all():
                blk["graders"][g] = dict(status=f"unavailable on {int((~np.isfinite(sc)).sum())} clips")
                continue
            S[f"grader:{g}"], Bt[f"grader:{g}"] = summarise(sc, pos, cse, can, ua, picks, cfo)
            blk["graders"][g] = S[f"grader:{g}"]
            OOF[f"grader:{g}"] = sc
        for arm in list(S):
            if not arm.startswith("grader:"):
                blk["arms"][arm] = S[arm]
        for da, db, note in DIFFS:
            if da in S and db in S:
                blk["differences"].append(diff(da, db, S, Bt, note))
        if "VJ-dense" in S:
            for g in GRADERS:
                if f"grader:{g}" in S:
                    blk["differences"].append(diff("VJ-dense", f"grader:{g}", S, Bt,
                                                   "7.3 VJ-dense vs the grader" + (" (context: bugged X3D)"
                                                                                  if "bug" in g else "")))
        # M and B quantities (the decision on the primary contrast; reported for S4 vs S3)
        if all(k in S for k in ("VJ-dense", "VJ-shuffled", "X3D-dense")):
            blk["M_B"] = mb_quantities(S, Bt)
        else:
            blk["M_B"] = None
        res["contrasts"][cname] = blk
        boots[cname] = dict(S=S, Bt=Bt, OOF=OOF, pos=pos, an=can, se=cse, fo=cfo, idx=idx, ua=ua, picks=picks)
        print(f"[assemble] {cname}: {len(S)} scores summarised ({time.time() - t0:.0f}s)", flush=True)

    # ---- section 6 decision (primary)
    P = res["contrasts"]["severe_vs_mild"]
    if P["M_B"] is None:
        res["decision"] = None
        deviations.append(dict(what="the decision arms (VJ-dense, VJ-shuffled, X3D-dense) are not all available",
                               could_change_decision="yes: no decision"))
    else:
        mb = P["M_B"]
        cell, meaning = CELLS[(mb["M"], mb["B"])]
        res["decision"] = dict(cell=cell, meaning=meaning, **mb,
                               rule="M = CI_M[0] > 0; B = d_B >= 0.02 AND CI_B[0] > 0 (exact float64, no tolerance)",
                               valid=not test_reasons,
                               label=("MEANINGLESS: test run (" + "; ".join(test_reasons) + ")") if test_reasons
                               else "registered run",
                               motion_caveat=MOTION_CAVEAT if mb["M"] else None,
                               kill_caveat=KILL_CAVEAT if cell == "KILL" else None,
                               calibration_note=CALIBRATION_NOTE)
    # ---- 7.7 shuffle diagnostics (primary common set)
    bp = boots["severe_vs_mild"]
    res["shuffle_diagnostics"] = shuffle_diagnostics(a, Pz, bp)
    # ---- 7.4 LOAO stacks (primary)
    res["loao_stacks"] = loao_stacks(bp)
    # ---- 7.8 robustness (primary)
    res["robustness"] = robustness(bp)
    # ---- grader bars: all seizure clips vs the common set + the registered-value check
    res["grader_bars"] = grader_bars(Pz, bp)
    if not res["grader_bars"]["registered_check"]["all_ok"]:
        deviations.append(dict(what="the grader bars on all seizure clips do not reproduce the registered section 5 "
                                    "values at 4 decimals", could_change_decision="no (the grader is not a gate)"))
    res["deviations"] = deviations
    res["runtime_s"] = round(time.time() - t0, 1)
    C.atomic_json(os.path.join(out, "step0_results.json"), res)
    atomic_text(os.path.join(out, "step0_results.txt"), render(res))
    print(render_decision(res), flush=True)
    print(f"wrote {out}/step0_results.json and step0_results.txt ({time.time() - t0:.0f}s)", flush=True)
    return res


def arm_dim(prep, arm):
    src, parts, _ = ARMS[arm]
    dims = prep["vjepa" if src == "vjepa" else "x3d"]["dims"]
    return sum(dims[p.partition(":")[0]] for p in parts)


def mb_quantities(S, Bt):
    dm = diff("VJ-dense", "VJ-shuffled", S, Bt)
    db = diff("VJ-dense", "X3D-dense", S, Bt)
    M = bool(dm["ci_ws"][0] > 0)
    B = bool(db["d_ws"] >= B_MARGIN and db["ci_ws"][0] > 0)
    return dict(d_M=dm["d_ws"], CI_M=dm["ci_ws"], d_B=db["d_ws"], CI_B=db["ci_ws"], M=M, B=B,
                WS_VJ_dense=S["VJ-dense"]["ws"], WS_VJ_shuffled=S["VJ-shuffled"]["ws"],
                WS_X3D_dense=S["X3D-dense"]["ws"],
                nonfinite_draws=dict(VJ_dense=S["VJ-dense"]["ws_nonfinite_draws"],
                                     VJ_shuffled=S["VJ-shuffled"]["ws_nonfinite_draws"],
                                     X3D_dense=S["X3D-dense"]["ws_nonfinite_draws"],
                                     d_M=dm["nonfinite_ws_draws"], d_B=db["nonfinite_ws_draws"]))


def shuffle_diagnostics(a, Pz, bp):
    rows = Pz["row"][bp["idx"]]
    pos, se, an, ua, picks = bp["pos"], bp["se"], bp["an"], bp["ua"], bp["picks"]
    out = {}
    for name, path in (("V-JEPA 2", a.vjepa), ("X3D", a.x3d)):
        Z = np.load(path, allow_pickle=False)
        cs = cosine_rows(Z["dense_mean"][rows].astype(np.float64), Z["shuffled_mean"][rows].astype(np.float64))
        r, _ = summarise(1.0 - cs, pos, se, an, ua, picks, None)
        out[name] = dict(cos_all=q3(cs), cos_severe=q3(cs[pos]), cos_mild=q3(cs[~pos]),
                         one_minus_cos_as_severity_score=dict(ws=r["ws"], ws_ci=r["ws_ci"], pooled=r["pooled"],
                                                              pooled_ci=r["pooled_ci"]))
    return out


def loao_stacks(bp):
    S, OOF, pos, an, se, ua, picks = bp["S"], bp["OOF"], bp["pos"], bp["an"], bp["se"], bp["ua"], bp["picks"]
    g = "grader:x3dfix_ens"
    if g not in OOF:
        return dict(status="fixed-ensemble grader unavailable")
    y = pos.astype(float)
    base = PA.loao(OOF[g][:, None], y, an)
    rb, bb = summarise(base, pos, se, an, ua, picks, None)
    out = dict(features="probe_analysis.loao (fit_predict: median impute, standardise, class-balanced L2 lam=1), "
                        "leave-one-animal-out over the animals of the severe-vs-mild common set",
               caveat="for a held-out animal, the OOF scores of animals in other folds came from probes trained "
                      "with it (the accepted second-order leak, as for the EGRG gate)",
               grader_alone=dict(ws=rb["ws"], ws_ci=rb["ws_ci"], pooled=rb["pooled"], pooled_ci=rb["pooled_ci"]),
               stacks={})
    for arm in ("VJ-dense", "VJ-shuffled", "X3D-dense"):
        if arm not in OOF:
            continue
        st = PA.loao(np.c_[OOF[g], OOF[arm]], y, an)
        rs, bs = summarise(st, pos, se, an, ua, picks, None)
        Dw, Dp = bs[:, 1] - bb[:, 1], bs[:, 0] - bb[:, 0]
        out["stacks"][arm] = dict(ws=rs["ws"], ws_ci=rs["ws_ci"], pooled=rs["pooled"], pooled_ci=rs["pooled_ci"],
                                  d_ws_vs_grader_alone=rs["ws"] - rb["ws"], ci_d_ws=C.ci95(Dw),
                                  nonfinite_d_ws_draws=int((~np.isfinite(Dw)).sum()),
                                  d_pooled_vs_grader_alone=rs["pooled"] - rb["pooled"], ci_d_pooled=C.ci95(Dp),
                                  role="test" if arm == "VJ-dense" else "control")
    return out


def robustness(bp):
    OOF, pos, an, se = bp["OOF"], bp["pos"], bp["an"], bp["se"]
    need = ("VJ-dense", "VJ-shuffled", "X3D-dense")
    if not all(k in OOF for k in need):
        return dict(status="decision arms unavailable")
    out = dict(drop_one_animal={}, per_animal_ws={})
    for q in PAIR_HEAVY:
        keep = an != q
        if keep.all():
            out["drop_one_animal"][q] = dict(status="animal not in the set")
            continue
        ua, picks = C.animal_picks(an[keep], len(bp["picks"]), seed=PICK_SEED)
        S, Bt = {}, {}
        for k in need:
            S[k], Bt[k] = summarise(OOF[k][keep], pos[keep], se[keep], an[keep], ua, picks, None, pooled_boot=False)
        mb = mb_quantities(S, Bt)
        out["drop_one_animal"][q] = {k: mb[k] for k in ("d_M", "CI_M", "d_B", "CI_B", "M", "B",
                                                        "WS_VJ_dense", "WS_VJ_shuffled", "WS_X3D_dense")}
    ua = np.unique(an)
    parts = {k: C.ws_parts(OOF[k], pos, se, an, ua) for k in need}
    N = parts["VJ-dense"][1]
    per, tally = {}, {}
    for i, q in enumerate(ua):
        if N[i] <= 0:
            continue
        per[q] = dict(pairs=int(N[i]), **{k: float(parts[k][0][i] / N[i]) for k in need})
    for other in ("VJ-shuffled", "X3D-dense"):
        d = np.array([per[q]["VJ-dense"] - per[q][other] for q in per])
        tally[f"VJ-dense vs {other}"] = dict(favour_VJ_dense=int((d > 0).sum()), ties=int((d == 0).sum()),
                                             favour_other=int((d < 0).sum()), animals=int(len(d)))
    out["per_animal_ws"] = dict(animals=per, tally=tally)
    return out


def grader_bars(Pz, bp):
    y5, an, se, fo = Pz["y5"], Pz["animal"], Pz["session"], Pz["fold"]
    pos_all = y5 >= 3
    ua, picks = C.animal_picks(an, len(bp["picks"]), seed=PICK_SEED)
    rows = {}
    check = {}
    for v in ("x3dfix", "x3dbug"):
        vals = dict(ws=[], pooled=[])
        for g in [f"{v}_s1", f"{v}_s2", f"{v}_s3", f"{v}_ens"]:
            sc = Pz[f"g_{g}"]
            if not np.isfinite(sc).all():
                rows[g] = dict(status="unavailable")
                vals["ws"].append(None)
                vals["pooled"].append(None)
                continue
            r_all, _ = summarise(sc, pos_all, se, an, ua, picks, None)
            r_cm = bp["S"].get(f"grader:{g}")
            rows[g] = dict(all_seizure=dict(ws=r_all["ws"], ws_ci=r_all["ws_ci"], pooled=r_all["pooled"],
                                            pooled_ci=r_all["pooled_ci"], n=int(len(sc))),
                           common_set=None if r_cm is None else dict(ws=r_cm["ws"], ws_ci=r_cm["ws_ci"],
                                                                     pooled=r_cm["pooled"],
                                                                     pooled_ci=r_cm["pooled_ci"]),
                           unique_scores=int(len(np.unique(sc))))
            vals["ws"].append(round(r_all["ws"], 4))
            vals["pooled"].append(round(r_all["pooled"], 4))
        check[v] = dict(got=vals, registered=PREREG_BARS[v], ok=vals == PREREG_BARS[v])
    return dict(score="log(clip(P[:,2],1e-300)) - log(clip(P[:,1],1e-300)) of probs_g3, val_ep12; ensemble = "
                      "severity logit of the probability-averaged posterior of seeds 1-3",
                bars=rows, registered_check=dict(per_variant=check, all_ok=all(c["ok"] for c in check.values())))


# ============================================================================ rendering

def f4(x):
    return "   nan" if x is None or not np.isfinite(x) else f"{x:.4f}"


def fci(ci):
    return f"[{f4(ci[0])}, {f4(ci[1])}]" if ci else "[n/a]"


def fd(x):
    return "    nan" if x is None or not np.isfinite(x) else f"{x:+.4f}"


def fdci(ci):
    return f"[{fd(ci[0])}, {fd(ci[1])}]"


def render_decision(res):
    d = res.get("decision")
    if not d:
        return "DECISION: none" + (f" ({res.get('stop_reason')})" if res.get("stop_reason") else "")
    L = [f"DECISION: {d['cell']}  -- {d['meaning']}",
         f"  [{d['label']}]",
         f"  M (motion)       d_M = WS(VJ-dense) - WS(VJ-shuffled) = {f4(d['WS_VJ_dense'])} - {f4(d['WS_VJ_shuffled'])}"
         f" = {fd(d['d_M'])}  CI_M {fdci(d['CI_M'])}  -> M = {d['M']}",
         f"  B (better start) d_B = WS(VJ-dense) - WS(X3D-dense)   = {f4(d['WS_VJ_dense'])} - {f4(d['WS_X3D_dense'])}"
         f" = {fd(d['d_B'])}  CI_B {fdci(d['CI_B'])}  -> B = {d['B']} (needs d_B >= 0.02 and CI_B[0] > 0)",
         f"  non-finite bootstrap draws: {d['nonfinite_draws']}"]
    return "\n".join(L)


def render(res):
    L = ["Step 0: frozen V-JEPA 2 severity probe on unseen animals (grader/step0_vjepa_prereg.md)",
         f"created {res['created']}   status {res['status']}"]
    if res.get("test_run"):
        L.append("*** TEST RUN: numbers and decision are MEANINGLESS: " + "; ".join(res["test_reasons"]) + " ***")
    L.append("")
    L.append(render_decision(res))
    d = res.get("decision")
    if d and d.get("motion_caveat"):
        L += ["", "  " + d["motion_caveat"]]
        sd = res.get("shuffle_diagnostics", {})
        for bk, v in sd.items():
            c = v["cos_all"]
            L.append(f"  shuffle diagnostics {bk}: cos(dense_mean, shuffled_mean) median {c['median']:.4f} "
                     f"IQR [{c['q25']:.4f}, {c['q75']:.4f}]; severe {v['cos_severe']['median']:.4f}, mild "
                     f"{v['cos_mild']['median']:.4f}; WS of 1-cos {f4(v['one_minus_cos_as_severity_score']['ws'])} "
                     f"{fci(v['one_minus_cos_as_severity_score']['ws_ci'])}")
    if d and d.get("kill_caveat"):
        L += ["", "  " + d["kill_caveat"]]
    if d and d.get("calibration_note"):
        L += ["", "  " + d["calibration_note"]]
    if res.get("stop_reason"):
        L += ["", f"STOPPED: {res['stop_reason']}"]
    dr = res["drops"]
    L += ["", "Clips (section 3)",
          f"  seizure clips {dr['n_seizure']}; common set {dr['n_common']}; dropped {dr['n_dropped']} "
          f"(stop above {DROP_LIMIT})",
          f"  dropped by reason (a clip can have several): {dr['by_reason']}",
          f"  non-finite seizure rows per array: {dr['per_array_nonfinite_seizure_rows']}",
          f"  V-JEPA failed listed {dr['failed_listed_total']}, not_extracted listed {dr['not_extracted_listed_total']} "
          f"(non-seizure {dr['not_extracted_nonseizure']}); X3D failed listed {dr['x3d_failed_listed']}; bugged grader "
          f"covers the common set: {dr['bugged_grader_covers_common']}"]
    du = res["dumps"]
    L.append(f"  grader dumps: {len(du['loaded'])} loaded (fold path sets asserted: {du['checked_fold_sets']}), "
             f"missing {len(du['missing'])}, double-softmax repaired {len(du['double_softmax_repaired'])}")
    if "counts" not in res:
        L += _render_tail(res)
        return "\n".join(L) + "\n"
    ca = res["counts"]["all_seizure"]
    L.append(f"  all seizure clips: {ca['n']} clips, {ca['sessions']} sessions, {ca['animals']} animals, "
             f"stages {ca['stages']}; registered-table check {'OK' if res['counts']['registered_table_check']['all_ok'] else 'MISMATCH'}")
    cc = res["counts"]["common"]
    L.append(f"  common set: {cc['n']} clips, {cc['sessions']} sessions, {cc['animals']} animals, stages {cc['stages']}")
    for cname, blk in res["contrasts"].items():
        k = blk["counts"]
        L += ["", f"== {cname}: {blk['label']} ==",
              f"  {k['n_pos']} positive / {k['n_neg']} negative clips (stages {k['stages']}), {k['animals']} animals, "
              f"{k['sessions']} sessions, {k['sessions_with_both']} with both classes, {k['within_pairs']} "
              f"within-session pairs from {k['animals_with_pairs']} animals; single-snippet clips "
              f"{blk['single_snippet_clips']}",
              "  pair share by animal: " + ", ".join(f"{q} {v:.0%}" for q, v in list(k["pair_share_by_animal"].items())[:8]),
              "  fold | held-out animals            |  clips |  pos |   neg | sess both | pairs"]
        for r in k["per_fold"]:
            L.append(f"  {r['fold']:4d} | {' '.join(r['animals']):27s} | {r['n']:6d} | {r['n_pos']:4d} | {r['n_neg']:5d} |"
                     f" {r['sessions_with_both']:9d} | {r['within_pairs']}")
        ip = blk["inner_fold_pairs"]
        L.append(f"  smallest inner fold: {ip['min_pairs']} pairs (outer fold {ip['min_in_outer_fold']}); registered "
                 f"{ip.get('registered_min')}")
        if blk["missing_jobs"]:
            L.append(f"  NOT RUN (test subset): {blk['missing_jobs']}")
        L.append("  arm           dim  |   WS   [95% CI]          | pooled [95% CI]          | C per fold (boundary) | "
                 "fits maxiter/!success | inner modes")
        for arm, r in blk["arms"].items():
            L.append(f"  {arm:12s} {r['dim']:5d} | {f4(r['ws'])} {fci(r['ws_ci'])} | {f4(r['pooled'])} {fci(r['pooled_ci'])} |"
                     f" {','.join(f'{c:g}' for c in r['C_per_fold'])} ({r['boundary_hits']}) | "
                     f"{r['convergence']['n_reached_maxiter']}/{r['convergence']['n_success_false']} of "
                     f"{r['convergence']['n_fits']} | {r['inner_modes']}")
        L.append("  per-fold WS: " + "; ".join(
            f"{arm} " + " ".join(f4(v['ws']) for v in r["per_fold"].values()) for arm, r in blk["arms"].items()))
        if not CONTRASTS[cname]["primary"]:
            L.append(f"  {S4S3_GRADER_NOTE}")
        L.append("  grader (fixed X3D bar; bugged = context)    |   WS   [95% CI]          | pooled [95% CI]")
        for g, r in blk["graders"].items():
            if "ws" in r:
                L.append(f"  {g:43s} | {f4(r['ws'])} {fci(r['ws_ci'])} | {f4(r['pooled'])} {fci(r['pooled_ci'])}")
            else:
                L.append(f"  {g:43s} | {r['status']}")
        L.append("  paired differences (a - b)               |  dWS    [95% CI]            | dpooled [95% CI]")
        for x in blk["differences"]:
            L.append(f"  {x['a'] + ' - ' + x['b']:41s} | {fd(x['d_ws'])} {fdci(x['ci_ws'])} | {fd(x['d_pooled'])} "
                     f"{fdci(x['ci_pooled'])}  {x['note']}")
        if blk.get("M_B") and not CONTRASTS[cname]["primary"]:
            mb = blk["M_B"]
            L.append(f"  M/B quantities here (reported, not gating): d_M {fd(mb['d_M'])} {fdci(mb['CI_M'])} M would be "
                     f"{mb['M']}; d_B {fd(mb['d_B'])} {fdci(mb['CI_B'])} B would be {mb['B']}")
    L += _render_tail(res)
    return "\n".join(L) + "\n"


def _render_extraction(res):
    L = ["", "Extraction (the V-JEPA 2 file's provenance)"]
    ex = res.get("extraction")
    if not ex:
        L.append("  no extraction provenance (stand-in / synthetic file)")
    elif "error" in ex:
        L.append(f"  {ex['error']}")
    else:
        segs = ex.get("segments") or []
        wrote = set(ex.get("segments_with_rows") or [])
        gpu = [g for g in segs if g.get("gpu") not in (None, "cpu")]
        m = ex.get("model") or {}
        ct = ex.get("counts") or {}
        L.append(f"  model {m.get('repo')}@{str(m.get('revision'))[:8]}; precision {ex.get('precision')}; "
                 f"{len(segs)} segments ({len(wrote)} wrote rows, {len(gpu)} on a GPU); failed {ct.get('failed')}, "
                 f"not extracted {ct.get('not_extracted')}, single-snippet {ct.get('single_snippet')} (among "
                 f"extracted rows)")
        mc = [g["precision_min_cos"] for g in segs if g.get("precision_min_cos") is not None]
        og = [g["precision_bf16_over_gap"] for g in segs if g.get("precision_bf16_over_gap")]
        if mc:
            L.append(f"  bf16 vs fp32 (first batch of each GPU segment): min token-mean cos {min(mc):.6f} "
                     f"(1 - cos {1 - min(mc):.3g}; abort below 0.99)"
                     + (f"; bf16 error / fp32 dense-vs-shuffled gap, max over segments: "
                        f"{max(o['mean'] for o in og):.3g} for *_mean (the decision's readout; should be << 1), "
                        f"{max(o['tv'] for o in og):.3g} for *_tv (the non-gating TT readout; 0.13 in a CPU "
                        f"bf16 simulation)" if og else ""))
        else:
            L.append("  no bf16-vs-fp32 precision record (CPU / test extraction)")
        idv = [g[k] for g in gpu for k in ("block23_rel_err", "direct_call_rel_err") if g.get(k) is not None]
        if idv:
            L.append(f"  GPU identity checks (block-23 hook, direct model call): max {max(idv):.3g} relative "
                     f"(registered 1e-5; the GPU aborts above 1e-3)")
        codes = {json.dumps(g.get("code"), sort_keys=True) for g in segs if g.get("seg_id") in wrote}
        L.append(f"  code of the segments that wrote rows: {len(codes)} state(s); equal to --prepare's: "
                 f"{codes == {json.dumps(ex.get('prepared_code'), sort_keys=True)}}")
    if res.get("vjepa_identity_problems"):
        pr = res["vjepa_identity_problems"]
        L.append(f"  differs from the registered extraction in {len(pr)} ways (allowed only in a test run): "
                 + "; ".join(pr[:6]) + (" ..." if len(pr) > 6 else ""))
    return L


def _render_tail(res):
    L = []
    sd = res.get("shuffle_diagnostics")
    if sd:
        L += ["", "Shuffle diagnostics (7.7, severe vs mild common set)"]
        for bk, v in sd.items():
            L.append(f"  {bk}: cos median {v['cos_all']['median']:.4f} IQR [{v['cos_all']['q25']:.4f}, "
                     f"{v['cos_all']['q75']:.4f}]; severe {v['cos_severe']['median']:.4f} "
                     f"[{v['cos_severe']['q25']:.4f}, {v['cos_severe']['q75']:.4f}]; mild {v['cos_mild']['median']:.4f} "
                     f"[{v['cos_mild']['q25']:.4f}, {v['cos_mild']['q75']:.4f}]; 1-cos as severity score WS "
                     f"{f4(v['one_minus_cos_as_severity_score']['ws'])} {fci(v['one_minus_cos_as_severity_score']['ws_ci'])}")
    ls = res.get("loao_stacks")
    if ls and "grader_alone" in ls:
        L += ["", "LOAO stacks (7.4, severe vs mild): stack - grader-alone stack",
              f"  grader alone (fixed ensemble, LOAO): WS {f4(ls['grader_alone']['ws'])} {fci(ls['grader_alone']['ws_ci'])}"]
        for arm, r in ls["stacks"].items():
            L.append(f"  + {arm:12s} ({r['role']:7s}): WS {f4(r['ws'])}; dWS {fd(r['d_ws_vs_grader_alone'])} "
                     f"{fdci(r['ci_d_ws'])}; dpooled {fd(r['d_pooled_vs_grader_alone'])} {fdci(r['ci_d_pooled'])}")
        L.append(f"  caveat: {ls['caveat']}")
    rb = res.get("robustness")
    if rb and "drop_one_animal" in rb:
        L += ["", "Robustness (7.8): drop one pair-heavy animal (OOF not refitted; picks redrawn)"]
        for q, r in rb["drop_one_animal"].items():
            if "d_M" in r:
                L.append(f"  -{q}: d_M {fd(r['d_M'])} {fdci(r['CI_M'])} M={r['M']}; d_B {fd(r['d_B'])} {fdci(r['CI_B'])} "
                         f"B={r['B']}")
        for k, t in rb["per_animal_ws"]["tally"].items():
            L.append(f"  per-animal WS, {k}: favour VJ-dense {t['favour_VJ_dense']}, ties {t['ties']}, favour other "
                     f"{t['favour_other']} (of {t['animals']} pair-holding animals)")
    gb = res.get("grader_bars")
    if gb:
        L += ["", f"Grader bars on all seizure clips vs the common set; registered-value check "
                  f"{'OK' if gb['registered_check']['all_ok'] else 'MISMATCH ' + json.dumps(gb['registered_check']['per_variant'])}"]
        for g, r in gb["bars"].items():
            if "all_seizure" in r:
                cmn = r["common_set"]
                L.append(f"  {g:11s} all {r['all_seizure']['n']}: WS {f4(r['all_seizure']['ws'])} pooled "
                         f"{f4(r['all_seizure']['pooled'])} | common: WS {f4(cmn['ws']) if cmn else 'n/a'} pooled "
                         f"{f4(cmn['pooled']) if cmn else 'n/a'} | unique scores {r['unique_scores']}")
    L += _render_extraction(res)
    bu = res.get("budget", {})
    L += ["", "Budget: " +(f"sacct {bu['sacct_gpu_hours']:.2f} GPU-h over jobs {bu['job_ids']}"
                            if "sacct_gpu_hours" in bu else bu.get("status", json.dumps(bu)[:200]))]
    pv = res.get("provenance", {})
    L += ["", "Provenance (commit of the last change; * = uncommitted changes)"]
    for k in ("prereg", "extract_script", "analysis_script", "probe_analysis", "ttg_common", "dhlib"):
        g = pv.get(k, {})
        if g.get("exists"):
            L.append(f"  {k:16s} {g.get('last_commit') or 'UNCOMMITTED'}{'*' if g.get('uncommitted_changes') else ''}"
                     f"  sha256 {g['sha256'][:16]}")
        else:
            L.append(f"  {k:16s} (not found: {g.get('file')})")
    fs = pv.get("features_sha256", {})
    L.append(f"  features sha256: vjepa {fs.get('vjepa')}  x3d {fs.get('x3d')}")
    L.append(f"  BLAS: {res.get('blas', {}).get('set')}")
    L += ["", "Deviations" + ("" if res.get("deviations") else ": none")]
    for x in res.get("deviations", []):
        L.append(f"  - {x['what']} (could change the decision: {x['could_change_decision']})")
    return L


# ============================================================================ main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--vjepa", default=REAL_VJEPA, help="V-JEPA 2 features.npz (relative: EEG_ROOT's)")
    ap.add_argument("--x3d", default=REAL_X3D, help="Stage 2 X3D features.npz (relative: EEG_ROOT's)")
    ap.add_argument("--vsubj", default=REAL_VSUBJ, help="root of the x3d{fix,bug}_dual_s*_fold* dumps")
    ap.add_argument("--out", default=REAL_OUT, help=f"{REAL_OUT} (real) or {TMP_PREFIX}<name> (tests)")
    ap.add_argument("--stage", choices=["all", "prepare", "jobs", "assemble"], default="all")
    ap.add_argument("--jobs", default="all", help="comma list of contrast:arm, or 'all' (--stage jobs / all)")
    ap.add_argument("--job_index", type=int, default=None, help="run registered job i (0..17; SLURM arrays)")
    ap.add_argument("--list_jobs", action="store_true")
    ap.add_argument("--extract_script", default=EXTRACT_SCRIPT, help="recorded (commit, sha256) in the results")
    ap.add_argument("--items", choices=["f32index", "discover"], default="f32index",
                    help="item source; both are fingerprint-checked against the registered list")
    # test-only switches: any of them marks the run as a test and refuses the real output directory
    ap.add_argument("--dry_run", action="store_true", help="tests: 200 reps, 3 C values, 1,500 training clips/fold")
    ap.add_argument("--limit", type=int, default=0, help="tests: training clips per outer fold (seeded)")
    ap.add_argument("--reps", type=int, default=None, help=f"tests: bootstrap reps (registered {REPS})")
    ap.add_argument("--arms", default=None, help="tests: comma list of arms to run (default all)")
    ap.add_argument("--allow_partial", action="store_true", help="tests: assemble with missing jobs")
    ap.add_argument("--skip_sha", action="store_true", help="tests: do not hash the feature files")
    a = ap.parse_args()

    if a.list_jobs:
        for i, (c, arm) in enumerate(JOBS):
            print(f"{i:2d} {c}:{arm}")
        return 0
    a.vjepa, a.x3d, a.vsubj = resolve(a.vjepa), resolve(a.x3d), resolve(a.vsubj)
    a.extract_script = os.path.realpath(a.extract_script)
    if a.arms:
        a.arms = [s.strip() for s in a.arms.split(",") if s.strip()]
        bad = [s for s in a.arms if s not in ARMS]
        if bad:
            die(f"unknown arms {bad}; known {list(ARMS)}")
    if a.reps is None:
        a.reps = 200 if a.dry_run else REPS
    out = check_out_dir(a.out)
    test_reasons = []
    if a.dry_run:
        test_reasons.append("--dry_run")
    if a.limit:
        test_reasons.append(f"--limit {a.limit}")
    if a.reps != REPS:
        test_reasons.append(f"--reps {a.reps}")
    if a.arms:
        test_reasons.append(f"--arms {a.arms}")
    if a.allow_partial:
        test_reasons.append("--allow_partial")
    if a.skip_sha:
        test_reasons.append("--skip_sha")
    if a.vjepa != REAL_VJEPA:
        test_reasons.append(f"V-JEPA file is not the registered {REAL_VJEPA}")
    if a.x3d != os.path.realpath(REAL_X3D):
        test_reasons.append("X3D file is not Stage 2's")
    if a.vsubj != os.path.realpath(REAL_VSUBJ):
        test_reasons.append("grader dumps are not output/ttg_vsubj")
    if os.path.exists(a.vjepa):
        try:
            Zp = np.load(a.vjepa, allow_pickle=False)
            for k in ("synthetic", "standin", "synthetic_kind"):
                if k in Zp.files and Zp[k].size == 1 and bool(Zp[k].item()):
                    test_reasons.append(f"V-JEPA file is synthetic ({k}={Zp[k].item()})")
                    break
        except Exception:                                          # noqa: BLE001
            pass
    if out == REAL_OUT and test_reasons:
        die("the real output directory only takes the registered run; test switches / inputs: " + "; ".join(test_reasons))
    a.registered = not test_reasons          # prepare then also asserts the V-JEPA file's registered identity
    for what, p in (("--vjepa", a.vjepa), ("--x3d", a.x3d), ("--vsubj", a.vsubj)):
        if not os.path.exists(p):
            die(f"{what} {p} does not exist")
    work = os.path.join(out, WORK)
    if a.stage in ("all", "prepare"):
        os.makedirs(work, exist_ok=True)
    elif not os.path.isdir(work):
        die(f"{work} missing: run --stage prepare first")
    print(f"step0_vjepa_analysis: out {out}; BLAS {BLAS_ENV}" + (f"; TEST: {test_reasons}" if test_reasons else ""),
          flush=True)

    if a.stage in ("all", "prepare"):
        prep = stage_prepare(a, out, work)
        if prep["status"] != "OK":
            stage_assemble(a, out, work, test_reasons)
            return 2
    if a.stage in ("all", "jobs"):
        prep, Pz = load_prepare(work, a)
        if prep["status"] != "OK":
            die(f"prepare STOPPED: {prep['stop_reason']}")
        if a.job_index is not None:
            if not 0 <= a.job_index < len(JOBS):
                die(f"--job_index {a.job_index} outside 0..{len(JOBS) - 1}")
            todo = [JOBS[a.job_index]]
        elif a.jobs == "all":
            todo = JOBS
        else:
            todo = []
            for s in a.jobs.split(","):
                c, _, arm = s.strip().partition(":")
                if (c, arm) not in JOBS:
                    die(f"unknown job {s!r}; see --list_jobs")
                todo.append((c, arm))
        if a.arms:
            todo = [j for j in todo if j[1] in a.arms]
        for c, arm in todo:
            run_job(a, work, prep, Pz, c, arm)
    if a.stage in ("all", "assemble"):
        stage_assemble(a, out, work, test_reasons)
    return 0


if __name__ == "__main__":
    sys.exit(main())
