# EEG-Gated Racine Grader — training and scoring code

The code behind the **EEG-Gated Racine Grader (EGRG)** and the Two-Timescale Grader (TT-X3D)
programme that led to it: a video seizure grader (X3D-M with shared 3- and 5-class heads), an
EEG seizure detector (TCN), and a 3-parameter gate in which **EEG and video decide seizure vs
non-seizure together and video alone grades severity**. Full write-up: the *EEG-Gated Racine
Grader* report linked from the top-level README.

This directory holds **code only**. The parent repository `EEG` (`EEG_ROOT`, default
`/work/mech-ai-scratch/alloy/EEG`) holds everything else: the clips (`data/` keys,
`data_full/` videos), the frame and segment caches, the base trainers these scripts import
(`train_pooled.py`, `train_classifier.py`, `train_pooled_eeg.py` and their helpers
`preds_io`, `frame_cache`, ...), the aligned-split checker `check_eeg_align.py`, the stored
reference runs, and every output these scripts write. The code was moved here from
`EEG/ttg/` on 2026-09-24. Runs already queued from `EEG/ttg/` keep using that copy.

## Headline result

Unseen animals: protocol B, subject-disjoint 5-fold out-of-fold (`split_subjects`, seed 49;
the EEG and video folds hold identical animals, checked by `eeg/check_folds.py`). 20 animals,
24,452 clips with both modalities (the 45 video clips without EEG are dropped and counted).
Fixed X3D-M dual-head grader (seed 1, last epoch 12) plus the TCN detector (seed 1, last epoch
30). The gate is fitted leave-one-animal-out. Both networks are read at their pre-registered
last epoch, with no epoch selection.

| task | video alone | EGRG | EGRG − video, bootstrap mean [95% CI] | severe hits |
|---|---|---|---|---|
| 3-class | 0.710 | **0.750** | +0.040 [+0.024, +0.058] | 426 → 424 / 1257 |
| 5-class | 0.514 | **0.557** | +0.042 [+0.029, +0.056] | 419 → 417 / 1257 |

Macro-F1. The difference column is the mean of the paired animal-clustered bootstrap (as in the
stored `results.txt`); the point differences are +0.0398 (3-class) and +0.0433 (5-class).
Per-class recall, hits / clips, video → EGRG:

- 3-class: non-seizure 11775 → 12162 / 12343 · mild (S2–3) 8955 → 9933 / 10852 · severe (S4–5) 426 → 424 / 1257
- 5-class: non-seizure 11787 → 12169 / 12343 · S2 211 → 386 / 1457 · S3 7845 → 8254 / 9395 · S4 302 → 300 / 1063 · S5 47 → 47 / 194

How to read it:

- **The gain is detection.** Detection macro-F1 goes from 0.927 to 0.984. Severity is graded
  by video's within-seizure split by construction, so severe recall does not improve (it loses
  2 clips at both tasks). The within-session severe-vs-mild AUROC stays video's (0.744 at
  3-class).
- **The CIs understate the uncertainty.** They come from an animal-clustered bootstrap
  (2,000 replicates) that holds the gate's leave-one-animal-out outputs fixed, so the
  variability of fitting the gate is not propagated.
- **This is one video seed × one EEG seed.** The seed-spread runs (`eeg/video_subject_s23.tsv`,
  `eeg/eeg_subject_s3.tsv`) were still queued when the code was moved and are not in these
  numbers.

`eeg/joint_gate.py` run from this directory reproduces every number of the stored run
(`$EEG_ROOT/output/ttg_eeg_gate/final/B_x3dfix_eegs1`): EGRG 0.7496809630 / 0.5572929862.
Its `results.json` and `gate.json` differ from the stored ones only in the creation time, the
run time, `argv`, the dhlib path and the descriptive source strings (`ttg/` → `grader/`).

## What each file does

| path | what it does |
|---|---|
| `ttg_common.py` | Shared helpers. Resolves `EEG_ROOT` and puts it on `sys.path`; `enter_eeg_root()` chdirs there. Also: exact frame decoding, snippet geometry, AUROC / within-session AUROC / animal bootstrap, `cls_report`, atomic writes. |
| `dhlib.py` | The decision-headroom loaders, **vendored** from `$EEG_ROOT/output/ttg_ref/decision-headroom/dhlib.py` with the code unchanged. Handles path alignment, double-softmax repair, path-less admission by label sequence, `split_sessions` / `split_subjects`, confusion-matrix metrics and the bootstrap. `DHLIB_DIR` overrides where scripts look for it. |
| `train_grader.py` | Video grader trainer. X3D-M (optionally with the head-softmax bug fixed, `--fix_x3d`) or SlowFast, with dual g3/g5 heads. Resumable mid-epoch on the scavenger partition. Dumps `val_ep{E}.npz` every epoch. |
| `build_f16_cache.py` | Rebuilds `cache_frames/f16s224` from `data_full/`: resumable, never fills a failed clip, bit-verified. |
| `stage1_eval.py` | Stage-1 gates: whether the X3D fix holds, the ranking against SlowFast, dual vs dedicated heads, and the shift from rebuilding the cache. Reads the runs listed in `stage1.tsv`. |
| `stage1_ensemble.py` | Swaps the fixed X3D into the stored top-3 × 5 ensemble, or adds it. Upper bounds, labelled as such. |
| `stage1b_prereg.md`, `stage1b_eval.py` | Stage 1b pre-registration (augmentation, bounded logits, early epoch) and its evaluation, exactly as registered. |
| `probe_extract.py`, `probe_analysis.py` | Stage 2: frozen X3D features (sparse / dense / shuffled) and the pre-registered linear-probe kill test of the dense pathway (outcome: KILL). |
| `sbatch_grader.sh`, `sbatch_cache.sh`, `sbatch_probe.sh`, `submit_ttg.sh` | SLURM drivers (scavenger partition, requeue-safe) and the submission helper. |
| `stage1.tsv`, `stage1b.tsv` | Config tables for `sbatch_grader.sh`, one array task per line. |
| `eeg/train_eeg_det.py` | EGRG EEG detector: `train_pooled_eeg`'s TCN, binary, reported at a fixed last epoch, with per-epoch clip and window dumps. Resumable, bit-identical on CPU. |
| `eeg/joint_gate.py` | Fits and scores EGRG against video alone and the post-hoc gate. Protocol A is the aligned 5,279-clip split; protocol B is the subject-disjoint out-of-fold split. Writes `results.json`, `results.txt` and, for B, `gate.json`. `--regression` re-derives the prototype (46 checks). |
| `eeg/check_folds.py` | Proves the EEG and video subject folds hold the same animals. Imported by `joint_gate.py`. |
| `eeg/verify_eeg_det.py`, `eeg/verify_joint_gate.py` | CPU verification suites for the detector (against the original trainer) and for the gate (synthetic dumps, real tiny runs, frozen gates). |
| `eeg/sbatch_eeg_det.sh`, `eeg/submit_eeg.sh` | SLURM driver and submission helper for the detector. |
| `eeg/*.tsv` | Tables: `eeg_aligned`, `eeg_subject`, `eeg_subject_s3` (detector); `video_subject`, `video_subject_s23` (video folds, driven by `sbatch_grader.sh`). |
| `figures/make_figures.py` | Model diagrams (`fig_video_grader`, `fig_eeg_detector`, `fig_egrg_system`; `.pdf` / `.svg` / `.png`), written next to the script. |

## Running

Environment: the `eeg` conda env (`/work/mech-ai-scratch/alloy/.conda/envs/eeg`: Python 3.11,
torch 2.4.1+cu121, pytorchvideo, OpenCV, scikit-learn 1.5.2). Set
`PYTHONDONTWRITEBYTECODE=1`. Unlike the ensembling scripts in this repo, the grader needs
torch, and scikit-learn fits the gate.

```bash
export EEG_ROOT=/work/mech-ai-scratch/alloy/EEG        # the default; set it if the tree moves
export PYTHONDONTWRITEBYTECODE=1
PY=/work/mech-ai-scratch/alloy/.conda/envs/eeg/bin/python
cd /work/mech-ai-scratch/alloy/video-eeg-ensembling    # any cwd works

# the headline (protocol B, fixed X3D seed 1 x TCN seed 1)
$PY grader/eeg/joint_gate.py --protocol B --video 'output/ttg_vsubj/x3dfix_dual_s1_fold{f}' --video_epoch 12 \
    --eeg 'output/ttg_eeg/subject/tcn_bin_fold{f}_s1' --eeg_epoch 30 --rows headline \
    --out $EEG_ROOT/output/ttg_eeg_gate/<name>
$PY grader/eeg/joint_gate.py --regression --out $EEG_ROOT/output/ttg_eeg_gate/<name>   # must print 46/46

# CPU smoke tests (numbers meaningless)
$PY grader/train_grader.py --arch x3d --fix_x3d --heads dual --seed 1 --dry_run --allow_cpu
$PY grader/eeg/train_eeg_det.py --split subject --fold 0 --seed 1 --epochs 2 --limit 800 --allow_cpu \
    --threads 4 --output $EEG_ROOT/output/ttg_eeg_test/<name>
```

**Path rules.**

- Every script that imports a parent-repo module (the two trainers, the cache and probe
  scripts, `stage1_eval.py`, everything in `eeg/`) puts `EEG_ROOT` on `sys.path` explicitly,
  so the base trainers are imported from the parent repo whatever the cwd.
  `stage1_ensemble.py`, `stage1b_eval.py`, `dhlib.py` and `make_figures.py` import nothing
  from it; `dhlib.py` reads `EEG_ROOT` only to find the stored runs.
- The trainers, the cache builder, the probe scripts and the stage-1 evaluators **chdir to
  `EEG_ROOT`**, because `train_pooled.discover()` and every cache key are relative paths. Any
  relative path you pass them (`--cache_dir`, `--cache`, `--out`) is therefore resolved
  against `EEG_ROOT`, exactly as when they had to be started from there. The one exception is
  `stage1_eval.py --table`, which is resolved against your cwd and then against `grader/`,
  because the tables live here.
- `joint_gate.py` and `check_folds.py` do not chdir. Relative `--video` / `--eeg` run
  directories are resolved against `EEG_ROOT`; `--out` and `--frozen_gate` against the cwd.
- Every output must resolve under `$EEG_ROOT/output/ttg_*` (`ttg_eeg*` for the `eeg/`
  scripts), and the scripts refuse anything else. The two trainers and `probe_extract.py`
  also require it to be absolute. `build_f16_cache.py --out` may also resolve under
  `$EEG_ROOT/cache_frames/`, where the real cache goes. Nothing is ever written into this
  repository, except the figures by `make_figures.py`.
- `stage1_ensemble.py` and `stage1b_eval.py` overwrite their stored result by default. Pass
  `--out` (also under `$EEG_ROOT/output/ttg_*`) to re-derive it somewhere else.

**SLURM.** Run the helpers on a login node, from any cwd:

```bash
DRY_RUN=1 bash grader/submit_ttg.sh stage1 [table]   # preview. submit_ttg.sh SUBMITS unless DRY_RUN=1
bash grader/eeg/submit_eeg.sh aligned|subject|all|video   # dry run by default; DRY_RUN=0 submits
# the seed-spread tables have no helper shortcut:
T=grader/eeg/eeg_subject_s3.tsv; N=$(grep -cvE '^[[:space:]]*(#|$)' $T)
mkdir -p $EEG_ROOT/logs/ttg && sbatch --array=0-$((N-1)) grader/eeg/sbatch_eeg_det.sh $T
# what task i of a table would run, without SLURM:
DRY_RUN=1 SLURM_ARRAY_TASK_ID=0 bash grader/sbatch_grader.sh grader/eeg/video_subject_s23.tsv
```

How the drivers find things:

- **Scripts.** Under `sbatch`, `$0` is a spooled copy, so a driver uses its own directory only
  when that directory is a grader checkout (`train_grader.py`, `eeg/train_eeg_det.py` and
  `dhlib.py` present), i.e. on a direct `bash` call. Otherwise it uses `$GRADER_DIR` (the
  helpers export it), then the checkout `sbatch` was run from (`$SLURM_SUBMIT_DIR/grader`,
  or `$SLURM_SUBMIT_DIR` itself or its parent when that is `grader/` or `grader/eeg/`), then
  the default `/work/mech-ai-scratch/alloy/video-eeg-ensembling/grader`. The old `EEG/ttg/`
  copy has no `dhlib.py`, so it is never picked up by accident.
- **Tables.** A relative table path is resolved against the directory `sbatch` / `bash` was
  run from, then against `grader/` (and `grader/eeg/` for the detector). A table found in none
  of them is refused. It is never looked up under `EEG_ROOT`, so `ttg/stage1.tsv` cannot
  silently read the old copy.
- **Python.** It runs with cwd `$EEG_ROOT` and writes to the absolute output in the table
  (checked after resolving symlinks, as the Python side does).
- **Logs.** They go to `$EEG_ROOT/logs/ttg/`. The `#SBATCH --output` lines hard-code
  `/work/mech-ai-scratch/alloy/EEG/logs/ttg`, so edit them if `EEG_ROOT` moves.

**Cache dry run.** `DRY_RUN=1 bash grader/sbatch_cache.sh` exits 1 with "index.json exists
(complete cache); pass --overwrite to rebuild" once `cache_frames/f16s224` is built, as the
original did. To see the plan, set `TTG_CACHE_OUT` to a test directory under
`output/ttg_tmp/`.

**Known issue (inherited, not from the move).** A reviewer saw one trainer exit with a
segfault (rc 139) during interpreter shutdown, just after it had written `last.pt` and printed
`checkpointed ...; exit 3`. It did not recur in 16 further stops. The checkpoint was good. The
drivers treat 139 as a failure, so after a time-limit (USR1) stop the task would not requeue
itself and would have to be resubmitted (it resumes from `last.pt`). Preempted tasks are
unaffected, because SLURM requeues them itself.

**Figures.** `python3 grader/figures/make_figures.py` (matplotlib only, no `EEG_ROOT`). The
committed figures were made with matplotlib 3.10.7 (the miniconda `python3`), which
reproduces the PNGs byte for byte. The eeg env's matplotlib 3.9.2 gives pixel-identical PNGs
but different PDF/SVG metadata.

## Where outputs go (all under `$EEG_ROOT`)

| path | written by |
|---|---|
| `output/ttg_stage1/`, `output/ttg_stage1b/` | `train_grader.py` via `stage1.tsv` / `stage1b.tsv`; `stage1_eval.py`, `stage1_ensemble.py` (`ens/`), `stage1b_eval.py` |
| `output/ttg_vsubj/` | `train_grader.py` subject folds (`eeg/video_subject*.tsv`) |
| `output/ttg_eeg/aligned/`, `output/ttg_eeg/subject/` | `eeg/train_eeg_det.py` (`eeg/eeg_*.tsv`) |
| `output/ttg_eeg_gate/` | `eeg/joint_gate.py`, `eeg/check_folds.py --json` |
| `output/ttg_probe/`, `output/ttg_tmp/` | `probe_extract.py`, `probe_analysis.py`; smoke tests |
| `cache_frames/f16s224/` | `build_f16_cache.py` |
| `logs/ttg/` | the SLURM drivers |

## Checks run after the move (2026-09-24, CPU)

- `--help` for every script; `bash -n` for every driver.
- DRY_RUN of every line of every table (68 commands). They are identical to the old
  `EEG/ttg` drivers' output except for the script path.
- `joint_gate.py --regression`: 46/46 checks, `results.txt` identical.
- The protocol-B headline above: every number identical to the stored run.
- `stage1b_eval.py`, `stage1_ensemble.py`, `stage1_eval.py`, `check_folds.py` reproduce their
  stored outputs.
- Tiny CPU runs of both trainers through their SLURM drivers are bit-identical (dumps,
  weights, `run_key`) to the old code run with the same arguments.
- `eeg/verify_joint_gate.py all`: 71/71 checks (synthetic dumps, real tiny runs of both
  trainers, frozen gates).
- `eeg/verify_eeg_det.py errors`: 4/4 checks; `firstbatch --setting subject`: 11/11 (the
  port's detector against the parent trainer, bit-identical weights after 3 steps).
- Drivers: a relative table found neither in the cwd nor in `grader/` is refused
  (`ttg/stage1.tsv` from the repo root exits 1). Simulated `sbatch` spooled copies pick
  `$GRADER_DIR`, then the checkout named by `$SLURM_SUBMIT_DIR`, then the default. They
  never pick `EEG/ttg`. With a symlinked `EEG_ROOT`, rows written with the link path are
  accepted, and `..` escapes are refused.
- The probe extraction dry run gives bit-identical features. `probe_analysis.py` runs in
  its quick mode.
- `make_figures.py` reproduces the figures.
