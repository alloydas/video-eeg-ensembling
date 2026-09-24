# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

Code over a video–EEG rodent seizure project, in two parts:

- **Ensembling and fusion** (`ensembles/`, `fusion/`): reads the stored `val_preds.npz` /
  `val_clip_preds.npz` files and averages them. No training.
- **The EEG-Gated Racine Grader** (`grader/`): training, scoring and SLURM code for a video
  grader, an EEG seizure detector and the gate that joins them. `grader/README.md` has the
  file map, path rules and run commands. It moved here from `EEG/ttg/` on 2026-09-24.

**This repo holds code only, never data.** Clips, caches, checkpoints, stored runs, logs and
every output stay in the parent `EEG` checkout: `EEG_ROOT`, default
`/work/mech-ai-scratch/alloy/EEG`, which is the `EEG-seizure-classification` repo. Grader
scripts write only under `$EEG_ROOT/output/ttg_*`, `cache_frames/` and `logs/ttg/`, and refuse
anything else. The one exception is the figures from `grader/figures/make_figures.py`. Do not
put grader code back into the `EEG` repo.

## The prediction-file contract — get these wrong and the numbers are silently inflated

- **Align members on the stored clip `path`, never on row index.** Nine of forty video runs per
  task store rows in a different order. Row-index averaging mixes probability vectors across
  clips and pushes detection to 0.986, which is wrong.
- **Detect and invert double-softmaxed posteriors** before averaging. Four video runs per task
  store `softmax(softmax(x))`; `is_double_softmax` / `unsquash` handle it.
- **Runs with no `path`** (the HuggingFace TimeSformer and VideoMAE trainers) may only be
  admitted if their raw label sequence equals the reference run's, i.e. they are already in
  reference order. Otherwise drop them and say so.
- **Never assume two runs share a validation set.** Video and EEG here do not: 5,289 vs 5,319
  clips, overlapping by 2,830. Check before comparing or fusing.
- Report `dropped` and `double-softmax repaired` counts in any output. A silent drop is a bug.

## Environment

Every script reads `EEG_ROOT` for the run tree.

- `ensembles/`, `fusion/`: `numpy` + `scipy`; `scikit-learn` only for metrics. The original
  analysis box has no sklearn in its base interpreter, and its shared environment must not be
  installed into. Use a venv with `--system-site-packages`.
- `grader/`: the `eeg` conda env, `/work/mech-ai-scratch/alloy/.conda/envs/eeg/bin/python`
  (torch 2.4.1+cu121, pytorchvideo, OpenCV, scikit-learn 1.5.2, which also fits the gate). Set
  `PYTHONDONTWRITEBYTECODE=1`. The trainers import `train_pooled*.py` from `EEG_ROOT` and chdir
  there, because every cache key is a relative path.
- GPU work goes through SLURM, from a login node:
  - `grader/submit_ttg.sh` **submits unless `DRY_RUN=1`**; `grader/eeg/submit_eeg.sh` only
    dry-runs unless `DRY_RUN=0`.
  - The mech-ai account sits at its 17-GPU cap, so the drivers use the scavenger QoS with
    `--requeue`, and trainers resume mid-epoch from `last.pt`.
  - `rtx_pro_6000` nodes cannot run torch 2.4.1+cu121, and the drivers' `--constraint`
    excludes them.
- Data hazards:
  - `EEG/data/*.mp4` are zero-byte placeholders. Decode frames from `data_full/`, but key the
    clips by their `data/` path.
  - The only frame caches are `cache_frames/f32s224` and the rebuilt `f16s224`.

## Conventions

- Metrics are macro-averaged precision / recall / F1, binary or one-vs-rest-macro AUROC, and
  multiclass MCC — matching the parent project's result tables. Do not silently switch to
  accuracy or weighted averages.
- Report **per-class recall with class counts** beside any macro number. A macro-F1 gain that
  hides a severe-class regression is a regression.
- **Selecting a fusion weight, a threshold or a member subset on the test set is cheating.**
  If you do it as an upper bound, label it as such in the output, as `ens_sel.py` does.
- Inference about the parent cohort is clustered on **animal**, never on clip.

## Established results — do not re-derive, challenge only with evidence

- Video ensembling gains +0.0032 / +0.0107 / +0.0136 macro-F1 at detection / 3-class / 5-class;
  curation is unnecessary, and seed-averaging one backbone captures most of the gain.
- EEG ensembling **loses** 0.0068 / 0.0118 / 0.0199 against its own best single run (TCN at
  every task) and no honest selection rule fixes it; test-set-selected rules only reach parity.
- The cause is pool spread (EEG single-run sd 0.098–0.131 vs video 0.025–0.029), **not** error
  correlation — the irreducible error share is 2.3% EEG vs 2.2% video at 5-class. The mechanism
  behind the failure is *not* established; do not write as though it is.
- Video ensembling erases the parent paper's 0.006 macro-F1 detection margin for EEG, but the
  two are scored on different clip sets. Recomputed on the 2,830 shared clips, EEG leads in
  point estimate, but ensemble against ensemble the gap is unresolved at the animal level:
  +0.0074 [−0.0166, +0.0305]. Details are in the *Two Signals, One Grader* report.
- **EEG-Gated Racine Grader** (`grader/`). Tested on unseen animals: subject-disjoint 5-fold
  out-of-fold, 20 animals, 24,452 clips. Both networks are read at their pre-registered last
  epoch, and the gate is fitted leave-one-animal-out.
  - 3-class 0.710 → **0.750**; 5-class 0.514 → **0.557** (video alone → EGRG).
  - **The gain is detection** (0.927 → 0.984). Severity is video's within-seizure split by
    construction, and severe recall does not move (426 → 424 of 1,257).
  - Across three EEG seeds the gain moves by at most 0.002 (sd 0.001). The **video seed
    spread is not yet measured**, so quote the headline as one video network.
  - On the matched 5,279-clip split, EGRG beats the unfitted post-hoc gate by only
    +0.0012 [−0.0014, +0.0055] at 3-class.
- **Video-only changes did not improve the ensemble.** They are negatives; do not re-propose
  them without new evidence.
  - Fixing X3D's train-mode head softmax lifts X3D alone at 5-class (0.582 → 0.641) but costs
    3-class (0.798 → 0.769) and helps neither ensemble.
  - Augmentation, tanh-bounded logits and early-epoch reporting were pre-registered, and none
    was adopted.
  - The dense-temporal pathway was killed at its frozen-probe gate: frame-shuffled input
    matches it, so its gain is appearance, not motion.

## `fusion/` — read this before touching it

Cross-modal fusion of the video and EEG posteriors. **The results committed in this
directory were measured on the wrong set.** The 3-class re-measurement on the matched split
was done on 2026-09-22 and is in the *Two Signals, One Grader* report (linked from the
README). Its corrections are folded in below.

### The split problem

Both training pipelines shuffle sessions with seed 49, but they shuffle *different lists*:
`train_pooled.discover()` finds 604 sessions with video, the EEG segment cache holds 601
(three RN216 sessions have video but no usable EEG window). A three-element difference
re-permutes everything, so the two 20% prefixes agree on only **56 sessions / 2,830 clips**.
This is documented verbatim in `train_pooled_eeg.py:145-178` in the parent repo.

That 2,830-clip intersection is a bad set for this question: it **drops 3 of 18 animals**
(RN216, RN222, RN229), cuts the inverse-Simpson effective animal count from 10.8 to 6.2,
holds only **44% of the severe clips**, is **6x enriched in Stage 2**, has shorter seizures
(median 44.3 s vs 55.2 / 58.7 s), and shifts each modality's own baseline by up to 0.056 —
**amplifying whichever modality already leads on that task**. Never report a fusion number
on it.

### Use the aligned split instead

`train_pooled_eeg.video_val_sessions()` (P1.1) takes the session universe from the VIDEO
item list, so the EEG validation set is a strict **subset** of video's. `output/v3_eegalign/`
holds runs trained that way. `fusion/align_aligned_split.py` builds the pairing and verifies
it: **5,279 clips, 17 animals, labels agreeing 5,279/5,279, 40 video members and 10 EEG
members.** Start there.

Coverage limits:

- `v3_eegalign` and `v3_eeg250` cover the **3-class task only**, GRU and TCN, 5 seeds each.
- `output/ttg_eeg/aligned/` adds binary TCN detectors on the same split: 3 seeds, the
  grader's protocol A.
- No 5-class EEG runs exist on the aligned split.

### What the (biased-set) results already establish

The direction is consistent enough that the aligned re-run is expected to sharpen it, not
reverse it. Do not let a macro-F1 gain talk you out of it:

- **Detection: fusion wins.** 0.9745 -> 0.9904, +0.0158 [+0.0067, +0.0252].
- **Severity: every posterior-combination rule loses severe recall.** Video 41/88; average
  38, geometric 37, stacking 30. In 4,000 animal resamples, averaging never once beat video.
- **Severe recall is monotone in the video weight** (0.216 at pure EEG to 0.466 at pure
  video). There is no interior optimum — the macro-F1 peak near w = 0.5 is just where EEG's
  class-0 gain offsets the severe recall it destroys.
- **Within session, no variant beats video** (0.9014 vs 0.9014 at best). EEG severity
  collapses out of session: 0.8185 pooled -> 0.6123 within. Video loses only 0.048.
- **Rules that "raise severe recall" are prior reweighting, and video alone does it better**:
  video/prior gets 88/88 severe at macro-F1 0.7252 against fused rank-averaging's 88/88 at
  0.6164.
- **Max-confidence is not a disguised detection gate.** On the 2,830 clips it picks video on
  94–97% of clips. On the matched split it picks EEG for 18–24% of clips, loses 9 severe clips
  at argmax on `v3_eegalign`, and lowers the severe-vs-mild ranking on both EEG sets.
- **The bias figures** (shorter seizures, up to 0.056 baseline shift) reproduce as
  descriptions of this sample, but neither is systematic within animals. The sample is biased
  by which animals it holds.

What the matched split (5,279 clips, 200 severe) establishes:

- **Averaging costs severe recall and severe ranking.**
  - `v3_eegalign`: 99 → 88 of 200, interval excludes zero.
  - `v3_eeg250`: 99 → 95, not significant.
  - Severe-vs-mild AUROC falls on both: −0.0198 and −0.0186, intervals below zero.

  The plan of record's "−0.0765 severe recall, P = 0.001" does not reproduce, and neither does
  the 2,830-clip "costs about three clips".
- **The gate keeps severity where video had it.** EEG supplies only the seizure mass, and video
  keeps its mild/severe split. It gains +0.0115 at w = 0.5, and +0.0124 at the w = 0.6 picked
  leave-one-animal-out, with severe recall unchanged for w ≤ 0.6. `grader/` is this idea,
  trained and fitted.

### Rules for this directory

- Report **per-class recall with counts** beside every macro number, and report severity
  **within session** as well as pooled. Stage 5 is 14 clips on the old set.
- A fusion weight or threshold chosen on the test set is an upper bound — label it.
- Before claiming EEG adds severity information, run the **video-only control**: the same
  reweighting applied to video posteriors alone. For detection and the gate, the control
  recovers little (+0.0025 vs +0.0159; +0.0022 vs +0.0115), so those gains come from EEG.
  Averaging's severe loss at the operating point, though, is a threshold effect.

## Related repositories

Each is a separate folder with its own CLAUDE.md. Do not re-add their code here.

- `rodent-eeg-band-analysis` — epoch-export verification and frequency-band structure.
- `rodent-seizure-preictal` — whether anything changes before seizure onset (it does not).
- `EEG-seizure-classification` — the parent project, checked out at `EEG_ROOT`: base
  trainers, data, caches, checkpoints, every run output, and the paper.
