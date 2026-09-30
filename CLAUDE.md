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
  - Fixed X3D, mean of 3 video × 3 EEG seeds: 3-class 0.708 → **0.749**, 5-class
    0.503 → **0.547** (video alone → EGRG). All 9 seed pairs gain, by +0.0406 ± 0.0008 and
    +0.0437 ± 0.0009; the smallest gains are +0.039 and +0.042.
  - The level is set by the video seed (5-class sd 0.010), not the EEG seed (≤ 0.001). The
    first-reported 0.750 / 0.557 was video seed 1, the best of three at 5-class. Quote the
    seed mean.
  - **The gain is detection** (0.927 → 0.985). Severity is video's within-seizure split by
    construction, and severe hits move by −2 to +2 of 1,257 across the 9 pairs.
  - The original (bugged) X3D is 0.009 better at 3-class with the gate (0.758), but it finds
    0 of 194 Stage-5 clips in every seed. The fixed X3D is the recommended grader.
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
- **A pretrained video JEPA was killed at its pre-registered gate** (Step 0,
  `grader/step0_vjepa_prereg.md`, 2026-09-25). The model was frozen V-JEPA 2 ViT-L, with a linear
  probe and within-session severe-vs-mild AUROC on unseen animals.
  - Dense: 0.681. Frame-shuffled: 0.681 (d = +0.0005 [−0.019, +0.024]).
  - Frozen Kinetics X3D: 0.728. The fine-tuned X3D grader: 0.744.
  - Frame order carries no severity signal for either backbone on unseen animals (X3D 0.728 vs
    0.731 shuffled).
  - Do not fund JEPA / future-latent pretraining for severity without new evidence.
  - Not registered, a hypothesis only: V-JEPA 2 on the sparse whole-clip frames reached 0.747
    (vs X3D sparse 0.709). Testing it needs its own pre-registration.

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

## `rawdata/` — the days-long recordings the clips were cut from

The raw recordings are READ-ONLY, at `/work/mech-ai/alloydas/EEG/Data` (2.56 TB, 23 animal folders,
about 47 days each). They were uploaded 2026-09-28 and audited read-only by `rawdata/`. Outputs are
in `$EEG_ROOT/output/raw_audit/`.

- **Layout.**
  - One camera shows two animals. The first-named animal in the camera name is on the left.
  - Every camera file is stored under both partners' folders: 986 GB of byte-identical duplicates.
  - Per-animal crop boxes (unscaled, pixel-exact across days): `raw_audit/align/crop_boxes.csv`.
- **Clocks.**
  - The mp4 file name is the UTC **close** time.
  - The true start is XML `DSI_utc_start_time + DSI_sync_offset/1e7`.
  - The EDF header start is naive local time, about 18.4 s before the first video of a day folder.
    It drops the repeated hour at the 2023-11-05 fall-back.
- **EEG channel.** It is labelled `ECG`, or `EEG` for RN219 and RN243. Read `EEG` first, then `ECG`.
- **Problems.**
  - Unreadable: one mp4, `RN199-RN201.20231213214000`. At most about 2 h of it is recoverable.
  - Unfinalised: one EDF, `DONE_RN203/11-15-2023`, with 23.6 h readable.
  - 58 EDFs are under 1 h, so those days have video without EEG.
- **90 labelled clips have video from the wrong time** (EEG and label are right):
  - 10 from the 10-13 room-D file, about 175 min off;
  - 50 after a +61 s PC-clock step in the 11-03 and 12-05 room-C files;
  - 27 cut 1 h early across the DST change;
  - 3 in the repeated hour.

  By class: 43 non-seizure, 35 Stage 2, 10 Stage 3, 2 Stage 4. Also, one non-seizure clip
  (`RN242/10-12-2023/clip_03_vs_seizure_04_…`) contains an annotated Stage 3 seizure.
- **Step 1, done 2026-09-29** (`rawdata/recut_clips.py`, `rescore_recut.py`; outputs in
  `$EEG_ROOT/output/ttg_recut/`; nothing in `data_full/` was modified).
  - 172 candidate clips were each measured by EDF-Activity vs crop-motion lag.
  - 81 were re-cut to the right time, reproducing the parent ffmpeg pipeline exactly.
  - 11 are excluded: 5 labels have no video; 5 on 12-05 are unresolved; 1 is the mislabelled negative.
  - Re-scored without retraining, the change is below seed noise: 9-pair EGRG 3-class
    0.7488 → 0.7485, 5-class 0.5471 → 0.5452. Retraining is not warranted on its own.
  - **5 "seizure" labels have no seizure in EEG or in the corrected video.** They are RN229
    s39/s42, RN210 s22/s23 and RN213 s41. Two are duplicate labels of a neighbouring seizure,
    shifted by a clock change. So some xlsx annotations inherit the clock defects.
  - EGRG's Stage-2 hits partly came from quiet wrong-time video under an EEG seizure. With the
    correct video, the video grades those clips S3/S4.
  - Cameras RN242-RN243 and RN229 run 2e-4 slow: up to +5 s late at the end of a 7.5 h file,
    across all their clips. Not corrected.
- **Step 2, the false-alarm scan: STOPPED by its pre-registered validation** (2026-09-30,
  `rawdata/scan_prereg.md`, `$EEG_ROOT/output/ttg_scan/`). No window was scanned; it used 0.6 GPU-h.
  - Re-decoding the same frames from the raw video (differences of about 0.8 grey levels from the
    crf-18 training cache) flips 4–5% of per-head video argmax decisions. Keyframe snapping flips
    12–14%. The EGRG seizure flag agrees 99.4% and 98.5% respectively.
  - **So the video grader is brittle to invisible encoding differences.** A deployment on
    freshly decoded video will not reproduce the clip-level decisions exactly.
  - Also found:
    - Room-C files `.20231109055600` are about 13 s off the EDF clock, so 52 labelled clips may
      have video 13 s off. This was not corrected in step 1.
    - RN203's box includes the bottom of RN199's cage, and the labelled RN204 box includes
      RN201's: camera cross-talk.
    - RN227 seizure_110's clip EEG matches no raw EDF segment within ±48 h.
    - RN243's Stage 2 annotation is its cage-mate's seizure.
- **Annotations.**
  - The xlsx event logs are the only source.
  - Every labelled seizure matches one event exactly.
  - 328 annotated Stage 2–5 seizures with video were never clipped (S4 24, S5 1).
  - 782 have EEG but no video.
- **RN201, RN203, RN243** (not in the labelled 20) have 0, 0 and 6 annotated events in about 47
  days each. They are plausibly controls, but no document says so.

## Related repositories

Each is a separate folder with its own CLAUDE.md. Do not re-add their code here.

- `rodent-eeg-band-analysis` — epoch-export verification and frequency-band structure.
- `rodent-seizure-preictal` — whether anything changes before seizure onset (it does not).
- `EEG-seizure-classification` — the parent project, checked out at `EEG_ROOT`: base
  trainers, data, caches, checkpoints, every run output, and the paper.
