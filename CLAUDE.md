# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

Ensembling of validation posteriors already written to disk by a video–EEG rodent seizure
project. It contains **no training code and no data** — it reads stored `val_preds.npz` /
`val_clip_preds.npz` files and averages them. The modelling repo and the band-level analysis
are separate (`EEG-seizure-classification`, `rodent-eeg-band-analysis`).

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

`numpy` + `scipy` for everything; `scikit-learn` only for metrics. The original analysis box
has no sklearn in its base interpreter and its shared environment must not be installed into —
use a venv with `--system-site-packages`. Every script reads `EEG_ROOT` for the run tree.

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
  two are scored on different clip sets, so neither direction is established until both are
  recomputed on the 2,830-clip intersection.

## `fusion/` — read this before touching it

Cross-modal fusion of the video and EEG posteriors. **The results currently in this
directory were measured on the wrong set.** Fixing that is the open work.

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

Coverage limit: `v3_eegalign` covers the **3-class task only**, GRU and TCN, 5 seeds each.
Detection and 5-class fusion on the aligned split need EEG retraining first.

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
- **Max-confidence gating picks video on 94-97% of clips** — it is the detection gate wearing
  a different name.

Correction to the plan of record: the standing "-0.0765 severe recall, P = 0.001" does **not**
reproduce. Here it is -0.0341 with a sign test of p = 0.50. The honest statement is *fusion
never helps severe recall and sometimes costs about three clips*, not *significantly hurts*.

### Rules for this directory

- Report **per-class recall with counts** beside every macro number, and report severity
  **within session** as well as pooled. Stage 5 is 14 clips on the old set.
- A fusion weight or threshold chosen on the test set is an upper bound — label it.
- Before claiming EEG adds severity information, run the **video-only control**: the same
  reweighting applied to video posteriors alone. Every apparent fusion gain so far has been
  a threshold move video can make by itself.

## Related repositories

Each is a separate folder with its own CLAUDE.md. Do not re-add their code here.

- `rodent-eeg-band-analysis` — epoch-export verification and frequency-band structure.
- `rodent-seizure-preictal` — whether anything changes before seizure onset (it does not).
- `EEG-seizure-classification` — the parent project: training code, checkpoints, the paper.
