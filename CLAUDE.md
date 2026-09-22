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
