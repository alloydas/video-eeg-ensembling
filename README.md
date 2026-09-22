# Video–EEG model ensembling

Ensembling the validation posteriors that a video–EEG rodent seizure project already wrote to
disk — 175 stored runs, no retraining, no GPU. The result is asymmetric: **averaging improves
video at every task and makes EEG worse at every task.**

**[Forty Runs, No Retraining](docs/ensembling.html)** is the self-contained report. Open it in
a browser.

## The result

| pool | task | best single | all-member ensemble | change |
|---|---|---|---|---|
| video | detection | 0.9760 | **0.9792** | +0.0032 |
| video | 3-class | 0.8007 | **0.8114** | +0.0107 |
| video | 5-class | 0.6843 | **0.6979** | +0.0136 |
| EEG | detection | **0.9793** | 0.9725 | −0.0068 |
| EEG | 3-class | **0.7769** | 0.7651 | −0.0118 |
| EEG | 5-class | **0.5807** | 0.5608 | −0.0199 |

Macro-F1. Video: 40 runs, 10 backbones × 4 seeds, 5,289 clips. EEG: 35 runs, 7 architectures
× 5 seeds, 5,319 clips.

**Why:** pool quality, not error structure. Video members span 0.943–0.976 at detection; EEG
members span 0.227–0.777 at 3-class, mixing strong recurrent models with classical baselines
and a few collapsed runs. An unweighted mean is pulled toward the middle of that spread.

**What it is not:** the natural story — that EEG models fail on the *same* clips, so averaging
cannot help — is contradicted. The irreducible error share is 2.3% for EEG against 2.2% for
video at 5-class, and on severe clips EEG is *more* recoverable (5.0% vs 14.6%). The effect is
reproducible; the mechanism is not established.

Curating the EEG pool does not rescue it. Four selection rules were tried, two of them
*cheating* by selecting on the reported test scores, and even those only reach parity with the
best single run.

## What is here

| path | what it does |
|---|---|
| `ensembles/ens_both.py` | Seed / architecture ensembling for both modalities, with the alignment and repair steps below. |
| `ensembles/ens_sel.py` | Selective ensembling — deep-only, drop-collapsed, and two oracle rules. |
| `ensembles/ens_err.py` | Splits error into the part every member shares and the part averaging can recover. |
| `ensembles/align_modalities.py` | Builds the clip intersection between the two modalities' validation sets. |

## Three things that change the numbers

1. **Row order differs between runs.** Nine of forty video runs per task store their
   predictions in a different order. Members must be aligned on the stored clip `path` —
   averaging by row index mixes probability vectors across clips and *inflates* detection to
   0.986.
2. **Four video runs per task store double-softmaxed posteriors** and must be inverted before
   averaging (`unsquash`).
3. **Runs without a stored `path`** (the HuggingFace trainers) are admitted only if their raw
   label sequence matches the reference's, i.e. they are already in reference order.

## Running it

Python 3.11+ with `numpy`, `scipy` and `scikit-learn`. Point at the run tree and go:

```bash
export EEG_ROOT=/path/to/the/analysis/tree
python ensembles/ens_both.py     # per-task, per-backbone, all-member
python ensembles/ens_sel.py      # selective ensembling
python ensembles/ens_err.py      # shared vs recoverable error
```

Install `scikit-learn` into a virtual environment, not a shared base interpreter.

## Caveat that governs everything

The two modalities are scored on **different validation sets** — 5,289 and 5,319 clips,
overlapping by only 2,830. The table above is valid within each modality and is **not** a
modality comparison. `align_modalities.py` builds the intersection for anyone who needs one.

The companion analysis of the underlying frequency bands lives in a separate repository,
`rodent-eeg-band-analysis`.
