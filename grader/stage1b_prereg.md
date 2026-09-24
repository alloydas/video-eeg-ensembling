# Stage 1b pre-registration — written 2026-09-23 ~20:05 CDT, BEFORE any Stage 1b run

Observation that motivated it (Stage 1, seeds 1-3, session split seed 49):
within-session severe-vs-mild AUROC of the fixed X3D peaks at epochs 1-3 (mean ~0.84-0.85) and
falls to ~0.79-0.80 at epoch 12 as training loss reaches ~0.002 (memorisation) in all 9 fixed runs;
the bugged X3D (bounded logits, loss floor ~0.59) ends at ~0.81-0.82. Choosing an epoch from
those curves would be selection on the scored set, so the test below uses FRESH seeds 4-6.

Arms (x3d, --fix_x3d, --heads dual, session split seed 49, 12 epochs, lr 1e-4, batch 8, seeds 4,5,6):
  R0  control (no augmentation, unbounded logits)
  R1  --augment (flip, random-resized-crop 0.8-1.0, brightness/contrast +-0.1)
  R2  --logit_bound 1.0 (logits = tanh(z); a principled version of what the X3D bug did)
  R3  = R0 read at the PRE-REGISTERED early epoch 3 (no extra runs)

Primary endpoint: 3-seed mean within-session severe-vs-mild AUROC of the g3 head, at epoch 12 for
R0/R1/R2 and epoch 3 for R3; compared with R0 at epoch 12 (paired by seed where applicable).
Secondary: g3 and g5 macro-F1 with per-class recall and counts; g5 Stage-5 recall x/34; effect of
swapping each arm into the stored top-3x5 ensemble (best-epoch members are upper bounds, noted).
Adoption rule: an arm is adopted if its primary endpoint beats R0@12 by >= +0.02 AND its g3
macro-F1 is not lower than R0@12 by more than 0.01. Otherwise reported as not adopted.
Noise note: single-run within-session AUROC sd ~0.016-0.033; 3-seed means reduce this by ~1.7x.
