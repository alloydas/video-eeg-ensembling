# Step 0 pre-registration: frozen V-JEPA 2 severity probe on unseen animals — written 2026-09-25 ~10:45 CDT, BEFORE any V-JEPA 2 extraction

Revised 2026-09-25 ~11:05 CDT, after an independent review and before any V-JEPA 2 extraction or any probe fit
on the subject folds. No V-JEPA 2 number existed when the revision was made. The pre-revision draft was never
committed; the changes are listed under **Changes after critique** at the end.

Question. The EEG-Gated Racine Grader already does detection; what it has left to gain is severity ranking. Before
any JEPA pretraining on the raw video: does a frozen, pretrained video-JEPA representation (V-JEPA 2 ViT-L) carry
severe-vs-mild information on UNSEEN animals (a) beyond our current frozen starting point (Kinetics X3D-M), and
(b) as MOTION, meaning information that depends on frame order, rather than appearance? Stage 2 (`probe_extract.py`,
`probe_analysis.py`) asked (b) of X3D only, on the session split. It never ran on subject-disjoint folds.

Why ViT-L. ViT-L is the size we could realistically continue-pretrain on the raw video under the 17-GPU,
preemptible budget, so it is the backbone Step 0 must judge.

Looked at before writing or revising (no V-JEPA 2 forward pass, and no probe fitted on the subject folds):

- the label and fold counts below, the within-session pair counts per outer and inner fold, and native video sizes;
- the fine-tuned-grader score of section 5. On the 24,452-clip EGRG set it reproduces the quoted protocol-B values:
  fixed X3D 0.7438 / 0.7414 / 0.7394, 3-seed ensemble 0.7558; bugged X3D 0.7492 / 0.7590 / 0.7561, ensemble 0.7795.
  Recomputed on all 12,140 seizure clips, which is the Step 0 set, the values are those in section 5;
- the shuffle-permutation anchor of section 2.

## 1. Model facts (verified 2026-09-25, not assumed)

- Checkpoint `facebook/vjepa2-vitl-fpc64-256`, pinned revision `b3c1679b7c34d3255ef3547f27c7b226aefab26f` (MIT).
  Class `VJEPA2Model`. It needs transformers >= 4.53, pinned to `transformers==4.53.0`, the version whose
  `models/vjepa2/modeling_vjepa2.py` was read. It runs from the venv `/work/mech-ai-scratch/alloy/.venvs/vjepa2`
  (`--system-site-packages` from the eeg env; pins in `grader/requirements-vjepa2.txt`), with
  `HF_HOME=/work/mech-ai-scratch/alloy/hf_cache`. Download the model on the login node and run jobs with
  `HF_HUB_OFFLINE=1`. No encoder bug fix exists between 4.53.0 and 4.57.6. The only encoder change in that range
  replaces an einsum with an elementwise multiply, which is numerically identical under torch 2.4.1 in fp32 and bf16.
- `config.json` gives hidden_size 1024, 24 layers, 16 heads, mlp_ratio 4, patch_size 16, tubelet_size 2,
  crop_size = image_size 256, frames_per_clip 64 and layer_norm_eps 1e-6. The predictor (12 layers, width 384) is
  never used.
- Tokens. `Conv3d`, kernel = stride = (2, 16, 16): a 16-frame 256x256 view gives 8 x 16 x 16 = **2,048 tokens**,
  flattened time-major (token index = 256 t + 16 h + w, t = 0..7).
- Position. 3D RoPE is computed from the token index (frame id = index // 256), with no learned position table.
  `grid_size` is `crop_size // patch_size`; `grid_depth` is computed but never used. Any even frame count is
  therefore valid, and 16 frames need no interpolation. frames_per_clip = 64 does not enter the forward pass.
  H = W must be exactly 256: the 16 x 16 grid is taken from the config, so any other size would mis-assign height
  and width ids without raising an error. Under autocast RoPE runs in bf16, as in Meta's bf16 pretraining.
- Pretraining (arXiv 2506.09985, Table 9). The primary phase uses **16 frames at 4 fps, 256 crop, random-resize
  aspect ratio [0.75, 1.35]**, and the cooldown uses 64 frames at 4 fps. The paper's frozen evaluations use 16-frame
  clips at 256. The model card's "we use 64 frames" is a recommendation for inference, not a constraint.
- Normalisation. The repo has no `preprocessor_config.json` (404). `video_preprocessor_config.json` gives rescale
  1/255 and mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225), i.e. ImageNet; it also gives shortest_edge 292
  and crop 256. Read the constants from the pinned snapshot and assert them. The processor's spatial steps (resize
  the shortest edge to 292 bilinear, centre-crop 256) are **not** used; see section 2.
  - The extraction script normalises by itself: x = (u8 / 255 − mean_c) / std_c on the (B, C, T, H, W) uint8
    tensor, and then permutes to (B, T, C, H, W).
  - **Never call `train_pooled.norm_batch`** (`tp.norm_batch`), and never reuse `probe_extract.py`'s forward loop
    unchanged: `norm_batch` applies the Kinetics constants `KINETICS_MEAN` / `KINETICS_STD` (0.45 / 0.225).
- Encoder only: `model.encoder(pixel_values_videos=x).last_hidden_state` has shape (B, 2048, 1024) and includes the
  encoder's final LayerNorm. `model(x, skip_predictor=True)` is equivalent. Do **not** call `get_vision_features()`:
  in 4.53.0 it calls `forward()` without `skip_predictor`, so the predictor runs as well (fixed in 4.57.6). The
  input layout is **(B, T, C, H, W)**; the embedding permutes it to (B, C, T, H, W) internally.
- Intermediate layers. Block i (i = 0..23) is `model.encoder.layer[i]`; its output is element [0] of the returned
  tuple. The final LayerNorm is `model.encoder.layernorm`. Do **not** use `output_hidden_states`: its entry i is the
  *input* of block i, and only its last entry is normalised.
- Output range. The final LayerNorm weights of the pinned checkpoint lie in 1.04 to 5.01 and its biases within
  ±0.27, so outputs are at most about 160 in magnitude and f16 storage cannot overflow.
- Precision: `eval()`, `requires_grad_(False)`, fp32 weights, `attn_implementation="sdpa"`, and
  `torch.autocast("cuda", dtype=torch.bfloat16)`. Cast `last_hidden_state` and every hooked block output to fp32
  before any LayerNorm-then-mean, mean or difference. CPU tests run in fp32.

## 2. Views and extraction (`$EEG_ROOT/output/ttg_vjepa/`)

Items: `train_pooled.discover()` sorted by path, 24,497 clips. The fingerprint must equal Stage 2's `75e31ebd:24497`.
Frames are always decoded from `data_full/` (the f16s224 cache is 224 and is not used). Decoding converts BGR to RGB
and resizes the **full frame** to 256x256 with `cv2.INTER_AREA`, the same rule as Stage 2 at 224. Native frames are
small and vary in shape: in a 300-clip sample they include 320x480, 480x480, 344x228 and 360x210 (w/h 0.67 to 1.71),
all at 15 fps. For most clips one axis is therefore upsampled; cv2 4.10's INTER_AREA is close to bilinear when
enlarging (checked). The full-frame squash is kept for comparability with X3D. The frame indices are exactly
Stage 2's:

- **sparse16**: `C.decode_linspace(full, 16, 256)`, the historical linspace frames, at least 1.2 s apart.
- **dense**: `starts, single = C.snippet_starts(n_header, 8, 16, 2, 150)`, then
  `C.decode_seek_snippets(full, starts, 16, 2, 256)`. This gives 8 snippets x 16 frames at stride 2 (7.5 fps,
  2.13 s each).
- **dense_shuffled**: the same frames, reordered with `probe_extract.py`'s code.
  - Seed: `rng = np.random.default_rng(C.clip_seed(p))`, where `p` is the **full item string, including
    `/video.mp4`** (`data/.../video.mp4`, as at `probe_extract.py` lines 104 and 117). The docstring's "crc32 of
    the data/ key" is loose: seeding with `C.key_of(p)` changes every permutation and raises no error.
  - Permutations: `perm = [rng.permutation(16) for _ in range(8)]`. Frame t of shuffled snippet j is frame
    `perm[j][t]` of dense snippet j (`xd[i][:, pm[i]]` in (C, T, H, W)).
  - Anchor: the first sorted clip, `data/Data_RN197_cropped/10-12-2023 (2)/clip_01_vs_seizure_02_Stage_3_20231012_162605/video.mp4`,
    has `clip_seed` 3765462539 and `perm[0]` = [11, 14, 5, 8, 6, 1, 10, 13, 15, 7, 12, 3, 4, 0, 2, 9]. The wrong
    `key_of` rule would give [10, 13, 8, 14, 4, 5, 7, 6, 3, 0, 12, 2, 1, 11, 15, 9].

Assert per clip that `starts`, `n_header` and `single_snippet` equal Stage 2's `features.npz`, and run
`--verify_seek 20` at 256. Each clip has 17 views, each (16, 3, 256, 256) after normalisation. Shards, the resume
logic, retries, `--retry_failed`, atomic writes and the abort when more than 0.5% of clips fail are all as in
`probe_extract.py`. A failed clip gets NaN rows and is listed in `failed`, never filled.

**Order of work.** Every file keeps rows in sorted-path order, but the work is done in two passes so that a budget
stop cannot void Step 0:

- Pass 1 extracts the 12,140 seizure rows (`y5` >= 1), in sorted order, 512 rows per shard.
- Pass 2 extracts the 12,357 non-seizure rows the same way.
- The merge reassembles sorted-path order and asserts it against the fingerprinted item list.

Non-seizure clips never enter Step 0 and are about half the compute; section 8 says what happens when pass 2 does
not finish.

Stored per clip (`features.npz`: `path`, `y5`, `starts`, `n_header`, `single_snippet`, `failed`, `not_extracted`,
plus provenance including the revision, package versions, precision, the GPU check of section 8 and the extraction
script's commit):

- `sparse` [N,1024] f32: the mean of the 2,048 last-layer tokens.
- `dense_mean` and `shuffled_mean` [N,1024] f32: the mean over the 8 snippets of each snippet's token mean.
- `dense_tt` and `shuffled_tt` [N,8,8,1024] f16: for snippet j and temporal index t, the mean of tokens
  256t to 256t+255.
- `dense_tv` and `shuffled_tv` [N,1024] f32: the TT "token speed" v of section 7, computed on the GPU in fp32 from
  the fp32 temporal-token means (not from the f16 `*_tt`).
- `sparse_ml`, `dense_ml` and `shuffled_ml` [N,3,1024] f32: for blocks 17, 19 and 21 of `model.encoder.layer`
  (0-indexed; block 23 is the last).
  - A forward hook takes the block output (element [0], cast to fp32).
  - The hook applies `model.encoder.layernorm` to each token, then takes the mean over the 2,048 tokens.
  - Dense and shuffled entries are the mean over the 8 snippets of these snippet means, as for `*_mean`.
  - The encoder's final LayerNorm puts the four readout layers on one scale, and it makes the same path on block 23
    reproduce `last_hidden_state`; the dry run asserts this. Use hooks, not `output_hidden_states` (section 1).
- `perm` [N,8,16] int8: the shuffle permutations actually applied.

The merge asserts, for every clip and element, that:

- |`mean_{j,t}` of `*_tt` − `*_mean`| <= 1e-3 · (1 + max_{j,t} |`*_tt`|);
- the v recomputed from the f16 `*_tt` agrees with `*_tv` within the same tolerance;
- `perm` equals the permutations regenerated from the seed rule.

Both sides of each check come from the same fp32 tokens, so the only gap is f16 rounding (at most 2^-11 relative per
value, 2^-10 for a difference). A larger gap, or any non-finite stored value, means a bug and aborts. Expected size
is about 7.8 GB: `*_tt` 6.4 GB, `*_ml` 0.9 GB, `*_mean` and `sparse` 0.3 GB, `*_tv` 0.2 GB. The filesystem is at
98%, with 3.2 TB free.

## 3. Clips, labels, folds

- Scored clips: **seizure clips only** (`y5` >= 1): 12,140 clips, 595 sessions, 20 animals (S2 1,459, S3 9,419,
  S4 1,068, S5 194). Non-seizure clips are extracted but never enter Step 0.
- **Primary: severe vs mild**, with the g3 grouping: severe = S4+S5 (`y5` in {3,4}, 1,262 clips), mild = S2+S3
  (`y5` in {1,2}, 10,878 clips).
- **Secondary: S4 vs S3**: 1,068 vs 9,419; S2 and S5 are excluded.
- **Common clip set.** A seizure clip is used, in training and in scoring, for **every** arm only if it has:
  - finite values in every stored V-JEPA 2 array of section 2;
  - finite X3D `sparse`, `dense_mean` and `shuffled_mean` (`output/ttg_probe/features.npz`: 0 failed, dim 2048);
  - a `val_ep12.npz` posterior from all 3 fixed-X3D seeds.

  Anything else is dropped from all arms and counted by reason. Labels must agree across the path, both feature
  files and every dump (assert). If more than 0.5% of the 12,140 clips are dropped (over 60), stop and report.
  The dumps were checked on 2026-09-25 to cover exactly the 24,497 sorted discover() keys.
- **Folds (protocol B)**: `train_pooled.split_subjects(sorted items, 49, k, 5)`, k = 0..4, which is identical to
  `dhlib.split_subjects` and to the EGRG folds.
  - For every variant, seed and fold k, assert directly that the set of `path` strings in the dump equals
    {i[0] for i in `dhlib.split_subjects(items, 49, k, 5)[1]`}, and that the dump's `y5` agrees. Otherwise abort.
  - Do not use `joint_gate._check_fold_animals`: it needs `check_folds.expected_folds()`, which loads the EEG
    segment cache, and Step 0 uses no EEG.
  - Checked on 2026-09-25: the assertion holds for all 30 `x3dfix` / `x3dbug` dumps, and none is double-softmaxed.

| fold | held-out animals | seizure | severe (S4/S5) | mild (S2/S3) | sessions with both | within-session pairs |
|---|---|---|---|---|---|---|
| 0 | RN213 RN223 RN235 RN237 | 1,922 | 56 (49/7) | 1,866 (415/1,451) | 27 | 371 |
| 1 | RN197 RN208 RN215 RN238 | 2,098 | 179 (140/39) | 1,919 (39/1,880) | 45 | 7,227 |
| 2 | RN216 RN229 RN242 RN245 | 2,813 | 470 (415/55) | 2,343 (136/2,207) | 61 | 11,770 |
| 3 | RN199 RN210 RN219 RN224 | 3,741 | 370 (298/72) | 3,371 (824/2,547) | 64 | 8,647 |
| 4 | RN204 RN222 RN227 RN244 | 1,566 | 187 (166/21) | 1,379 (45/1,334) | 41 | 2,500 |

Within-session pairs are concentrated. Five animals hold 85% of the 30,515 severe-mild pairs: RN197 23%, RN229 21%,
RN242 18%, RN210 13%, RN224 10%. 18 of the 20 animals hold severe-mild pairs; RN215 and RN216 hold none, but they
still enter pooled AUROC and the bootstrap. S4 vs S3 has 25,521 pairs from 17 animals.

The inner folds of section 4 all hold within-session pairs. On the full seizure set, the smallest inner fold holds
1,746 severe-mild pairs and 1,517 S4-S3 pairs, both in outer fold 3.

## 4. Probe (identical for every feature arm, fitted separately per arm and per contrast)

- For outer fold k: train on the common-set contrast clips of the 16 other animals, and score fold k's clips
  **once**. Pool the out-of-fold (OOF) scores over the 5 folds, one score per clip.
- Model: `probe_analysis.fit_logreg`. This is sklearn-equivalent L2 logistic regression with
  `class_weight='balanced'` and the intercept unpenalised, solved with L-BFGS-B at maxiter 2000. Its input is
  `probe_analysis.standardise` (training-clip mean and sd + 1e-6). Import these functions; do not copy them.
  `probe_analysis.probe` is **not** used: its criterion is the mean pooled AUROC over 5 folds.
- Convergence is logged. `fit_logreg` gains a backward-compatible keyword `return_info=False`. With `True` it also
  returns `nit`, `success`, and the inf-norm of the objective's gradient at the solution. The default path is
  unchanged, so Stage 2 is unaffected. The change is committed with the analysis script (section 8).
  - Every fit's info is kept.
  - The results report, per (arm, contrast), how many fits reached maxiter or returned `success` False.
  - Such fits are used as returned. They are never refitted.
- C grid: {1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1, 3, 10}, 15 values. This is
  Stage 2's grid extended downward by one decade, because Stage 2's X3D probes chose 1e-4, and 3e-5 for S4 vs S3
  dense.
- Inner CV, **training animals only**. The inner folds are `probe_analysis.animal_folds(train_animals, k=4)`:
  whole animals, greedily balanced by the contrast's clip count.
  - For each inner fold f, standardise on the other 3 inner folds.
  - Fit along the grid in increasing C, warm-started from the previous C (as `probe_analysis.probe` does), and
    score fold f.
- **Criterion (decision 4).** crit(C) is the **unweighted mean over the 4 inner folds** of s_f(C):
  - s_f(C) is the within-session AUROC (`C.within_auroc`) of inner fold f's held-out scores, if that fold holds at
    least one within-session pair;
  - otherwise it is that fold's pooled AUROC (`C.auroc`);
  - a fold lacking a class entirely is skipped.

  Each use of the fallback or a skip is recorded. None is expected (section 3). Folds are not weighted by their
  pair counts, which range from 1,517 to 9,437.
- Choose `argmax round(crit, 6)`; ties go to the smaller C. Then refit on all training clips of fold k, standardised
  on them and cold-started (`w0=None`, as in Stage 2). A C at either grid end is reported as a boundary hit, and the
  grid is **not** extended.
- **Primary metric: WS**, the within-session AUROC of the pooled OOF scores (`C.within_auroc`: severe-mild pairs
  within the same session, ties count 1/2, pair-weighted). Pooled AUROC (`C.auroc`) is reported beside it.
- **CIs**: `ua, picks = C.animal_picks(animals of the contrast's common set, 2000, seed=11)` and `C.boot_auc`, which
  resamples every animal in that set (all 20 for severe vs mild). Every arm and every difference within a contrast
  uses the **same** picks, so the differences are paired draw by draw. The CI is `C.ci95` (2.5 / 97.5 percentiles of
  finite draws), and the count of non-finite draws is reported. Point estimates are full-sample values.
- **Full precision.** Every point estimate, difference and CI is computed from unrounded float64 values.
  - A difference's point estimate is the difference of the two unrounded full-sample WS values.
  - Its CI is `C.ci95` of the unrounded paired draws, `boot_a[:, 1] − boot_b[:, 1]`.
  - `probe_analysis.auc_summary` and `probe_analysis.delta` round to 4 decimals before differencing, so they must
    not feed any quantity of section 6.
  - Rounding is for display only.

## 5. Arms, all on the same clips and folds

- **VJ-dense** (the test arm), **VJ-shuffled** (motion control) and **VJ-sparse**: V-JEPA 2 `dense_mean` /
  `shuffled_mean` / `sparse`, 1024-d.
- **X3D-dense** (starting-point control), **X3D-sparse** and **X3D-shuffled**: Stage 2's frozen Kinetics X3D-M
  `dense_mean` / `sparse` / `shuffled_mean` features, 2048-d.
- **TT-dense** and **TT-shuffled** (secondary temporal-token readout; section 7.5), 2048-d.
- **ML-dense** and **ML-shuffled** (multi-layer readout; section 7.6), 4096-d, **primary contrast only**.
- **Grader** (fixed X3D, informative bar, not a gate). The score is
  `log(clip(P[:,2], 1e-300)) - log(clip(P[:,1], 1e-300))` of `probs_g3` from
  `output/ttg_vsubj/x3dfix_dual_s{1,2,3}_fold{f}/val_ep12.npz`, at the pre-registered last epoch 12. It is reported
  per seed and for the ensemble, where the ensemble is the severity logit of the **probability-averaged** posterior
  (`joint_gate.ensemble`). This is the reading of "3-seed mean" that reproduces the quoted 0.7558.
  - Every dump is checked with `dhlib.is_double_softmax`; any repair is counted.
  - The bugged X3D (`x3dbug_*`), scored the same way, is reported as context.
  - Bars recomputed on 2026-09-25 on all 12,140 seizure clips, 0 dropped, severe vs mild. About 8% of grader scores
    are tied (seed 1: 11,174 unique of 12,140), and ties count 1/2.

    | grader | seed 1 | seed 2 | seed 3 | ensemble |
    |---|---|---|---|---|
    | fixed X3D, WS | 0.7438 | 0.7415 | 0.7396 | 0.7559 |
    | fixed X3D, pooled AUROC | 0.7846 | 0.7940 | 0.7643 | 0.7966 |
    | bugged X3D, WS | 0.7496 | 0.7592 | 0.7564 | 0.7797 |
    | bugged X3D, pooled AUROC | 0.7299 | 0.7561 | 0.7578 | 0.7814 |

    If the common set drops any clip, the bars are recomputed on it and both versions are reported.

## 6. Pre-registered decision (primary contrast, WS, full precision)

    d_M = WS(VJ-dense) - WS(VJ-shuffled)          CI_M = C.ci95(boot(VJ-dense)[:,1] - boot(VJ-shuffled)[:,1])
    d_B = WS(VJ-dense) - WS(X3D-dense)            CI_B = C.ci95(boot(VJ-dense)[:,1] - boot(X3D-dense)[:,1])

    M (motion)       = CI_M[0] > 0
    B (better start) = d_B >= 0.02  AND  CI_B[0] > 0

    M and B  -> GO:          JEPA domain pretraining on the raw video is funded (temporal prediction is justified).
    B only   -> GO-ENCODER:  fine-tune V-JEPA 2 as the video grader directly; do NOT claim temporal prediction
                             helps; domain pretraining optional.
    M only   -> WEAK:        motion present but no better start; report, do not fund pretraining yet.
    neither  -> KILL the JEPA-for-severity plan (the EEG-gated grader stays).

Every WS value and draw is unrounded float64 (section 4). The comparisons are exact float64 comparisons, with no
tolerance. The decision is computed by the analysis script and written to `step0_results.json` together with its
inputs as raw floats: d_M, CI_M, d_B, CI_B, both WS values of each difference, and the non-finite draw counts.
Nothing else changes it. Whenever M holds, the results print the section 9 caveat on what M means for V-JEPA 2
next to it, with the diagnostics of section 7.7.

## 7. Reported, never gating

1. Every arm's WS and pooled AUROC with CIs, per contrast, alongside:
   - the class counts, animals, sessions and within-session pairs;
   - the per-fold WS;
   - the chosen C per fold and any boundary hits;
   - the convergence counts (section 4);
   - the dropped and failed counts by reason.
2. Paired differences:
   - VJ-dense − VJ-sparse;
   - VJ-sparse − X3D-sparse;
   - X3D-dense − X3D-shuffled, Stage 2's kill test re-read on unseen animals;
   - the M and B quantities for S4 vs S3.
3. **VJ-dense vs the grader**: WS(VJ-dense) − WS(grader), paired CI, for each fixed seed and the fixed ensemble.
4. **LOAO stack.** `probe_analysis.loao` (the Stage-2 `fit_predict`: median impute, standardise, class-balanced L2
   with lam = 1, leave-one-animal-out over the 20 animals) is fitted twice:
   - on [fixed-ensemble grader score, VJ-dense OOF score];
   - on [fixed-ensemble grader score] alone.

   Report the ΔWS (stack − grader-alone) with its paired CI, and the same stack with VJ-shuffled or X3D-dense in
   place of VJ-dense as controls. Caveat: for a held-out animal, the OOF scores of animals in other folds came from
   probes trained with it. This is the same accepted second-order leak as the EGRG gate.
5. **Secondary temporal-token readout (TT)**, hyperparameters fixed now. For view V in {dense, shuffled} and clip i:
   - u_{j,t} is the fp32 mean of tokens 256t to 256t+255 of snippet j (the fp32 source of `V_tt[i,j,t]`);
   - m = `V_mean[i]`, which equals the mean over (j, t) of u_{j,t};
   - v = `V_tv[i]` = mean over j = 0..7 and t = 0..6 of |u_{j,t+1} − u_{j,t}|, elementwise, i.e. the mean of 56
     absolute first differences ("token speed"). It is computed in fp32 on the GPU, because f16 steps of about 1e-3
     to 2e-3 at magnitudes 2 to 4 would add a positive noise floor to small temporal differences;
   - x_TT = [m, v], 2048-d.

   TT goes through the same probe, grid, inner CV, folds and picks. Report WS(TT-dense), WS(TT-shuffled),
   TT-dense − TT-shuffled and TT-dense − VJ-dense (paired CIs). Because m is exactly the VJ feature, TT-dense −
   VJ-dense isolates what v adds. Clips that use a single snippet are kept, and their count is reported.

   A positive TT-dense − TT-shuffled bound without M would be noted as "order information that mean pooling hides".
   It does not change section 6.
6. **Multi-layer readout (ML)**, primary contrast only, hyperparameters fixed now. For view V in {dense, shuffled},
   x_ML = [`V_ml[:,0]`, `V_ml[:,1]`, `V_ml[:,2]`, `V_mean`], 4096-d: blocks 17, 19, 21 and the last layer, each
   LayerNorm-then-mean-pooled. It goes through the same probe, grid, inner CV, folds and picks. Report WS(ML-dense),
   WS(ML-shuffled), ML-dense − X3D-dense (B's quantity under this readout) and ML-dense − ML-shuffled (M's quantity
   under it), with paired CIs. The ML readout does not change section 6.
7. **Shuffle diagnostics**, for both backbones on the common severe-vs-mild set:
   - the per-clip cosine similarity between `dense_mean` and `shuffled_mean`, as median and IQR, overall and per
     class (severe / mild);
   - the WS of the score 1 − cos with severe positive, i.e. whether the shuffle's effect on the representation
     itself tracks severity.

   These are printed next to M.
8. **Robustness of M and B.** These are recomputed with each of the five pair-heavy animals (RN197, RN229, RN242,
   RN210, RN224) removed in turn.
   - The OOF scores are not refitted. The animal's clips leave the scoring, and the picks are redrawn with
     `C.animal_picks(remaining animals, 2000, seed=11)`.
   - Report d_M, CI_M, d_B, CI_B and whether M and B would hold.
   - Also report the per-animal WS (U_a / N_a from `C.ws_parts`) of each of the 18 pair-holding animals, and count
     how many favour VJ-dense over VJ-shuffled and over X3D-dense. Ties are counted separately.

## 8. Budget, failures, stopping, code freeze

- The extraction runs on 1 GPU, `a100|h200|l40s` (all supported by torch 2.4.1+cu121; `C.check_cuda_arch()`), on
  the scavenger partition with `--requeue`, following `grader/sbatch_probe.sh`. The orchestrator submits it.
- Precision check. The first GPU segment runs its first batch twice, once under bf16 autocast and once in fp32.
  - It records, per view type, the cosine between the two token means.
  - The check goes into the provenance, and a cosine below 0.99 aborts.
- **The cap is 8 GPU-hours** of cumulative allocated GPU time over every job of the run, including preempted
  segments.
  - `budget.json` in the output directory records each segment's SLURM job id, start time and last-write time.
  - It is rewritten at every shard flush, at least every 10 minutes, and from a SIGTERM handler, since preemption
    and requeue send SIGTERM before the kill.
  - At the start of each segment, the running total is the larger of the `budget.json` total and the `sacct`
    elapsed total for the recorded job ids, when `sacct` answers. The source used is logged.
  - No new shard starts if the total plus one shard's measured duration would exceed 8 h.
  - `sacct` totals are reported as the authoritative figure.
- The estimate is about 1.65 TFLOP per view, times 17 views, times 24,497 clips, about 7e17 FLOP. That is roughly
  2 h at 100 TFLOP/s sustained bf16, or about 3.6 clips/s. Stage 2's decode-plus-X3D pipeline ran at 9.82 clips/s
  (`logs/ttg/probe_16583661.out`), so decoding is not the bottleneck. Pass 1 (seizure clips) is about half of the
  compute.
- If the cap is hit:
  - **during pass 1**: the run stops and is reported with the number of clips done, and **no decision is made on a
    partial set**. Any extension is a logged Deviation.
  - **during pass 2**: the scored set is complete. The merge writes `features.npz` with the unextracted non-seizure
    rows NaN and listed in `not_extracted` (not in `failed`; they do not count toward the 0.5% failure abort). The
    decision is made, and the shortfall is reported.
- **Budget overage.** If `features.npz` completes but `sacct` shows more than 8 h, because `budget.json` lagged, the
  decision is still made from the complete features. The overage is reported as a Deviation, with its size.
- **Code freeze.** Before the real `ttg_vjepa/features.npz` exists:
  - The analysis script and the `fit_logreg` keyword are committed, and they run end to end on stand-in features
    under `$EEG_ROOT/output/ttg_tmp/vjepa_standin/`. The run covers every arm, both contrasts, the decision and the
    output files.
  - Stand-ins: the X3D arrays copied into the V-JEPA 2 slots (dim 2048), plus seeded synthetic `*_tt`, `*_tv` and
    `*_ml` that satisfy the file's consistency checks. The stand-in decision is meaningless and is labelled so.
  - `step0_results.json` records the commit hashes of this pre-registration, the extraction script and the analysis
    script, and the sha256 of both features files.
  - Any later change to the analysis code is a Deviation.
- The analysis runs on CPU only, one job per (contrast, arm), 18 in all: 10 arms for severe vs mild, 8 for S4 vs S3.
  - Each job does 5 outer folds x (4 inner folds x 15 C + 1 refit) = 305 fits. Stage 2 took about 5 s per 2048-d
    fit (2,164 s for its 6 probes plus the kill test), so a job takes about 25 to 45 min at 2048-d, less at 1024-d,
    and more at 4096-d.
  - Jobs run as scavenger CPU jobs with `--requeue` and a fixed `OMP_NUM_THREADS=4`, which is recorded. Each job
    caches its per-outer-fold OOF scores atomically, so a preempted job resumes.
  - The assembly step (bootstraps, stacks, diagnostics, decision) runs after all jobs have finished. Every step is
    deterministic.
- Tests go under `$EEG_ROOT/output/ttg_tmp/vjepa_<name>/`. Required before the real run: a CPU dry run on the first
  3 sorted clips (fp32, about 84 TFLOP).
  - It runs under
    `srun -A mech-ai-scavenger -q scavenger -p scavenger -c 8 --mem=32G -t 01:00:00`, never on the login node:
    10 clips would be about 280 TFLOP, or 15 to 30 min.
  - It checks the token shape (1, 2048, 1024).
  - It checks the normalisation constants against the snapshot.
  - It asserts the tensor actually fed to the model. For one known (snippet, t, h, w) and each RGB channel c,
    `x_fed[b, t, c, h, w]` must equal (`u8[b, c, t, h, w]` / 255 − mean_c) / std_c within 1e-6. This catches Kinetics
    constants, a wrong channel order and a permutation applied before normalising.
  - It asserts the shuffle. For each shuffled snippet j and each t, the fed frame t equals the fed dense frame
    `perm[j][t]` along the time axis of (B, T, C, H, W). The first clip's `perm[0]` must equal the section 2 anchor.
  - It checks `starts` against Stage 2's.
  - It checks the `*_tt` / `*_mean` and f16-v / `*_tv` consistency.
  - It asserts that the ML hook path applied to block 23 reproduces the mean of `last_hidden_state` within 1e-5
    relative.
  - It runs `--verify_seek`.

## 9. Known limitations, fixed now (they are not grounds for re-running)

- **Readout asymmetry against V-JEPA 2.** The two backbones are not read on equal terms.
  - X3D's pooled 2048-d vector is its native readout: its supervised head global-average-pools and then projects.
  - V-JEPA 2 is evaluated in its paper with 4-layer attentive probes. For motion tasks (Jester, Diving-48) these
    attend to blocks 17, 19, 21 and 23 of ViT-L, not the last layer alone (arXiv 2506.09985, Tables 16-17, as read
    in review).
  - A last-layer mean with a linear probe is therefore weaker for V-JEPA 2 than for X3D, and it discards where and
    when things happen.
  - B carries a structural false-negative risk. A KILL means "not linearly available in mean-pooled frozen
    last-layer tokens". The ML readout (section 7.6) measures part of this risk but does not change the decision.
- **What M means for V-JEPA 2.** V-JEPA 2 was pretrained to predict latents of temporally coherent video, so a
  within-snippet shuffle is out of distribution for it. The shuffle can disturb its appearance encoding as well as
  remove order. VJ-dense − VJ-shuffled > 0 can therefore occur without any severity-relevant motion. X3D showed no
  such degradation in Stage 2 (shuffled >= dense on the session split). Because "M and B" reads as "temporal
  prediction is justified", this caveat is printed with M whenever M holds (section 6), alongside the shuffle
  diagnostics of section 7.7.
- **What the shuffle control tests.** Shuffling within a 2.13-s snippet keeps the set of frames, so motion magnitude
  partly survives as appearance diversity. M tests order dependence, not motion versus no motion.
- **Input mismatch.** Squashing the full frame to a square distorts aspect by 0.67 to 1.71, against the pretraining
  range of [0.75, 1.35]. Dense snippets run at 7.5 fps, against pretraining's 4 fps. sparse16, at 1.2 s or more
  between frames (with one tubelet fusing frames at least 1.2 s apart), is far outside that range.
- **Precision.** One fold assignment and 20 animals, with 85% of the pairs in 5 animals, so the CIs will be wide.
  The percentile animal bootstrap holds the probe fits fixed, so fitting variability is not propagated (as with
  EGRG), and the CIs are probably too narrow. Section 7.8 is the pre-registered check on this. It does not gate.
- **Scope.** ViT-L only, frozen only. Step 0 says nothing about what fine-tuning V-JEPA 2 would reach.

## 10. Outputs and deviations

`$EEG_ROOT/output/ttg_vjepa/`: `shards/`, `features.npz`, `budget.json`, `step0_results.json` and
`step0_results.txt`. The results must print:

- the decision, and the M and B quantities with their CIs, displayed at 4 decimals; the raw float64 values are in
  the JSON;
- every count in section 3;
- the budget used, from `sacct`;
- the commit hashes of section 8.

This pre-registration is committed before any extraction starts. Nothing in this document is chosen after seeing
V-JEPA 2 numbers. Any departure from it (code, clips, budget, a bug fix) is listed in a **Deviations** section of
the results, with its reason and whether it could change the decision. It is never edited into this
pre-registration.

## Changes after critique (2026-09-25 ~11:05 CDT, before any extraction)

Blocking:

- **Inner-CV criterion (section 4).** It is now the unweighted mean of the four per-inner-fold within-session
  AUROCs, with a per-fold fallback to pooled AUROC, as decision 4 says. It had been the pair-weighted within-session
  AUROC of the pooled inner scores.
- **Gating arithmetic (sections 4 and 6).** M and B are now evaluated on unrounded float64 values, with no tolerance.
  `auc_summary` / `delta` (4-decimal rounding) are barred from the gate, and the raw values go to
  `step0_results.json`.

Nonblocking, adopted:

- Normalisation: never `tp.norm_batch`, plus a dry-run assertion on the fed tensor.
- Shuffle seed: the full `/video.mp4` string, with an anchor permutation; `perm` is stored; a dry-run frame-order
  assertion.
- Limitations: a caveat on what M means for V-JEPA 2, with cosine diagnostics (section 7.7).
- Limitations: the readout asymmetry is stated plainly. Blocks 17 / 19 / 21 are stored via hooks, with the final
  LayerNorm, and read as the non-gating ML arm.
- TT: v is computed in fp32 on the GPU (`*_tv`), with an f16 agreement check. m is taken from `*_mean`.
- Code freeze: a stand-in end-to-end run, with commit hashes and feature sha256 recorded.
- Robustness: drop-one-animal for the five pair-heavy animals, plus per-animal counts (section 7.8).
- C grid: extended to 1e-6 and 3e-6. L-BFGS convergence is logged through a backward-compatible
  `fit_logreg(return_info=...)`.
- Budget: seizure clips first; `budget.json` written at every flush and on SIGTERM; `sacct` reconciliation; the pass
  1 / pass 2 stop rules; the overage rule.
- Dry run: 3 clips under `srun -c 8`. Analysis: runtime and job layout stated.
- Grader bars: recomputed and written in (section 5); the probability-averaged ensemble reading is kept.
- Fold check: asserted directly against `dhlib.split_subjects`, not `joint_gate._check_fold_animals`.
- Process: the revision line under the header, and the pre-registration committed before extraction.

Added by the reviser, not asked for by the review; all are non-gating or pure safety checks:

- The bf16-vs-fp32 cosine check on the first GPU batch (section 8), which aborts below 0.99.
- The ML readout applies the final LayerNorm before the mean, with the block-23 identity as a check.
- The WS of 1 − cos as a severity score (section 7.7).
- The explicit cold-start refit.
- The inner-fold pair minima.

Unchanged: the orchestrator's decisions 1 to 8. The review's answers to the open questions are kept:

- 4 inner folds;
- TT as [m, v];
- the bugged X3D reported as context;
- scoring on the 12,140 seizure clips;
- the full-frame squash.
