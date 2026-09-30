# Pre-registration: the EEG-gated grader on continuous recordings — written 2026-09-29, BEFORE any scanning

Status when written:

- No network has been run on any raw-recording window.
- No crop box has been derived for RN201, RN203 or RN243, and no frame of those animals has been viewed.

What was looked at before writing:

- the audit outputs (`raw_audit/per_animal.json`, `events.csv`, `files.csv`, `align/crop_boxes.csv`,
  `align/clips.csv`);
- the fold assignment;
- the gate code (`grader/eeg/joint_gate.py`);
- step 1's re-scored grid (`ttg_recut/rescore/summary.json`).

The sha256 of this file goes into every stage's `run.json`. Any change made after scanning starts is logged in §12,
with its time and reason. Whether RN201, RN203 and RN243 are controls is **unknown**, because no document says so
(the user has been asked). Nothing below depends on the answer. Only the wording of §11 does.

## 1. Question

At its fixed operating point, how often does the EEG-Gated Racine Grader flag a seizure? The grader is the fixed
X3D-M at epoch 12, plus the TCN detector at epoch 30, plus the EGRG gate. It is measured in continuous recordings
of the three animals with no labelled seizures (RN201, RN203, RN243). The comparison is the seizure-free time of
their camera partners (RN199, RN204, RN242). The same operating point is then used to measure sensitivity on the
partners' annotated seizures. Nothing is retrained. The gate is not refitted on scan data, and no threshold is
moved.

## 2. Animals, cameras, models

| camera (room) | no-label animal (side, box) | partner (side, box, OOF fold) | video∩EDF h, no-label / partner (audit) | partner events with video + EDF |
|---|---|---|---|---|
| RN199-RN201 (C) | RN201, right, derived in §8 | RN199, left, (0,180,400,360), fold 3 | 1,414 / 961 | 803 (S2 422, S3 320, S4 36, S5 25), 783 clipped |
| RN203-RN204 (C) | RN203, left, derived in §8 | RN204, right, (480,0,320,480), fold 4 | 1,399 / 1,400 | 50 Stage 2–5 (all clipped) + 3 Spikes |
| RN242-RN243 (D) | RN243, right, derived in §8 | RN242, left, (48,264,352,216), fold 2 | 1,445 / 1,464 | 1,412 (S2 111, S3 1,104, S4 186, S5 11), 1,403 clipped |

The fold numbers come from `split_subjects` with seed 49, and `rescore/summary.json` gives the same folds.
`check_folds` asserts them again at run time. RN243 has 6 annotated events: 4 Spikes, one Stage 2 (10-12, which may
belong to RN242) and one Stage 5 (10-16). RN201 and RN203 have none.

- **Networks.**
  - Video: `ttg_vsubj/x3dfix_dual_s{1,2,3}_fold{f}/last.pt`. Each is asserted to be the state after epoch 12 of 12,
    and is run through `train_grader`'s eval path (GraderNet, `tp.norm_batch`, `tp._autocast`, `head_probs`) as
    in `rescore_recut.py`, on CUDA.
  - EEG: `ttg_eeg/subject/tcn_bin_fold{f}_s{1,2,3}/final.pt`. Each is asserted equal to `last.pt`'s model at epoch
    30 of 30, and run in fp32.
- **Configuration.** A configuration is (video seed, EEG seed, fold).
  - A partner uses only its own out-of-fold fold: 9 configurations (protocol B).
  - A no-label animal uses all 5 folds: 45 configurations.
  - Every summary is given as the mean and the range (min–max) over configurations. For no-label animals the
    per-fold means are also listed.

## 3. Windows and exclusions

- **Grid.** Non-overlapping 60-s windows `[m, m+60)` at whole UTC minutes, on the true video clock. Frame j of a
  file is at `DSI_utc_start_time + DSI_sync_offset/1e7 + pts_j`. The frame pts and keyframe flags come from
  `ffprobe -show_packets`.
  - RN242-RN243's pts run 2e-4 slow. This is not corrected; it is at most 5.4 s at the end of a file.
  - Both animals of a camera share the grid.
- **EDF clock.** An EDF sample k is at the naive local time `header start + k/1000`. It is mapped to UTC as UTC−5
  before naive 2023-11-05 01:00, and UTC−6 from that time on. The EDF's naive 01:00–02:00 on 11-05 is the CST hour.
  The CDT hour, 06:00–07:00 UTC, has no EEG.
  - EDFs are placed by their header time, not by their folder, so the misfiled `10-12-2023 (3)` EDFs are handled.
  - The room-C starts at +198.5 s and −748 s on 10-19 and 11-17 need no correction. On those days EDF Activity and
    video motion agree at −1 s.
- **Inclusion.** A window is scanned only when all of the following hold:
  1. One readable video file that is not a defect file covers all 60 s with frames, with no pts gap over 1 s.
  2. Exactly one EDF covers all 60 s with complete records.
  3. The EEG is not in dropout (see the dropout reason below).
  4. Its clock block is not excluded.
  5. All 16 of its input frames decode.
- **Accounting.** Every minute from the animal's first to its last recorded instant gets exactly one status. An
  excluded minute takes the first reason that applies, in this order:
  - `no video`: rollovers (about 1 min per file), gaps across day folders, and times with no recording.
  - `unreadable file`: `RN199-RN201.20231213214000.mp4`. It is not recovered.
  - `timing-defect file`: whole files, on the cameras that have them. These are room D `…20231014004100` (10-13) and
    room C `…20231025153704`, `…20231103122100` and `…20231205222300`.
  - `EDF stub`: a day folder whose EDF is under 1 h.
  - `DST hour`.
  - `unfinalised EDF`: `DONE_RN203/11-15-2023` after its 8,508 complete records (23.6 h are used).
  - `EDF gap`.
  - `no 1-kHz channel`.
  - `EDF ambiguous`: two EDFs cover the minute.
  - `partial window`.
  - `EEG dropout`: the window's 60,000 samples at 1 kHz hold a run of at least 1,000 identical consecutive values.
  - `clock`: see below.
  - `decode error`.

  Scanned and excluded hours are reported per reason. Nothing is dropped silently.
- **Clock check.** It uses motion only, and is finished before any network output is read. The unit is a 1-h block
  per camera file and box.
  - Motion is the mean |difference| of consecutive keyframes (2.13 s apart) inside the box. It is interpolated to
    1 Hz and rank-transformed.
  - Motion is correlated with EDF `Activity` at lags of ±120 s. On 2023-11-05 lags of ±3,600 ± 120 s are also
    searched.
  - A block is excluded when its measurement is strong (`recut_clips.STRONG` / `is_strong`) and its peak lies more
    than 10 s from the aligned value (Activity lagging motion by 0.74 s, as in `recut_clips.py`).
  - Weak blocks are kept. The report gives the number of blocks, the number that are strong, the number that are
    strong and consistent, and the number excluded.

## 4. Inputs

- **Video (primary mode K).**
  1. Let n be the number of frames in the window. The positions are `train_classifier.sample_frame_indices(n, 16)`,
     the f16s224 linspace rule.
  2. Each position is snapped to the nearest keyframe in the same file, with ties going to the earlier one. The
     keyframe interval is 32 frames, so a frame moves by at most 1.07 s.
  3. Only keyframes are decoded, with PyAV `skip_frame="NONKEY"` to `bgr24`.
  4. Each frame is cropped `[y:y+h, x:x+w]` with the animal's box, converted BGR→RGB and resized to 224×224 with
     `cv2.INTER_AREA`.

  Mode E (the fallback, §7) differs only in step 2: it decodes the exact frames, sequentially.
- **EEG.**
  - The window's 60,000 samples come from the first channel whose label contains `EEG`, or else the first containing
    `ECG` (case-insensitive). This is `build_stage_segments_pooled.one()`, and the channel must be at 1 kHz.
  - The samples are read with a numpy EDF record reader using MNE's physical scaling.
  - The processing is exactly that of one 60-s clip: `resample_poly(sig, 1, 8)` on these samples alone, then
    float32, 6-s windows at a 3-s stride (19 windows), then the per-window z-score of `load_cache`.
  - Then come the TCN softmax and `aggregate_clip(..., "logmean")`, which gives P_E(sz).

## 5. Gates and decisions

- **Gate.** q = σ(a·logit P_V(sz) + b·logit P_E(sz) + c), as in `joint_gate.py`.
  - P_V(sz) = 1 − P_V(non-seizure) from the task head. The logit is clipped at 1e-6. P_E is the logmean score.
  - The fit is `LogisticRegression(C=1, max_iter=5000)` on seizure vs not.
  - The data are the patched protocol-B OOF dumps: `ttg_recut/vsubj_patched` and the `tcn_bin` epoch-30 dumps,
    24,441 clips. They are loaded with `joint_gate`'s loaders, which log path alignment and drops.
  - Gates are fitted per (video seed, EEG seed) and per task:
    - **G_all**, fitted on all 20 animals, is used for the no-label animals.
    - **G_−P**, fitted without partner P (its LOAO gate), is used for P.
- **Reproduction gate (stop on failure).** The same code, run leave-one-animal-out over all 20 animals, must
  reproduce `rescore/summary.json` (x3dfix, patched) exactly: the per-pair `egrg_hits` and `egrg_f1` of all 9 pairs,
  for both g3 and g5.
- **Decision.** class = argmax [1 − q, q·P_V(g|sz)], unchanged. A window is **flagged** when class ≠ non-seizure,
  which is equivalent to q > 1/(1 + max_g P_V(g|sz)). At g5 that threshold lies between 0.5 and 0.8.
  - **Primary task: g5.** Its grades are Racine stages. g3 is reported alongside, with its own head and its own gate.
  - Context systems:
    - video alone: argmax P_V ≠ non-seizure, one configuration per video network;
    - EEG alone: P_E(sz) > 0.5, one configuration per EEG network.

## 6. Outputs

For each animal, system (EGRG, video alone, EEG alone) and task:

- scanned hours, and excluded hours by reason (§3);
- **flagged windows per scanned hour**, for any grade and for each grade;
- **flagged episodes per scanned day** (scanned h / 24). Consecutive flagged windows are merged. An episode whose
  start is ≤ 300 s after the previous episode's end is merged with it, measured in wall time;
- day-to-day spread: the median and IQR over days of the configuration-mean rate;
- a descriptive hour-of-day profile;
- q quantiles.

**Partners' seizure-free time.** A partner window counts only when both of these hold:

- it is ≥ 600 s from every annotated event of that animal, of any label (Spikes included, and events without video
  included);
- its day folder has an xlsx.

The partners' rates are reported on this time and on all time. RN243 gets the same ±10-min rule around its own 6
events. Its Stage 2 and Stage 5 events are described one by one.

**Primary comparison (matched), one per camera.**

- Both animals are scored with the partner's fold-f networks (9 configurations) and the partner's G_−P gate.
- The time is matched: windows scanned for both animals and ≥ 600 s from any event of either.
- The only difference left is the animal. The report gives r_N, r_P, r_N / r_P and r_N − r_P per configuration,
  their mean and range, and the sign count over the 9 configurations.

**Sensitivity (partners, at the same operating point).**

- An annotated Stage 2–5 event is **scorable** when every window overlapping [start, end] (UTC, with the xlsx naive
  times mapped as in §3) is scanned. Events that are not scorable are counted by reason.
- An event is **detected** when at least one overlapping window is flagged.
- It is reported by stage with counts, split into clipped and never-clipped events, together with a chance level of
  Σ 1 − (1 − f)^k. Here f is the configuration's seizure-free flag share and k is the event's number of windows.
- Also reported:
  - the table of annotated stage against the highest flagged grade;
  - the same measurements for video alone and EEG alone.

**What the gate does.** Per configuration, the 2×2 table of video-alone flag against EEG-alone flag, with the EGRG
flag share in each cell.

**Cross-talk.** The no-label animal's flag rate in windows that overlap the partner's events ±60 s, against its
matched-time rate.

**OOF reference.** This is computed in S0 from the dumps only, before any scan output exists. It is the per-config
non-seizure-clip flag share × 60, i.e. the flags per hour if every window behaved like an OOF non-seizure clip.

- The x3dfix 9-pair mean at g5 is 168.4 of 12,338 = 1.37%, or **0.82 per hour for EGRG**.
- For video alone it is 574 of 12,338 = 4.65%, or **2.79 per hour**.
- The reference for sensitivity is the OOF seizure-clip detection share.

Outputs:

- `results/summary.{json,txt}`;
- per-window predictions in `preds/<animal>/<day>.npz`. They hold P_V (g3, g5) for each video network, and P_E with
  the 19 window posteriors for each EEG network.

## 7. Step A: validation, run before the scan

**Sample.** 360 labelled clips with seed 20260929:

- 120 non-seizure, and 60 each of S2, S3, S4 and S5. If fewer S5 clips are eligible, all of them are used and the
  sample is topped up from S4;
- drawn round-robin over animals in a seeded order;
- at least 15 clips each from RN199, RN204 and RN242.

A clip is eligible when all of these hold:

- it is in the patched OOF dumps;
- it is not among step 1's 172 candidates or 11 exclusions;
- its duration is 50–75 s, so that the snapping geometry is close to that of a 60-s window;
- it lies inside one readable raw file that is not a defect file.

The raw frame of clip frame j is r0 + j. Here r0 is the keyframe at or before the cutter's seek, or the one before
it, whichever has the higher NCC with clip frame 0. The NCC must be ≥ 0.99, or the clip is redrawn.

**Arms.** Each is predicted by the clip's 3 OOF video networks on CUDA, with its own f16s224 frame count n.

- **C**: the stored `cache_frames/f16s224` row. This is the noise-floor control. If C is not within median |Δp|
  ≤ 0.01, the inference code is wrong and must be fixed first.
- **K**: raw keyframe-snapped frames with the labelled animal's `crop_boxes.csv` box (§4).
- **E**: raw exact frames r0 + `sample_frame_indices(n, 16)`, with the same crop.

**Tolerance, fixed now.** It is judged against the stored `val_ep12.npz` (patched), for each head separately:

- argmax agreement ≥ **97%** of (clip, network) rows;
- median |Δp| ≤ **0.05**, where |Δp| is the largest absolute class difference in a row.

Both heads must pass. Also reported, for information only:

- the p95 of |Δp|;
- the mean signed Δ P_V(sz);
- EGRG flag agreement, using the stored EEG and G_all;
- the macro-F1 of each arm.

**EEG arm R.** The clip's samples are read from the raw EDF, starting at its `Clip start` on the EDF clock (the
audit found a lag of 0 samples), with the clip EDF's sample count, and processed as in §4.

- R is compared with the stored `val_clip_ep30` `p_logmean` of the 3 OOF TCNs. It must reach argmax agreement
  ≥ **99%** and median |Δp| ≤ **0.01**.
- Also reported: the raw samples against the clip EDF's samples, and how many clips the dropout rule would remove.

**Decision.**

- If K passes, the scan uses mode K.
- If K fails and E passes, the scan uses mode E, which decodes every frame sequentially, per file, on CPU.
- If K and E both fail, or R fails, the work **stops**. The result is reported, and an amendment is needed.

The decision goes into `stepA/decision.json` before any raw window is scanned.

## 8. Crop boxes for RN201, RN203, RN243

Boxes are native-resolution, even-pixel crops of the 800×600 frame (the `crop_clips.py` grid). The first-named
animal is on the left.

**Candidates from the room conventions.**

- RN201: **(416,180,384,360)**. RN199's (0,180,400,360) pairs with this right box on RN245-RN216, the other room-C
  camera with that left box.
- RN203: **(0,0,480,480)**. RN204's (480,0,320,480) pairs with this left box on three room-C cameras.
- RN243: room D has no shared convention. The candidate keeps RN242's y and h, (·,264,·,216), and sets x and w
  to the right-hand cage walls in the median image.

**Check.** Frames are sampled on 3 fixed days: the first, the middle and the last day folder that has video. Each
sample is one keyframe pair every 10 min over 24 h. From these, the per-pixel median image and a motion map
(consecutive-keyframe |Δ|) are made.

- A box is accepted when, on each day, ≥ 90% of the motion energy outside the partner's box lies inside it, and it
  does not overlap the partner's box.
- Otherwise it is grown to the smallest even-pixel box that passes, and the change is recorded.
- PNGs of the median image and the motion map, with both boxes drawn, go to `boxes/`, along with `boxes.json`
  (holding its sha256).
- The boxes are frozen before any network sees a raw window.

The boxes of the labelled animals are used as they are.

## 9. Gallery (for human review; not part of the numbers above)

- **Selection.**
  - For each no-label animal, the **10 highest-q episodes**. They are built from consensus windows: windows flagged
    at g5 in ≥ 50% of the animal's 45 configurations, merged by the §6 rule. They are ranked by the maximum of q̄,
    the configuration mean.
  - Also each partner's 10 highest-q̄ seizure-free episodes, so that both sides get the same review.
  - Also 5 blinded controls per no-label animal: windows flagged by no configuration, drawn with a fixed seed.
- **Each item.** Items are exported with exact-frame decoding. They cover the episode ±60 s, capped at 6 min around
  the window with the highest q̄. Each item has:
  - the cropped clip at native resolution (libx264, crf 18);
  - a full-frame clip at 400×300, to judge cross-talk;
  - an EEG PNG with the 1-kHz trace, the 125-Hz trace, the per-6-s P_E(sz), the per-window q̄ and video's grade;
  - `item.json`.
- **Blinding.** Items get random IDs, and the key goes in `gallery/key.json`. The user labels each item as
  seizure (with stage), not seizure or unsure before the key is opened.

## 10. Order of work, budget, resumability

- **Code.** New code goes in `rawdata/scan.py` (sub-commands for the stages below) and `rawdata/submit_scan.sh`.
  - Every write goes through `out_path()` under `$EEG_ROOT/output/ttg_scan/`, logs included. The raw tree, `data/`,
    `data_full/`, `cache_frames/` and every other output directory are opened read-only.
  - Each camera file is read once from the first animal folder, after its name and size are checked against the
    partner's copy.
  - No change to `grader/` is planned. If one is needed, it is an option whose default is bit-identical, proven by
    `joint_gate.py --regression` (46/46) and by reproducing the protocol-B grid.
- **Stages.**
  - **S0 (CPU).** Window tables. Gate fits and the reproduction gate. The OOF reference.
  - **S1 (CPU).** Boxes (§8).
  - **S2.** Step A (§7).
  - **S3 pilot.** The first day folder with ≥ 20 h scanned for both RN203 and RN204, run end to end. Only its
    throughput and integrity are inspected, not its rates.
  - **S4.** Decoding (CPU) and the clock check, then inference (GPU).
  - **S5 (CPU).** Decisions and summaries.
  - **S6 (CPU).** Gallery.
- **SLURM.**
  - `SLURM_*` is stripped first.
  - CPU jobs run with `-A mech-ai-scavenger -q scavenger -p scavenger`.
  - GPU jobs use the same flags plus `--gres=gpu:1 --constraint="a100|h200|l40s" --requeue`.
  - The eeg env imports slowly (8–47 min), so each job handles many units.
- **GPU ≤ 6.0 h in total.** This includes step A, the pilot and preempted time.
  - Each GPU job appends its start and end to `budget/gpu.jsonl`, which is reconciled with `sacct`.
  - A job will not start when the ledger plus its `--time` would exceed 6 h.
  - The expected load is about 4.5 M X3D-M forwards: about 85 k windows × 15 networks for each no-label animal,
    plus (58 + 84 + 88) k windows × 3 for the partners. The EEG load is negligible. The scan therefore needs about
    250 forwards/s.
  - **If the pilot projects a total above 6 h, the work stops and the projection is reported.** There is no ad-hoc
    reduction. If the ledger reaches 6 h during the scan, the work also stops, and what is finished is reported.
  - CPU hours are recorded, with no cap.
- **Disk.** The filesystem is 99% full, with 2.7 TB free.
  - Frame shards are raw uint8 arrays, 2.4 MB per window. They are transient, and at most **200 GB** exist at once:
    decoding pauses above that.
  - A shard is deleted only after the predictions of all its networks are written and their row counts checked.
- **Resumability.**
  - The units are the camera file (decoding) and the (animal, day) shard (inference). Each is written atomically,
    with a done-marker that holds its input hashes, and a requeued job skips finished units.
  - On SIGTERM a job finishes its batch, then exits 3.

## 11. What will and will not be claimed

- **Wording.** The numbers are flag rates at a fixed operating point, not seizure counts. No flagged window is
  called a seizure or a false alarm without the gallery review.
  - "False-alarm rate" is used only if the user confirms that RN201, RN203 and RN243 are controls, and then it is
    "the flag rate on a control animal".
  - The partners' seizure-free rate is an **upper bound** on their false-alarm rate, because the annotations may be
    incomplete.
- **Inference.** There are 3 camera pairs, so no p-values and no cohort inference are given, and the results are
  described pair by pair. Any analysis chosen after scan results are seen is labelled post hoc.
- **Sensitivity.** About 99% of the partners' events (2,236 of 2,265) were labelled clips scored in protocol B.
  Sensitivity therefore mostly re-measures OOF detection in a new window geometry. The never-clipped events (29)
  are the new part.
- **Limits stated in advance.**
  - The no-label boxes are not validated against clips (§8).
  - The mode-K input is validated only on labelled animals.
  - The gate was fitted at about 50% seizure prevalence, and continuous data are far rarer in seizures. The
    operating point is kept anyway, and a prior-corrected gate would be a separate pre-registration.

## 12. Deviations (filled in during the work)

None yet.
