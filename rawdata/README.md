# rawdata/ — read-only audit of the raw DSI/Noldus tree

Checks every file under `/work/mech-ai/alloydas/EEG/Data` for integrity and coverage:
23 animal folders, about 13,500 files, 2.56 TB. It reads headers, sidecars and three
frames per video. It never decodes a whole video and never writes under the raw tree.

| file | what it does |
|---|---|
| `audit.py` | `list`, `unit --animal A` / `unit --index i` (one animal folder, resumable per file), `merge` |
| `submit_audit.sh` | SLURM array on the CPU scavenger QoS, one task per animal folder, then `merge` |

Everything is written to `$EEG_ROOT/output/raw_audit/` (the default `EEG_ROOT` is
`/work/mech-ai-scratch/alloy/EEG`). `out_path()` refuses any path outside that directory.

| output | contents |
|---|---|
| `files.csv` | one row per file: size, status (`ok`/`warn`/`error`), problem codes, and the fields for its kind |
| `events.csv` | one row per xlsx event: label, duration, start/end, the video that covers it and the seek into it, same-day EDF coverage, the matching labelled clip dir |
| `duplicates.csv` | files stored more than once (name + size + sha1 of the first and last 1 MiB), content-only matches, and the bytes de-duplication would save |
| `per_animal.json` | per-animal totals, per-day-folder and per-calendar-day hours, gaps between consecutive video files, EDF gaps, cameras and partners, bad files, partner asymmetry, and the RN197 `seizure_clips/` breakdown |
| `audit_summary.txt` | the human-readable report, including a "not checked" list |
| `units/<animal>.files.jsonl`, `units/<animal>.json` | the per-file cache (used to resume) and the finished unit |

## Run

```bash
bash rawdata/submit_audit.sh              # all 23 folders, waits, then merges
ONLY=3,7 bash rawdata/submit_audit.sh     # re-run selected array indices (audit.py list)
$PY rawdata/audit.py merge                # re-merge only; this is cheap
```

`$PY` is `/work/mech-ai-scratch/alloy/.conda/envs/eeg/bin/python`. `audit.py` uses
**only the standard library**. Video goes through `ffprobe`/`ffmpeg` in
`/work/mech-ai/alloy/miniconda3/bin`, and xlsx through a small zipfile/ElementTree
reader, which agreed with openpyxl on 40 of 40 sampled files. The first run imported
cv2 and PyAV inside the unit jobs. Their libraries page in at about 2.6 MB/s from the
conda env's filesystem (`/work/mech-ai-scratch`, 99% full), and 23 tasks sat for over
15 minutes without finishing a single mp4. `AUDIT_CV2=1` turns the cv2 open / frame-count
check back on.

A requeued task skips every file already in its jsonl cache. The cache key is
`relpath|size|mtime|FILE_VERSION`. Bump `FILE_VERSION` only when the per-file probe
changes. `AUDIT_VERSION` covers the per-animal and merge logic. After a change there,
`audit.py merge --refinalize` rebuilds every unit from the caches without re-probing a
file.

## What each file gets

- **mp4**
  - the top-level box walk: whether `moov` and `mdat` are present, and whether a box runs past EOF
  - the Noldus trailer, 320 bytes after `moov`, which holds the start time to the millisecond in both local time and UTC
  - ffprobe: codec, WxH, fps, `nb_frames` and duration
  - `ffmpeg … -vf showinfo -f null -` decodes three points: the first frame, the frame at
    mid-file, and every frame from the keyframe at or before end−10 s through EOF. It
    records the pts, luma mean and SD of each, the last decodable frame's gap to the
    declared duration, and any decoder error lines
  - sha1 of the head and tail
  - every XML sidecar field
- **edf**
  - the header only: start, nrec, record duration, labels, samples per record
  - expected size = header + nrec·Σsamples·2, and a flag on truncation or `nrec = -1`
  - the zero fraction and ECG spread of the first and last complete record
- **xlsx**: every event row, parsed from the header names, with the unit row skipped.
- **RN197 `seizure_clips/`**
  - `info.txt`, `_clip_log.csv`, `video.mp4` and `eeg.edf`, parsed and cross-checked
    against the raw video's XML start and against the labelled tree (by directory name
    only)
  - zero-byte files

## Timing conventions (measured on all 5,200 videos; numbers are in audit_summary.txt)

- The `<camera>.<YYYYMMDDhhmmss>.mp4` in a file name is the **UTC time the file was
  closed**, not its start. File-name time − (XML UTC start + stream duration) has a
  median of −0.27 s, and p5/p95 of −5.3/+0.4 s. The residual grows with file length
  (stream clock vs wall clock). 24 unique files break this rule by more than 30 s; they
  are listed below.
- `DSI_utc_start_time` is the first frame in whole UTC seconds. `DSI_sync_offset` is the
  sub-second remainder in 100 ns units (0.012–0.998 s). It equals the Noldus trailer's
  UTC milliseconds × 10⁴ in 5,198 of the 5,198 files that have a trailer. True start =
  `utc + sync_offset/1e7`.
- `DSI_local_start_time` is America/Chicago wall time: UTC−5 before 2023-11-05 (2,103
  files) and UTC−6 after (3,097). It is exactly Central(UTC) in 5,099 files and 1 s
  earlier in 101.
- Every file has its own XML start; no two files of a camera share one. Within a day
  folder, consecutive files are separated by 47.5–60.2 s (median 59.7 s). All cameras in
  a room roll over together: 1402C and 1402D in `Dsi_VideoFullPath`.
- A day folder is one acquisition session. The EDF header start, in local wall time,
  comes 18.5 s before the first video (median 18.46 s, p5–p95 17.9–18.9 s, n = 1,049).
  The exceptions are room 1402C on 10-19-2023 (+198.5 s) and on 11-17-2023 (−748 s: the
  video starts before the EDF).
- **Frames do not fill the wall-clock span** in 24 unique files (47 stored copies):
  - on 2023-10-13 in room 1402D, 7.14 h of frames cover 10.06 h
  - on 2023-10-25 in room 1402C, 22 s of frames cover 4.3 h
  - on 2023-11-03 and 2023-12-05 in room 1402C, 60 s is missing from each file

  Inside these files, `start + pts` is not clock time.

---

# `align_clips.py` — labelled clips mapped onto the raw recordings

Maps the labelled clip dirs (`$EEG_ROOT/data*/Data_<A>_cropped/<session>/<clip>/`) of the 20
labelled animals onto the raw mp4 and EDF files, read-only. Outputs go only to
`$EEG_ROOT/output/raw_audit/align/`. `out_path()` refuses any other path, including the raw tree,
`data/` and `data_full/`.

| step | command | output |
|---|---|---|
| 1 | `align_clips.py inventory` parses every clip dir name and `info.txt`, replicates `train_pooled.discover()` (ROOTS and STAGE are read from `train_pooled.py` with `ast`, so torch is not imported), compares the result with `cache_frames/f32s224/index.json`, and maps every clip onto its raw mp4 and EDF from recorded times alone | `clips.csv`, `inventory.json` |
| 2 | `align_clips.py sample` draws a stratified sample: 4 seizure and 2 non-seizure clips per animal, on evenly spaced sessions, with at least one severe clip when the animal has one, plus a `defect` stratum of clips cut from timing-defect raw files | `sample.csv`, `sample_notes.json` |
| 3b | `align_clips.py scan [name …]` correlates EDF Activity with video motion at points along the timing-defect files and the DST session (the list is in `SCANS`) | `scans/<name>.json` |
| 3 | `submit_align.sh` runs `align_clips.py match --index i` as a scavenger array with one task per animal (resumable per clip), then `merge` | `units/<A>.jsonl`, `matches.csv`, `crop_boxes.csv`, `per_animal.csv`, `align_summary.txt`, `align.json` |

What `match` does for each sampled clip:

- **Crop box.** It takes three clip frames (at ¼, ½ and the end of the clip) and the raw frames at the
  cutter's predicted seek plus the same times. It computes a normalised cross-correlation map
  (FFT with integral images) for each scale in {0.5 … 1.25}, averages the maps over the three pairs,
  and takes the argmax as the box `(x, y, w, h)` in raw 800×600 pixels.
- **Frame offset.** It decodes every clip frame and every raw frame within ±30 s of the predicted
  seek, cropped to the box and downsampled 4×. For each candidate offset it scores the mean
  per-frame NCC of the raw intensity, of the background-removed frames (per-sequence temporal
  median subtracted) and of the frame differences. Offsets are whole frames. It also checks drift
  by aligning each third of the clip separately, and computes a full-resolution MAE on one aligned
  frame.
- **Wide check.** The same frame-offset search is repeated over ±300 s at 8× downsampling. That
  decode also supplies the per-frame motion used for the EDF Activity comparison.
- **Keyframe model.** `ffprobe` lists the packets around the seek. `d_key_frames` is the matched
  first frame's index minus that of the keyframe at or before the seek. It is 0 when the clip was
  cut with `-c copy`, which is the cutter's default.
- **EEG.** It reads the clip's `eeg.edf` 1 kHz channel (label `ECG`, otherwise `EEG`) and the raw
  EDF's same channel at the implied offset `clip start − EDF header start`, ±30 s, and runs a 1-D
  normalised cross-correlation. It reports the lag in samples, the peak NCC and the regression
  slope. `eeg_minus_video_s` is the EDF clock of EEG sample 0 minus the video clock of video frame 0,
  where the video clock is the XML UTC start + `sync_offset` + pts.
- **Activity vs motion.** It correlates the EDF `Activity` channel (1 Hz) with the per-second
  video motion in the animal's crop over ±300 s, at lags of −30 to +30 s. This is a coarse,
  independent check that the EDF clock and the video clock agree.

Python is the miniconda3 base interpreter (`/work/mech-ai/alloy/miniconda3/bin/python`: numpy
2.3, scipy 1.16), not the `eeg` env. On these nodes the `eeg` env takes about 7 minutes just to
import numpy (it pages in from `/work/mech-ai-scratch`). Nothing was installed.

### Strata

| stratum | what | why |
|---|---|---|
| `main` | 4 seizure clips (at least one Stage ≥ 4 when the animal has one) and 2 non-seizure clips per animal, on evenly spaced sessions | success rate, crop boxes and offsets |
| `defect` | up to 4 clips per timing-defect raw file, at the smallest, middle and largest seek | whether the missing wall time comes before the clip; Activity search ±4 h |
| `dst` | 2 seizure clips per animal and case after the 2023-11-05 fall-back | the cutter subtracted naive local times |
| `clock` | 1 seizure clip per animal on 10-19-2023 and 11-17-2023, where room 1402C's EDF starts 198.5 s and −748.4 s from the first video instead of 18.5 s. Room 1402D on the same days is the control. | whether the EDF clock or the video clock moves on those days |

### EDF clock across the 2023-11-05 fall-back

The EDF clock is naive local wall time. The room-D session `11-03-2023 to 11-05-2023` is one EDF that
runs 72.17 h, while its real span (10:22:15 CDT to 10:32:35 CST) is 73.17 h, so one hour is not in the
file. On RN224, EDF Activity against video motion (EDF time mapped to UTC as start + elapsed) peaks at
−1 s at 25.3 h and 37.2 h into the EDF, before the change, and at +3599 s at 41.9 h and 59.8 h, after
it. `align_clips.py` therefore maps EDF time to UTC as `local_to_utc(start + elapsed)`.

What this means for the cutter's naive arithmetic:

- The EEG cut is at the naive label time, in every case.
- If the video file started on the same side of the change as the label, the video cut is at the
  same moment.
- If the video file started before the change and the label is after it, the video cut is 1 h early
  (`video_seek_error_s` = +3600). This is 30 labelled clips.

### Timing-defect files (from `align_clips.py scan`)

Each scan correlates EDF Activity with video motion in 20-minute windows along one file.

- `…20231014004100.mp4` (room D, 2023-10-13), measured on RN242: the lag is about −10,524 s from the
  first window (pts 100 s) through pts 23,000 s. The frames therefore begin about 175.4 min after the
  XML start, and every labelled clip cut from this file shows video about 175 min after its label and
  its EEG.
- `…20231103122100.mp4` and `…20231205222300.mp4` (room C), measured on RN235's camera: the lag is −1 s
  early in the file and −61 s late in it. On 11-03 the last clean window ends at pts 11,700 s and
  the first shifted window starts at 12,000 s. On 12-05 the last clean window ends at 13,200 s, the
  13,500 s window is mixed, and the window from 15,000 s is shifted. Clips cut after that point show
  video 61 s late. Only this camera was scanned; that the other room-C cameras lose the same minute
  is inferred from their shared rollover, and is supported by one RN199 clip (−61 s at pts 16,005 s)
  and one RN238 clip (−1 s at pts 6,849 s).

### How a clip's frames map to the raw clock (measured on 120 of 120 main-stratum clips)

- The cutter's seek is `clip start − Central(int(DSI_utc_start_time))`. It matches info.txt's
  `Video seek` to within 0.02 s in 12,344 of 12,344 non-seizure clips.
- Clip frame 0 is the raw keyframe at or before that seek (GOP 32 frames = 2.133 s; the cut used
  `-c copy`). In 2 of 120 clips it is one GOP earlier, where the seek fell 4 ms after a keyframe.
- The raw clock time of clip frame j is `DSI_utc_start_time + DSI_sync_offset/1e7 + pts(keyframe) +
  j/15`. On that clock, frame 0 sits −1.8 to +0.7 s (median −0.56 s) from info.txt's `Clip start`.
- The clip EEG starts exactly at `Clip start` on the EDF clock (lag 0 samples in 165 of 165 matchable
  clips). Within a clip, EEG minus video is therefore +0.68 s (median; range −0.66 to +1.81 s), and
  the EDF header's whole-second start adds up to 1 s of unknown to that.

# `annotations.py` — the seizure annotations against the labelled clips

Read-only. Outputs go to `$EEG_ROOT/output/raw_audit/annotations/` (`out_path()` refuses anything
else); `summary.txt` there is the human-readable report. Standard library only: numpy and pandas
page in too slowly from the 99%-full conda filesystem, and the first scavenger run stalled on the
import.

| sub-command | what it does |
|---|---|
| `labelled [--animal A]` | lists `data/` and `data_full/` clip dirs and parses every `data_full` info.txt |
| `xlsxzip` | zip-member time (the export time) and sheet dimension of all 1,083 xlsx |
| `epochs [--animal A]` | the `rn243_scale_audit.py` line-length detector over the DSI epoch export, for all 23 animals |
| `onset [--animal A]` | 1-s EEG line length from −60 to +90 s around up to 40 annotated onsets per animal (EDF records only) |
| `detprobe [--animal A]` | ictal and post-ictal line length around unannotated detections and annotated seizures |
| `clipprobe`, `gop` | ffprobe duration of 15 data_full clips per animal; keyframe spacing of 4 raw files |
| `report` | joins the caches into CSV/JSON outputs and `summary.txt` |
| `all` | every unit, then `detprobe`, then `report` |

`bash rawdata/submit_annotations.sh` runs `all` on one scavenger node (`CMD=report` for the report
only). Each unit writes `cache/<unit>_<animal>.json` and is skipped when that file exists.

# `recut_clips.py` — re-cut the labelled clips whose video is from the wrong time

Measures, for every labelled clip that could be affected, where its video really is. Clips whose video is
from the wrong time are re-cut from the raw recording with the parent pipeline's own two ffmpeg steps.
Everything is written to `$EEG_ROOT/output/ttg_recut/`, and `out_path()` refuses any other path. Nothing
under the raw tree, `data/`, `data_full/`, `cache_frames/` or another run directory is touched.
`bash rawdata/submit_recut.sh motion|analyze|recut|verify|all` runs the stages on the CPU scavenger QoS.

| stage | what | output |
|---|---|---|
| `plan` | candidates: every `discover()` clip cut from a timing-defect file (10-13 room D, 11-03 and 12-05 room C), or from a file spanning the 2023-11-05 fall-back, or labelled 01:00–02:00 on 11-05 (172). Also builds the decode jobs and picks 5 unaffected reproduction controls. | `plan/` |
| `motion` | one full decode per camera file (19): the mean \|frame difference\| per frame in every known animal box, pts from the integer pts × stream time base | `motion/*.npz` |
| `activity` | the 1 Hz EDF `Activity` channel, and per-record ECG flatness (telemetry gaps) | `activity/*.npz` |
| `measure` | per clip: Activity over the label span ±30 min against box motion over ±4 h (masked NCC at 1 s), refined ±30 s and to 0.1 s; the partner animal on the same camera over the same window; clip-sized scan windows every 600 s along every decoded file | `measure/clips.csv`, `measure/profiles.csv` |
| `calibrate` | the same windows scored against the cause model, to choose the motion transform and the "strong" thresholds | `calibrate/*.csv` |
| `decide` | aligned / misaligned / no_video / unresolved / flagged, against the cause model | `measure/decisions.csv`, `exclude.csv` |
| `recut` | `ffmpeg -ss <seek> -i raw -t <dur> -c copy -avoid_negative_ts make_zero`, then `crop_clips.py`'s re-encode (`-vf crop=trunc(iw*wf/2)*2:… -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p -an -threads 2`) | `data_full/<clip rel>/video.mp4`, `manifest.csv` |
| `repro` | the same two steps at the ORIGINAL seek for 5 unaffected clips | `repro/` |
| `verify` | decodes; frame count; NCC of every frame against the raw frames at the corrected time; Activity vs motion at the new and at the original location | `verify/verify.csv`, `verify/repro.csv` |
| `report` | | `summary.txt`, `summary.json` |

**The offset convention.** The offset is the raw pts that shows the labelled instant, minus the pts the cutter
used, on the cutter's nominal timeline `int(DSI_utc_start_time) + pts`. It is the lag of `verify/v_activity.py`.
Expected values: 10-13 room D about −10,523 s; clock step −59.8 to −60.8 s; DST +3600 s (the video was cut 1 h
early, so the right frames are 1 h later).

**Method notes (all measured here):**
- Motion is rank-transformed over the searched ±4 h series. The values are 0.01–0.3 grey levels, so
  `log1p` is linear there, and one person or light change that meets one Activity count dominated Pearson r:
  14% of strong scan windows were wrong with `log1p`, and none with `rank`.
- A measurement is strong when r ≥ 0.10, z ≥ 4 over the lag curve, it leads the next peak (> 15 s away) by
  ≥ 0.04, and it is sharp: peak minus the best r 4–15 s away ≥ 0.02. Of 842 scan windows with a model
  truth, 384 are strong and none contradicts the cause model (`calibrate/rank_hp0_w3600.csv`).
- An aligned clip does not measure 0. The cutter ignores `DSI_sync_offset`, and Activity lags motion by
  λ = 0.74 s (median of 28 clip controls and 154 aligned scan windows). The offset is therefore
  lag − (−sync − λ), in the cutter's own convention.
- RN242-RN243 and RN229 stamp pts slow by 2.0e-4 (frames are real time at 15 fps: `nb_frames/15` equals the
  wall span), so the offset drifts up to +5.4 s along a 7.5 h file.
  - The expected value is therefore an interval, from no drift to full drift.
  - A DST clip late in its file has its true frames in the camera's NEXT file, where the drift starts again.
    The nominal timeline jumps by 5.4 s at that boundary, so windows are split there.
- The 3 clips labelled in the repeated hour measure +3600 s. So the EDF's naive 01:00–02:00 on 11-05 is the
  CST hour, and the EDF dropped the CDT one.
- Weak clips take their partner's strong measurement, or the family's consensus: ≥ 3 measurements on the
  same file and side within ±1 h, within ±1.5 s of each other, from ≥ 2 independent units or with a strong
  member, and inside the model interval. As a last resort they take the model midpoint, and only when the
  family's strong clips on ≥ 2 cameras agree with the model to within 2 s and the interval is ≤ 6 s wide.
  Otherwise they are unresolved.
- On 10-13 no file of either camera covers local 09:34–12:32. Clips labelled there have no video at all
  (`no_video`).

### Results (2026-09-29, `ttg_recut/summary.txt`)

- **Candidates: 172.**
  - 81 aligned (the before-change and before-step controls).
  - 81 misaligned and re-cut.
  - 5 `no_video`.
  - 5 unresolved.
  - 0 flagged.
- **Wrong-time video: 86 clips** (42 non-seizure, 32 Stage 2, 10 Stage 3, 2 Stage 4): 81 re-cut plus 5 never recorded.
- **Re-cut clips by class:** 39 / 30 / 10 / 2.
- **Offsets used (s):**
  - DST +3600.0, range +3599.7 to +3605.2 (n = 27).
  - Repeated hour +3600.2 (n = 3).
  - 10-13 −10,523.3, range −10,524.5 to −10,523.0 (n = 5).
  - Clock step −60.4, range −61.0 to −59.7 (n = 46).
- **Source of the offset:** 35 measured, 7 partner, 36 family consensus, 3 model midpoint.
  - 9 re-cuts come from the camera's next file.
  - One (RN229 `seizure_41`) runs 8.7 s past its file's end and is truncated there, as the parent cutter
    would truncate it.
- **Against the brief's 90:** all 81 re-cuts are among them, and 5 are `no_video`. 4 are unresolved: 12-05
  clock-step clips on RN235 (whose partner RN222 has no EEG that day) and on RN238 (whose partner RN237
  measures −66.4 s, against the model's −60.5 s).
  - One more unresolved clip (RN238 `seizure_02`, 12-05, before the step) is expected aligned but was not confirmed.
- **Verification:** all 81 decode.
  - Each starts at the keyframe at or before the corrected seek, and its frames match the raw frames there
    (mean NCC ≥ 0.9997).
  - Frame counts equal the copy-cut expectation.
- **Activity check, ±15 min window:**
  - The best lag lies within ±3 s of the method baseline for 70 of 81 at the new location, and for 2 of 81 at
    the original.
  - r at lag 0 is higher at the new location in 75 of 79.
  - 52 pass (23 of them with z ≥ 4), and 28 are weak: too little Activity, a median of 13 counts against 86.
  - None has a clear peak elsewhere.
  - The clip-only check (60–120 one-second bins) is indeterminate for 78 of 81.
- **Reproduction:** the same two ffmpeg steps at the original seek reproduce 5 unaffected `data_full` clips
  frame for frame (MAE 0.0). The files differ only in container and encoder tags: ffmpeg 4.3 and x264 core 161
  here, against 4.4 and 163 there.

# `rescore_recut.py` — the EEG-gated grader re-scored with the re-cut clips (no retraining)

Stages: `cache`, `infer`, `patch`, `grid`, `compare`, `analyse` (the module docstring has the details).
Everything is written under `$EEG_ROOT/output/ttg_recut/`. No `grader/` script was changed: `joint_gate.py` runs
in-process with two logged adaptations, the `--out` prefix and the per-fold clip-set check minus the excluded
clips. `bash rawdata/submit_rescore.sh prep|grid|analyse` runs the same stages on the CPU scavenger QoS; its
header lists what was actually run.

**Steps and checks (2026-09-29):**

- **Cache.** `cache_f16s224/` holds the 81 re-cut clips, built with the f16s224 rule (`ttg_common.decode_linspace`).
  - All 81 rows are bit-identical to `train_classifier.load_clip`.
  - 3 unaffected clips rebuilt from `data_full/` are bit-identical to `cache_frames/f16s224`.
  - So are the 5 reproduction controls from `ttg_recut/repro/`. That tests the whole re-cut chain.
- **Inference.** 30 runs (`ttg_vsubj/{x3dfix,x3dbug}_dual_s{1,2,3}_fold{0-4}`) were run on CPU in fp32.
  - Each `last.pt` was checked to be the state after epoch 12 of 12.
  - The run key matches `config.json`, and the history and dump epochs are 12.
  - **Equivalence** against the stored `val_ep12.npz`, which was written under CUDA fp16 autocast:
    - 150 unaffected control rows and 486 re-cut rows re-predicted on their ORIGINAL frames.
    - Argmax is identical on 1272 of 1272 (row, head) pairs.
    - Mean |Δp| is 2–4e-4. The maximum is 0.0205, and the largest |Δ log p| (p > 1e-3) is 0.16.
  - The tolerance |Δp| ≤ 0.02, set before the check, was exceeded on 2 rows (0.0205 and 0.0200). The patch
    proceeded on the argmax criterion under `--accept_precision_noise`. This is recorded in
    `infer/equivalence.json` and in every `patch.json`.
  - CPU fp16 autocast on one run is no closer (max 4e-3 against fp32's 2.5e-3 there).
  - **Fragile predictions.** 22 of the 972 new (row, head) predictions have a top-2 margin inside 2 × the
    measured noise. 16 of those changed argmax, so they could land either way under GPU fp16.
- **Patch.** The patched dumps are in `vsubj_patched/<run>/val_ep12.npz`:
  - 486 rows replaced (81 clips × 6 runs). argmax changed on 249 (g3) and 256 (g5) of them.
  - 66 rows removed (11 excluded clips × 6).
  - Every other field is kept.
  - `vsubj_exclonly/` holds the same dumps with the removals only.
  - The patched dumps were written with numpy 2.3.4, since `patch` needs numpy only, and joint_gate read them
    with 1.26.4.
- **Wrapper faithfulness.** `grid/repro_B_{x3dfix,x3dbug}_grid3x3` is the stored dumps run through the wrapper.
  Against `ttg_eeg_gate/seeds/B_*_grid3x3`, it differs in 0 fields of `results.json` and `gate.json` (outside
  created / argv / secs), and `results.txt` is identical after the timestamp line.
- **Grids.** They are in `grid/B_{x3dfix,x3dbug}_grid3x3` (patched) and `grid/exclonly_B_*`. The per-pair
  numbers in `rescore/summary.json` equal the grids' exactly.

**Results.** 9 single-video × single-EEG pairs. Means are given as before → after.
- "Before" is the stored dumps: 24,452 clips.
- "After" is the patched dumps: 24,441 clips, with the 11 excluded clips dropped and counted.
- The exclusions alone move every number by ≤ 0.0003.

| | x3dfix g3 | x3dfix g5 | x3dbug g3 | x3dbug g5 |
|---|---|---|---|---|
| video alone | 0.7082 → 0.7086 | 0.5034 → 0.5032 | 0.7175 → 0.7177 | 0.4469 → 0.4463 |
| EGRG | 0.7488 → 0.7485 | 0.5471 → 0.5452 | 0.7578 → 0.7575 | 0.5032 → 0.5013 |
| EGRG − video | +0.0406 → +0.0399 | +0.0437 → +0.0420 | +0.0403 → +0.0398 | +0.0563 → +0.0550 |
| EGRG Stage-2 hits (g5) / mild hits (g3) | 9989.4 → 9976.8 of 10852 → 10846 | 361.1 → 347.9 of 1457 → 1452 | 9968.8 → 9959.8 | 460.7 → 446.1 |
| severe hits, video / EGRG (of 1257) | 406.7 / 405.7 → 405.3 / 405.6 | 401.0 / 401.0 → 399.7 / 400.9 | 444.0 / 446.0 → 442.3 / 446.0 | 425.0 / 426.7 → 423.0 / 426.7 |
| within-session severe AUROC (video's) | 0.7415 → 0.7358 | 0.7403 → 0.7344 | 0.7548 → 0.7525 | 0.7515 → 0.7488 |

- **Predictions that changed class (mean per pair).**
  - Video: 40–43 of the 81 re-cut clips, and 0 others.
  - EGRG on the re-cut clips: 6.8 (x3dfix g3), 20.1 (x3dfix g5), 11.9 (x3dbug g3), 28.4 (x3dbug g5).
  - EGRG elsewhere: 2–4 clips, through the gate refit (about 2 of them from the exclusions alone).
- **The corrected video sees the seizures.** Of the 42 re-cut seizure clips, a majority of video seeds call
  seizure on 7 before and 35 after (x3dfix; x3dbug: 8 → 33).
- **EGRG gets the corrected clips *less* often right.** Per pair on the 81 clips: g3 77.6 → 71.9 and g5
  58.8 → 47.0 (x3dfix). There are two reasons:
  - **Stage 2 is overgraded.** On the 30 Stage-2 clips, the wrong-time video looked quiet, and EGRG gated them
    in on the EEG and took a within-seizure split that fell on S2 (13.9 per pair). The real video is graded S3
    or S4 (23.7 + 4.0 per pair, S2 2.3). Part of EGRG's Stage-2 hits therefore come from quiet video under an
    EEG seizure.
  - **5 re-cut seizure clips are EEG-disputed.** The EEG detector at the label says no seizure: 3-seed P(sz)
    0.02–0.29, against a base rate of 556 / 12,109 seizure clips below 0.5. The ORIGINAL video showed a seizure
    in 6 of 6 runs, and the re-cut video shows none in 6 of 6.
    - They are RN229 `seizure_39` and `seizure_42` (S4, DST), RN210 `seizure_22` and `seizure_23` (S3, DST),
      and RN213 `seizure_41` (S3, clock step).
    - Two are demonstrably duplicate labels:
      - RN229 `seizure_42`'s original window is `seizure_41`'s corrected window (5 s apart, the same footage).
      - RN213 `seizure_41`'s original window is `seizure_42`'s corrected window (1.8 s apart; the labels are
        59 s apart, the size of the clock step).
    - For these 5 the premise "label and EEG right, video wrong" fails.
    - Removing them from before and after (the `-D` rows, a sensitivity analysis chosen on the scored clips)
      leaves the result unchanged: x3dfix EGRG g5 0.5472 → 0.5453.
- **Headline ensemble rows (grid, animal bootstrap).**
  - x3dfix g3: EGRG − video +0.0366 [+0.0219, +0.0546] → +0.0358 [+0.0213, +0.0535].
  - x3dfix g5: +0.0474 → +0.0452.

**Retraining.**
- **Exposure.** Every run shares split_subjects(seed 49) folds, so each fold's training set holds the same
  affected clips whatever the seed or recipe.
  - Wrong-time clips in training: 61–82 per fold (70 / 69 / 62 / 61 / 82). That is 0.3–0.4% of the ~17–21k
    training clips.
  - By class that is 30–39 non-seizure (0.33%), 22–32 Stage 2 (1.8–3.5% of Stage-2 training), 6–9 Stage 3
    (0.1%) and 0–2 Stage 4 (0.2%).
  - Each fold also holds 2–5 unresolved clips and 0–1 mislabelled negative.
- **Effect.** It is not measured. Expected to be below seed noise: video 5-class sd 0.010 across seeds, while
  the whole val-side correction moved EGRG by ≤ 0.002.
- **Not recommended now.** Fix the labels first (the 5 EEG-disputed clips, including 2 duplicates across clock
  changes; the mislabelled negative; the 5 unresolved). Retrain once, together with larger data changes such as
  the 328 unclipped annotated seizures.
