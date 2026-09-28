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
