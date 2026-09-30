#!/usr/bin/env python3
"""Cut the annotated Stage 2-5 seizures that were never clipped, as the parent pipeline would have, plus a paired
non-seizure clip for each new seizure that has none.

Writes ONLY under $EEG_ROOT/output/ttg_newclips/ (out_path() refuses anything else). The raw tree, data/,
data_full/, cache_frames/ and every other output/ directory are opened read-only. Nothing is committed.

Inputs
------
  raw_audit/annotations/candidates_unclipped_events.csv   the 328 unclipped Stage 2-5 events in labelled animals
  raw_audit/files.csv (via recut_clips.Raw)                 camera files, XML starts, EDFs
  the raw session folders (xlsx, mp4 + XML, EDF)            read exactly as the parent cutters read them
  $EEG_ROOT/data_full/Data_<A>_cropped/<session>/            existing clips: names, info.txt (reproduction checks)
  ttg_eeg/subject/tcn_bin_fold{f}_s{s}/final.pt              the EEG detector (label sanity, CPU)

The parent pipeline (EEG repo, read-only), reproduced here
---------------------------------------------------------
  sh/_cut_worker.sh runs, per raw session folder, with a 90-min timeout each:
    cut_seizure_clips.py     --pre 10 --post 10 --mode copy
    cut_non_seizure_clips.py --pre 10 --post 10 --safety 30 --seed 42 --mode copy
  then crop_clips.py (per-animal box, libx264 crf 18 preset fast, yuv420p, -an), which copies info.txt and eeg.edf.
  * seizure clip: onset - 10 s .. end + 10 s. The video file is the session file whose [Central(int(DSI_utc_start)),
    + cv2 frames / cv2 fps) holds the onset; seek = max(0, onset - start - 10); duration min(clip, file - seek).
    Name seizure_<idx:02d>_<label>_<onset %Y%m%d_%H%M%S>, idx = xlsx data-row index (row + 1).
  * non-seizure clip: for EVERY xlsx row in order (clip_n = 1, 2, ...), a window of the same length
    (10 + duration + 10 s) drawn uniformly from the free time of the seizure's own video (falling back to the other
    session videos), where free = outside every xlsx event widened by 40 s on each side; random.seed(42) once per
    session. Name clip_<n:02d>_vs_seizure_<idx:02d>_<label>_<onset>.
  * EEG: mne 1.12.1 read_raw_edf(preload=False).crop(clip start - EDF header start, clip end - EDF start)
    .load_data(), then edf_clip_writer.write_edf_clip (edfio 0.4.16, per-channel physical range, 1-s records,
    trimmed to whole seconds). Reproduced byte for byte (see `verify`).
  The seizure cutter was KILLED by the 90-min guard on the large sessions (MNE hanging on a multi-day EDF), which is
  why 314 of the events are the tails of those sessions; the non-seizure cutter died the same way in 3 sessions.
  Because the negative sampler is seeded, the negatives it never reached are the continuation of its random stream:
  `plan` replays each session from seed 42, checks the replay against every existing negative (name and info.txt,
  byte for byte), and takes the missing ones from the same stream.

Deviations from the parent (all recorded per clip in the manifest)
------------------------------------------------------------------
  * EDF channel labels: the source EDF has two channels labelled 'ECG [FIR-HP: 5Hz'; MNE de-duplicates them to
    'ECG [FIR-HP: 5Hz-0'/'-1' (18 chars) and the cutters' 16-char rename turned the second into 'ECG [FIR-HP: 5_0'.
    Here every channel keeps its source label. Everything else in eeg.edf is the parent's bytes.
  * video in another day folder (8 events of room D 10-12): the parent found no video in the xlsx's folder; the
    camera file in <day>(2) is used, with the parent's seek arithmetic on that file.
  * second xlsx in a folder (5 events): the parent read only xlsx_files[0]. These are numbered after the session's
    highest index (the parent's row + 1 would repeat an index in the session), and their negatives block the
    events of BOTH xlsx files (blocking only its own file is how RN242 clip_03_vs_seizure_04 landed on a seizure).
  * timing corrections (10-13 room D, 11-03/12-05 clock steps, DST, the .20231109055600 files) are applied where a
    clip falls in an affected span; none is expected here (see `plan` output) and every clip is measured anyway.

Sub-commands (resumable; interpreters: 'eeg' = the eeg conda env, 'base' = miniconda3 base)
------------------------------------------------------------------------------------------
  plan      eeg   parse xlsx (openpyxl), cv2 video intervals, MNE EDF headers; seizure specs; negative replay
                  with the reproduction check; names and collision check; timing causes; decode jobs; controls
  cut       base  ffmpeg copy cut + crop encode, info.txt, for every planned clip and the 5 controls
  eeg       eeg   eeg.edf for every planned clip and the 5 controls
  motion    base  one full decode per camera file: per-frame motion in every known box (recut_clips)
  activity  base  EDF Activity (1 Hz) per EDF needed (recut_clips)
  measure   base  Activity vs crop motion per clip (+ partner, + clip-sized scan windows along every file)
  decide    base  aligned / flagged / unmeasured per clip, against the timing model of its file
  recut     base  re-cut any clip whose measured offset needs a correction (none expected)
  sanity    eeg   EEG line-length ratio and the TCN detector per new clip, with reference distributions
  verify    base  decode, frame count, crop box, frames vs raw, EEG vs raw EDF, the 5 controls
  report    base  manifest.csv, summary.txt / summary.json, integration.txt
"""
import argparse
import csv
import datetime as dt
import glob
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter, OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import align_clips as AC  # noqa: E402  (stdlib at import: EDF header / reader, time helpers, decode)
import recut_clips as RC  # noqa: E402  (stdlib at import: Raw, motion, activity, measure_one, cut/crop commands)

RAW = AC.RAW
EEG_ROOT = AC.EEG_ROOT
AUDIT = AC.AUDIT
FULL = AC.FULL
DATA = AC.DATA
OUT = os.path.join(EEG_ROOT, "output", "ttg_newclips")
RC.OUT = OUT          # recut_clips' out_path (motion/, activity/, plan/jobs.json) now resolves under OUT only
CAND_CSV = os.path.join(AUDIT, "annotations", "candidates_unclipped_events.csv")
EVENTS_CSV = os.path.join(AUDIT, "events.csv")
ESUBJ = os.path.join(EEG_ROOT, "output", "ttg_eeg", "subject")
RECUT = os.path.join(EEG_ROOT, "output", "ttg_recut")
FFMPEG, FFPROBE = AC.FFMPEG, AC.FFPROBE
CENTRAL = RC.CENTRAL

# the parent's command-line settings (sh/_cut_worker.sh)
PRE_BUFFER_S, POST_BUFFER_S, SAFETY_S, NS_SEED = 10.0, 10.0, 30.0, 42
STAGE = {"Stage_2": 1, "Stage_3": 2, "Stage_4": 3, "Stage_5": 4}          # train_pooled.STAGE
LABELLED = ["RN197", "RN199", "RN204", "RN213", "RN216", "RN222", "RN235", "RN237", "RN238", "RN245",
            "RN208", "RN210", "RN215", "RN219", "RN223", "RN224", "RN227", "RN229", "RN242", "RN244"]  # ROOTS
LAMBDA_S = 0.739      # Activity lags motion (recut_clips `decide`, ttg_recut/measure/baseline.json); fixed, not refit
ALIGNED_TOL_S = 3.0
STAMP_13S = "20231109055600"   # room-C files ~13 s off the EDF clock (step 2, ttg_scan)
CONTROLS = [          # 5 existing clips re-made by this pipeline (video and EEG), in sessions that get new clips
    ("RN235", "11-30-2023 to 12-04-2023", "seizure_02_Stage_3_20231130_154740"),
    ("RN213", "11-30-2023 to 12-04-2023", None),        # the last complete negative before the kill (found in plan)
    ("RN197", "10-26-2023", None),                      # the last seizure clip before the tail
    ("RN199", "10-26-2023", None),                      # the last seizure clip before the tail
    ("RN242", "10-12-2023", "clip_01_vs_seizure_02_Stage_3_20231012_192747"),
]


# ============================================================================ guards and io
def _under(p, root):
    p, r = os.path.realpath(p), os.path.realpath(root)
    return p == r or p.startswith(r + os.sep)


def out_path(*parts):
    """Every write goes through here: under OUT only; never the raw tree, data/, data_full/, caches or other runs."""
    p = os.path.join(OUT, *parts)
    if not _under(p, OUT) or _under(p, RAW) or _under(p, DATA) or _under(p, FULL) or \
            _under(p, os.path.join(EEG_ROOT, "cache_frames")):
        raise SystemExit(f"refusing to write outside {OUT}: {p}")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def write_text(rel, text):
    p = out_path(rel)
    with open(p + ".tmp", "w") as fh:
        fh.write(text)
    os.replace(p + ".tmp", p)


def write_csv(rel, cols, rows):
    p = out_path(rel)
    with open(p + ".tmp", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: RC._cell(r.get(k)) for k in cols})
    os.replace(p + ".tmp", p)


def write_jsonl(rel, rows):
    p = out_path(rel)
    with open(p + ".tmp", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")
    os.replace(p + ".tmp", p)


def read_jsonl(rel):
    p = os.path.join(OUT, rel)
    if not os.path.exists(p):
        return []
    with open(p) as fh:
        return [json.loads(x) for x in fh if x.strip()]


read_csv, fnum, nsec, from_nsec = RC.read_csv, RC.fnum, RC.nsec, RC.from_nsec
PD = AC.parse_dt


def rel_of(animal, session, clip_dir):
    return f"Data_{animal}_cropped/{session}/{clip_dir}"


def key_of(animal, session, clip_dir):
    return f"data/{rel_of(animal, session, clip_dir)}/video.mp4"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(cmd, timeout=3600):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


# ============================================================================ the parent's session reader
# Copied from cut_seizure_clips.py / cut_non_seizure_clips.py (EEG repo) line for line where it matters; the only
# change is datetime.fromtimestamp -> Central wall time explicitly (the cutting box ran on Central time: its logs print
# XML starts as Central wall time, e.g. RN242-RN243.20231012200400 -> 2023-10-12 09:40:53).

def _xml_start_time(fpath):
    from xml.etree import ElementTree as ET
    base = os.path.splitext(fpath)[0]
    for ext in (".XML", ".xml"):
        cand = base + ext
        if os.path.exists(cand):
            try:
                node = ET.parse(cand).getroot().find("DSI_utc_start_time")
                if node is not None and node.text:
                    return AC.central_of_epoch(int(node.text.strip()))
            except Exception:
                return None
    return None


def parse_video_files(folder):
    p14, p12 = re.compile(r'(\d{14})'), re.compile(r'(\d{12})')
    videos = []
    files = sorted(set(glob.glob(os.path.join(glob.escape(folder), "*.mp4")) +
                       glob.glob(os.path.join(glob.escape(folder), "*", "*.mp4"))))
    for fpath in files:
        fname = os.path.basename(fpath)
        start_dt = _xml_start_time(fpath)
        if start_dt is None:
            for pat, fmt in [(p14, "%Y%m%d%H%M%S"), (p12, "%Y%m%d%H%M")]:
                m = pat.search(fname)
                if m:
                    try:
                        start_dt = dt.datetime.strptime(m.group(1), fmt)
                        break
                    except ValueError:
                        pass
        if start_dt:
            videos.append((start_dt, fpath, fname))
    return sorted(videos, key=lambda x: x[0])


_CV2 = {}


def get_video_info(fpath):
    """cv2 CAP_PROP_FRAME_COUNT / CAP_PROP_FPS, as the cutters (cached per file)."""
    if fpath not in _CV2:
        import cv2
        cap = cv2.VideoCapture(fpath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        dur_s = frames / fps if fps > 0 and frames > 0 else None
        if not dur_s:
            raise SystemExit(f"cv2 gives no duration for {fpath}; the parent would have used ffmpeg -i")
        _CV2[fpath] = (dur_s, round(fps, 3), width, height, frames, fps)
    return _CV2[fpath]


def build_video_intervals(videos):
    out = []
    for i, (start_dt, fpath, fname) in enumerate(videos):
        dur_s, fps, width, height, frames, fps_raw = get_video_info(fpath)
        out.append({"start": start_dt, "end": start_dt + dt.timedelta(seconds=dur_s), "dur_s": dur_s, "fps": fps,
                    "width": width, "height": height, "fpath": fpath, "fname": fname,
                    "relpath": os.path.relpath(fpath, RAW), "cv2_frames": frames, "cv2_fps": fps_raw})
    return out


_EDFH = {}


def parse_edf_files(folder):
    import mne
    recs = []
    files = sorted(set(glob.glob(os.path.join(glob.escape(folder), "*.edf")) +
                       glob.glob(os.path.join(glob.escape(folder), "*.EDF")) +
                       glob.glob(os.path.join(glob.escape(folder), "*", "*.edf"))))
    for fpath in files:
        if fpath not in _EDFH:
            raw = mne.io.read_raw_edf(fpath, preload=False, verbose=False)
            sd = raw.info["meas_date"]
            if sd is None:
                raise SystemExit(f"{fpath}: no meas_date (the parent would parse the file name)")
            _EDFH[fpath] = dict(start=sd.replace(tzinfo=None), dur_s=raw.n_times / raw.info["sfreq"],
                                n_ch=len(raw.ch_names), sfreq=raw.info["sfreq"], fpath=fpath,
                                fname=os.path.basename(fpath), source="header", relpath=os.path.relpath(fpath, RAW),
                                ch_names=list(raw.ch_names))
            raw.close()
        e = dict(_EDFH[fpath])
        e["end"] = e["start"] + dt.timedelta(seconds=e["dur_s"])
        recs.append(e)
    recs.sort(key=lambda x: x["start"])
    return recs


def find_edf(t, edf_records):
    for e in edf_records:
        if e["start"] <= t < e["end"]:
            return e
    return None


def find_video(t, intervals):
    for v in intervals:
        if v["start"] <= t < v["end"]:
            return v
    return None


def _xlsx_rows(excel_path):
    import openpyxl
    wb = openpyxl.load_workbook(excel_path)
    ws = wb.active
    headers = [str(c.value).strip().lower() if c.value else "" for c in next(ws.iter_rows(min_row=1, max_row=1))]

    def col(*names):
        for n in names:
            for i, h in enumerate(headers):
                if n in h:
                    return i
        return None

    ix = dict(label=col("label", "stage", "type"), dur=col("dur", "length"), start=col("start"), end=col("end"),
              spike=col("spike"))
    if ix["start"] is None:
        raise SystemExit(f"{excel_path}: no Start column ({headers})")
    return list(ws.iter_rows(min_row=2, values_only=True)), ix


def parse_annotations_seizure(excel_path):
    """cut_seizure_clips.parse_annotations (no --filter)."""
    rows, ix = _xlsx_rows(excel_path)
    out = []
    for i, row in enumerate(rows):
        start = row[ix["start"]]
        if start is None:
            continue
        if not isinstance(start, (int, float)) and not hasattr(start, 'hour'):
            try:
                dt.datetime.strptime(str(start).strip(), "%H:%M:%S")
            except Exception:
                continue
        label = row[ix["label"]] if ix["label"] is not None else None
        label = str(label).strip() if label is not None else "Unknown"
        dur_raw = row[ix["dur"]] if ix["dur"] is not None else None
        try:
            duration = float(dur_raw) if dur_raw is not None else 30.0
        except (ValueError, TypeError):
            duration = 30.0
        end = row[ix["end"]] if ix["end"] is not None else None
        spikes = row[ix["spike"]] if ix["spike"] is not None else 0
        out.append({"idx": i + 1, "label": label, "duration": duration, "start": start, "end": end, "spikes": spikes})
    return out


def parse_annotations_nonseizure(excel_path):
    """cut_non_seizure_clips.parse_annotations (no --filter): all rows, end filled, spikes or 0."""
    out = []
    for r in parse_annotations_seizure(excel_path):
        sz_end = r["end"] if r["end"] else r["start"] + dt.timedelta(seconds=r["duration"])
        out.append(dict(r, end=sz_end, spikes=r["spikes"] or 0))
    return out


def build_blocked_intervals(all_seizures):
    margin = dt.timedelta(seconds=PRE_BUFFER_S + SAFETY_S)
    blocked = []
    for sz in all_seizures:
        blocked.append((sz["start"] - margin, sz["end"] + dt.timedelta(seconds=POST_BUFFER_S + SAFETY_S)))
    blocked.sort(key=lambda x: x[0])
    merged = []
    for b in blocked:
        if merged and b[0] <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b[1]))
        else:
            merged.append(list(b))
    return [(dt.datetime(*b[0].timetuple()[:6], b[0].microsecond), dt.datetime(*b[1].timetuple()[:6], b[1].microsecond))
            for b in merged]


def find_free_windows(video, blocked_intervals, clip_dur_s):
    vid_start, vid_end = video["start"], video["end"]
    need = dt.timedelta(seconds=clip_dur_s)
    free, cursor = [], vid_start
    for b_start, b_end in blocked_intervals:
        b_start, b_end = max(b_start, vid_start), min(b_end, vid_end)
        if b_start >= vid_end or b_end <= vid_start:
            continue
        if cursor < b_start:
            free.append((cursor, b_start))
        cursor = max(cursor, b_end)
    if cursor < vid_end:
        free.append((cursor, vid_end))
    return [(s, e) for s, e in free if (e - s) >= need]


def sample_random_window(rng, free_windows, clip_dur_s):
    need = dt.timedelta(seconds=clip_dur_s)
    weights, starts = [], []
    for s, e in free_windows:
        usable_end = e - need
        if usable_end > s:
            weights.append((usable_end - s).total_seconds())
            starts.append(s)
    if not weights:
        return None
    total = sum(weights)
    r = rng.uniform(0, total)
    cum = 0
    for w, s in zip(weights, starts):
        cum += w
        if r <= cum:
            return s + dt.timedelta(seconds=rng.uniform(0, w))
    return starts[-1]


def _hms(s):
    s = max(0, int(s))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def safe_label(label):
    return re.sub(r"[^\w]", "_", label)


def seizure_info_text(sz, sz_end, clip_start, clip_end, video, edf):
    """cut_seizure_clips.process_session's info.txt, byte for byte."""
    s = (f"Seizure index  : {sz['idx']}\n"
         f"Label          : {sz['label']}\n"
         f"Seizure start  : {sz['start']}\n"
         f"Seizure end    : {sz_end}\n"
         f"Duration (s)   : {sz['duration']:.2f}\n"
         f"Spikes         : {sz['spikes']}\n"
         f"Clip start     : {clip_start}\n"
         f"Clip end       : {clip_end}\n"
         f"Pre-buffer (s) : {PRE_BUFFER_S}\n"
         f"Post-buffer(s) : {POST_BUFFER_S}\n"
         f"Video file     : {video['fname'] if video else 'N/A'}\n"
         f"EDF file       : {edf['fname'] if edf else 'N/A'}\n")
    if edf:
        s += (f"EEG channels   : {edf['n_ch']}\n"
              f"EEG sfreq (Hz) : {edf['sfreq']}\n"
              f"EDF date src   : {edf['source']}\n")
    if video:
        s += (f"Video FPS      : {video['fps']}\n"
              f"Video res      : {video['width']}x{video['height']}\n")
    return s


def nonseizure_info_text(clip_n, sz, clip_start_dt, clip_end_dt, clip_dur_s, video, seek_s, edf):
    """cut_non_seizure_clips.process_session's info.txt, byte for byte."""
    s = (f"Non-seizure clip index   : {clip_n}\n"
         f"Matched seizure index    : {sz['idx']}\n"
         f"Matched seizure label    : {sz['label']}\n"
         f"Matched seizure start    : {sz['start']}\n"
         f"Matched seizure end      : {sz['end']}\n"
         f"Matched seizure dur (s)  : {sz['duration']:.2f}\n"
         f"Clip start               : {clip_start_dt}\n"
         f"Clip end                 : {clip_end_dt}\n"
         f"Clip duration (s)        : {clip_dur_s:.2f}\n"
         f"Pre-buffer (s)           : {PRE_BUFFER_S}\n"
         f"Post-buffer (s)          : {POST_BUFFER_S}\n"
         f"Safety margin (s)        : {SAFETY_S}\n"
         f"Video file               : {video['fname']}\n"
         f"Video seek               : {_hms(seek_s)} ({seek_s:.2f}s)\n")
    if edf:
        s += (f"EDF file                 : {edf['fname']}\n"
              f"EEG channels             : {edf['n_ch']}\n"
              f"EEG sfreq (Hz)           : {edf['sfreq']}\n")
    return s


def seizure_spec(sz, vid_ivals, edf_records, other_ivals=None):
    """The seizure cutter's decisions for one xlsx row (process_session's loop body)."""
    sz_start = sz["start"]
    sz_end = sz["end"] if sz["end"] else sz_start + dt.timedelta(seconds=sz["duration"])
    clip_start = sz_start - dt.timedelta(seconds=PRE_BUFFER_S)
    clip_end = sz_end + dt.timedelta(seconds=POST_BUFFER_S)
    clip_dur_s = (clip_end - clip_start).total_seconds()
    video = find_video(sz_start, vid_ivals)
    video_src = "session" if video else None
    if video is None and other_ivals:
        video = find_video(sz_start, other_ivals)
        video_src = "other_day_folder" if video else None
    edf = find_edf(sz_start, edf_records)
    d = dict(kind="seizure", sz_idx=sz["idx"], label=sz["label"], sz_start=sz_start, sz_end=sz_end,
             sz_dur=sz["duration"], spikes=sz["spikes"], clip_start=clip_start, clip_end=clip_end,
             clip_dur_s=clip_dur_s, video_src=video_src,
             clip_dir=f"seizure_{sz['idx']:02d}_{safe_label(sz['label'])}_{sz_start.strftime('%Y%m%d_%H%M%S')}")
    if video:
        raw_seek = (sz_start - video["start"]).total_seconds()
        seek_s = max(0.0, raw_seek - PRE_BUFFER_S)
        d.update(video=_vfields(video), seek_s=seek_s, v_clip_dur_s=min(clip_dur_s, video["dur_s"] - seek_s))
    if edf:
        d.update(edf=_efields(edf))
    d["info_text"] = seizure_info_text(sz, sz_end, clip_start, clip_end, video, edf)
    return d


def _vfields(v):
    return {k: v[k] for k in ("start", "end", "dur_s", "fps", "width", "height", "fpath", "fname", "relpath",
                              "cv2_frames", "cv2_fps")}


def _efields(e):
    return {k: e[k] for k in ("start", "end", "dur_s", "n_ch", "sfreq", "fpath", "fname", "relpath", "source",
                              "ch_names")}


def replay_negatives(all_seizures, target_seizures, vid_ivals, edf_records, blocked_from=None):
    """cut_non_seizure_clips.process_session's sampling loop with random.seed(42): every target in order.

    blocked_from: the events whose +-40 s spans are blocked (the parent: all_seizures of the same xlsx)."""
    rng = random.Random(NS_SEED)
    blocked = build_blocked_intervals(blocked_from if blocked_from is not None else all_seizures)
    out = []
    for clip_n, sz in enumerate(target_seizures, start=1):
        sz_start = sz["start"]
        clip_dur_s = PRE_BUFFER_S + sz["duration"] + POST_BUFFER_S
        source_video = find_video(sz_start, vid_ivals)
        cands = vid_ivals if source_video is None else [source_video] + [v for v in vid_ivals if v is not source_video]
        clip_start_dt, chosen = None, None
        for v in cands:
            fw = find_free_windows(v, blocked, clip_dur_s)
            if not fw:
                continue
            s = sample_random_window(rng, fw, clip_dur_s)
            if s is not None:
                clip_start_dt, chosen = s, v
                break
        rec = dict(clip_n=clip_n, sz=sz, clip_dur_s=clip_dur_s, source_video=source_video["fname"] if source_video else None)
        if clip_start_dt is None:
            rec["status"] = "NO_FREE_WINDOW"
            out.append(rec)
            continue
        clip_end_dt = clip_start_dt + dt.timedelta(seconds=clip_dur_s)
        seek_s = (clip_start_dt - chosen["start"]).total_seconds()
        edf = find_edf(clip_start_dt, edf_records)
        rec.update(status="ok", clip_start=clip_start_dt, clip_end=clip_end_dt, video=_vfields(chosen), seek_s=seek_s,
                   v_clip_dur_s=min(clip_dur_s, chosen["dur_s"] - seek_s), edf=_efields(edf) if edf else None,
                   clip_dir=(f"clip_{clip_n:02d}_vs_seizure_{sz['idx']:02d}_{safe_label(sz['label'])}"
                             f"_{sz_start.strftime('%Y%m%d_%H%M%S')}"),
                   info_text=nonseizure_info_text(clip_n, sz, clip_start_dt, clip_end_dt, clip_dur_s, chosen, seek_s,
                                                  edf))
        out.append(rec)
    return out


# ============================================================================ 1. plan
def timing_cause(raw, fname, cs):
    """Known clock defects of the camera file at label time cs (naive local) -> (cause, expected offset note)."""
    v = raw.vid.get(fname)
    if v is None:
        return "unknown_file", None
    stamp = v["stamp"]
    if stamp == RC.STAMP_1013:
        return "roomD_1013_gap", "~-10523 s"
    if stamp in RC.STAMPS_STEP:
        return "roomC_clock_step", "-59.8..-60.8 s after the step"
    if stamp == STAMP_13S:
        return "roomC_13s_file", "~+13 s (measure)"
    if "frames_missing_vs_wallclock" in (v["problems"] or ""):
        return "timing_defect_other", None
    st = RC.utc_naive(v["utc_int"])
    if cs.date() == dt.date(2023, 11, 5) and cs.hour == 1:
        return "dst_ambiguous_hour", "+3600 s"
    if st < RC.FALL_UTC and (st + dt.timedelta(seconds=v["dur"]) > RC.FALL_UTC or cs >= RC.DST_DAY_1):
        return ("dst_after_change", "+3600 s") if cs >= RC.DST_DAY_2 else ("dst_before_change", "0")
    return "none", "0"


def existing_dirs(animal, session):
    out = {}
    for root, tag in ((FULL, "data_full"), (DATA, "data")):
        p = os.path.join(root, f"Data_{animal}_cropped", session)
        if os.path.isdir(p):
            for d in os.listdir(p):
                out.setdefault(d, []).append(tag)
    return out


def all_existing_names(animal):
    """Every clip dir name of the animal in data/ and data_full/ (any session) -> [session, ...]."""
    out = defaultdict(list)
    for root in (FULL, DATA):
        for p in glob.glob(os.path.join(root, f"Data_{animal}_cropped", "*", "*")):
            out[os.path.basename(p)].append(os.path.basename(os.path.dirname(p)))
    return out


def cmd_plan(args):
    t0 = time.time()
    raw = RC.Raw()
    cands = [r for r in read_csv(CAND_CSV) if r["animal"] in LABELLED]
    lab = Counter(r["label"] for r in cands)
    print(f"candidates: {len(cands)} in labelled animals {dict(Counter(r['animal'] for r in cands))}, labels {dict(lab)}")
    if any(r["label"] not in ("Stage 2", "Stage 3", "Stage 4", "Stage 5") for r in cands):
        raise SystemExit("a candidate in a labelled animal is not Stage 2-5")
    events = read_csv(EVENTS_CSV)
    sessions = OrderedDict()
    for r in cands:
        sessions.setdefault((r["animal"], r["day_folder"]), []).append(r)
    clips, excluded, repro_existing, sess_info = [], [], [], {}
    controls = []
    for (animal, day), rows in sessions.items():
        folder = os.path.join(RAW, animal, day)
        session = day
        if not os.path.isdir(os.path.join(FULL, f"Data_{animal}_cropped", session)):
            raise SystemExit(f"no data_full session dir for {animal}/{session}")
        vid_ivals = build_video_intervals(parse_video_files(folder))
        edf_records = parse_edf_files(folder)
        xlsx_all = sorted(glob.glob(os.path.join(glob.escape(folder), "*.xlsx")) +
                          glob.glob(os.path.join(glob.escape(folder), "*.XLSX")))
        parsed = {x: parse_annotations_seizure(x) for x in xlsx_all}
        ex = existing_dirs(animal, session)
        # which xlsx the parent read: the one whose rows the existing dirs reproduce (name = idx + label + onset)
        names_by_x = {}
        for x, evs in parsed.items():
            nm = set()
            for sz in evs:
                tag = f"_{sz['idx']:02d}_{safe_label(sz['label'])}_{sz['start'].strftime('%Y%m%d_%H%M%S')}"
                nm.add(tag)
            names_by_x[x] = nm
        hit = {x: sum(1 for d in ex if any(d.endswith(t) for t in nm)) for x, nm in names_by_x.items()}
        parent_x = max(hit, key=lambda x: hit[x]) if len(xlsx_all) > 1 else xlsx_all[0]
        if len(xlsx_all) > 1 and sorted(hit.values())[-2] == hit[parent_x]:
            raise SystemExit(f"{animal}/{day}: cannot tell which xlsx the parent read: {hit}")
        second_x = [x for x in xlsx_all if x != parent_x]
        print(f"\n== {animal}/{day}: {len(vid_ivals)} videos, {len(edf_records)} EDFs, xlsx {len(xlsx_all)} "
              f"(parent read {os.path.basename(parent_x)}: {len(parsed[parent_x])} rows), {len(ex)} existing dirs, "
              f"{len(rows)} candidates", flush=True)
        other_ivals = {}
        for r in rows:
            if r["video_day_folder"] and r["video_day_folder"] != day:
                of = os.path.join(RAW, animal, r["video_day_folder"])
                if of not in other_ivals:
                    other_ivals[of] = build_video_intervals(parse_video_files(of))
        oiv = [v for vs in other_ivals.values() for v in vs]
        # ---- reproduce every existing dir of the session from the parent xlsx
        sz_rows = parsed[parent_x]
        spec_by_name = {}
        for sz in sz_rows:
            s = seizure_spec(sz, vid_ivals, edf_records)
            spec_by_name[s["clip_dir"]] = s
        all_ns = parse_annotations_nonseizure(parent_x)
        replay = replay_negatives(all_ns, all_ns, vid_ivals, edf_records)
        for rp in replay:
            if rp["status"] == "ok":
                spec_by_name[rp["clip_dir"]] = dict(kind="nonseizure", **{k: v for k, v in rp.items() if k != "sz"},
                                                    sz_idx=rp["sz"]["idx"], label=rp["sz"]["label"],
                                                    sz_start=rp["sz"]["start"], sz_end=rp["sz"]["end"],
                                                    sz_dur=rp["sz"]["duration"])
        n_ok = Counter()
        for d in sorted(ex):
            if not (d.startswith("seizure_") or d.startswith("clip_")):
                continue
            s = spec_by_name.get(d)
            info_p = os.path.join(FULL, f"Data_{animal}_cropped", session, d, "info.txt")
            have_info = os.path.exists(info_p)
            same = None
            if s is not None and have_info:
                same = open(info_p).read() == s["info_text"]
            files_ = sorted(os.listdir(os.path.dirname(info_p))) if os.path.isdir(os.path.dirname(info_p)) else []
            kind = "seizure" if d.startswith("seizure_") else "nonseizure"
            repro_existing.append(dict(animal=animal, session=session, clip_dir=d, kind=kind, name_reproduced=s is not None,
                                       has_info=have_info, info_identical=same, files="|".join(files_),
                                       in_data="data" in ex[d], in_full="data_full" in ex[d]))
            n_ok[(kind, s is not None, same)] += 1
        print(f"   existing dirs reproduced (kind, name, info identical): {dict(n_ok)}")
        bad = [x for x in repro_existing if x["animal"] == animal and x["session"] == session and
               (not x["name_reproduced"] or x["info_identical"] is False)]
        if bad:
            raise SystemExit(f"{animal}/{day}: {len(bad)} existing dirs NOT reproduced, e.g. {bad[:3]}")
        # ---- controls
        for ca, cs_, cd in CONTROLS:
            if (ca, cs_) != (animal, session):
                continue
            if cd is None:
                if animal == "RN213":
                    full_ns = [x for x in ex if x.startswith("clip_") and os.path.exists(
                        os.path.join(FULL, f"Data_{animal}_cropped", session, x, "eeg.edf"))]
                    cd = max(full_ns, key=lambda x: int(x.split("_")[1]))
                else:
                    full_sz = [x for x in ex if x.startswith("seizure_") and os.path.exists(
                        os.path.join(FULL, f"Data_{animal}_cropped", session, x, "eeg.edf"))]
                    cd = max(full_sz, key=lambda x: int(x.split("_")[1]))
            controls.append(dict(spec_by_name[cd], animal=animal, session=session, clip_dir=cd, control=True))
        # ---- new seizure clips
        # the parent's seizure cutter gives EVERY row of its xlsx a dir index (row + 1), cut or not
        max_sz = max([int(d.split("_")[1]) for d in ex if d.startswith("seizure_")] + [sz["idx"] for sz in sz_rows] + [0])
        max_ns = max([int(d.split("_")[1]) for d in ex if d.startswith("clip_")] + [rp["clip_n"] for rp in replay] + [0])
        new_negs_needed = []
        for r in rows:
            xl = os.path.join(RAW, r["xlsx"])
            if xl not in parsed:
                raise SystemExit(f"candidate xlsx not in the folder: {r['xlsx']}")
            m = [sz for sz in parsed[xl] if hasattr(sz["start"], "hour") and sz["label"] == r["label"]
                 and abs((sz["start"] - PD(r["start"])).total_seconds()) < 0.002]
            if len(m) != 1:
                raise SystemExit(f"candidate {r['animal']} {r['start']} matches {len(m)} xlsx rows")
            sz = m[0]
            if sz["idx"] != int(r["row"]) + 1:
                raise SystemExit(f"idx {sz['idx']} != row + 1 for {r}")
            base = dict(animal=animal, session=session, day_folder=day, xlsx=r["xlsx"], xlsx_row=int(r["row"]),
                        outcome=r["outcome"], parent_idx=sz["idx"], parent_xlsx=os.path.relpath(parent_x, RAW),
                        existing_negative=(json.loads(r["non_seizure_dirs"])[0] if r["non_seizure_dirs"] not in ("", "[]")
                                           else None), deviations=[])
            if xl != parent_x:
                max_sz += 1
                sz = dict(sz, idx=max_sz)
                base["deviations"].append(f"second xlsx in folder (parent read {os.path.basename(parent_x)}): numbered "
                                          f"{max_sz} after the session's highest index (parent rule would give "
                                          f"{base['parent_idx']})")
            s = seizure_spec(sz, vid_ivals, edf_records, oiv)
            s.update(base)
            if s.get("video_src") == "other_day_folder":
                s["deviations"].append(f"video from day folder {r['video_day_folder']} (parent: no video in {day})")
            if s.get("video") is None:
                s.update(status="excluded", reason=(
                    f"no camera file holds the onset under the parent's rule (cv2 file ends; audit: {r['outcome']}, "
                    f"seek {r['video_seek_s']} s into {r['video'].split('/')[-1]}, i.e. the onset is in the file's last "
                    f"second and the next file starts 60 s later): a video clip would show only the pre-buffer"))
                excluded.append(s)
                print(f"   EXCLUDED {s['clip_dir']}: {s['reason'][:120]}")
                continue
            if s.get("edf") is None:
                raise SystemExit(f"no EDF for {s['clip_dir']}")
            s["status"] = "planned"
            clips.append(s)
            if base["existing_negative"] is None:
                new_negs_needed.append((xl, sz, s))
        # ---- new negatives: parent xlsx from the replayed stream; second xlsx: fresh seed 42, both files blocked
        rp_by_idx = {rp["sz"]["idx"]: rp for rp in replay}
        second_needed = [t for t in new_negs_needed if t[0] != parent_x]
        for xl, sz, s in new_negs_needed:
            if xl != parent_x:
                continue
            rp = rp_by_idx[sz["idx"]]
            if rp["status"] != "ok":
                raise SystemExit(f"replay found no free window for {s['clip_dir']}")
            if rp["clip_dir"] in ex:
                raise SystemExit(f"replayed negative already exists: {rp['clip_dir']}")
            clips.append(_neg_spec(rp, s, animal, session, day, parent_x, "continuation of the session's seed-42 stream"))
        if second_needed:
            all_union = [dict(x) for xx in xlsx_all for x in parse_annotations_nonseizure(xx)]
            for xl in sorted({t[0] for t in second_needed}):
                ns_rows = parse_annotations_nonseizure(xl)
                rp2 = replay_negatives(ns_rows, ns_rows, vid_ivals, edf_records, blocked_from=all_union)
                by = {r_["sz"]["idx"]: r_ for r_ in rp2}
                for xl_, sz, s in second_needed:
                    if xl_ != xl:
                        continue
                    r_ = by[s["parent_idx"]]
                    if r_["status"] != "ok":
                        raise SystemExit(f"no free window for the negative of {s['clip_dir']}")
                    max_ns += 1
                    sz2 = dict(r_["sz"], idx=sz["idx"])
                    r_ = dict(r_, clip_n=max_ns, sz=sz2,
                              clip_dir=(f"clip_{max_ns:02d}_vs_seizure_{sz['idx']:02d}_{safe_label(sz['label'])}"
                                        f"_{sz['start'].strftime('%Y%m%d_%H%M%S')}"),
                              info_text=nonseizure_info_text(max_ns, sz2, r_["clip_start"], r_["clip_end"],
                                                             r_["clip_dur_s"], r_["video"], r_["seek_s"],
                                                             r_["edf"]))
                    ns = _neg_spec(r_, s, animal, session, day, xl,
                                   f"second xlsx: seed 42 over {os.path.basename(xl)}, events of all "
                                   f"{len(xlsx_all)} xlsx blocked; numbered {max_ns} after the session's highest")
                    clips.append(ns)
        sess_info[f"{animal}/{day}"] = dict(
            videos=[dict(fname=v["fname"], start=str(v["start"]), dur_s=v["dur_s"], fps=v["fps"], frames=v["cv2_frames"])
                    for v in vid_ivals],
            edfs=[dict(fname=e["fname"], start=str(e["start"]), dur_s=e["dur_s"], n_ch=e["n_ch"]) for e in edf_records],
            xlsx=[os.path.basename(x) for x in xlsx_all], parent_xlsx=os.path.basename(parent_x),
            parent_rows=len(sz_rows), existing=len(ex), candidates=len(rows),
            reproduced={f"{k[0]}|name={k[1]}|info={k[2]}": n for k, n in n_ok.items()},
            other_folders=sorted(os.path.relpath(k, RAW) for k in other_ivals))
    # ---- names: no collision with any existing dir of the animal (any session), nor among the new ones
    seen = Counter((c["animal"], c["session"], c["clip_dir"]) for c in clips)
    dup = [k for k, n in seen.items() if n > 1]
    if dup:
        raise SystemExit(f"new clip names collide with each other: {dup}")
    coll = []
    names_cache = {}
    for c in clips:
        names = names_cache.setdefault(c["animal"], all_existing_names(c["animal"]))
        if c["clip_dir"] in names:
            coll.append((c["animal"], c["session"], c["clip_dir"], names[c["clip_dir"]]))
        idx_clash = [d for d in existing_dirs(c["animal"], c["session"])
                     if d.split("_")[:2] == c["clip_dir"].split("_")[:2]]
        c["index_clash_in_session"] = "|".join(sorted(idx_clash))
    if coll:
        raise SystemExit(f"new clip names collide with existing dirs: {coll[:5]}")
    # ---- timing causes, keys, labels
    for c in clips + controls:
        c.setdefault("deviations", [])
        c["key"] = key_of(c["animal"], c["session"], c["clip_dir"])
        c["rel"] = rel_of(c["animal"], c["session"], c["clip_dir"])
        if c["kind"] == "seizure":
            m = re.search(r"Stage_[0-9]+", c["clip_dir"])
            c["y"] = STAGE.get(m.group()) if m else None
        else:
            c["y"] = 0
        cause, note = timing_cause(raw, c["video"]["fname"], PD(str(c["clip_start"])))
        c["timing_cause"], c["timing_expected"] = cause, note
        v = raw.vid.get(c["video"]["fname"])
        c["camera"] = v["camera"] if v else None
        c["sync_s"] = v["sync"] if v else None
        c["rho"] = v["rho"] if v else None
        # the cutter's nominal timeline uses Central(int(utc)); both must agree with our interval start
        if v is not None and abs(nsec(PD(str(c["video"]["start"]))) - (v["u0"] - v["off"])) > 1e-6:
            raise SystemExit(f"video start mismatch {c['video']['fname']}")
        c["truncated_s"] = max(0.0, c["clip_dur_s"] - c["v_clip_dur_s"])
    print("\ntiming causes of the new clips:", dict(Counter(c["timing_cause"] for c in clips)))
    print("new clips:", dict(Counter((c["kind"], c.get("y")) for c in clips)))
    print("deviations:", dict(Counter(d.split(":")[0].split(" (")[0] for c in clips for d in c["deviations"])))
    write_jsonl("plan/clips.jsonl", clips)
    write_jsonl("plan/excluded.jsonl", excluded)
    write_jsonl("plan/controls.jsonl", controls)
    write_csv("plan/reproduce_existing.csv", ["animal", "session", "clip_dir", "kind", "name_reproduced", "has_info",
                                              "info_identical", "files", "in_data", "in_full"], repro_existing)
    write_text("plan/sessions.json", json.dumps(sess_info, indent=1, default=str))
    # decode jobs (recut_clips format): every camera file a new clip or control is cut from
    jobs = {}
    for c in clips + controls:
        v = raw.vid[c["video"]["fname"]]
        if v["name"] not in jobs:
            jobs[v["name"]] = dict(id=f"{v['camera']}__{v['stamp']}", name=v["name"],
                                   path=c["video"]["fpath"], camera=v["camera"], animals=RC.cam_animals(v), t=None)
    jl = sorted(jobs.values(), key=lambda j: j["id"])
    for j in jl:
        j["boxes"] = [(a,) + RC.crop_box(a) for a in j["animals"]]
    write_text("plan/jobs.json", json.dumps(jl, indent=1))
    print(f"decode jobs: {len(jl)};  controls: {[c['rel'] for c in controls]}")
    print(f"plan done in {time.time() - t0:.0f} s")


def _neg_spec(rp, seizure, animal, session, day, xlsx, how):
    sz = rp["sz"]
    d = dict(kind="nonseizure", animal=animal, session=session, day_folder=day, xlsx=os.path.relpath(xlsx, RAW),
             xlsx_row=seizure["xlsx_row"], outcome=seizure["outcome"], parent_idx=seizure["parent_idx"],
             parent_xlsx=seizure["parent_xlsx"], clip_n=rp["clip_n"], sz_idx=sz["idx"], label=sz["label"],
             sz_start=sz["start"], sz_end=sz["end"], sz_dur=sz["duration"], clip_start=rp["clip_start"],
             clip_end=rp["clip_end"], clip_dur_s=rp["clip_dur_s"], video=rp["video"], seek_s=rp["seek_s"],
             v_clip_dur_s=rp["v_clip_dur_s"], edf=rp["edf"], clip_dir=rp["clip_dir"], info_text=rp["info_text"],
             paired_seizure=seizure["clip_dir"], negative_rule=how, status="planned", deviations=[],
             video_src=("seizure_file" if rp["video"]["fname"] == rp.get("source_video") else
                        "session_file_fallback"))
    if "second xlsx" in how:
        d["deviations"].append(how)
    if d["edf"] is None:
        raise SystemExit(f"no EDF for the negative {d['clip_dir']}")
    return d


# ============================================================================ 2. cut (video) + info.txt
def load_plan(include_controls=True):
    clips = read_jsonl("plan/clips.jsonl")
    ctr = read_jsonl("plan/controls.jsonl") if include_controls else []
    return clips, ctr


def clip_paths(c):
    """(final dir, uncropped copy-cut work file) for a new clip or a control."""
    if c.get("control"):
        return out_path("repro", "data_full", c["rel"], "x"), out_path("repro", "work", c["rel"], "video.mp4")
    return out_path("data_full", c["rel"], "x"), out_path("work", "cut", c["rel"], "video.mp4")


def video_source(c):
    """(raw path, seek, v_clip_dur) actually used: the parent's, or the corrected one from `recut`."""
    if c.get("corrected_seek_s") is not None:
        return c["corrected_fpath"], float(c["corrected_seek_s"]), float(c["corrected_v_clip_dur_s"])
    return c["video"]["fpath"], float(c["seek_s"]), float(c["v_clip_dur_s"])


def cmd_cut(args):
    clips, ctr = load_plan()
    todo = clips + ctr

    def one(c):
        dpath, work = clip_paths(c)
        d = os.path.dirname(dpath)
        dst = os.path.join(d, "video.mp4")
        info = os.path.join(d, "info.txt")
        if os.path.exists(dst) and os.path.exists(info) and not args.overwrite:
            return c["rel"], "cached"
        src, seek, vdur = video_source(c)
        for p in (work, dst):
            if os.path.exists(p):
                os.remove(p)
        r1 = RC.cut_copy(src, seek, vdur, work)
        if r1.returncode != 0 or not os.path.exists(work):
            return c["rel"], f"cut_failed: {r1.stderr.strip()[-200:]}"
        r2 = RC.crop_encode(work, dst, c["animal"])
        if r2.returncode != 0 or not os.path.exists(dst):
            return c["rel"], f"crop_failed: {r2.stderr.strip()[-200:]}"
        with open(info + ".tmp", "w") as fh:
            fh.write(c["info_text"])
        os.replace(info + ".tmp", info)
        return c["rel"], "ok"

    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        res = list(ex.map(one, todo))
    st = Counter(s.split(":")[0] for _, s in res)
    for r, s in res:
        if not s.startswith(("ok", "cached")):
            print("  ", r, s)
    write_csv("logs/cut_status.csv", ["rel", "status"], [dict(rel=r, status=s) for r, s in res])
    print(f"cut: {dict(st)} in {time.time() - t0:.0f} s")


# ============================================================================ 3. eeg.edf
def write_edf_clip_labels(raw, out, labels, record_s=1.0):
    """EEG repo edf_clip_writer.write_edf_clip, line for line, except that each channel's EDF label is the SOURCE
    EDF's label instead of MNE's name cut to 16 characters (see the module doc)."""
    import numpy as np
    from edfio import Edf, EdfSignal
    sf = float(raw.info["sfreq"])
    data = raw.get_data()
    n = int(data.shape[1] // (sf * record_s) * (sf * record_s))
    if n <= 0:
        raise ValueError(f"window shorter than one {record_s}s data record")
    data = data[:, :n]
    signals = []
    for k, ch in enumerate(raw.ch_names):
        v = np.asarray(data[k], dtype=np.float64)
        finite = v[np.isfinite(v)]
        if finite.size == 0:
            lo, hi = -1.0, 1.0
            v = np.zeros_like(v)
        else:
            lo, hi = float(finite.min()), float(finite.max())
            v = np.nan_to_num(v, nan=lo, posinf=hi, neginf=lo)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = lo - 1e-6, lo + 1e-6
        signals.append(EdfSignal(v, sampling_frequency=sf, label=labels[k], physical_range=(lo, hi)))
    Edf(signals, data_record_duration=record_s).write(out)
    return n / sf


def parent_labels(ch_names):
    """The cutters' 16-char rename (cut_seizure_clips.cut_eeg_segment) -> the labels the parent wrote."""
    rename, used = {}, set(ch_names)
    for ch in ch_names:
        if len(ch) > 16:
            base, suffix, cand = ch[:14], 0, ch[:16]
            while cand in used and cand != ch:
                cand = f"{base}_{suffix}"
                suffix += 1
            rename[ch] = cand
            used.discard(ch)
            used.add(cand)
    return [rename.get(c, c)[:16] for c in ch_names]


def cmd_eeg(args):
    import mne
    mne.set_log_level("ERROR")
    import warnings
    warnings.filterwarnings("ignore")
    clips, ctr = load_plan()
    rows = []
    t0 = time.time()
    for i, c in enumerate(clips + ctr):
        dpath, _ = clip_paths(c)
        d = os.path.dirname(dpath)
        dst = os.path.join(d, "eeg.edf")
        row = dict(rel=c["rel"], control=bool(c.get("control")))
        if os.path.exists(dst) and not args.overwrite:
            row["status"] = "cached"
            rows.append(row)
            continue
        e = c["edf"]
        src = e["fpath"]
        t1 = time.time()
        raw = mne.io.read_raw_edf(src, preload=False, verbose=False)
        es = raw.info["meas_date"].replace(tzinfo=None)
        if str(es) != str(e["start"]):
            raise SystemExit(f"{src}: meas_date {es} != planned {e['start']}")
        cs, ce = PD(str(c["clip_start"])), PD(str(c["clip_end"]))
        t_start_s = max(0.0, (cs - es).total_seconds())
        t_end_s = min(float(e["dur_s"]), (ce - es).total_seconds())
        if t_end_s <= t_start_s:
            row["status"] = f"bad_window {t_start_s} {t_end_s}"
            rows.append(row)
            raw.close()
            continue
        raw.crop(tmin=t_start_s, tmax=t_end_s)
        raw.load_data(verbose=False)
        h = AC.edf_header(src)
        src_labels = h["labels"]
        if len(src_labels) != len(raw.ch_names) or any(not n.startswith(l) for n, l in zip(raw.ch_names, src_labels)):
            raise SystemExit(f"{src}: MNE channels {raw.ch_names} do not map onto header labels {src_labels}")
        plab = parent_labels(raw.ch_names)
        tmp = dst + ".tmp.edf"
        write_edf_clip_labels(raw, tmp, src_labels)
        os.replace(tmp, dst)
        if c.get("control"):         # also the parent's exact labels, to compare byte for byte with data_full
            write_edf_clip_labels(raw, os.path.join(d, "eeg_parentlabels.edf"), plab)
        raw.close()
        row.update(status="ok", t_start_s=t_start_s, t_end_s=t_end_s, n_ch=len(src_labels),
                   labels="|".join(src_labels), parent_labels="|".join(plab),
                   label_changed="|".join(f"{p}->{s}" for p, s in zip(plab, src_labels) if p != s), secs=time.time() - t1)
        rows.append(row)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(clips) + len(ctr)}  {time.time() - t0:.0f} s", flush=True)
    write_csv("logs/eeg_status.csv", ["rel", "control", "status", "t_start_s", "t_end_s", "n_ch", "labels",
                                      "parent_labels", "label_changed", "secs"], rows)
    print(f"eeg: {dict(Counter(r['status'].split()[0] for r in rows))} in {time.time() - t0:.0f} s")


# ============================================================================ 4-5. motion, activity (recut_clips)
def cmd_motion(args):
    jobs = RC.load_jobs()
    sel = jobs if args.index is None else [jobs[i] for i in range(args.index, len(jobs), args.stride)]
    for j in sel:
        print(RC.run_motion_job(j, args.threads), flush=True)


def activity_needs():
    """(animal, EDF relpath) -> naive [lo, hi]: every animal box on every decoded camera file, and every clip."""
    raw = RC.Raw()
    clips, ctr = load_plan()
    needs = {}

    def add(k, lo, hi):
        needs[k] = (min(needs[k][0], lo), max(needs[k][1], hi)) if k in needs else (lo, hi)

    pad = dt.timedelta(seconds=RC.SEARCH_S + 1800)
    for j in RC.load_jobs():
        v = raw.vid[j["name"]]
        s = from_nsec(v["u0"] - v["off"])
        e = s + dt.timedelta(seconds=v["dur"])
        for a in j["animals"]:
            ed = raw.edf_for(a, s, e)
            if ed is not None:
                add((a, ed["relpath"]), s - pad, e + pad)
    for c in clips + ctr:
        cs = PD(str(c["clip_start"]))
        add((c["animal"], c["edf"]["relpath"]), cs - pad, cs + pad)
    return needs


def cmd_activity(args):
    import numpy as np
    needs = activity_needs()
    print(f"{len(needs)} EDF reads")

    def one(item):
        (a, rel), (lo, hi) = item
        p = RC.edf_npz(rel)
        if os.path.exists(p):
            z = np.load(p)
            if z["t0"] <= nsec(lo) + 10 and z["t0"] + len(z["act"]) >= nsec(hi) - 10:
                return f"{rel}: cached"
        t0 = time.time()
        d = RC.read_activity(rel, lo, hi)
        np.savez_compressed(RC.ensure_dir(p) + ".tmp.npz", animal=a, relpath=rel, **d)
        os.replace(p + ".tmp.npz", p)
        return f"{rel}: {len(d['act'])} s, nonzero={int((d['act'] >= 0.5).sum())}, {time.time() - t0:.0f}s"

    with ThreadPoolExecutor(args.workers) as ex:
        for msg in ex.map(one, sorted(needs.items())):
            print(msg, flush=True)


# ============================================================================ 6. measure
MEAS_COLS = ["rel", "animal", "kind", "y", "camera", "raw_video", "timing_cause", "clip_start", "clip_end",
             "seek_s", "sync_s", "baseline_s", "edf", "win_s", "act_nonzero", "coarse_off", "coarse_r", "coarse_z",
             "second_r", "second_off", "margin", "sharp", "offset_s", "offset_r", "r_at0", "lag_s", "strong", "err",
             "partner", "p_edf", "p_lag_s", "p_offset_s", "p_offset_r", "p_coarse_z", "p_margin", "p_sharp", "p_strong",
             "p_err"]
PROF_COLS = ["file", "camera", "animal", "edf", "t_center", "pts_center", "win_s", "act_nonzero", "lag_s", "offset_s",
             "offset_r", "coarse_z", "margin", "sharp", "strong", "err"]


def baseline_of(v):
    """The lag a correctly cut clip measures: -DSI_sync_offset - lambda (recut_clips.method_baseline)."""
    return -v["sync"] - LAMBDA_S


def cmd_measure(args):
    raw = RC.Raw()
    TL = RC.Timeline(raw)
    ACT = RC.Activity()
    clips, ctr = load_plan()
    rows = []

    def one(c):
        v = raw.vid[c["video"]["fname"]]
        cs, ce = PD(str(c["clip_start"])), PD(str(c["clip_end"]))
        t0_, t1_ = nsec(cs), nsec(ce)
        edf_rel = c["edf"]["relpath"]
        row = dict(rel=c["rel"], animal=c["animal"], kind=c["kind"], y=c["y"], camera=v["camera"], raw_video=v["name"],
                   timing_cause=c["timing_cause"], clip_start=str(cs), clip_end=str(ce), seek_s=c["seek_s"],
                   sync_s=v["sync"], baseline_s=baseline_of(v), edf=edf_rel)
        if ACT.get(edf_rel) is None:
            row["err"] = "no_activity_npz"
            return row
        seg = RC.clip_segment(ACT, edf_rel, t0_, t1_)
        m = RC.measure_one(TL, ACT, v["camera"], c["animal"], edf_rel, v["off"], t0_, t1_, seg)
        m.pop("_at", None)
        row.update({k: m.get(k) for k in MEAS_COLS if k in m})
        if m.get("offset_s") is not None:
            row["lag_s"] = m["offset_s"]
            row["offset_s"] = m["offset_s"] - baseline_of(v)
        row["strong"] = RC.is_strong(m)
        for pa in RC.cam_animals(v):
            if pa == c["animal"]:
                continue
            ped = raw.edf_for(pa, cs - dt.timedelta(hours=1), ce + dt.timedelta(hours=1))
            if ped is None or ACT.get(ped["relpath"]) is None:
                continue
            pseg = RC.clip_segment(ACT, ped["relpath"], t0_, t1_)
            pm = RC.measure_one(TL, ACT, v["camera"], pa, ped["relpath"], v["off"], t0_, t1_, pseg)
            pm.pop("_at", None)
            row.update(partner=pa, p_edf=ped["relpath"], p_lag_s=pm.get("offset_s"),
                       p_offset_s=(pm["offset_s"] - baseline_of(v)) if pm.get("offset_s") is not None else None,
                       p_offset_r=pm.get("offset_r"), p_coarse_z=pm.get("coarse_z"), p_margin=pm.get("margin"),
                       p_sharp=pm.get("sharp"), p_strong=RC.is_strong(pm), p_err=pm.get("err"))
            break
        return row

    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        rows = list(ex.map(one, clips + ctr))
    write_csv("measure/clips.csv", MEAS_COLS, rows)
    print(f"clip measurements: {len(rows)}, strong {sum(1 for r in rows if r.get('strong'))}, "
          f"partner strong {sum(1 for r in rows if r.get('p_strong'))}  ({time.time() - t0:.0f} s)", flush=True)

    # clip-sized scan windows along every decoded file, every box: the file's clock profile
    import numpy as np
    jobs_ = RC.load_jobs()
    tasks = []
    for j in jobs_:
        v = raw.vid[j["name"]]
        s0 = v["u0"] - v["off"]
        for a in j["animals"]:
            ed = raw.edf_for(a, from_nsec(s0), from_nsec(s0 + v["dur"]))
            if ed is None or ACT.get(ed["relpath"]) is None:
                continue
            for tc in np.arange(s0 + RC.PROFILE_WIN_S / 2, s0 + v["dur"] - RC.PROFILE_WIN_S / 2 + 1, RC.PROFILE_STEP_S):
                tasks.append((j["name"], a, ed["relpath"], float(tc)))

    def prof_one(t):
        name, a, rel, tc = t
        v = raw.vid[name]
        seg = RC.clip_segment(ACT, rel, tc - 1, tc + 1)
        if seg is None:
            return None
        w0, w1 = tc - RC.PROFILE_WIN_S / 2 + RC.HALF_WIN_S, tc + RC.PROFILE_WIN_S / 2 - RC.HALF_WIN_S
        m = RC.measure_one(TL, ACT, v["camera"], a, rel, v["off"], w0, w1, seg, refine=False)   # integer lag
        m.pop("_at", None)
        lag = m.get("offset_s")
        return dict(file=name, camera=v["camera"], animal=a, edf=rel, t_center=AC.iso(from_nsec(tc)),
                    pts_center=tc - (v["u0"] - v["off"]), win_s=m.get("win_s"), act_nonzero=m.get("act_nonzero"),
                    lag_s=lag, offset_s=(lag - baseline_of(v)) if lag is not None else None,
                    offset_r=m.get("offset_r"), coarse_z=m.get("coarse_z"), margin=m.get("margin"),
                    sharp=m.get("sharp"), strong=RC.is_strong(m), err=m.get("err"))

    with ThreadPoolExecutor(args.workers) as ex:
        prof = [p for p in ex.map(prof_one, tasks) if p is not None]
    write_csv("measure/profiles.csv", PROF_COLS, prof)
    print(f"scan windows: {len(prof)}, strong {sum(1 for p in prof if p['strong'])}  ({time.time() - t0:.0f} s)")


# ============================================================================ 7. decide
DEC_COLS = ["rel", "animal", "kind", "y", "camera", "raw_video", "timing_cause", "expected_interval", "offset_s",
            "offset_r", "strong", "p_offset_s", "p_strong", "local_n", "local_units", "local_n_strong",
            "local_median_s", "local_spread_s", "file_n_strong", "file_median_s", "file_max_abs_s", "offset_used_s",
            "offset_source", "status", "reason"]


def cmd_decide(args):
    import numpy as np
    raw = RC.Raw()
    ms = {r["rel"]: r for r in read_csv(os.path.join(OUT, "measure", "clips.csv"))}
    prof = read_csv(os.path.join(OUT, "measure", "profiles.csv"))
    clips, ctr = load_plan()
    pool = defaultdict(list)          # file -> (t, offset, animal, strong, tag)
    for m in ms.values():
        t = nsec(PD(m["clip_start"]))
        if fnum(m.get("offset_s")) is not None and not m.get("err"):
            pool[m["raw_video"]].append((t, float(m["offset_s"]), m["animal"], m.get("strong") == "True",
                                         f"clip:{m['rel']}"))
        if fnum(m.get("p_offset_s")) is not None and not m.get("p_err"):
            pool[m["raw_video"]].append((t, float(m["p_offset_s"]), m["partner"], m.get("p_strong") == "True",
                                         f"partner:{m['rel']}"))
    for p in prof:
        if fnum(p.get("offset_s")) is not None and not p.get("err"):
            pool[p["file"]].append((nsec(PD(p["t_center"])), float(p["offset_s"]), p["animal"], p["strong"] == "True",
                                    f"scan:{p['file']}:{p['animal']}:{p['t_center']}"))
    file_stats = {}
    for f, items in pool.items():
        xs = [x[1] for x in items if x[3]]
        file_stats[f] = dict(n=len(xs), med=float(np.median(xs)) if xs else None,
                             max_abs=float(max(abs(x) for x in xs)) if xs else None)
    rows = []
    for c in clips + ctr:
        m = ms.get(c["rel"], {})
        v = raw.vid[c["video"]["fname"]]
        p = float(c["seek_s"])
        sub = {"none": "normal"}.get(c["timing_cause"], c["timing_cause"])
        if sub == "normal":
            lo_, hi_ = sorted((RC.model_offset(v, p, "normal", True), RC.model_offset(v, p, "normal", False)))
        elif sub == "roomC_13s_file":
            lo_, hi_ = 11.0, 15.0
        else:
            lo_, hi_ = RC.model_interval(v, p, sub, raw)
        dist = lambda x: 0.0 if lo_ <= x <= hi_ else min(abs(x - lo_), abs(x - hi_))  # noqa: E731
        d = dict(rel=c["rel"], animal=c["animal"], kind=c["kind"], y=c["y"], camera=v["camera"], raw_video=v["name"],
                 timing_cause=c["timing_cause"], expected_interval=f"{lo_:+.2f}..{hi_:+.2f}",
                 offset_s=fnum(m.get("offset_s")), offset_r=fnum(m.get("offset_r")), strong=m.get("strong") == "True",
                 p_offset_s=fnum(m.get("p_offset_s")), p_strong=m.get("p_strong") == "True")
        fs = file_stats.get(v["name"], {})
        d.update(file_n_strong=fs.get("n"), file_median_s=fs.get("med"), file_max_abs_s=fs.get("max_abs"))
        t = nsec(PD(str(c["clip_start"])))
        cl = RC.consensus([x for x in pool[v["name"]] if abs(x[0] - t) <= RC.CONSENSUS_S and x[4] != f"clip:{c['rel']}"])
        d.update(local_n=cl["n"], local_units=cl["units"], local_n_strong=cl["n_strong"], local_median_s=cl["median"],
                 local_spread_s=cl["spread"])
        used = src = status = reason = None
        if d["strong"]:
            if dist(d["offset_s"]) <= ALIGNED_TOL_S:
                used, src = d["offset_s"], "measured"
            else:
                status, reason = "flagged", (f"strong own measurement {d['offset_s']:+.1f} s is outside the expected "
                                             f"{d['expected_interval']} s")
        elif d["p_strong"]:
            if dist(d["p_offset_s"]) <= ALIGNED_TOL_S:
                used, src = d["p_offset_s"], "partner"
            else:
                status, reason = "flagged", f"partner's strong measurement {d['p_offset_s']:+.1f} s is outside the model"
        weak_note = ""
        if used is None and status is None:
            if (cl["median"] is not None and cl["n"] >= 3 and (cl["units"] >= 2 or cl["n_strong"] >= 1)):
                if dist(cl["median"]) <= ALIGNED_TOL_S:
                    used, src = cl["median"], "local_consensus"
                elif cl["n_strong"] >= 1:
                    status, reason = "flagged", f"local consensus {cl['median']:+.1f} s (with a strong member) is outside the model"
                else:     # a cluster of weak windows only: not evidence against the file's strong measurements
                    weak_note = (f"weak-only local cluster at {cl['median']:+.1f} s (n={cl['n']}, units={cl['units']}, "
                                 f"0 strong) ignored; ")
            if used is None and status is None:
                # no usable measurement near the clip: fall back on the file, if every strong measurement on it (clips,
                # partners, scan windows, both boxes) agrees with the model
                fstrong = [x[1] for x in pool[v["name"]] if x[3]]
                if len(fstrong) >= 3 and all(dist(x) <= ALIGNED_TOL_S for x in fstrong):
                    used, src = float(np.median(fstrong)), "file_strong_all_agree"
                    reason = weak_note
                else:
                    status, reason = "unmeasured", (weak_note + f"weak own measurement (r={RC._f(d['offset_r'])}), no "
                                                    f"strong partner, no local consensus (n={cl['n']}, units="
                                                    f"{cl['units']}), file strong n={len(fstrong)}")
        if used is not None:
            status = "aligned"
            reason = (reason or "") + f"{src} {used:+.2f} s within {ALIGNED_TOL_S} s of {d['expected_interval']}"
            if src == "file_strong_all_agree":
                reason += f" ({d['file_n_strong']} strong measurements on {v['name']}, max |offset| {d['file_max_abs_s']:.2f} s)"
            if sub != "normal":
                status = "needs_correction"
        d.update(offset_used_s=used, offset_source=src, status=status, reason=reason)
        rows.append(d)
    write_csv("measure/decisions.csv", DEC_COLS, rows)
    print("status:", dict(Counter((r["status"], r["offset_source"]) for r in rows)))
    print("by kind:", dict(Counter((r["kind"], r["status"]) for r in rows)))
    fl = [r for r in rows if r["status"] in ("flagged", "needs_correction")]
    for r in fl:
        print("  ", r["rel"], r["status"], r["reason"])
    print("files:", {f: s for f, s in sorted(file_stats.items()) if s["n"]})


def cmd_recut(args):
    dec = read_csv(os.path.join(OUT, "measure", "decisions.csv"))
    todo = [d for d in dec if d["status"] == "needs_correction"]
    print(f"{len(todo)} clips need a timing correction")
    if todo:
        raise SystemExit("corrections are not expected for these files; inspect measure/decisions.csv before "
                         "implementing them (recut_clips.cmd_decide shows the file-boundary logic)")


# ============================================================================ 8. label sanity
def read_clip_eeg(path):
    """The builder's channel (first label containing EEG, else ECG, as build_stage_segments_pooled.one()),
    physical, and its sampling rate."""
    h = AC.edf_header(path)
    labs = h["labels"]
    eeg = [i for i, l in enumerate(labs) if "EEG" in l.upper()]
    ecg = [i for i, l in enumerate(labs) if "ECG" in l.upper()]
    ch = (eeg or ecg)[0]
    x, _ = AC.edf_read(h, ch, 0, h["nrec_file"] * h["nsamp"][ch])
    return x, h["nsamp"][ch] / h["rdur"], labs[ch]


def ll_ratio(x, fs, sz_rel0, sz_rel1):
    """Line length (mean |diff|) over the labelled span / over the pre-buffer [0, onset)."""
    import numpy as np
    a, b = int(round(sz_rel0 * fs)), int(round(sz_rel1 * fs))
    pre = x[:a]
    sz = x[a:min(b, len(x))]
    if len(pre) < fs * 5 or len(sz) < fs * 5:
        return None
    return float(np.mean(np.abs(np.diff(sz))) / max(np.mean(np.abs(np.diff(pre))), 1e-12))


def fold_of_animals(seed=49, n_folds=5):
    """train_pooled.split_subjects: subjects sorted, random.Random(seed).shuffle, groups subs[k::n_folds]."""
    subs = sorted(LABELLED)
    rng = random.Random(seed)
    rng.shuffle(subs)
    return {s: k for k in range(n_folds) for s in subs[k::n_folds]}


def cmd_sanity(args):
    """Label sanity per new clip: the TCN detector (needs torch: the eeg env) and the line-length ratio; then the
    reference distributions from the existing clips (numpy only; --ref_only redoes just that part)."""
    import numpy as np
    folds = fold_of_animals()
    clips, _ = load_plan(include_controls=False)
    # reference: the stored OOF epoch-30 detector posteriors of every existing clip (3 seeds, the animal's own fold)
    ref = defaultdict(dict)
    for f in range(5):
        for s in (1, 2, 3):
            z = np.load(os.path.join(ESUBJ, f"tcn_bin_fold{f}_s{s}", "val_clip_ep30.npz"), allow_pickle=True)
            for pth, p5, y5 in zip(z["path"], z["p_logmean"], z["y5"]):
                ref[str(pth)][s] = (float(p5), int(y5))
    ref_rows = []
    for pth, d in ref.items():
        if len(d) == 3:
            ref_rows.append(dict(path=pth, y5=list(d.values())[0][1], p=float(np.mean([v[0] for v in d.values()])),
                                 animal=pth.split("/")[1].split("_")[1]))
    seen_fold = defaultdict(set)
    for f in range(5):
        z = np.load(os.path.join(ESUBJ, f"tcn_bin_fold{f}_s1", "val_clip_ep30.npz"), allow_pickle=True)
        for pth in z["path"]:
            seen_fold[str(pth).split("/")[1].split("_")[1]].add(f)
    bad = {a: fs_ for a, fs_ in seen_fold.items() if fs_ != {folds[a]}}
    if bad:
        raise SystemExit(f"split_subjects(seed 49) replica disagrees with the stored OOF folds: {bad}")
    if not args.ref_only:
        score_new_clips(clips, folds, args, ref)
    sanity_reference(clips, ref, ref_rows, args)


def score_new_clips(clips, folds, args, ref):
    import numpy as np
    sys.path.insert(0, os.path.join(REPO, "grader"))
    import scan_false_alarms as SF
    import torch
    torch.set_num_threads(args.threads)
    runs = [SF.eeg_run(s, f) for f in range(5) for s in (1, 2, 3)]
    nets = SF.EEGNets(runs, torch.device("cpu"))
    out = []
    t0 = time.time()
    for c in clips:
        d = os.path.join(OUT, "data_full", c["rel"])
        x, fs, lab = read_clip_eeg(os.path.join(d, "eeg.edf"))
        cs = PD(str(c["clip_start"]))
        row = dict(rel=c["rel"], animal=c["animal"], kind=c["kind"], y=c["y"], label=c["label"], channel=lab,
                   fold=folds[c["animal"]], dur_s=len(x) / fs)
        if c["kind"] == "seizure":
            r0 = (PD(str(c["sz_start"])) - cs).total_seconds()
            r1 = (PD(str(c["sz_end"])) - cs).total_seconds()
            row["ll_ratio"] = ll_ratio(x, fs, r0, r1)
        if fs != 1000:
            raise SystemExit(f"{c['rel']}: EEG at {fs} Hz")
        segs = SF.eeg_preprocess(x.astype(np.float64))
        row["n_windows"] = len(segs)
        ps = {}
        for run_ in runs:
            Pw = nets.predict_windows(segs, run_)
            ps[run_] = float(nets.clip_scores(Pw, np.array([0, len(segs)]))[0]) if len(segs) else float("nan")
        own = [ps[SF.eeg_run(s, folds[c["animal"]])] for s in (1, 2, 3)]
        row.update(p_oof_mean=float(np.mean(own)), p_oof_min=float(np.min(own)), p_oof_max=float(np.max(own)),
                   p_all15_mean=float(np.mean(list(ps.values()))), p_runs=json.dumps(ps))
        out.append(row)
    print(f"scored {len(out)} new clips in {time.time() - t0:.0f} s", flush=True)
    # scoring-path check: the 5 controls are existing clips with stored OOF posteriors (GPU, epoch 30)
    ctl = []
    for c in read_jsonl("plan/controls.jsonl"):
        x, fs, lab = read_clip_eeg(os.path.join(OUT, "repro", "data_full", c["rel"], "eeg.edf"))
        segs = SF.eeg_preprocess(x.astype(np.float64))
        f = folds[c["animal"]]
        for s_ in (1, 2, 3):
            Pw = nets.predict_windows(segs, SF.eeg_run(s_, f))
            p = float(nets.clip_scores(Pw, np.array([0, len(segs)]))[0])
            st = ref.get("data/" + c["rel"], {}).get(s_)
            ctl.append(dict(rel=c["rel"], fold=f, seed=s_, p_cpu=p, p_stored=st[0] if st else None,
                            abs_diff=abs(p - st[0]) if st else None))
    write_csv("sanity/controls.csv", ["rel", "fold", "seed", "p_cpu", "p_stored", "abs_diff"], ctl)
    print("controls: max |p_cpu - p_stored| =", max((r["abs_diff"] for r in ctl if r["abs_diff"] is not None),
                                                   default=None), flush=True)
    write_csv("sanity/clips.csv", ["rel", "animal", "kind", "y", "label", "channel", "fold", "dur_s", "n_windows",
                                   "ll_ratio", "p_oof_mean", "p_oof_min", "p_oof_max", "p_all15_mean", "p_runs"], out)


def sanity_reference(clips, ref, ref_rows, args):
    import numpy as np
    # LL-ratio reference: existing seizure clips of the same animals (data_full eeg.edf + info.txt), a fixed sample
    rng = random.Random(20260929)
    ll_ref = []
    for a in sorted({c["animal"] for c in clips}):
        dirs = sorted(glob.glob(os.path.join(FULL, f"Data_{a}_cropped", "*", "seizure_*")))
        dirs = [d_ for d_ in dirs if os.path.exists(os.path.join(d_, "eeg.edf")) and
                os.path.exists(os.path.join(d_, "info.txt"))]
        for d_ in rng.sample(dirs, min(len(dirs), args.ref_per_animal)):
            info = AC.parse_info(os.path.join(d_, "info.txt"))
            kv = {k.strip(): v.strip() for k, v in (l.split(":", 1) for l in open(os.path.join(d_, "info.txt"))
                                                     if ":" in l)}
            se = PD(kv.get("Seizure end", ""))
            if info["info_clip_start"] is None or info["info_sz_start"] is None or se is None:
                continue
            try:
                x, fs, _ = read_clip_eeg(os.path.join(d_, "eeg.edf"))
            except Exception:
                continue
            rr = ll_ratio(x, fs, (info["info_sz_start"] - info["info_clip_start"]).total_seconds(),
                          (se - info["info_clip_start"]).total_seconds())
            key = "data/" + os.path.relpath(d_, FULL)
            pr = ref.get(key)
            ll_ref.append(dict(path=key, animal=a, label=info["info_label"], ll_ratio=rr,
                               p_oof_mean=float(np.mean([v[0] for v in pr.values()])) if pr and len(pr) == 3 else None))
    write_csv("sanity/ll_reference.csv", ["path", "animal", "label", "ll_ratio", "p_oof_mean"], ll_ref)
    ref_sz = [r for r in ref_rows if r["y5"] > 0]
    ref_ns = [r for r in ref_rows if r["y5"] == 0]
    new = read_csv(os.path.join(OUT, "sanity", "clips.csv"))
    animals = sorted({c["animal"] for c in clips})
    per_animal = {}
    for a in animals:
        rs = [r for r in ref_sz if r["animal"] == a]
        rn = [r for r in ref_ns if r["animal"] == a]
        ns = [r for r in new if r["animal"] == a and r["kind"] == "seizure"]
        nn = [r for r in new if r["animal"] == a and r["kind"] == "nonseizure"]
        lr = [float(r["ll_ratio"]) for r in ll_ref if r["animal"] == a and fnum(r.get("ll_ratio")) is not None]
        per_animal[a] = dict(existing_sz=len(rs), existing_sz_p_below_05=sum(1 for r in rs if r["p"] < 0.5),
                             new_sz=len(ns), new_sz_p_below_05=sum(1 for r in ns if float(r["p_oof_mean"]) < 0.5),
                             existing_neg=len(rn), existing_neg_p_above_05=sum(1 for r in rn if r["p"] >= 0.5),
                             new_neg=len(nn), new_neg_p_above_05=sum(1 for r in nn if float(r["p_oof_mean"]) >= 0.5),
                             ll_ref_n=len(lr), ll_ref_median=float(np.median(lr)) if lr else None,
                             ll_new_median=float(np.median([float(r["ll_ratio"]) for r in ns if r["ll_ratio"]]))
                             if ns else None)
    llall = [float(r["ll_ratio"]) for r in ll_ref if fnum(r.get("ll_ratio")) is not None]
    js = dict(ref_seizure_n=len(ref_sz), ref_seizure_p_below_05=sum(1 for r in ref_sz if r["p"] < 0.5),
              ref_nonseizure_n=len(ref_ns), ref_nonseizure_p_above_05=sum(1 for r in ref_ns if r["p"] >= 0.5),
              ll_ref_n=len(llall), ll_ref_p05=float(np.percentile(llall, 5)), ll_ref_p10=float(np.percentile(llall, 10)),
              ll_ref_median=float(np.median(llall)),
              ll_ref_both_low=sum(1 for r in ll_ref if fnum(r.get("ll_ratio")) is not None and
                                  float(r["ll_ratio"]) < np.percentile(llall, 10) and r.get("p_oof_mean") not in (None, "")
                                  and float(r["p_oof_mean"]) < 0.5),
              per_animal=per_animal)
    write_text("sanity/reference.json", json.dumps(js, indent=1))
    print(json.dumps(js))


def cmd_context(args):
    """Where is the EEG seizure? For every new seizure clip whose OOF detector P(sz) < 0.5, and every new negative
    with P >= 0.5: the animal's 3 OOF TCNs over the raw EDF, onset -600..+600 s, 6-s windows every 3 s (the
    builder's preprocessing), smoothed over 5 windows; the longest run of window P >= 0.5 near the label."""
    import numpy as np
    sys.path.insert(0, os.path.join(REPO, "grader"))
    import scan_false_alarms as SF
    import torch
    torch.set_num_threads(args.threads)
    folds = fold_of_animals()
    clips, _ = load_plan(include_controls=False)
    san = {r["rel"]: r for r in read_csv(os.path.join(OUT, "sanity", "clips.csv"))}
    todo = [c for c in clips if (c["kind"] == "seizure" and float(san[c["rel"]]["p_oof_mean"]) < 0.5) or
            (c["kind"] == "nonseizure" and float(san[c["rel"]]["p_oof_mean"]) >= 0.5)]
    runs = sorted({SF.eeg_run(s_, folds[c["animal"]]) for c in todo for s_ in (1, 2, 3)})
    nets = SF.EEGNets(runs, torch.device("cpu"))
    out = []
    for c in todo:
        h = AC.edf_header(c["edf"]["fpath"])
        ch = AC.edf_pick(h)
        on = PD(str(c["sz_start"])) if c["kind"] == "seizure" else PD(str(c["clip_start"]))
        t0 = (on - h["start"]).total_seconds() - 600
        s0 = int(round(max(0.0, t0) * 1000))
        x, _ = AC.edf_read(h, ch, s0, s0 + 1200 * 1000)
        segs = SF.eeg_preprocess(x)
        P = np.mean([nets.predict_windows(segs, SF.eeg_run(s_, folds[c["animal"]]))[:, 1] for s_ in (1, 2, 3)], 0)
        tw = s0 / 1000 + 3.0 * np.arange(len(P)) + 3.0 - (on - h["start"]).total_seconds()   # window centre - onset
        Ps = np.convolve(P, np.ones(5) / 5, mode="same")
        dur = float(c["sz_dur"]) if c["kind"] == "seizure" else float(c["clip_dur_s"])
        inside = (tw >= 0) & (tw <= dur)
        runs_ = []
        on_ = Ps >= 0.5
        i = 0
        while i < len(on_):
            if on_[i]:
                j = i
                while j + 1 < len(on_) and on_[j + 1]:
                    j += 1
                runs_.append((tw[i], tw[j], float(Ps[i:j + 1].max())))
                i = j + 1
            else:
                i += 1
        best = max(runs_, key=lambda r: r[1] - r[0]) if runs_ else None
        near = min(runs_, key=lambda r: 0 if r[0] <= 0 <= r[1] else min(abs(r[0]), abs(r[1]))) if runs_ else None
        out.append(dict(rel=c["rel"], kind=c["kind"], label=c["label"], dur_s=dur,
                        p_clip_oof=float(san[c["rel"]]["p_oof_mean"]), p_inside_mean=float(P[inside].mean()) if inside.any() else None,
                        p_inside_max=float(P[inside].max()) if inside.any() else None,
                        n_runs=len(runs_), longest_run=f"{best[0]:+.0f}..{best[1]:+.0f} s (max {best[2]:.2f})" if best else "",
                        nearest_run=f"{near[0]:+.0f}..{near[1]:+.0f} s (max {near[2]:.2f})" if near else "",
                        profile_1min=" ".join(f"{float(P[(tw >= a) & (tw < a + 60)].max()):.2f}" if ((tw >= a) & (tw < a + 60)).any()
                                              else "nan" for a in range(-600, 600, 60))))
        print(out[-1]["rel"][-50:], out[-1]["label"], f"P in span {out[-1]['p_inside_mean']}", "nearest", out[-1]["nearest_run"],
              "longest", out[-1]["longest_run"], flush=True)
    write_csv("sanity/context.csv", ["rel", "kind", "label", "dur_s", "p_clip_oof", "p_inside_mean", "p_inside_max",
                                     "n_runs", "longest_run", "nearest_run", "profile_1min"], out)


# ============================================================================ 9. verify
VER_COLS = ["rel", "control", "kind", "animal", "decodes", "dec_errors", "w", "h", "box_w", "box_h", "crop_dims_ok",
            "nb", "nb_expected", "nb_intended", "frames_as_expected", "within_keyframe", "keyframe_pts", "seek_s",
            "v_clip_dur_s", "ncc_mean", "ncc_min", "ncc_best_minus_key_frames", "frames_match_raw", "eeg_ok",
            "eeg_nrec", "eeg_nrec_expected", "eeg_ch", "eeg_lag_samples", "eeg_max_abs_diff", "eeg_quant_step",
            "eeg_corr", "eeg_aux_zero", "eeg_labels", "err"]


def eeg_vs_raw(clip_edf, src_edf, clip_start, clip_end, edf_start, edf_dur_s):
    """Every 1-kHz channel of the clip EDF against the source EDF at sample round((clip start - EDF start) * fs), with
    a +-2 s lag search on the builder's channel; max |diff| is compared with the clip's own quantization step."""
    import numpy as np
    hc, hs = AC.edf_header(clip_edf), AC.edf_header(src_edf)
    res = dict(eeg_nrec=hc["nrec_file"], eeg_ch=hc["ns"], eeg_labels="|".join(hc["labels"]))
    t0 = (clip_start - edf_start).total_seconds()
    s0 = int(round(t0 * 1000))
    worst, cors, quant, aux_zero = 0.0, [], 0.0, True
    lag = None
    for k in range(hc["ns"]):
        fs_c = hc["nsamp"][k] / hc["rdur"]
        fs_s = hs["nsamp"][k] / hs["rdur"]
        xc, _ = AC.edf_read(hc, k, 0, hc["nrec_file"] * hc["nsamp"][k])
        if fs_s < 1000:          # Activity / SignalStr / Temp: MNE 1.12.1 gives zeros for them (parent format)
            aux_zero = aux_zero and bool(np.all(np.abs(xc) < 1e-9))
            continue
        xs, _ = AC.edf_read(hs, k, s0, s0 + len(xc))
        q = (hc["pmax"][k] - hc["pmin"][k]) / (hc["dmax"][k] - hc["dmin"][k])
        quant = max(quant, q)
        worst = max(worst, float(np.abs(xs - xc).max()) / q)
        cors.append(float(np.corrcoef(xs, xc)[0, 1]))
        if lag is None:     # the builder's channel is the first 1-kHz channel
            y, s0b = AC.edf_read(hs, k, max(0, s0 - 2000), s0 + len(xc) + 2000)
            r = AC.ncc_1d(xc, y)
            lag = int(np.argmax(r)) - (s0 - s0b)
    # mne crop keeps samples round(t_start*fs) .. round(t_end*fs) inclusive; the writer keeps whole 1-s records
    t_s = max(0.0, (clip_start - edf_start).total_seconds())
    t_e = min(edf_dur_s, (clip_end - edf_start).total_seconds())
    res.update(eeg_lag_samples=lag, eeg_max_abs_diff=worst, eeg_quant_step=quant, eeg_corr=min(cors) if cors else None,
               eeg_aux_zero=aux_zero, eeg_nrec_expected=(int(round(t_e * 1000)) - int(round(t_s * 1000)) + 1) // 1000)
    res["eeg_ok"] = bool(lag == 0 and worst <= 0.51 and res["eeg_nrec"] == res["eeg_nrec_expected"])
    return res


def cmd_verify(args):
    raw = RC.Raw()
    clips, ctr = load_plan()
    dec = {r["rel"]: r for r in read_csv(os.path.join(OUT, "measure", "decisions.csv"))} \
        if os.path.exists(os.path.join(OUT, "measure", "decisions.csv")) else {}

    def one(c):
        dpath, work = clip_paths(c)
        d = os.path.dirname(dpath)
        p = os.path.join(d, "video.mp4")
        row = dict(rel=c["rel"], control=bool(c.get("control")), kind=c["kind"], animal=c["animal"])
        try:
            pr = RC.probe(p)
            x, y, bw, bh = RC.crop_box(c["animal"])
            src, seek, vdur = video_source(c)
            key = RC.keyframe_at_or_before(src, seek)
            exp = (seek - (key if key is not None else seek) + vdur) * 15
            row.update(decodes=pr["nread"] > 0 and pr["dec_errors"] == 0, dec_errors=pr["dec_errors"], w=pr["w"],
                       h=pr["h"], box_w=bw, box_h=bh, crop_dims_ok=(pr["w"], pr["h"]) == (bw, bh), nb=pr["nread"],
                       nb_expected=exp, nb_intended=vdur * 15, frames_as_expected=abs(pr["nread"] - exp) <= 2,
                       within_keyframe=-2 <= pr["nread"] - vdur * 15 <= 34,
                       keyframe_pts=key, seek_s=seek, v_clip_dur_s=vdur)
            nc = RC.ncc_vs_raw(p, src, key if key is not None else seek, c["animal"], pr["dur"])
            row.update({k: nc.get(k) for k in ("ncc_mean", "ncc_min", "ncc_best_minus_key_frames")})
            dk = nc.get("ncc_best_minus_key_frames")
            row["frames_match_raw"] = bool((nc.get("ncc_mean") or 0) >= 0.95 and dk is not None and abs(dk) <= 1)
            row.update(eeg_vs_raw(os.path.join(d, "eeg.edf"), c["edf"]["fpath"], PD(str(c["clip_start"])),
                                  PD(str(c["clip_end"])), PD(str(c["edf"]["start"])), float(c["edf"]["dur_s"])))
        except Exception as e:  # noqa
            row["err"] = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        return row

    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        rows = list(ex.map(one, clips + ctr))
    write_csv("verify/verify.csv", VER_COLS, rows)
    for k in ("decodes", "crop_dims_ok", "frames_as_expected", "within_keyframe", "frames_match_raw", "eeg_ok",
              "eeg_aux_zero"):
        print(f"  {k}: {dict(Counter(str(r.get(k)) for r in rows))}")
    print(f"verify: {len(rows)} in {time.time() - t0:.0f} s", flush=True)

    # the 5 controls against the existing data_full clips: video frames and the EEG bytes
    rr = []
    for c in ctr:
        d = os.path.join(OUT, "repro", "data_full", c["rel"])
        o = os.path.join(FULL, c["rel"])
        pa, pb = RC.probe(os.path.join(d, "video.mp4")), RC.probe(os.path.join(o, "video.mp4"))
        row = dict(rel=c["rel"], kind=c["kind"], nb=pa["nread"], orig_nb=pb["nread"], w=pa["w"], h=pa["h"],
                   orig_w=pb["w"], orig_h=pb["h"])
        row.update(RC.compare_clips(os.path.join(d, "video.mp4"), os.path.join(o, "video.mp4"), pa["w"], pa["h"]))
        a = open(os.path.join(d, "eeg_parentlabels.edf"), "rb").read()
        b = open(os.path.join(o, "eeg.edf"), "rb").read()
        f = open(os.path.join(d, "eeg.edf"), "rb").read()
        row.update(eeg_parent_labels_identical=a == b, eeg_bytes=len(b),
                   eeg_fixed_labels_ndiff_bytes=sum(1 for x, y in zip(f, b) if x != y) + abs(len(f) - len(b)),
                   info_identical=open(os.path.join(d, "info.txt")).read() == open(os.path.join(o, "info.txt")).read())
        rr.append(row)
        print("control", json.dumps(row, default=str), flush=True)
    write_csv("verify/controls.csv", ["rel", "kind", "nb", "orig_nb", "w", "h", "orig_w", "orig_h", "frame_shift",
                                      "ncc_mean", "ncc_min", "mae_mean", "mae_max", "n_compared",
                                      "eeg_parent_labels_identical", "eeg_bytes", "eeg_fixed_labels_ndiff_bytes",
                                      "info_identical"], rr)


# ============================================================================ 10. report (manifest, summary, integration)
MAN_COLS = ["key", "animal", "session", "clip_dir", "kind", "y", "stage", "label", "onset", "seizure_end",
            "duration_s", "clip_start", "clip_end", "clip_dur_s", "xlsx", "xlsx_row", "parent_idx", "sz_idx", "clip_n",
            "paired_seizure", "existing_negative", "raw_video", "video_src", "seek_s", "keyframe_pts", "v_clip_dur_s",
            "truncated_s", "raw_edf", "edf_start", "edf_offset_s", "timing_cause", "correction_applied_s",
            "measured_offset_s", "measured_r", "measured_strong", "partner_offset_s", "offset_used_s",
            "offset_source", "alignment", "alignment_reason", "ll_ratio", "p_eeg_oof", "p_eeg_all15", "label_flags", "verify_ok",
            "deviations", "status", "note"]


def label_checks(clips, sanity, ll_ref):
    """Per new clip: flags from the EEG (line length, detector), duplicates across clock changes, overlaps."""
    import numpy as np
    events = read_csv(EVENTS_CSV)
    ev_by_a = defaultdict(list)
    for e in events:
        s = PD(e["start"])
        if s is None:
            continue
        en = PD(e["end"]) or s + dt.timedelta(seconds=float(e["duration_s"] or 0))
        ev_by_a[e["animal"]].append(dict(start=s, end=en, label=e["label"], dur=float(e["duration_s"] or 0),
                                         xlsx=e["xlsx"], row=e["row"]))
    # existing clip windows per animal (info.txt clip start/end) for overlap checks
    clipped = read_csv(os.path.join(AUDIT, "align", "clips.csv"))
    win_by_a = defaultdict(list)
    for r in clipped:
        cs, ce = PD(r.get("info_clip_start")), PD(r.get("info_clip_end"))
        if cs and ce:
            win_by_a[r["animal"]].append((cs, ce, f"{r['session']}/{r['clip_dir']}", r["kind"]))
    new_win = defaultdict(list)
    for c in clips:
        new_win[c["animal"]].append((PD(str(c["clip_start"])), PD(str(c["clip_end"])), f"{c['session']}/{c['clip_dir']}",
                                     c["kind"]))
    llr = [r["ll_ratio"] for r in ll_ref if fnum(r.get("ll_ratio")) is not None]
    cpath = os.path.join(OUT, "sanity", "context.csv")
    ctx = {r["rel"]: r for r in read_csv(cpath)} if os.path.exists(cpath) else {}
    ll_p05 = float(np.percentile([float(x) for x in llr], 5)) if llr else None
    ll_p10 = float(np.percentile([float(x) for x in llr], 10)) if llr else None
    CLOCK_STEPS = [("clock_step_60s", 60.4, 3.0), ("dst_3600s", 3600.0, 6.0), ("roomD_10523s", 10523.3, 10.0),
                   ("roomC_13s", 13.0, 2.0)]
    out = {}
    for c in clips:
        fl = []
        s = sanity.get(c["rel"], {})
        on, en = PD(str(c["sz_start"])), PD(str(c["sz_end"]))
        cx = ctx.get(c["rel"])
        if c["kind"] == "seizure":
            p = fnum(s.get("p_oof_mean"))
            lr = fnum(s.get("ll_ratio"))
            eeg_no = p is not None and p < 0.5
            ll_no = lr is not None and ll_p10 is not None and lr < ll_p10
            # clip-level rule (set first): both detectors low. The window-level context (cmd_context) then decides:
            # a detector run (5-window-smoothed P >= 0.5) inside the labelled span = the EEG shows a seizure, shorter
            # than the label; no run and max window P < 0.5 inside a span >= 10 s long = EEG-disputed (step 1's 5).
            if eeg_no and cx is not None:
                run_in = cx.get("nearest_run") and any(
                    float(a) <= float(c["sz_dur"]) and float(b) >= 0 for a, b in
                    [re.match(r"([+-]?\d+)\.\.([+-]?\d+)", cx["nearest_run"]).groups()])
                pmax = fnum(cx.get("p_inside_max"))
                if run_in:
                    fl.append(f"eeg_seizure_shorter_than_label(p_oof={p:.2f}, detector run {cx['nearest_run']} from "
                              f"onset" + (f", ll_ratio={lr:.2f}<p10" if ll_no else "") + ")")
                elif float(c["sz_dur"]) >= 10 and pmax is not None and pmax < 0.5:
                    fl.append(f"EEG_DISPUTED(p_oof={p:.2f}, no detector run within +-600 s of onset reaches the span, "
                              f"max window P in span {pmax:.2f}, ll_ratio={lr:.2f})")
                else:
                    fl.append(f"eeg_detector_low(p_oof={p:.2f}, max window P in span {pmax:.2f}, "
                              f"labelled duration {float(c['sz_dur']):.1f} s" + (", too short for a 6-s window at 80%"
                                                                               if float(c["sz_dur"]) < 10 else "") + ")")
                if eeg_no and ll_no:
                    fl.append(f"clip_rule_both_low(p_oof={p:.2f},ll_ratio={lr:.2f}<p10 {ll_p10:.2f})")
            elif eeg_no:
                fl.append(f"eeg_detector_low(p_oof={p:.2f})")
            elif ll_no:
                fl.append(f"ll_ratio_low({lr:.2f}<p10 {ll_p10:.2f})")
            # duplicates of a neighbouring event across a clock change (same animal, any xlsx)
            for e in ev_by_a[c["animal"]]:
                dts = (e["start"] - on).total_seconds()
                if abs(dts) < 1e-3:
                    continue
                for nm, stp, tol in CLOCK_STEPS:
                    if abs(abs(dts) - stp) <= tol and abs(e["dur"] - float(c["sz_dur"])) <= max(10.0, 0.3 * float(c["sz_dur"])):
                        fl.append(f"possible_duplicate_{nm}(other event {e['label']} at {e['start']}, {dts:+.1f} s, "
                                  f"dur {e['dur']:.1f} vs {float(c['sz_dur']):.1f})")
            # overlap with another annotated event of the same animal (a second label on the same seizure)
            for e in ev_by_a[c["animal"]]:
                if e["start"] == on:
                    continue
                if e["start"] < en and e["end"] > on:
                    fl.append(f"overlaps_event({e['label']} {e['start']}..{e['end']})")
        else:
            cs, ce = PD(str(c["clip_start"])), PD(str(c["clip_end"]))
            for e in ev_by_a[c["animal"]]:
                if e["start"] - dt.timedelta(seconds=40) < ce and e["end"] + dt.timedelta(seconds=40) > cs:
                    fl.append(f"NEGATIVE_NEAR_EVENT({e['label']} {e['start']})")
            p = fnum(s.get("p_oof_mean"))
            if p is not None and p >= 0.5:
                if cx is not None and cx.get("nearest_run"):
                    a, b = (float(x) for x in re.match(r"([+-]?\d+)\.\.([+-]?\d+)", cx["nearest_run"]).groups())
                    if a <= float(c["clip_dur_s"]) and b >= 0 and b - a >= 15:
                        fl.append(f"NEGATIVE_EEG_SEIZURE_LIKE(p_oof={p:.2f}, detector run {cx['nearest_run']} from clip "
                                  f"start: an unannotated seizure?)")
                    else:
                        fl.append(f"negative_eeg_detector_high(p_oof={p:.2f})")
                else:
                    fl.append(f"negative_eeg_detector_high(p_oof={p:.2f})")
        cs, ce = PD(str(c["clip_start"])), PD(str(c["clip_end"]))
        for (a0, a1, nm, kd) in win_by_a[c["animal"]] + [w for w in new_win[c["animal"]] if w[2] != f"{c['session']}/{c['clip_dir']}"]:
            if a0 < ce and a1 > cs:
                if c["kind"] == "seizure" and kd == "nonseizure":
                    fl.append(f"existing_negative_contains_this_seizure({nm})")
                elif c["kind"] == "nonseizure" and kd == "seizure":
                    fl.append(f"NEGATIVE_OVERLAPS_SEIZURE_CLIP({nm})")
                elif c["kind"] == "seizure" and kd == "seizure":
                    fl.append(f"overlaps_seizure_clip({nm})")
        out[c["rel"]] = fl
    return out, dict(ll_p05=ll_p05, ll_p10=ll_p10, ll_ref_n=len(llr))


def cmd_report(args):
    import numpy as np
    clips, ctr = load_plan()
    excl = read_jsonl("plan/excluded.jsonl")
    ms = {r["rel"]: r for r in read_csv(os.path.join(OUT, "measure", "clips.csv"))}
    dec = {r["rel"]: r for r in read_csv(os.path.join(OUT, "measure", "decisions.csv"))}
    ver = {r["rel"]: r for r in read_csv(os.path.join(OUT, "verify", "verify.csv"))}
    san = {r["rel"]: r for r in read_csv(os.path.join(OUT, "sanity", "clips.csv"))} \
        if os.path.exists(os.path.join(OUT, "sanity", "clips.csv")) else {}
    ll_ref = read_csv(os.path.join(OUT, "sanity", "ll_reference.csv")) \
        if os.path.exists(os.path.join(OUT, "sanity", "ll_reference.csv")) else []
    flags, llinfo = label_checks(clips, san, ll_ref)
    rows = []
    for c in clips:
        m, d, v, s = ms.get(c["rel"], {}), dec.get(c["rel"], {}), ver.get(c["rel"], {}), san.get(c["rel"], {})
        vok = all(v.get(k) == "True" for k in ("decodes", "crop_dims_ok", "frames_as_expected", "frames_match_raw",
                                                "eeg_ok"))
        fl = flags.get(c["rel"], [])
        hard = [f for f in fl if f.split("(")[0].isupper()]
        status = "ok"
        if not vok:
            status = "verify_failed"
        elif d.get("status") in ("flagged", "needs_correction"):
            status = "timing_flagged"
        elif hard:
            status = "label_flagged"
        cs = PD(str(c["clip_start"]))
        rows.append(dict(
            key=c["key"], animal=c["animal"], session=c["session"], clip_dir=c["clip_dir"], kind=c["kind"], y=c["y"],
            stage=c["label"] if c["kind"] == "seizure" else "non-seizure", label=c["label"], onset=c["sz_start"],
            seizure_end=c["sz_end"], duration_s=c["sz_dur"], clip_start=c["clip_start"], clip_end=c["clip_end"],
            clip_dur_s=c["clip_dur_s"], xlsx=c["xlsx"], xlsx_row=c["xlsx_row"], parent_idx=c["parent_idx"],
            sz_idx=c["sz_idx"], clip_n=c.get("clip_n"), paired_seizure=c.get("paired_seizure"),
            existing_negative=c.get("existing_negative") if c["kind"] == "seizure" else None,
            raw_video=c["video"]["relpath"], video_src=c.get("video_src"), seek_s=c["seek_s"],
            keyframe_pts=v.get("keyframe_pts"), v_clip_dur_s=c["v_clip_dur_s"], truncated_s=c["truncated_s"],
            raw_edf=c["edf"]["relpath"], edf_start=c["edf"]["start"],
            edf_offset_s=(cs - PD(str(c["edf"]["start"]))).total_seconds(), timing_cause=c["timing_cause"],
            correction_applied_s=0.0, measured_offset_s=m.get("offset_s"), measured_r=m.get("offset_r"),
            measured_strong=m.get("strong"), partner_offset_s=m.get("p_offset_s"),
            offset_used_s=d.get("offset_used_s"), offset_source=d.get("offset_source"), alignment=d.get("status"),
            alignment_reason=d.get("reason"),
            ll_ratio=s.get("ll_ratio"), p_eeg_oof=s.get("p_oof_mean"), p_eeg_all15=s.get("p_all15_mean"),
            label_flags=" ; ".join(fl), verify_ok=vok, deviations=" ; ".join(c.get("deviations") or []), status=status,
            note=c.get("negative_rule") or ""))
    for c in excl:
        rows.append(dict(key=key_of(c["animal"], c["session"], c["clip_dir"]), animal=c["animal"],
                         session=c["session"], clip_dir=c["clip_dir"], kind="seizure", stage=c["label"],
                         label=c["label"], onset=c["sz_start"], seizure_end=c["sz_end"], duration_s=c["sz_dur"],
                         clip_start=c["clip_start"], clip_end=c["clip_end"], xlsx=c["xlsx"], xlsx_row=c["xlsx_row"],
                         parent_idx=c["parent_idx"], sz_idx=c["sz_idx"], status="excluded", note=c["reason"],
                         raw_edf=c.get("edf", {}).get("relpath")))
    write_csv("manifest.csv", MAN_COLS, rows)
    items = write_items(clips)
    integ = write_integration(clips, rows)
    write_summary(clips, excl, rows, ms, dec, ver, san, ll_ref, llinfo, integ)
    print("manifest:", len(rows), dict(Counter((r["kind"], r["status"]) for r in rows)), f"items: {len(items)}")


NAMES5 = ["non-seizure", "Stage2", "Stage3", "Stage4", "Stage5"]
GROUP3 = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}


def write_items(clips):
    """The discover() tuples of the new clips (key, y, 'subj/sess', subj): an extra item list for a later retrain."""
    items = [dict(key=c["key"], y=c["y"], sess=f"{c['animal']}/{c['session']}", subj=c["animal"],
                  full_rel=f"output/ttg_newclips/data_full/{c['rel']}") for c in clips]
    write_csv("items.csv", ["key", "y", "sess", "subj", "full_rel"], items)
    return items


def split_sessions_seed(items, seed=49, val_frac=0.2, nc=5):
    """train_pooled.split_sessions (5-class NC): the first seed >= 49 whose val and train both hold every class."""
    sessions = sorted({i[2] for i in items})
    for s in range(seed, seed + 500):
        rng = random.Random(s)
        ss = sessions[:]
        rng.shuffle(ss)
        val = set(ss[:max(1, round(len(ss) * val_frac))])
        ctr = Counter(i[1] for i in items if i[2] not in val)
        cva = Counter(i[1] for i in items if i[2] in val)
        if all(ctr[c] > 0 for c in range(nc)) and all(cva[c] > 0 for c in range(nc)):
            return val, s
    raise SystemExit("no session split")


def eeg_windows(n_samples_1k, sz_rel, y):
    """build_stage_segments_pooled.one(): windows of 6 s every 3 s at 125 Hz, labelled y if >= 4.8 s in the seizure."""
    n = -(-n_samples_1k // 8)             # resample_poly(sig, 1, 8) output length
    sf, cs, ss = 125.0, 750, 375
    labs = []
    i = 0
    while i + cs <= n:
        w0, w1 = i / sf, (i + cs) / sf
        lab = 0
        if sz_rel is not None:
            ov = max(0.0, min(w1, sz_rel[1]) - max(w0, sz_rel[0]))
            if ov >= 0.8 * 6.0:
                lab = y
        labs.append(lab)
        i += ss
    return labs


def write_integration(clips, rows):
    clipped = read_csv(os.path.join(AUDIT, "align", "clips.csv"))
    disc = [(f"data/Data_{r['animal']}_cropped/{r['session']}/{r['clip_dir']}/video.mp4", int(r["y"]),
             f"{r['animal']}/{r['session']}", r["animal"], r.get("full_eeg_bytes") not in ("", "0", None))
            for r in clipped if r["in_discover"] == "True"]
    new = [(c["key"], int(c["y"]), f"{c['animal']}/{c['session']}", c["animal"], True) for c in clips]
    flagged = {r["key"] for r in rows if r.get("status") in ("label_flagged", "timing_flagged", "verify_failed")}
    folds = fold_of_animals()
    L = []
    P = L.append
    P("INTEGRATION NOTE: what a later retrain needs to include the new clips (nothing here was merged)")
    P("")
    P(f"new clips: {len(new)} = {dict(Counter(NAMES5[i[1]] for i in new))}; of them flagged in the manifest: {len(flagged)} "
      f"(1 more event was excluded and has no clip)")
    P(f"existing discover() items: {len(disc)}; with eeg.edf in data_full (the EEG side): {sum(1 for i in disc if i[4])}")
    P("")
    P("1. Class balance (clips; 5-class, and 3-class mild = S2+S3, severe = S4+S5)")
    b, a_ = Counter(i[1] for i in disc), Counter(i[1] for i in disc + new)
    for k in range(5):
        P(f"   {NAMES5[k]:12s} {b[k]:6d} ({b[k] / len(disc):6.2%}) -> {a_[k]:6d} ({a_[k] / (len(disc) + len(new)):6.2%})"
          f"   +{a_[k] - b[k]} ({(a_[k] - b[k]) / max(b[k], 1):+.1%})")
    b3, a3 = Counter(GROUP3[i[1]] for i in disc), Counter(GROUP3[i[1]] for i in disc + new)
    P(f"   3-class: non {b3[0]} -> {a3[0]}, mild {b3[1]} -> {a3[1]} (+{a3[1] - b3[1]}), severe {b3[2]} -> {a3[2]} "
      f"(+{a3[2] - b3[2]}); severe share of seizures {b3[2] / (b3[1] + b3[2]):.2%} -> {a3[2] / (a3[1] + a3[2]):.2%}")
    P(f"   seizure : non-seizure {sum(b[k] for k in range(1, 5))}:{b[0]} -> {sum(a_[k] for k in range(1, 5))}:{a_[0]} "
      f"(the new set adds 327 seizures and 178 negatives because 150 of the new seizures already had a negative)")
    P("")
    P("2. By fold (train_pooled.split_subjects seed 49; subject folds do not change: no new animal)")
    for f in range(5):
        va = sorted(a for a, k in folds.items() if k == f)
        bv = Counter(i[1] for i in disc if folds[i[3]] == f)
        nv = Counter(i[1] for i in new if folds[i[3]] == f)
        bt = Counter(i[1] for i in disc if folds[i[3]] != f)
        nt = Counter(i[1] for i in new if folds[i[3]] != f)
        P(f"   fold {f}: val animals {va}")
        P(f"      val   +{sum(nv.values()):3d}: " + ", ".join(f"{NAMES5[k]} {bv[k]}+{nv[k]}" for k in range(5)))
        P(f"      train +{sum(nt.values()):3d}: " + ", ".join(f"{NAMES5[k]} {bt[k]}+{nt[k]}" for k in range(5)))
    P("   new clips by animal and fold: " + ", ".join(f"{a} (fold {folds[a]}) {n}" for a, n in
                                                     sorted(Counter(i[3] for i in new).items())))
    P("")
    val_b, seed_b = split_sessions_seed([i[:4] for i in disc])
    val_a, seed_a = split_sessions_seed([i[:4] for i in disc + new])
    newsess = {i[2] for i in new} - {i[2] for i in disc}
    P("3. Session-disjoint split (train_pooled.split_sessions seed 49, 5-class NC; the aligned EEG split")
    P(f"   train_pooled_eeg.video_val_sessions uses the same universe): new sessions {len(newsess)}; seed used before "
      f"{seed_b}, after {seed_a}; val sessions {len(val_b)} -> {len(val_a)}, identical: {val_a == val_b}")
    nv = [i for i in new if i[2] in val_b]
    P(f"   new clips in the seed-49 val sessions: {len(nv)} {dict(Counter(NAMES5[i[1]] for i in nv))} "
      f"(sessions {sorted({i[2] for i in nv})})")
    P("")
    # EEG windows the builder would add
    wl = Counter()
    for c in clips:
        p = os.path.join(OUT, "data_full", c["rel"], "eeg.edf")
        if not os.path.exists(p):
            continue
        h = AC.edf_header(p)
        n = h["nrec_file"] * 1000
        cs = PD(str(c["clip_start"]))
        sz = ((PD(str(c["sz_start"])) - cs).total_seconds(), (PD(str(c["sz_end"])) - cs).total_seconds()) \
            if c["kind"] == "seizure" else None
        for lab in eeg_windows(n, sz, c["y"]):
            wl[lab] += 1
    P("4. EEG segment cache (build_stage_segments_pooled defaults: 6-s windows, 3-s stride, 125 Hz, label if >= 80% "
      "inside the seizure)")
    P(f"   windows added: {sum(wl.values())} by window label " + ", ".join(f"{NAMES5[k]} {wl[k]}" for k in range(5)))
    P("")
    P("5. What a retrain has to do (in this order)")
    P("   a. Fix labels first: drop or relabel the new clips flagged in manifest.csv (status label_flagged /")
    P("      timing_flagged / verify_failed) and apply step 1 (ttg_recut: 81 re-cut videos, 11 exclusions).")
    P("   b. Files. EITHER copy output/ttg_newclips/data_full/Data_<A>_cropped/<session>/<clip>/ into data_full/ (same")
    P("      relative paths; plan checked there is no name collision with any data/ or data_full/ dir) AND create the")
    P("      zero-byte key data/<same rel>/video.mp4, so that train_pooled.discover() and")
    P("      build_stage_segments_pooled (--root data_full) pick them up with no code change; OR append")
    P("      output/ttg_newclips/items.csv to discover()'s list (key = data/... path, y, 'subj/sess', subj) and map")
    P("      the keys to output/ttg_newclips/data_full/ wherever data_full/ is read (ttg_common.full_path,")
    P("      build_f16_cache, build_stage_segments_pooled --root). The first needs no special case; it is the")
    P("      recommended one.")
    P("   c. Frame caches. cache_frames/f16s224 and f32s224 are memmaps indexed by the discover() key list")
    P("      (index.json 'paths'); FrameCache falls back to decoding the data/ key for a path missing from the")
    P("      index, and data/ mp4s are zero-byte, so both caches must be rebuilt (grader/build_f16_cache.py decodes")
    P("      data_full/<rel> and keeps data/<rel> keys; the same for 32x224), or extended by appending rows and")
    P("      paths. Row order changes, so any consumer must key on path (CLAUDE.md contract), never on row.")
    P("   d. EEG segment caches: re-run build_stage_segments_pooled.py --root data_full with the knobs of every")
    P("      cache in use (cache_bestcfg/seg_w6.0_s3.0_d8.npz for the grader TCN, stage_segments_pooled_v3.npz,")
    P("      the 250-Hz one for v3_eeg250). The builder sorts by clip path, so clip_id values shift: key on path.")
    P("   e. Splits: subject folds unchanged; session universe unchanged (see 3), so the seed-49 session split and")
    P("      the aligned split keep their session sets, and their val sets grow by the new clips in val sessions.")
    P("      Stored val_preds / val_clip_preds do not contain the new clips: any comparison with an old run must")
    P("      restrict to the old clip set (align on path).")
    P("   f. Existing gaps you may want to close in the same pass: 45 discover() clips have no eeg.edf in data_full")
    P("      (25 of them also no info.txt: the clip being cut when the 90-min guard killed the cutter); this")
    P("      pipeline reproduces their specs exactly (plan/reproduce_existing.csv) and could write them.")
    write_text("integration.txt", "\n".join(L) + "\n")
    print("\n".join(L))
    return dict(new_by_class=dict(Counter(NAMES5[i[1]] for i in new)), windows=dict(wl),
                val_sessions_identical=val_a == val_b, new_in_session_val=len(nv))


def write_summary(clips, excl, rows, ms, dec, ver, san, ll_ref, llinfo, integ):
    import numpy as np
    rep = read_csv(os.path.join(OUT, "plan", "reproduce_existing.csv"))
    ctl = read_csv(os.path.join(OUT, "verify", "controls.csv"))
    L = []
    P = L.append
    P(f"ttg_newclips: unclipped annotated Stage 2-5 seizures cut as new labelled clips  ({time.strftime('%F %T')})")
    P("")
    P(f"candidates 328 -> seizure clips {sum(1 for c in clips if c['kind'] == 'seizure')}, excluded {len(excl)}; "
      f"new negatives {sum(1 for c in clips if c['kind'] == 'nonseizure')} (150 seizures reuse an existing negative)")
    P("by class: " + str(dict(Counter((c['kind'], c['label'] if c['kind'] == 'seizure' else '-') for c in clips))))
    P("by animal/session: " + str(dict(Counter(f"{c['animal']}/{c['session']}" for c in clips))))
    for c in excl:
        P(f"excluded: {c['animal']}/{c['session']}/{c['clip_dir']}: {c['reason']}")
    P("")
    rc = Counter((r["kind"], r["name_reproduced"], r["info_identical"]) for r in rep)
    P(f"parent reproduction (plan): {len(rep)} existing dirs in the 15 sessions; (kind, name reproduced, info.txt "
      f"byte-identical): {dict(rc)}  (None = no info.txt: the clip being cut when the cutter was killed)")
    P("")
    newrel = {c["rel"] for c in clips}
    dec = {k: r for k, r in dec.items() if k in newrel}
    ms = {k: r for k, r in ms.items() if k in newrel}
    sts = Counter(r["status"] for r in dec.values())
    P(f"timing: causes {dict(Counter(c['timing_cause'] for c in clips))}; decisions {dict(sts)}; sources "
      f"{dict(Counter(r['offset_source'] for r in dec.values()))}")
    offs = [float(r["offset_used_s"]) for r in dec.values() if fnum(r.get("offset_used_s")) is not None]
    if offs:
        P(f"   offsets used: median {np.median(offs):+.2f} s, range {min(offs):+.2f}..{max(offs):+.2f}, n={len(offs)}")
    strong = [float(m["offset_s"]) for m in ms.values() if m.get("strong") == "True"]
    if strong:
        P(f"   own strong measurements: n={len(strong)}, median {np.median(strong):+.2f} s, "
          f"|offset|<=3 s: {sum(1 for x in strong if abs(x) <= 3)}")
    P("")
    vk = ["decodes", "crop_dims_ok", "frames_as_expected", "within_keyframe", "frames_match_raw", "eeg_ok",
          "eeg_aux_zero"]
    newv = [v for v in ver.values() if v.get("control") != "True"]
    P("verify (new clips): " + ", ".join(f"{k} {dict(Counter(v.get(k) for v in newv))}" for k in vk))
    nm = [float(v["ncc_mean"]) for v in newv if fnum(v.get("ncc_mean")) is not None]
    if nm:
        P(f"   frames vs raw NCC mean: min {min(nm):.5f}; EEG max |diff| / quant step: max "
          f"{max(float(v['eeg_max_abs_diff']) for v in newv if fnum(v.get('eeg_max_abs_diff')) is not None):.3f}")
    for c in ctl:
        P(f"control {c['rel']}: frames {c['nb']}/{c['orig_nb']} shift {c['frame_shift']} MAE {c['mae_mean']}/"
          f"{c['mae_max']}; eeg parent-label bytes identical {c['eeg_parent_labels_identical']}; fixed-label bytes "
          f"differing {c['eeg_fixed_labels_ndiff_bytes']}; info.txt identical {c['info_identical']}")
    P("")
    sz = [s for s in san.values() if s["kind"] == "seizure"]
    ps = [float(s["p_oof_mean"]) for s in sz]
    P(f"label sanity: new seizure clips {len(sz)}: OOF detector P(sz) < 0.5 in {sum(1 for x in ps if x < 0.5)}; "
      f"LL ratio p10 of {llinfo.get('ll_ref_n')} existing seizure clips (same animals) = {llinfo.get('ll_p10')}")
    fl = Counter(f.split("(")[0] for r in rows for f in (r.get("label_flags") or "").split(" ; ") if f)
    P(f"   flags: {dict(fl)}")
    js = json.load(open(os.path.join(OUT, "sanity", "reference.json")))
    P(f"   reference: stored OOF P(sz) < 0.5 in {js['ref_seizure_p_below_05']} of {js['ref_seizure_n']} existing seizure "
      f"clips; P >= 0.5 in {js['ref_nonseizure_p_above_05']} of {js['ref_nonseizure_n']} existing negatives; both "
      f"detectors low (the clip rule) in {js['ll_ref_both_low']} of {js['ll_ref_n']} sampled existing seizure clips")
    P("   possible_duplicate_* = another annotated event of the animal a clock-step interval away (60 s, 3600 s, 10523 s,")
    P("   13 s) with a similar duration; a duplicate label would point at EEG with no seizure (step 1's RN229 s42 and")
    P("   RN213 s41), and every one found here has EEG seizure evidence at its own time, so none is a duplicate.")
    P("   rule history: the clip-level rule set first (OOF P < 0.5 and LL ratio < p10) marked 3 clips (clip_rule_both_low);")
    P("   the window-level context (sanity/context.csv) shows a 15-27 s detector run (max P 0.90-0.97) inside each of")
    P("   their labelled spans, so they are kept with a soft flag; EEG_DISPUTED (no run, max window P < 0.5 in a")
    P("   span >= 10 s) and NEGATIVE_EEG_SEIZURE_LIKE (a run >= 15 s inside a negative) are the hard flags.")
    for r in rows:
        f_ = r.get("label_flags") or ""
        if r.get("status") == "ok" and any(k in f_ for k in ("possible_duplicate", "negative_eeg", "overlaps",
                                                               "existing_negative", "eeg_detector_low",
                                                               "clip_rule_both_low")):
            P(f"   note            {r['key']}  p_oof={r.get('p_eeg_oof')} ll={r.get('ll_ratio')}  {f_}")
    for r in rows:
        if r.get("status") not in ("ok",):
            P(f"   {r['status']:15s} {r['key']}  {r.get('label_flags') or r.get('note')}")
    P("")
    P("status: " + str(dict(Counter((r['kind'], r['status']) for r in rows))))
    P("integration: see integration.txt; " + json.dumps(integ, default=str))
    write_text("summary.txt", "\n".join(L) + "\n")
    write_text("summary.json", json.dumps(dict(
        n_seizure=sum(1 for c in clips if c["kind"] == "seizure"), n_negative=sum(1 for c in clips if c["kind"] == "nonseizure"),
        n_excluded=len(excl), status=dict(Counter(r["status"] for r in rows)),
        by_class=dict(Counter(r["stage"] for r in rows if r["status"] != "excluded")),
        timing=dict(sts), flags=dict(fl), ll=llinfo, integration=integ,
        reproduction={f"{k[0]}|{k[1]}|{k[2]}": n for k, n in rc.items()}), indent=1, default=str))
    print("\n".join(L))


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["plan", "cut", "eeg", "motion", "activity", "measure", "decide", "recut",
                                        "sanity", "context", "verify", "report"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--ref_per_animal", type=int, default=150)
    ap.add_argument("--ref_only", action="store_true", help="sanity: redo only the reference distributions")
    a = ap.parse_args()
    return globals()[f"cmd_{a.command}"](a)


if __name__ == "__main__":
    sys.exit(main())
