#!/usr/bin/env python3
"""Re-cut the labelled clips whose VIDEO is from the wrong time (their EEG and label are right).

Read-only on the raw tree, data/, data_full/, cache_frames/ and every existing output/ run dir.
Writes ONLY under $EEG_ROOT/output/ttg_recut/ (out_path() refuses anything else).

Offset convention (used everywhere below)
----------------------------------------
    offset_s = (raw pts that really shows the labelled instant) - (raw pts the parent cutter used)

on the nominal video timeline int(DSI_utc_start_time) + pts, which is what the cutter assumed. It equals the
lag L of rawdata/verify/v_activity.py (L maximises corr(motion(t + L), Activity(t))).
  - 10-13 room-D file (frames begin ~10,525 s after the XML start):  offset ~ -10,523 s
  - 11-03 / 12-05 room-C PC-clock step (labels after the step):      offset ~ -60 / -61 s
  - DST fall-back (video started CDT, label CST):                     offset = +3600 s (the video was cut
    1 h EARLY, so the right frames are 1 h LATER in the file; the brief's "~ -3600" is this, sign flipped)

Sub-commands (in order; each is resumable)
------------------------------------------
  plan      candidates (every discover() clip cut from a timing-defect raw file or around the 2023-11-05
            fall-back), the cause model, the decode jobs and 5 unaffected reproduction controls
  motion    one ffmpeg decode per camera file: per-frame motion energy (mean |frame diff|) in every animal box
  activity  per EDF: the 1 Hz Activity channel and per-record ECG flatness (telemetry gaps)
  measure   per candidate: EDF Activity vs crop motion over +-4 h (1 s NCC, masked), refined +-30 s and to
            0.1 s; the partner animal on the same camera as a same-time second opinion; dense scans along files
  decide    aligned / misaligned / unresolved / flagged, with the family cross-check -> decisions.csv, exclude.csv
  recut     the parent cutter (copy mode) + crop_clips.py (re-encode) at the corrected time -> data_full/, manifest.csv
  repro     the same two ffmpeg steps on 5 unaffected clips at their ORIGINAL seek -> repro/
  verify    decode, frame count, NCC against raw frames, clip-file motion vs Activity (new and original)
  report    summary.txt / summary.json

Parent pipeline reproduced (EEG repo, read-only):
  cut_seizure_clips.py / cut_non_seizure_clips.py, --mode copy (sh/_cut_worker.sh):
      ffmpeg -y -ss <seek> -i <raw.mp4> -t <v_clip_dur> -c copy -avoid_negative_ts make_zero video.mp4
      seek = clip start - datetime.fromtimestamp(int(DSI_utc_start_time)) (naive Central), v_clip_dur =
      min(clip end - clip start, video duration - seek)
  crop_clips.py (sh/_crop_all.sh, _fill_crops.sh, _crop_roomd.sh give the per-animal fractions):
      ffmpeg -y -loglevel error -nostdin -i video.mp4 -vf crop=trunc(iw*wf/2)*2:trunc(ih*hf/2)*2:
      trunc(iw*xf/2)*2:trunc(ih*yf/2)*2 -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p -an -threads 2 out.mp4

Needs numpy + scipy only. The miniconda3 base interpreter imports them in 0.2 s; the `eeg` env pages numpy in
for minutes on the compute nodes (see rawdata/README.md). ffmpeg/ffprobe from /work/mech-ai/alloy/miniconda3/bin.
"""
import argparse
import csv
import datetime as dt
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import align_clips as AC  # noqa: E402  (helpers only: EDF header, ffmpeg decode, keyframes, info parsing)

RAW = AC.RAW
EEG_ROOT = AC.EEG_ROOT
AUDIT = AC.AUDIT
ALIGN = os.path.join(AUDIT, "align")
DATA = AC.DATA
FULL = AC.FULL
OUT = os.path.join(EEG_ROOT, "output", "ttg_recut")
FFMPEG, FFPROBE = AC.FFMPEG, AC.FFPROBE
CENTRAL = ZoneInfo("America/Chicago")
UTC = dt.timezone.utc
EPOCH0 = dt.datetime(2023, 1, 1)
FALL_UTC = dt.datetime(2023, 11, 5, 7, 0, 0)                  # 02:00 CDT -> 01:00 CST
DST_DAY_1 = dt.datetime(2023, 11, 5, 1, 0, 0)                 # naive local: start of the repeated hour
DST_DAY_2 = dt.datetime(2023, 11, 5, 2, 0, 0)                 # naive local: end of the repeated hour
STAMP_1013 = "20231014004100"
STAMPS_STEP = ("20231103122100", "20231205222300")
MISLABELLED_NEG = ("RN242", "10-12-2023", "clip_03_vs_seizure_04_Stage_4_20231012_234446")

HALF_WIN_S = 1800         # Activity template: clip span +- 30 min (clipped to the clip's clock segment)
SEARCH_S = 4 * 3600       # coarse search +-4 h
REFINE_S = 30             # refine +-30 s around the coarse peak
MIN_OVERLAP = 0.6         # a lag is scored only where >= 60% of the template bins have video
ALIGNED_TOL_S = 3.0       # |offset| <= 3 s is aligned
FAMILY_TOL_S = 3.0        # a strong measurement must sit within 3 s of its cause's expected offset
FAMILY_TIGHT_S = 2.0      # a family "agrees tightly" when its strong residuals are all within 2 s of their median
# a measurement is "strong" when its peak r, robust z over the +-4 h lag curve and lead over the next peak (>15 s
# away) pass these; set by `calibrate` (3600 s windows along every decoded file, scored against the cause model):
# 384 of 842 windows with a model truth are strong and none contradicts the model (calibrate/rank_hp0_w3600.csv)
STRONG = dict(r=0.10, z=4.0, margin=0.04, sharp=float(os.environ.get("RECUT_SHARP", "0.02")))
DECODE_NEXT_S = 5400      # DST: the camera's next file, first 1.5 h (the +3600 s frames of late clips live there)
PROFILE_WIN_S = 3600      # scan windows are clip-sized (the clip window is its span +- HALF_WIN_S)
PROFILE_STEP_S = 600
MOT_TRANSFORM = os.environ.get("RECUT_MOT_TRANSFORM", "rank")   # rank | log | log1p (see mot_transform)
HP_WIN_S = int(os.environ.get("RECUT_HP_WIN_S", "0"))   # high-pass: subtract a centred running mean (0 = off)
ACT_ON = math.log1p(0.5)   # log1p(Activity) at or above this = at least one Activity count in that second

# Per-animal crop fractions of the parent pipeline, copied verbatim from the EEG repo's sh/_crop_all.sh and
# sh/_fill_crops.sh (room C: wf hf xf yf) and sh/_crop_roomd.sh (room D). (wf, hf, xf, yf) = crop_clips.py's
# (--left_frac, --top_frac, --x_frac, --y_frac). RN243 is not labelled; its box is used only as a partner.
CROP_FRACS = {
    "RN197": (0.60, 0.80, 0.00, 0.00), "RN213": (0.40, 0.80, 0.60, 0.00), "RN222": (0.60, 0.80, 0.00, 0.00),
    "RN237": (0.60, 0.80, 0.00, 0.00), "RN204": (0.40, 0.80, 0.60, 0.00), "RN235": (0.40, 0.80, 0.60, 0.00),
    "RN238": (0.40, 0.80, 0.60, 0.00), "RN199": (0.50, 0.60, 0.00, 0.30), "RN245": (0.50, 0.60, 0.00, 0.30),
    "RN216": (0.48, 0.60, 0.52, 0.30),
    "RN208": (0.43, 0.39, 0.00, 0.33), "RN223": (0.42, 0.38, 0.55, 0.35), "RN210": (0.43, 0.38, 0.10, 0.42),
    "RN224": (0.40, 0.38, 0.53, 0.44), "RN215": (0.42, 0.35, 0.00, 0.20), "RN219": (0.43, 0.35, 0.50, 0.22),
    "RN227": (0.40, 0.33, 0.00, 0.42), "RN244": (0.46, 0.36, 0.52, 0.42), "RN242": (0.44, 0.36, 0.06, 0.44),
    "RN243": (0.40, 0.37, 0.50, 0.45), "RN229": (0.45, 0.35, 0.00, 0.22),
}


# ----------------------------------------------------------------------------- guards / io
def _under(p, root):
    p, r = os.path.realpath(p), os.path.realpath(root)
    return p == r or p.startswith(r + os.sep)


def out_path(*parts):
    """Every write goes through here: under OUT only; never the raw tree, data/, data_full/ or other runs."""
    p = os.path.join(OUT, *parts)
    if not _under(p, OUT) or _under(p, RAW) or _under(p, DATA) or _under(p, FULL) or \
            _under(p, os.path.join(EEG_ROOT, "cache_frames")):
        raise SystemExit(f"refusing to write outside {OUT}: {p}")
    return p


def ensure_dir(p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def write_text(rel, text):
    p = ensure_dir(out_path(rel))
    with open(p + ".tmp", "w") as fh:
        fh.write(text)
    os.replace(p + ".tmp", p)


def write_csv(rel, cols, rows):
    p = ensure_dir(out_path(rel))
    with open(p + ".tmp", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _cell(r.get(k)) for k in cols})
    os.replace(p + ".tmp", p)


def _cell(v):
    if v is None:
        return ""
    if type(v).__module__ == "numpy" and hasattr(v, "item"):
        v = v.item()                                    # numpy scalar -> Python scalar (repr would say np.float64)
    if isinstance(v, float):
        return "" if math.isnan(v) else repr(v)       # full precision: seeks are ~2e4 s and need ms
    if isinstance(v, (list, tuple)):
        return "|".join(map(str, v))
    return v


def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def fnum(x):
    try:
        return None if x in (None, "") else float(x)
    except (TypeError, ValueError):
        return None


def nsec(t):
    """Naive datetime -> seconds since 2023-01-01 (used for naive local AND naive UTC axes)."""
    return (t - EPOCH0).total_seconds()


def from_nsec(s):
    return EPOCH0 + dt.timedelta(seconds=float(s))


def utc_naive(epoch):
    return dt.datetime.fromtimestamp(int(epoch), tz=UTC).replace(tzinfo=None)


def central_offset_s(epoch):
    """UTC - Central wall time at a UTC epoch, in seconds (18000 CDT, 21600 CST)."""
    return -dt.datetime.fromtimestamp(int(epoch), tz=UTC).astimezone(CENTRAL).utcoffset().total_seconds()


def clip_rel(animal, session, clip_dir):
    return f"Data_{animal}_cropped/{session}/{clip_dir}"


def clip_key(animal, session, clip_dir):
    """The discover() path key: data/Data_<A>_cropped/<session>/<clip>/video.mp4."""
    return f"data/{clip_rel(animal, session, clip_dir)}/video.mp4"


def crop_expr(animal):
    wf, hf, xf, yf = CROP_FRACS[animal]
    return f"crop=trunc(iw*{wf}/2)*2:trunc(ih*{hf}/2)*2:trunc(iw*{xf}/2)*2:trunc(ih*{yf}/2)*2"


def crop_box(animal, W=800, H=600):
    wf, hf, xf, yf = CROP_FRACS[animal]
    t = lambda v: int(math.trunc(v / 2)) * 2  # noqa: E731  (ffmpeg's trunc(.../2)*2 on doubles)
    return t(W * xf), t(H * yf), t(W * wf), t(H * hf)


def run(cmd, timeout=3600):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


# ----------------------------------------------------------------------------- raw tables
class Raw:
    """Video files (unique by name) per camera, EDFs per animal, from raw_audit/files.csv."""

    def __init__(self):
        rows = read_csv(os.path.join(AUDIT, "files.csv"))
        self.vid, self.by_cam = {}, defaultdict(list)
        self.edfs = defaultdict(list)
        for r in rows:
            if r["kind"] == "video" and r.get("xml_utc_start") and r.get("duration_s"):
                n = r["name"]
                own = r["animal_folder"] in (r.get("camera") or "").split("-")
                if n in self.vid and not own:
                    continue
                u = int(r["xml_utc_start"])
                nb = int(fnum(r["nb_frames"]) or 0)
                sd = fnum(r["stream_duration"]) or fnum(r["duration_s"])
                wms = fnum(r.get("wall_minus_stream_s")) or 0.0
                self.vid[n] = dict(
                    name=n, relpath=r["relpath"], camera=r.get("camera"), stamp=n.split(".")[-2],
                    utc_int=u, u0=nsec(utc_naive(u)), sync=fnum(r.get("sync_offset_s")) or 0.0,
                    off=central_offset_s(u), dur=fnum(r["duration_s"]), nb=nb, stream=sd, wms=wms,
                    rho=(sd * 15.0 / nb) if nb else 1.0,
                    gap=(sd + wms) - (nb / 15.0 if nb else sd),    # wall span not covered by frames at 15 fps
                    problems=r.get("problems") or "", status=r.get("status"),
                    animals=sorted(set(filter(None, (r.get("xml_animals") or "").split("|")))))
            elif r["kind"] == "edf":
                st = AC.parse_dt(r.get("edf_start"))
                hrs = fnum(r.get("edf_hours"))
                if st is not None and hrs:
                    self.edfs[r["animal_folder"]].append(dict(relpath=r["relpath"], name=r["name"], start=st,
                                                              hours=hrs, status=r.get("status")))
        for v in self.vid.values():
            self.by_cam[v["camera"]].append(v)
        for c in self.by_cam:
            self.by_cam[c].sort(key=lambda v: v["utc_int"])

    def next_file(self, v):
        lst = self.by_cam[v["camera"]]
        i = lst.index(v)
        return lst[i + 1] if i + 1 < len(lst) else None

    def file_at(self, camera, u):
        """The camera file whose nominal span [u0, u0 + dur) holds nominal UTC second u."""
        for v in self.by_cam.get(camera, []):
            if v["u0"] <= u < v["u0"] + v["dur"]:
                return v
        return None

    def edf_for(self, animal, t0, t1):
        """The animal's EDF (own folder) overlapping naive [t0, t1] the most."""
        best, bo = None, 0.0
        for e in self.edfs.get(animal, []):
            s, en = e["start"], e["start"] + dt.timedelta(hours=e["hours"])
            o = (min(en, t1) - max(s, t0)).total_seconds()
            if o > bo:
                best, bo = e, o
        return best


def cam_animals(v):
    """Animals whose box we know on this camera (camera name first-named = left)."""
    names = [a for a in (v["camera"] or "").split("-") if a]
    return [a for a in names if a in CROP_FRACS]


# ----------------------------------------------------------------------------- 1. plan
CAND_COLS = ["key", "animal", "session", "clip_dir", "kind", "label", "y", "cause", "sub", "raw_video", "stamp",
             "camera", "info_clip_start", "info_clip_end", "clip_dur_s", "info_seek_s", "seek_pred_s", "raw_edf",
             "edf_start_local", "edf_offset_s", "dst_ambiguous", "video_start_utc", "file_dur_s", "rho", "gap_s",
             "orch90", "note"]


def cutter_seek(cs, v):
    """The parent cutter's seek: clip start - datetime.fromtimestamp(int(DSI_utc_start_time)) on a Central box."""
    return nsec(cs) - (v["u0"] - v["off"])


def classify_cause(r, v):
    stamp = v["stamp"]
    cs = AC.parse_dt(r["info_clip_start"])
    started_before = utc_naive(v["utc_int"]) < FALL_UTC
    spans = started_before and utc_naive(v["utc_int"]) + dt.timedelta(seconds=v["dur"]) > FALL_UTC
    amb = cs.date() == dt.date(2023, 11, 5) and cs.hour == 1
    if "frames_missing_vs_wallclock" in v["problems"]:
        if stamp == STAMP_1013:
            return "roomD_1013_gap"
        if stamp in STAMPS_STEP:
            return "roomC_clock_step"
        return "timing_defect_other"
    if amb or spans or (started_before and cs >= DST_DAY_1):
        return "dst"
    return None


def cmd_plan(args):
    raw = Raw()
    clips = read_csv(os.path.join(ALIGN, "clips.csv"))
    disc = [r for r in clips if r["in_discover"] == "True"]
    print(f"clips.csv: {len(clips)} dirs, {len(disc)} in discover()")
    # the per-animal boxes must be exactly the audit's pixel-exact crop boxes
    boxes = {}
    for b in read_csv(os.path.join(ALIGN, "crop_boxes.csv")):
        boxes[b["animal"]] = tuple(int(b[k]) for k in ("x", "y", "w", "h"))
    for a, bx in boxes.items():
        if crop_box(a) != bx:
            raise SystemExit(f"crop fraction table disagrees with crop_boxes.csv for {a}: {crop_box(a)} vs {bx}")
    print(f"crop fractions reproduce crop_boxes.csv for all {len(boxes)} labelled animals")
    cands, missing = [], []
    for r in disc:
        v = raw.vid.get((r.get("raw_video") or "").split("/")[-1]) if r.get("raw_video") else None
        if v is None or not r.get("info_clip_start"):
            missing.append(r)
            continue
        cause = classify_cause(r, v)
        if cause is None:
            continue
        cs = AC.parse_dt(r["info_clip_start"])
        c = {k: r.get(k) for k in CAND_COLS if k in r}
        # clips.csv keeps 6 significant digits; recompute the cutter's seek and duration exactly
        c["seek_pred_s"] = cutter_seek(cs, v)
        c["clip_dur_s"] = (AC.parse_dt(r["info_clip_end"]) - cs).total_seconds()
        c.update(key=clip_key(r["animal"], r["session"], r["clip_dir"]), cause=cause, stamp=v["stamp"],
                 camera=v["camera"], video_start_utc=AC.iso(utc_naive(v["utc_int"])), file_dur_s=v["dur"],
                 rho=v["rho"], gap_s=v["gap"])
        if cause == "dst":
            if cs.date() == dt.date(2023, 11, 5) and cs.hour == 1:
                c["sub"] = "dst_ambiguous_hour"
            elif cs >= DST_DAY_2 and utc_naive(v["utc_int"]) < FALL_UTC:
                c["sub"] = "dst_after_change"
            else:
                c["sub"] = "dst_before_change"
        cands.append(c)
    # the orchestrator's 90 (clips.csv logic): 10 + 50 + 27 + 3; which of ours match it is decided in `report`
    write_csv("plan/candidates.csv", CAND_COLS, cands)
    print(f"candidates: {len(cands)}  by cause {dict(Counter(c['cause'] for c in cands))}  "
          f"by sub {dict(Counter(c.get('sub') for c in cands if c.get('sub')))}")
    if missing:
        print(f"  {len(missing)} discover() clips have no raw video / info mapping (not candidates): "
              f"{[m['clip_dir'] for m in missing[:5]]}")

    # decode jobs: every candidate's camera file in full; DST: also the camera's next file, first 1.5 h
    jobs = {}

    def add(v, t=None):
        j = jobs.setdefault(v["name"], dict(id=f"{v['camera']}__{v['stamp']}", name=v["name"],
                                            path=os.path.join(RAW, v["relpath"]), camera=v["camera"],
                                            animals=cam_animals(v), t=t))
        if t is None:
            j["t"] = None
        elif j["t"] is not None:
            j["t"] = max(j["t"], t)

    for c in cands:
        v = raw.vid[c["raw_video"].split("/")[-1]]
        add(v)
        if c["cause"] == "dst":
            nx = raw.next_file(v)
            if nx is not None:
                add(nx, DECODE_NEXT_S)
    jl = sorted(jobs.values(), key=lambda j: j["id"])
    for j in jl:
        j["boxes"] = [(a,) + crop_box(a) for a in j["animals"]]
    write_text("plan/jobs.json", json.dumps(jl, indent=1))
    print(f"decode jobs: {len(jl)}")
    for i, j in enumerate(jl):
        print(f"  [{i}] {j['id']}  animals={j['animals']}  t={j['t']}")

    # reproduction controls: 5 unaffected clips from the audit's main stratum with a high-grade match and a
    # clean keyframe model, one each from 5 different cameras / geometries, fixed choice (not tuned)
    mt = read_csv(os.path.join(ALIGN, "matches.csv"))
    cand_keys = {c["key"] for c in cands}
    want = [("RN197", "seizure"), ("RN238", "nonseizure"), ("RN199", "seizure"), ("RN224", "nonseizure"),
            ("RN242", "seizure")]
    rep = []
    for a, kind in want:
        pool = sorted((m for m in mt if m["stratum"] == "main" and m["animal"] == a and m["kind"] == kind
                       and m["grade"] == "high" and m.get("d_key_frames") == "0"
                       and clip_key(a, m["session"], m["clip_dir"]) not in cand_keys),
                      key=lambda m: (m["session"], m["clip_dir"]))
        if pool:
            m = pool[len(pool) // 2]
            rep.append(dict(key=clip_key(a, m["session"], m["clip_dir"]), animal=a, session=m["session"],
                            clip_dir=m["clip_dir"], kind=kind, raw_video=m["raw_video"]))
    write_csv("plan/repro.csv", ["key", "animal", "session", "clip_dir", "kind", "raw_video"], rep)
    print(f"reproduction controls: {[r['key'] for r in rep]}")


# ----------------------------------------------------------------------------- 2. motion
MLINE = re.compile(r"frame:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:\s*(-?[\d.eE+-]+)")
MOTION_VERSION = 2   # 2: pts from the integer pts x stream time base (pts_time is printed with 6 significant digits)
VLINE = re.compile(r"lavfi\.signalstats\.YAVG=([\d.eE+-]+|nan)")


def load_jobs():
    return json.load(open(out_path("plan/jobs.json")))


def motion_npz(job):
    # RECUT_MOTION_SUBDIR lets a provisional analysis read a saved copy while the decode is re-run
    return out_path(os.environ.get("RECUT_MOTION_SUBDIR", "motion"), f"{job['id']}.npz")


def run_motion_job(job, threads=0):
    import numpy as np
    dst = motion_npz(job)
    if os.path.exists(dst):
        z = np.load(dst)
        ver = int(z["version"]) if "version" in z.files else 0
        if list(z["animals"]) == job["animals"] and float(z["t"]) == float(job["t"] or 0) and ver == MOTION_VERSION:
            return f"{job['id']}: cached"
    pr = run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=time_base", "-of",
              "csv=p=0", job["path"]], timeout=120)
    tb_num, tb_den = (int(x) for x in pr.stdout.strip().split("/"))
    tmp = out_path("motion", "tmp", job["id"])
    os.makedirs(tmp, exist_ok=True)
    k = len(job["boxes"])
    fc = "[0:v]format=gray,split=%d%s" % (k, "".join(f"[s{i}]" for i in range(k)))
    for i, (a, x, y, w, h) in enumerate(job["boxes"]):
        sw, sh = max(8, (w // 4) * 2), max(8, (h // 4) * 2)
        fc += (f";[s{i}]crop={w}:{h}:{x}:{y},scale={sw}:{sh}:flags=area,tblend=all_mode=difference,signalstats,"
               f"metadata=mode=print:key=lavfi.signalstats.YAVG:file={tmp}/{a}.txt[o{i}]")
    cmd = [FFMPEG, "-nostdin", "-v", "error", "-threads", str(threads)]
    if job["t"]:
        cmd += ["-t", str(job["t"])]
    cmd += ["-i", job["path"], "-filter_complex", fc]
    for i in range(k):
        cmd += ["-map", f"[o{i}]", "-f", "null", "-"]
    t0 = time.time()
    r = run(cmd, timeout=6 * 3600)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed {job['id']}: {r.stderr[-500:]}")
    pts, mots = None, []
    for (a, *_bx) in job["boxes"]:
        p_, m_ = [], []
        with open(os.path.join(tmp, f"{a}.txt")) as fh:
            for line in fh:
                mm = MLINE.match(line)
                if mm:
                    P = int(mm.group(2))
                    t_ = P * tb_num / tb_den
                    if len(p_) < 200 and abs(t_ - float(mm.group(3))) > 1e-3:
                        raise RuntimeError(f"{job['id']}: pts {P} x {tb_num}/{tb_den} != pts_time {mm.group(3)}")
                    p_.append(t_)
                    continue
                mv = VLINE.search(line)
                if mv:
                    m_.append(float(mv.group(1)))
        n = min(len(p_), len(m_))
        p_, m_ = np.array(p_[:n]), np.array(m_[:n], dtype=np.float32)
        if pts is None:
            pts = p_
        elif len(p_) != len(pts) or np.abs(p_ - pts).max() > 1e-3:
            raise RuntimeError(f"{job['id']}: pts differ between boxes")
        mots.append(m_)
    np.savez_compressed(ensure_dir(dst) + ".tmp.npz", pts=pts, mot=np.stack(mots), animals=np.array(job["animals"]),
                        t=float(job["t"] or 0), name=job["name"], stderr=r.stderr[-2000:], version=MOTION_VERSION,
                        time_base=f"{tb_num}/{tb_den}")
    os.replace(dst + ".tmp.npz", dst)
    shutil.rmtree(tmp, ignore_errors=True)
    nonmono = int((np.diff(pts) <= 0).sum())
    return (f"{job['id']}: {len(pts)} frames, pts {pts[0]:.3f}..{pts[-1]:.1f}, nonmonotonic={nonmono}, "
            f"stderr_lines={len(r.stderr.splitlines())}, {time.time() - t0:.0f}s")


def cmd_motion(args):
    jobs = load_jobs()
    sel = jobs if args.index is None else [jobs[args.index]]
    if args.workers <= 1 or len(sel) == 1:
        for j in sel:
            print(run_motion_job(j, args.threads), flush=True)
        return
    with ThreadPoolExecutor(args.workers) as ex:
        for msg in ex.map(lambda j: run_motion_job(j, args.threads), sel):
            print(msg, flush=True)


# ----------------------------------------------------------------------------- 3. activity
def edf_npz(relpath):
    return out_path("activity", re.sub(r"[^\w.-]+", "_", relpath) + ".npz")


def read_activity(relpath, t_from, t_to):
    """1 Hz Activity (physical) and a per-second 'ECG flat' flag (first 1000 ECG samples of the record constant)."""
    import numpy as np
    path = os.path.join(RAW, relpath)
    h = AC.edf_header(path)
    ia, ie = AC.edf_pick(h, "activity"), AC.edf_pick(h)
    rd, na, ne = h["rdur"], h["nsamp"][ia], h["nsamp"][ie]
    if abs(na / rd - 1.0) > 1e-9:
        raise RuntimeError(f"{relpath}: Activity is {na / rd} Hz, expected 1 Hz")
    r0 = max(0, int((t_from - h["start"]).total_seconds() // rd))
    r1 = min(h["nrec_file"], int(math.ceil((t_to - h["start"]).total_seconds() / rd)))
    nread = min(ne, 1000)
    lo = min(h["off"][ia], h["off"][ie])
    hi = max(h["off"][ia] + na, h["off"][ie] + nread)
    g = (h["pmax"][ia] - h["pmin"][ia]) / (h["dmax"][ia] - h["dmin"][ia])
    act = np.zeros((max(0, r1 - r0), na), np.float32)
    flat = np.zeros(max(0, r1 - r0), bool)
    with open(path, "rb") as fh:
        for i, rec in enumerate(range(r0, r1)):
            fh.seek(h["hdr"] + rec * h["recbytes"] + 2 * lo)
            x = np.frombuffer(fh.read(2 * (hi - lo)), dtype="<i2")
            a = x[h["off"][ia] - lo: h["off"][ia] - lo + na].astype(np.float64)
            act[i] = (a - h["dmin"][ia]) * g + h["pmin"][ia]
            e = x[h["off"][ie] - lo: h["off"][ie] - lo + nread]
            flat[i] = len(e) == 0 or e.min() == e.max()
    t0 = nsec(h["start"]) + r0 * rd
    return dict(t0=t0, act=act.reshape(-1), flat=np.repeat(flat, na), edf_start=nsec(h["start"]),
                edf_end=nsec(h["start"]) + h["nrec_file"] * rd, label_act=h["labels"][ia], label_ecg=h["labels"][ie])


def activity_needs(raw, cands):
    """(animal, edf relpath) -> naive [t_from, t_to] to read."""
    needs = {}
    jobs = load_jobs()
    for j in jobs:
        v = raw.vid[j["name"]]
        # the file's span on the naive local axis the cutter used, +- the search range
        s = from_nsec(v["u0"] - v["off"])
        e = s + dt.timedelta(seconds=v["dur"] if not j["t"] else j["t"])
        for a in j["animals"]:
            ed = raw.edf_for(a, s, e)
            if ed is None:
                continue
            lo, hi = s - dt.timedelta(seconds=SEARCH_S + 1800), e + dt.timedelta(seconds=SEARCH_S + 1800)
            k = (a, ed["relpath"])
            if k in needs:
                needs[k] = (min(needs[k][0], lo), max(needs[k][1], hi))
            else:
                needs[k] = (lo, hi)
    for c in cands:
        if c.get("raw_edf"):
            cs = AC.parse_dt(c["info_clip_start"])
            k = (c["animal"], c["raw_edf"])
            lo, hi = cs - dt.timedelta(seconds=SEARCH_S + 1800), cs + dt.timedelta(seconds=SEARCH_S + 1800)
            needs[k] = (min(needs[k][0], lo), max(needs[k][1], hi)) if k in needs else (lo, hi)
    return needs


def cmd_activity(args):
    import numpy as np
    raw = Raw()
    cands = read_csv(out_path("plan/candidates.csv"))
    needs = activity_needs(raw, cands)
    print(f"{len(needs)} EDF reads")

    def one(item):
        (a, rel), (lo, hi) = item
        p = edf_npz(rel)
        if os.path.exists(p):
            z = np.load(p)
            if z["t0"] <= nsec(lo) + 10 and z["t0"] + len(z["act"]) >= nsec(hi) - 10:
                return f"{rel}: cached"
        t0 = time.time()
        d = read_activity(rel, lo, hi)
        np.savez_compressed(ensure_dir(p) + ".tmp.npz", animal=a, relpath=rel, **d)
        os.replace(p + ".tmp.npz", p)
        return (f"{rel}: {len(d['act'])} s, nonzero={int((d['act'] >= 0.5).sum())}, flat_s={int(d['flat'].sum())}, "
                f"{time.time() - t0:.0f}s")

    with ThreadPoolExecutor(args.workers) as ex:
        for msg in ex.map(one, sorted(needs.items())):
            print(msg, flush=True)


# ----------------------------------------------------------------------------- 4. measure
class Timeline:
    """Per (camera, animal): per-frame motion on the nominal UTC axis int(DSI_utc_start_time) + pts."""

    def __init__(self, raw):
        import numpy as np
        self.raw = raw
        self.frames = defaultdict(list)      # (camera, animal) -> [(u array, mot array, name)]
        for j in load_jobs():
            p = motion_npz(j)
            if not os.path.exists(p):
                continue
            z = np.load(p)
            v = raw.vid[j["name"]]
            for i, a in enumerate(z["animals"]):
                self.frames[(v["camera"], str(a))].append((v["u0"] + z["pts"], z["mot"][i].astype(np.float64),
                                                          j["name"]))

    def bins(self, camera, animal, U0, N, shift=0.0):
        """1 Hz mean motion for nominal seconds [U0, U0 + N) (bin i holds u - shift in [U0 + i, U0 + i + 1))."""
        import numpy as np
        s = np.zeros(N)
        c = np.zeros(N)
        for u, m, _ in self.frames.get((camera, animal), []):
            i0, i1 = np.searchsorted(u, [U0 + shift, U0 + N + shift])      # pts are monotonic within a file
            if i1 <= i0:
                continue
            k = np.floor(u[i0:i1] - shift - U0).astype(np.int64)
            ok = (k >= 0) & (k < N)
            if ok.any():
                s += np.bincount(k[ok], weights=m[i0:i1][ok], minlength=N)
                c += np.bincount(k[ok], minlength=N)
        v = c >= 8
        out = np.full(N, np.nan)
        out[v] = s[v] / c[v]
        return out


class Activity:
    def __init__(self):
        self.cache = {}

    def get(self, rel):
        import numpy as np
        if rel not in self.cache:
            p = edf_npz(rel)
            self.cache[rel] = dict(np.load(p)) if os.path.exists(p) else None
        return self.cache[rel]

    def window(self, rel, t1, t2):
        """log1p(Activity) for naive seconds [t1, t2) (integer grid), NaN where outside / ECG flat."""
        import numpy as np
        d = self.get(rel)
        n = int(t2 - t1)
        out = np.full(n, np.nan)
        if d is None:
            return out
        i0 = int(round(t1 - float(d["t0"])))
        a, f = d["act"], d["flat"]
        lo, hi = max(0, i0), min(len(a), i0 + n)
        if hi > lo:
            seg = np.log1p(np.maximum(a[lo:hi], 0).astype(np.float64))
            seg[f[lo:hi]] = np.nan
            out[lo - i0: hi - i0] = seg
        return out

    def flat_runs(self, rel, min_len=60):
        """Naive-second intervals where the ECG is flat for >= min_len s (telemetry gaps)."""
        import numpy as np
        d = self.get(rel)
        if d is None:
            return []
        f = d["flat"].astype(np.int8)
        e = np.flatnonzero(np.diff(np.concatenate([[0], f, [0]])))
        runs = [(float(d["t0"]) + s, float(d["t0"]) + t) for s, t in zip(e[::2], e[1::2]) if t - s >= min_len]
        return runs


def highpass(x, w=None):
    """x minus its centred running mean over w bins (NaN-aware). Slow level changes in motion (room lights, people
    working in the room) otherwise give broad spurious NCC peaks that pull windows from anywhere in the file."""
    import numpy as np
    w = HP_WIN_S if w is None else w
    if not w:
        return x
    v = np.isfinite(x).astype(float)
    x0 = np.where(v > 0, x, 0.0)
    k = np.ones(int(w))
    s = np.convolve(x0, k, mode="same")
    n = np.convolve(v, k, mode="same")
    out = x - s / np.maximum(n, 1)
    out[(v == 0) | (n < w / 3)] = np.nan
    return out


def mot_transform(m):
    """Motion values are ~0.01-0.3 grey levels, where log1p is linear, so one large spike (a person, the lights)
    that happens to meet one Activity count dominates Pearson r. 'rank' maps the searched series to its percentile
    ranks (bounded, outlier-proof); 'log' uses log(m + 0.01) (0.01 ~ the motion noise floor)."""
    import numpy as np
    if MOT_TRANSFORM == "log1p":
        return np.log1p(m)
    if MOT_TRANSFORM == "log":
        return np.log(m + 0.01)
    out = np.full(len(m), np.nan)
    g = np.isfinite(m)
    if g.any():
        from scipy.stats import rankdata
        out[g] = rankdata(m[g]) / g.sum()
    return out


def masked_ncc(a, m):
    """Pearson r of template a (len L) against every window of m (len N), both with NaN = missing.

    Returns (r[N-L+1], n_overlap[N-L+1])."""
    import numpy as np
    from scipy.signal import fftconvolve
    va, vm = np.isfinite(a).astype(float), np.isfinite(m).astype(float)
    a0, m0 = np.where(va > 0, a, 0.0), np.where(vm > 0, m, 0.0)

    def xc(x, y):     # sum_i x[i] y[s+i]
        return fftconvolve(y, x[::-1], mode="valid")

    n = xc(va, vm)
    sa, saa = xc(a0, vm), xc(a0 * a0, vm)
    sm, smm = xc(va, m0), xc(va, m0 * m0)
    sam = xc(a0, m0)
    n_ = np.maximum(n, 1)
    cov = sam - sa * sm / n_
    var = np.maximum(saa - sa * sa / n_, 0) * np.maximum(smm - sm * sm / n_, 0)
    r = cov / np.sqrt(np.maximum(var, 1e-12))
    r[(var <= 1e-9)] = np.nan
    return r, np.round(n).astype(int)


def sharpness(rr, k):
    """Peak r minus the best r 4-15 s away: a measurement must localise the offset to the +-3 s tolerance; a broad
    plateau (long activity bouts) does not, whatever its height."""
    import numpy as np
    side = np.concatenate([rr[max(0, k - 15): max(0, k - 3)], rr[k + 4: k + 16]])
    side = side[np.isfinite(side)]
    return float(rr[k] - side.max()) if len(side) else float("nan")


def pearson(a, b):
    import numpy as np
    g = np.isfinite(a) & np.isfinite(b)
    if g.sum() < 30 or a[g].std() == 0 or b[g].std() == 0:
        return float("nan"), int(g.sum())
    return float(np.corrcoef(a[g], b[g])[0, 1]), int(g.sum())


def segment_bounds(t_lo, t_hi, flat_runs):
    """Clock discontinuities on the label (EDF naive) axis: the DST repeated hour boundaries and telemetry gaps."""
    b = [nsec(DST_DAY_1), nsec(DST_DAY_2)]
    for s, e in flat_runs:
        b += [s, e]
    return sorted(x for x in b if t_lo < x < t_hi)


def measure_one(TL, ACT, camera, animal, edf_rel, off, t_clip0, t_clip1, t_seg, search=SEARCH_S, refine=True):
    """Offset of the video (camera/animal motion) against the EDF Activity over the naive window.

    t_clip0/1: the labelled span (naive s). t_seg: (lo, hi) naive bounds of the clip's clock segment.
    Returns a dict with the coarse peak, refined integer and 0.1 s offsets, and scores."""
    import numpy as np
    t1 = max(math.floor(t_clip0 - HALF_WIN_S), t_seg[0])
    t2 = min(math.ceil(t_clip1 + HALF_WIN_S), t_seg[1])
    t1, t2 = int(math.ceil(t1)), int(math.floor(t2))
    res = dict(win_lo=t1, win_hi=t2, win_s=t2 - t1)
    if t2 - t1 < 120:
        res["err"] = "window_too_short"
        return res
    a = highpass(ACT.window(edf_rel, t1, t2))
    L = len(a)
    fa = np.isfinite(a)
    res.update(act_valid=int(fa.sum()), act_nonzero=int((np.nan_to_num(a) >= ACT_ON).sum()),
               act_sd=float(np.nanstd(a)) if fa.any() else float("nan"))
    if fa.sum() < 120 or np.nanstd(a) == 0:
        res["err"] = "activity_flat_or_missing"
        return res
    # nominal UTC second of template bin 0 under offset 0 is t1 + off
    U0 = int(t1 + off - search)
    N = L + 2 * search
    m = highpass(mot_transform(TL.bins(camera, animal, U0, N)))
    r, n = masked_ncc(a, m)                   # index s <-> offset s - search
    ok = (n >= MIN_OVERLAP * fa.sum()) & np.isfinite(r)
    res["lags_scored"] = int(ok.sum())
    if ok.sum() < 60:
        res["err"] = "no_video_overlap"
        return res
    rr = np.where(ok, r, np.nan)
    k = int(np.nanargmax(rr))
    offs = np.arange(len(rr)) - search
    med = float(np.nanmedian(rr))
    mad = float(np.nanmedian(np.abs(rr - med))) * 1.4826 + 1e-9
    r2 = rr.copy()
    r2[max(0, k - 15): k + 16] = np.nan
    second = float(np.nanmax(r2)) if np.isfinite(r2).any() else float("nan")
    k2 = int(np.nanargmax(r2)) if np.isfinite(r2).any() else None
    res.update(coarse_off=int(offs[k]), coarse_r=float(rr[k]), coarse_z=(float(rr[k]) - med) / mad,
               second_r=second, second_off=int(offs[k2]) if k2 is not None else None,
               margin=float(rr[k]) - second if np.isfinite(second) else float("nan"),
               r_noise_med=med, r_noise_sd=mad, overlap=int(n[k]))
    at = lambda o: float(rr[o + search]) if 0 <= o + search < len(rr) and np.isfinite(rr[o + search]) else float("nan")  # noqa: E731
    res["r_at0"] = at(0)
    res["_at"] = at
    if not refine:
        res.update(offset_s=float(offs[k]), offset_r=float(rr[k]), sharp=sharpness(rr, k))
        return res
    # refine: integer peak within +-REFINE_S of the coarse peak, then 0.1 s re-binning of the frames
    lo_, hi_ = max(0, k - REFINE_S), min(len(rr), k + REFINE_S + 1)
    kk = lo_ + int(np.nanargmax(rr[lo_:hi_]))
    oi = int(offs[kk])
    res.update(refine_off_int=oi, refine_r_int=float(rr[kk]))
    res["sharp"] = sharpness(rr, kk)
    best = (float(rr[kk]), 0.0)
    for d10 in range(-15, 16):
        sh = d10 / 10.0
        # re-bin at a sub-second shift; the rank transform is taken over the same +-4 h series as the coarse search
        mb = highpass(mot_transform(TL.bins(camera, animal, U0, N, shift=sh)))
        k0 = int(t1 + off + oi) - U0
        rv, _ = pearson(a, mb[k0: k0 + L])
        if np.isfinite(rv) and rv > best[0] + 1e-12:
            best = (rv, sh)
    res.update(offset_s=oi + best[1], offset_r=best[0])
    return res


def clip_segment(ACT, edf_rel, t0, t1, extra=()):
    """The clip's clock segment on the label axis: EDF span, cut at the DST boundaries, telemetry gaps and any
    extra bound (content_boundary)."""
    d = ACT.get(edf_rel)
    if d is None:
        return None
    lo, hi = float(d["edf_start"]), float(d["edf_end"])
    for b in sorted(segment_bounds(lo, hi, ACT.flat_runs(edf_rel)) + [x for x in extra if lo < x < hi]):
        if b <= t0:
            lo = max(lo, b)
        elif b >= t1:
            hi = min(hi, b)
    return lo, hi


def model_offset(v, p, sub, drift=True):
    """Expected offset from the cause model (see module doc); p = the cutter's pts.

    drift=True: the frames are real time at 15 fps and the pts are dilated by rho = stream_duration * 15 / nb_frames
    (RN242-RN243 and RN229: rho - 1 = 2.0e-4, so up to +5.4 s at the end of a 7.5 h file); drift=False: the pts are
    real time. The dense scans follow the drift model on RN242's 10-13 file end to end and on the DST files; a DST
    clip late in its file has its frames in the camera's NEXT file, where the pts start again (content_model)."""
    rho = v["rho"] if drift else 1.0
    gap = v["gap"] if drift else v["wms"]
    A = {"dst_after_change": 3600.0, "roomD_1013_gap": -gap, "step_after": -gap}.get(sub, 0.0)
    return rho * (p + A) - p


DRIFT_MIN = 5e-5      # |rho - 1| above this: the camera's pts drift against real time (RN242-RN243, RN229)


def content_next(raw, v, p, sub):
    """For a DST clip whose true frames (real time = cutter's time + 3600 s) are past this file's last frame: the
    camera's next file, else None."""
    if raw is None or sub != "dst_after_change":
        return None
    u_real = v["u0"] + p + 3600.0
    if u_real < v["u0"] + v["dur"] / v["rho"]:
        return None
    g = raw.next_file(v)
    return g if g is not None and u_real >= g["u0"] else None


def content_boundary(v):
    """Label (naive) time at which a DST clip's true frames cross from this file into the next one. On a drifting
    camera the nominal timeline jumps back by (rho - 1) * duration (5.4 s) there, so windows must not straddle it."""
    start = utc_naive(v["utc_int"])
    if abs(v["rho"] - 1) < DRIFT_MIN or not (start < FALL_UTC < start + dt.timedelta(seconds=v["dur"])):
        return None                                   # only files that span the 2023-11-05 fall-back
    return v["u0"] + v["dur"] / v["rho"] - 3600.0 - v["off"]


def model_interval(v, p, sub, raw=None):
    """[no drift, full drift] of the cause model; for a DST clip whose frames are in the next file g, the drift is
    g's, accrued from g's start: 3600 + (rho_g - 1) * (u_real - u0_g)."""
    g = content_next(raw, v, p, sub)
    if g is not None:
        a, b = 3600.0, 3600.0 + (g["rho"] - 1.0) * (v["u0"] + p + 3600.0 - g["u0"])
    else:
        a, b = model_offset(v, p, sub, True), model_offset(v, p, sub, False)
    return min(a, b), max(a, b)


def side_of(v, t, flat_runs):
    """The cause sub-family of naive label time t on camera file v (for scan windows)."""
    if v["stamp"] == STAMP_1013:
        return "roomD_1013_gap"
    if v["stamp"] in STAMPS_STEP:
        runs = [rn for rn in flat_runs if rn[1] - rn[0] >= 300
                and v["u0"] - v["off"] <= rn[0] <= v["u0"] - v["off"] + v["dur"]]
        if not runs:
            return "step_no_gap_found"
        g = max(runs, key=lambda rn: rn[1] - rn[0])
        return "step_before" if t < g[0] else ("step_after" if t >= g[1] else "step_in_gap")
    if utc_naive(v["utc_int"]) < FALL_UTC < utc_naive(v["utc_int"]) + dt.timedelta(seconds=v["dur"]):
        return ("dst_before_change" if t < nsec(DST_DAY_1) else
                "dst_ambiguous_hour" if t < nsec(DST_DAY_2) else "dst_after_change")
    return "normal"


MEAS_COLS = ["key", "animal", "session", "clip_dir", "y", "cause", "sub", "camera", "raw_video", "info_clip_start",
             "seek_pred_s",
             "clip_dur_s", "edf", "seg_lo", "seg_hi", "gap_lo", "gap_hi", "expected_s", "expected_nodrift_s",
             "expected_alt_s",
             "win_s", "act_nonzero", "act_sd", "lags_scored", "coarse_off", "coarse_r", "coarse_z", "second_r",
             "second_off", "margin", "sharp", "refine_off_int", "refine_r_int", "offset_s", "offset_r", "r_at0",
             "r_at_expected", "r_at_expected_alt", "strong", "err",
             "partner", "p_edf", "p_offset_s", "p_offset_r", "p_coarse_z", "p_margin", "p_sharp", "p_r_at0",
             "p_r_at_expected",
             "p_strong", "p_err"]


def is_strong(m):
    return (m.get("err") is None and (m.get("offset_r") or 0) >= STRONG["r"] and (m.get("coarse_z") or 0) >= STRONG["z"]
            and (m.get("margin") or 0) >= STRONG["margin"] and _finite(m.get("sharp")) and m["sharp"] >= STRONG["sharp"]
            and abs((m.get("offset_s") or 0) - (m.get("coarse_off") or 0)) <= 2)


def cmd_measure(args):
    import numpy as np
    raw = Raw()
    TL = Timeline(raw)
    ACT = Activity()
    cands = read_csv(out_path("plan/candidates.csv"))
    rows = []
    for c in cands:
        v = raw.vid[c["raw_video"].split("/")[-1]]
        cs, ce = AC.parse_dt(c["info_clip_start"]), AC.parse_dt(c["info_clip_end"])
        t0, t1 = nsec(cs), nsec(ce)
        p = fnum(c["seek_pred_s"])
        row = {k: c.get(k) for k in MEAS_COLS if k in c}
        row["edf"] = c.get("raw_edf")
        sub = c.get("sub") or ""
        # step family: which side of the room's telemetry gap (EDF flat run) the label sits on
        if c["cause"] == "roomC_clock_step":
            runs = [rn for rn in ACT.flat_runs(c["raw_edf"]) if rn[1] - rn[0] >= 300
                    and v["u0"] - v["off"] <= rn[0] <= v["u0"] - v["off"] + v["dur"]]
            if runs:
                g = max(runs, key=lambda rn: rn[1] - rn[0])
                row.update(gap_lo=AC.iso(from_nsec(g[0])), gap_hi=AC.iso(from_nsec(g[1])))
                sub = "step_before" if t1 <= g[0] else ("step_after" if t0 >= g[1] else "step_in_gap")
            else:
                sub = "step_no_gap_found"
        elif c["cause"] == "roomD_1013_gap":
            sub = "roomD_1013_gap"
        row["sub"] = sub
        exp = model_offset(v, p, sub)
        row["expected_s"] = exp
        row["expected_nodrift_s"] = model_offset(v, p, sub, drift=False)
        if sub in ("dst_ambiguous_hour",):
            row["expected_alt_s"] = model_offset(v, p, "dst_after_change")
        elif sub in ("step_in_gap", "step_no_gap_found"):
            row["expected_alt_s"] = model_offset(v, p, "step_after")
        xb = [b for b in (content_boundary(v),) if b is not None]
        seg = clip_segment(ACT, c["raw_edf"], t0, t1, xb) if c.get("raw_edf") else None
        if seg is None:
            row["err"] = "no_edf_activity"
            rows.append(row)
            continue
        row.update(seg_lo=AC.iso(from_nsec(seg[0])), seg_hi=AC.iso(from_nsec(seg[1])))
        m = measure_one(TL, ACT, v["camera"], c["animal"], c["raw_edf"], v["off"], t0, t1, seg)
        at = m.pop("_at", None)
        row.update({k: m.get(k) for k in m})
        if at is not None:
            row["r_at_expected"] = at(int(round(exp)))
            if row.get("expected_alt_s") not in (None, ""):
                row["r_at_expected_alt"] = at(int(round(row["expected_alt_s"])))
        row["strong"] = is_strong(m)
        # partner animal on the same camera, same naive window (same acquisition PC clock)
        for pa in cam_animals(v):
            if pa == c["animal"]:
                continue
            ped = raw.edf_for(pa, cs - dt.timedelta(hours=1), ce + dt.timedelta(hours=1))
            if ped is None or ACT.get(ped["relpath"]) is None:
                continue
            pseg = clip_segment(ACT, ped["relpath"], t0, t1, xb)
            pm = measure_one(TL, ACT, v["camera"], pa, ped["relpath"], v["off"], t0, t1, pseg)
            pat = pm.pop("_at", None)
            row.update(partner=pa, p_edf=ped["relpath"], p_offset_s=pm.get("offset_s"), p_offset_r=pm.get("offset_r"),
                       p_coarse_z=pm.get("coarse_z"), p_margin=pm.get("margin"), p_sharp=pm.get("sharp"),
                       p_r_at0=pm.get("r_at0"),
                       p_r_at_expected=pat(int(round(exp))) if pat else None, p_strong=is_strong(pm), p_err=pm.get("err"))
            break
        rows.append(row)
        print(f"{c['animal']:6s} {sub:20s} {c['clip_dir'][:44]:44s} exp={exp:9.1f} off={row.get('offset_s')} "
              f"r={_f(row.get('offset_r'))} z={_f(row.get('coarse_z'))} mg={_f(row.get('margin'))} "
              f"strong={row['strong']} | partner {row.get('partner')} off={row.get('p_offset_s')} "
              f"z={_f(row.get('p_coarse_z'))} {row.get('err') or ''}", flush=True)
    write_csv("measure/clips.csv", MEAS_COLS, rows)

    # dense scans along every decoded file (both animals): the clock profile the families must agree with
    prof = []
    for j in load_jobs():
        v = raw.vid[j["name"]]
        span = v["dur"] if not j["t"] else j["t"]
        s0 = v["u0"] - v["off"]
        for a in j["animals"]:
            ed = raw.edf_for(a, from_nsec(s0), from_nsec(s0 + span))
            if ed is None or ACT.get(ed["relpath"]) is None:
                continue
            for tc in np.arange(s0 + PROFILE_WIN_S / 2, s0 + span - PROFILE_WIN_S / 2 + 1, PROFILE_STEP_S):
                seg = clip_segment(ACT, ed["relpath"], tc - 1, tc + 1,
                                   [b for b in (content_boundary(v),) if b is not None])
                if seg is None:
                    continue
                w0, w1 = tc - PROFILE_WIN_S / 2 + HALF_WIN_S, tc + PROFILE_WIN_S / 2 - HALF_WIN_S
                m = measure_one(TL, ACT, v["camera"], a, ed["relpath"], v["off"], w0, w1, seg)
                m.pop("_at", None)
                prof.append(dict(file=j["name"], camera=v["camera"], animal=a, edf=ed["relpath"],
                                 sub=side_of(v, tc, ACT.flat_runs(ed["relpath"])),
                                 t_center=AC.iso(from_nsec(tc)), pts_center=tc - s0, win_s=m.get("win_s"),
                                 coarse_off=m.get("coarse_off"), offset_s=m.get("offset_s"), offset_r=m.get("offset_r"),
                                 coarse_z=m.get("coarse_z"), margin=m.get("margin"), sharp=m.get("sharp"),
                                 r_at0=m.get("r_at0"), strong=is_strong(m), err=m.get("err")))
        print(f"profile {j['id']}: {sum(1 for p_ in prof if p_['file'] == j['name'])} windows", flush=True)
    write_csv("measure/profiles.csv", ["file", "camera", "animal", "edf", "sub", "t_center", "pts_center", "win_s",
                                       "coarse_off", "offset_s", "offset_r", "coarse_z", "margin", "sharp", "r_at0",
                                       "strong", "err"], prof)


def _f(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


# ----------------------------------------------------------------------------- 4b. calibrate
def expected_along(raw, ACT, v, animal, edf_rel, t):
    """The cause model's interval [no drift, full drift] for a window centred at naive time t of camera file v
    (None = no true match expected: before the 10-13 frames, inside a telemetry gap, at the DST boundary)."""
    p = t - (v["u0"] - v["off"])
    if v["stamp"] == STAMP_1013:
        return model_interval(v, p, "roomD_1013_gap") if p > v["gap"] + 900 else None
    if v["stamp"] in STAMPS_STEP:
        runs = [rn for rn in ACT.flat_runs(edf_rel) if rn[1] - rn[0] >= 300]
        if not runs:
            return None
        g = max(runs, key=lambda rn: rn[1] - rn[0])
        return (model_interval(v, p, "step_before") if t < g[0] - 900 else
                model_interval(v, p, "step_after") if t > g[1] + 900 else None)
    if utc_naive(v["utc_int"]) < FALL_UTC < utc_naive(v["utc_int"]) + dt.timedelta(seconds=v["dur"]):
        if abs(t - nsec(DST_DAY_1)) < 900:
            return None
        cb = content_boundary(v)
        if cb is not None and abs(t - cb) < 900:
            return None
        return model_interval(v, p, "dst_before_change" if t < nsec(DST_DAY_1) else "dst_after_change", raw)
    return model_interval(v, p, "normal")


def cmd_calibrate(args):
    """Dense windows along every decoded camera file (both animals), measured like a clip and scored against the
    cause model, to set the transform and the 'strong' thresholds by their false-alarm rate. Truth here is only
    the piecewise cause model (0 / -gap / +3600 +- camera drift), never a clip's own label."""
    import numpy as np
    raw = Raw()
    TL = Timeline(raw)
    ACT = Activity()
    rows = []
    for j in load_jobs():
        v = raw.vid[j["name"]]
        span = v["dur"] if not j["t"] else j["t"]
        s0 = v["u0"] - v["off"]
        for a in j["animals"]:
            ed = raw.edf_for(a, from_nsec(s0), from_nsec(s0 + span))
            if ed is None or ACT.get(ed["relpath"]) is None:
                continue
            for tc in np.arange(s0 + args.win / 2, s0 + span - args.win / 2 + 1, args.step):
                seg = clip_segment(ACT, ed["relpath"], tc - 1, tc + 1,
                                   [b for b in (content_boundary(v),) if b is not None])
                if seg is None:
                    continue
                w0 = tc - args.win / 2 + HALF_WIN_S
                w1 = tc + args.win / 2 - HALF_WIN_S
                m = measure_one(TL, ACT, v["camera"], a, ed["relpath"], v["off"], w0, w1, seg, refine=False)
                m.pop("_at", None)
                iv = expected_along(raw, ACT, v, a, ed["relpath"], tc)
                lag = m.get("offset_s")
                # 0.74 s = the method baseline lambda (see method_baseline); distance to the model interval
                off = None if lag is None else lag + v["sync"] + 0.74
                ok = None if (iv is None or off is None) else (
                    (0.0 if iv[0] <= off <= iv[1] else min(abs(off - iv[0]), abs(off - iv[1]))) <= 3.0)
                exp = None if iv is None else (iv[0] + iv[1]) / 2
                rows.append(dict(file=j["name"], animal=a, pts=tc - s0, win=m.get("win_s"), expected=exp, lag=lag,
                                 r=m.get("coarse_r"), z=m.get("coarse_z"), margin=m.get("margin"),
                                 sharp=m.get("sharp"), nz=m.get("act_nonzero"), consistent=ok, err=m.get("err")))
        print(f"calibrate {j['id']}: {len(rows)} windows so far", flush=True)
    tag = f"{MOT_TRANSFORM}_hp{HP_WIN_S}_w{args.win}"
    write_csv(f"calibrate/{tag}.csv", ["file", "animal", "pts", "win", "expected", "lag", "r", "z", "margin", "sharp",
                                       "nz", "consistent", "err"], rows)
    sc = [r for r in rows if r["consistent"] is not None]
    print(f"{tag}: {len(rows)} windows, {len(sc)} with a model truth")
    for rt in (0.10, 0.15, 0.20, 0.25):
        for mt in (0.02, 0.04, 0.06):
            for zt in (4, 6):
                st = [r for r in sc if (r["r"] or 0) >= rt and (r["margin"] or 0) >= mt and (r["z"] or 0) >= zt]
                bad = sum(1 for r in st if not r["consistent"])
                print(f"  r>={rt:.2f} margin>={mt:.2f} z>={zt}: strong {len(st):4d}  wrong {bad:3d}  "
                      f"({bad / max(1, len(st)):.1%})")
    for sh in (0.0, 0.01, 0.02, 0.03, 0.05):
        st = [r for r in sc if (r["r"] or 0) >= STRONG["r"] and (r["margin"] or 0) >= STRONG["margin"]
              and (r["z"] or 0) >= STRONG["z"] and _finite(r.get("sharp")) and r["sharp"] >= sh]
        bad = sum(1 for r in st if not r["consistent"])
        print(f"  STRONG r/margin/z and sharp>={sh:.2f}: strong {len(st):4d}  wrong {bad:3d}")


# ----------------------------------------------------------------------------- 5. decide
DEC_COLS = ["key", "animal", "session", "clip_dir", "y", "cause", "sub", "family", "camera", "raw_video",
            "seek_pred_s", "clip_dur_s", "sync_s", "baseline_s", "expected_s", "expected_nodrift_s",
            "expected_alt_s", "expected_interval", "measured_lag_s", "offset_s", "measured_r", "coarse_z", "margin",
            "strong", "residual_s", "partner", "p_offset_s", "p_strong", "local_n_pool", "local_n", "local_units",
            "local_n_strong", "local_median_s", "local_spread_s", "own_in_consensus", "content_file",
            "family_n_strong",
            "family_median_residual_s", "family_spread_s",
            "offset_used_s", "offset_source", "status", "reason", "new_raw_video", "new_seek_s", "v_clip_dur_s",
            "truncated_s", "orch90"]


def family_of(m):
    return f"{m['sub']}:{m['raw_video'].split('.')[-2]}"


def orchestrator_90(raw):
    """The brief's 90 re-derived from clips.csv as its description reads (for the comparison only)."""
    clips = read_csv(os.path.join(ALIGN, "clips.csv"))
    out = {}
    FLAT = {"20231103122100": (dt.datetime(2023, 11, 3, 3, 36, 6), dt.datetime(2023, 11, 3, 3, 52, 23)),
            "20231205222300": (dt.datetime(2023, 12, 5, 13, 37, 13), dt.datetime(2023, 12, 5, 13, 54, 3))}
    for r in clips:
        if r["in_discover"] != "True" or not r.get("raw_video") or not r.get("info_clip_start"):
            continue
        v = raw.vid.get(r["raw_video"].split("/")[-1])
        if v is None:
            continue
        cs = AC.parse_dt(r["info_clip_start"])
        k = clip_key(r["animal"], r["session"], r["clip_dir"])
        if v["stamp"] == STAMP_1013:
            out[k] = "roomD_1013_gap"
        elif v["stamp"] in FLAT and cs >= FLAT[v["stamp"]][1]:
            out[k] = "roomC_clock_step"
        elif utc_naive(v["utc_int"]) < FALL_UTC:
            if cs.date() == dt.date(2023, 11, 5) and cs.hour == 1:
                out[k] = "dst_ambiguous_hour"
            elif cs >= DST_DAY_2:
                out[k] = "dst_after_change"
    return out


def method_baseline(raw, ms, prof):
    """The Activity-vs-motion lag of a CORRECTLY cut clip is not 0: the cutter ignores DSI_sync_offset (the frames
    really start sync s after int(utc)) and Activity lags motion by some latency lam. So an aligned clip measures
    lag = -sync - lam. lam is estimated from every strong aligned measurement (clip controls expected at 0 and
    scan windows with |lag| <= 5 s); a clip's clock offset is then lag - (-sync - lam), in the cutter's convention."""
    import numpy as np
    xs = []
    for m in ms:
        if m["sub"] in ("dst_before_change", "step_before") and m.get("strong") == "True":
            lag = float(m["offset_s"])
            if abs(lag) <= 5:
                xs.append(-(lag + raw.vid[m["raw_video"].split("/")[-1]]["sync"]))
    ys = []
    for p in prof:
        if p.get("strong") == "True" and p.get("offset_s") not in (None, "") and abs(float(p["offset_s"])) <= 5:
            ys.append(-(float(p["offset_s"]) + raw.vid[p["file"]]["sync"]))
    allv = xs + ys
    lam = float(np.median(allv)) if allv else 0.0
    return lam, dict(lam=lam, n_clip_controls=len(xs), n_scan_windows=len(ys),
                     clip_controls_median=float(np.median(xs)) if xs else None,
                     scan_windows_median=float(np.median(ys)) if ys else None,
                     iqr=[float(np.percentile(allv, 25)), float(np.percentile(allv, 75))] if allv else None)


CONSENSUS_S = 3600    # family consensus for a weak clip: measurements on the same file and side within +-1 h
CLUSTER_S = 1.5       # ... that sit within +-1.5 s of the modal offset


def consensus(items):
    """Modal cluster of (t, offset, animal, strong, tag) measurements. Independent units: different animals, or the
    same animal with windows that do not overlap (> 3600 s apart). A spurious peak lands anywhere in +-4 h, so
    several independent units agreeing within +-1.5 s (and, checked by the caller, with the cause model) cannot be
    chance."""
    import numpy as np
    out = dict(n=0, units=0, n_strong=0, median=None, spread=None, n_pool=len(items))
    if not items:
        return out
    def n_units(mem):
        u = 0
        for a in {x[2] for x in mem}:
            last = None
            for tt in sorted(x[0] for x in mem if x[2] == a):
                if last is None or tt - last > 3600:
                    u += 1
                    last = tt
        return u

    def score(c):
        mem = [x for x in items if abs(x[1] - c) <= CLUSTER_S]
        return (n_units(mem), sum(1 for x in mem if x[3]), len(mem))

    best = max((x[1] for x in items), key=score)             # most independent units, then strong, then members
    mem = [x for x in items if abs(x[1] - best) <= CLUSTER_S]
    med = float(np.median([x[1] for x in mem]))
    mem = [x for x in items if abs(x[1] - med) <= CLUSTER_S]
    units = n_units(mem)
    out.update(n=len(mem), units=units, n_strong=sum(1 for x in mem if x[3]),
               median=float(np.median([x[1] for x in mem])),
               spread=float(max(abs(x[1] - med) for x in mem)))
    return out


def cmd_decide(args):
    """Classify every candidate. Offsets are baseline-corrected (method_baseline). A clip's expected offset is the
    cause model's interval [no drift, full drift] (model_interval); two-hypothesis clips (ambiguous DST hour,
    telemetry gap) have two intervals.

      strong own measurement within 3 s of an interval  -> use it (source 'measured')
      strong own measurement elsewhere                  -> flagged (not corrected)
      weak own measurement                              -> the partner animal's strong measurement in the same window
                                                           (same camera, same clock) if it fits an interval; else the
                                                           local consensus: >= 3 strong measurements (clips and scan
                                                           windows, both animals) on the same camera file and side,
                                                           centred within 3 h, all within 2 s of their median and
                                                           fitting an interval; else unresolved
    """
    import numpy as np
    raw = Raw()
    ms = read_csv(out_path("measure/clips.csv"))
    cstart = {c["key"]: c["info_clip_start"] for c in read_csv(out_path("plan/candidates.csv"))}
    for m in ms:
        m["info_clip_start"] = m.get("info_clip_start") or cstart[m["key"]]
    prof = read_csv(out_path("measure/profiles.csv")) if os.path.exists(out_path("measure/profiles.csv")) else []
    o90 = orchestrator_90(raw)
    lam, lam_info = method_baseline(raw, ms, prof)
    print("method baseline:", json.dumps(lam_info))
    base = lambda v: -v["sync"] - lam  # noqa: E731

    def dist(x, ivs):
        return min(0.0 if lo <= x <= hi else min(abs(x - lo), abs(x - hi)) for lo, hi in ivs)

    def cside(v, p, sub):
        g = content_next(raw, v, p, sub)
        return f"next:{g['name']}" if g is not None else ""

    # pool of ALL measurements (strong or weak) per (camera file, side, file holding the true frames): clips,
    # partners and scan windows, baseline-corrected, each tagged with its animal, window centre and strength
    pool = defaultdict(list)
    for m in ms:
        v = raw.vid[m["raw_video"].split("/")[-1]]
        t = nsec(AC.parse_dt(m["info_clip_start"]))
        rs = cside(v, float(m["seek_pred_s"]), m["sub"])
        if fnum(m.get("offset_s")) is not None and not m.get("err"):
            pool[(v["name"], m["sub"], rs)].append((t, float(m["offset_s"]) - base(v), m["animal"], m.get("strong") == "True",
                                                f"clip:{m['clip_dir']}"))
        if fnum(m.get("p_offset_s")) is not None and not m.get("p_err"):
            pool[(v["name"], m["sub"], rs)].append((t, float(m["p_offset_s"]) - base(v), m["partner"],
                                                m.get("p_strong") == "True", f"partner:{m['clip_dir']}"))
    for pr in prof:
        if pr.get("offset_s") and not pr.get("err"):
            v = raw.vid[pr["file"]]
            pool[(pr["file"], pr["sub"], cside(v, float(pr["pts_center"]), pr["sub"]))].append((nsec(AC.parse_dt(pr["t_center"])), float(pr["offset_s"]) - base(v),
                                                  pr["animal"], pr.get("strong") == "True", f"scan:{pr['t_center']}"))
    def intervals(v, p, sub):
        ivs = [model_interval(v, p, sub, raw)]
        if sub == "dst_ambiguous_hour":
            ivs.append(model_interval(v, p, "dst_after_change", raw))
        elif sub in ("step_in_gap", "step_no_gap_found"):
            ivs.append(model_interval(v, p, "step_after", raw))
        return ivs

    # per family (cause x file stamp): do its strong clip measurements agree with the cause model?
    fm_d, fm_c = defaultdict(list), defaultdict(set)
    for m in ms:
        if m.get("strong") == "True":
            v = raw.vid[m["raw_video"].split("/")[-1]]
            x = float(m["offset_s"]) - base(v)
            fm_d[family_of(m)].append(dist(x, intervals(v, float(m["seek_pred_s"]), m["sub"])))
            fm_c[family_of(m)].add(v["camera"])
    family_model = {f: dict(n=len(xs), max=max(xs), cameras=len(fm_c[f]),
                            tight=len(xs) >= 5 and len(fm_c[f]) >= 2 and max(xs) <= FAMILY_TIGHT_S)
                    for f, xs in fm_d.items()}
    rows, fam = [], defaultdict(list)
    for m in ms:
        v = raw.vid[m["raw_video"].split("/")[-1]]
        p = float(m["seek_pred_s"])
        sub = m["sub"]
        ivs = intervals(v, p, sub)
        rside = cside(v, p, sub)
        d = {k: m.get(k) for k in DEC_COLS if k in m}
        d["content_file"] = rside[5:] if rside else v["name"]
        b = base(v)
        meas = fnum(m.get("offset_s"))
        off = meas - b if meas is not None else None
        pof = fnum(m.get("p_offset_s")) - b if fnum(m.get("p_offset_s")) is not None else None
        strong, pstrong = m.get("strong") == "True", m.get("p_strong") == "True"
        d.update(family=family_of(m), sync_s=v["sync"], baseline_s=b, measured_lag_s=meas, offset_s=off,
                 expected_interval=" | ".join(f"{lo:+.1f}..{hi:+.1f}" for lo, hi in ivs),
                 measured_r=fnum(m.get("offset_r")), strong=strong, p_offset_s=pof, p_strong=pstrong,
                 orch90=o90.get(m["key"], ""))
        used, src, status, reason = None, None, None, None
        cdur_ = float(m["clip_dur_s"])
        if sub == "roomD_1013_gap" and p + cdur_ + max(hi for _, hi in ivs) <= 0:
            # the whole labelled span precedes the file's first frame under every model offset, and no file of this
            # camera covers it (the previous file ends ~3 min before this one's XML start): never recorded
            d.update(status="no_video", residual_s=None,
                     reason=(f"label span ends {-(p + cdur_ + max(hi for _, hi in ivs)):.0f} s before the first frame "
                             f"of {v['name']} (frames begin {v['gap']:.0f} s after its XML start; no {v['camera']} file "
                             f"covers the span); own measurement {_f(off, 1)} s, r={_f(m.get('offset_r'))}"))
            rows.append(d)
            continue
        if strong:
            d["residual_s"] = dist(off, ivs)
            if d["residual_s"] <= FAMILY_TOL_S:
                used, src = off, "measured"
                fam[d["family"]].append(d["residual_s"])
                if len(ivs) > 1:
                    reason = f"two hypotheses ({d['expected_interval']}): the measurement {off:+.1f} s picks one"
            else:
                status = "flagged"
                reason = (f"strong measurement {off:+.1f} s (r={float(m['offset_r']):.3f}, z={float(m['coarse_z']):.1f},"
                          f" margin={float(m['margin']):.3f}) is {d['residual_s']:.1f} s from the cause model "
                          f"{d['expected_interval']}")
        elif pstrong and dist(pof, ivs) <= FAMILY_TOL_S:
            used, src = pof, "partner"
            reason = (f"weak own measurement (r={_f(m.get('offset_r'))}, margin={_f(m.get('margin'))}); partner "
                      f"{m['partner']} on the same camera and window measures {pof:+.1f} s")
        else:
            t = nsec(AC.parse_dt(m["info_clip_start"]))
            items = list(pool[(v["name"], sub, rside)])
            cl = consensus([x for x in items if abs(x[0] - t) <= CONSENSUS_S and x[4] != f"clip:{m['clip_dir']}"])
            d.update(local_n=cl["n"], local_units=cl["units"], local_n_strong=cl["n_strong"],
                     local_median_s=cl["median"], local_spread_s=cl["spread"], local_n_pool=cl["n_pool"])
            d["own_in_consensus"] = (off is not None and cl["median"] is not None and abs(off - cl["median"]) <= CLUSTER_S)
            if (cl["median"] is not None and cl["n"] >= 3 and (cl["units"] >= 2 or cl["n_strong"] >= 1)
                    and dist(cl["median"], ivs) <= FAMILY_TOL_S):
                used, src = cl["median"], "family"
                reason = (f"weak own measurement (r={_f(m.get('offset_r'))}, margin={_f(m.get('margin'))}"
                          f"{', agrees' if d['own_in_consensus'] else ', does not agree'}); family on {v['name']} "
                          f"({sub}, +-{CONSENSUS_S / 3600:.0f} h): {cl['n']} of {cl['n_pool']} measurements from "
                          f"{cl['units']} independent units ({cl['n_strong']} strong) agree within {cl['spread']:.2f} s")
            else:
                # last resort: the cause model itself, when (a) this family's strong clips on other cameras agree
                # with it tightly, (b) its interval is narrow enough that any point in it is within the +-3 s
                # tolerance of the midpoint, (c) there is one hypothesis only, and (d) no local evidence contradicts it
                fm = family_model.get(d["family"])
                width = ivs[0][1] - ivs[0][0]
                contra = ((pstrong and dist(pof, ivs) > FAMILY_TOL_S) or
                          (cl["median"] is not None and (cl["units"] >= 2 or cl["n_strong"] >= 1)
                           and dist(cl["median"], ivs) > FAMILY_TOL_S))
                if fm and fm["tight"] and len(ivs) == 1 and width <= 2 * ALIGNED_TOL_S and not contra:
                    used, src = (ivs[0][0] + ivs[0][1]) / 2, "family_model"
                    reason = (f"weak own measurement (r={_f(m.get('offset_r'))}, sharp={_f(m.get('sharp'))}); no local "
                              f"consensus (n={cl['n']}, units={cl['units']}); family {d['family']}: {fm['n']} strong "
                              f"clips on {fm['cameras']} cameras all within {fm['max']:.2f} s of the cause model, whose "
                              f"interval here is {d['expected_interval']} -> its midpoint")
                else:
                    status = "unresolved"
                    reason = (f"weak measurement (r={_f(m.get('offset_r'))}, z={_f(m.get('coarse_z'), 1)}, "
                              f"margin={_f(m.get('margin'))}, sharp={_f(m.get('sharp'))}"
                              f"{', ' + m['err'] if m.get('err') else ''}); partner "
                              f"{'strong at ' + format(pof, '+.1f') if pstrong else 'weak or absent'}; family cluster "
                              f"n={cl['n']} of {cl['n_pool']}, units={cl['units']}"
                              + (f", median {cl['median']:+.1f} s" if cl["median"] is not None else "")
                              + f"; model {d['expected_interval']}"
                              + ("; local evidence contradicts the model" if contra else
                                 "" if fm and fm["tight"] else f"; family {d['family']} has no tight strong set"))
        if used is not None:
            d.update(offset_used_s=used, offset_source=src)
            if abs(used) <= ALIGNED_TOL_S:
                status = "aligned"
            else:
                # where the right frames are: same camera, nominal UTC u*. Causes that are a property of the defect
                # file itself (10-13 gap, clock step) must stay inside that file.
                u = v["u0"] + p + used
                g = raw.file_at(v["camera"], u)
                cdur = float(m["clip_dur_s"])
                if m["cause"] in ("roomD_1013_gap", "roomC_clock_step") and (g is None or g["name"] != v["name"]):
                    status = "no_video"
                    reason = (f"corrected pts {p + used:.1f} s is before the first frame of {v['name']}: the video "
                              f"for this label time was never recorded ({src} offset {used:+.1f} s)")
                elif g is None:
                    status, reason = "no_video", f"no {v['camera']} file covers the corrected time"
                else:
                    seek = u - g["u0"]
                    vcd = min(cdur, g["dur"] - seek)
                    status = "misaligned"
                    d.update(new_raw_video=g["relpath"], new_seek_s=seek, v_clip_dur_s=vcd,
                             truncated_s=max(0.0, cdur - vcd))
                    if vcd < cdur - 1.0:
                        reason = (reason + "; " if reason else "") + \
                            f"corrected span runs {cdur - vcd:.1f} s past the end of {g['name']}"
        d.update(status=status, reason=reason)
        rows.append(d)
    fstat = {f: dict(n=len(rs), med=float(np.median(rs)), spread=float(max(rs))) for f, rs in fam.items()}
    for d in rows:
        fs = fstat.get(d["family"], {})
        d.update(family_n_strong=fs.get("n", 0), family_median_residual_s=fs.get("med"),
                 family_spread_s=fs.get("spread"))
    write_csv("measure/decisions.csv", DEC_COLS, rows)
    fam_rows = [dict(family=f, **s) for f, s in sorted(fstat.items())]
    write_csv("measure/families.csv", ["family", "n", "med", "spread"], fam_rows)
    write_text("measure/baseline.json", json.dumps(lam_info, indent=1))
    ex = [dict(key=d["key"], animal=d["animal"], session=d["session"], clip_dir=d["clip_dir"], y=d["y"],
               cause=d["sub"], status=d["status"], reason=d["reason"], measured_offset_s=d["offset_s"],
               corr_r=d["measured_r"], expected_interval=d.get("expected_interval")) for d in rows
          if d["status"] in ("unresolved", "flagged", "no_video")]
    a, s, cdir = MISLABELLED_NEG
    ex.append(dict(key=clip_key(a, s, cdir), animal=a, session=s, clip_dir=cdir, y=0, cause="mislabelled_negative",
                   status="exclude_from_class0",
                   reason="class-0 clip contains an annotated Stage 3 seizure (raw_audit/annotations); drop it from "
                          "class 0 until it is re-added as a seizure clip"))
    write_csv("exclude.csv", ["key", "animal", "session", "clip_dir", "y", "cause", "status", "reason",
                              "measured_offset_s", "corr_r", "expected_interval"], ex)
    for f in fam_rows:
        print(f"family {f['family']:40s} n_strong={f['n']:3d} distance to model interval: median {f['med']:.2f} s, "
              f"max {f['spread']:.2f} s")
    print("status:", dict(Counter(d["status"] for d in rows)))
    for k, n in sorted(Counter((d["sub"], d["status"], d.get("offset_source") or "-") for d in rows).items()):
        print(f"  {k[0]:22s} {k[1]:12s} {k[2]:16s} {n}")


# ----------------------------------------------------------------------------- 6. recut / repro (the parent pipeline)
def cut_copy(raw_path, seek, vdur, dst):
    """cut_seizure_clips.cut_video_clip(mode='copy'), argument for argument (str() of Python floats)."""
    return run([FFMPEG, "-y", "-ss", str(float(seek)), "-i", raw_path, "-t", str(float(vdur)), "-c", "copy",
                "-avoid_negative_ts", "make_zero", dst], timeout=300)


def crop_encode(src, dst, animal):
    """crop_clips.crop_one(), argument for argument."""
    return run([FFMPEG, "-y", "-loglevel", "error", "-nostdin", "-i", src, "-vf", crop_expr(animal),
                "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", "-an", "-threads", "2",
                dst], timeout=900)


def make_clip(raw_path, seek, vdur, animal, cut_dst, out_dst):
    ensure_dir(cut_dst)
    ensure_dir(out_dst)
    for p in (cut_dst, out_dst):
        if os.path.exists(p):
            os.remove(p)
    r1 = cut_copy(raw_path, seek, vdur, cut_dst)
    if r1.returncode != 0 or not os.path.exists(cut_dst):
        return f"cut_failed: {r1.stderr.strip().splitlines()[-1][:200] if r1.stderr.strip() else r1.returncode}"
    r2 = crop_encode(cut_dst, out_dst, animal)
    if r2.returncode != 0 or not os.path.exists(out_dst):
        return f"crop_failed: {r2.stderr.strip()[:200]}"
    return "ok"


def keyframe_at_or_before(path, t):
    pk = AC.keyframes_near(path, t - 8, t + 1)
    kf = [p for p, k in pk if k and p <= t + 1e-6]
    return kf[-1] if kf else None


MAN_COLS = ["clip_key", "animal", "session", "clip_dir", "y", "cause", "sub", "measured_lag_s", "baseline_s",
            "measured_offset_s", "offset_used_s",
            "offset_source", "corr_r", "coarse_z", "margin", "partner", "partner_offset_s", "expected_offset_s",
            "expected_interval", "old_raw_video", "old_seek_pts", "old_keyframe_pts", "new_raw_video", "new_seek_pts",
            "new_keyframe_pts", "clip_dur_s", "v_clip_dur_s", "truncated_s", "status", "out", "reason"]


def cmd_recut(args):
    dec = read_csv(out_path("measure/decisions.csv"))
    todo = [d for d in dec if d["status"] == "misaligned"]
    print(f"{len(todo)} misaligned clips to re-cut")
    # drop outputs of earlier runs that are no longer misaligned (only ever inside OUT)
    keep = {clip_rel(d["animal"], d["session"], d["clip_dir"]) for d in todo}
    for sub_ in ("data_full", os.path.join("work", "cut")):
        for f in glob.glob(out_path(sub_, "Data_*", "*", "*", "video.mp4")):
            rel = os.path.relpath(os.path.dirname(f), out_path(sub_))
            if rel not in keep:
                os.remove(out_path(sub_, rel, "video.mp4"))
                print(f"  removed stale {sub_}/{rel}/video.mp4")

    def one(d):
        rel = clip_rel(d["animal"], d["session"], d["clip_dir"])
        out_dst = out_path("data_full", rel, "video.mp4")
        cut_dst = out_path("work", "cut", rel, "video.mp4")
        new_raw = os.path.join(RAW, d["new_raw_video"])
        seek, vdur = float(d["new_seek_s"]), float(d["v_clip_dur_s"])
        st = make_clip(new_raw, seek, vdur, d["animal"], cut_dst, out_dst)
        old_raw = os.path.join(RAW, d["raw_video"])
        return dict(clip_key=d["key"], animal=d["animal"], session=d["session"], clip_dir=d["clip_dir"], y=d["y"],
                    cause=d["cause"], sub=d["sub"], measured_lag_s=d["measured_lag_s"], baseline_s=d["baseline_s"],
                    measured_offset_s=d["offset_s"], offset_used_s=d["offset_used_s"],
                    offset_source=d["offset_source"], corr_r=d["measured_r"], coarse_z=d["coarse_z"],
                    margin=d["margin"], partner=d.get("partner"), partner_offset_s=d.get("p_offset_s"),
                    expected_offset_s=d["expected_s"], expected_interval=d.get("expected_interval"),
                    reason=d.get("reason"), old_raw_video=d["raw_video"],
                    old_seek_pts=float(d["seek_pred_s"]),
                    old_keyframe_pts=keyframe_at_or_before(old_raw, float(d["seek_pred_s"])),
                    new_raw_video=d["new_raw_video"], new_seek_pts=seek,
                    new_keyframe_pts=keyframe_at_or_before(new_raw, seek), clip_dur_s=d["clip_dur_s"],
                    v_clip_dur_s=vdur, truncated_s=d["truncated_s"], status=st,
                    out=os.path.relpath(out_dst, EEG_ROOT))

    with ThreadPoolExecutor(args.workers) as ex:
        man = list(ex.map(one, todo))
    write_csv("manifest.csv", MAN_COLS, man)
    print("recut status:", dict(Counter(m["status"] for m in man)))


def cmd_repro(args):
    rep = read_csv(out_path("plan/repro.csv"))
    clips = {clip_key(r["animal"], r["session"], r["clip_dir"]): r for r in read_csv(os.path.join(ALIGN, "clips.csv"))}
    raw = Raw()
    out = []
    for r in rep:
        c = clips[r["key"]]
        v = raw.vid[c["raw_video"].split("/")[-1]]
        rel = clip_rel(r["animal"], r["session"], r["clip_dir"])
        # the cutter's seek at full precision (clip start - Central(int(utc))); info.txt's "Video seek" is 2-decimal
        cs = AC.parse_dt(c["info_clip_start"])
        seek = cutter_seek(cs, v)
        cdur = (AC.parse_dt(c["info_clip_end"]) - cs).total_seconds()
        vdur = min(cdur, v["dur"] - seek)
        st = make_clip(os.path.join(RAW, v["relpath"]), seek, vdur, r["animal"],
                       out_path("repro", "work", rel, "video.mp4"), out_path("repro", "data_full", rel, "video.mp4"))
        out.append(dict(r, seek=seek, v_clip_dur=vdur, status=st))
        print(r["key"], st, flush=True)
    write_csv("repro/repro_runs.csv", ["key", "animal", "session", "clip_dir", "kind", "raw_video", "seek",
                                       "v_clip_dur", "status"], out)


# ----------------------------------------------------------------------------- 7. verify
def probe(path):
    r = run([FFPROBE, "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
             "stream=width,height,nb_frames,nb_read_frames,duration,r_frame_rate", "-of", "json", path], timeout=300)
    s = (json.loads(r.stdout or "{}").get("streams") or [{}])[0]
    dec = run([FFMPEG, "-nostdin", "-v", "error", "-i", path, "-f", "null", "-"], timeout=300)
    return dict(w=int(s.get("width", 0)), h=int(s.get("height", 0)), nb=int(s.get("nb_frames", 0) or 0),
                nread=int(s.get("nb_read_frames", 0) or 0), dur=float(s.get("duration", 0) or 0),
                fps=s.get("r_frame_rate"), dec_errors=len([x for x in dec.stderr.splitlines() if x.strip()]))


def frames_gray(path, w, h, ss=None, t=None, vf=None, scale=None):
    """All frames (gray) of a clip or a raw window; returns (frames, pts)."""
    chain = []
    if vf:
        chain.append(vf)
    ow, oh = w, h
    if scale:
        ow, oh = max(8, (w // scale) // 2 * 2), max(8, (h // scale) // 2 * 2)
        chain.append(f"scale={ow}:{oh}:flags=area")
    return AC.decode(path, ss, t, ",".join(chain) or "null", ow, oh)


def zrows(X):
    import numpy as np
    X = X.reshape(len(X), -1).astype(np.float64)
    X = X - X.mean(1, keepdims=True)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def ncc_vs_raw(clip_path, raw_path, key_pts, animal, dur, scale=2, search=15):
    """Frame-sequence NCC of the clip against the raw frames from its keyframe, over +-search frames."""
    import numpy as np
    x, y, w, h = crop_box(animal)
    cf, _ = frames_gray(clip_path, w, h, scale=scale)
    ss = max(0.0, key_pts - search / 15.0 - 0.5)
    rf, rpts = frames_gray(raw_path, w, h, ss=ss, t=dur + 2 * search / 15.0 + 2.0, vf=f"crop={w}:{h}:{x}:{y}",
                           scale=scale)
    if len(cf) < 10 or len(rf) < len(cf):
        return dict(ncc_err=f"frames clip={len(cf)} raw={len(rf)}")
    C, R = zrows(cf), zrows(rf)
    M = len(C)
    D = len(R) - M + 1
    if D < 1:
        M = len(R) - 2 * search
        C = C[:M]
        D = len(R) - M + 1
    G = C @ R.T
    ii = np.arange(M)[:, None]
    S = G[ii, ii + np.arange(D)[None, :]].mean(0)
    k = int(np.argmax(S))
    k0 = int(np.argmin(np.abs(rpts - key_pts)))       # the raw index of the keyframe the cut should start at
    per = G[np.arange(M), np.arange(M) + k]
    return dict(ncc_mean=float(S[k]), ncc_min=float(per.min()), ncc_best_minus_key_frames=k - k0,
                ncc_at_key=float(S[k0]) if 0 <= k0 < D else None, n_clip_frames=len(cf))


def clip_motion_vs_activity(clip_path, animal, t_frame0, edf_rel, ACT, lag_max=90):
    """1 Hz motion of the clip file itself on the label clock vs EDF Activity: best lag in +-lag_max s, r at 0."""
    import numpy as np
    x, y, w, h = crop_box(animal)
    cf, cpts = frames_gray(clip_path, w, h, scale=4)
    if len(cf) < 30:
        return dict(err="few_frames")
    mot = np.abs(np.diff(cf.astype(np.float32), axis=0)).mean((1, 2))
    tj = t_frame0 + (cpts[1:] - cpts[0])
    b0 = int(math.floor(tj.min()))
    nb = int(math.floor(tj.max())) - b0 + 1
    k = np.floor(tj - b0).astype(int)
    s, c = np.bincount(k, weights=mot, minlength=nb), np.bincount(k, minlength=nb)
    mb = mot_transform(np.where(c >= 8, s / np.maximum(c, 1), np.nan))
    a = ACT.window(edf_rel, b0 - lag_max, b0 + nb + lag_max)
    rs = {}
    for L in range(-lag_max, lag_max + 1):
        # motion(t) vs Activity(t - L): the clip shows what happened L s after its label time
        seg = a[lag_max + L: lag_max + L + nb]
        r, n = pearson(mb, seg)
        if np.isfinite(r):
            rs[L] = r
    if not rs:
        return dict(err="activity_flat", act_nonzero=int((np.nan_to_num(a) >= ACT_ON).sum()))
    vals = np.array(list(rs.values()))
    Lb = max(rs, key=rs.get)
    med, sd = float(np.median(vals)), float(np.std(vals)) + 1e-9
    return dict(lag=Lb, r_best=rs[Lb], r0=rs.get(0, float("nan")), z_best=(rs[Lb] - med) / sd,
                act_nonzero=int((np.nan_to_num(a[lag_max: lag_max + nb]) >= ACT_ON).sum()), bins=nb)


VER_COLS = ["clip_key", "animal", "cause", "sub", "offset_used_s", "baseline_s", "status", "decodes",
            "win_new_lag", "win_new_r", "win_new_r0", "win_new_z", "win_orig_lag", "win_orig_r", "win_orig_r0",
            "win_orig_z", "win_act_nonzero", "window_check", "window_clear", "win_new_within_3s", "win_orig_within_3s", "nb_expected", "orig_nb_expected",
            "frames_as_expected", "orig_truncated_s", "dec_errors", "w", "h",
            "orig_w", "orig_h", "nb", "orig_nb", "d_frames", "dur", "orig_dur", "within_keyframe", "ncc_mean",
            "ncc_min", "ncc_best_minus_key_frames", "frames_match_raw", "new_lag", "new_r_best", "new_r0",
            "new_z_best", "new_act_nonzero", "orig_lag", "orig_r_best", "orig_r0", "orig_z_best", "orig_act_nonzero",
            "new_aligned", "orig_aligned", "activity_check"]


def _finite(x):
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def window_lag(TL, ACT, camera, animal, edf_rel, off, t0, t1, shift, half=900, lag_max=120):
    """Activity over the label span +- half vs the raw crop motion at nominal u = t + off + shift + L, for
    L in +-lag_max: best L, its r, r at L = 0 and the robust z of the peak on the lag curve."""
    import numpy as np
    a = ACT.window(edf_rel, int(t0 - half), int(t1 + half))
    L = len(a)
    U0 = int(t0 - half + off + shift) - lag_max
    m = mot_transform(TL.bins(camera, animal, U0, L + 2 * lag_max))
    if np.isfinite(a).sum() < 120 or np.nanstd(a) == 0:
        return dict(err="activity_flat")
    r, n = masked_ncc(a, m)
    r = np.where(n >= MIN_OVERLAP * np.isfinite(a).sum(), r, np.nan)
    if not np.isfinite(r).any():
        return dict(err="no_video")
    k = int(np.nanargmax(r))
    med = float(np.nanmedian(r))
    mad = float(np.nanmedian(np.abs(r - med))) * 1.4826 + 1e-9
    return dict(lag=k - lag_max, r_best=float(r[k]), r0=float(r[lag_max]) if np.isfinite(r[lag_max]) else None,
                z_best=(float(r[k]) - med) / mad, act_nonzero=int((np.nan_to_num(a) >= ACT_ON).sum()))


def win_peak(res, thr_z=4.0, thr_r=0.1):
    return not res.get("err") and _finite(res.get("r_best")) and res["r_best"] >= thr_r and res["z_best"] >= thr_z


def act_peak(res, thr_r=0.4, thr_z=3.5):
    """Is there a clear motion-vs-Activity peak for this clip (r >= 0.4 and 3.5 SD above the lag curve)?"""
    return (not res.get("err") and _finite(res.get("r_best")) and res["r_best"] >= thr_r
            and res["z_best"] >= thr_z)


def act_verdict(res, base):
    """True: a clear peak within +-3 s of the method baseline (the lag a correctly cut clip shows, -sync - lam);
    False: no clear peak there (none at all, or elsewhere)."""
    return bool(act_peak(res) and abs(res["lag"] - base) <= ALIGNED_TOL_S)


def cmd_verify(args):
    raw = Raw()
    ACT = Activity()
    TL = Timeline(raw)
    man = read_csv(out_path("manifest.csv"))
    clips = {clip_key(r["animal"], r["session"], r["clip_dir"]): r for r in read_csv(os.path.join(ALIGN, "clips.csv"))}

    def one(m):
        rel = clip_rel(m["animal"], m["session"], m["clip_dir"])
        newp = out_path("data_full", rel, "video.mp4")
        orig = os.path.join(FULL, rel, "video.mp4")
        c = clips[m["clip_key"]]
        row = {k: m.get(k) for k in VER_COLS if k in m}
        if m["status"] != "ok" or not os.path.exists(newp):
            row["decodes"] = False
            return row
        pn, po = probe(newp), probe(orig)
        # frames a copy-mode cut from keyframe K for vdur s of the seek should hold (CFR 15 fps after the crop step)
        nk_, ok0 = float(m["new_keyframe_pts"] or m["new_seek_pts"]), float(m["old_keyframe_pts"] or m["old_seek_pts"])
        vo = raw.vid[m["old_raw_video"].split("/")[-1]]
        o_vdur = min(float(m["clip_dur_s"]), vo["dur"] - float(m["old_seek_pts"]))
        exp_new = (float(m["new_seek_pts"]) - nk_ + float(m["v_clip_dur_s"])) * 15
        exp_old = (float(m["old_seek_pts"]) - ok0 + o_vdur) * 15
        row.update(decodes=pn["nread"] > 0 and pn["dec_errors"] == 0, dec_errors=pn["dec_errors"], w=pn["w"], h=pn["h"],
                   orig_w=po["w"], orig_h=po["h"], nb=pn["nread"], orig_nb=po["nread"], d_frames=pn["nread"] - po["nread"],
                   dur=pn["dur"], orig_dur=po["dur"], nb_expected=exp_new, orig_nb_expected=exp_old,
                   frames_as_expected=abs(pn["nread"] - exp_new) <= 2,
                   orig_truncated_s=max(0.0, float(m["clip_dur_s"]) - o_vdur),
                   within_keyframe=abs(pn["nread"] - po["nread"]) <= 33 or
                   abs((pn["nread"] - exp_new) - (po["nread"] - exp_old)) <= 2)
        newraw = os.path.join(RAW, m["new_raw_video"])
        nk = float(m["new_keyframe_pts"]) if m.get("new_keyframe_pts") else float(m["new_seek_pts"])
        try:
            row.update(ncc_vs_raw(newp, newraw, nk, m["animal"], pn["dur"]))
            dk = row.get("ncc_best_minus_key_frames")
            row["frames_match_raw"] = bool((row.get("ncc_mean") or 0) >= 0.95 and dk is not None and abs(dk) <= 1)
        except Exception as e:  # noqa
            row["ncc_err"] = repr(e)
        # label-clock time of frame 0 = clip start - (seek - keyframe)
        cs = nsec(AC.parse_dt(c["info_clip_start"]))
        ok_ = float(m["old_keyframe_pts"]) if m.get("old_keyframe_pts") else float(m["old_seek_pts"])
        edf = c.get("raw_edf")
        # the new clip is placed on the label clock by the corrected mapping: frame 0 <-> clip start - (seek - key)
        rn = clip_motion_vs_activity(newp, m["animal"], cs - (float(m["new_seek_pts"]) - nk), edf, ACT)
        ro = clip_motion_vs_activity(orig, m["animal"], cs - (float(m["old_seek_pts"]) - ok_), edf, ACT)
        row.update(new_lag=rn.get("lag"), new_r_best=rn.get("r_best"), new_r0=rn.get("r0"), new_z_best=rn.get("z_best"),
                   new_act_nonzero=rn.get("act_nonzero"), orig_lag=ro.get("lag"), orig_r_best=ro.get("r_best"),
                   orig_r0=ro.get("r0"), orig_z_best=ro.get("z_best"), orig_act_nonzero=ro.get("act_nonzero"))
        base = float(m["baseline_s"])
        row["baseline_s"] = base
        # window level (primary): the raw crop motion at the corrected location (identical to the new clip's frames,
        # see the NCC above) and at the original location, against Activity over the label span +- 15 min
        v = raw.vid[m["old_raw_video"].split("/")[-1]]
        t0 = nsec(AC.parse_dt(c["info_clip_start"]))
        t1 = nsec(AC.parse_dt(c["info_clip_end"]))
        wn = window_lag(TL, ACT, v["camera"], m["animal"], edf, v["off"], t0, t1, float(m["offset_used_s"]))
        wo = window_lag(TL, ACT, v["camera"], m["animal"], edf, v["off"], t0, t1, 0.0)
        row.update(win_new_lag=wn.get("lag"), win_new_r=wn.get("r_best"), win_new_r0=wn.get("r0"),
                   win_new_z=wn.get("z_best"), win_orig_lag=wo.get("lag"), win_orig_r=wo.get("r_best"),
                   win_orig_r0=wo.get("r0"), win_orig_z=wo.get("z_best"), win_act_nonzero=wn.get("act_nonzero"))
        # best lag of the +-120 s search within +-3 s of the method baseline (chance: 7 of 241 lags, ~3%);
        # 'clear' additionally needs the peak >= 4 robust SD above the lag curve
        lag_ok = lambda w: (not w.get("err") and _finite(w.get("r_best")) and w["r_best"] >= 0.1  # noqa: E731
                            and abs(w["lag"] - base) <= ALIGNED_TOL_S)
        new_ok, orig_ok = lag_ok(wn), lag_ok(wo)
        # fail = a clear peak (z >= 4) somewhere else; weak = no clear peak (too little Activity around the clip)
        row["window_check"] = ("indeterminate" if wn.get("err") else "pass" if new_ok and not orig_ok else
                               "both_aligned" if new_ok and orig_ok else "fail" if win_peak(wn) else "weak")
        row["win_new_within_3s"] = bool(not wn.get("err") and _finite(wn.get("lag")) and abs(wn["lag"] - base) <= 3)
        row["win_orig_within_3s"] = bool(not wo.get("err") and _finite(wo.get("lag")) and abs(wo["lag"] - base) <= 3)
        row["window_clear"] = bool(new_ok and not orig_ok and win_peak(wn))
        row["new_aligned"] = act_verdict(rn, base)
        row["orig_aligned"] = act_verdict(ro, base)
        # same label span, same Activity: a clear peak at 0 for the new clip but not for the original is a pass;
        # no clear peak for the new clip means the Activity is too quiet over the clip to decide
        row["activity_check"] = ("pass" if row["new_aligned"] and not row["orig_aligned"] else
                                 "both_aligned" if row["new_aligned"] else
                                 "fail" if act_peak(rn) else "indeterminate")
        return row

    with ThreadPoolExecutor(args.workers) as ex:
        rows = list(ex.map(one, man))
    write_csv("verify/verify.csv", VER_COLS, rows)
    print("decodes:", Counter(str(r.get("decodes")) for r in rows), " within_keyframe:",
          Counter(str(r.get("within_keyframe")) for r in rows), " frames_match_raw:",
          Counter(str(r.get("frames_match_raw")) for r in rows), " frames_as_expected:",
          Counter(str(r.get("frames_as_expected")) for r in rows), " window_check:",
          Counter(str(r.get("window_check")) for r in rows), " clip activity_check:",
          Counter(str(r.get("activity_check")) for r in rows))

    # reproduction controls: our pipeline at the original seek vs the original data_full clip
    rep = read_csv(out_path("repro/repro_runs.csv")) if os.path.exists(out_path("repro/repro_runs.csv")) else []
    rr = []
    for r in rep:
        rel = clip_rel(r["animal"], r["session"], r["clip_dir"])
        a, b = out_path("repro", "data_full", rel, "video.mp4"), os.path.join(FULL, rel, "video.mp4")
        row = dict(key=r["key"], status=r["status"])
        if r["status"] == "ok":
            pa, pb = probe(a), probe(b)
            row.update(w=pa["w"], h=pa["h"], orig_w=pb["w"], orig_h=pb["h"], nb=pa["nread"], orig_nb=pb["nread"],
                       dur=pa["dur"], orig_dur=pb["dur"], fps=pa["fps"], orig_fps=pb["fps"])
            row.update(compare_clips(a, b, pa["w"], pa["h"]))
        rr.append(row)
        print("repro", json.dumps(row, default=str), flush=True)
    write_csv("verify/repro.csv", ["key", "status", "w", "h", "orig_w", "orig_h", "nb", "orig_nb", "dur", "orig_dur",
                                   "fps", "orig_fps", "frame_shift", "ncc_mean", "ncc_min", "mae_mean", "mae_max",
                                   "n_compared"], rr)


def compare_clips(a, b, w, h):
    """Frame-by-frame agreement of two clips of the same geometry (full resolution, gray)."""
    import numpy as _np
    fa, _ = AC.decode(a, None, None, "null", w, h)
    fb, _ = AC.decode(b, None, None, "null", w, h)
    best = None
    for sh in range(-3, 4):
        A_ = fa[max(0, sh): len(fa) + min(0, sh)] if sh >= 0 else fa[:len(fa) + sh]
        B_ = fb[max(0, -sh): len(fb)]
        if sh >= 0:
            A_, B_ = fa[sh:], fb
        else:
            A_, B_ = fa, fb[-sh:]
        n = min(len(A_), len(B_))
        if n < 10:
            continue
        za, zb = zrows(A_[:n]), zrows(B_[:n])
        ncc = (za * zb).sum(1)
        if best is None or ncc.mean() > best["ncc_mean"]:
            mae = _np.abs(A_[:n].astype(_np.float32) - B_[:n].astype(_np.float32)).mean((1, 2))
            best = dict(frame_shift=sh, ncc_mean=float(ncc.mean()), ncc_min=float(ncc.min()),
                        mae_mean=float(mae.mean()), mae_max=float(mae.max()), n_compared=n)
    return best or {}


# ----------------------------------------------------------------------------- 8. report
def cmd_report(args):
    import numpy as np
    raw = Raw()
    dec = read_csv(out_path("measure/decisions.csv"))
    man = read_csv(out_path("manifest.csv")) if os.path.exists(out_path("manifest.csv")) else []
    ver = read_csv(out_path("verify/verify.csv")) if os.path.exists(out_path("verify/verify.csv")) else []
    rep = read_csv(out_path("verify/repro.csv")) if os.path.exists(out_path("verify/repro.csv")) else []
    exc = read_csv(out_path("exclude.csv"))
    fam = read_csv(out_path("measure/families.csv"))
    base = json.load(open(out_path("measure/baseline.json")))
    cal = sorted(glob.glob(out_path("calibrate", "*.csv")))
    names = {0: "non-seizure", 1: "Stage 2", 2: "Stage 3", 3: "Stage 4", 4: "Stage 5"}
    ycount = lambda rows: {names[int(k)]: v for k, v in sorted(Counter(r["y"] for r in rows).items())}  # noqa: E731
    o90 = {d["key"]: d["orch90"] for d in dec if d.get("orch90")}
    by = defaultdict(list)
    for d in dec:
        by[d["status"]].append(d)
    mis = {d["key"] for d in by["misaligned"]}
    wrong_time = [d for d in dec if d["status"] in ("misaligned", "no_video")]
    wt = {d["key"] for d in wrong_time}
    ver_ok = [v for v in ver if v.get("decodes") == "True"]
    s = dict(
        candidates=len(dec), candidates_by_sub=dict(Counter(d["sub"] for d in dec)),
        by_status={k: len(v) for k, v in by.items()},
        by_sub_status_source={f"{k[0]}|{k[1]}|{k[2]}": n for k, n in
                              sorted(Counter((d["sub"], d["status"], d.get("offset_source") or "-") for d in dec).items())},
        wrong_time=len(wrong_time), wrong_time_by_class=ycount(wrong_time),
        wrong_time_by_cause=dict(Counter(d["sub"] for d in wrong_time)),
        misaligned=len(mis), misaligned_by_class=ycount(by["misaligned"]),
        misaligned_by_cause=dict(Counter(d["sub"] for d in by["misaligned"])),
        offsets_used_by_cause={c: dict(n=len(xs), median=float(np.median(xs)), min=float(min(xs)), max=float(max(xs)))
                               for c, xs in ((c, [float(d["offset_used_s"]) for d in by["misaligned"] if d["sub"] == c])
                                             for c in sorted({d["sub"] for d in by["misaligned"]}))},
        excluded=len(exc), excluded_by_status=dict(Counter(e["status"] for e in exc)), excluded_by_class=ycount(exc),
        orch90=len(o90), orch90_by_cause=dict(Counter(o90.values())),
        orch90_wrong_time=len(set(o90) & wt), orch90_misaligned=len(set(o90) & mis),
        orch90_status=dict(Counter(next(d["status"] for d in dec if d["key"] == k) for k in o90)),
        orch90_not_wrong_time=[(k, next(d["status"] for d in dec if d["key"] == k), o90[k]) for k in sorted(set(o90) - wt)],
        wrong_time_not_in_orch90=[(k, next(d["sub"] for d in dec if d["key"] == k)) for k in sorted(wt - set(o90))],
        method_baseline=base, families=fam, calibration_files=[os.path.basename(c) for c in cal],
        recut=dict(Counter(m["status"] for m in man)),
        verify=dict(n=len(ver), decodes=dict(Counter(v.get("decodes") for v in ver)),
                    within_keyframe=dict(Counter(v.get("within_keyframe") for v in ver)),
                    frames_match_raw=dict(Counter(v.get("frames_match_raw") for v in ver)),
                    ncc_mean_min=min((float(v["ncc_mean"]) for v in ver_ok if v.get("ncc_mean")), default=None),
                    frames_as_expected=dict(Counter(v.get("frames_as_expected") for v in ver)),
                    window_check=dict(Counter(v.get("window_check") for v in ver)),
                    window_clear=dict(Counter(v.get("window_clear") for v in ver)),
                    window_best_lag_within_3s=dict(new=sum(v.get("win_new_within_3s") == "True" for v in ver),
                                                   orig=sum(v.get("win_orig_within_3s") == "True" for v in ver)),
                    window_r0_new_gt_orig=sum(1 for v in ver if _finite(v.get("win_new_r0")) and _finite(v.get("win_orig_r0"))
                                              and float(v["win_new_r0"]) > float(v["win_orig_r0"])),
                    window_r0_pairs=sum(1 for v in ver if _finite(v.get("win_new_r0")) and _finite(v.get("win_orig_r0"))),
                    clip_activity_check=dict(Counter(v.get("activity_check") for v in ver))),
        repro=rep)
    write_text("summary.json", json.dumps(s, indent=1, default=str))
    L = ["ttg_recut: labelled clips whose video is from the wrong time, measured and re-cut", "",
         f"candidates {len(dec)}: {dict(Counter(d['sub'] for d in dec))}",
         f"status: {s['by_status']}",
         f"wrong-time video: {len(wrong_time)} = misaligned (re-cut) {len(mis)} + no video recorded "
         f"{len(by['no_video'])}; by class {s['wrong_time_by_class']}",
         f"re-cut by cause: {s['misaligned_by_cause']}", "offsets used (s): " +
         "; ".join(f"{c} n={v['n']} median {v['median']:+.1f} [{v['min']:+.1f}, {v['max']:+.1f}]"
                   for c, v in s["offsets_used_by_cause"].items()),
         f"excluded {len(exc)}: {s['excluded_by_status']} by class {s['excluded_by_class']}",
         f"method baseline: lambda = {base['lam']:.3f} s (Activity lags motion; + the ignored DSI_sync_offset), "
         f"from {base['n_clip_controls']} clip controls and {base['n_scan_windows']} scan windows", "",
         f"orchestrator's 90: {s['orch90_by_cause']}; here {s['orch90_wrong_time']} are wrong-time "
         f"({s['orch90_misaligned']} re-cut), status {s['orch90_status']}"]
    for k, st, c in s["orch90_not_wrong_time"]:
        L.append(f"   in the 90, not wrong-time here: {k}  [{c}] -> {st}")
    for k, c in s["wrong_time_not_in_orch90"]:
        L.append(f"   wrong-time here, not in the 90: {k}  [{c}]")
    L += ["", f"re-cut: {s['recut']}", f"verify: {s['verify']}", "reproduction controls (unaffected clips):"]
    for r in rep:
        L.append(f"   {r['key']}: frames {r.get('nb')}/{r.get('orig_nb')} shift {r.get('frame_shift')} "
                 f"NCC mean {r.get('ncc_mean')} MAE mean {r.get('mae_mean')} max {r.get('mae_max')}")
    write_text("summary.txt", "\n".join(L) + "\n")
    print("\n".join(L))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("plan")
    p = sp.add_parser("motion")
    p.add_argument("--index", type=int)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--threads", type=int, default=0)
    p = sp.add_parser("activity")
    p.add_argument("--workers", type=int, default=4)
    sp.add_parser("measure")
    p = sp.add_parser("calibrate")
    p.add_argument("--win", type=float, default=3600)
    p.add_argument("--step", type=float, default=600)
    sp.add_parser("decide")
    p = sp.add_parser("recut")
    p.add_argument("--workers", type=int, default=4)
    sp.add_parser("repro")
    p = sp.add_parser("verify")
    p.add_argument("--workers", type=int, default=4)
    sp.add_parser("report")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    {"plan": cmd_plan, "motion": cmd_motion, "activity": cmd_activity, "measure": cmd_measure,
     "calibrate": cmd_calibrate, "decide": cmd_decide,
     "recut": cmd_recut, "repro": cmd_repro, "verify": cmd_verify, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
