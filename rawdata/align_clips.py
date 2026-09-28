#!/usr/bin/env python3
"""Map the LABELLED CLIPS onto the raw DSI/Noldus recordings, read-only.

Sub-commands
------------
  inventory   parse every labelled clip dir of the 20 labelled animals (data/ and data_full/),
              read each info.txt, replicate train_pooled.discover(), compare with the frame-cache
              index, and map every clip onto its raw mp4 and raw EDF by recorded times only
              (no decoding)                                    -> clips.csv, inventory.json
  sample      stratified sample for the image/EEG check      -> sample.csv
  list        print the array index -> animal table used by `match --index`
  match       for the sampled clips of one animal: recover the crop box by normalised template
              matching over a small grid of scales, the time offset by frame-sequence correlation
              within +-30 s (and a +-300 s wide check), the EEG lag by 1 kHz cross-correlation,
              and a coarse EDF-Activity vs video-motion clock check  -> units/<animal>.jsonl
  merge       tables and the summary                         -> matches.csv, crop_boxes.csv,
              per_animal.csv, eeg.csv, align_summary.txt, align.json

Reads   /work/mech-ai/alloydas/EEG/Data (raw, never written), $EEG_ROOT/data, $EEG_ROOT/data_full,
        $EEG_ROOT/cache_frames/f32s224/index.json, $EEG_ROOT/output/raw_audit/files.csv
Writes  ONLY $EEG_ROOT/output/raw_audit/align/ (out_path() refuses anything else).

Needs numpy + scipy (the miniconda3 base interpreter imports them in 0.2 s; the `eeg` conda env
takes ~7 min to page numpy in from /work/mech-ai-scratch) and ffmpeg/ffprobe from
/work/mech-ai/alloy/miniconda3/bin. Video frames are decoded only in short windows.
"""
import argparse
import ast
import csv
import datetime as dt
import glob
import json
import os
import re
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from zoneinfo import ZoneInfo

RAW = os.path.realpath(os.environ.get("RAW_ROOT", "/work/mech-ai/alloydas/EEG/Data"))
EEG_ROOT = os.path.realpath(os.environ.get("EEG_ROOT", "/work/mech-ai-scratch/alloy/EEG"))
AUDIT = os.path.join(EEG_ROOT, "output", "raw_audit")
OUT = os.path.join(AUDIT, "align")
DATA = os.path.join(EEG_ROOT, "data")
FULL = os.path.join(EEG_ROOT, "data_full")
CACHE_INDEX = os.path.join(EEG_ROOT, "cache_frames", "f32s224", "index.json")
TRAIN_POOLED = os.path.join(EEG_ROOT, "train_pooled.py")
FFBIN = "/work/mech-ai/alloy/miniconda3/bin"
FFMPEG = os.environ.get("FFMPEG", os.path.join(FFBIN, "ffmpeg"))
FFPROBE = os.environ.get("FFPROBE", os.path.join(FFBIN, "ffprobe"))
CENTRAL = ZoneInfo("America/Chicago")
UTC = dt.timezone.utc
ALIGN_VERSION = 3          # per-clip match logic; a bump re-runs every sampled clip

NARROW_S = 30.0            # time-offset search window around the predicted seek (brief: +-30 s)
WIDE_S = 300.0             # wide check + Activity/motion window
DS_NARROW = 4              # downsample factor for the +-30 s frame correlation
DS_WIDE = 8                # downsample factor for the +-300 s check
SCALES = (0.5, 0.6, 0.667, 0.75, 0.8, 0.9, 1.0, 1.1, 1.2, 1.25)
EEG_WIN_S = 30.0           # EEG lag search window
LONG_LAG_S = 600           # Activity/motion long-lag search, main stratum
LONG_LAG_DEFECT_S = 4 * 3600   # ... and for clips cut from timing-defect files (175 min missing on 10-13)
LONG_LAG_CLOCK_S = 1800    # ... and for the clock-anomaly day folders
CLOCK_DAYS = ("10-19-2023", "11-17-2023")
WIDE_MULT = {"main": 1, "defect": 4, "clock": 3}


# ----------------------------------------------------------------------------- guards / io
def _under(path, root):
    p, r = os.path.realpath(path), os.path.realpath(root)
    return p == r or p.startswith(r + os.sep)


def out_path(*parts):
    """Every write goes through here: under OUT only, never under RAW or elsewhere in the repo."""
    p = os.path.join(OUT, *parts)
    if not _under(p, OUT) or _under(p, RAW) or _under(p, DATA) or _under(p, FULL):
        raise SystemExit(f"refusing to write outside {OUT}: {p}")
    return p


def write_text(relname, text):
    p = out_path(relname)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, p)


def write_csv(relname, cols, rows):
    p = out_path(relname)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _cell(r.get(k)) for k in cols})
    os.replace(tmp, p)


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, (list, tuple)):
        return "|".join(map(str, v))
    return v


def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def ropen(path):
    if not (_under(path, RAW) or _under(path, FULL) or _under(path, DATA) or _under(path, EEG_ROOT)):
        raise SystemExit(f"unexpected read path {path}")
    return open(path, "rb")


# ----------------------------------------------------------------------------- time helpers
def parse_dt(s):
    if s in (None, ""):
        return None
    if isinstance(s, dt.datetime):
        return s
    s = str(s).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return dt.datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None


def iso(x):
    return None if x is None else x.isoformat(sep=" ")


def central_of_epoch(epoch):
    """Naive Central wall time of a UTC epoch (what datetime.fromtimestamp gave on the cutter box)."""
    return dt.datetime.fromtimestamp(epoch, tz=UTC).astimezone(CENTRAL).replace(tzinfo=None)


def local_to_utc(naive_local):
    """Naive Central wall time -> naive UTC (fold=0 in the repeated 01:00-02:00 of 2023-11-05)."""
    return naive_local.replace(tzinfo=CENTRAL).astimezone(UTC).replace(tzinfo=None)


def utc_to_local(naive_utc_dt):
    return naive_utc_dt.replace(tzinfo=UTC).astimezone(CENTRAL).replace(tzinfo=None)


def utc_offset_s(naive_local):
    return naive_local.replace(tzinfo=CENTRAL).utcoffset().total_seconds()


def naive_utc(epoch):
    return dt.datetime.fromtimestamp(epoch, tz=UTC).replace(tzinfo=None)


def secs(a, b):
    return None if a is None or b is None else (a - b).total_seconds()


def fnum(x):
    try:
        return None if x in (None, "") else float(x)
    except ValueError:
        return None


def ambiguous_dst(t):
    return t is not None and t.date() == dt.date(2023, 11, 5) and t.hour == 1


# ----------------------------------------------------------------------------- parent-repo config
def parent_config():
    """ROOTS and STAGE from train_pooled.py, read with ast (importing it would import torch)."""
    src = open(TRAIN_POOLED).read()
    tree = ast.parse(src)
    got = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            n = node.targets[0].id
            if n in ("ROOTS", "STAGE"):
                got[n] = ast.literal_eval(node.value)
    return got["ROOTS"], got["STAGE"]


def discover_replica(roots, stage):
    """Line-for-line replica of train_pooled.discover() (glob run from EEG_ROOT, as the trainers chdir)."""
    items = []
    cwd = os.getcwd()
    os.chdir(EEG_ROOT)
    try:
        for subj, root in roots.items():
            if not os.path.isdir(root):
                continue
            for d in glob.glob(f"{root}/*/seizure_*") + glob.glob(f"{root}/*/clip_*_vs_seizure_*"):
                mp4 = os.path.join(d, "video.mp4")
                if not os.path.exists(mp4):
                    continue
                b, sess = os.path.basename(d), os.path.basename(os.path.dirname(d))
                if b.startswith("seizure_"):
                    m = re.search(r"Stage_[0-9]+", b)
                    if not m or m.group() not in stage:
                        continue
                    y = stage[m.group()]
                else:
                    y = 0
                items.append((mp4, y, f"{subj}/{sess}", subj))
    finally:
        os.chdir(cwd)
    return items


# ----------------------------------------------------------------------------- clip names / info.txt
RX_SZ = re.compile(r"^seizure_(\d+)_(.+)_(\d{8}_\d{6})$")
RX_NS = re.compile(r"^clip_(\d+)_vs_seizure_(\d+)_(.+)_(\d{8}_\d{6})$")


def parse_clip_name(name):
    m = RX_SZ.match(name)
    if m:
        return dict(kind="seizure", clip_idx=int(m.group(1)), sz_idx=int(m.group(1)), label=m.group(2),
                    name_time=dt.datetime.strptime(m.group(3), "%Y%m%d_%H%M%S"))
    m = RX_NS.match(name)
    if m:
        return dict(kind="nonseizure", clip_idx=int(m.group(1)), sz_idx=int(m.group(2)), label=m.group(3),
                    name_time=dt.datetime.strptime(m.group(4), "%Y%m%d_%H%M%S"))
    return dict(kind="other", clip_idx=None, sz_idx=None, label=None, name_time=None)


def parse_info(path):
    try:
        txt = open(path, errors="replace").read()
    except OSError:
        return None
    kv = {}
    for line in txt.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            kv[k.strip()] = v.strip()
    out = dict(
        info_sz_idx=kv.get("Seizure index") or kv.get("Matched seizure index"),
        info_ns_idx=kv.get("Non-seizure clip index"),
        info_label=kv.get("Label") or kv.get("Matched seizure label"),
        info_sz_start=parse_dt(kv.get("Seizure start") or kv.get("Matched seizure start")),
        info_clip_start=parse_dt(kv.get("Clip start")),
        info_clip_end=parse_dt(kv.get("Clip end")),
        info_video=kv.get("Video file"),
        info_edf=kv.get("EDF file"),
        info_n_ch=kv.get("EEG channels"),
        info_seek_s=None,
    )
    m = re.search(r"\(([\d.]+)s\)", kv.get("Video seek", ""))
    if m:
        out["info_seek_s"] = float(m.group(1))
    return out


# ----------------------------------------------------------------------------- raw index (from the audit)
class RawIndex:
    """mp4 and EDF rows of raw_audit/files.csv, by name and by animal."""

    def __init__(self):
        rows = read_csv(os.path.join(AUDIT, "files.csv"))
        self.videos_by_name = defaultdict(list)
        self.edfs_by_name = defaultdict(list)
        self.videos_by_animal = defaultdict(dict)       # animal -> name -> row (dedup across partner copies)
        for r in rows:
            if r["kind"] == "video":
                utc = r.get("xml_utc_start")
                r["_trunc_local"] = central_of_epoch(int(utc)) if utc else None
                so = fnum(r.get("sync_offset_s")) or 0.0
                r["_true_local"] = r["_trunc_local"] + dt.timedelta(seconds=so) if r["_trunc_local"] else None
                r["_dur"] = fnum(r.get("duration_s"))
                self.videos_by_name[r["name"]].append(r)
                animals = set(filter(None, (r.get("xml_animals") or "").split("|"))) | {r["animal"]}
                for a in animals:
                    self.videos_by_animal[a].setdefault(r["name"], r)
            elif r["kind"] == "edf":
                r["_start"] = parse_dt(r.get("edf_start"))
                self.edfs_by_name[r["name"]].append(r)

    @staticmethod
    def _prefer(rows, animal):
        own = [r for r in rows if r["animal_folder"] == animal]
        return (own or rows)[0], ("own" if own else "partner")

    def video(self, name, animal):
        rows = self.videos_by_name.get(name) or []
        if not rows:
            return None, "none"
        r, how = self._prefer(rows, animal)
        return r, "name_" + how

    def video_covering(self, animal, t):
        hits = []
        for r in self.videos_by_animal.get(animal, {}).values():
            if r["_trunc_local"] is None or r["_dur"] is None:
                continue
            s = secs(t, r["_trunc_local"])
            if 0 <= s < r["_dur"]:
                hits.append(r)
        return hits

    def edf(self, name, animal):
        rows = self.edfs_by_name.get(name) or []
        if not rows:
            return None, "none"
        r, how = self._prefer(rows, animal)
        return r, "name_" + how


# ----------------------------------------------------------------------------- 1. inventory
INV_COLS = ["animal", "session", "clip_dir", "kind", "clip_idx", "sz_idx", "label", "y", "name_time",
            "in_data", "data_video", "in_full", "full_video_bytes", "full_info", "full_eeg_bytes", "full_other_files",
            "in_discover", "in_cache", "exclusion",
            "info_label", "info_sz_idx", "info_ns_idx", "info_sz_start", "info_clip_start", "info_clip_end",
            "clip_dur_s", "info_video", "info_seek_s", "info_edf", "info_n_ch", "name_vs_info",
            "raw_video", "raw_lookup", "raw_status", "raw_problems", "raw_camera", "raw_trunc_start_local",
            "raw_true_start_local", "raw_sync_s", "raw_dur_s", "seek_pred_s", "info_seek_minus_pred_s",
            "clip_in_file", "cover_by_time", "cover_agrees", "dst_ambiguous",
            "raw_edf", "edf_lookup", "edf_start_local", "edf_dur_s", "edf_1k_label", "edf_offset_s",
            "clip_in_edf", "edf_status", "edf_problems"]


def cmd_inventory(args):
    t0 = time.time()
    roots, stage = parent_config()
    animals = sorted(roots)
    print(f"{len(animals)} labelled animals from train_pooled.ROOTS; STAGE={stage}")
    disc = discover_replica(roots, stage)
    disc_paths = {p for p, *_ in disc}
    print(f"discover() replica: {len(disc)} items")
    try:
        cache_paths = set(json.load(open(CACHE_INDEX))["paths"])
    except Exception as e:  # noqa
        cache_paths = set()
        print("cache index unreadable:", e)
    raw = RawIndex()
    rows = []
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=args.threads)   # NFS metadata is latency-bound
    for a in animals:
        rel_root = roots[a]                               # data/Data_<A>_cropped
        droot, froot = os.path.join(EEG_ROOT, rel_root), os.path.join(FULL, os.path.basename(rel_root))
        keys = set()
        for base, tag in ((droot, "data"), (froot, "full")):
            if not os.path.isdir(base):
                continue
            sess_dirs = [s for s in os.scandir(base) if s.is_dir()]
            for sname, subs in zip([s.name for s in sess_dirs],
                                   pool.map(lambda p: [c.name for c in os.scandir(p) if c.is_dir()],
                                            [s.path for s in sess_dirs])):
                keys.update((sname, c) for c in subs)

        def one(key):
            sess, cdir = key
            r = dict(animal=a, session=sess, clip_dir=cdir)
            r.update(parse_clip_name(cdir))
            lab = r["label"]
            if r["kind"] == "nonseizure":
                r["y"] = 0
            else:
                m = re.search(r"Stage_[0-9]+", lab or "")
                r["y"] = stage.get(m.group()) if m else None
            dpath, fpath = os.path.join(droot, sess, cdir), os.path.join(froot, sess, cdir)
            try:
                dn = set(os.listdir(dpath))
                r["in_data"] = True
            except OSError:
                dn, r["in_data"] = set(), False
            r["data_video"] = "video.mp4" in dn
            try:
                fn = {e.name: e for e in os.scandir(fpath)}
                r["in_full"] = True
            except OSError:
                fn, r["in_full"] = {}, False
            r["full_video_bytes"] = fn["video.mp4"].stat().st_size if "video.mp4" in fn else None
            r["full_info"] = "info.txt" in fn
            r["full_eeg_bytes"] = fn["eeg.edf"].stat().st_size if "eeg.edf" in fn else None
            fi = os.path.join(fpath, "info.txt")
            r["full_other_files"] = "|".join(sorted(set(fn) - {"video.mp4", "info.txt", "eeg.edf"})) or None
            rel_mp4 = f"{rel_root}/{sess}/{cdir}/video.mp4"
            r["in_discover"] = rel_mp4 in disc_paths
            r["in_cache"] = rel_mp4 in cache_paths
            # why discover() would skip it
            ex = []
            if r["kind"] == "other":
                ex.append("name_not_seizure_or_clip_vs")
            if not r["data_video"]:
                ex.append("no_video_in_data")
            if r["kind"] == "seizure" and r["y"] is None:
                ex.append(f"stage_not_in_STAGE:{lab}")
            r["exclusion"] = "|".join(ex)
            info = parse_info(fi) if r["full_info"] else None
            if info:
                r.update(info)
                cs, ce = info["info_clip_start"], info["info_clip_end"]
                r["clip_dur_s"] = secs(ce, cs)
                # name <-> info consistency
                bad = []
                if r["name_time"] and info["info_sz_start"] and info["info_sz_start"].replace(microsecond=0) != r["name_time"]:
                    bad.append("time")
                if lab and info["info_label"] and info["info_label"].replace(" ", "_") != lab:
                    bad.append("label")
                if r["sz_idx"] is not None and info["info_sz_idx"] not in (None, "") and int(info["info_sz_idx"]) != r["sz_idx"]:
                    bad.append("sz_idx")
                if r["kind"] == "nonseizure" and info["info_ns_idx"] not in (None, "") and int(info["info_ns_idx"]) != r["clip_idx"]:
                    bad.append("ns_idx")
                if r["kind"] == "seizure" and cs and info["info_sz_start"] and abs(secs(info["info_sz_start"], cs) - 10.0) > 1e-3:
                    bad.append("prebuffer_ne_10s")
                r["name_vs_info"] = "|".join(bad) or "ok"
                map_raw(r, raw, a)
            return r

        rows.extend(pool.map(one, sorted(keys)))
        print(f"  {a}: {sum(1 for x in rows if x['animal'] == a)} clip dirs  ({time.time() - t0:.0f}s)")
    for r in rows:
        for k in ("name_time", "info_sz_start", "info_clip_start", "info_clip_end", "raw_trunc_start_local",
                  "raw_true_start_local", "edf_start_local"):
            if isinstance(r.get(k), dt.datetime):
                r[k] = iso(r[k])
    write_csv("clips.csv", INV_COLS, rows)
    summ = inventory_summary(rows, disc, cache_paths)
    write_text("inventory.json", json.dumps(summ, indent=1, default=str))
    print(json.dumps(summ["totals"], indent=1, default=str))


def map_raw(r, raw, a):
    cs = r.get("info_clip_start")
    if not cs:
        return
    r["dst_ambiguous"] = ambiguous_dst(cs)
    v, how = (raw.video(r["info_video"], a) if r.get("info_video") not in (None, "", "N/A") else (None, "none"))
    cov = raw.video_covering(a, cs)
    r["cover_by_time"] = "|".join(sorted(x["name"] for x in cov)) or None
    if v is None and cov:
        v, how = cov[0], "time"
    r["raw_lookup"] = how
    if v is not None:
        r["raw_video"] = v["relpath"]
        r["raw_status"], r["raw_problems"], r["raw_camera"] = v["status"], v.get("problems"), v.get("camera")
        r["raw_trunc_start_local"], r["raw_true_start_local"] = v["_trunc_local"], v["_true_local"]
        r["raw_sync_s"], r["raw_dur_s"] = fnum(v.get("sync_offset_s")), v["_dur"]
        if v["_trunc_local"] is not None:
            sp = secs(cs, v["_trunc_local"])
            r["seek_pred_s"] = sp
            if r.get("info_seek_s") is not None:
                r["info_seek_minus_pred_s"] = r["info_seek_s"] - sp
            dur = v["_dur"] or 0
            r["clip_in_file"] = ("yes" if 0 <= sp and sp + (r.get("clip_dur_s") or 0) <= dur + 0.5
                                 else "start_only" if 0 <= sp < dur else "no")
        r["cover_agrees"] = v["name"] in (r["cover_by_time"] or "").split("|")
    e, how = (raw.edf(r["info_edf"], a) if r.get("info_edf") not in (None, "", "N/A") else (None, "none"))
    r["edf_lookup"] = how
    if e is not None:
        r["raw_edf"] = e["relpath"]
        r["edf_start_local"] = e["_start"]
        hrs = fnum(e.get("edf_hours"))
        r["edf_dur_s"] = hrs * 3600 if hrs is not None else None
        r["edf_1k_label"] = e.get("edf_1khz_label") or e.get("edf_ecg_label")
        r["edf_status"], r["edf_problems"] = e["status"], e.get("problems")
        if e["_start"] is not None:
            o = secs(cs, e["_start"])
            r["edf_offset_s"] = o
            if r["edf_dur_s"] is not None:
                r["clip_in_edf"] = ("yes" if 0 <= o and o + (r.get("clip_dur_s") or 0) <= r["edf_dur_s"] + 1
                                    else "start_only" if 0 <= o < r["edf_dur_s"] else "no")


def inventory_summary(rows, disc, cache_paths):
    C = Counter
    tot = dict(
        clip_dirs=len(rows),
        by_kind=dict(C(r["kind"] for r in rows)),
        in_data=sum(r["in_data"] for r in rows), in_full=sum(r["in_full"] for r in rows),
        data_video_present=sum(r["data_video"] for r in rows),
        full_video_nonzero=sum(1 for r in rows if (r["full_video_bytes"] or 0) > 0),
        full_video_zero=sum(1 for r in rows if r["full_video_bytes"] == 0),
        full_video_missing=sum(1 for r in rows if r["in_full"] and r["full_video_bytes"] is None),
        full_info=sum(r["full_info"] for r in rows),
        full_eeg=sum(1 for r in rows if (r["full_eeg_bytes"] or 0) > 0),
        discover_replica=len(disc), discover_flag=sum(r["in_discover"] for r in rows),
        cache_index=len(cache_paths), in_cache=sum(r["in_cache"] for r in rows),
        discover_eq_cache=({p for p, *_ in disc} == cache_paths),
        discover_class_counts=dict(sorted(C(y for _, y, *_ in disc).items())),
        exclusion=dict(C(r["exclusion"] or "none" for r in rows)),
        labels=dict(C((r["kind"], r["label"]) for r in rows).most_common()),
    )
    tot["labels"] = {f"{k[0]}:{k[1]}": v for k, v in tot["labels"].items()}
    tot["not_in_discover_by_reason"] = dict(C(r["exclusion"] or "?" for r in rows if not r["in_discover"]))
    tot["in_discover_not_in_full"] = sum(1 for r in rows if r["in_discover"] and not r["in_full"])
    tot["in_full_not_in_discover"] = sum(1 for r in rows if r["in_full"] and not r["in_discover"])
    d = [r for r in rows if r["in_discover"]]
    tot["discover_with_info"] = sum(1 for r in d if r["full_info"])
    tot["name_vs_info"] = dict(C(r.get("name_vs_info") or "no_info" for r in d))
    tot["raw_lookup"] = dict(C(r.get("raw_lookup") or "no_info" for r in d))
    tot["clip_in_file"] = dict(C(r.get("clip_in_file") or "n/a" for r in d))
    tot["cover_agrees"] = dict(C(str(r.get("cover_agrees")) for r in d))
    tot["raw_status"] = dict(C(r.get("raw_status") or "n/a" for r in d))
    tot["raw_timing_defect"] = sum(1 for r in d if "frames_missing_vs_wallclock" in (r.get("raw_problems") or ""))
    tot["info_seek_minus_pred_abs_gt_0.02"] = sum(1 for r in d if r.get("info_seek_minus_pred_s") is not None
                                                   and abs(r["info_seek_minus_pred_s"]) > 0.02)
    tot["info_seek_checked"] = sum(1 for r in d if r.get("info_seek_minus_pred_s") is not None)
    tot["edf_lookup"] = dict(C(r.get("edf_lookup") or "no_info" for r in d))
    tot["clip_in_edf"] = dict(C(r.get("clip_in_edf") or "n/a" for r in d))
    tot["edf_1k_label"] = dict(C(r.get("edf_1k_label") or "n/a" for r in d))
    tot["dst_ambiguous"] = sum(1 for r in d if r.get("dst_ambiguous"))
    per = {}
    for a in sorted({r["animal"] for r in rows}):
        ra = [r for r in rows if r["animal"] == a]
        da = [r for r in ra if r["in_discover"]]
        per[a] = dict(dirs=len(ra), discover=len(da), seizure=sum(r["kind"] == "seizure" for r in da),
                      nonseizure=sum(r["kind"] == "nonseizure" for r in da),
                      sessions=len({r["session"] for r in da}),
                      info=sum(1 for r in da if r["full_info"]), eeg=sum(1 for r in da if (r["full_eeg_bytes"] or 0) > 0),
                      raw_found=sum(1 for r in da if r.get("raw_video")),
                      clip_in_file=sum(1 for r in da if r.get("clip_in_file") == "yes"),
                      edf_found=sum(1 for r in da if r.get("raw_edf")),
                      clip_in_edf=sum(1 for r in da if r.get("clip_in_edf") == "yes"),
                      timing_defect=sum(1 for r in da if "frames_missing_vs_wallclock" in (r.get("raw_problems") or "")),
                      cameras=dict(Counter(r.get("raw_camera") for r in da if r.get("raw_camera"))))
    return dict(generated=dt.datetime.now().isoformat(sep=" ", timespec="seconds"), totals=tot, per_animal=per)


# ----------------------------------------------------------------------------- 2. sample
SAMPLE_COLS = ["stratum", "animal", "session", "clip_dir", "kind", "label", "y", "clip_idx", "info_clip_start",
               "clip_dur_s", "raw_video", "raw_camera", "seek_pred_s", "raw_problems"]


def session_date(s):
    m = re.search(r"(\d{1,2})-(\d{1,2})-(\d{4})", s)
    return dt.date(int(m.group(3)), int(m.group(1)), int(m.group(2))) if m else dt.date(2100, 1, 1)


def cmd_sample(args):
    import random
    rows = read_csv(out_path("clips.csv"))
    rng = random.Random(args.seed)

    def eligible(r):
        return (r["in_discover"] == "True" and r["full_info"] == "True" and (fnum(r["full_video_bytes"]) or 0) > 0
                and r.get("raw_video") and r.get("clip_in_file") in ("yes", "start_only")
                and r.get("raw_status") not in ("error",)
                and "frames_missing_vs_wallclock" not in (r.get("raw_problems") or ""))

    out, notes = [], {}
    for a in sorted({r["animal"] for r in rows}):
        ra = [r for r in rows if r["animal"] == a and r["in_discover"] == "True"]
        el = [r for r in ra if eligible(r)]
        notes[a] = dict(discover=len(ra), eligible=len(el))
        for kind, n in (("seizure", args.n_seizure), ("nonseizure", args.n_nonseizure)):
            pool = [r for r in el if r["kind"] == kind]
            sessions = sorted({r["session"] for r in pool}, key=lambda s: (session_date(s), s))
            if not sessions:
                continue
            # evenly spaced sessions across the recording, then one clip per chosen session
            k = min(n, len(sessions))
            pick_s = [sessions[round(i * (len(sessions) - 1) / max(1, k - 1))] for i in range(k)] if k > 1 else [sessions[len(sessions) // 2]]
            pick_s = list(dict.fromkeys(pick_s))
            chosen, used_labels = [], set()
            want_severe = kind == "seizure" and any(int(r["y"] or 0) >= 3 for r in pool)
            for j, s in enumerate(pick_s):
                cand = [r for r in pool if r["session"] == s]
                if want_severe and not any(int(c["y"] or 0) >= 3 for c in chosen):
                    sev = [c for c in cand if int(c["y"] or 0) >= 3]
                    if sev and (j == len(pick_s) - 1 or rng.random() < 0.5):
                        cand = sev
                fresh = [c for c in cand if c["label"] not in used_labels]
                c = rng.choice(fresh or cand)
                chosen.append(c)
                used_labels.add(c["label"])
            if want_severe and not any(int(c["y"] or 0) >= 3 for c in chosen):
                sev = [r for r in pool if int(r["y"] or 0) >= 3 and r not in chosen]
                if sev:
                    chosen[-1] = rng.choice(sev)
            # fewer sessions than n: top up from the remaining clips
            while len(chosen) < n and len(chosen) < len(pool):
                rest = [r for r in pool if r not in chosen]
                chosen.append(rng.choice(rest))
            for c in chosen:
                out.append(dict(c, stratum="main"))
    # the labelled clips cut from timing-defect raw files (audit: 62); a few, as a separate stratum
    defect = [r for r in rows if r["in_discover"] == "True" and r["full_info"] == "True"
              and "frames_missing_vs_wallclock" in (r.get("raw_problems") or "")]
    by_file = defaultdict(list)
    for r in defect:
        by_file[r["raw_video"].split("/")[-1].split(".")[-2]].append(r)
    for f, rs in sorted(by_file.items()):
        rs = sorted(rs, key=lambda r: fnum(r["seek_pred_s"]) or 0)
        k = args.n_defect_per_file
        pick = rs if len(rs) <= k else [rs[round(i * (len(rs) - 1) / (k - 1))] for i in range(k)]
        for c in dict.fromkeys(id(x) for x in pick):
            out.append(dict(next(x for x in pick if id(x) == c), stratum="defect"))
    # DST: clips after the 2023-11-05 fall-back whose EDF or video started before it
    vutc = {f["relpath"]: int(f["xml_utc_start"]) for f in read_csv(os.path.join(AUDIT, "files.csv"))
            if f["kind"] == "video" and f["xml_utc_start"]}
    dst = defaultdict(list)
    for r in rows:
        if not (eligible(r) and r["kind"] == "seizure" and r.get("edf_start_local")):
            continue
        vs = dt.datetime.fromtimestamp(vutc[r["raw_video"]], tz=UTC).astimezone(CENTRAL)
        v_off = vs.utcoffset().total_seconds()
        e_off = utc_offset_s(parse_dt(r["edf_start_local"]))
        l_off = utc_offset_s(parse_dt(r["info_clip_start"]))
        if v_off != l_off:
            dst[("video_started_before_label_after", r["animal"])].append(r)
        elif e_off != l_off:
            dst[("edf_started_before_video_label_after", r["animal"])].append(r)
    for (kind_, a), v in sorted(dst.items()):
        v = sorted(v, key=lambda r: -int(r["y"] or 0))
        top = [r for r in v if int(r["y"] or 0) >= 2] or v
        k = 2
        for c in rng.sample(top, min(k, len(top))):
            out.append(dict(c, stratum="dst"))
    notes_dst = {f"{k[0]}:{k[1]}": len(v) for k, v in sorted(dst.items())}

    # day folders whose EDF header start is not ~18.5 s before the first video (audit: room 1402C on
    # 10-19-2023 +198.5 s, 11-17-2023 -748.4 s): does the EDF clock or the video clock move there?
    cl = defaultdict(list)
    for r in rows:
        if eligible(r) and r["kind"] == "seizure" and r["raw_video"].split("/")[1] in CLOCK_DAYS:
            cl[(r["raw_video"].split("/")[1], r["animal"])].append(r)
    picks = [rng.choice(v) for _, v in sorted(cl.items())]
    rng.shuffle(picks)
    for c in sorted(picks[: args.n_clock], key=lambda r: (r["raw_video"], r["animal"])):
        out.append(dict(c, stratum="clock"))
    write_csv("sample.csv", SAMPLE_COLS, out)
    print(f"sampled {len(out)} clips: {dict(Counter((r['stratum'], r['kind']) for r in out))}")
    print("defect-file groups:", {k: len(v) for k, v in by_file.items()})
    write_text("sample_notes.json", json.dumps(dict(seed=args.seed, per_animal=notes, dst_groups=notes_dst,
                                                     defect_groups={k: len(v) for k, v in by_file.items()}), indent=1))


# ----------------------------------------------------------------------------- video helpers
SHOWINFO = re.compile(r"n:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:(-?[\d.]+)")


def ffprobe_stream(path):
    r = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,nb_frames,duration,r_frame_rate,avg_frame_rate,time_base",
                        "-of", "json", path], capture_output=True, text=True, timeout=120)
    s = json.loads(r.stdout or "{}").get("streams", [{}])[0]
    return dict(w=int(s.get("width", 0)), h=int(s.get("height", 0)), nb_frames=int(s.get("nb_frames", 0) or 0),
                duration=float(s.get("duration", 0) or 0), r_fps=s.get("r_frame_rate"), avg_fps=s.get("avg_frame_rate"))


def decode(path, ss, t, vf, w, h, nframes=None, timeout=900):
    """Decode gray frames; returns (frames uint8 [n,h,w], absolute pts seconds [n])."""
    import numpy as np
    cmd = [FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "info"]
    if ss and ss > 0:
        cmd += ["-ss", f"{ss:.4f}"]
    cmd += ["-i", path]
    if t:
        cmd += ["-t", f"{t:.4f}"]
    if nframes:
        cmd += ["-frames:v", str(nframes)]
    cmd += ["-an", "-sn", "-vsync", "0", "-vf", f"{vf},format=gray,showinfo", "-f", "rawvideo",
            "-pix_fmt", "gray", "pipe:1"]
    r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    buf = r.stdout
    n = len(buf) // (w * h)
    fr = np.frombuffer(buf[: n * w * h], dtype=np.uint8).reshape(n, h, w)
    pts = [float(m.group(3)) for m in SHOWINFO.finditer(r.stderr.decode("latin1", "replace"))]
    base = ss if ss and ss > 0 else 0.0
    pts = np.array(pts[:n], dtype=np.float64) + base
    if len(pts) < n:
        fr = fr[: len(pts)]
    return fr, pts


def keyframes_near(path, t0, t1):
    r = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time,flags",
                        "-read_intervals", f"{max(0.0, t0):.3f}%{t1:.3f}", "-of", "csv=p=0", path],
                       capture_output=True, text=True, timeout=120)
    pk = []
    for line in r.stdout.splitlines():
        p = line.split(",")
        try:
            pk.append((float(p[0]), "K" in p[1]))
        except (ValueError, IndexError):
            pass
    pk.sort()
    return pk


def ncc_map(img, tpl):
    import numpy as np
    from scipy.signal import fftconvolve
    img = img.astype(np.float64)
    tpl = tpl.astype(np.float64)
    th, tw = tpl.shape
    t0 = tpl - tpl.mean()
    tn = np.sqrt((t0 ** 2).sum())
    num = fftconvolve(img, t0[::-1, ::-1], mode="valid")
    ii = np.pad(img, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    ii2 = np.pad(img ** 2, ((1, 0), (1, 0))).cumsum(0).cumsum(1)

    def box(I):
        return I[th:, tw:] - I[:-th, tw:] - I[th:, :-tw] + I[:-th, :-tw]

    s, s2 = box(ii), box(ii2)
    var = np.maximum(s2 - s * s / (th * tw), 1e-6)
    return num / (np.sqrt(var) * tn + 1e-9)


def zn(X):
    import numpy as np
    X = X.reshape(len(X), -1).astype(np.float32)
    X = X - X.mean(1, keepdims=True)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-6)
    return X


def diag_scores(Cn, Rn):
    """S[d] = mean_j <Cn[j], Rn[j+d]> for every d with the whole clip inside the raw window."""
    import numpy as np
    M, N = len(Cn), len(Rn)
    D = N - M + 1
    if D < 1:
        return None
    G = Cn @ Rn.T
    ii = np.arange(M)[:, None]
    return G[ii, ii + np.arange(D)[None, :]].mean(0)


def peak_stats(S, excl=3):
    import numpy as np
    k = int(np.argmax(S))
    mask = np.ones(len(S), bool)
    mask[max(0, k - excl): k + excl + 1] = False
    second = float(S[mask].max()) if mask.any() else float("nan")
    med = float(np.median(S))
    mad = float(np.median(np.abs(S - med))) * 1.4826 + 1e-9
    return dict(arg=k, peak=float(S[k]), second=second, margin=float(S[k] - second), z=float((S[k] - med) / mad))


# ----------------------------------------------------------------------------- EDF helpers
def edf_header(path):
    with ropen(path) as fh:
        h = fh.read(256)
        f = lambda a, b: h[a:b].decode("latin1").strip()  # noqa: E731
        ns = int(f(252, 256))
        sh = fh.read(256 * ns)
    fields, o = {}, 0
    for name, w in (("label", 16), ("transducer", 80), ("dim", 8), ("pmin", 8), ("pmax", 8),
                    ("dmin", 8), ("dmax", 8), ("prefilter", 80), ("nsamp", 8), ("sreserved", 32)):
        fields[name] = [sh[o + i * w: o + (i + 1) * w].decode("latin1").strip() for i in range(ns)]
        o += w * ns
    hdr = int(f(184, 192))
    rdur = float(f(244, 252))
    nsamp = [int(x) for x in fields["nsamp"]]
    size = os.path.getsize(path)
    recbytes = 2 * sum(nsamp)
    nrec = int(f(236, 244))
    nrec_file = (size - hdr) // recbytes
    try:
        d, m, y = (int(x) for x in f(168, 176).split("."))
        H, M, S = (int(x) for x in f(176, 184).split("."))
        start = dt.datetime(2000 + y if y < 85 else 1900 + y, m, d, H, M, S)
    except ValueError:
        start = None
    return dict(path=path, start=start, hdr=hdr, rdur=rdur, ns=ns, labels=fields["label"], nsamp=nsamp,
                pmin=[float(x) for x in fields["pmin"]], pmax=[float(x) for x in fields["pmax"]],
                dmin=[float(x) for x in fields["dmin"]], dmax=[float(x) for x in fields["dmax"]],
                recbytes=recbytes, nrec=nrec if nrec > 0 else nrec_file, nrec_file=nrec_file,
                off=[sum(nsamp[:i]) for i in range(ns)])


def edf_pick(h, want="eeg"):
    labels = h["labels"]
    if want == "activity":
        return next((i for i, l in enumerate(labels) if l.lower().startswith("activity")), None)
    for pred in (lambda l: l == "ECG", lambda l: "ECG" in l and "FIR" not in l, lambda l: l == "EEG",
                 lambda l: "EEG" in l):
        i = next((i for i, l in enumerate(labels) if pred(l)), None)
        if i is not None:
            return i
    # generic labels (Channel-657...): the 1 kHz channel that is not the second (HP) copy
    fs = [n / h["rdur"] for n in h["nsamp"]]
    return next((i for i, x in enumerate(fs) if x >= 500), None)


def edf_read(h, ch, s0, s1, digital=False):
    """Samples [s0, s1) of channel ch (sample indices at that channel's rate), physical unless digital."""
    import numpy as np
    n = h["nsamp"][ch]
    s0, s1 = max(0, int(s0)), min(int(s1), h["nrec_file"] * n)
    if s1 <= s0:
        return np.zeros(0), s0
    r0, r1 = s0 // n, (s1 - 1) // n
    out = []
    with ropen(h["path"]) as fh:
        for rec in range(r0, r1 + 1):
            fh.seek(h["hdr"] + rec * h["recbytes"] + 2 * h["off"][ch])
            out.append(np.frombuffer(fh.read(2 * n), dtype="<i2"))
    x = np.concatenate(out)[s0 - r0 * n: s1 - r0 * n].astype(np.float64)
    if not digital:
        g = (h["pmax"][ch] - h["pmin"][ch]) / (h["dmax"][ch] - h["dmin"][ch])
        x = (x - h["dmin"][ch]) * g + h["pmin"][ch]
    return x, s0


def ncc_1d(x, y):
    """NCC of template x at every valid position in y."""
    import numpy as np
    from scipy.signal import fftconvolve
    L = len(x)
    x0 = x - x.mean()
    xn = np.sqrt((x0 ** 2).sum())
    num = fftconvolve(y, x0[::-1], mode="valid")
    c1 = np.concatenate([[0.0], np.cumsum(y)])
    c2 = np.concatenate([[0.0], np.cumsum(y * y)])
    s, s2 = c1[L:] - c1[:-L], c2[L:] - c2[:-L]
    var = np.maximum(s2 - s * s / L, 1e-30)
    return num / (np.sqrt(var) * xn + 1e-30)


# ----------------------------------------------------------------------------- 3. match one clip
def match_clip(s, clips_row, vrow, cams=None):
    import numpy as np
    from scipy.ndimage import zoom
    res = dict(key=clip_key(s), version=ALIGN_VERSION, stratum=s["stratum"], animal=s["animal"],
               session=s["session"], clip_dir=s["clip_dir"], kind=s["kind"], label=s["label"],
               raw_video=s["raw_video"], raw_camera=s.get("raw_camera"), stage_errors=[])
    t_start = time.time()
    clip = os.path.join(FULL, f"Data_{s['animal']}_cropped", s["session"], s["clip_dir"], "video.mp4")
    rawp = os.path.join(RAW, s["raw_video"])
    ci = ffprobe_stream(clip)
    ri = ffprobe_stream(rawp)
    res.update(clip_w=ci["w"], clip_h=ci["h"], clip_nb_frames=ci["nb_frames"], clip_duration=ci["duration"],
               clip_fps=ci["r_fps"], raw_w=ri["w"], raw_h=ri["h"])
    seek = float(clips_row["seek_pred_s"])
    res["seek_pred_s"] = seek
    cw, ch, W, H = ci["w"], ci["h"], ri["w"], ri["h"]
    cdur = ci["duration"]

    # ---- (a) crop box: NCC of 3 clip frames against the raw frames at the predicted times, over scales
    tcs = [min(1.0, cdur / 4), cdur / 2, max(0.0, cdur - 1.5)]
    cfr = [decode(clip, tc, None, "null", cw, ch, nframes=1)[0] for tc in tcs]
    rfr = [decode(rawp, seek + tc, None, "null", W, H, nframes=1)[0] for tc in tcs]
    pairs = [(c[0].astype(np.float32), r[0].astype(np.float32)) for c, r in zip(cfr, rfr) if len(c) and len(r)]
    res["crop_pairs"] = len(pairs)
    if not pairs:
        res["stage_errors"].append("crop:no_frames")
        return res
    best = None
    per_scale = {}
    for sc in SCALES:
        th, tw = int(round(ch * sc)), int(round(cw * sc))
        if th > H or tw > W or th < 16 or tw < 16:
            continue
        acc = None
        for c, r in pairs:
            tpl = c if sc == 1.0 else zoom(c, (th / ch, tw / cw), order=1)
            m = ncc_map(r, tpl)
            acc = m if acc is None else acc + m
        acc /= len(pairs)
        k = np.unravel_index(int(np.argmax(acc)), acc.shape)
        v = float(acc[k])
        # second-best peak at least 12 px away
        m2 = acc.copy()
        m2[max(0, k[0] - 12): k[0] + 13, max(0, k[1] - 12): k[1] + 13] = -1
        per_scale[sc] = round(v, 4)
        if best is None or v > best["ncc"]:
            best = dict(scale=sc, x=int(k[1]), y=int(k[0]), w=tw, h=th, ncc=v, second=float(m2.max()))
    res.update(crop_scale=best["scale"], crop_x=best["x"], crop_y=best["y"], crop_w=best["w"], crop_h=best["h"],
               crop_ncc=best["ncc"], crop_ncc_second=best["second"], crop_ncc_by_scale=per_scale)
    # per-pair NCC at the chosen box (does every pair agree?)
    x, y, w, h = best["x"], best["y"], best["w"], best["h"]
    pn = []
    for c, r in pairs:
        tpl = c if best["scale"] == 1.0 else zoom(c, (h / ch, w / cw), order=1)
        a, b = tpl - tpl.mean(), r[y:y + h, x:x + w] - r[y:y + h, x:x + w].mean()
        pn.append(float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9)))
    res["crop_ncc_pairs"] = [round(v, 4) for v in pn]

    # ---- (b) time offset: frame-sequence correlation within +-NARROW_S of the predicted seek
    ow, oh = max(8, cw // DS_NARROW), max(8, ch // DS_NARROW)
    cf, cpts = decode(clip, 0, None, f"scale={ow}:{oh}:flags=area", ow, oh)
    ss = max(0.0, seek - NARROW_S)
    rf, rpts = decode(rawp, ss, (seek - ss) + cdur + NARROW_S, f"crop={w}:{h}:{x}:{y},scale={ow}:{oh}:flags=area", ow, oh)
    res.update(clip_frames_decoded=len(cf), raw_frames_decoded=len(rf))
    if len(cf) < 10 or len(rf) <= len(cf):
        res["stage_errors"].append(f"time:too_few_frames clip={len(cf)} raw={len(rf)}")
        return res
    cfl, rfl = cf.astype(np.float32), rf.astype(np.float32)
    S_A = diag_scores(zn(cfl), zn(rfl))
    S_B = diag_scores(zn(cfl - np.median(cfl, 0)), zn(rfl - np.median(rfl, 0)))
    S_C = diag_scores(zn(np.diff(cfl, axis=0)), zn(np.diff(rfl, axis=0)))
    pa, pb, pc = peak_stats(S_A), peak_stats(S_B), peak_stats(S_C)
    comb = S_B + S_C
    pbc = peak_stats(comb)
    d = pbc["arg"]
    res.update(tA=pa, tB=pb, tC=pc, tBC=pbc)
    pts0 = float(rpts[d])
    res["pts0_s"] = pts0
    res["d_seek_s"] = pts0 - seek
    res["argmax_spread_frames"] = int(max(pa["arg"], pb["arg"], pc["arg"]) - min(pa["arg"], pb["arg"], pc["arg"]))
    # per-frame NCC at the chosen alignment, and drift across three clip segments
    M = len(cf)
    Cb, Rb = zn(cfl - np.median(cfl, 0)), zn(rfl - np.median(rfl, 0))
    Ca, Ra = zn(cfl), zn(rfl)
    fa = (Ca * Ra[d: d + M]).sum(1)
    res["frame_nccA_min"], res["frame_nccA_median"] = float(fa.min()), float(np.median(fa))
    seg_off = []
    for k in range(3):
        j0, j1 = k * M // 3, (k + 1) * M // 3
        if j1 - j0 < 10:
            continue
        best_o, best_v = None, -9
        for o in range(max(0, d - 45), min(len(rf) - j1, d + 45) + 1):
            v = float((Cb[j0:j1] * Rb[o + j0: o + j1]).sum(1).mean() + 0.0)
            if v > best_v:
                best_o, best_v = o, v
        seg_off.append(best_o - d if best_o is not None else None)
    res["segment_offsets_frames"] = seg_off
    res["mae_ds_at_best"] = float(np.abs(cfl - rfl[d: d + M]).mean())
    res["mae_ds_at_best_plus1"] = float(np.abs(cfl - rfl[d + 1: d + 1 + M]).mean()) if d + 1 + M <= len(rf) else None
    res["mae_ds_at_best_minus1"] = float(np.abs(cfl - rfl[d - 1: d - 1 + M]).mean()) if d >= 1 else None

    # content: clip motion in the labelled seizure span vs the pre-buffer. The clip's frame 0 sits d_seek
    # before the cutter's clip start, so the span is [10 - d_seek, dur_info - 10 - d_seek] in clip time.
    # Non-seizure clips get the same relative span as a null reference.
    try:
        cdi = fnum(clips_row.get("clip_dur_s")) or cdur
        motc = np.abs(np.diff(cfl, axis=0)).mean((1, 2))
        tj = (cpts[1:] - cpts[0]) if len(cpts) == len(cf) else np.arange(1, M) / 15.0
        t0, t1 = 10.0 - res["d_seek_s"], cdi - 10.0 - res["d_seek_s"]
        pre = (tj >= 0.5) & (tj <= t0 - 1.0)
        szm = (tj >= t0) & (tj <= t1)
        if pre.sum() > 10 and szm.sum() > 10:
            res["vid_motion_pre"] = float(motc[pre].mean())
            res["vid_motion_span"] = float(motc[szm].mean())
            res["vid_motion_ratio"] = float(motc[szm].mean() / (motc[pre].mean() + 1e-6))
    except Exception as e:  # noqa
        res["stage_errors"].append(f"content:{e!r}")

    # full-resolution check of one aligned frame pair (the clip's middle frame)
    j = M // 2
    try:
        cfull, _ = decode(clip, 0, None, f"select=eq(n\\,{j})", cw, ch, nframes=1)
        rfull, rp = decode(rawp, max(0.0, float(rpts[d + j]) - 0.02), None, f"crop={w}:{h}:{x}:{y}", w, h, nframes=1)
        if len(cfull) and len(rfull):
            a = cfull[0].astype(np.float32)
            b = rfull[0].astype(np.float32)
            if best["scale"] != 1.0:
                b = zoom(b, (ch / h, cw / w), order=1)[:ch, :cw]
            res["full_frame_mae"] = float(np.abs(a - b).mean())
            res["full_frame_pts_check"] = float(rp[0]) - float(rpts[d + j]) if len(rp) else None
    except Exception as e:  # noqa
        res["stage_errors"].append(f"fullcheck:{e}")

    # keyframe structure around the seek: copy-mode cuts start at the keyframe at or before the seek
    pk = keyframes_near(rawp, seek - 6, seek + 3)
    kf = [p for p, k in pk if k and p <= seek + 1e-6]
    if pk and kf:
        kpts = kf[-1]
        allp = [p for p, _ in pk]
        idx_k = int(np.argmin(np.abs(np.array(allp) - kpts)))
        idx_0 = int(np.argmin(np.abs(np.array(allp) - pts0)))
        res["keyframe_before_seek_s"] = kpts
        res["seek_minus_keyframe_s"] = seek - kpts
        res["d_key_frames"] = idx_0 - idx_k if abs(allp[idx_0] - pts0) < 0.02 else None
        firstge = next((p for p in allp if p >= seek - 1e-6), None)
        res["first_frame_ge_seek_s"] = firstge
        res["gop_s"] = float(np.median(np.diff([p for p, k in pk if k]))) if sum(k for _, k in pk) > 1 else None
    true_start = parse_dt(clips_row["raw_true_start_local"])
    trunc_start = parse_dt(clips_row["raw_trunc_start_local"])
    vid_utc = naive_utc(int(vrow["xml_utc_start"])) + dt.timedelta(seconds=fnum(vrow.get("sync_offset_s")) or 0.0)
    res["vid_start_utc"] = iso(vid_utc)
    cstart = parse_dt(clips_row["info_clip_start"])
    res["raw_sync_s"] = fnum(clips_row["raw_sync_s"])
    res["clock_off_s"] = secs(true_start + dt.timedelta(seconds=pts0), cstart)       # raw clock - info time, frame 0
    res["clock_off_trunc_s"] = secs(trunc_start + dt.timedelta(seconds=pts0), cstart)
    name_t = parse_dt(clips_row["name_time"])
    if s["kind"] == "seizure" and name_t is not None:
        # raw clock of the frame the clip places at the annotated seizure start, minus the name time
        sz = parse_dt(clips_row["info_sz_start"])
        res["name_off_s"] = secs(true_start + dt.timedelta(seconds=pts0 + secs(sz, cstart)), name_t)

    # ---- DST (2023-11-05 fall-back). The cutter subtracted naive local times. The EDF clock is naive local
    # wall time: the EDF that spans the change is exactly 1 h shorter than its real span, and the
    # Activity/motion lag steps from -1 s to +3599 s across the change (see README). So EEG cuts are right,
    # while a video that started before the change, cut for a label after it, is cut (v_off - l_off) s early.
    v_off = round((true_start - vid_utc).total_seconds())      # the video start's exact UTC offset
    l_off = utc_offset_s(cstart)
    res["video_seek_error_s"] = v_off - l_off                   # true seek - cut seek
    edf_start = parse_dt(clips_row.get("edf_start_local"))
    if edf_start is not None:
        res["edf_start_other_side_s"] = utc_offset_s(edf_start) - l_off
    if s["stratum"] == "dst":
        for tag, alt in (("minus1h", seek - 3600.0), ("plus1h", seek + 3600.0)):
            # the moment 1 h either side may sit in the camera's previous or next file
            path2, alt2 = rawp, alt
            if cams is not None and not (0 <= alt <= (fnum(vrow.get("duration_s")) or 0) - cdur):
                t_utc = vid_utc + dt.timedelta(seconds=alt)
                for u0, rel2, d2 in cams.get(vrow.get("camera"), []):
                    if u0 <= t_utc and t_utc + dt.timedelta(seconds=cdur) <= u0 + dt.timedelta(seconds=d2):
                        path2, alt2 = os.path.join(RAW, rel2), (t_utc - u0).total_seconds()
                        res[f"alt_file_{tag}"] = rel2
                        break
            res[f"alt_seek_{tag}_s"] = alt2
            try:
                af, apts = decode(path2, max(0.0, alt2), cdur, f"crop={w}:{h}:{x}:{y},scale={ow}:{oh}:flags=area", ow, oh)
                alt = alt2
                if len(af) > 20 and alt >= 0:
                    cdi = fnum(clips_row.get("clip_dur_s")) or cdur
                    ma = np.abs(np.diff(af.astype(np.float32), axis=0)).mean((1, 2))
                    tj = apts[1:] - alt
                    pre = (tj >= 0.5) & (tj <= 9.0)
                    szm = (tj >= 10.0) & (tj <= cdi - 10.0)
                    res[f"vid_motion_ratio_{tag}"] = float(ma[szm].mean() / (ma[pre].mean() + 1e-6))
            except Exception as e:  # noqa
                res["stage_errors"].append(f"dst_video_{tag}:{e!r}")

    # ---- (c) wide check (+-WIDE_S at 1/DS_WIDE) and per-second motion for the Activity comparison
    try:
        ww, wh = max(6, cw // DS_WIDE), max(6, ch // DS_WIDE)
        cfw, _ = decode(clip, 0, None, f"scale={ww}:{wh}:flags=area", ww, wh)
        wide_s = WIDE_S * WIDE_MULT.get(s["stratum"], 1)
        ssw = max(0.0, seek - wide_s)
        rfw, rptsw = decode(rawp, ssw, (seek - ssw) + cdur + wide_s, f"crop={w}:{h}:{x}:{y},scale={ww}:{wh}:flags=area", ww, wh)
        res["wide_window_s"] = wide_s
        res["wide_frames"] = len(rfw)
        if len(rfw) > len(cfw) + 10:
            c32, r32 = cfw.astype(np.float32), rfw.astype(np.float32)
            Sw = diag_scores(zn(c32 - np.median(c32, 0)), zn(r32 - np.median(r32, 0)))
            Sw2 = diag_scores(zn(np.diff(c32, axis=0)), zn(np.diff(r32, axis=0)))
            comb_w = Sw + Sw2
            pw = peak_stats(comb_w, excl=5)
            res["wide"] = pw
            res["wide_pts0_s"] = float(rptsw[pw["arg"]])
            res["wide_minus_narrow_s"] = float(rptsw[pw["arg"]]) - pts0
            inwin = np.abs(rptsw[: len(comb_w)] - seek) <= NARROW_S
            res["wide_best_in_narrow"] = float(comb_w[inwin].max()) if inwin.any() else None
            res["wide_best_outside_narrow"] = float(comb_w[~inwin].max()) if (~inwin).any() else None
            mot = np.abs(np.diff(r32, axis=0)).mean((1, 2))
            res["_motion"] = dict(pts=rptsw[1:].round(4).tolist(), mot=mot.round(4).tolist())
    except Exception as e:  # noqa
        res["stage_errors"].append(f"wide:{e}")

    # ---- (d) EEG: clip eeg.edf against the raw EDF at the implied offset; Activity vs motion
    try:
        eeg_check(s, clips_row, res, vid_utc, pts0)
    except Exception as e:  # noqa
        res["stage_errors"].append(f"eeg:{e!r}")
        res["eeg_trace"] = traceback.format_exc()[-800:]
    res.pop("_motion", None)
    res["secs"] = round(time.time() - t_start, 1)
    return res


def eeg_check(s, cr, res, vid_utc, pts0):
    import numpy as np
    cedf = os.path.join(FULL, f"Data_{s['animal']}_cropped", s["session"], s["clip_dir"], "eeg.edf")
    res["clip_eeg_present"] = os.path.exists(cedf) and os.path.getsize(cedf) > 0
    if not cr.get("raw_edf"):
        res["eeg_status"] = "no_raw_edf"
        return
    hr = edf_header(os.path.join(RAW, cr["raw_edf"]))
    ch_r = edf_pick(hr)
    cstart = parse_dt(cr["info_clip_start"])
    o = secs(cstart, hr["start"])
    res.update(raw_edf=cr["raw_edf"], raw_edf_label=hr["labels"][ch_r] if ch_r is not None else None,
               raw_edf_labels="|".join(hr["labels"]),
               edf_offset_s=o, raw_edf_start=iso(hr["start"]))
    # coarse clock check: EDF Activity (1 Hz) vs video motion in the animal's crop (needs the wide decode)
    mo = res.get("_motion")
    ia = edf_pick(hr, "activity")
    if mo and ia is not None and hr["nsamp"][ia] / hr["rdur"] == 1.0:
        # video frame -> EDF elapsed seconds, EDF clock = naive local wall time
        pts_a = np.array(mo["pts"])
        offs = {}
        vt = np.empty(len(pts_a))
        for i_, pp in enumerate(pts_a):
            u = vid_utc + dt.timedelta(seconds=float(pp))
            hk = u.replace(minute=0, second=0, microsecond=0)
            if hk not in offs:
                offs[hk] = (utc_to_local(hk) - hk).total_seconds()
            vt[i_] = (u - hr["start"]).total_seconds() + offs[hk]
        mot = np.array(mo["mot"])
        b0, b1 = int(np.floor(vt.min())), int(np.floor(vt.max()))
        nb = b1 - b0 + 1
        mb = np.full(nb, np.nan)
        idx = np.floor(vt).astype(int) - b0
        sums = np.bincount(idx, weights=mot, minlength=nb)
        cnts = np.bincount(idx, minlength=nb)
        ok = cnts >= 8
        mb[ok] = sums[ok] / cnts[ok]
        act, a0 = edf_read(hr, ia, b0, b1 + 1)
        if len(act) == nb and ok.sum() > 60:
            la = np.log1p(np.maximum(act, 0))
            lm = np.log1p(mb)
            rs = {}
            for L in range(-30, 31):
                # motion bin i+L vs activity bin i
                i0, i1 = max(0, -L), min(nb, nb - L)
                a_, m_ = la[i0:i1], lm[i0 + L:i1 + L]
                g = np.isfinite(m_)
                if g.sum() > 60 and a_[g].std() > 0 and m_[g].std() > 0:
                    rs[L] = float(np.corrcoef(a_[g], m_[g])[0, 1])
            res["act_curve"] = {str(k): round(v, 4) for k, v in rs.items()}
            # long-lag search: where along the EDF does this stretch of video motion fit best?
            # (a frame gap before the clip in a timing-defect file shows up as a lag of the gap size)
            LAG = {"defect": LONG_LAG_DEFECT_S, "clock": LONG_LAG_CLOCK_S}.get(s["stratum"], LONG_LAG_S)
            al, a0 = edf_read(hr, ia, b0 - LAG, b1 + 1 + LAG)
            tpl = np.where(np.isfinite(lm), lm, np.nanmean(lm))
            if len(al) > len(tpl) + 10:
                cc = ncc_1d(tpl, np.log1p(np.maximum(al, 0)))
                kk = int(np.argmax(cc))
                m2 = cc.copy()
                m2[max(0, kk - 5): kk + 6] = -1
                res.update(act_long_range_s=LAG, act_long_lag_s=int(b0 - a0 - kk), act_long_r=float(cc[kk]),
                           act_long_second=float(m2.max()),
                           act_long_r_at0=float(cc[b0 - a0]) if 0 <= b0 - a0 < len(cc) else None)
            if rs:
                Lb = max(rs, key=rs.get)
                res.update(act_lag_s=Lb, act_r_best=rs[Lb], act_r0=rs.get(0),
                           act_r_second=max((v for k, v in rs.items() if abs(k - Lb) > 2), default=None),
                           act_bins=int(ok.sum()), act_nonzero=int((act > 0).sum()))
    if not res["clip_eeg_present"]:
        res["eeg_status"] = "no_clip_eeg"
        return
    hc = edf_header(cedf)
    ch_c = edf_pick(hc)
    res["clip_edf_label"] = hc["labels"][ch_c] if ch_c is not None else None
    res["clip_edf_labels"] = "|".join(hc["labels"])
    # what the parent reader would take: the first label containing 'ECG'
    res["clip_parent_reader_label"] = next((l for l in hc["labels"] if "ECG" in l), None)
    res["clip_edf_start_hdr"] = iso(hc["start"])
    if ch_c is None or ch_r is None:
        res["eeg_status"] = "no_channel"
        return
    fs_c, fs_r = hc["nsamp"][ch_c] / hc["rdur"], hr["nsamp"][ch_r] / hr["rdur"]
    res.update(clip_eeg_fs=fs_c, raw_eeg_fs=fs_r)
    x, _ = edf_read(hc, ch_c, 0, hc["nrec_file"] * hc["nsamp"][ch_c])
    xd, _ = edf_read(hc, ch_c, 0, hc["nrec_file"] * hc["nsamp"][ch_c], digital=True)
    res["clip_eeg_seconds"] = len(x) / fs_c
    res["clip_eeg_unique"] = int(len(np.unique(xd)))
    if fs_c != fs_r:
        res["eeg_status"] = f"fs_mismatch {fs_c} vs {fs_r}"
        return
    p = int(round(o * fs_r))
    lo = p - int(EEG_WIN_S * fs_r)
    y, y0 = edf_read(hr, ch_r, lo, p + len(x) + int(EEG_WIN_S * fs_r))
    if len(y) < len(x) + 10 or x.std() == 0:
        res["eeg_status"] = "raw_window_short_or_flat_clip"
        return
    c = ncc_1d(x, y)
    k = int(np.argmax(c))
    lag = (y0 + k) - p
    m = c.copy()
    m[max(0, k - 100): k + 101] = -1
    seg = y[k: k + len(x)]
    slope = float(np.polyfit(seg, x, 1)[0]) if seg.std() > 0 else None
    res.update(eeg_lag_samples=int(lag), eeg_lag_ms=lag / fs_r * 1000, eeg_ncc=float(c[k]),
               eeg_ncc_second=float(m.max()), eeg_ncc_at_implied=float(c[p - y0]) if 0 <= p - y0 < len(c) else None,
               eeg_slope=slope, eeg_resid_rel=float(np.std(x - seg * (slope or 1)) / (x.std() + 1e-30)))
    # EEG-vs-video offset of the clip: EDF clock of EEG sample 0 minus video clock of frame 0, both in UTC
    # (EDF clock taken as naive local wall time)
    eeg0 = local_to_utc(hr["start"] + dt.timedelta(seconds=(p + lag) / fs_r))
    vid0 = vid_utc + dt.timedelta(seconds=pts0)
    res["eeg_minus_video_s"] = secs(eeg0, vid0)
    # content: EEG line length in the labelled span [10, dur-10] vs the pre-buffer [0, 9] (EEG starts at clip start)
    cdi = fnum(cr.get("clip_dur_s"))
    if cdi:
        ll = np.abs(np.diff(x))
        i0, i1, i9 = int(10 * fs_c), int((cdi - 10) * fs_c), int(9 * fs_c)
        if i1 - i0 > fs_c and i9 > fs_c:
            res["eeg_ll_ratio"] = float(ll[i0:i1].mean() / (ll[:i9].mean() + 1e-30))
            if s["stratum"] == "dst":
                # the same measure one hour either side (a continuous-time EDF would put the label at +1 h)
                for tag, sh in (("minus1h", -3600), ("plus1h", 3600)):
                    pa = p + int(sh * fs_r)
                    ya, _ = edf_read(hr, ch_r, pa, pa + len(x))
                    if len(ya) == len(x):
                        la = np.abs(np.diff(ya))
                        res[f"eeg_ll_ratio_{tag}"] = float(la[i0:i1].mean() / (la[:i9].mean() + 1e-30))
    res["eeg_status"] = eeg_status_of(res["eeg_ncc"])


def eeg_status_of(ncc):
    return "ok" if ncc > 0.99 else ("weak" if ncc > 0.5 else "no_match")


# ----------------------------------------------------------------------------- clock scans along one file
# name: (animal, raw video, raw EDF, crop box, window s, max lag s, offsets into the video, lags to report)
SCANS = {
    "1013_RN242": ("RN242", "RN242/10-13-2023/RN242-RN243.20231014004100.mp4", "RN242/10-13-2023/RN242-10-13-2023.edf",
                   (48, 264, 352, 216), 1200, 11000, (100, 1500, 3000, 4500, 6000, 7000, 8500, 12000, 18000, 23000),
                   (-1, 0, -10520, -10522, -10524, -10526)),
    "1103_RN235": ("RN235", "RN235/11-2-2023/RN222-RN235.20231103122100.mp4", "RN235/11-2-2023/RN235-11-2-2023.edf",
                   (480, 0, 320, 480), 1200, 300, (100, 3000, 6000, 9000, 10500, 12000, 15000, 18000, 21000, 23500),
                   (-1, 0, -60, -61, -62)),
    "1205_RN235": ("RN235", "RN235/12-05-2023/RN222-RN235.20231205222300.mp4", "RN235/12-05-2023/RN235-12-5-2023.edf",
                   (480, 0, 320, 480), 1200, 300, (100, 3000, 6000, 9000, 12000, 13500, 15000, 18000, 21000),
                   (-1, 0, -60, -61, -62)),
    "dst_RN224": ("RN224", None, "RN224/11-03-2023 to 11-05-2023/RN224-11-3-2023.edf", (424, 264, 320, 228), 1800, 7200,
                  ("RN224/11-03-2023 to 11-05-2023/RN210-RN224.20231104200900.mp4@12000",
                   "RN224/11-03-2023 to 11-05-2023/RN210-RN224.20231105034200.mp4@12000",
                   "RN224/11-03-2023 to 11-05-2023/RN210-RN224.20231105111500.mp4@3000",
                   "RN224/11-03-2023 to 11-05-2023/RN210-RN224.20231105111500.mp4@20000",
                   "RN224/11-03-2023 to 11-05-2023/RN210-RN224.20231105180600.mp4@7000",
                   "RN224/11-03-2023 to 11-05-2023/RN210-RN224.20231106083800.mp4@8000"),
                  (-1, 0, 3599, 3600)),
}


def cmd_scan(args):
    """EDF Activity vs video motion at points along one file: where does the video clock step?

    EDF time is mapped to UTC with the continuous model (start + elapsed, the start's own UTC offset), so a
    lag in the output is exactly the step a timing defect or the DST hour introduces."""
    import numpy as np
    vrows = {r["relpath"]: r for r in read_csv(os.path.join(AUDIT, "files.csv")) if r["kind"] == "video"}
    names = args.names or sorted(SCANS)
    for name in names:
        animal, vid, edfrel, (x, y, w, h), win, lagmax, offsets, lags = SCANS[name]
        hr = edf_header(os.path.join(RAW, edfrel))
        edf_utc = local_to_utc(hr["start"])
        ia = edf_pick(hr, "activity")
        ww, wh = max(6, w // DS_WIDE), max(6, h // DS_WIDE)
        out = []
        for o in offsets:
            rel, off = (o.split("@")[0], float(o.split("@")[1])) if isinstance(o, str) else (vid, float(o))
            vr = vrows[rel]
            vutc = naive_utc(int(vr["xml_utc_start"])) + dt.timedelta(seconds=fnum(vr.get("sync_offset_s")) or 0.0)
            fr, pts = decode(os.path.join(RAW, rel), off, win, f"crop={w}:{h}:{x}:{y},scale={ww}:{wh}:flags=area", ww, wh)
            if len(fr) < 100:
                out.append(dict(video=rel, offset=off, error=f"{len(fr)} frames"))
                continue
            mot = np.abs(np.diff(fr.astype(np.float32), axis=0)).mean((1, 2))
            vt = pts[1:] + secs(vutc, edf_utc)
            b0, b1 = int(np.floor(vt.min())), int(np.floor(vt.max()))
            nb = b1 - b0 + 1
            idx = np.floor(vt).astype(int) - b0
            s_, c_ = np.bincount(idx, weights=mot, minlength=nb), np.bincount(idx, minlength=nb)
            lm = np.log1p(np.where(c_ >= 8, s_ / np.maximum(c_, 1), np.nan))
            tpl = np.where(np.isfinite(lm), lm, np.nanmean(lm))
            al, a0 = edf_read(hr, ia, b0 - lagmax, b1 + 1 + lagmax)
            cc = ncc_1d(tpl, np.log1p(np.maximum(al, 0)))
            lg = b0 - a0 - np.arange(len(cc))
            top = []
            for k in np.argsort(-cc):
                if all(abs(lg[k] - t[0]) > 10 for t in top):
                    top.append((int(lg[k]), round(float(cc[k]), 3)))
                if len(top) == 5:
                    break
            r_at = {str(L): round(float(cc[np.argmin(np.abs(lg - L))]), 3) for L in lags}
            out.append(dict(video=rel, offset=off, edf_elapsed_h=round(b0 / 3600, 2), bins=nb, top=top, r_at=r_at,
                            median=round(float(np.median(cc)), 3), sd=round(float(np.std(cc)), 3)))
            print(name, json.dumps(out[-1]), flush=True)
        write_text(os.path.join("scans", f"{name}.json"), json.dumps(dict(name=name, animal=animal, edf=edfrel,
                                                                          box=[x, y, w, h], window_s=win, lagmax_s=lagmax,
                                                                          points=out), indent=1))


def clip_key(s):
    return f"{s['animal']}|{s['session']}|{s['clip_dir']}|{s['stratum']}"


def cmd_list(args):
    sm = read_csv(out_path("sample.csv"))
    an = sorted({r["animal"] for r in sm})
    for i, a in enumerate(an):
        print(i, a, sum(r["animal"] == a for r in sm))


def cmd_match(args):
    sm = read_csv(out_path("sample.csv"))
    an = sorted({r["animal"] for r in sm})
    a = args.animal or an[args.index]
    clips = {(r["animal"], r["session"], r["clip_dir"]): r for r in read_csv(out_path("clips.csv"))}
    vrows = {r["relpath"]: r for r in read_csv(os.path.join(AUDIT, "files.csv")) if r["kind"] == "video"}
    cams, seen = defaultdict(list), set()
    for r in vrows.values():
        if r.get("xml_utc_start") and r.get("duration_s") and r["name"] not in seen:
            seen.add(r["name"])
            u0 = naive_utc(int(r["xml_utc_start"])) + dt.timedelta(seconds=fnum(r.get("sync_offset_s")) or 0.0)
            cams[r.get("camera")].append((u0, r["relpath"], float(r["duration_s"])))
    todo = [r for r in sm if r["animal"] == a]
    unit = out_path("units", f"{a}.jsonl")
    os.makedirs(os.path.dirname(unit), exist_ok=True)
    done = set()
    if os.path.exists(unit):
        for line in open(unit):
            try:
                j = json.loads(line)
                if j.get("version") == ALIGN_VERSION:
                    done.add(j["key"])
            except json.JSONDecodeError:
                pass
    print(f"{a}: {len(todo)} sampled, {len(done)} already done", flush=True)
    for s in todo:
        if clip_key(s) in done:
            continue
        cr = clips[(s["animal"], s["session"], s["clip_dir"])]
        try:
            res = match_clip(s, cr, vrows[s["raw_video"]], cams)
        except Exception as e:  # noqa
            res = dict(key=clip_key(s), version=ALIGN_VERSION, stratum=s["stratum"], animal=a, session=s["session"],
                       clip_dir=s["clip_dir"], kind=s["kind"], label=s["label"],
                       stage_errors=[f"fatal:{e!r}"], trace=traceback.format_exc()[-1500:])
        with open(unit, "a") as fh:
            fh.write(json.dumps(res, default=str) + "\n")
        print(f"  {s['clip_dir']}: crop=({res.get('crop_x')},{res.get('crop_y')},{res.get('crop_w')},{res.get('crop_h')})"
              f" s={res.get('crop_scale')} ncc={res.get('crop_ncc')}  d_seek={res.get('d_seek_s')}"
              f"  dkey={res.get('d_key_frames')}  eeg={res.get('eeg_status')} lag={res.get('eeg_lag_ms')}"
              f"  err={res.get('stage_errors')}  {res.get('secs')}s", flush=True)


# ----------------------------------------------------------------------------- 4. merge
def grade(r):
    """Confidence of the clip -> raw match (crop box AND frame offset together).

    high   the crop NCC >= 0.95, the three frame-sequence scores (raw intensity, background-removed,
           frame difference) peak within 1 frame of each other, the three clip thirds give the same
           offset (+-1 frame: no drop/dup drift), the frame-difference peak stands >= 0.1 above any
           other offset and >= 20 robust SDs above the median, every aligned frame has intensity
           NCC >= 0.95, and one full-resolution aligned frame differs by <= 3 grey levels (MAE)
    medium crop NCC >= 0.9, and either the peaks agree or the thirds agree, with a frame-difference
           z >= 8 or a background-removed margin >= 0.05
    low    crop NCC >= 0.8 only;  fail otherwise
    """
    if not r.get("tBC"):
        return "fail"
    b, c = r["tB"], r["tC"]
    cn = r.get("crop_ncc") or 0
    agree = (r.get("argmax_spread_frames") if r.get("argmax_spread_frames") is not None else 99) <= 1
    segs = [x for x in (r.get("segment_offsets_frames") or []) if x is not None]
    stable = len(segs) >= 2 and max(map(abs, segs)) <= 1
    mae = r.get("full_frame_mae")
    if (cn >= 0.95 and agree and stable and c["margin"] >= 0.1 and c["z"] >= 20
            and (r.get("frame_nccA_min") or 0) >= 0.95 and mae is not None and mae <= 3):
        return "high"
    if cn >= 0.9 and (agree or stable) and (c["z"] >= 8 or b["margin"] >= 0.05):
        return "medium"
    if cn >= 0.8:
        return "low"
    return "fail"


MATCH_COLS = ["stratum", "animal", "session", "clip_dir", "kind", "label", "raw_video", "raw_camera",
              "clip_w", "clip_h", "clip_nb_frames", "clip_duration", "crop_scale", "crop_x", "crop_y", "crop_w",
              "crop_h", "crop_ncc", "crop_ncc_second", "crop_ncc_pairs", "grade", "seek_pred_s", "pts0_s",
              "d_seek_s", "keyframe_before_seek_s", "seek_minus_keyframe_s", "first_frame_ge_seek_s", "d_key_frames",
              "gop_s", "raw_sync_s", "clock_off_s", "clock_off_trunc_s", "name_off_s", "tA_peak", "tB_peak",
              "tB_margin", "tB_z", "tC_peak", "tC_z", "argmax_spread_frames", "segment_offsets_frames",
              "frame_nccA_min", "frame_nccA_median", "mae_ds_at_best", "mae_ds_at_best_plus1", "mae_ds_at_best_minus1",
              "full_frame_mae", "wide_minus_narrow_s", "wide_best_in_narrow", "wide_best_outside_narrow",
              "eeg_status", "clip_eeg_present", "clip_edf_label", "raw_edf", "raw_edf_label", "edf_offset_s",
              "clip_eeg_unique", "clip_eeg_seconds", "eeg_lag_samples", "eeg_lag_ms", "eeg_ncc", "eeg_ncc_second",
              "eeg_ncc_at_implied", "eeg_slope", "eeg_resid_rel", "eeg_minus_video_s", "clip_edf_labels",
              "clip_parent_reader_label", "raw_edf_labels", "room", "act_lag_s", "act_r_best",
              "act_r0", "act_r_second", "act_bins", "act_nonzero", "act_long_range_s", "act_long_lag_s",
              "act_long_r", "act_long_second", "act_long_r_at0", "wide_window_s", "vid_motion_ratio", "eeg_ll_ratio",
              "video_seek_error_s", "edf_start_other_side_s", "vid_motion_ratio_minus1h", "vid_motion_ratio_plus1h",
              "eeg_ll_ratio_minus1h", "eeg_ll_ratio_plus1h", "stage_errors", "secs"]


def q(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def dist(xs, nd=3):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "n=0"
    return (f"n={len(xs)} min={min(xs):.{nd}f} p5={q(xs, .05):.{nd}f} med={q(xs, .5):.{nd}f} "
            f"p95={q(xs, .95):.{nd}f} max={max(xs):.{nd}f}")


def cmd_merge(args):
    sm = read_csv(out_path("sample.csv"))
    res = {}
    for f in sorted(glob.glob(out_path("units", "*.jsonl"))):
        for line in open(f):
            try:
                j = json.loads(line)
            except json.JSONDecodeError:
                continue
            if j.get("version") == ALIGN_VERSION:
                res[j["key"]] = j
    room = {}
    for f in read_csv(os.path.join(AUDIT, "files.csv")):
        if f["kind"] == "video":
            m = re.search(r"(1402[A-Z])", f.get("xml_fullpath") or "")
            room[f["relpath"]] = m.group(1) if m else None
    rows, missing = [], []
    for s in sm:
        r = res.get(clip_key(s))
        if r is None:
            missing.append(clip_key(s))
            continue
        for t in ("tA", "tB", "tC"):
            for k in ("peak", "margin", "z"):
                if r.get(t):
                    r[f"{t}_{k}"] = r[t][k]
        r["grade"] = grade(r)
        r["room"] = room.get(r.get("raw_video"))
        if r.get("eeg_status") is None and r.get("eeg_ncc") is not None:
            r["eeg_status"] = eeg_status_of(r["eeg_ncc"])      # version-3 runs reached the end without setting it
        rows.append(r)
    write_csv("matches.csv", MATCH_COLS, rows)
    main = [r for r in rows if r["stratum"] == "main"]
    L = []
    P = L.append
    P(f"CLIP -> RAW ALIGNMENT  {dt.datetime.now():%Y-%m-%d %H:%M:%S}  align_version={ALIGN_VERSION}")
    P(f"sampled {len(sm)}  matched-records {len(rows)}  missing {len(missing)}")
    for m in missing:
        P(f"  MISSING {m}")
    inv = json.load(open(out_path("inventory.json")))
    P("")
    P("INVENTORY (step 1)")
    for k, v in inv["totals"].items():
        P(f"  {k}: {v}")
    P("")
    P("PER-ANIMAL INVENTORY")
    for a, v in inv["per_animal"].items():
        P(f"  {a}: {v}")
    P("")
    P("TIME-ALIGNMENT GRADES (main stratum)")
    P(f"  {dict(Counter(r['grade'] for r in main))}  by kind: "
      f"{dict(Counter((r['kind'], r['grade']) for r in main))}")
    P("")
    P("OFFSETS (main stratum, graded high/medium)")
    good = [r for r in main if r["grade"] in ("high", "medium")]
    for k, nd in (("d_seek_s", 3), ("seek_minus_keyframe_s", 3), ("d_key_frames", 0), ("clock_off_s", 3),
                  ("clock_off_trunc_s", 3), ("name_off_s", 3), ("raw_sync_s", 3), ("wide_minus_narrow_s", 3),
                  ("full_frame_mae", 2), ("mae_ds_at_best", 2), ("crop_ncc", 4)):
        P(f"  {k:24s} {dist([fnum(r.get(k)) for r in good], nd)}")
    P(f"  d_key_frames counts: {dict(Counter(r.get('d_key_frames') for r in good))}")
    P("")
    # crop boxes per animal/camera
    boxes = defaultdict(list)
    for r in rows:
        if r.get("crop_x") is None:
            continue
        boxes[(r["animal"], r.get("raw_camera"))].append(r)
    crow = []
    for (a, cam), rs in sorted(boxes.items()):
        c = Counter((r["crop_x"], r["crop_y"], r["crop_w"], r["crop_h"], r["crop_scale"]) for r in rs)
        (bx, by, bw, bh, bs), n = c.most_common(1)[0]
        sess = sorted({r["session"] for r in rs}, key=lambda s: (session_date(s), s))
        dev = [max(abs(r["crop_x"] - bx), abs(r["crop_y"] - by)) for r in rs]
        crow.append(dict(animal=a, camera=cam, x=bx, y=by, w=bw, h=bh, scale=bs, n_clips=len(rs), n_same_box=n,
                         max_dev_px=max(dev), n_days=len(sess), first_day=sess[0], last_day=sess[-1],
                         boxes_seen=";".join(f"{k[0]},{k[1]},{k[2]},{k[3]}@{k[4]}x{v}" for k, v in c.items()),
                         side=("left" if bx + bw / 2 < 400 else "right"),
                         crop_ncc_min=min(r["crop_ncc"] for r in rs)))
    write_csv("crop_boxes.csv", ["animal", "camera", "x", "y", "w", "h", "scale", "side", "n_clips", "n_same_box",
                                 "max_dev_px", "n_days", "first_day", "last_day", "crop_ncc_min", "boxes_seen"], crow)
    P("CROP BOXES (raw 800x600 px; x,y = top-left; per animal x camera; all strata)")
    for c in crow:
        P(f"  {c['animal']:6s} {str(c['camera']):12s} box=({c['x']},{c['y']},{c['w']},{c['h']}) scale={c['scale']} "
          f"{c['side']:5s} same={c['n_same_box']}/{c['n_clips']} maxdev={c['max_dev_px']}px days={c['n_days']} "
          f"({c['first_day']}..{c['last_day']}) ncc_min={c['crop_ncc_min']:.3f}")
    P("")
    # per animal
    prow = []
    for a in sorted({r["animal"] for r in rows}):
        ra = [r for r in main if r["animal"] == a]
        g = Counter(r["grade"] for r in ra)
        prow.append(dict(animal=a, n=len(ra), high=g.get("high", 0), medium=g.get("medium", 0), low=g.get("low", 0),
                         fail=g.get("fail", 0), days=len({r["session"] for r in ra}),
                         crop_ncc_med=q([r.get("crop_ncc") for r in ra], .5),
                         d_seek_med=q([r.get("d_seek_s") for r in ra if r["grade"] in ("high", "medium")], .5),
                         d_key_frames=";".join(str(r.get("d_key_frames")) for r in ra),
                         clock_off_med=q([r.get("clock_off_s") for r in ra if r["grade"] in ("high", "medium")], .5),
                         clock_off_range=(lambda v: f"{min(v):.3f}..{max(v):.3f}" if v else "")(
                             [r["clock_off_s"] for r in ra if r.get("clock_off_s") is not None and r["grade"] in ("high", "medium")]),
                         eeg_ok=sum(1 for r in ra if r.get("eeg_status") == "ok"),
                         eeg_lag_ms=";".join(str(r.get("eeg_lag_ms")) for r in ra),
                         eeg_minus_video_med=q([r.get("eeg_minus_video_s") for r in ra if r.get("eeg_status") == "ok"], .5),
                         act_lag=";".join(str(r.get("act_lag_s")) for r in ra)))
    write_csv("per_animal.csv", list(prow[0].keys()) if prow else ["animal"], prow)
    P("PER ANIMAL (main stratum)")
    for p in prow:
        P("  " + "  ".join(f"{k}={_cell(v)}" for k, v in p.items()))
    P("")
    P("EEG (all strata)")
    P(f"  status: {dict(Counter(r.get('eeg_status') for r in rows))}")
    ok = [r for r in rows if r.get("eeg_status") == "ok"]
    for k, nd in (("eeg_lag_ms", 1), ("eeg_ncc", 6), ("eeg_slope", 4), ("eeg_minus_video_s", 3),
                  ("clip_eeg_unique", 0)):
        P(f"  {k:20s} {dist([fnum(r.get(k)) for r in ok], nd)}")
    P(f"  lag counts (samples): {dict(Counter(r.get('eeg_lag_samples') for r in ok))}")
    P(f"  clip EDF 1 kHz label used: {dict(Counter(r.get('clip_edf_label') for r in rows if r.get('clip_eeg_present')))}"
      f"   parent reader ('ECG' in label) would take: {dict(Counter(r.get('clip_parent_reader_label') for r in rows if r.get('clip_eeg_present')))}")
    P("")
    P("EDF ACTIVITY (1 Hz) vs VIDEO MOTION in the animal's crop, +-300 s around each clip; r(L) = corr(activity[t],")
    P("motion[t+L]) on log1p values, 1-s bins on the EDF clock, video mapped by XML start + sync_offset + pts.")
    P("A peak at L > 0 means the video clock runs L s late relative to the EDF clock.")
    P(f"  per-clip best lag (s): {dict(sorted(Counter(r.get('act_lag_s') for r in rows if r.get('act_lag_s') is not None).items()))}")
    act = {}
    mainsel = lambda r: r["stratum"] == "main"  # noqa: E731
    for grp, sel in [("main all", mainsel), ("main 1402C", lambda r: mainsel(r) and r.get("room") == "1402C"),
                     ("main 1402D", lambda r: mainsel(r) and r.get("room") == "1402D"),
                     ("clock", lambda r: r["stratum"] == "clock"), ("defect", lambda r: r["stratum"] == "defect")] + \
            [(a, (lambda a: lambda r: mainsel(r) and r["animal"] == a)(a)) for a in sorted({r["animal"] for r in rows})]:
        curves = [r["act_curve"] for r in rows if r.get("act_curve") and sel(r)]
        if not curves:
            continue
        mean = {}
        for lag in range(-30, 31):
            v = [c[str(lag)] for c in curves if str(lag) in c]
            if len(v) >= max(1, len(curves) // 2):
                mean[lag] = sum(v) / len(v)
        if not mean:
            continue
        Lb = max(mean, key=mean.get)
        second = max((v for k, v in mean.items() if abs(k - Lb) > 2), default=None)
        act[grp] = dict(n=len(curves), best_lag=Lb, r_best=round(mean[Lb], 4), r0=round(mean.get(0, float("nan")), 4),
                        r_second_gt2s=None if second is None else round(second, 4),
                        n_best_within_2s=sum(1 for r in rows if r.get("act_curve") and sel(r) and r.get("act_lag_s") is not None
                                             and abs(r["act_lag_s"]) <= 2))
        P(f"  {grp:11s} n={len(curves):3d}  mean-curve peak L={Lb:+d}s r={mean[Lb]:.3f}  r(0)={mean.get(0, float('nan')):.3f}"
          f"  best elsewhere (|dL|>2)={second if second is None else round(second, 3)}"
          f"  clips with own peak |L|<=2: {act[grp]['n_best_within_2s']}/{len(curves)}")
    P(f"  act_r_best {dist([fnum(r.get('act_r_best')) for r in rows], 3)}")
    P(f"  act_r0     {dist([fnum(r.get('act_r0')) for r in rows], 3)}")
    P("")
    P("DEFECT STRATUM (clips cut from raw files whose frames do not fill the wall-clock span)")
    for r in rows:
        if r["stratum"] == "defect":
            P(f"  {r['animal']} {r['session']}/{r['clip_dir']} raw={r['raw_video']} grade={r['grade']} "
              f"d_seek={_cell(r.get('d_seek_s'))} wide-narrow={_cell(r.get('wide_minus_narrow_s'))} "
              f"d_key={r.get('d_key_frames')} eeg={r.get('eeg_status')} eeg-video={_cell(r.get('eeg_minus_video_s'))} "
              f"act_lag={r.get('act_lag_s')} r={_cell(r.get('act_r_best'))} long_lag={r.get('act_long_lag_s')}"
              f" r={_cell(r.get('act_long_r'))} 2nd={_cell(r.get('act_long_second'))} r@0={_cell(r.get('act_long_r_at0'))}")
    P("")
    P("CLOCK SCANS ALONG ONE FILE (EDF Activity vs video motion, 20-30 min windows; EDF mapped as start + elapsed)")
    P("  top = best lags (s) with r; a step from ~-1 s to another lag locates missing wall time")
    brackets = {}
    for f in sorted(glob.glob(out_path("scans", "*.json"))):
        sc = json.load(open(f))
        P(f"  {sc['name']}  box={sc['box']}  window={sc['window_s']} s  lags searched +-{sc['lagmax_s']} s")
        normal, shifted = [], []
        for pt in sc["points"]:
            if "top" not in pt:
                P(f"    offset {pt['offset']:>7.0f}  {pt.get('error')}")
                continue
            (l1, r1), (l2, r2) = pt["top"][0], pt["top"][1]
            P(f"    offset {pt['offset']:>7.0f} s  EDF {pt['edf_elapsed_h']:>6.2f} h  top {l1:+6d} s r={r1:.3f} "
              f"(next {l2:+d} r={r2:.3f})  r at {pt['r_at']}  sd={pt['sd']}")
            if abs(l1 + 1) <= 4:
                normal.append(pt["offset"])
            elif r1 - r2 >= 0.02:
                shifted.append((pt["offset"], l1))
        brackets[os.path.basename(f)[:-5]] = (normal, shifted)
    P("")
    P("LABELLED CLIPS IN TIMING-DEFECT FILES (discover() clips; gap position from the scans above)")
    gap = {"20231014004100": (0.0, 0.0)}
    for key, tag in (("1103_RN235", "20231103122100"), ("1205_RN235", "20231205222300")):
        if key in brackets:
            normal, shifted = brackets[key]
            if normal and shifted:
                lo = max(o for o in normal if o < min(o2 for o2, _ in shifted))
                gap[tag] = (lo + 1200.0, min(o for o, _ in shifted))   # a normal window covers [o, o + 1200]
    dc = Counter()
    for r in read_csv(out_path("clips.csv")):
        if r["in_discover"] != "True" or "frames_missing_vs_wallclock" not in (r.get("raw_problems") or ""):
            continue
        tag = r["raw_video"].split(".")[-2]
        sp_ = fnum(r["seek_pred_s"]) or 0.0
        lo, hi = gap.get(tag, (None, None))
        if tag == "20231014004100":
            pos = "whole file shifted"
        elif lo is None:
            pos = "gap not located"
        elif sp_ + (fnum(r["clip_dur_s"]) or 0) <= lo:
            pos = "before the gap (aligned)"
        elif sp_ >= hi:
            pos = "after the gap (video late)"
        else:
            pos = "inside the bracket (uncertain)"
        dc[(tag, pos, r["kind"])] += 1
    P(f"  gap brackets (pts s): {gap}")
    for k, v in sorted(dc.items()):
        P(f"  {k[0]}  {k[1]:32s} {k[2]:10s} {v}")
    P("")
    P("CLOCK STRATUM (room 1402C day folders whose EDF starts 198.5 s / -748.4 s from the first video instead of 18.5 s)")
    for r in rows:
        if r["stratum"] == "clock":
            P(f"  {r['animal']} {r['session']}/{r['clip_dir']} raw={r['raw_video']} grade={r['grade']} "
              f"d_key={r.get('d_key_frames')} eeg={r.get('eeg_status')} lag={r.get('eeg_lag_ms')}ms "
              f"act_lag={r.get('act_lag_s')} r={_cell(r.get('act_r_best'))} r0={_cell(r.get('act_r0'))} "
              f"long_lag={r.get('act_long_lag_s')} r={_cell(r.get('act_long_r'))} 2nd={_cell(r.get('act_long_second'))}")
    P("")
    P("CONTENT (labelled span [10 s, dur-10 s] vs the 10 s pre-buffer; seizure clips vs the same span of non-seizure clips)")
    for kind in ("seizure", "nonseizure"):
        rk = [r for r in main if r["kind"] == kind]
        P(f"  {kind:10s} video motion ratio {dist([fnum(r.get('vid_motion_ratio')) for r in rk], 2)}")
        P(f"  {kind:10s} EEG line-length    {dist([fnum(r.get('eeg_ll_ratio')) for r in rk], 2)}")
    P("")
    # every labelled clip (inventory), not just the sample: which ones straddle the DST change?
    vutc = {f["relpath"]: int(f["xml_utc_start"]) for f in read_csv(os.path.join(AUDIT, "files.csv"))
            if f["kind"] == "video" and f["xml_utc_start"]}
    dcount = Counter()
    for r in read_csv(out_path("clips.csv")):
        if r["in_discover"] != "True" or not r.get("raw_video"):
            continue
        v_off = dt.datetime.fromtimestamp(vutc[r["raw_video"]], tz=UTC).astimezone(CENTRAL).utcoffset().total_seconds()
        l_off = utc_offset_s(parse_dt(r["info_clip_start"]))
        e_off = utc_offset_s(parse_dt(r["edf_start_local"])) if r.get("edf_start_local") else l_off
        if ambiguous_dst(parse_dt(r["info_clip_start"])):
            dcount[("clip start in the repeated hour 01:00-02:00 (unresolvable)", r["animal"], r["kind"])] += 1
        if v_off != l_off:
            dcount[(f"video started before, label after: video cut {v_off - l_off:+.0f} s off", r["animal"], r["kind"])] += 1
        elif e_off != l_off:
            dcount[("EDF started before, video and label after: consistent", r["animal"], r["kind"])] += 1
    P("DST EXPOSURE, all discover() clips with an info.txt:")
    for k, v in sorted(dcount.items()):
        P(f"  {k[0]:55s} {k[1]} {k[2]:10s} {v}")
    P("")
    P("DST STRATUM (after the 2023-11-05 fall-back; the cutter used naive local times)")
    P("  ratios are labelled-span / pre-buffer: EEG line length and video motion, at the clip as cut and 1 h either side")
    for r in rows:
        if r["stratum"] == "dst":
            P(f"  {r['animal']} {r['session']}/{r['clip_dir']} video_seek_error={_cell(r.get('video_seek_error_s'))} "
              f"edf_other_side={_cell(r.get('edf_start_other_side_s'))}  EEG ll: -1h={_cell(r.get('eeg_ll_ratio_minus1h'))} "
              f"as-cut={_cell(r.get('eeg_ll_ratio'))} +1h={_cell(r.get('eeg_ll_ratio_plus1h'))}  video motion: "
              f"-1h={_cell(r.get('vid_motion_ratio_minus1h'))} as-cut={_cell(r.get('vid_motion_ratio'))} "
              f"+1h={_cell(r.get('vid_motion_ratio_plus1h'))}  act_lag={r.get('act_lag_s')} r={_cell(r.get('act_r_best'))} "
              f"r0={_cell(r.get('act_r0'))} long={r.get('act_long_lag_s')} grade={r['grade']} eeg={r.get('eeg_status')} "
              f"eeg-video={_cell(r.get('eeg_minus_video_s'))}")
    P("")
    P("NON-HIGH MAIN CLIPS")
    for r in main:
        if r["grade"] != "high":
            P(f"  {r['animal']} {r['session']}/{r['clip_dir']} grade={r['grade']} crop_ncc={_cell(r.get('crop_ncc'))} "
              f"tA={_cell(r.get('tA_peak'))} tB={_cell(r.get('tB_peak'))}/{_cell(r.get('tB_margin'))} "
              f"tCz={_cell(r.get('tC_z'))} spread={r.get('argmax_spread_frames')} segs={r.get('segment_offsets_frames')} "
              f"errors={r.get('stage_errors')}")
    write_text("align_summary.txt", "\n".join(L) + "\n")
    write_text("align.json", json.dumps(dict(n_sampled=len(sm), n_records=len(rows), missing=missing,
                                             grades=dict(Counter(r["grade"] for r in main)),
                                             crop_boxes=crow, per_animal=prow, activity=act), indent=1, default=str))
    print("\n".join(L[:200]))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("inventory")
    p.add_argument("--threads", type=int, default=32)
    p = sp.add_parser("sample")
    p.add_argument("--n-seizure", type=int, default=4)
    p.add_argument("--n-nonseizure", type=int, default=2)
    p.add_argument("--n-defect-per-file", type=int, default=4)
    p.add_argument("--n-clock", type=int, default=16)
    p.add_argument("--seed", type=int, default=20260928)
    sp.add_parser("list")
    p = sp.add_parser("match")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--animal")
    g.add_argument("--index", type=int)
    p = sp.add_parser("scan")
    p.add_argument("names", nargs="*", help=f"any of {sorted(SCANS)} (default: all)")
    sp.add_parser("merge")
    a = ap.parse_args()
    os.makedirs(out_path(), exist_ok=True)
    dict(inventory=cmd_inventory, sample=cmd_sample, list=cmd_list, match=cmd_match, scan=cmd_scan,
         merge=cmd_merge)[a.cmd](a)


if __name__ == "__main__":
    main()
