#!/usr/bin/env python
"""Read-only inventory of the seizure ANNOTATIONS and their agreement with the labelled clips.

    RAW      /work/mech-ai/alloydas/EEG/Data                (never written; xlsx zip headers and a
                                                             few EDF records per sampled event are read)
    AUDIT    $EEG_ROOT/output/raw_audit/{events,files}.csv  (the integrity audit; read)
    LABELLED $EEG_ROOT/data/Data_<A>_cropped/<day>/<dir>/   (zero-byte placeholders: names only)
             $EEG_ROOT/data_full/Data_<A>_cropped/...       (real clips: info.txt read, a sample ffprobed)
    EXPORT   $EEG_ROOT/raw_epoch_features/Room?/<A>/*.csv   (DSI 10-s epoch features; a few columns read)
    OUT      $EEG_ROOT/output/raw_audit/annotations/        (the ONLY place written)

Sub-commands (each heavy unit caches under OUT/cache/ and is skipped when its cache exists):
    labelled   [--animal A]   list data/ and data_full/ clip dirs, parse every data_full info.txt
    xlsxzip                   zip-member timestamps + sheet dimension of every raw xlsx (export time)
    epochs     [--animal A]   line-length robust-z detector over the epoch export, per animal
    onset      [--animal A]   EDF samples around up to N annotated onsets per animal: 1-s line length
    detprobe   [--animal A]   EDF around unannotated line-length detections and annotated seizures:
                              ictal and post-ictal line length (needs the epochs cache)
    clipprobe                 ffprobe duration of a sample of data_full seizure clips
    gop                       keyframe spacing of 4 raw recordings (packet flags, first 120 s)
    report                    join everything; write the CSV/JSON outputs and summary.txt
    all        [--workers W]  every unit above (process pool), then report

Standard library only (numpy/pandas page in too slowly from the 99%-full conda filesystem).
"""
import argparse
import csv
import datetime as dt
import glob
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import time
import traceback
import zipfile
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict

RAW = os.path.realpath(os.environ.get("RAW_ROOT", "/work/mech-ai/alloydas/EEG/Data"))
EEG_ROOT = os.path.realpath(os.environ.get("EEG_ROOT", "/work/mech-ai-scratch/alloy/EEG"))
AUDIT = os.path.join(EEG_ROOT, "output", "raw_audit")
OUT = os.path.join(AUDIT, "annotations")
CACHE = os.path.join(OUT, "cache")
LAB = os.path.join(EEG_ROOT, "data")
LABFULL = os.path.join(EEG_ROOT, "data_full")
EXPORT = os.path.join(EEG_ROOT, "raw_epoch_features")
RN243_AUDIT = os.path.join(EEG_ROOT, "output", "rn243_scale", "rn243_scale_audit.json")
FFPROBE = os.environ.get("FFPROBE", "/work/mech-ai/alloy/miniconda3/bin/ffprobe")

LABELLED_ANIMALS = ("RN197 RN199 RN204 RN208 RN210 RN213 RN215 RN216 RN219 RN222 RN223 RN224 "
                    "RN227 RN229 RN235 RN237 RN238 RN242 RN244 RN245").split()
UNLABELLED_ANIMALS = ("RN201", "RN203", "RN243")
ALL_ANIMALS = sorted(LABELLED_ANIMALS + list(UNLABELLED_ANIMALS))
DISCOVER_STAGES = {"Stage_2", "Stage_3", "Stage_4", "Stage_5"}      # train_pooled.STAGE
SEIZURE_LABELS = {"Stage 2", "Stage 3", "Stage 4", "Stage 5"}
MATCH_TOL_S = 60.0
PARTNER_TOL_S = 60.0

# epoch-export detector: identical definition to rn243_scale_audit.py
EPOCH_S, MIN_DUR, MERGE_GAP = 10.0, 4.0, 4.0
EXPORT_DROP_DIRS = {os.path.join("RoomD", "RN245")}      # lab said discard; 23-epoch rail stub
EXPORT_COLS = ["animal_id", "day_folder", "recording_id", "epoch_index", "channel_index",
               "epoch_start_native", "epoch_duration_s", "features_valid",
               "annotation_overlap_s", "line_length_uV_per_s"]
ANNOT_PAD_S = 60.0          # a detection within +-60 s of any annotated event counts as annotated
SENS_PAD_S = 10.0           # an annotated event is "hit" if a detection touches [start-10, end+10]

# onset profiles
ONSET_PER_ANIMAL = int(os.environ.get("ONSET_PER_ANIMAL", "40"))
PRE_S, POST_S = 60, 90       # seconds read before / after the annotated start
BASE = (-60, -20)            # baseline bins [-60, -20)
SEARCH = (-20, 40)           # change-point search bins
K_MAD, RUN = 5.0, 3          # change = RUN consecutive 1-s bins above median + K_MAD * robust SD
CLIPPROBE_PER_ANIMAL = 15


# ------------------------------------------------------------------ write guard

def _under(path, root):
    p, r = os.path.realpath(path), os.path.realpath(root)
    return p == r or p.startswith(r + os.sep)


def out_path(*parts):
    p = os.path.join(OUT, *parts)
    if not _under(p, OUT) or _under(p, RAW):
        raise RuntimeError(f"refusing to write outside {OUT}: {p}")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def write_atomic(path, text):
    path = out_path(os.path.relpath(path, OUT)) if os.path.isabs(path) else out_path(path)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def write_json(rel, obj):
    write_atomic(rel, json.dumps(obj, indent=1, default=str))


def write_csv(rel, cols, rows):
    import io
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: _cell(r.get(k)) for k in cols})
    write_atomic(rel, buf.getvalue())


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return "" if math.isnan(v) else f"{v:.6g}" if abs(v) < 1e6 else f"{v:.1f}"
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(v, default=str)
    return v


def ropen(path, mode="rb"):
    if "w" in mode or "a" in mode or "+" in mode:
        raise RuntimeError("read-only")
    return open(path, mode)


# ------------------------------------------------------------------ small helpers

def parse_dt(s):
    if s is None or s == "" or s == "None":
        return None
    if isinstance(s, dt.datetime):
        return s
    s = str(s).strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return dt.datetime.strptime(s[:26], fmt)
        except ValueError:
            pass
    return None


EPOCH0 = dt.datetime(1970, 1, 1)


def tsec(d):
    return (d - EPOCH0).total_seconds()


def animal_of(folder):
    return folder.replace("DONE_", "")


def q(xs, p):
    xs = sorted(x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x)))
    if not xs:
        return None
    k = (len(xs) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def dist(xs, nd=2):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return "n=0"
    f = lambda v: f"{v:.{nd}f}"
    return (f"n={len(xs)} min={f(min(xs))} p5={f(q(xs, .05))} p25={f(q(xs, .25))} median={f(q(xs, .5))} "
            f"p75={f(q(xs, .75))} p95={f(q(xs, .95))} max={f(max(xs))}")


def label_of_dirname(lbl):
    """'Stage_3' -> 'Stage 3'; 'Spikes' -> 'Spikes'."""
    return lbl.replace("_", " ")


SZ_RE = re.compile(r"^seizure_(\d+)_(.+)_(\d{8})_(\d{6})$")
NS_RE = re.compile(r"^clip_(\d+)_vs_seizure_(\d+)_(.+)_(\d{8})_(\d{6})$")


def parse_info(path):
    d = {}
    with ropen(path, "r") as f:
        for line in f:
            if ":" in line:
                k, v = line.split(":", 1)
                d[k.strip()] = v.strip()
    return d


def load_csv(path):
    with ropen(path, "r") as f:
        return list(csv.DictReader(f))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ================================================================== unit: labelled tree

def unit_labelled(animal):
    cp = os.path.join(CACHE, f"labelled_{animal}.json")
    if os.path.exists(cp):
        return "cached"
    rows = {}
    for tree, root in (("data", LAB), ("full", LABFULL)):
        aroot = os.path.join(root, f"Data_{animal}_cropped")
        if not os.path.isdir(aroot):
            continue
        for day in sorted(os.listdir(aroot)):
            dpath = os.path.join(aroot, day)
            if not os.path.isdir(dpath):
                continue
            for name in os.listdir(dpath):
                if not (name.startswith("seizure_") or name.startswith("clip_")):
                    continue
                p = os.path.join(dpath, name)
                if not os.path.isdir(p):
                    continue
                r = rows.setdefault((day, name), {"animal": animal, "day": day, "name": name})
                files = {}
                for fn in os.listdir(p):
                    try:
                        files[fn] = os.stat(os.path.join(p, fn)).st_size
                    except OSError:
                        files[fn] = None
                r[f"in_{tree}"] = True
                r[f"{tree}_files"] = files
                if tree == "full" and "info.txt" in files and files["info.txt"]:
                    try:
                        r["info"] = parse_info(os.path.join(p, "info.txt"))
                    except Exception as e:
                        r["info_error"] = repr(e)
    out = sorted(rows.values(), key=lambda r: (r["day"], r["name"]))
    write_json(os.path.relpath(cp, OUT), {"animal": animal, "rows": out})
    return f"{len(out)} dirs"


# ================================================================== unit: xlsx zip times

def unit_xlsxzip():
    cp = os.path.join(CACHE, "xlsx_zip.json")
    if os.path.exists(cp):
        return "cached"
    files = [r for r in load_csv(os.path.join(AUDIT, "files.csv")) if r["kind"] == "xlsx"]
    out = []
    for r in files:
        p = os.path.join(RAW, r["relpath"])
        rec = {"relpath": r["relpath"], "animal": r["animal"], "animal_folder": r["animal_folder"],
               "day_folder": r["day_folder"], "n_events": r.get("n_events")}
        try:
            with ropen(p) as fh:
                z = zipfile.ZipFile(fh)
                infos = {i.filename: i for i in z.infolist()}
                sh = infos.get("xl/worksheets/sheet1.xml")
                rec["members"] = len(infos)
                rec["has_docprops"] = any(n.startswith("docProps/") for n in infos)
                rec["zip_time"] = dt.datetime(*sh.date_time).isoformat() if sh else None
                times = sorted({dt.datetime(*i.date_time).isoformat() for i in infos.values()})
                rec["zip_times_distinct"] = len(times)
                head = z.read("xl/worksheets/sheet1.xml")[:600].decode("utf-8", "replace") if sh else ""
                m = re.search(r'dimension ref="([^"]+)"', head)
                rec["dimension"] = m.group(1) if m else None
        except Exception as e:
            rec["error"] = repr(e)
        out.append(rec)
    write_json(os.path.relpath(cp, OUT), out)
    return f"{len(out)} xlsx"


# ================================================================== unit: epoch export detector

def _extract_intervals(times, win=EPOCH_S, min_dur=MIN_DUR, merge_gap=MERGE_GAP):
    """sweep_session.extract_intervals on already-thresholded hit start times."""
    times = sorted(times)
    if not times:
        return []
    merged = [[times[0], times[0] + win]]
    for s in times[1:]:
        e = s + win
        if s - merged[-1][1] <= merge_gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [tuple(m) for m in merged if m[1] - m[0] >= min_dur]


def _events_by_animal():
    ev = defaultdict(list)
    for r in load_csv(os.path.join(AUDIT, "events.csv")):
        s, e = parse_dt(r["start"]), parse_dt(r["end"])
        if s is None:
            continue
        if e is None:
            e = s + dt.timedelta(seconds=float(r["duration_s"] or 0))
        ev[r["animal"]].append((tsec(s), tsec(e), r["label"]))
    for a in ev:
        ev[a].sort()
    return ev


def export_files():
    out = defaultdict(list)
    skipped = []
    for p in sorted(glob.glob(os.path.join(EXPORT, "*", "*", "*"))):
        rel = os.path.relpath(p, EXPORT)
        if ".interrupted-" in os.path.basename(p) or not p.endswith(".csv"):
            skipped.append(rel)
            continue
        if os.path.dirname(rel) in EXPORT_DROP_DIRS:
            skipped.append(rel)
            continue
        out[animal_of(os.path.basename(os.path.dirname(p)))].append(p)
    return out, skipped


def thresholds():
    d = json.load(ropen(RN243_AUDIT, "r"))
    return {k: float(v) for k, v in d["calibration"]["thresholds"].items() if v is not None}


def _iso_seconds(v):
    try:
        return tsec(dt.datetime.fromisoformat(v[:26]))
    except (ValueError, TypeError):
        d = parse_dt(v)
        return tsec(d) if d else None


def _f(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def read_export(path):
    """One export CSV -> (raw rows, deduped rows sorted by (recording_id, epoch_index)).

    Dedupe on (recording_id, epoch_index), keeping the lowest channel_index (the 1 kHz channel;
    RoomC/DONE_RN203/11-15-2023 exports three). Row = (day, t_s, dur_s, valid, overlap_s, line_length)."""
    with open(path, "r", newline="") as f:
        rd = csv.reader(f)
        hdr = next(rd)
        ix = {c: i for i, c in enumerate(hdr)}
        miss = [c for c in EXPORT_COLS if c not in ix]
        if miss:
            raise ValueError(f"missing columns {miss}")
        i_d, i_r, i_e, i_c, i_t, i_u, i_v, i_o, i_l = (ix[c] for c in (
            "day_folder", "recording_id", "epoch_index", "channel_index", "epoch_start_native",
            "epoch_duration_s", "features_valid", "annotation_overlap_s", "line_length_uV_per_s"))
        need = max(i_d, i_r, i_e, i_c, i_t, i_u, i_v, i_o, i_l)
        best, n0 = {}, 0
        for row in rd:
            if len(row) <= need:
                continue
            n0 += 1
            key = (row[i_r], int(float(row[i_e])) if row[i_e] else -1)
            ch = _f(row[i_c])
            ch = 99.0 if ch is None else ch
            prev = best.get(key)
            if prev is None or ch < prev[0]:
                best[key] = (ch, row[i_d], row[i_t], row[i_u], row[i_v], row[i_o], row[i_l])
    out = []
    for key in sorted(best):
        _, d, t, u, v, o, l = best[key]
        out.append((d, _iso_seconds(t), _f(u) or 0.0, _f(v) == 1.0, _f(o) or 0.0, _f(l)))
    return n0, out


def unit_epochs(animal):
    cp = os.path.join(CACHE, f"epochs_{animal}.json")
    if os.path.exists(cp):
        return "cached"
    files, _ = export_files()
    thr = thresholds()
    evs = _events_by_animal().get(animal, [])
    ev_s = [e[0] for e in evs]
    ev_pad = sorted((s - ANNOT_PAD_S, e + ANNOT_PAD_S) for s, e, _ in evs)
    pad_starts = [a for a, _ in ev_pad]
    # running max of pad ends lets an interval query use one bisect
    run_max, m = [], -1e18
    for _, b in ev_pad:
        m = max(m, b)
        run_max.append(m)

    def near_annotation(s, e):
        i = bisect_right(pad_starts, e) - 1
        return i >= 0 and run_max[i] >= s

    res = {"animal": animal, "thresholds": thr, "sessions": [], "files": []}
    per_thr = {k: {"n_det": 0, "n_det_ge20": 0, "n_unannot": 0, "n_unannot_ge20": 0} for k in thr}
    all_det = []
    seizure_ev = [(s, e, l) for s, e, l in evs if l in SEIZURE_LABELS]
    covered_ev = set()
    ev_hit = {k: set() for k in thr}
    tot_h = valid_h = 0.0
    annot_epochs = annot_epochs_near_xlsx = 0
    for p in files.get(animal, []):
        rel = os.path.relpath(p, EXPORT)
        try:
            n0, rows = read_export(p)
        except Exception as e:
            res["files"].append({"file": rel, "error": repr(e)})
            continue
        if not rows:
            res["files"].append({"file": rel, "rows": 0})
            continue
        t = [r[1] for r in rows]
        dur = [r[2] for r in rows]
        ov = [r[4] for r in rows]
        ok = [r[3] and r[5] is not None and r[5] > 0 and r[1] is not None for r in rows]
        x = [r[5] for r in rows]
        z = [None] * len(rows)
        stat = {"file": rel, "rows": int(n0), "epochs": len(rows), "valid": sum(ok),
                "hours": sum(dur) / 3600, "valid_hours": sum(d for d, o in zip(dur, ok) if o) / 3600,
                "day_folder": rows[0][0]}
        tot_h += stat["hours"]
        valid_h += stat["valid_hours"]
        # export annotation vs xlsx: every epoch the export marks annotated should touch an xlsx event
        for ti, o in zip(t, ov):
            if not (o > 0) or ti is None:
                continue
            annot_epochs += 1
            j = bisect_right(ev_s, ti + EPOCH_S + 1) - 1
            while j >= 0 and ev_s[j] > ti - 3600:
                if evs[j][0] - 1 <= ti + EPOCH_S and evs[j][1] + 1 >= ti:
                    annot_epochs_near_xlsx += 1
                    break
                j -= 1
        if sum(ok) >= 30:
            lx = [math.log10(v) for v, o in zip(x, ok) if o]
            med = statistics.median(lx)
            mad = statistics.median([abs(v - med) for v in lx]) * 1.4826
            if not mad > 0:
                mad = statistics.pstdev(lx)
            if mad > 0:
                z = [(math.log10(v) - med) / mad if o else None for v, o in zip(x, ok)]
                stat.update(z_med=med, z_mad=mad)
        tv = [v for v in t if v is not None]
        t0, t1 = (min(tv), max(tv) + EPOCH_S) if tv else (None, None)
        stat.update(t0=t0, t1=t1)
        if t0 is not None:
            lo, hi = bisect_left(ev_s, t0), bisect_right(ev_s, t1)
            for s, e, l in evs[lo:hi]:
                if l in SEIZURE_LABELS:
                    covered_ev.add((s, e))
        for k, th in thr.items():
            hits = [(ti, zi) for ti, zi in zip(t, z) if zi is not None and zi >= th]
            iv = _extract_intervals([h[0] for h in hits])
            zz = {round(a_, 1): b_ for a_, b_ in hits}
            n_u = n_u20 = n20 = 0
            for s, e in iv:
                d = e - s
                na = near_annotation(s, e)
                n20 += d >= 20
                if not na:
                    n_u += 1
                    n_u20 += d >= 20
                if k == "0.5" and (animal in UNLABELLED_ANIMALS or not na) and d >= 20:
                    zin = [zz[k_] for k_ in zz if s - 0.05 <= k_ < e]
                    mz = max(zin) if zin else float("nan")
                    all_det.append({"file": rel, "day_folder": stat["day_folder"],
                                    "start": (EPOCH0 + dt.timedelta(seconds=s)).isoformat(sep=" "),
                                    "end": (EPOCH0 + dt.timedelta(seconds=e)).isoformat(sep=" "),
                                    "dur_s": d, "max_z": mz, "near_annotation": na})
                # sensitivity: seizure events touched by this detection
                lo = bisect_left(ev_s, s - 3600)
                for es, ee, el in evs[lo:bisect_right(ev_s, e + SENS_PAD_S)]:
                    if el in SEIZURE_LABELS and es - SENS_PAD_S <= e and ee + SENS_PAD_S >= s:
                        ev_hit[k].add((es, ee))
            pt = per_thr[k]
            pt["n_det"] += len(iv)
            pt["n_det_ge20"] += n20
            pt["n_unannot"] += n_u
            pt["n_unannot_ge20"] += n_u20
            stat[f"det_{k}"] = len(iv)
            stat[f"unannot_ge20_{k}"] = n_u20
        res["sessions"].append(stat)
    for k in thr:
        pt = per_thr[k]
        pt["seizure_events_covered"] = len(covered_ev)
        pt["seizure_events_hit"] = len(ev_hit[k] & covered_ev)
    res.update(per_threshold=per_thr, hours=tot_h, valid_hours=valid_h,
               n_seizure_events_xlsx=len(seizure_ev), n_events_xlsx=len(evs),
               export_annotated_epochs=annot_epochs, export_annotated_epochs_touching_xlsx=annot_epochs_near_xlsx,
               detections_listed=all_det, n_files=len(files.get(animal, [])))
    write_json(os.path.relpath(cp, OUT), res)
    return f"{len(res['sessions'])} files {valid_h:.0f} valid h"


# ================================================================== unit: EDF onset profiles

def edf_header(path):
    with ropen(path) as f:
        h = f.read(256)
        ns = int(h[252:256].decode().strip())
        s = f.read(ns * 256)
    fld = lambda a, b, w: [s[a * ns + i * w: a * ns + (i + 1) * w].decode("latin1").strip() for i in range(ns)]
    labels = fld(0, 0, 16)
    nsamp = [int(s[216 * ns + i * 8: 216 * ns + (i + 1) * 8].decode().strip()) for i in range(ns)]
    return {"ns": ns, "labels": labels, "nsamp": nsamp,
            "hdr_bytes": int(h[184:192].decode().strip()),
            "nrec": int(h[236:244].decode().strip()),
            "rec_dur": float(h[244:252].decode().strip()),
            "start": dt.datetime.strptime(h[168:176].decode() + " " + h[176:184].decode(), "%d.%m.%y %H.%M.%S")}


def pick_channel(hd):
    fs = [n / hd["rec_dur"] for n in hd["nsamp"]]
    for want in ("ECG", "EEG"):
        for i, l in enumerate(hd["labels"]):
            if l == want and abs(fs[i] - 1000) < 1:
                return i
    for i, l in enumerate(hd["labels"]):
        if ("ECG" in l or "EEG" in l) and "FIR" not in l and abs(fs[i] - 1000) < 1:
            return i
    return None


def read_window(path, hd, ch, t_from, t_to):
    """Samples of channel ch for [t_from, t_to) seconds from the EDF start (records read only)."""
    from array import array
    rd = hd["rec_dur"]
    r0, r1 = int(math.floor(t_from / rd)), int(math.ceil(t_to / rd))
    recb = sum(hd["nsamp"]) * 2
    size = os.path.getsize(path)
    nrec_file = (size - hd["hdr_bytes"]) // recb
    if r0 < 0 or r1 > nrec_file:
        return None, None
    with ropen(path) as f:
        f.seek(hd["hdr_bytes"] + r0 * recb)
        buf = f.read((r1 - r0) * recb)
    a = array("h")
    a.frombytes(buf)
    if sys.byteorder == "big":
        a.byteswap()
    tot, c0, n = sum(hd["nsamp"]), sum(hd["nsamp"][:ch]), hd["nsamp"][ch]
    sig = array("h")
    for r in range(r1 - r0):
        sig.extend(a[r * tot + c0: r * tot + c0 + n])
    return sig, r0 * rd


def unit_onset(animal):
    cp = os.path.join(CACHE, f"onset_{animal}.json")
    if os.path.exists(cp):
        return "cached"
    evs = [r for r in load_csv(os.path.join(AUDIT, "events.csv")) if r["animal"] == animal]
    edfs = [r for r in load_csv(os.path.join(AUDIT, "files.csv")) if r["kind"] == "edf" and r["animal"] == animal]
    edf_iv = [(parse_dt(r["edf_start"]), parse_dt(r["edf_end"]), r["relpath"]) for r in edfs
              if parse_dt(r["edf_start"]) and parse_dt(r["edf_end"])]
    pool = [r for r in evs if (r["label"] in SEIZURE_LABELS or animal in UNLABELLED_ANIMALS)]
    rng = random.Random(f"onset-{animal}")
    rng.shuffle(pool)
    out = {"animal": animal, "n_pool": len(pool), "events": [], "skipped": Counter()}
    hdr_cache = {}
    for r in pool:
        if len(out["events"]) >= ONSET_PER_ANIMAL:
            break
        st = parse_dt(r["start"])
        cand = [e for e in edf_iv if e[0] <= st - dt.timedelta(seconds=PRE_S + 1)
                and st + dt.timedelta(seconds=POST_S + 1) <= e[1]]
        if not cand:
            out["skipped"]["no_edf_window"] += 1
            continue
        rel = cand[0][2]
        p = os.path.join(RAW, rel)
        try:
            hd = hdr_cache.get(rel) or edf_header(p)
            hdr_cache[rel] = hd
            ch = pick_channel(hd)
            if ch is None:
                out["skipped"]["no_1khz_channel"] += 1
                continue
            t0 = (st - hd["start"]).total_seconds()
            sig, sig_t0 = read_window(p, hd, ch, t0 - PRE_S - 1, t0 + POST_S + 1)
            if sig is None:
                out["skipped"]["window_outside_records"] += 1
                continue
            ll = []
            for k in range(-PRE_S, POST_S):
                i0 = int(round((t0 + k - sig_t0) * 1000))
                seg = sig[i0:i0 + 1000]
                ll.append(sum(abs(seg[j + 1] - seg[j]) for j in range(999)) / 999 if len(seg) == 1000 and i0 >= 0 else None)
            base = [v for v in ll[BASE[0] + PRE_S: BASE[1] + PRE_S] if v is not None]
            med = statistics.median(base) if base else 0.0
            mad = statistics.median([abs(v - med) for v in base]) * 1.4826 if base else 0.0
            rec = {"row": int(r["row"]), "day_folder": r["day_folder"], "label": r["label"], "start": r["start"],
                   "duration_s": float(r["duration_s"]), "edf": rel, "channel": hd["labels"][ch],
                   "base_med": med, "base_mad": mad,
                   "flat_frac": sum(1 for v in ll if v == 0) / len(ll)}
            if not (mad > 0 and med > 0):
                rec["degenerate_baseline"] = True
                out["events"].append(rec)
                continue
            zz = [None if v is None else (v - med) / mad for v in ll]

            def change_point(thr):
                for k in range(SEARCH[0], SEARCH[1]):
                    i = k + PRE_S
                    w = zz[i:i + RUN]
                    if len(w) == RUN and all(v is not None and v > thr for v in w):
                        return k
                return None

            def med_of(xs):
                xs = [v for v in xs if v is not None]
                return statistics.median(xs) if xs else None
            rec["change_s_k3"] = change_point(3.0)
            pre_base = [v for v in zz[:PRE_S + BASE[1]] if v is not None]
            rec.update(z=[None if v is None else round(v, 2) for v in zz],
                       change_s=change_point(K_MAD), pre10_median_z=med_of(zz[PRE_S - 10: PRE_S]),
                       post10_median_z=med_of(zz[PRE_S: PRE_S + 10]),
                       base_raised=bool(pre_base and max(pre_base) > 3 * K_MAD))
            out["events"].append(rec)
        except Exception as e:
            out["skipped"]["error"] += 1
            out.setdefault("errors", []).append(f"{rel}: {e!r}")
    out["skipped"] = dict(out["skipped"])
    write_json(os.path.relpath(cp, OUT), out)
    return f"{len(out['events'])} events"


# ================================================================== unit: detection probe (post-ictal check)

DETPROBE_UNLAB, DETPROBE_LAB = 80, 10


def _ll_profile(sig, sig_t0, t_rel0, n_s):
    """1-s line length for n_s seconds starting t_rel0 seconds after the EDF start."""
    out = []
    for k in range(n_s):
        i0 = int(round((t_rel0 + k - sig_t0) * 1000))
        seg = sig[i0:i0 + 1000]
        out.append(sum(abs(seg[j + 1] - seg[j]) for j in range(999)) / 999 if len(seg) == 1000 and i0 >= 0 else None)
    return out


def unit_detprobe(animal):
    """EEG around (a) annotated seizures and (b) unannotated line-length detections: is the
    event followed by post-ictal suppression, as seizures are and most artifacts are not?"""
    cp = os.path.join(CACHE, f"detprobe_{animal}.json")
    if os.path.exists(cp):
        return "cached"
    ep = json.load(ropen(os.path.join(CACHE, f"epochs_{animal}.json"), "r"))
    rng = random.Random(f"detprobe-{animal}")
    dets = [d for d in ep["detections_listed"] if not d["near_annotation"]]
    cap = DETPROBE_UNLAB if animal in UNLABELLED_ANIMALS else DETPROBE_LAB
    wins = [("detection", parse_dt(d["start"]), parse_dt(d["end"]), d.get("max_z")) for d in rng.sample(dets, min(cap, len(dets)))]
    if animal in LABELLED_ANIMALS:
        evs = [r for r in load_csv(os.path.join(AUDIT, "events.csv")) if r["animal"] == animal and r["label"] in SEIZURE_LABELS]
        for r in rng.sample(evs, min(DETPROBE_LAB, len(evs))):
            wins.append((f"annotated {r['label']}", parse_dt(r["start"]), parse_dt(r["end"]), None))
    edfs = [(parse_dt(r["edf_start"]), parse_dt(r["edf_end"]), r["relpath"])
            for r in load_csv(os.path.join(AUDIT, "files.csv"))
            if r["kind"] == "edf" and r["animal"] == animal and r.get("edf_start") and r.get("edf_end")]
    out = {"animal": animal, "n_detections_available": len(dets), "windows": [], "skipped": Counter()}
    hdr_cache = {}
    for kind, s, e, mz in wins:
        dur = (e - s).total_seconds()
        cand = [x for x in edfs if x[0] <= s - dt.timedelta(seconds=61) and e + dt.timedelta(seconds=61) <= x[1]]
        if not cand:
            out["skipped"]["no_edf_window"] += 1
            continue
        rel = cand[0][2]
        p = os.path.join(RAW, rel)
        try:
            hd = hdr_cache.get(rel) or edf_header(p)
            hdr_cache[rel] = hd
            ch = pick_channel(hd)
            if ch is None:
                out["skipped"]["no_1khz_channel"] += 1
                continue
            t0 = (s - hd["start"]).total_seconds()
            sig, sig_t0 = read_window(p, hd, ch, t0 - 61, t0 + dur + 61)
            if sig is None:
                out["skipped"]["window_outside_records"] += 1
                continue
            n = int(dur) + 120
            ll = _ll_profile(sig, sig_t0, t0 - 60, n)
            base = [v for v in ll[0:40] if v is not None]
            med = statistics.median(base) if base else 0
            mad = statistics.median([abs(v - med) for v in base]) * 1.4826 if base else 0
            rec = {"kind": kind, "start": s.isoformat(sep=" "), "dur_s": dur, "max_z_epoch": mz, "edf": rel,
                   "base_med": med, "base_mad": mad}
            if not (med > 0 and mad > 0):
                rec["degenerate_baseline"] = True
                out["windows"].append(rec)
                continue
            z = [None if v is None else (v - med) / mad for v in ll]

            def mz_(a, b):
                xs = [v for v in z[60 + a: 60 + b] if v is not None]
                return statistics.median(xs) if xs else None
            ie = int(dur)
            rec.update(ictal_median_z=mz_(0, ie), post_5_35_median_z=mz_(ie + 5, ie + 35),
                       post_35_60_median_z=mz_(ie + 35, ie + 60), pre_20_0_median_z=mz_(-20, 0),
                       post_ratio=(statistics.median([v for v in ll[60 + ie + 5: 60 + ie + 35] if v is not None]) / med
                                   if any(v is not None for v in ll[60 + ie + 5: 60 + ie + 35]) else None))
            out["windows"].append(rec)
        except Exception as ex:
            out["skipped"]["error"] += 1
            out.setdefault("errors", []).append(f"{rel}: {ex!r}")
    out["skipped"] = dict(out["skipped"])
    write_json(os.path.relpath(cp, OUT), out)
    return f"{len(out['windows'])} windows"


# ================================================================== unit: clip probe

def ffprobe_json(args, timeout=120):
    r = subprocess.run([FFPROBE, "-v", "error", "-of", "json"] + args, capture_output=True, text=True,
                       timeout=timeout)
    return json.loads(r.stdout or "{}"), r.stderr.strip()


def unit_clipprobe():
    cp = os.path.join(CACHE, "clipprobe.json")
    if os.path.exists(cp):
        return "cached"
    out = {"clips": []}
    for a in LABELLED_ANIMALS:
        root = os.path.join(LABFULL, f"Data_{a}_cropped")
        dirs = sorted(glob.glob(os.path.join(root, "*", "seizure_*")))
        rng = random.Random(f"clip-{a}")
        for d in rng.sample(dirs, min(CLIPPROBE_PER_ANIMAL, len(dirs))):
            v, inf = os.path.join(d, "video.mp4"), os.path.join(d, "info.txt")
            rec = {"animal": a, "dir": os.path.relpath(d, LABFULL)}
            try:
                info = parse_info(inf) if os.path.exists(inf) else {}
                cs, ce = parse_dt(info.get("Clip start")), parse_dt(info.get("Clip end"))
                ss = parse_dt(info.get("Seizure start"))
                rec["intended_s"] = (ce - cs).total_seconds() if cs and ce else None
                rec["onset_minus_clipstart_s"] = (ss - cs).total_seconds() if cs and ss else None
                j, err = ffprobe_json(["-show_entries", "format=duration:stream=nb_frames,r_frame_rate,start_time,width,height",
                                       "-select_streams", "v:0", v])
                rec["video_s"] = float(j.get("format", {}).get("duration", "nan"))
                st = (j.get("streams") or [{}])[0]
                rec.update(nb_frames=st.get("nb_frames"), wh=f"{st.get('width')}x{st.get('height')}",
                           start_time=st.get("start_time"), err=err[:200])
                if rec["intended_s"] is not None:
                    rec["excess_s"] = rec["video_s"] - rec["intended_s"]
            except Exception as e:
                rec["error"] = repr(e)
            out["clips"].append(rec)
    write_json(os.path.relpath(cp, OUT), out)
    return f"{len(out['clips'])} clips"


def unit_gop():
    """Keyframe spacing of the source recordings, from packet flags over the first 120 s (no decoding)."""
    cp = os.path.join(CACHE, "gop.json")
    if os.path.exists(cp):
        return "cached"
    files = [r for r in load_csv(os.path.join(AUDIT, "files.csv")) if r["kind"] == "video" and r["status"] == "ok"]
    by_room = defaultdict(list)
    for r in files:
        by_room["1402C" if "1402C" in (r.get("xml_path_prefix") or "") else "1402D"].append(r)
    out = []
    for room, rs in sorted(by_room.items()):
        for r in random.Random(room).sample(rs, 2):
            j, err = ffprobe_json(["-select_streams", "v:0", "-read_intervals", "%+120", "-show_entries",
                                   "packet=pts_time,flags", os.path.join(RAW, r["relpath"])], timeout=300)
            key = [float(x["pts_time"]) for x in j.get("packets", []) if "K" in x.get("flags", "") and "pts_time" in x]
            gaps = [b - a for a, b in zip(key, key[1:])]
            out.append({"room": room, "relpath": r["relpath"], "n_key": len(key), "gap_median_s": q(gaps, .5),
                        "gap_min_s": min(gaps) if gaps else None, "gap_max_s": max(gaps) if gaps else None,
                        "first": key[:4], "err": err[:200]})
    write_json(os.path.relpath(cp, OUT), out)
    return f"{len(out)} files"


# ================================================================== report

def match_dirs_to_events(dirs, evs):
    """One-to-one greedy match by |dt| (<= MATCH_TOL_S), label agreement as tie-break.

    dirs: list of dicts with t_ref (datetime), t_prec ('ms' | 's'), label.
    evs : list of dicts with t (datetime), label.  Returns {dir_i: (ev_j, dt_s)}."""
    ev_t = sorted((tsec(e["t"]), j) for j, e in enumerate(evs))
    ts = [x for x, _ in ev_t]
    pairs = []
    for i, d in enumerate(dirs):
        if d.get("t_ref") is None:
            continue
        td = tsec(d["t_ref"])
        lo, hi = bisect_left(ts, td - MATCH_TOL_S - 1), bisect_right(ts, td + MATCH_TOL_S + 1)
        for _, j in ev_t[lo:hi]:
            te = evs[j]["t"]
            if d["t_prec"] == "s":
                te = te.replace(microsecond=0)
            dts = (te - d["t_ref"]).total_seconds()
            if abs(dts) <= MATCH_TOL_S:
                pairs.append((abs(dts), evs[j]["label"] != d["label"], i, j, dts))
    pairs.sort()
    used_d, used_e, m = set(), set(), {}
    for _, _, i, j, dts in pairs:
        if i in used_d or j in used_e:
            continue
        used_d.add(i)
        used_e.add(j)
        m[i] = (j, dts)
    return m


def report():
    t_start = time.time()
    events = load_csv(os.path.join(AUDIT, "events.csv"))
    files = load_csv(os.path.join(AUDIT, "files.csv"))
    per_animal_audit = json.load(ropen(os.path.join(AUDIT, "per_animal.json"), "r"))
    for e in events:
        e["t"] = parse_dt(e["start"])
        e["t_end"] = parse_dt(e["end"]) or e["t"] + dt.timedelta(seconds=float(e["duration_s"] or 0))
        e["dur"] = float(e["duration_s"] or 0)
    ev_by_a = defaultdict(list)
    for e in events:
        ev_by_a[e["animal"]].append(e)
    for a in ev_by_a:
        ev_by_a[a].sort(key=lambda e: e["t"])

    # ---- videos and EDFs per animal (for coverage before onset)
    vids = defaultdict(list)
    for r in files:
        if r["kind"] == "video" and r.get("start_local") and r.get("end_local"):
            vids[r["animal"]].append((parse_dt(r["start_local"]), parse_dt(r["end_local"]), r["relpath"],
                                      "frames_missing_vs_wallclock" in (r.get("problems") or "")))
    for a in vids:
        vids[a].sort()
    edfs = defaultdict(list)
    for r in files:
        if r["kind"] == "edf" and r.get("edf_start") and r.get("edf_end"):
            edfs[r["animal"]].append((parse_dt(r["edf_start"]), parse_dt(r["edf_end"]), r["relpath"],
                                      r.get("edf_1khz_label") or r.get("edf_ecg_label")))
    for a in edfs:
        edfs[a].sort()

    def video_run_start(a, t):
        """Start of the continuous video run containing t (file rollovers <= 120 s are bridged)."""
        vs = vids.get(a, [])
        idx = [i for i, v in enumerate(vs) if v[0] <= t < v[1]]
        if not idx:
            return None, None
        i = idx[0]
        file_start = vs[i][0]
        run = vs[i][0]
        while i > 0 and (vs[i][0] - vs[i - 1][1]).total_seconds() <= 120 and vs[i - 1][1] <= vs[i][0] + dt.timedelta(seconds=1):
            i -= 1
            run = vs[i][0]
        return file_start, run

    def edf_start_for(a, t):
        for s, e, rel, lab in edfs.get(a, []):
            if s <= t < e:
                return s, lab
        return None, None

    # ---- labelled tree
    lab_rows = []
    missing_units = []
    for a in LABELLED_ANIMALS:
        p = os.path.join(CACHE, f"labelled_{a}.json")
        if not os.path.exists(p):
            missing_units.append(f"labelled_{a}")
            continue
        lab_rows += json.load(ropen(p, "r"))["rows"]
    for r in lab_rows:
        r["in_data"] = bool(r.get("in_data"))
        r["in_full"] = bool(r.get("in_full"))
        m = SZ_RE.match(r["name"])
        n = NS_RE.match(r["name"])
        if m:
            r["kind"] = "seizure"
            r["idx"], r["label"] = int(m.group(1)), label_of_dirname(m.group(2))
            r["t_name"] = dt.datetime.strptime(m.group(3) + m.group(4), "%Y%m%d%H%M%S")
        elif n:
            r["kind"] = "non_seizure"
            r["ns_idx"], r["idx"], r["label"] = int(n.group(1)), int(n.group(2)), label_of_dirname(n.group(3))
            r["t_name"] = dt.datetime.strptime(n.group(4) + n.group(5), "%Y%m%d%H%M%S")
        else:
            r["kind"] = "unparsed"
            r["t_name"] = None
            r["label"] = None
        info = r.get("info") or {}
        r["info_sz_start"] = parse_dt(info.get("Seizure start") or info.get("Matched seizure start"))
        r["info_clip_start"] = parse_dt(info.get("Clip start"))
        r["info_clip_end"] = parse_dt(info.get("Clip end"))
        r["info_label"] = info.get("Label") or info.get("Matched seizure label")
        dfiles = r.get("data_files") or {}
        r["discover"] = (r["in_data"] and "video.mp4" in dfiles and
                         (r["kind"] == "non_seizure" or
                          (r["kind"] == "seizure" and r["label"].replace(" ", "_") in DISCOVER_STAGES)))
        if r["info_sz_start"] is not None:
            r["t_ref"], r["t_prec"] = r["info_sz_start"], "ms"
        else:
            r["t_ref"], r["t_prec"] = r["t_name"], "s"

    # ---- match seizure dirs and non-seizure partners to xlsx events
    for a in LABELLED_ANIMALS:
        evs = ev_by_a.get(a, [])
        for kind in ("seizure", "non_seizure"):
            ds = [r for r in lab_rows if r["animal"] == a and r["kind"] == kind]
            m = match_dirs_to_events(ds, evs)
            for i, d in enumerate(ds):
                if i in m:
                    j, dts = m[i]
                    e = evs[j]
                    d["ev_row"], d["ev_day"], d["ev_label"], d["dt_s"] = int(e["row"]), e["day_folder"], e["label"], dts
                    d["ev_key"] = (a, e["xlsx"], int(e["row"]))      # day_folder+row is NOT unique: 3 folders hold 2 xlsx
                    d["label_agree"] = e["label"] == d["label"]
                    d["idx_agree"] = d["idx"] == int(e["row"]) + 1
                    d["day_agree"] = e["day_folder"] == d["day"]
                    e.setdefault(f"{kind}_dirs", []).append(f"{d['day']}/{d['name']}")
                    e[f"{kind}_dt_s"] = dts
                    if kind == "seizure":
                        e["seizure_in_data"] = e.get("seizure_in_data", False) or d["in_data"]
                        e["seizure_discover"] = e.get("seizure_discover", False) or d["discover"]
                        e["seizure_in_full"] = e.get("seizure_in_full", False) or d["in_full"]
                else:
                    # nearest event for reporting
                    if d.get("t_ref") is not None and evs:
                        near = min(evs, key=lambda e: abs((e["t"] - d["t_ref"]).total_seconds()))
                        d["nearest_dt_s"] = (near["t"] - d["t_ref"]).total_seconds()
                        d["nearest_label"] = near["label"]

    # ---- non-seizure clips that overlap annotated events (any label)
    for a in LABELLED_ANIMALS:
        evs = ev_by_a.get(a, [])
        es = [tsec(e["t"]) for e in evs]
        for d in (r for r in lab_rows if r["animal"] == a and r["kind"] == "non_seizure"):
            cs, ce = d.get("info_clip_start"), d.get("info_clip_end")
            if cs is None or ce is None:
                continue
            s, e_ = tsec(cs), tsec(ce)
            hits, hits30 = [], []
            for j in range(max(0, bisect_left(es, s - 7200)), bisect_right(es, e_ + 30)):
                ev = evs[j]
                a0, a1 = tsec(ev["t"]), tsec(ev["t_end"])
                if a0 < e_ and a1 > s:
                    hits.append(f"{ev['label']}@{ev['start']}")
                if a0 - 30 < e_ and a1 + 30 > s:
                    hits30.append(f"{ev['label']}@{ev['start']}")
            d["overlaps_events"] = hits
            d["within30_events"] = hits30

    # ---- per-event outcome and reasons
    lab_days = defaultdict(set)
    for r in lab_rows:
        lab_days[r["animal"]].add(r["day"])
    prev_end = {}
    for a, evs in ev_by_a.items():
        last = None
        for e in evs:
            e["gap_prev_event_s"] = (e["t"] - last).total_seconds() if last else None
            last = max(last, e["t_end"]) if last else e["t_end"]
    used_xlsx = {e["xlsx"] for e in events if e.get("seizure_dirs") or e.get("non_seizure_dirs")}
    xlsx_per_folder = defaultdict(set)
    for r in files:
        if r["kind"] == "xlsx":
            xlsx_per_folder[(r["animal_folder"], r["day_folder"])].add(r["relpath"])
    for e in events:
        a = e["animal"]
        e["clipped"] = bool(e.get("seizure_dirs"))
        vday = e["video"].split("/")[1] if e.get("video") else None
        e["video_day_folder"] = vday
        if a in UNLABELLED_ANIMALS:
            reason = "animal_not_in_labelled_set"
        elif e["clipped"]:
            reason = "clipped" if e.get("seizure_discover") else ("clipped_not_in_discover" if e["label"] in SEIZURE_LABELS else "clipped_label_excluded_by_discover")
        elif e["label"] not in SEIZURE_LABELS:
            reason = f"label_{e['label'].replace(' ', '_')}"
        elif len(xlsx_per_folder[(e["animal_folder"], e["day_folder"])]) > 1 and e["xlsx"] not in used_xlsx:
            reason = "second_xlsx_in_folder_never_cut"
        elif not e.get("video"):
            reason = "no_video_no_video_folder" if "no video" in e["day_folder"] else "no_video_outside_any_video"
        elif vday != e["day_folder"]:
            reason = "video_in_other_day_folder"
        elif e.get("video_wallclock_defect"):
            reason = "video_timing_defect_file"
        elif e["day_folder"] not in lab_days.get(a, set()):
            reason = "day_folder_absent_from_labelled_tree"
        elif e.get("video_covers_end") == "False":
            reason = "video_ends_before_event_end"
        else:
            reason = "video_ok_not_clipped"
        e["outcome"] = reason
        # onset / pre-onset coverage (Q4)
        fs, run = video_run_start(a, e["t"])
        es, elab = edf_start_for(a, e["t"])
        e["pre_video_file_s"] = (e["t"] - fs).total_seconds() if fs else None
        e["pre_video_run_s"] = (e["t"] - run).total_seconds() if run else None
        e["pre_edf_s"] = (e["t"] - es).total_seconds() if es else None
        e["edf_1khz_label"] = elab

    # ---- partner coincidences
    partners = set()
    for folder, u in per_animal_audit.items():
        for p in u.get("partners", []) or []:
            a, b = sorted((animal_of(folder), animal_of(p)))
            if a != b and a in ALL_ANIMALS and b in ALL_ANIMALS:
                partners.add((a, b))
    partner_rows = []
    for a, b in sorted(partners):
        ta = [tsec(e["t"]) for e in ev_by_a.get(a, [])]
        tb = sorted(tsec(e["t"]) for e in ev_by_a.get(b, []))

        def coinc(shift):
            n = 0
            for x in ta:
                i = bisect_left(tb, x + shift - PARTNER_TOL_S)
                if i < len(tb) and tb[i] <= x + shift + PARTNER_TOL_S:
                    n += 1
            return n
        obs = coinc(0.0)
        shifts = [s * 600.0 for s in range(-36, 37) if abs(s) >= 3]      # +-30 min .. +-6 h
        null = [coinc(s) for s in shifts]
        exact = 0
        for x in ta:
            i = bisect_left(tb, x - 1)
            if i < len(tb) and tb[i] <= x + 1:
                exact += 1
        partner_rows.append({"animal_a": a, "animal_b": b, "events_a": len(ta), "events_b": len(tb),
                             "a_within60s_of_b": obs, "a_within1s_of_b": exact,
                             "null_mean_shifted": statistics.mean(null) if null else None,
                             "null_max_shifted": max(null) if null else None})

    # ---- xlsx export times
    xz = json.load(ropen(os.path.join(CACHE, "xlsx_zip.json"), "r")) if os.path.exists(os.path.join(CACHE, "xlsx_zip.json")) else None
    if xz is None:
        missing_units.append("xlsxzip")
    xlsx_rows = []
    if xz:
        for r in xz:
            r["zip_date"] = (r.get("zip_time") or "")[:10]
            xlsx_rows.append(r)
    export_batches = defaultdict(Counter)
    for r in xlsx_rows:
        export_batches[r["zip_date"]][animal_of(r["animal_folder"])] += 1

    # ---- epochs
    ep = {}
    for a in ALL_ANIMALS:
        p = os.path.join(CACHE, f"epochs_{a}.json")
        if os.path.exists(p):
            ep[a] = json.load(ropen(p, "r"))
        else:
            missing_units.append(f"epochs_{a}")
    # ---- onset
    on = {}
    for a in ALL_ANIMALS:
        p = os.path.join(CACHE, f"onset_{a}.json")
        if os.path.exists(p):
            on[a] = json.load(ropen(p, "r"))
        else:
            missing_units.append(f"onset_{a}")
    cp = json.load(ropen(os.path.join(CACHE, "clipprobe.json"), "r")) if os.path.exists(os.path.join(CACHE, "clipprobe.json")) else None
    if cp is None:
        missing_units.append("clipprobe")

    # ================================================================ outputs
    # 1. events by animal and day
    day_rows = []
    by_day = defaultdict(list)
    for e in events:
        by_day[(e["animal_folder"], e["day_folder"])].append(e)
    xl_by_day = {(r["animal_folder"], r["day_folder"]): r for r in xlsx_rows}
    all_days = set(by_day) | set(xl_by_day)
    labels_all = ["Stage 1", "Stage 2", "Stage 3", "Stage 4", "Stage 5", "Spikes"]
    for k in sorted(all_days):
        es = by_day.get(k, [])
        c = Counter(e["label"] for e in es)
        row = {"animal_folder": k[0], "animal": animal_of(k[0]), "day_folder": k[1], "n_events": len(es),
               "clipped": sum(e["clipped"] for e in es),
               "stage2_5_not_clipped": sum(1 for e in es if e["label"] in SEIZURE_LABELS and not e["clipped"]),
               "xlsx_export_time": (xl_by_day.get(k) or {}).get("zip_time")}
        for l in labels_all:
            row[l.replace(" ", "_")] = c.get(l, 0)
        day_rows.append(row)
    write_csv("events_by_animal_day.csv",
              ["animal", "animal_folder", "day_folder", "n_events"] + [l.replace(" ", "_") for l in labels_all] +
              ["clipped", "stage2_5_not_clipped", "xlsx_export_time"], day_rows)

    # 2. every event with its outcome
    ev_cols = ["animal", "day_folder", "xlsx", "row", "label", "start", "end", "duration_s", "spikes",
               "outcome", "clipped", "seizure_dirs", "seizure_dt_s", "seizure_in_data", "seizure_discover", "seizure_in_full",
               "non_seizure_dirs", "video", "video_day_folder", "video_seek_s", "video_covers_end", "video_wallclock_defect",
               "in_same_day_edf", "in_any_edf", "edf_1khz_label", "pre_video_file_s", "pre_video_run_s", "pre_edf_s",
               "gap_prev_event_s"]
    write_csv("events_annotated.csv", ev_cols, events)

    # 3. labelled dirs
    d_cols = ["animal", "day", "name", "kind", "idx", "label", "t_name", "info_sz_start", "info_label",
              "info_clip_start", "info_clip_end", "in_data", "in_full", "discover", "data_files", "full_files",
              "ev_day", "ev_row", "ev_label", "dt_s", "label_agree", "idx_agree", "day_agree", "nearest_dt_s",
              "nearest_label", "overlaps_events", "within30_events"]
    write_csv("labelled_dirs.csv", d_cols, lab_rows)

    # 4. candidates for new labelled data
    cands = [e for e in events if e["outcome"] in ("video_ok_not_clipped", "video_in_other_day_folder",
                                                   "second_xlsx_in_folder_never_cut",
                                                   "day_folder_absent_from_labelled_tree", "video_ends_before_event_end",
                                                   "video_timing_defect_file")
             or (e["animal"] in UNLABELLED_ANIMALS and e.get("video"))]
    write_csv("candidates_unclipped_events.csv", ev_cols, cands)

    # 5. partner coincidences
    write_csv("partner_coincidence.csv", list(partner_rows[0].keys()) if partner_rows else ["none"], partner_rows)

    # 6. xlsx export times
    write_csv("xlsx_export_times.csv", ["animal_folder", "animal", "day_folder", "relpath", "n_events", "zip_time",
                                        "zip_times_distinct", "has_docprops", "dimension", "error"], xlsx_rows)

    # 7. detector per animal
    det_rows, det_list = [], []
    for a in ALL_ANIMALS:
        r = ep.get(a)
        if not r:
            continue
        row = {"animal": a, "labelled": a in LABELLED_ANIMALS, "export_files": r["n_files"],
               "hours": r["hours"], "valid_hours": r["valid_hours"], "xlsx_events": r["n_events_xlsx"],
               "xlsx_seizure_events": r["n_seizure_events_xlsx"],
               "annotated_epochs_in_export": r["export_annotated_epochs"],
               "annotated_epochs_touching_xlsx_event": r["export_annotated_epochs_touching_xlsx"]}
        for k, pt in r["per_threshold"].items():
            vh = max(r["valid_hours"], 1e-9)
            row[f"z{k}_threshold"] = r["thresholds"][k]
            row[f"z{k}_det"] = pt["n_det"]
            row[f"z{k}_det_ge20_per_day"] = 24 * pt["n_det_ge20"] / vh
            row[f"z{k}_unannot_ge20"] = pt["n_unannot_ge20"]
            row[f"z{k}_unannot_ge20_per_day"] = 24 * pt["n_unannot_ge20"] / vh
            row[f"z{k}_unannot_per_day"] = 24 * pt["n_unannot"] / vh
            row[f"z{k}_sensitivity"] = (pt["seizure_events_hit"] / pt["seizure_events_covered"]
                                        if pt["seizure_events_covered"] else None)
            row[f"z{k}_seizure_events_covered"] = pt["seizure_events_covered"]
        row["annotated_seizures_per_day"] = 24 * r["n_seizure_events_xlsx"] / max(r["valid_hours"], 1e-9)
        det_rows.append(row)
        if a in UNLABELLED_ANIMALS:
            for d in r["detections_listed"]:
                d = dict(d, animal=a)
                t = parse_dt(d["start"])
                vs = [v for v in vids.get(a, []) if v[0] <= t < v[1]]
                d["video"] = vs[0][2] if vs else None
                d["video_seek_s"] = (t - vs[0][0]).total_seconds() if vs else None
                det_list.append(d)
    if det_rows:
        write_csv("detector_per_animal.csv", list(det_rows[0].keys()), det_rows)
    write_csv("unlabelled_animals_eeg_detections.csv",
              ["animal", "day_folder", "start", "end", "dur_s", "max_z", "near_annotation", "video", "video_seek_s", "file"],
              sorted(det_list, key=lambda d: (d["animal"], d["start"])))

    # 8. onset
    onset_rows, prof = [], defaultdict(list)
    for a, r in on.items():
        for e in r["events"]:
            e2 = {k: v for k, v in e.items() if k != "z"}
            e2["animal"] = a
            onset_rows.append(e2)
            if e.get("z") and a in LABELLED_ANIMALS:
                prof[e["label"]].append(e["z"])
                prof["all"].append(e["z"])
    write_csv("onset_edf_sample.csv", ["animal", "day_folder", "row", "label", "start", "duration_s", "edf", "channel",
                                       "base_med", "base_mad", "flat_frac", "degenerate_baseline", "change_s", "change_s_k3",
                                       "pre10_median_z", "post10_median_z", "base_raised"], onset_rows)
    prof_out = {}
    for lab, zs in prof.items():
        cols = list(zip(*zs))
        prof_out[lab] = {"n": len(zs), "seconds": list(range(-PRE_S, POST_S)),
                         "median_z": [q([v for v in c if v is not None], .5) for c in cols],
                         "p25_z": [q([v for v in c if v is not None], .25) for c in cols],
                         "p75_z": [q([v for v in c if v is not None], .75) for c in cols]}
    write_json("onset_profiles.json", prof_out)

    # ================================================================ per-animal table
    table = []
    for a in ALL_ANIMALS:
        es = ev_by_a.get(a, [])
        lab = [r for r in lab_rows if r["animal"] == a]
        sz = [r for r in lab if r["kind"] == "seizure"]
        ns = [r for r in lab if r["kind"] == "non_seizure"]
        oc = Counter(e["outcome"] for e in es)
        c = Counter(e["label"] for e in es)
        row = {
            "animal": a, "in_labelled_set": a in LABELLED_ANIMALS,
            "xlsx_files": sum(1 for r in xlsx_rows if animal_of(r["animal_folder"]) == a),
            "xlsx_nonempty": sum(1 for r in xlsx_rows if animal_of(r["animal_folder"]) == a and int(r.get("n_events") or 0) > 0),
            "events": len(es), "S1": c.get("Stage 1", 0), "S2": c.get("Stage 2", 0), "S3": c.get("Stage 3", 0),
            "S4": c.get("Stage 4", 0), "S5": c.get("Stage 5", 0), "Spikes": c.get("Spikes", 0),
            "sz_dirs_data": sum(r["in_data"] for r in sz), "sz_dirs_full": sum(r["in_full"] for r in sz),
            "sz_dirs_discover": sum(r["discover"] for r in sz),
            "sz_dirs_matched60": sum(1 for r in sz if r.get("ev_row") is not None),
            "sz_dirs_matched1": sum(1 for r in sz if r.get("ev_row") is not None and abs(r["dt_s"]) <= 1),
            "sz_dirs_unmatched": sum(1 for r in sz if r.get("ev_row") is None),
            "sz_dirs_label_disagree": sum(1 for r in sz if r.get("ev_row") is not None and not r["label_agree"]),
            "ns_dirs_discover": sum(r["discover"] for r in ns),
            "ns_partner_unmatched": sum(1 for r in ns if r.get("ev_row") is None),
            "ns_overlap_event": sum(1 for r in ns if r.get("overlaps_events")),
            "ns_within30_event": sum(1 for r in ns if r.get("within30_events")),
            "events_clipped": sum(e["clipped"] for e in es),
            "S2_5_unclipped": sum(1 for e in es if e["label"] in SEIZURE_LABELS and not e["clipped"]),
        }
        for k in ("video_ok_not_clipped", "second_xlsx_in_folder_never_cut", "video_in_other_day_folder", "video_timing_defect_file",
                  "day_folder_absent_from_labelled_tree", "video_ends_before_event_end",
                  "no_video_no_video_folder", "no_video_outside_any_video", "clipped_not_in_discover"):
            row[k] = oc.get(k, 0)
        dr = next((d for d in det_rows if d["animal"] == a), None)
        if dr:
            row["eeg_valid_h"] = dr["valid_hours"]
            row["annot_sz_per_day"] = dr["annotated_seizures_per_day"]
            row["det_z0.5_unannot_ge20_per_day"] = dr["z0.5_unannot_ge20_per_day"]
            row["det_z0.5_sensitivity"] = dr["z0.5_sensitivity"]
        table.append(row)
    write_csv("per_animal_annotations.csv", list(table[0].keys()), table)

    # ================================================================ summary text
    L = []
    P = L.append
    P(f"ANNOTATION INVENTORY  {time.strftime('%Y-%m-%d %H:%M:%S')}   (read-only; outputs in {OUT})")
    P(f"inputs: {AUDIT}/events.csv ({len(events)} xlsx event rows), files.csv, per_animal.json; labelled trees {LAB}, {LABFULL};")
    P(f"        epoch export {EXPORT}; RN243 calibration {RN243_AUDIT}")
    P(f"missing units: {missing_units or 'none'}")
    P("")
    P("1. ANNOTATION SOURCES AND VOCABULARY")
    P(f"  xlsx files: {len(xlsx_rows)}; with >=1 event: {sum(1 for r in xlsx_rows if int(r.get('n_events') or 0) > 0)}; "
      f"docProps present: {sum(1 for r in xlsx_rows if r.get('has_docprops'))}; read errors: {sum(1 for r in xlsx_rows if r.get('error'))}")
    P(f"  label vocabulary (events.csv): {dict(Counter(e['label'] for e in events).most_common())}")
    P(f"  duration [s] by label:")
    for l in labels_all:
        P(f"    {l:8s} {dist([e['dur'] for e in events if e['label'] == l], 1)}")
    multi = defaultdict(set)
    for e in events:
        multi[(e["animal_folder"], e["day_folder"])].add(e["xlsx"])
    for r in xlsx_rows:
        multi[(r["animal_folder"], r["day_folder"])].add(r["relpath"])
    mm = {k: sorted(os.path.basename(x) for x in v) for k, v in multi.items() if len(v) > 1}
    P(f"  day folders holding more than one xlsx: {len(mm)}  (cut_seizure_clips.py uses only xlsx_files[0] of a folder)")
    for k, v in sorted(mm.items()):
        n = Counter(os.path.basename(e["xlsx"]) for e in events if (e["animal_folder"], e["day_folder"]) == k)
        P(f"    {k[0]}/{k[1]}: " + ", ".join(f"{x} ({n.get(x, 0)} events)" for x in v))
    P(f"  'EEG (ave)' column non-empty: {sum(1 for e in events if e.get('eeg_ave'))};  spike-count column: {dist([float(e['spikes']) for e in events if e.get('spikes') not in (None, '')], 0)}")
    # which recording days carry annotations: EEG-covered vs video-only days
    edf_h, vid_h = defaultdict(float), defaultdict(float)
    for r in files:
        k = (r["animal_folder"], r["day_folder"])
        if r["kind"] == "edf" and r.get("edf_hours"):
            edf_h[k] += float(r["edf_hours"])
        if r["kind"] == "video" and r.get("hours"):
            vid_h[k] += float(r["hours"])
    n_ev_day = Counter((e["animal_folder"], e["day_folder"]) for e in events)
    xt = Counter()
    for k in set(edf_h) | set(vid_h):
        cat = ("EEG>=1h" if edf_h[k] >= 1 else "EEG<1h") + ("+video>=1h" if vid_h[k] >= 1 else "+video<1h")
        xt[(cat, animal_of(k[0]) in LABELLED_ANIMALS, n_ev_day[k] > 0)] += 1
    P("  animal-day folders by coverage (labelled animals only): with >=1 event / with none")
    for cat in ("EEG>=1h+video>=1h", "EEG<1h+video>=1h", "EEG>=1h+video<1h", "EEG<1h+video<1h"):
        P(f"    {cat:20s} {xt[(cat, True, True)]:4d} / {xt[(cat, True, False)]:4d}")
    P(f"  xlsx export (zip member) dates -> animals (files):")
    for d in sorted(export_batches):
        P(f"    {d or '?'}: {dict(sorted(export_batches[d].items()))}")
    P("")
    P("2. LABELLED CLIPS vs ANNOTATIONS (seizure dirs matched one-to-one to xlsx events of the same animal, |dt| <= 60 s)")
    szall = [r for r in lab_rows if r["kind"] == "seizure"]
    nsall = [r for r in lab_rows if r["kind"] == "non_seizure"]
    P(f"  labelled dirs: seizure {len(szall)} (in data/ {sum(r['in_data'] for r in szall)}, in data_full/ {sum(r['in_full'] for r in szall)}), "
      f"non-seizure {len(nsall)} (data/ {sum(r['in_data'] for r in nsall)}, data_full/ {sum(r['in_full'] for r in nsall)}), unparsed {sum(1 for r in lab_rows if r['kind'] == 'unparsed')}")
    P(f"  discover()-equivalent items: {sum(r['discover'] for r in lab_rows)} (seizure {sum(r['discover'] for r in szall)}, non-seizure {sum(r['discover'] for r in nsall)}); expected 24,497")
    P(f"  seizure dir labels: {dict(Counter(r['label'] for r in szall).most_common())}")
    P(f"  seizure dirs only in data_full/ (not in data/, so not in discover()): {sum(1 for r in szall if r['in_full'] and not r['in_data'])}; only in data/: {sum(1 for r in szall if r['in_data'] and not r['in_full'])}")
    P(f"  seizure dirs with info.txt: {sum(1 for r in szall if r.get('info'))}; info 'Seizure start' == dir-name time (to the second): "
      f"{sum(1 for r in szall if r.get('info_sz_start') and r['info_sz_start'].replace(microsecond=0) == r['t_name'])}")
    m = [r for r in szall if r.get("ev_row") is not None]
    P(f"  matched: {len(m)} of {len(szall)};  |dt| distribution [s]: {dist([abs(r['dt_s']) for r in m], 3)}")
    P(f"  matched with |dt| <= 1 s: {sum(1 for r in m if abs(r['dt_s']) <= 1)};  label agrees: {sum(r['label_agree'] for r in m)};  "
      f"dir index == xlsx row+1: {sum(r['idx_agree'] for r in m)};  same day folder: {sum(r['day_agree'] for r in m)}")
    um = [r for r in szall if r.get("ev_row") is None]
    P(f"  UNMATCHED seizure dirs (label with no annotation within 60 s): {len(um)}")
    for r in um[:20]:
        P(f"    {r['animal']}/{r['day']}/{r['name']}  nearest event dt={r.get('nearest_dt_s')} {r.get('nearest_label')}")
    multi = [e for e in events if len(e.get("seizure_dirs") or []) > 1]
    P(f"  events matched by >1 seizure dir: {len(multi)}")
    P(f"  non-seizure dirs: paired-seizure event found {sum(1 for r in nsall if r.get('ev_row') is not None)} of {len(nsall)}; "
      f"with clip times (info.txt) {sum(1 for r in nsall if r.get('info_clip_start'))}")
    ev_index = {(e["animal"], e["xlsx"], int(e["row"])): e for e in events}
    orphan = [r for r in nsall if r.get("ev_key") and not ev_index[r["ev_key"]].get("seizure_dirs")]
    P(f"  non-seizure dirs whose paired annotated event has NO seizure dir: {len(orphan)}; paired labels "
      f"{dict(Counter(r['label'] for r in orphan))}; paired-event outcomes {dict(Counter(ev_index[r['ev_key']]['outcome'] for r in orphan))}")
    for r in [r for r in nsall if r.get("overlaps_events")][:10]:
        P(f"    overlap: {r['animal']}/{r['day']}/{r['name']} clip {r['info_clip_start']}..{r['info_clip_end']} "
          f"discover={r['discover']} overlaps {r['overlaps_events']}")
    P(f"  non-seizure clips OVERLAPPING an annotated event: {sum(1 for r in nsall if r.get('overlaps_events'))}; "
      f"within 30 s of one: {sum(1 for r in nsall if r.get('within30_events'))}; of these in discover(): "
      f"{sum(1 for r in nsall if r.get('overlaps_events') and r['discover'])} / {sum(1 for r in nsall if r.get('within30_events') and r['discover'])}")
    P("")
    P("  event outcomes (labelled animals):")
    oc = Counter((e["label"] if e["label"] in SEIZURE_LABELS else "other", e["outcome"]) for e in events if e["animal"] in LABELLED_ANIMALS)
    for (l, o), n in sorted(oc.items()):
        P(f"    {l:8s} {o:40s} {n}")
    cands_s = [e for e in cands if e["label"] in SEIZURE_LABELS and e["animal"] in LABELLED_ANIMALS]
    P(f"  CANDIDATE new seizure clips (Stage 2-5, labelled animals, has video): {len(cands_s)}; "
      f"by stage {dict(Counter(e['label'] for e in cands_s))}; with same-day EDF {sum(1 for e in cands_s if e['in_same_day_edf'] == 'True')}")
    P(f"    by reason: {dict(Counter(e['outcome'] for e in cands_s))}")
    P(f"    by animal: {dict(Counter(e['animal'] for e in cands_s))}")
    # are the unclipped video-ok events the TAIL of their xlsx (an interrupted cutting run) or interleaved?
    last_clipped = defaultdict(int)
    for e in events:
        if e["clipped"]:
            last_clipped[e["xlsx"]] = max(last_clipped[e["xlsx"]], int(e["row"]))
    vo = [e for e in events if e["outcome"] == "video_ok_not_clipped"]
    tail = [e for e in vo if e["xlsx"] in last_clipped and int(e["row"]) > last_clipped[e["xlsx"]]]
    none = [e for e in vo if e["xlsx"] not in last_clipped]
    P(f"    'video_ok_not_clipped': {len(vo)} = {len(tail)} after the last clipped row of their xlsx (cutting stopped part-way) "
      f"+ {len(none)} in xlsx files with no clipped event at all {sorted({e['xlsx'] for e in none})} "
      f"+ {len(vo) - len(tail) - len(none)} interleaved")
    P(f"    by animal/day: {dict(Counter(e['animal'] + '/' + e['day_folder'] for e in vo))}")
    P("")
    P("3. UNLABELLED ANIMALS (RN201, RN203, RN243)")
    for a in UNLABELLED_ANIMALS:
        es = ev_by_a.get(a, [])
        P(f"  {a}: xlsx files {sum(1 for r in xlsx_rows if animal_of(r['animal_folder']) == a)}, events {len(es)} "
          f"{dict(Counter(e['label'] for e in es))}")
        for e in es:
            P(f"    {e['day_folder']:16s} {e['label']:8s} {e['start']}  {e['dur']:.1f} s  spikes={e['spikes']}  video={'yes' if e.get('video') else 'no'}")
    P("  camera-partner coincidences (events of A within 60 s of an event of B; null = same count with B shifted by 30 min..6 h):")
    for r in partner_rows:
        P(f"    {r['animal_a']}({r['events_a']}) ~ {r['animal_b']}({r['events_b']}): {r['a_within60s_of_b']} within 60 s, "
          f"{r['a_within1s_of_b']} within 1 s; shifted-null mean {r['null_mean_shifted']:.1f} max {r['null_max_shifted']}")
    for a in UNLABELLED_ANIMALS:
        for e in ev_by_a.get(a, []):
            for pr in sorted(partners):
                if a in pr:
                    b = pr[0] if pr[1] == a else pr[1]
                    near = [x for x in ev_by_a.get(b, []) if abs((x["t"] - e["t"]).total_seconds()) <= PARTNER_TOL_S]
                    for x in near:
                        P(f"    {a} {e['label']} {e['start']} coincides with {b} {x['label']} {x['start']} ({x['dur']:.1f} s)")
    P("  EEG line-length detector on the DSI epoch export (robust z of log10 line length within session, valid epochs,")
    P("  thresholds from rn243_scale_audit calibration; a detection is 'annotated' if within 60 s of any xlsx event):")
    P(f"    {'animal':7s} {'lab':3s} {'valid_h':>8s} {'ann_sz/d':>8s} {'sens@.5':>7s} {'all>=20s/d@.5':>13s} {'unann>=20s/d @.5':>16s} {'@.7':>7s} {'@.9':>7s}")
    for d in sorted(det_rows, key=lambda d: d["annotated_seizures_per_day"]):
        f = lambda v: "" if v is None else f"{v:.3f}"
        P(f"    {d['animal']:7s} {'Y' if d['labelled'] else 'n':3s} {d['valid_hours']:8.1f} {d['annotated_seizures_per_day']:8.3f} "
          f"{f(d.get('z0.5_sensitivity')):>7s} {f(d.get('z0.5_det_ge20_per_day')):>13s} {f(d.get('z0.5_unannot_ge20_per_day')):>16s} "
          f"{f(d.get('z0.7_unannot_ge20_per_day')):>7s} {f(d.get('z0.9_unannot_ge20_per_day')):>7s}")
    lab_u = [d["z0.5_unannot_ge20_per_day"] for d in det_rows if d["labelled"]]
    low_u = [d["z0.5_unannot_ge20_per_day"] for d in det_rows if d["labelled"] and d["annotated_seizures_per_day"] < 0.5]
    P(f"    labelled animals' unannotated >=20 s detections/day @.5: {dist(lab_u, 3)}")
    P(f"    ... of the labelled animals with < 0.5 annotated seizures/day: {dist(low_u, 3)}")
    # post-ictal check of detections vs annotated seizures
    dp_rows = []
    for a in ALL_ANIMALS:
        pth = os.path.join(CACHE, f"detprobe_{a}.json")
        if not os.path.exists(pth):
            missing_units.append(f"detprobe_{a}")
            continue
        for w in json.load(ropen(pth, "r"))["windows"]:
            w = dict(w, animal=a)
            w["group"] = ("annotated seizure (labelled animals)" if w["kind"].startswith("annotated")
                          else f"unannotated detection {a}" if a in UNLABELLED_ANIMALS
                          else "unannotated detection (labelled animals)")
            dp_rows.append(w)
    write_csv("detection_probe.csv", ["animal", "group", "kind", "start", "dur_s", "max_z_epoch", "edf", "base_med",
                                      "base_mad", "degenerate_baseline", "pre_20_0_median_z", "ictal_median_z",
                                      "post_5_35_median_z", "post_35_60_median_z", "post_ratio"], dp_rows)
    P("  EDF check of events (1-s line length, z vs the event's own -60..-20 s baseline): ictal and post-ictal (end+5..+35 s)")
    for g in sorted({w["group"] for w in dp_rows}):
        ws = [w for w in dp_rows if w["group"] == g and not w.get("degenerate_baseline")]
        post = [w.get("post_5_35_median_z") for w in ws if w.get("post_5_35_median_z") is not None]
        P(f"    {g:42s} n={len(ws):3d}  ictal z {dist([w.get('ictal_median_z') for w in ws], 1)}")
        P(f"    {'':42s}        post z  {dist(post, 2)};  post z < -0.5: {sum(1 for v in post if v < -0.5)}/{len(post)}"
          f";  post ratio (LL/baseline) median {q([w.get('post_ratio') for w in ws], .5)}")
    ex = [(d["animal"], d["annotated_epochs_in_export"], d["annotated_epochs_touching_xlsx_event"]) for d in det_rows]
    P(f"    export-annotated epochs that touch an xlsx event (clock check): {sum(x[2] for x in ex)} / {sum(x[1] for x in ex)}")
    P("")
    P("4. ONSET TIMES")
    szm = [e for e in events if e["animal"] in LABELLED_ANIMALS and e["label"] in SEIZURE_LABELS and e["clipped"]]
    P(f"  labelled (clipped) Stage 2-5 events: {len(szm)}; xlsx start with sub-second precision: "
      f"{sum(1 for e in szm if e['t'].microsecond)}")
    ns_ok = [r for r in szall if r.get("info_clip_start") and r.get("info_sz_start")]
    P(f"  info.txt seizure start - clip start [s]: {dist([(r['info_sz_start'] - r['info_clip_start']).total_seconds() for r in ns_ok], 3)}")
    if cp:
        P(f"  data_full clip video duration - intended (pre+dur+post) [s], sample: {dist([c.get('excess_s') for c in cp['clips']], 2)}")
    gp = os.path.join(CACHE, "gop.json")
    if os.path.exists(gp):
        for k in json.load(ropen(gp, "r")):
            P(f"  raw keyframe spacing {k['room']} {k['relpath']}: median {k['gap_median_s']} s "
              f"(min {k['gap_min_s']}, max {k['gap_max_s']}), {k['n_key']} keyframes in 120 s")
    else:
        missing_units.append("gop")
    P(f"  pre-onset continuous video within the same file [min]: {dist([e['pre_video_file_s'] / 60 for e in szm if e['pre_video_file_s'] is not None], 1)}")
    P(f"  pre-onset continuous video, file rollovers bridged [min]: {dist([e['pre_video_run_s'] / 60 for e in szm if e['pre_video_run_s'] is not None], 1)}")
    P(f"  pre-onset EDF in the same file [min]: {dist([e['pre_edf_s'] / 60 for e in szm if e['pre_edf_s'] is not None], 1)}")
    P(f"  gap since previous annotated event end (any label) [min]: {dist([e['gap_prev_event_s'] / 60 for e in szm if e['gap_prev_event_s'] is not None], 1)}")
    for h in (1, 5, 10, 30, 60, 240):
        n = sum(1 for e in szm if e["pre_video_run_s"] and e["pre_edf_s"] and e["pre_video_run_s"] >= h * 60
                and e["pre_edf_s"] >= h * 60 and (e["gap_prev_event_s"] is None or e["gap_prev_event_s"] >= h * 60)
                and not e.get("video_wallclock_defect"))
        P(f"    clipped seizures with >= {h:3d} min of video AND EDF before onset and no annotated event in that window: {n}")
    oe = [e for r in on.values() for e in r["events"] if r["animal"] in LABELLED_ANIMALS]
    P(f"  EDF onset sample: {len(oe)} events from {len([a for a in on if a in LABELLED_ANIMALS])} labelled animals "
      f"(skipped {dict(sum((Counter(r['skipped']) for r in on.values()), Counter()))})")
    ch = [e["change_s"] for e in oe if not e.get("degenerate_baseline")]
    P(f"    EEG change point (first {RUN} consecutive 1-s bins > baseline median + {K_MAD:g} robust SD, searched {SEARCH[0]}..{SEARCH[1]} s) relative to annotated start:")
    P(f"    found {sum(1 for c in ch if c is not None)} / {len(ch)}; {dist([c for c in ch if c is not None], 1)}")
    for w in (2, 5, 10):
        P(f"      within +-{w} s: {sum(1 for c in ch if c is not None and abs(c) <= w)}")
    P(f"    median z in the 10 s before the annotated start: {dist([e.get('pre10_median_z') for e in oe], 2)}")
    P(f"    median z in the 10 s after it:                   {dist([e.get('post10_median_z') for e in oe], 2)}")
    ch3 = [e.get("change_s_k3") for e in oe if not e.get("degenerate_baseline")]
    P(f"    same with 3 robust SD: found {sum(1 for c in ch3 if c is not None)} / {len(ch3)}; {dist([c for c in ch3 if c is not None], 1)}")
    for w in (2, 5, 10):
        P(f"      within +-{w} s: {sum(1 for c in ch3 if c is not None and abs(c) <= w)}")
    P(f"    events whose baseline (-60..-20 s) already holds a bin > {3 * K_MAD:g} robust SD: {sum(1 for e in oe if e.get('base_raised'))}")
    for lab in ("Stage 2", "Stage 3", "Stage 4", "Stage 5"):
        cc = [e["change_s"] for e in oe if e["label"] == lab and not e.get("degenerate_baseline")]
        P(f"    {lab}: change found {sum(1 for c in cc if c is not None)}/{len(cc)}  {dist([c for c in cc if c is not None], 1)}")
    if "all" in prof_out:
        mz = prof_out["all"]["median_z"]
        P("    median z profile (all stages), seconds -10..+15: " +
          " ".join(f"{s}:{mz[s + PRE_S]:.1f}" for s in range(-10, 16) if mz[s + PRE_S] is not None))
    P("")
    P(f"elapsed {time.time() - t_start:.1f} s")
    write_atomic("summary.txt", "\n".join(L) + "\n")
    print("\n".join(L))


# ================================================================== driver

def run_unit(spec):
    kind, arg = spec
    t = time.time()
    try:
        fn = {"labelled": unit_labelled, "epochs": unit_epochs, "onset": unit_onset,
              "detprobe": unit_detprobe}.get(kind)
        msg = fn(arg) if fn else {"xlsxzip": unit_xlsxzip, "clipprobe": unit_clipprobe, "gop": unit_gop}[kind]()
        return f"{kind} {arg or ''}: {msg} ({time.time() - t:.0f} s)"
    except Exception:
        return f"{kind} {arg or ''}: FAILED\n{traceback.format_exc()}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["labelled", "xlsxzip", "epochs", "onset", "detprobe", "clipprobe", "gop", "report", "all"])
    ap.add_argument("--animal")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True) if _under(CACHE, OUT) else None
    if a.cmd == "report":
        return report()
    if a.cmd in ("xlsxzip", "clipprobe", "gop"):
        print(run_unit((a.cmd, None)))
        return
    if a.cmd in ("labelled", "epochs", "onset", "detprobe"):
        animals = [a.animal] if a.animal else (LABELLED_ANIMALS if a.cmd == "labelled" else ALL_ANIMALS)
        for x in animals:
            print(run_unit((a.cmd, x)), flush=True)
        return
    specs = ([("epochs", x) for x in ALL_ANIMALS] + [("labelled", x) for x in LABELLED_ANIMALS] +
             [("onset", x) for x in ALL_ANIMALS] + [("xlsxzip", None), ("clipprobe", None), ("gop", None)])
    from multiprocessing import get_context
    with get_context("fork").Pool(a.workers) as pool:
        for msg in pool.imap_unordered(run_unit, specs):
            log(msg)
        # detprobe needs the epochs caches
        for msg in pool.imap_unordered(run_unit, [("detprobe", x) for x in ALL_ANIMALS]):
            log(msg)
    report()


if __name__ == "__main__":
    main()
