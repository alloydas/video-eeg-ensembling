#!/usr/bin/env python
"""Read-only integrity and coverage audit of the raw DSI/Noldus tree.

    RAW   /work/mech-ai/alloydas/EEG/Data/<animal>/<day folder>/...   (never written)
    OUT   $EEG_ROOT/output/raw_audit/                                  (the only place written)

Sub-commands
    list                  print the animal folders (one unit each)
    unit  --animal A      audit every file under RAW/A; resumable per file
    merge                 join the unit results into files.csv, per_animal.json,
                          events.csv, duplicates.csv, audit_summary.txt

Per file it reads headers, sidecars and a few frames, never whole videos:
  mp4   size; top-level box walk (moov present? box overruns EOF?); the Noldus trailer
        after moov (start time with ms, local and UTC); ffprobe fps / frames / WxH /
        duration; cv2 open + frame count + first frame; PyAV decode of the first frame,
        one frame at mid-file and the last decodable frame (seek to <=10 s before the
        end, decode to EOF); sha1 of the first and last 1 MiB; the XML sidecar, every
        field.
  XML   parsed with its mp4 (a row of its own too, so every file appears).
  edf   256-byte header + signal headers only; expected size, truncation, nrec=-1, the
        last complete record's zero fraction and ECG spread.
  xlsx  every event row; label vocabulary.
  other RN197 seizure_clips/ (info.txt, _clip_log.csv, video.mp4, eeg.edf) and zero-byte
        files: identified and cross-checked.
"""
import argparse
import csv
import datetime as dt
import glob
import hashlib
import io
import json
import os
import re
import statistics
import struct
import subprocess
import sys
import threading
import time
import traceback
import xml.etree.ElementTree as ET
import zipfile
from array import array
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from zoneinfo import ZoneInfo

RAW = os.path.realpath(os.environ.get("RAW_ROOT", "/work/mech-ai/alloydas/EEG/Data"))
EEG_ROOT = os.path.realpath(os.environ.get("EEG_ROOT", "/work/mech-ai-scratch/alloy/EEG"))
OUT = os.path.join(EEG_ROOT, "output", "raw_audit")
LABELLED = os.path.join(EEG_ROOT, "data")          # listed (names only), never opened for write
CENTRAL = ZoneInfo("America/Chicago")
UTC = dt.timezone.utc
FFBIN = "/work/mech-ai/alloy/miniconda3/bin"
FFPROBE = os.environ.get("FFPROBE", os.path.join(FFBIN, "ffprobe"))
FFMPEG = os.environ.get("FFMPEG", os.path.join(FFBIN, "ffmpeg"))
# cv2 is opt-in: its 66 MB .so pages in at ~2.6 MB/s from the conda env's filesystem and
# stalled every scavenger task for >15 min. ffprobe/ffmpeg live on the faster /work/mech-ai.
USE_CV2 = os.environ.get("AUDIT_CV2", "0") == "1"
MIB = 1 << 20
AUDIT_VERSION = 4      # unit/merge logic; a bump forces re-finalize (cheap)
FILE_VERSION = 2       # per-file probe logic; a bump forces re-probing every file

LABELLED_ANIMALS = ("RN197 RN199 RN204 RN208 RN210 RN213 RN215 RN216 RN219 RN222 RN223 RN224 "
                    "RN227 RN229 RN235 RN237 RN238 RN242 RN244 RN245").split()

# Problems that make a file unusable or incomplete. Everything else is a warning.
ERRORS = {
    "zero_byte", "unreadable", "mp4_no_moov", "mp4_no_mdat", "mp4_box_overruns_eof",
    "ffprobe_error", "no_video_stream", "first_frame_fail", "mid_frame_fail",
    "last_frame_fail", "last_frame_early", "decode_errors", "xml_missing", "xml_parse_error",
    "mp4_missing_for_xml", "edf_bad_header", "edf_truncated", "edf_extra_bytes", "edf_nrec_minus1",
    "edf_last_record_zero", "xlsx_open_error",
}
# Readable files whose timebase does not match the wall clock: video time != clock time.
TIMING = {"frames_missing_vs_wallclock", "stream_longer_than_wallclock", "xml_local_ne_central(utc)",
          "sync_offset_ne_trailer_ms"}
# Readable and consistent, but a consumer keyed on names/labels would mis-handle them.
COMPAT = {"edf_no_ecg", "xml_camera_ne_filename", "xml_path_name_ne_file", "folder_animal_not_in_xml_ids",
          "folder_animal_not_in_camera", "edf_misfiled_vs_video"}
WALL_TOL_S = 30.0     # clock drift between stream pts and the wall clock reaches ~10 s over a 9.8 h file


def severity(problems):
    for name, group in (("error", ERRORS), ("timing", TIMING), ("compat", COMPAT)):
        if any(p in group for p in problems):
            return name
    return "warn" if problems else "ok"


BENIGN_H264 = re.compile(r"Missing reference picture|decode_slice_header error|reference picture missing")
DTS_WARN = re.compile(r"non monotonically increasing dts")


def reclassify(r):
    """Downgrade decoder messages that every decode start of these streams produces. The
    streams open on non-IDR keyframes, so h264 logs missing references until the next
    keyframe; first/mid/last frames still decode. Idempotent (works on the cached list)."""
    p = r["problems"]
    if "decode_errors" in p:
        raw = r.get("ffmpeg_errors") or ""
        msgs = [m for m in raw.split(" | ") if m]
        if len(raw) >= 600 and msgs:
            msgs = msgs[:-1]            # the last piece may be cut mid-message
        body = [m.split(": ", 1)[1] if ": " in m else m for m in msgs]
        if body and all(BENIGN_H264.search(m) or DTS_WARN.search(m) for m in body) \
                and r.get("first_ok") and r.get("last_ok") and r.get("mid_ok") is not False:
            p.remove("decode_errors")
            if any(BENIGN_H264.search(m) for m in body):
                p.append("h264_missing_ref_at_decode_start")
            if any(DTS_WARN.search(m) for m in body):
                p.append("nonmonotonic_dts_near_end")
    if "ffprobe_error" in p and r.get("codec"):
        lines = [l for l in (r.get("ffprobe_error") or "").splitlines() if l.strip()]
        if lines and all(BENIGN_H264.search(l) for l in lines):
            p.remove("ffprobe_error")
            if "h264_missing_ref_at_decode_start" not in p:
                p.append("h264_missing_ref_at_decode_start")
    if r["kind"] in ("edf", "clip_edf") and r.get("edf_labels") and r.get("edf_fs"):
        labs, fss = r["edf_labels"].split("|"), r["edf_fs"].split("|")
        r["edf_1khz_label"] = next((l for l, f in zip(labs, fss) if f == "1000"), None)


# ----------------------------------------------------------------------------- guards
def _under(path, root):
    p, r = os.path.realpath(path), os.path.realpath(root)
    return p == r or p.startswith(r + os.sep)


def out_path(*parts):
    """Every write goes through here: under OUT, never under RAW."""
    p = os.path.join(OUT, *parts)
    if not _under(p, OUT) or _under(p, RAW):
        raise SystemExit(f"refusing to write outside {OUT}: {p}")
    return p


def write_atomic(path, text):
    path = out_path(os.path.relpath(path, OUT))
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)


def ropen(path):
    """Read-only binary open of a raw file."""
    return open(path, "rb")


# ----------------------------------------------------------------------------- helpers
def sha1_head_tail(path, size):
    with ropen(path) as fh:
        head = fh.read(MIB)
        if size > MIB:
            fh.seek(max(0, size - MIB))
            tail = fh.read(MIB)
        else:
            tail = head
    return hashlib.sha1(head).hexdigest(), hashlib.sha1(tail).hexdigest()


def iso(x):
    if x is None:
        return None
    if isinstance(x, dt.datetime):
        return x.isoformat(sep=" ")
    return str(x)


def naive_utc(epoch):
    return dt.datetime.fromtimestamp(epoch, tz=UTC).replace(tzinfo=None)


def local_to_utc_naive(naive_local):
    """Central wall time -> naive UTC (fold=0 for the ambiguous 01:00-02:00 on 2023-11-05)."""
    return naive_local.replace(tzinfo=CENTRAL).astimezone(UTC).replace(tzinfo=None)


def utc_to_local_naive(naive_utc_dt):
    return naive_utc_dt.replace(tzinfo=UTC).astimezone(CENTRAL).replace(tzinfo=None)


def secs(a, b):
    """a - b in seconds, or None."""
    if a is None or b is None:
        return None
    return round((a - b).total_seconds(), 3)


def parse_dt(s):
    if s in (None, ""):
        return None
    if isinstance(s, dt.datetime):
        return s
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return dt.datetime.strptime(str(s), fmt)
        except ValueError:
            pass
    return None


def animal_of_folder(folder):
    m = re.search(r"(RN\d+)", folder)
    return m.group(1) if m else folder


def day_folder_dates(name):
    """'10-13-2023-10-14-2023', '10-20-2023 to 10-22-2023', '11-06-2023 (no video)',
    '10-12-2023 (2)' -> (first date, last date)."""
    ds = []
    for m in re.finditer(r"(\d{1,2})-(\d{1,2})-(\d{4})", name):
        try:
            ds.append(dt.date(int(m.group(3)), int(m.group(1)), int(m.group(2))))
        except ValueError:
            pass
    if not ds:
        return None, None
    return ds[0], ds[-1]


# ----------------------------------------------------------------------------- mp4
KNOWN_BOXES = {b"ftyp", b"moov", b"mdat", b"free", b"skip", b"wide", b"uuid", b"meta", b"pdin",
               b"moof", b"mfra", b"styp", b"sidx", b"udta", b"junk", b"pnot"}


def mp4_boxes(path, size):
    """Walk the top-level boxes. Stops at the first header that is not a known 4CC."""
    boxes, off, trailing = [], 0, 0
    overrun = False
    with ropen(path) as fh:
        while off + 8 <= size:
            fh.seek(off)
            h = fh.read(16)
            bsz, typ = struct.unpack(">I4s", h[:8])
            hdr = 8
            if bsz == 1 and len(h) >= 16:
                bsz, hdr = struct.unpack(">Q", h[8:16])[0], 16
            elif bsz == 0:
                bsz = size - off
            if typ not in KNOWN_BOXES or bsz < hdr:
                trailing = size - off
                break
            if off + bsz > size:
                overrun = True
                boxes.append((typ.decode("latin1"), off, bsz))
                break
            boxes.append((typ.decode("latin1"), off, bsz))
            off += bsz
        else:
            trailing = size - off
    return boxes, trailing, overrun


def noldus_trailer(path, size):
    """Noldus Media Recorder appends ~320 bytes after moov: UTF-16 'Noldus', 'Media Recorder
    4.0', a version string, then 'data' + 4 bytes + two 12-byte records
    (u16 year, u8 month, u8 day, u8 hour, u8 min, u8 sec, u8 x, u32 ms): local then UTC."""
    n = min(size, 4096)
    with ropen(path) as fh:
        fh.seek(size - n)
        tail = fh.read(n)
    res = {"trailer_present": False}
    i = tail.rfind("Noldus".encode("utf-16-le"))
    j = tail.rfind("data".encode("utf-16-le"))
    if i < 0 or j < 0 or j < i:
        return res
    res["trailer_present"] = True
    res["trailer_bytes"] = n - i
    txt = tail[i:j].decode("utf-16-le", "replace").replace("\x00", " ")
    res["trailer_text"] = " ".join(txt.split())[:120]
    recs = tail[j + 8 + 4: j + 8 + 4 + 24]
    out = []
    for k in range(2):
        r = recs[12 * k: 12 * (k + 1)]
        if len(r) < 12:
            out.append(None)
            continue
        y, mo, d, hh, mi, ss, x, ms = struct.unpack("<HBBBBBBI", r)
        try:
            out.append((dt.datetime(y, mo, d, hh, mi, ss, ms * 1000), x, ms))
        except ValueError:
            out.append(None)
    if out[0]:
        res["trailer_local"] = iso(out[0][0]); res["trailer_local_x"] = out[0][1]; res["trailer_local_ms"] = out[0][2]
    if out[1]:
        res["trailer_utc"] = iso(out[1][0]); res["trailer_utc_x"] = out[1][1]; res["trailer_utc_ms"] = out[1][2]
    return res


def ffprobe(path):
    cmd = [FFPROBE, "-v", "error", "-show_entries",
           "format=duration,size,bit_rate,start_time,format_name,nb_streams:"
           "stream=index,codec_type,codec_name,profile,pix_fmt,width,height,r_frame_rate,"
           "avg_frame_rate,nb_frames,duration,start_time,time_base",
           "-of", "json", path]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return None, "ffprobe timeout"
    err = p.stderr.strip()
    try:
        j = json.loads(p.stdout or "{}")
    except json.JSONDecodeError:
        return None, (err or "ffprobe bad json")[:300]
    return j, err[:300] if (err or p.returncode) else None


def _frac(s):
    try:
        a, b = s.split("/")
        return float(a) / float(b) if float(b) else None
    except Exception:
        return None


SHOWINFO = re.compile(r"n:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:(\S+).*?iskey:(\d).*?"
                      r"mean:\[([^\]]*)\]\s*stdev:\[([^\]]*)\]")
FF_ERR = re.compile(r"error|corrupt|invalid|truncat|non-existing|moov atom|missing|concealing|overread", re.I)


def ff_showinfo(args, timeout=600):
    """Run ffmpeg -> null with showinfo; return (frames, error lines). Each frame is
    (n, pts_time, iskey, luma mean, luma stdev)."""
    cmd = [FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "info", *args, "-an", "-vf", "showinfo", "-f", "null", "-"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return [], ["ffmpeg timeout"]
    frames, errs = [], []
    for line in p.stderr.splitlines():
        if "showinfo" in line:
            m = SHOWINFO.search(line)
            if m:
                frames.append((int(m.group(1)), float(m.group(3)), int(m.group(4)),
                               float(m.group(5).split()[0]), float(m.group(6).split()[0])))
        elif FF_ERR.search(line):
            errs.append(line.strip()[:160])
    if p.returncode != 0 and not errs:
        errs.append(f"ffmpeg exit {p.returncode}")
    return frames, errs


def ff_frames(path, dur_s):
    """First frame, the frame at mid-file, and the last decodable frame (decode from a
    keyframe <= 10 s before the end to EOF). Timestamps are the file's own (-copyts)."""
    r, errs = {}, []
    fr, e = ff_showinfo(["-copyts", "-i", path, "-frames:v", "1"])
    errs += [f"first: {x}" for x in e]
    r["first_ok"] = bool(fr)
    if fr:
        r["first_pts_s"], r["first_mean"], r["first_std"] = fr[0][1], fr[0][3], fr[0][4]
    if dur_s and dur_s > 0:
        fr, e = ff_showinfo(["-copyts", "-ss", f"{dur_s / 2:.3f}", "-i", path, "-frames:v", "1"])
        errs += [f"mid: {x}" for x in e]
        r["mid_ok"] = bool(fr)
        if fr:
            r["mid_pts_s"], r["mid_mean"], r["mid_std"] = fr[0][1], fr[0][3], fr[0][4]
    fr = []
    for back in (10, 120, 1200):
        fr, e = ff_showinfo(["-copyts", "-sseof", f"-{back}", "-i", path])
        errs += [f"last(-{back}s): {x}" for x in e]
        if fr:
            break
    r["last_ok"] = bool(fr)
    if fr:
        r["last_pts_s"], r["last_mean"], r["last_std"] = fr[-1][1], fr[-1][3], fr[-1][4]
        r["tail_frames_decoded"] = len(fr)
    if errs:
        r["ffmpeg_errors"] = " | ".join(errs)[:600]
        r["n_ffmpeg_errors"] = len(errs)
    return r


def cv2_probe(path):
    import cv2
    r = {}
    try:
        cap = cv2.VideoCapture(path)
        r["cv2_opened"] = bool(cap.isOpened())
        if r["cv2_opened"]:
            r["cv2_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            r["cv2_fps"] = round(cap.get(cv2.CAP_PROP_FPS), 4)
            r["cv2_w"], r["cv2_h"] = int(cap.get(3)), int(cap.get(4))
            ok, _ = cap.read()
            r["cv2_first_ok"] = bool(ok)
        cap.release()
    except Exception as e:
        r["cv2_error"] = str(e)[:200]
    return r


def parse_xml(path):
    r = {"xml_present": True}
    try:
        with ropen(path) as fh:
            raw = fh.read()
        root = ET.fromstring(raw)
    except Exception as e:
        r["xml_error"] = str(e)[:200]
        return r
    known = {"DSI_utc_start_time", "DSI_camera_name", "DSI_animal_ID", "DSI_version", "Duration",
             "Dsi_NativeSize", "Dsi_VideoFullPath", "DSI_local_start_time", "DSI_sync_offset"}
    t = lambda k: (root.findtext(k) or "").strip() or None
    r["xml_utc_start"] = t("DSI_utc_start_time")
    r["xml_camera"] = t("DSI_camera_name")
    ids = root.find("DSI_animal_ID")
    r["xml_animal_ids"] = "|".join((e.text or "").strip() for e in ids) if ids is not None else None
    r["xml_version"] = t("DSI_version")
    r["xml_duration"] = t("Duration")
    r["xml_native_w"] = t("Dsi_NativeSize/width")
    r["xml_native_h"] = t("Dsi_NativeSize/height")
    r["xml_fullpath"] = t("Dsi_VideoFullPath")
    r["xml_local_raw"] = t("DSI_local_start_time")
    r["xml_sync_offset"] = t("DSI_sync_offset")
    other = [c.tag for c in root if c.tag not in known]
    r["xml_other_tags"] = "|".join(other) if other else None
    # decode
    try:
        u = int(r["xml_utc_start"])
        r["xml_utc_iso"] = iso(naive_utc(u))
    except Exception:
        pass
    try:
        p = [int(x) for x in r["xml_local_raw"].split(":")]
        r["xml_local_iso"] = iso(dt.datetime(1900 + p[0], p[1] + 1, p[2], p[3], p[4], p[5]))
    except Exception:
        pass
    if r["xml_fullpath"]:
        parts = r["xml_fullpath"].split("\\")
        r["xml_path_name"] = parts[-1]
        r["xml_path_day"] = parts[-2] if len(parts) >= 2 else None
        r["xml_path_prefix"] = "\\".join(parts[:-2])
    return r


MP4_NAME = re.compile(r"^(?P<cam>.+)\.(?P<ts>\d{14})\.mp4$", re.I)


def audit_mp4(path, rec, with_xml=True):
    size = rec["size"]
    probs = rec["problems"]
    m = MP4_NAME.match(rec["name"])
    if m:
        rec["camera"] = m.group("cam")
        try:
            rec["fn_time"] = iso(dt.datetime.strptime(m.group("ts"), "%Y%m%d%H%M%S"))
        except ValueError:
            rec["fn_time"] = None
    if size == 0:
        probs.append("zero_byte")
        return
    boxes, trailing, overrun = mp4_boxes(path, size)
    types = [b[0] for b in boxes]
    rec["boxes"] = ",".join(types)
    rec["moov_offset"] = next((b[1] for b in boxes if b[0] == "moov"), None)
    rec["mdat_bytes"] = next((b[2] for b in boxes if b[0] == "mdat"), None)
    rec["trailing_bytes"] = trailing
    if "moov" not in types:
        probs.append("mp4_no_moov")
    if "mdat" not in types:
        probs.append("mp4_no_mdat")
    if overrun:
        probs.append("mp4_box_overruns_eof")
    rec.update(noldus_trailer(path, size))
    if not rec.get("trailer_present"):
        probs.append("noldus_trailer_missing")
    j, err = ffprobe(path)
    if err:
        rec["ffprobe_error"] = err
        probs.append("ffprobe_error")
    vs = None
    if j:
        fmt = j.get("format", {})
        rec["fmt_duration"] = float(fmt["duration"]) if fmt.get("duration") not in (None, "N/A") else None
        rec["fmt_name"] = fmt.get("format_name")
        rec["bit_rate"] = int(fmt["bit_rate"]) if str(fmt.get("bit_rate", "")).isdigit() else None
        sts = j.get("streams", [])
        rec["n_streams"] = len(sts)
        rec["stream_types"] = ",".join(s.get("codec_type", "?") for s in sts)
        vs = next((s for s in sts if s.get("codec_type") == "video"), None)
    if vs is None:
        probs.append("no_video_stream")
    else:
        rec["codec"] = vs.get("codec_name")
        rec["pix_fmt"] = vs.get("pix_fmt")
        rec["width"], rec["height"] = vs.get("width"), vs.get("height")
        rec["r_fps"] = _frac(vs.get("r_frame_rate", ""))
        rec["avg_fps"] = round(_frac(vs.get("avg_frame_rate", "")) or 0, 5) or None
        rec["nb_frames"] = int(vs["nb_frames"]) if str(vs.get("nb_frames", "")).isdigit() else None
        rec["stream_duration"] = float(vs["duration"]) if vs.get("duration") not in (None, "N/A") else None
    dur = rec.get("stream_duration") or rec.get("fmt_duration")
    rec["duration_s"] = dur
    rec["hours"] = round(dur / 3600, 5) if dur else None
    if rec.get("width") and (rec["width"], rec["height"]) != (800, 600) and rec["kind"] == "video":
        probs.append("resolution_not_800x600")
    if rec.get("r_fps") and abs(rec["r_fps"] - 15) > 0.01 and rec["kind"] == "video":
        probs.append("fps_not_15")
    if USE_CV2:
        rec.update(cv2_probe(path))
        if rec.get("cv2_opened") is False:
            probs.append("cv2_open_fail")
        if rec.get("cv2_frames") and rec.get("nb_frames") and rec["cv2_frames"] != rec["nb_frames"]:
            probs.append("cv2_nframes_differs")
    if "moov" in types:
        rec.update(ff_frames(path, dur))
        if rec.get("first_ok") is False:
            probs.append("first_frame_fail")
        if rec.get("mid_ok") is False:
            probs.append("mid_frame_fail")
        if rec.get("last_ok") is False:
            probs.append("last_frame_fail")
        if rec.get("last_ok") and dur and rec.get("last_pts_s") is not None:
            rec["last_frame_gap_s"] = round(dur - rec["last_pts_s"], 4)
            if rec["last_frame_gap_s"] > 2.0:
                probs.append("last_frame_early")
        if rec.get("ffmpeg_errors"):
            probs.append("decode_errors")
        for k in ("first", "mid", "last"):
            if rec.get(f"{k}_ok") and (rec.get(f"{k}_std") or 0) < 2.0:
                probs.append(f"{k}_frame_blank")
    rec["hash_head"], rec["hash_tail"] = sha1_head_tail(path, size)
    if with_xml:
        xp = None
        for ext in (".XML", ".xml"):
            c = path[:-4] + ext
            if os.path.exists(c):
                xp = c
                break
        if xp is None:
            probs.append("xml_missing")
            rec["xml_present"] = False
        else:
            x = parse_xml(xp)
            rec.update(x)
            if "xml_error" in x:
                probs.append("xml_parse_error")


# ----------------------------------------------------------------------------- edf
EDF_STD_LABELS = ["Activity", "ECG", "ECG [FIR-HP: 5Hz", "SignalStr", "Temp"]


def audit_edf(path, rec):
    size = rec["size"]
    probs = rec["problems"]
    if size == 0:
        probs.append("zero_byte")
        return
    try:
        with ropen(path) as fh:
            h = fh.read(256)
            if len(h) < 256:
                probs.append("edf_bad_header")
                return
            f = lambda a, b: h[a:b].decode("latin1").strip()
            rec["edf_version"] = f(0, 8)
            rec["edf_patient"] = f(8, 88)
            rec["edf_recording"] = f(88, 168)
            rec["edf_startdate"] = f(168, 176)
            rec["edf_starttime"] = f(176, 184)
            hdr = int(f(184, 192))
            rec["edf_reserved"] = f(192, 236)
            nrec = int(f(236, 244))
            rdur = float(f(244, 252))
            ns = int(f(252, 256))
            rec["edf_header_bytes"], rec["edf_nrec"], rec["edf_rec_dur"], rec["edf_ns"] = hdr, nrec, rdur, ns
            sh = fh.read(256 * ns)
            if len(sh) < 256 * ns:
                probs.append("edf_bad_header")
                return
            fields, o = {}, 0
            for name, w in (("label", 16), ("transducer", 80), ("dim", 8), ("pmin", 8), ("pmax", 8),
                            ("dmin", 8), ("dmax", 8), ("prefilter", 80), ("nsamp", 8), ("sreserved", 32)):
                fields[name] = [sh[o + i * w: o + (i + 1) * w].decode("latin1").strip() for i in range(ns)]
                o += w * ns
            labels = fields["label"]
            nsamp = [int(x) for x in fields["nsamp"]]
            rec["edf_labels"] = "|".join(labels)
            rec["edf_nsamp"] = "|".join(map(str, nsamp))
            rec["edf_dims"] = "|".join(fields["dim"])
            rec["edf_fs"] = "|".join(f"{n / rdur:g}" if rdur else "?" for n in nsamp)
            if labels != EDF_STD_LABELS:
                probs.append("edf_channels_nonstandard")
            # parent pipeline: first channel whose label contains 'ECG'
            ecg = next((i for i, l in enumerate(labels) if "ECG" in l), None)
            if ecg is None:
                probs.append("edf_no_ecg")
            else:
                rec["edf_ecg_label"] = labels[ecg]
                rec["edf_ecg_fs"] = nsamp[ecg] / rdur if rdur else None
                rec["edf_ecg_phys"] = f"{fields['pmin'][ecg]}..{fields['pmax'][ecg]} {fields['dim'][ecg]}"
                rec["edf_ecg_dig"] = f"{fields['dmin'][ecg]}..{fields['dmax'][ecg]}"
            if hdr != 256 * (ns + 1):
                probs.append("edf_bad_header")
            recbytes = 2 * sum(nsamp)
            rec["edf_record_bytes"] = recbytes
            ncomplete = (size - hdr) // recbytes if recbytes else 0
            rec["edf_records_in_file"] = ncomplete
            rec["edf_partial_bytes"] = (size - hdr) - ncomplete * recbytes if recbytes else None
            if nrec == -1:
                probs.append("edf_nrec_minus1")
                rec["edf_expected_size"] = None
            else:
                exp = hdr + nrec * recbytes
                rec["edf_expected_size"] = exp
                rec["edf_size_diff"] = size - exp
                if size < exp:
                    probs.append("edf_truncated")
                elif size > exp:
                    probs.append("edf_extra_bytes")
            usable = ncomplete if nrec == -1 else min(nrec, ncomplete)
            rec["edf_hours"] = round(usable * rdur / 3600, 5)
            rec["edf_hours_header"] = round(nrec * rdur / 3600, 5) if nrec >= 0 else None
            # start
            try:
                dd, mm, yy = [int(x) for x in rec["edf_startdate"].split(".")]
                hh, mi, ss = [int(x) for x in rec["edf_starttime"].split(".")]
                yy += 1900 if yy >= 85 else 2000
                st = dt.datetime(yy, mm, dd, hh, mi, ss)
                rec["edf_start"] = iso(st)
                rec["edf_end"] = iso(st + dt.timedelta(seconds=usable * rdur))
            except Exception:
                probs.append("edf_bad_startdatetime")
            # last complete record: all-zero? ECG flat?
            if ncomplete >= 1 and recbytes:
                for tag, idx in (("first", 0), ("last", usable - 1)):
                    if idx < 0:
                        continue
                    fh.seek(hdr + idx * recbytes)
                    a = array("h")
                    a.frombytes(fh.read(recbytes))
                    if sys.byteorder != "little":
                        a.byteswap()
                    rec[f"edf_{tag}_rec_zero_frac"] = round(a.count(0) / len(a), 4)
                    if ecg is not None:
                        o0 = sum(nsamp[:ecg])
                        rec[f"edf_{tag}_rec_ecg_std_dig"] = round(statistics.pstdev(a[o0: o0 + nsamp[ecg]]), 2)
                if rec.get("edf_last_rec_zero_frac") == 1.0:
                    probs.append("edf_last_record_zero")
                if ecg is not None and rec.get("edf_last_rec_ecg_std_dig", 1) == 0:
                    probs.append("edf_last_record_ecg_flat")
    except Exception as e:
        rec["edf_error"] = str(e)[:200]
        probs.append("edf_bad_header")
    # date in the file name, e.g. RN201-10-9-2023.edf or 'RN201-10-12-2023 (2).edf'
    m = re.search(r"(\d{1,2})-(\d{1,2})-(\d{4})", rec["name"])
    if m and rec.get("edf_start"):
        fd = dt.date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
        rec["edf_name_date"] = fd.isoformat()
        if fd != parse_dt(rec["edf_start"]).date():
            probs.append("edf_start_date_ne_filename")


# ----------------------------------------------------------------------------- xlsx
XNS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
       "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
DATE_FMT_IDS = set(range(14, 23)) | {45, 46, 47}


def _is_date_code(code):
    c = re.sub(r'"[^"]*"|\[[^\]]*\]|\\.', "", code or "").lower()
    return c != "general" and bool(re.search(r"[dmyhs]", c))


def _col_index(ref):
    n = 0
    for ch in ref:
        if not ch.isalpha():
            break
        n = n * 26 + (ord(ch.upper()) - 64)
    return n - 1


def read_xlsx(data):
    """Minimal stdlib xlsx reader: [(sheet name, [row tuples])], dates as datetime
    (styles with a date number format), shared/inline strings as str."""
    z = zipfile.ZipFile(io.BytesIO(data))
    names = set(z.namelist())
    m = "{%s}" % XNS["m"]
    shared = []
    if "xl/sharedStrings.xml" in names:
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", XNS):
            shared.append("".join(t.text or "" for t in si.iter(m + "t")))
    date_styles = set()
    if "xl/styles.xml" in names:
        st = ET.fromstring(z.read("xl/styles.xml"))
        custom = {int(f.get("numFmtId")): f.get("formatCode", "")
                  for f in st.findall("m:numFmts/m:numFmt", XNS)}
        for i, xf in enumerate(st.findall("m:cellXfs/m:xf", XNS)):
            fid = int(xf.get("numFmtId", 0))
            if fid in DATE_FMT_IDS or (fid in custom and _is_date_code(custom[fid])):
                date_styles.add(i)
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    pr = wb.find("m:workbookPr", XNS)
    base = dt.datetime(1904, 1, 1) if pr is not None and pr.get("date1904") in ("1", "true") else dt.datetime(1899, 12, 30)
    rels = {}
    if "xl/_rels/workbook.xml.rels" in names:
        for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels")):
            rels[r.get("Id")] = r.get("Target")
    out = []
    for sh in wb.findall("m:sheets/m:sheet", XNS):
        tgt = rels.get(sh.get("{%s}id" % XNS["r"]), "")
        path = tgt.lstrip("/") if tgt.startswith("/") else "xl/" + tgt
        rows = []
        if path in names:
            for row in ET.fromstring(z.read(path)).iter(m + "row"):
                vals = {}
                for c in row.findall("m:c", XNS):
                    t, sidx = c.get("t", "n"), int(c.get("s", 0))
                    v = c.find("m:v", XNS)
                    txt = v.text if v is not None else None
                    if t == "s":
                        val = shared[int(txt)] if txt is not None else None
                    elif t == "inlineStr":
                        val = "".join(x.text or "" for x in c.iter(m + "t"))
                    elif t in ("str", "e"):
                        val = txt
                    elif t == "b":
                        val = bool(int(txt)) if txt else None
                    elif txt is None:
                        val = None
                    else:
                        num = float(txt)
                        if sidx in date_styles:
                            val = base + dt.timedelta(milliseconds=round(num * 86400000))
                        else:
                            val = int(num) if num.is_integer() else num
                    ref = c.get("r")
                    vals[_col_index(ref) if ref else len(vals)] = val
                if vals:
                    rows.append(tuple(vals.get(i) for i in range(max(vals) + 1)))
        out.append((sh.get("name"), rows))
    return out


def _cell_dt(v):
    if isinstance(v, dt.datetime):
        if v.year < 1900:
            return None, "time_only"
        return v, "datetime"
    if isinstance(v, dt.time):
        return None, "time_only"
    if v is None:
        return None, "empty"
    p = parse_dt(str(v).strip())
    return p, "string" if p else f"unparsed:{type(v).__name__}"


def audit_xlsx(path, rec):
    probs = rec["problems"]
    events = []
    if rec["size"] == 0:
        probs.append("zero_byte")
        return events
    try:
        with ropen(path) as fh:
            data = fh.read()
        sheets = read_xlsx(data)
    except Exception as e:
        rec["xlsx_error"] = f"{type(e).__name__}: {e}"[:200]
        probs.append("xlsx_open_error")
        return events
    rec["xlsx_sheets"] = "|".join(name for name, _ in sheets)
    nrows_total = 0
    for title, rows in sheets:
        rows = [r for r in rows if r is not None and any(v not in (None, "") for v in r)]
        nrows_total += len(rows)
        if not rows:
            continue
        head = [str(v).strip() if v is not None else "" for v in rows[0]]
        rec.setdefault("xlsx_header", "|".join(head))
        low = [h.lower() for h in head]
        col = lambda key: next((i for i, h in enumerate(low) if key in h), None)
        ci = {"label": col("label"), "dur": col("duration"), "start": col("start"), "end": col("end"),
              "spike": col("spike"), "eeg": col("eeg")}
        body = rows[1:]
        if body and all((v is None or str(v).startswith("[")) for v in body[0]):
            rec.setdefault("xlsx_units", "|".join("" if v is None else str(v) for v in body[0]))
            body = body[1:]
        for k, r in enumerate(body):
            g = lambda key: r[ci[key]] if ci[key] is not None and ci[key] < len(r) else None
            st, st_t = _cell_dt(g("start"))
            en, en_t = _cell_dt(g("end"))
            dur = g("dur")
            try:
                dur = float(dur) if dur is not None else None
            except (TypeError, ValueError):
                dur = None
            ev = {"sheet": title, "row": k + 1, "label": None if g("label") is None else str(g("label")).strip(),
                  "duration_s": dur, "start": iso(st), "end": iso(en), "start_type": st_t, "end_type": en_t,
                  "spikes": g("spike"), "eeg_ave": g("eeg"), "problems": []}
            if st is None:
                ev["problems"].append(f"start_{st_t}")
            if st and en and dur is not None:
                ev["end_minus_start_minus_dur_s"] = round((en - st).total_seconds() - dur, 3)
                if abs(ev["end_minus_start_minus_dur_s"]) > 0.01:
                    ev["problems"].append("duration_ne_end_minus_start")
            if st and en and en < st:
                ev["problems"].append("end_before_start")
            events.append(ev)
    rec["xlsx_rows_nonempty"] = nrows_total
    rec["n_events"] = len(events)
    rec["event_labels"] = "|".join(f"{k}:{v}" for k, v in sorted(Counter(e["label"] for e in events).items(), key=lambda kv: str(kv[0])))
    if any(e["problems"] for e in events):
        probs.append("xlsx_event_row_problems")
    return events


# ----------------------------------------------------------------------------- other
def audit_info_txt(path, rec):
    if rec["size"] == 0:
        rec["problems"].append("zero_byte")
        return
    with ropen(path) as fh:
        txt = fh.read(65536).decode("utf-8", "replace")
    kv = {}
    for line in txt.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            kv[k.strip()] = v.strip()
    rec["info"] = kv
    rec["identified_as"] = "clip info.txt (clip cutter metadata: seizure times, source video/EDF)"


def audit_clip_log(path, rec):
    if rec["size"] == 0:
        rec["problems"].append("zero_byte")
        return
    with ropen(path) as fh:
        txt = fh.read().decode("utf-8", "replace")
    rows = list(csv.DictReader(io.StringIO(txt)))
    rec["clip_log_rows"] = rows
    rec["n_rows"] = len(rows)
    rec["identified_as"] = "clip cutter _clip_log.csv (one row per cut seizure clip)"


def sniff(path, rec):
    if rec["size"] == 0:
        rec["problems"].append("zero_byte")
        rec["identified_as"] = "zero-byte file"
        return
    with ropen(path) as fh:
        head = fh.read(16)
    rec["magic_hex"] = head.hex()
    rec["identified_as"] = "unknown"


def classify(rel):
    parts = rel.split(os.sep)
    name = parts[-1]
    low = name.lower()
    in_clips = "seizure_clips" in parts
    if in_clips:
        if low == "video.mp4":
            return "clip_video"
        if low == "eeg.edf":
            return "clip_edf"
        if low == "info.txt":
            return "clip_info"
        if low == "_clip_log.csv":
            return "clip_log"
        return "clip_other"
    if len(parts) == 3:        # animal/day/file
        if low.endswith(".mp4"):
            return "video"
        if low.endswith(".xml"):
            return "xml"
        if low.endswith(".edf"):
            return "edf"
        if low.endswith(".xlsx"):
            return "xlsx"
    return "other"


def audit_file(path, rel):
    st = os.stat(path)
    parts = rel.split(os.sep)
    rec = {"animal_folder": parts[0], "animal": animal_of_folder(parts[0]),
           "day_folder": parts[1] if len(parts) > 2 else "", "relpath": rel, "name": parts[-1],
           "ext": os.path.splitext(parts[-1])[1].lower().lstrip("."), "size": st.st_size,
           "mtime": iso(dt.datetime.fromtimestamp(st.st_mtime)), "problems": []}
    kind = classify(rel)
    rec["kind"] = kind
    events = []
    try:
        if kind == "video":
            audit_mp4(path, rec)
        elif kind == "clip_video":
            audit_mp4(path, rec, with_xml=False)
        elif kind == "xml":
            if st.st_size == 0:
                rec["problems"].append("zero_byte")
            else:
                x = parse_xml(path)
                rec.update(x)
                if "xml_error" in x:
                    rec["problems"].append("xml_parse_error")
            if not any(os.path.exists(path[:-4] + e) for e in (".mp4", ".MP4")):
                rec["problems"].append("mp4_missing_for_xml")
        elif kind in ("edf", "clip_edf"):
            audit_edf(path, rec)
        elif kind == "xlsx":
            events = audit_xlsx(path, rec)
        elif kind == "clip_info":
            audit_info_txt(path, rec)
        elif kind == "clip_log":
            audit_clip_log(path, rec)
        else:
            sniff(path, rec)
    except Exception as e:
        rec["problems"].append("unreadable")
        rec["exception"] = f"{type(e).__name__}: {e}"[:300]
        rec["traceback"] = traceback.format_exc()[-800:]
    return rec, events


# ----------------------------------------------------------------------------- unit
def list_animals():
    return sorted(d for d in os.listdir(RAW) if os.path.isdir(os.path.join(RAW, d)))


def walk_files(animal_dir):
    out = []
    for root, dirs, files in os.walk(os.path.join(RAW, animal_dir)):
        dirs.sort()
        for f in sorted(files):
            p = os.path.join(root, f)
            out.append((p, os.path.relpath(p, RAW)))
    return out


def run_unit(animal_dir, workers, finalize_only=False):
    os.makedirs(out_path("units"), exist_ok=True)
    final = out_path("units", f"{animal_dir}.json")
    cache = out_path("units", f"{animal_dir}.files.jsonl")
    files = walk_files(animal_dir)
    done = {}
    if os.path.exists(cache):
        with open(cache) as fh:
            for line in fh:
                try:
                    j = json.loads(line)
                except json.JSONDecodeError:
                    continue            # a line cut by preemption
                done[j["key"]] = j
    todo = []
    for p, rel in files:
        st = os.stat(p)
        key = f"{rel}|{st.st_size}|{st.st_mtime_ns}|f{FILE_VERSION}"
        if key not in done:
            todo.append((p, rel, key))
    print(f"[{animal_dir}] {len(files)} files, {len(files) - len(todo)} cached, {len(todo)} to audit", flush=True)
    if finalize_only and todo:
        raise SystemExit(f"[{animal_dir}] {len(todo)} files not yet audited; run the unit job first")
    lock = threading.Lock()
    t0 = time.time()
    with open(cache, "a") as fh, ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(audit_file, p, rel): (p, rel, key) for p, rel, key in todo}
        for i, fu in enumerate(as_completed(futs), 1):
            p, rel, key = futs[fu]
            rec, events = fu.result()
            j = {"key": key, "rec": rec, "events": events}
            with lock:
                fh.write(json.dumps(j, default=str) + "\n")
                fh.flush()
            done[key] = j
            if i % 25 == 0 or i == len(todo):
                print(f"[{animal_dir}] {i}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
    keys = {f"{rel}|{os.stat(p).st_size}|{os.stat(p).st_mtime_ns}|f{FILE_VERSION}" for p, rel in files}
    recs = [done[k] for k in sorted(keys)]
    unit = finalize_unit(animal_dir, [r["rec"] for r in recs], [(r["rec"], r["events"]) for r in recs])
    write_atomic(final, json.dumps(unit, default=str))
    print(f"[{animal_dir}] complete: {len(recs)} files -> {final}", flush=True)


# ----------------------------------------------------------------------------- per-animal analysis
def interval_union(iv):
    iv = sorted((a, b) for a, b in iv if a is not None and b is not None and b > a)
    out = []
    for a, b in iv:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def union_hours(iv):
    return sum((b - a).total_seconds() for a, b in interval_union(iv)) / 3600


def intersect_hours(iv1, iv2):
    u1, u2 = interval_union(iv1), interval_union(iv2)
    i = j = 0
    tot = 0.0
    while i < len(u1) and j < len(u2):
        a, b = max(u1[i][0], u2[j][0]), min(u1[i][1], u2[j][1])
        if b > a:
            tot += (b - a).total_seconds()
        if u1[i][1] < u2[j][1]:
            i += 1
        else:
            j += 1
    return tot / 3600


def split_by_local_day(iv):
    """{date: hours} for intervals in naive local time, split at midnight."""
    d = defaultdict(float)
    for a, b in interval_union(iv):
        cur = a
        while cur < b:
            nxt = min(b, dt.datetime.combine(cur.date() + dt.timedelta(days=1), dt.time()))
            d[cur.date().isoformat()] += (nxt - cur).total_seconds() / 3600
            cur = nxt
    return d


def labelled_clip_index(animal):
    """Names of labelled seizure clip dirs for this animal (listing only)."""
    idx = {}
    root = os.path.join(LABELLED, f"Data_{animal}_cropped")
    if not os.path.isdir(root):
        return idx, False
    for d in glob.glob(os.path.join(root, "*", "seizure_*")):
        m = re.search(r"seizure_(\d+)_Stage_(\d+)_(\d{8})_(\d{6})$", os.path.basename(d))
        if m:
            t = dt.datetime.strptime(m.group(3) + m.group(4), "%Y%m%d%H%M%S")
            idx[t] = os.path.relpath(d, LABELLED)
    return idx, True


def video_timing(v, edf_by_day):
    """Derived timing for one top-level video record (mutates v)."""
    probs = v["problems"]
    try:
        u = int(v["xml_utc_start"])
    except Exception:
        u = None
    so = None
    try:
        so = int(v["xml_sync_offset"])
    except Exception:
        pass
    utc0 = naive_utc(u) if u is not None else None
    loc0 = parse_dt(v.get("xml_local_iso"))
    fn = parse_dt(v.get("fn_time"))
    dur = v.get("duration_s")
    if utc0 is not None:
        start = utc0 + dt.timedelta(seconds=(so or 0) / 1e7)
        v["start_utc"] = iso(start)
        v["start_local"] = iso(utc_to_local_naive(start))
        if dur:
            end = start + dt.timedelta(seconds=dur)
            v["end_utc"] = iso(end)
            v["end_local"] = iso(utc_to_local_naive(end))
            if fn is not None:
                v["fn_minus_end_utc_s"] = secs(fn, end)
        if fn is not None:
            v["fn_minus_start_utc_s"] = secs(fn, utc0)
            v["wall_end_local"] = iso(utc_to_local_naive(fn))
            if dur:
                # file name = UTC time the file was closed; frames should fill [start, close]
                v["wall_minus_stream_s"] = round(v["fn_minus_start_utc_s"] - dur, 3)
                if v["wall_minus_stream_s"] > WALL_TOL_S:
                    probs.append("frames_missing_vs_wallclock")
                elif v["wall_minus_stream_s"] < -WALL_TOL_S:
                    probs.append("stream_longer_than_wallclock")
        if loc0 is not None:
            v["xml_local_minus_utc_h"] = round((loc0 - utc0).total_seconds() / 3600, 4)
            v["xml_local_minus_tz_s"] = secs(loc0, utc_to_local_naive(utc0))
            if abs(v["xml_local_minus_tz_s"]) > 1:
                probs.append("xml_local_ne_central(utc)")
    if so is not None:
        v["sync_offset_s"] = so / 1e7
        if v.get("trailer_utc_ms") is not None:
            v["sync_eq_trailer_utc_ms"] = (so == v["trailer_utc_ms"] * 10000)
            if not v["sync_eq_trailer_utc_ms"]:
                probs.append("sync_offset_ne_trailer_ms")
    tu, tl = parse_dt(v.get("trailer_utc")), parse_dt(v.get("trailer_local"))
    if tu is not None and utc0 is not None:
        v["trailer_utc_minus_xml_utc_s"] = secs(tu, utc0)
    if tl is not None and loc0 is not None:
        v["trailer_local_minus_xml_local_s"] = secs(tl, loc0)
    if tu is not None and tl is not None:
        v["trailer_local_minus_utc_s"] = secs(tl, tu)
    # camera / XML consistency
    if v.get("xml_camera") and v.get("camera") and v["xml_camera"] != v["camera"]:
        probs.append("xml_camera_ne_filename")
    if v.get("xml_path_name") and v["xml_path_name"] != v["name"]:
        probs.append("xml_path_name_ne_file")
    if v.get("xml_path_day") is not None and v["xml_path_day"] != v["day_folder"]:
        v["xml_path_day_differs"] = True
    ids = [re.sub(r"-\d+$", "", s) for s in (v.get("xml_animal_ids") or "").split("|") if s]
    v["xml_animals"] = "|".join(ids)
    if ids and v["animal"] not in ids:
        probs.append("folder_animal_not_in_xml_ids")
    if v.get("camera") and v["animal"] not in v["camera"].split("-"):
        probs.append("folder_animal_not_in_camera")
    # EDF of the same day folder
    edfs = edf_by_day.get(v["day_folder"], [])
    if edfs and v.get("start_local"):
        e = min(edfs, key=lambda e: abs((parse_dt(e["edf_start"]) - parse_dt(v["start_local"])).total_seconds()))
        es, ee = parse_dt(e["edf_start"]), parse_dt(e["edf_end"])
        v["edf_same_day"] = e["name"]
        v["edf_start"] = e["edf_start"]
        v["start_local_minus_edf_start_s"] = secs(parse_dt(v["start_local"]), es)
        if v.get("end_local"):
            vs, ve = parse_dt(v["start_local"]), parse_dt(v["end_local"])
            ov = max(0.0, (min(ve, ee) - max(vs, es)).total_seconds())
            v["frac_in_same_day_edf"] = round(ov / max(1e-9, (ve - vs).total_seconds()), 4)
            v["end_local_minus_edf_end_s"] = secs(ve, ee)


def finalize_unit(animal_dir, recs, rec_events):
    animal = animal_of_folder(animal_dir)
    for r in recs:
        reclassify(r)
    videos = [r for r in recs if r["kind"] == "video"]
    edfs = [r for r in recs if r["kind"] == "edf"]
    xlsxs = [r for r in recs if r["kind"] == "xlsx"]
    edf_by_day = defaultdict(list)
    for e in edfs:
        if e.get("edf_start") and e.get("edf_end"):
            edf_by_day[e["day_folder"]].append(e)
    for v in videos:
        video_timing(v, edf_by_day)
    # rank within day folder, shared XML starts
    by_day = defaultdict(list)
    for v in videos:
        by_day[v["day_folder"]].append(v)
    for d, vs in by_day.items():
        vs.sort(key=lambda v: v.get("start_utc") or "")
        for i, v in enumerate(vs):
            v["rank_in_day"] = i
    xml_starts = Counter((v.get("camera"), v.get("xml_utc_start")) for v in videos if v.get("xml_utc_start"))
    for v in videos:
        v["xml_start_shared_by"] = xml_starts.get((v.get("camera"), v.get("xml_utc_start")), 0)

    # --- gaps / overlaps between consecutive video files, per camera (unique by name)
    gaps = []
    seen = {}
    dup_within = []
    for v in videos:
        if v["name"] in seen:
            dup_within.append([seen[v["name"]]["relpath"], v["relpath"]])
        else:
            seen[v["name"]] = v
    percam = defaultdict(list)
    for v in seen.values():
        if v.get("start_utc") and v.get("end_utc"):
            percam[v.get("camera")].append(v)
    for cam, vs in percam.items():
        vs.sort(key=lambda v: v["start_utc"])
        for a, b in zip(vs, vs[1:]):
            g = secs(parse_dt(b["start_utc"]), parse_dt(a["end_utc"]))
            gaps.append({"camera": cam, "prev": a["relpath"], "next": b["relpath"], "gap_s": g,
                         "prev_end_local": a["end_local"], "next_start_local": b["start_local"],
                         "same_day_folder": a["day_folder"] == b["day_folder"]})
    within = [g["gap_s"] for g in gaps if g["same_day_folder"]]
    across = [g["gap_s"] for g in gaps if not g["same_day_folder"]]

    def gstats(xs):
        if not xs:
            return {"n": 0}
        return {"n": len(xs), "min_s": min(xs), "median_s": statistics.median(xs), "max_s": max(xs),
                "n_overlap_lt_-1s": sum(x < -1 for x in xs), "n_gt_120s": sum(x > 120 for x in xs),
                "n_gt_1h": sum(x > 3600 for x in xs), "total_gap_h": round(sum(x for x in xs if x > 0) / 3600, 3)}

    # --- EDF gaps / overlaps
    edf_sorted = sorted([e for e in edfs if e.get("edf_start")], key=lambda e: e["edf_start"])
    edf_gaps = []
    for a, b in zip(edf_sorted, edf_sorted[1:]):
        edf_gaps.append({"prev": a["relpath"], "next": b["relpath"],
                         "gap_s": secs(parse_dt(b["edf_start"]), parse_dt(a["edf_end"]))})

    # --- intervals (naive local)
    v_iv = [(parse_dt(v["start_local"]), parse_dt(v["end_local"])) for v in seen.values() if v.get("end_local")]
    # --- EDFs whose span overlaps the videos of a different day folder far more than their own
    vid_iv_by_day = defaultdict(list)
    for v in videos:
        if v.get("end_local"):
            vid_iv_by_day[v["day_folder"]].append((parse_dt(v["start_local"]), parse_dt(v["end_local"])))
    for e in edfs:
        if not e.get("edf_end") or (e.get("edf_hours") or 0) < 0.1:
            continue
        ei = [(parse_dt(e["edf_start"]), parse_dt(e["edf_end"]))]
        ov = {d: intersect_hours(ei, iv) for d, iv in vid_iv_by_day.items()}
        if ov:
            best = max(ov, key=ov.get)
            e["edf_video_overlap_own_h"] = round(ov.get(e["day_folder"], 0.0), 3)
            if best != e["day_folder"] and ov[best] > 0.5 * e["edf_hours"] and ov[best] > 2 * ov.get(e["day_folder"], 0.0):
                e["edf_video_folder_guess"] = best
                e["problems"].append("edf_misfiled_vs_video")
    e_iv = [(parse_dt(e["edf_start"]), parse_dt(e["edf_end"])) for e in edfs if e.get("edf_end")]
    v_day, e_day = split_by_local_day(v_iv), split_by_local_day(e_iv)

    # --- events: coverage by EDF, video, labelled clips
    lab_idx, lab_exists = labelled_clip_index(animal)
    lab_used = set()
    events_out = []
    for r, evs in rec_events:
        if r["kind"] != "xlsx":
            continue
        for ev in evs:
            e = dict(ev)
            e.update({"animal": animal, "animal_folder": animal_dir, "day_folder": r["day_folder"], "xlsx": r["relpath"]})
            st = parse_dt(ev.get("start"))
            if st is not None:
                e_same = [x for x in edf_by_day.get(r["day_folder"], [])]
                e["in_same_day_edf"] = any(parse_dt(x["edf_start"]) <= st < parse_dt(x["edf_end"]) for x in e_same)
                e["in_any_edf"] = any(a <= st < b for a, b in e_iv)
                if not e["in_same_day_edf"]:
                    e["problems"] = e["problems"] + ["start_outside_same_day_edf"]
                wd = [v for v in seen.values() if "frames_missing_vs_wallclock" in v["problems"] and v.get("wall_end_local")
                      and parse_dt(v["start_local"]) <= st < parse_dt(v["wall_end_local"])]
                if wd:
                    e["video_wallclock_defect"] = wd[0]["name"]
                    e["problems"] = e["problems"] + ["in_video_with_missing_wallclock_time"]
                cov = [v for v in seen.values() if v.get("end_local") and parse_dt(v["start_local"]) <= st < parse_dt(v["end_local"])]
                if cov:
                    v = cov[0]
                    e["video"] = v["relpath"]
                    e["video_seek_s"] = secs(st, parse_dt(v["start_local"]))
                    en = parse_dt(ev.get("end"))
                    e["video_covers_end"] = bool(en and en <= parse_dt(v["end_local"]))
                else:
                    e["video"] = None
                key = st.replace(microsecond=0)
                hit = next((lab_idx[k] for k in (key, key - dt.timedelta(seconds=1), key + dt.timedelta(seconds=1)) if k in lab_idx), None)
                e["labelled_clip"] = hit
                if hit:
                    lab_used.add(hit)
            events_out.append(e)

    # --- per day folder
    days = defaultdict(lambda: {"n_mp4": 0, "n_xml": 0, "n_edf": 0, "n_xlsx": 0, "n_other": 0, "video_h": 0.0,
                                "edf_h": 0.0, "n_events": 0, "problems": Counter()})
    for r in recs:
        d = days[r["day_folder"] or "(root)"]
        k = r["kind"]
        if k == "video":
            d["n_mp4"] += 1
            d["video_h"] += r.get("hours") or 0
        elif k == "xml":
            d["n_xml"] += 1
        elif k == "edf":
            d["n_edf"] += 1
            d["edf_h"] += r.get("edf_hours") or 0
        elif k == "xlsx":
            d["n_xlsx"] += 1
            d["n_events"] += r.get("n_events") or 0
        else:
            d["n_other"] += 1
        for p in r["problems"]:
            d["problems"][p] += 1
    day_rows = []
    for name in sorted(days, key=lambda n: (day_folder_dates(n)[0] or dt.date.max, n)):
        d = days[name]
        vs = sorted(by_day.get(name, []), key=lambda v: v.get("start_local") or "")
        es = sorted(edf_by_day.get(name, []), key=lambda e: e["edf_start"])
        row = {"day_folder": name, **{k: v for k, v in d.items() if k != "problems"},
               "video_h": round(d["video_h"], 3), "edf_h": round(d["edf_h"], 3),
               "video_first_start_local": vs[0].get("start_local") if vs else None,
               "video_last_end_local": vs[-1].get("end_local") if vs else None,
               "edf_start": es[0]["edf_start"] if es else None, "edf_end": max(e["edf_end"] for e in es) if es else None,
               "edf_files": [f'{e["name"]} {e["edf_start"]} {e["edf_hours"]}h' for e in es],
               "video_in_edf_h": round(intersect_hours(
                   [(parse_dt(v["start_local"]), parse_dt(v["end_local"])) for v in vs if v.get("end_local")],
                   [(parse_dt(e["edf_start"]), parse_dt(e["edf_end"])) for e in es]), 3),
               "problems": dict(d["problems"])}
        flags = []
        if d["n_mp4"] == 0:
            flags.append("no_video")
        if d["n_edf"] == 0:
            flags.append("no_edf")
        if d["n_edf"] > 1:
            flags.append("multiple_edf")
        if d["n_xlsx"] == 0:
            flags.append("no_xlsx")
        row["flags"] = flags
        day_rows.append(row)

    extras = rn197_extras(recs, videos) if animal_dir == "RN197" else None   # may flag clip_incomplete
    for r in recs:
        r["status"] = severity(r["problems"])
    bad = sorted({r["relpath"] for r in recs if r["status"] == "error"})
    cams = Counter(v.get("camera") for v in videos)
    partners = sorted({a for v in videos for a in (v.get("camera") or "").split("-") if a and a != animal}
                      | {a for v in videos for a in (v.get("xml_animals") or "").split("|") if a and a != animal})
    ev_labels = Counter(e["label"] for e in events_out)
    unit = {
        "animal_folder": animal_dir, "animal": animal, "labelled": animal in LABELLED_ANIMALS,
        "audit_version": AUDIT_VERSION, "complete": True, "n_files": len(recs),
        "files_by_kind": dict(Counter(r["kind"] for r in recs)),
        "bytes_by_kind": {k: sum(r["size"] for r in recs if r["kind"] == k) for k in set(r["kind"] for r in recs)},
        "n_day_folders": len([d for d in days if d != "(root)"]),
        "video_h_sum": round(sum(v.get("hours") or 0 for v in videos), 3),
        "video_h_unique": round(sum(v.get("hours") or 0 for v in seen.values()), 3),
        "edf_h_sum": round(sum(e.get("edf_hours") or 0 for e in edfs), 3),
        "edf_h_union": round(union_hours(e_iv), 3),
        "video_h_union": round(union_hours(v_iv), 3),
        "video_in_edf_h": round(intersect_hours(v_iv, e_iv), 3),
        "video_span_local": [iso(min(a for a, _ in v_iv)), iso(max(b for _, b in v_iv))] if v_iv else None,
        "edf_span_local": [iso(min(a for a, _ in e_iv)), iso(max(b for _, b in e_iv))] if e_iv else None,
        "cameras": dict(cams), "partners": partners,
        "n_events": len(events_out), "event_labels": dict(ev_labels),
        "events_with_video": sum(1 for e in events_out if e.get("video")),
        "events_in_same_day_edf": sum(1 for e in events_out if e.get("in_same_day_edf")),
        "labelled_tree_exists": lab_exists, "labelled_seizure_dirs": len(lab_idx),
        "events_matched_to_labelled_clip": sum(1 for e in events_out if e.get("labelled_clip")),
        "labelled_seizure_dirs_unmatched": sorted(set(lab_idx.values()) - lab_used),
        "gaps_within_day_folder": gstats(within), "gaps_across_day_folders": gstats(across),
        "gaps": gaps, "edf_gaps": edf_gaps, "video_dup_names_within_animal": dup_within,
        "per_day_folder": day_rows,
        "per_calendar_day": [{"date": d, "video_h": round(v_day.get(d, 0), 3), "edf_h": round(e_day.get(d, 0), 3)}
                             for d in sorted(set(v_day) | set(e_day))],
        "problem_counts": dict(Counter(p for r in recs for p in r["problems"])),
        "bad_files": bad,
        "timing_defect_files": sorted(r["relpath"] for r in recs if r["status"] == "timing"),
        "edf_1khz_labels": dict(Counter(e.get("edf_1khz_label") for e in edfs)),
        "events_in_video_with_missing_wallclock_time": sum(1 for e in events_out if e.get("video_wallclock_defect")),
        "labelled_clips_in_video_with_missing_wallclock_time": sum(1 for e in events_out if e.get("video_wallclock_defect") and e.get("labelled_clip")),
        "records": recs, "events": events_out,
    }
    if extras is not None:
        unit["rn197_extras"] = extras
    return unit


def rn197_extras(recs, videos):
    """seizure_clips/: what is there, and does it agree with the raw videos and the labelled set."""
    clip_recs = [r for r in recs if r["kind"].startswith("clip_")]
    per_dir = defaultdict(dict)
    for r in clip_recs:
        parts = r["relpath"].split(os.sep)
        if r["kind"] == "clip_log":
            per_dir[os.sep.join(parts[:3])]["_log"] = r
        else:
            per_dir[os.sep.join(parts[:4])][r["kind"]] = r
    vids = {v["name"]: v for v in videos}
    src_edfs = {(r["day_folder"], r["name"]): r for r in recs if r["kind"] == "edf"}
    logs = {k: v["_log"] for k, v in per_dir.items() if "_log" in v}
    seek_by_folder = {}
    for k, lg in logs.items():
        for row in lg.get("clip_log_rows", []):
            seek_by_folder[os.path.join(k, row.get("Folder", ""))] = row
    rows = []
    for d, fs in sorted(per_dir.items()):
        if "_log" in fs and len(fs) == 1:
            continue
        info = (fs.get("clip_info") or {}).get("info", {})
        row = {"clip_dir": d, "has_video": "clip_video" in fs, "has_edf": "clip_edf" in fs, "has_info": "clip_info" in fs,
               "video_size": fs.get("clip_video", {}).get("size"), "edf_size": fs.get("clip_edf", {}).get("size"),
               "info_size": fs.get("clip_info", {}).get("size"),
               "video_dur_s": fs.get("clip_video", {}).get("duration_s"),
               "video_wh": f'{fs.get("clip_video", {}).get("width")}x{fs.get("clip_video", {}).get("height")}',
               "edf_hours": fs.get("clip_edf", {}).get("edf_hours"),
               "edf_problems": fs.get("clip_edf", {}).get("problems"),
               "video_problems": fs.get("clip_video", {}).get("problems"),
               "edf_start": fs.get("clip_edf", {}).get("edf_start"),
               "edf_ecg_phys": fs.get("clip_edf", {}).get("edf_ecg_phys"),
               "edf_ecg_std_dig_first": fs.get("clip_edf", {}).get("edf_first_rec_ecg_std_dig"),
               "edf_ecg_std_dig_last": fs.get("clip_edf", {}).get("edf_last_rec_ecg_std_dig"),
               "label": info.get("Label"), "seizure_start": info.get("Seizure start"),
               "clip_start": info.get("Clip start"),
               "src_video": info.get("Video file"), "src_edf": info.get("EDF file"),
               "eeg_note": info.get("EEG channels")}
        if row["edf_start"] and row["clip_start"]:
            row["clip_edf_start_minus_clip_start_s"] = secs(parse_dt(row["edf_start"]), parse_dt(row["clip_start"]))
        srcedf = src_edfs.get((d.split(os.sep)[1], row["src_edf"]))
        if row["edf_start"] and srcedf and srcedf.get("edf_start"):
            row["clip_edf_start_minus_source_edf_start_s"] = secs(parse_dt(row["edf_start"]), parse_dt(srcedf["edf_start"]))
        lab = os.path.join(LABELLED, "Data_RN197_cropped", d.split(os.sep)[1], d.split(os.sep)[-1])
        row["in_labelled_tree"] = os.path.isdir(lab)
        lg = seek_by_folder.get(d)
        if lg:
            row["log_video_status"], row["log_eeg_status"] = lg.get("Video Status"), lg.get("EEG Status")
            try:
                hh, mm, ss = [int(x) for x in lg["Video Seek"].split(":")]
                seek = hh * 3600 + mm * 60 + ss
                sz = parse_dt(lg["Seizure Start"])
                src = vids.get(row["src_video"] or "")
                # cut_seizure_clips.py logs "Video Seek" = seizure start - video start (before the
                # pre-buffer is subtracted), truncated to whole seconds
                row["seek_s"] = seek
                if src and src.get("xml_local_iso"):
                    implied = sz - dt.timedelta(seconds=seek)
                    row["seek_implied_start_minus_xml_local_s"] = secs(implied, parse_dt(src["xml_local_iso"]))
            except Exception:
                pass
        miss = [k for k, has in (("video.mp4", row["has_video"]), ("eeg.edf", row["has_edf"]), ("info.txt", row["has_info"])) if not has]
        zero = [k for k, s in (("video.mp4", row["video_size"]), ("eeg.edf", row["edf_size"]), ("info.txt", row["info_size"])) if s == 0]
        row["missing"], row["zero_byte"] = miss, zero
        rows.append(row)
    for r in clip_recs:          # mark incomplete clip dirs on the file records themselves
        d = os.sep.join(r["relpath"].split(os.sep)[:4])
        rr = next((x for x in rows if x["clip_dir"] == d), None)
        if rr and (rr["missing"] or rr["zero_byte"]) and "clip_incomplete" not in r["problems"]:   # warning
            r["problems"].append("clip_incomplete")
    return {"n_clip_dirs": len(rows), "n_logs": len(logs),
            "day_folders_with_clips": sorted({r["clip_dir"].split(os.sep)[1] for r in rows}),
            "day_folders_without_log": sorted({r["clip_dir"].split(os.sep)[1] for r in rows} -
                                              {k.split(os.sep)[1] for k in logs}),
            "missing_counts": dict(Counter(m for r in rows for m in r["missing"])),
            "zero_byte": [(r["clip_dir"], r["zero_byte"]) for r in rows if r["zero_byte"]],
            "in_labelled_tree": sum(r["in_labelled_tree"] for r in rows),
            "labels": dict(Counter(r["label"] for r in rows)),
            "eeg_status": dict(Counter((r.get("log_eeg_status") or "no-log")[:14] for r in rows)),
            "clip_edf_start_minus_clip_start_s": dist([r.get("clip_edf_start_minus_clip_start_s") for r in rows]),
            "clip_edf_start_minus_source_edf_start_s": dist([r.get("clip_edf_start_minus_source_edf_start_s") for r in rows]),
            "clip_edf_ecg_phys": dict(Counter(r.get("edf_ecg_phys") for r in rows if r["has_edf"])),
            "clip_edf_ecg_std_dig_first": dist([r.get("edf_ecg_std_dig_first") for r in rows]),
            "clip_edf_ecg_std_dig_last": dist([r.get("edf_ecg_std_dig_last") for r in rows]),
            "clips": rows}


# ----------------------------------------------------------------------------- merge
FILE_COLS = [
    "animal_folder", "animal", "day_folder", "relpath", "name", "kind", "ext", "size", "mtime", "status", "problems",
    # video
    "camera", "fn_time", "codec", "pix_fmt", "width", "height", "r_fps", "avg_fps", "nb_frames", "stream_duration",
    "fmt_duration", "duration_s", "hours", "bit_rate", "stream_types", "boxes", "moov_offset", "trailing_bytes",
    "cv2_opened", "cv2_frames", "cv2_fps", "cv2_w", "cv2_h", "cv2_first_ok",
    "first_ok", "first_pts_s", "first_mean", "first_std", "mid_ok", "mid_pts_s", "mid_mean", "mid_std",
    "last_ok", "last_pts_s", "last_mean", "last_std", "last_frame_gap_s", "tail_frames_decoded", "n_ffmpeg_errors",
    "ffmpeg_errors",
    "ffprobe_error", "hash_head", "hash_tail",
    "trailer_present", "trailer_text", "trailer_local", "trailer_local_ms", "trailer_utc", "trailer_utc_ms", "trailer_utc_x",
    "xml_present", "xml_utc_start", "xml_utc_iso", "xml_local_raw", "xml_local_iso", "xml_sync_offset", "xml_camera",
    "xml_animal_ids", "xml_animals", "xml_version", "xml_duration", "xml_native_w", "xml_native_h", "xml_fullpath",
    "xml_path_prefix", "xml_path_day", "xml_path_day_differs", "xml_other_tags", "xml_error",
    "start_utc", "end_utc", "start_local", "end_local", "wall_end_local", "wall_minus_stream_s",
    "fn_minus_end_utc_s", "fn_minus_start_utc_s",
    "xml_local_minus_utc_h", "xml_local_minus_tz_s", "sync_offset_s", "sync_eq_trailer_utc_ms",
    "trailer_utc_minus_xml_utc_s", "trailer_local_minus_xml_local_s", "trailer_local_minus_utc_s",
    "rank_in_day", "xml_start_shared_by", "edf_same_day", "edf_start", "start_local_minus_edf_start_s",
    "end_local_minus_edf_end_s", "frac_in_same_day_edf",
    # edf
    "edf_version", "edf_patient", "edf_recording", "edf_startdate", "edf_starttime", "edf_end", "edf_header_bytes",
    "edf_reserved", "edf_nrec", "edf_rec_dur", "edf_ns", "edf_labels", "edf_nsamp", "edf_fs", "edf_dims",
    "edf_ecg_label", "edf_ecg_fs", "edf_ecg_phys", "edf_ecg_dig", "edf_record_bytes", "edf_expected_size",
    "edf_size_diff", "edf_records_in_file", "edf_partial_bytes", "edf_hours", "edf_hours_header", "edf_name_date",
    "edf_first_rec_zero_frac", "edf_last_rec_zero_frac", "edf_first_rec_ecg_std_dig", "edf_last_rec_ecg_std_dig",
    "edf_1khz_label", "edf_video_overlap_own_h", "edf_video_folder_guess", "edf_error",
    # xlsx
    "xlsx_sheets", "xlsx_header", "xlsx_units", "xlsx_rows_nonempty", "n_events", "event_labels", "xlsx_error",
    # other
    "identified_as", "n_rows", "info", "magic_hex", "exception",
]
EVENT_COLS = ["animal", "animal_folder", "day_folder", "xlsx", "sheet", "row", "label", "duration_s", "start", "end",
              "start_type", "end_type", "spikes", "eeg_ave", "end_minus_start_minus_dur_s", "in_same_day_edf",
              "in_any_edf", "video", "video_seek_s", "video_covers_end", "video_wallclock_defect", "labelled_clip",
              "problems"]


def _cell(v):
    if isinstance(v, list):
        return "|".join(map(str, v))
    if isinstance(v, dict):
        return json.dumps(v, default=str)
    return "" if v is None else v


def write_csv(path, cols, rows):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({c: _cell(r.get(c)) for c in cols})
    write_atomic(path, buf.getvalue())


def q(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


def dist(xs, nd=3):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "n=0"
    return (f"n={len(xs)} min={round(min(xs), nd)} p5={round(q(xs, .05), nd)} median={round(q(xs, .5), nd)} "
            f"p95={round(q(xs, .95), nd)} max={round(max(xs), nd)}")


def run_merge(allow_partial=False, refinalize=False):
    animals = list_animals()
    if refinalize:          # re-run the per-animal step from the per-file caches (no file is re-probed)
        for a in animals:
            run_unit(a, 1, finalize_only=True)
    units, missing = {}, []
    for a in animals:
        p = out_path("units", f"{a}.json")
        if not os.path.exists(p):
            missing.append(a)
            continue
        with open(p) as fh:
            u = json.load(fh)
        if u.get("audit_version") != AUDIT_VERSION:
            missing.append(a + " (stale version)")
            continue
        units[a] = u
    if missing and not allow_partial:
        raise SystemExit(f"units missing: {missing}  (rerun them, or pass --allow-partial)")
    recs = [r for u in units.values() for r in u["records"]]
    events = [e for u in units.values() for e in u["events"]]
    for r in recs:
        r["status"] = severity(r["problems"])
    # the raw tree listing, to prove every file was covered
    listed = sum(len(walk_files(a)) for a in animals)

    # ---------------- duplicates
    dup_rows, gid = [], 0
    by_name = defaultdict(list)
    for r in recs:
        if r["kind"] in ("video", "xml", "edf", "xlsx"):
            by_name[(r["kind"], r["name"])].append(r)
    content = defaultdict(list)
    for r in recs:
        if r.get("hash_head") and r["size"] > 0:
            content[(r["size"], r["hash_head"], r["hash_tail"])].append(r)
    video_saving = 0
    name_groups_identical = name_groups_conflict = 0
    for (kind, name), rs in sorted(by_name.items()):
        if len(rs) < 2:
            continue
        gid += 1
        sigs = {(r["size"], r.get("hash_head"), r.get("hash_tail")) for r in rs}
        identical = len(sigs) == 1
        sav = rs[0]["size"] * (len(rs) - 1) if identical else 0
        if kind == "video":
            video_saving += sav
            if identical:
                name_groups_identical += 1
            else:
                name_groups_conflict += 1
        dup_rows.append({"group": gid, "match": "name+size+hash" if identical else "name_only_CONTENT_DIFFERS",
                         "kind": kind, "name": name, "n_copies": len(rs), "size": rs[0]["size"],
                         "sizes": "|".join(str(r["size"]) for r in rs), "identical": identical,
                         "animals": "|".join(r["animal_folder"] for r in rs),
                         "day_folders": "|".join(r["day_folder"] for r in rs),
                         "same_day_folder_name": len({r["day_folder"] for r in rs}) == 1,
                         "paths": "|".join(r["relpath"] for r in rs), "saving_bytes": sav})
    # same content under different names (not caught above)
    for key, rs in content.items():
        names = {r["name"] for r in rs}
        if len(rs) > 1 and len(names) > 1:
            gid += 1
            dup_rows.append({"group": gid, "match": "content_only_DIFFERENT_NAMES", "kind": rs[0]["kind"],
                             "name": "|".join(sorted(names)), "n_copies": len(rs), "size": key[0],
                             "sizes": "|".join(str(r["size"]) for r in rs), "identical": True,
                             "animals": "|".join(r["animal_folder"] for r in rs),
                             "day_folders": "|".join(r["day_folder"] for r in rs),
                             "paths": "|".join(r["relpath"] for r in rs), "saving_bytes": 0})
    write_csv(out_path("duplicates.csv"),
              ["group", "match", "kind", "name", "n_copies", "size", "sizes", "identical", "animals", "day_folders",
               "same_day_folder_name", "paths", "saving_bytes"], dup_rows)
    # partner asymmetry: camera files present under one partner and not the other
    cam_files = defaultdict(lambda: defaultdict(set))
    for r in recs:
        if r["kind"] == "video" and r.get("camera"):
            cam_files[r["camera"]][r["animal"]].add(r["name"])
    asym = []
    for cam, per in sorted(cam_files.items()):
        members = [a for a in cam.split("-")]
        present = {a: per.get(a, set()) for a in members}
        allnames = set().union(*present.values()) if present else set()
        for a in members:
            folder_exists = any(u["animal"] == a for u in units.values())
            miss = sorted(allnames - present[a])
            if miss:
                asym.append({"camera": cam, "animal": a, "animal_folder_exists": folder_exists,
                             "n_missing": len(miss), "of": len(allnames), "examples": miss[:5]})

    # ---------------- files.csv, events.csv
    write_csv(out_path("files.csv"), FILE_COLS, sorted(recs, key=lambda r: r["relpath"]))
    write_csv(out_path("events.csv"), EVENT_COLS, events)

    # ---------------- per_animal.json
    per_animal = {}
    for a, u in units.items():
        d = {k: v for k, v in u.items() if k not in ("records", "events")}
        d["duplicate_video_files_here"] = sum(1 for g in dup_rows if g["kind"] == "video" and a in g["animals"].split("|"))
        d["partner_asymmetry"] = [x for x in asym if x["animal"] == d["animal"]]
        per_animal[a] = d
    write_atomic(out_path("per_animal.json"), json.dumps(per_animal, indent=1, default=str))

    # ---------------- summary
    L = []
    P = L.append
    P(f"RAW AUDIT  {dt.datetime.now().isoformat(sep=' ', timespec='seconds')}   audit_version={AUDIT_VERSION}")
    P(f"raw root {RAW} (read-only)   outputs {OUT}")
    P(f"animal folders: {len(animals)}   units merged: {len(units)}   missing units: {missing or 'none'}")
    P(f"files listed now: {listed}   files audited: {len(recs)}   "
      f"{'ALL COVERED' if listed == len(recs) else 'MISMATCH -- rerun units'}")
    kinds = Counter(r["kind"] for r in recs)
    P("files by kind: " + ", ".join(f"{k}={n} ({sum(r['size'] for r in recs if r['kind'] == k) / 1e9:.1f} GB)" for k, n in kinds.most_common()))
    P(f"total bytes: {sum(r['size'] for r in recs) / 1e12:.3f} TB")
    st = Counter(r["status"] for r in recs)
    P(f"status: {dict(st)}")
    P("")
    P("PER ANIMAL  video_h = unique video hours in the folder (stream duration); edf_h = union of EDF spans;")
    P("            v&e = hours with both; ev = xlsx event rows; ev_vid = events inside a video; ev_lab = events matched")
    P("            to a labelled seizure clip dir; err/tim = files with integrity errors / timing defects")
    P(f"{'folder':<11}{'lab':<4}{'days':>5}{'mp4':>5}{'edf':>5}{'video_h':>9}{'edf_h':>9}{'v&e_h':>8}{'ev':>6}{'ev_vid':>7}"
      f"{'ev_lab':>7}{'err':>5}{'tim':>5}  1kHz EDF label(s)  cameras / partners")
    for a, u in units.items():
        P(f"{a:<11}{'Y' if u['labelled'] else 'n':<4}{u['n_day_folders']:>5}{u['files_by_kind'].get('video', 0):>5}"
          f"{u['files_by_kind'].get('edf', 0):>5}{u['video_h_unique']:>9.1f}{u['edf_h_union']:>9.1f}{u['video_in_edf_h']:>8.1f}"
          f"{u['n_events']:>6}{u['events_with_video']:>7}{u['events_matched_to_labelled_clip']:>7}{len(u['bad_files']):>5}"
          f"{len(u['timing_defect_files']):>5}  {','.join(f'{k}:{n}' for k, n in u['edf_1khz_labels'].items())}  "
          f"{','.join(f'{c}:{n}' for c, n in u['cameras'].items())} / {','.join(u['partners'])}")
    tot = lambda k: sum(u[k] for u in units.values())
    P(f"{'TOTAL':<15}{'':>5}{sum(u['files_by_kind'].get('video', 0) for u in units.values()):>5}"
      f"{sum(u['files_by_kind'].get('edf', 0) for u in units.values()):>5}{tot('video_h_unique'):>9.1f}{tot('edf_h_union'):>9.1f}"
      f"{tot('video_in_edf_h'):>8.1f}{tot('n_events'):>6}{tot('events_with_video'):>7}{tot('events_matched_to_labelled_clip'):>7}")
    uniq_v = {}
    for r in recs:
        if r["kind"] == "video":
            uniq_v.setdefault(r["name"], r)
    P(f"unique camera files: {len(uniq_v)} ({sum(r.get('hours') or 0 for r in uniq_v.values()):.1f} h, "
      f"{sum(r['size'] for r in uniq_v.values()) / 1e12:.3f} TB) out of {sum(1 for r in recs if r['kind'] == 'video')} stored")
    P("")
    P("PROBLEM COUNTS  (error = unreadable/incomplete; timing = video time != wall-clock time; compat = names/labels a")
    P("                 consumer would mis-handle; warn = informational)")
    pc = Counter(p for r in recs for p in r["problems"])
    for p, n in sorted(pc.items(), key=lambda kv: (["error", "timing", "compat", "warn"].index(severity([kv[0]])), -kv[1])):
        P(f"  {severity([p]):<7}{p:<36}{n}")
    P("")
    P("INTEGRITY ERRORS (every file)")
    for r in sorted(recs, key=lambda r: r["relpath"]):
        if r["status"] == "error":
            P(f"  {r['relpath']}  size={r['size']}  [{';'.join(r['problems'])}]  "
              + f"{r.get('ffmpeg_errors') or r.get('ffprobe_error') or r.get('exception') or ''}".replace(chr(10), ' ')[:160]
              + (f" records_in_file={r.get('edf_records_in_file')} partial_bytes={r.get('edf_partial_bytes')} usable_h={r.get('edf_hours')}" if r['kind'] in ('edf', 'clip_edf') and r['size'] else ""))
    P("")
    P(f"TIMING DEFECTS: files whose frames do not fill the span from XML start to the file-name close time (> {WALL_TOL_S:.0f} s)")
    tv = {}
    for r in recs:
        if r["kind"] == "video" and r["status"] == "timing":
            tv.setdefault(r["name"], []).append(r["animal_folder"])
    by_ts = defaultdict(list)
    for name, fol in tv.items():
        r = uniq_v[name]
        by_ts[(r.get("xml_local_iso"), r.get("fn_time"), round((r.get("wall_minus_stream_s") or 0) / 60, 1))].append(f"{r['camera']}")
    for (st, fn, miss), cams in sorted(by_ts.items()):
        r0 = next(uniq_v[n] for n in tv if uniq_v[n].get("xml_local_iso") == st and uniq_v[n].get("fn_time") == fn)
        P(f"  start {st} local, closed {fn} UTC: frames cover {r0.get('duration_s', 0) / 3600:.3f} h, "
          f"{miss:.1f} min of wall time missing; cameras {', '.join(sorted(cams))}")
    P(f"  -> {len(tv)} unique files, {sum(len(v) for v in tv.values())} stored copies; events inside such a file's wall span: "
      f"{sum(1 for e in events if e.get('video_wallclock_defect'))}, of which matched to labelled clips: "
      f"{sum(1 for e in events if e.get('video_wallclock_defect') and e.get('labelled_clip'))}")
    P("")
    P("COMPAT")
    cam_ren = Counter((r["camera"], r.get("xml_camera"), r.get("xml_animal_ids")) for r in recs
                      if r["kind"] == "video" and "xml_camera_ne_filename" in r["problems"])
    for (c, xc, ids), n in cam_ren.items():
        P(f"  file name camera {c} but XML DSI_camera_name {xc} and Dsi_VideoFullPath name {xc}.*; XML animal ids {ids}: {n} stored files")
    E_ = [r for r in recs if r["kind"] == "edf"]
    for lab, n in Counter(r.get("edf_1khz_label") for r in E_).most_common():
        an = Counter(r["animal_folder"] for r in E_ if r.get("edf_1khz_label") == lab)
        P(f"  EDF 1 kHz channel label {lab!r}: {n} files  {dict(an) if len(an) < 6 else str(len(an)) + ' animals'}")
    P("  (the parent pipeline reads the first channel whose label contains 'ECG'; 'EEG'-labelled files have none)")
    mis = [r for r in E_ if "edf_misfiled_vs_video" in r["problems"]]
    for key, rs in sorted(Counter((r["day_folder"], r.get("edf_video_folder_guess")) for r in mis).items()):
        P(f"  EDF in '{key[0]}' covers the videos of '{key[1]}': {rs} animals")
    P("")
    P("COVERAGE: day folders where video hours and EDF hours differ by > 1 h (grouped over animals)")
    cov = defaultdict(list)
    for a, u in units.items():
        for d in u["per_day_folder"]:
            if abs(d["video_h"] - d["edf_h"]) > 1 or d["flags"]:
                cov[(d["day_folder"], round(d["video_h"], 1), round(d["edf_h"], 2), tuple(d["flags"]))].append(a)
    for (day, vh, eh, fl), an in sorted(cov.items(), key=lambda kv: (day_folder_dates(kv[0][0])[0] or dt.date.max, kv[0][0])):
        P(f"  {day:<30} video {vh:>6.1f} h  EDF {eh:>7.2f} h  {list(fl) if fl else ''}  {', '.join(an)}")
    short = [r for r in E_ if (r.get("edf_hours") or 0) < 1]
    P(f"  EDF files shorter than 1 h: {len(short)} ({dict(Counter(r['animal_folder'] for r in short))})")
    P("")
    P("DUPLICATES")
    vids = [g for g in dup_rows if g["kind"] == "video"]
    P(f"  video file names stored more than once: {len(vids)} groups "
      f"({name_groups_identical} byte-identical by size+head/tail sha1, {name_groups_conflict} same name but different content)")
    P(f"  copies beyond the first: {sum(g['n_copies'] - 1 for g in vids)};  de-duplication would save "
      f"{video_saving / 1e9:.1f} GB of {sum(r['size'] for r in recs if r['kind'] == 'video') / 1e9:.1f} GB video")
    P(f"  groups by animal pair: " + ", ".join(f"{k}:{n}" for k, n in Counter(g['animals'] for g in vids).most_common()))
    for kind in ("edf", "xlsx", "xml"):
        gs = [g for g in dup_rows if g["kind"] == kind]
        P(f"  {kind} names stored more than once: {len(gs)} groups, identical {sum(g['identical'] for g in gs)}")
    cg = [g for g in dup_rows if g["match"].startswith("content_only")]
    P(f"  identical content under different names: {len(cg)} groups")
    for g in cg[:20]:
        P(f"    {g['kind']} {g['paths']}")
    diff_day = [g for g in vids if not g.get("same_day_folder_name")]
    P(f"  video duplicates filed under different day-folder names: {len(diff_day)}")
    for g in diff_day[:10]:
        P(f"    {g['paths']}")
    P(f"  partner asymmetry (camera file under one partner's folder only): {len(asym)} (camera, animal) pairs")
    for x in asym:
        P(f"    {x['camera']}: {x['animal']} lacks {x['n_missing']}/{x['of']} (folder exists={x['animal_folder_exists']}) e.g. {x['examples'][:2]}")
    P("")
    P("TIMING (top-level videos)")
    V = [r for r in recs if r["kind"] == "video"]
    P(f"  wall span (file-name close time - XML UTC start) - stream duration [s]: {dist([v.get('wall_minus_stream_s') for v in V])}")
    P(f"  fn_time - (xml_utc + sync_offset/1e7 + stream duration)  [s]: {dist([v.get('fn_minus_end_utc_s') for v in V])}")
    P(f"  fn_time - xml_utc (both read as UTC) [h]: {dist([(v.get('fn_minus_start_utc_s') or 0) / 3600 if v.get('fn_minus_start_utc_s') is not None else None for v in V])}")
    lm = Counter(v.get("xml_local_minus_utc_h") for v in V)
    P(f"  xml_local - xml_utc [h]: {dict(lm)}")
    P(f"  xml_local - America/Chicago(xml_utc) [s]: {dist([v.get('xml_local_minus_tz_s') for v in V])}")
    dst = Counter((v.get("xml_local_minus_utc_h"), (v.get("start_local") or "")[:10] >= "2023-11-05") for v in V)
    P(f"  (offset, after 2023-11-05 DST end) counts: {dict(dst)}")
    P(f"  DSI_sync_offset [1e-7 s units]: {dist([v.get('sync_offset_s') for v in V], 4)} s")
    eq = Counter(v.get("sync_eq_trailer_utc_ms") for v in V)
    P(f"  sync_offset == Noldus-trailer UTC milliseconds * 1e4: {dict(eq)}")
    P(f"  trailer_utc - xml_utc [s]: {dist([v.get('trailer_utc_minus_xml_utc_s') for v in V])}")
    P(f"  trailer_local - trailer_utc [s]: {dist([v.get('trailer_local_minus_utc_s') for v in V])}")
    for v in V:
        x = v.get("trailer_local_minus_utc_s")
        if x is not None and min(abs(x + 18000), abs(x + 21600)) > 1:
            P(f"    trailer local-UTC outlier: {v['relpath']}  {x} s  (xml local-utc {v.get('xml_local_minus_utc_h')} h)")
    P(f"  trailer UTC 'x' byte values (meaning unknown): {dict(Counter(v.get('trailer_utc_x') for v in V))}")
    P(f"  XML UTC start shared by >1 file of the same camera: {sum(1 for v in V if (v.get('xml_start_shared_by') or 0) > 1)} files")
    first = [v for v in V if v.get("rank_in_day") == 0]
    P(f"  first video of a day folder: start_local - EDF start [s]: {dist([v.get('start_local_minus_edf_start_s') for v in first])}")
    for v in sorted(first, key=lambda v: v["relpath"]):
        x = v.get("start_local_minus_edf_start_s")
        if x is not None and abs(x - 18.5) > 5:
            P(f"    outlier: {v['relpath']}  video start - EDF start = {x} s  (EDF {v.get('edf_same_day')} {v.get('edf_start')})")
    P(f"  later videos: start_local - EDF start [h]: {dist([(v.get('start_local_minus_edf_start_s') or 0) / 3600 for v in V if (v.get('rank_in_day') or 0) > 0 and v.get('start_local_minus_edf_start_s') is not None])}")
    P(f"  fraction of each video inside its day folder's EDF: {dist([v.get('frac_in_same_day_edf') for v in V])}")
    lastv = {}
    for v in V:
        k = (v["animal_folder"], v["day_folder"])
        if v.get("end_local") and (k not in lastv or v["end_local"] > lastv[k]["end_local"]):
            lastv[k] = v
    P(f"  last video of a day folder: end_local - EDF end [s]: {dist([v.get('end_local_minus_edf_end_s') for v in lastv.values()])}")
    allg = [g["gap_s"] for u in units.values() for g in u["gaps"] if g["same_day_folder"]]
    accg = [g["gap_s"] for u in units.values() for g in u["gaps"] if not g["same_day_folder"]]
    P(f"  gap between consecutive files of one camera, same day folder [s]: {dist(allg)}")
    P(f"  gap across day folders [s]: {dist(accg)}")
    P(f"  video duration [h]: {dist([v.get('hours') for v in V])}")
    P(f"  XML Duration field values: {dict(Counter(v.get('xml_duration') for v in V))};  NativeSize: "
      f"{dict(Counter((v.get('xml_native_w'), v.get('xml_native_h')) for v in V))}")
    P(f"  XML path prefixes: {dict(Counter(v.get('xml_path_prefix') for v in V))}")
    P(f"  XML path day folder differs from the folder it is stored in: {sum(1 for v in V if v.get('xml_path_day_differs'))}")
    P(f"  DSI versions: {dict(Counter(v.get('xml_version') for v in V))}")
    P(f"  Noldus trailer text: {dict(Counter(v.get('trailer_text') for v in V))}")
    E = [r for r in recs if r["kind"] == "edf"]
    P("")
    P("EDF")
    P(f"  channel sets: {dict(Counter(r.get('edf_labels') for r in E))}")
    P(f"  samples/record: {dict(Counter(r.get('edf_nsamp') for r in E))}   record dur: {dict(Counter(r.get('edf_rec_dur') for r in E))}")
    P(f"  ECG physical range: {dict(Counter(r.get('edf_ecg_phys') for r in E))}")
    P(f"  size - expected [bytes]: {dict(Counter(r.get('edf_size_diff') for r in E))}")
    P(f"  EDF hours: {dist([r.get('edf_hours') for r in E])}")
    P(f"  patient field == folder animal: {dict(Counter((r.get('edf_patient') or '').split('-')[0] == r['animal'] for r in E))}")
    eg = [g["gap_s"] for u in units.values() for g in u["edf_gaps"]]
    P(f"  gap between consecutive EDFs of an animal [s]: {dist(eg)}   overlaps(<-1s): {sum(1 for x in eg if x < -1)}")
    P("")
    P("EVENTS (xlsx)")
    P(f"  xlsx files: {len([r for r in recs if r['kind'] == 'xlsx'])}, with >=1 event: {sum(1 for r in recs if r['kind'] == 'xlsx' and r.get('n_events'))}")
    P(f"  headers: {dict(Counter(r.get('xlsx_header') for r in recs if r['kind'] == 'xlsx'))}")
    P(f"  events: {len(events)}   label vocabulary: {dict(Counter(e['label'] for e in events).most_common())}")
    P(f"  start cell types: {dict(Counter(e['start_type'] for e in events))}")
    P(f"  event problems: {dict(Counter(p for e in events for p in e['problems']))}")
    P(f"  covered by a video: {sum(1 for e in events if e.get('video'))}/{len(events)};  in same-day EDF: "
      f"{sum(1 for e in events if e.get('in_same_day_edf'))};  matched to a labelled seizure clip dir: "
      f"{sum(1 for e in events if e.get('labelled_clip'))}")
    P(f"  labelled seizure dirs with no matching raw event: "
      f"{sum(len(u['labelled_seizure_dirs_unmatched']) for u in units.values())} of {sum(u['labelled_seizure_dirs'] for u in units.values())}")
    if "RN197" in units and "rn197_extras" in units["RN197"]:
        x = units["RN197"]["rn197_extras"]
        P("")
        P("RN197 EXTRAS (seizure_clips/)")
        P(f"  clip dirs: {x['n_clip_dirs']} in {len(x['day_folders_with_clips'])} day folders; _clip_log.csv: {x['n_logs']}; "
          f"day folders without a log: {x['day_folders_without_log']}")
        P(f"  missing files: {x['missing_counts']};  zero-byte: {x['zero_byte']}")
        P(f"  same dir name exists in the labelled tree: {x['in_labelled_tree']}/{x['n_clip_dirs']}")
        P(f"  labels: {x['labels']};  log EEG status: {x['eeg_status']}")
        P(f"  clip eeg.edf header start - info.txt clip start [s]: {x['clip_edf_start_minus_clip_start_s']}")
        P(f"  clip eeg.edf header start - start of the source EDF named in info.txt [s]: {x['clip_edf_start_minus_source_edf_start_s']}")
        P(f"  clip eeg.edf ECG physical range: {len(x['clip_edf_ecg_phys'])} distinct, e.g. {list(x['clip_edf_ecg_phys'])[:3]}")
        P(f"  clip eeg.edf ECG std in digital units, first / last 1-s record: {x['clip_edf_ecg_std_dig_first']} / {x['clip_edf_ecg_std_dig_last']}")
        P(f"  seek-implied video start - XML local start [s]: {dist([c.get('seek_implied_start_minus_xml_local_s') for c in x['clips']])}")
        P(f"  clip video duration [s]: {dist([c.get('video_dur_s') for c in x['clips']])};  sizes WxH: {dict(Counter(c['video_wh'] for c in x['clips']))}")
    other = [r for r in recs if r["kind"] in ("other", "clip_other")]
    P(f"  unclassified files elsewhere: {len(other)} " + "; ".join(r["relpath"] for r in other[:20]))
    P("")
    P("NOT CHECKED")
    for s in NOT_CHECKED:
        P(f"  - {s}")
    write_atomic(out_path("audit_summary.txt"), "\n".join(L) + "\n")
    print("\n".join(L))


NOT_CHECKED = [
    "video content beyond the decoded points per file (first frame, one mid-file frame, and the last ~10 s "
    "decoded from the preceding keyframe); corruption or frozen/black stretches between them would not be seen",
    "OpenCV: cv2 was not used (AUDIT_CV2=0); decodability was tested with ffmpeg/ffprobe (libavcodec), "
    "while the parent pipeline decodes with cv2's own bundled FFmpeg build",
    "frame count by full decode: nb_frames is the container's index count, not a decoded count",
    "whether each video shows the animals its name says (no frame was inspected for identity or crop)",
    "EDF sample data beyond the first and last complete record (no flat-line / dropout / clipping scan, "
    "no check that the 'ECG' channel is EEG-like)",
    "the physical clock accuracy of any timestamp: relationships are between recorded fields only",
    "whether EDF start times are local wall time (inferred from agreement with the XML local start, not documented)",
    "full-file hashes: duplicates are matched on name + size + sha1 of the first and last 1 MiB",
    "xlsx formulas, formatting, hidden sheets' meaning; only cell values were read",
    "whether seizure events are correct or complete (no re-scoring)",
]


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("list")
    u = sp.add_parser("unit")
    u.add_argument("--animal", help="animal folder name")
    u.add_argument("--index", type=int, help="index into the sorted folder list (SLURM array)")
    u.add_argument("--workers", type=int, default=6)
    m = sp.add_parser("merge")
    m.add_argument("--allow-partial", action="store_true")
    m.add_argument("--refinalize", action="store_true", help="rebuild units/<animal>.json from the per-file caches first")
    a = ap.parse_args()
    if _under(OUT, RAW):
        raise SystemExit("OUT is under RAW; refusing")
    os.makedirs(OUT, exist_ok=True)
    if a.cmd == "list":
        for i, x in enumerate(list_animals()):
            print(i, x)
    elif a.cmd == "unit":
        animals = list_animals()
        animal = a.animal if a.animal else animals[a.index]
        if animal not in animals:
            raise SystemExit(f"no such animal folder: {animal}")
        run_unit(animal, a.workers)
    elif a.cmd == "merge":
        run_merge(a.allow_partial, a.refinalize)


if __name__ == "__main__":
    main()
