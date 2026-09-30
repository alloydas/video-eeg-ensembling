#!/usr/bin/env python3
"""The EEG-Gated Racine Grader on the continuous recordings: flag rates at the fixed operating point.

Implements the pre-registration rawdata/scan_prereg.md (its sha256 goes into every stage's run.json). Section
numbers below (§n) refer to it. Nothing is retrained, no gate is refitted on scan data, no threshold is moved.

Animals: the three with no labelled seizures (RN201, RN203, RN243) and their camera partners (RN199, RN204,
RN242). Grader: x3dfix X3D-M epoch 12 (ttg_vsubj/x3dfix_dual_s{1,2,3}_fold{f}/last.pt) + TCN detector epoch 30
(ttg_eeg/subject/tcn_bin_fold{f}_s{1,2,3}/final.pt) + the EGRG gate (grader/eeg/joint_gate.py).

Sub-commands (stage in the pre-registration):
  tables      S0  camera files (name+size checked against the partner's copy), ffprobe packets, EDF placement,
                  the per-animal minute table with every exclusion reason that needs no decoding (§3)
  gates       S0  reproduction gate (the step-1 grid, per pair, exactly), G_all / G_-P per (video seed, EEG seed,
                  task), and the OOF reference (§5, §6)
  boxes       S1  crop boxes for RN201 / RN203 / RN243 from 3 days of keyframe pairs (§8), PNGs, boxes.json
  boxes-freeze S1 writes boxes.json from the candidates + the check (run after `boxes` has made the images)
  stepa-prep  S2  sample 360 labelled clips, locate r0, build arms C/K/E (frames) and R (raw EDF EEG) (§7)
  stepa-gpu   S2  run the clips' OOF networks on arms C/K/E/R (GPU)
  stepa-eval  S2  tolerances and the decision -> stepA/decision.json
  decode      S4  one camera file: keyframes (mode K) -> per-animal frame shards, EEG windows, keyframe motion
  clock       S4  the motion-vs-Activity clock check per 1-h block per file and box (no network output read)
  infer       S4  GPU: consume shards, write per-window predictions, delete shards
  analyse     S5  every number of §6 -> ttg_scan/scan_results.{json,txt} and results/summary.{json,txt}
  gallery     S6  blinded review items (§9)
  status          progress of every stage, disk and the GPU ledger

Writes ONLY under $EEG_ROOT/output/ttg_scan/ (out_path() refuses anything else). The raw tree, data/, data_full/,
cache_frames/ and every other output directory are opened read-only.

Environment: tables / clock / analyse need numpy + scipy only (the miniconda3 base is fine); decode, boxes and
stepa-prep need av + cv2 (the eeg env); gates needs sklearn (the eeg env); the GPU stages need torch.
rawdata/submit_scan.sh runs everything on the scavenger QoS.
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
from collections import Counter, OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
GRADER = os.path.join(REPO, "grader")
sys.path.insert(0, HERE)
import align_clips as AC  # noqa: E402  (EDF header / reader, time helpers; stdlib-only at import)

EEG_ROOT = os.path.realpath(os.environ.get("EEG_ROOT", "/work/mech-ai-scratch/alloy/EEG"))
RAW = os.path.realpath(os.environ.get("RAW_ROOT", "/work/mech-ai/alloydas/EEG/Data"))
SCAN = os.path.join(EEG_ROOT, "output", "ttg_scan")
AUDIT = os.path.join(EEG_ROOT, "output", "raw_audit")
RECUT = os.path.join(EEG_ROOT, "output", "ttg_recut")
VSUBJ = os.path.join(EEG_ROOT, "output", "ttg_vsubj")
PATCHED = os.path.join(RECUT, "vsubj_patched")
ESUBJ = os.path.join(EEG_ROOT, "output", "ttg_eeg", "subject")
F16 = os.path.join(EEG_ROOT, "cache_frames", "f16s224")
FULL = os.path.join(EEG_ROOT, "data_full")
PREREG = os.path.join(HERE, "scan_prereg.md")
FFBIN = "/work/mech-ai/alloy/miniconda3/bin"
FFMPEG, FFPROBE = os.path.join(FFBIN, "ffmpeg"), os.path.join(FFBIN, "ffprobe")

UTC = dt.timezone.utc
CAMS = OrderedDict([
    ("RN199-RN201", dict(room="C", left="RN199", right="RN201", partner="RN199", nolabel="RN201")),
    ("RN203-RN204", dict(room="C", left="RN203", right="RN204", partner="RN204", nolabel="RN203")),
    ("RN242-RN243", dict(room="D", left="RN242", right="RN243", partner="RN242", nolabel="RN243")),
])
NOLABEL = ("RN201", "RN203", "RN243")
PARTNERS = ("RN199", "RN204", "RN242")
ANIMALS = ("RN199", "RN201", "RN203", "RN204", "RN242", "RN243")
CAM_OF = {a: c for c, d in CAMS.items() for a in (d["left"], d["right"])}
PARTNER_OF = {d["nolabel"]: d["partner"] for d in CAMS.values()}
NOLABEL_OF = {d["partner"]: d["nolabel"] for d in CAMS.values()}
PARTNER_FOLD = {"RN199": 3, "RN204": 4, "RN242": 2}         # split_subjects seed 49 (asserted by `gates`)
FOLDER = {"RN199": "RN199", "RN201": "DONE_RN201", "RN203": "DONE_RN203", "RN204": "RN204", "RN242": "RN242",
          "RN243": "RN243"}
DEFECT_STAMPS = {"D": ("20231014004100",), "C": ("20231025153704", "20231103122100", "20231205222300")}
UNREADABLE = "RN199-RN201.20231213214000.mp4"
FALL_NAIVE = dt.datetime(2023, 11, 5, 1, 0, 0)               # EDF naive clock: UTC-5 before, UTC-6 from here
UNFINALISED = ("RN203", "11-15-2023")
UNFINALISED_RECORDS = 8508
SEEDS = (1, 2, 3)
FOLDS = (0, 1, 2, 3, 4)
V_EPOCH, E_EPOCH = 12, 30
TASKS = ("g3", "g5")
T, S = 16, 224
FS, DECIM, WIN, STRIDE = 1000, 8, 750, 375                    # 1 kHz -> 125 Hz, 6-s windows, 3-s stride
NWIN = 19                                                     # (7500 - 750) / 375 + 1
DROPOUT_RUN = 1000
MAX_GAP_S = 1.0
EVENT_GUARD_S = 600
EPISODE_MERGE_S = 300
SHARD_CAP_BYTES = 200 * 10 ** 9
GPU_CAP_H = 6.0
ACT_LAG_S = -0.74       # motion(t + L) ~ Activity(t): Activity lags motion by 0.74 s (recut_clips.py) -> L = -0.74
CLOCK_TOL_S = 10.0
CLOCK_LAG_S = 120
STEPA_SEED = 20260929

# exclusion reasons, in the pre-registered order (§3); 'scanned' is not a reason
REASONS = ["no video", "unreadable file", "timing-defect file", "EDF stub", "DST hour", "unfinalised EDF",
           "EDF gap", "no 1-kHz channel", "EDF ambiguous", "partial window", "EEG dropout", "clock", "decode error"]
RCODE = {r: i + 1 for i, r in enumerate(REASONS)}             # 0 = scannable so far / scanned


# ============================================================================ guards and io

def _under(p, root):
    p, root = os.path.realpath(p), os.path.realpath(root)
    return p == root or p.startswith(root + os.sep)


def out_path(*parts):
    """Every write goes through here: under ttg_scan/ only."""
    p = os.path.join(SCAN, *parts)
    if not _under(p, SCAN):
        raise SystemExit(f"refusing to write outside {SCAN}: {p}")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


STAGE = os.environ.get("SCAN_STAGE")        # GPU jobs: a node-local copy of the inputs (see submit_scan.sh)


def rd(path):
    """Read path: its node-local staged copy when one exists (the scratch file system reads at 1-4 MB/s cold),
    else the path itself. Only inputs go through here; every write goes through out_path()."""
    if STAGE:
        q = os.path.join(STAGE, os.path.abspath(path).lstrip("/"))
        if os.path.exists(q):
            return q
    return path


def raw_path(rel):
    p = os.path.join(RAW, rel)
    if not _under(p, RAW):
        raise SystemExit(f"not under the raw tree: {p}")
    return p


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha_obj(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
    except ImportError:
        pass
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, (dt.datetime, dt.date)):
        return o.isoformat()
    return str(o)


def atomic_json(path, obj):
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=_json_default)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_npz(path, **arrays):
    import numpy as np
    tmp = path + f".tmp{os.getpid()}.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        return None if x in (None, "") else float(x)
    except ValueError:
        return None


def log(msg):
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def run_json(stage, extra=None):
    """Provenance of a stage run: the pre-registration's sha256 (the contract), argv, host, job."""
    d = dict(stage=stage, prereg=os.path.relpath(PREREG, REPO), prereg_sha256=sha256(PREREG), argv=sys.argv,
             host=os.uname().nodename, job=os.environ.get("SLURM_JOB_ID"), at=time.strftime("%F %T"),
             script_sha256=sha256(os.path.abspath(__file__)))
    if extra:
        d.update(extra)
    atomic_json(out_path(stage, "run.json") if not stage.endswith(".json") else out_path(stage), d)
    return d


def utc_of(s):
    """'YYYY-mm-dd HH:MM:SS[.f]' naive UTC -> epoch seconds."""
    t = AC.parse_dt(s)
    return None if t is None else t.replace(tzinfo=UTC).timestamp()


def naive_to_utc_s(naive):
    """EDF / xlsx naive local wall time -> UTC epoch s (§3: UTC-5 before naive 2023-11-05 01:00, UTC-6 after)."""
    off = 5 if naive < FALL_NAIVE else 6
    return (naive + dt.timedelta(hours=off)).replace(tzinfo=UTC).timestamp()


def epoch_iso(t):
    return dt.datetime.fromtimestamp(t, tz=UTC).strftime("%Y-%m-%d %H:%M:%S")


def sample_frame_indices(n_total, n_sample):
    """train_classifier.sample_frame_indices, verbatim."""
    import numpy as np
    if n_total <= 0:
        return None
    if n_total >= n_sample:
        return np.linspace(0, n_total - 1, n_sample).round().astype(int)
    idx = np.arange(n_total)
    pad = np.full(n_sample - n_total, n_total - 1, dtype=int)
    return np.concatenate([idx, pad])


def snap_to_keyframes(pos, kf):
    """Nearest keyframe index (file frame index) to each position; ties go to the earlier keyframe (§4)."""
    import numpy as np
    j = np.searchsorted(kf, pos)                 # kf[j-1] < pos <= kf[j]
    lo = kf[np.clip(j - 1, 0, len(kf) - 1)]
    hi = kf[np.clip(j, 0, len(kf) - 1)]
    return np.where(np.abs(pos - lo) <= np.abs(hi - pos), lo, hi)


# ============================================================================ S0: tables

def load_file_rows():
    rows = read_csv(os.path.join(AUDIT, "files.csv"))
    return rows


def camera_files(rows):
    """{camera: [file dicts sorted by true start]}: one entry per unique camera file, read from the first-named
    animal's folder after its name and size are checked against the partner's copy (§10)."""
    out = {}
    problems = []
    for cam, d in CAMS.items():
        copies = defaultdict(dict)
        for r in rows:
            if r["kind"] == "video" and r["camera"] == cam:
                copies[r["name"]][r["animal_folder"]] = r
        L = []
        for name in sorted(copies):
            cp = copies[name]
            fl, fr = FOLDER[d["left"]], FOLDER[d["right"]]
            if set(cp) != {fl, fr}:
                problems.append((cam, name, f"copies in {sorted(cp)}"))
            src = cp.get(fl) or cp.get(fr)
            sizes = {k: int(v["size"]) for k, v in cp.items()}
            if len(set(sizes.values())) != 1:
                problems.append((cam, name, f"sizes differ {sizes}"))
            xml_utc = src.get("xml_utc_start")
            sync = fnum(src.get("sync_offset_s"))
            xso = fnum(src.get("xml_sync_offset"))
            if xml_utc and xso is not None and sync is not None and abs(sync - xso / 1e7) > 1e-9:
                problems.append((cam, name, f"sync_offset_s {sync} != xml_sync_offset/1e7 {xso / 1e7}"))
            if sync is None and xso is not None:
                sync = xso / 1e7
            stamp = name.split(".")[-2]
            fn_close = dt.datetime.strptime(stamp, "%Y%m%d%H%M%S").replace(tzinfo=UTC).timestamp()
            t0 = int(xml_utc) + sync if xml_utc else None
            kind = "ok"
            if name == UNREADABLE:
                kind = "unreadable"
            elif stamp in DEFECT_STAMPS[d["room"]]:
                kind = "defect"
            elif src["status"] == "error":
                kind = "unreadable"
            L.append(dict(name=name, camera=cam, relpath=src["relpath"], size=int(src["size"]), stamp=stamp,
                          t0=t0, fn_close=fn_close, kind=kind, status=src["status"], problems=src["problems"],
                          nb_frames=int(fnum(src.get("nb_frames")) or 0), stream_s=fnum(src.get("stream_duration")),
                          day_folder={a: copies[name].get(FOLDER[a], {}).get("day_folder") for a in (d["left"], d["right"])},
                          copies={k: v["relpath"] for k, v in cp.items()}))
        L.sort(key=lambda v: (v["t0"] if v["t0"] is not None else v["fn_close"]))
        out[cam] = L
    return out, problems


def packets_path(name):
    return out_path("s0", "packets", name.replace(".mp4", ".npz"))


def probe_packets(v):
    """pts (stream ticks) and keyframe flags of every video packet, from ffprobe -show_packets (§3)."""
    import numpy as np
    dst = packets_path(v["name"])
    ck = f"{v['relpath']}|{v['size']}"
    if os.path.exists(dst):
        z = np.load(dst)
        if str(z["cache_key"]) == ck:
            return f"{v['name']}: cached"
    p = raw_path(v["relpath"])
    r = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=time_base",
                        "-of", "csv=p=0", p], capture_output=True, text=True, timeout=300)
    num, den = (int(x) for x in r.stdout.strip().split("/"))
    r = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts,flags",
                        "-of", "csv=p=0", p], capture_output=True, text=True, timeout=1800)
    pts, key_ = [], []
    for line in r.stdout.splitlines():
        a = line.split(",")
        if len(a) < 2 or a[0] in ("", "N/A"):
            continue
        pts.append(int(a[0]))
        key_.append("K" in a[1])
    pts = np.array(pts, np.int64)
    key_ = np.array(key_, bool)
    o = np.argsort(pts, kind="stable")
    atomic_npz(dst, pts=pts[o], key=key_[o], tb=np.array([num, den], np.int64), cache_key=np.array(ck),
               reordered=np.array(int((o != np.arange(len(o))).sum())), stderr=np.array(r.stderr[-2000:]))
    return f"{v['name']}: {len(pts)} packets, {int(key_.sum())} K, reordered {int((o != np.arange(len(o))).sum())}"


def load_packets(name):
    import numpy as np
    z = np.load(rd(packets_path(name)))
    num, den = (int(x) for x in z["tb"])
    return z["pts"], z["key"], num / den


def edf_table(rows):
    """Per animal: every EDF with its placement (§3), 1-kHz channel, complete records and stub flag."""
    out = {}
    xlsx = defaultdict(set)
    for r in rows:
        if r["kind"] == "xlsx":
            xlsx[r["animal_folder"]].add(r["day_folder"])
    for a in ANIMALS:
        L = []
        for r in rows:
            if r["kind"] != "edf" or r["animal_folder"] != FOLDER[a]:
                continue
            h = AC.edf_header(raw_path(r["relpath"]))
            labels = h["labels"]
            ch = next((i for i, l in enumerate(labels) if "EEG" in l.upper()), None)
            if ch is None:
                ch = next((i for i, l in enumerate(labels) if "ECG" in l.upper()), None)
            fs = h["nsamp"][ch] / h["rdur"] if ch is not None else None
            ia = next((i for i, l in enumerate(labels) if l.lower().startswith("activity")), None)
            nrec_hdr = int(AC.fnum(r.get("edf_nrec")) or 0)
            R = h["nrec_file"] if nrec_hdr <= 0 else min(nrec_hdr, h["nrec_file"])
            if (a, r["day_folder"]) == UNFINALISED:
                if R != UNFINALISED_RECORDS:
                    raise SystemExit(f"{r['relpath']}: {R} complete records, expected {UNFINALISED_RECORDS}")
            hours = R * h["rdur"] / 3600
            L.append(dict(relpath=r["relpath"], day_folder=r["day_folder"], start_naive=h["start"].isoformat(),
                          rdur=h["rdur"], records=R, nrec_header=nrec_hdr, nrec_file=h["nrec_file"], hours=hours,
                          stub=hours < 1.0, ch=ch, ch_label=labels[ch] if ch is not None else None,
                          ch_fs=fs, one_khz=(fs == FS), act=ia, labels=labels, dims=None,
                          unfinalised=(a, r["day_folder"]) == UNFINALISED))
        L.sort(key=lambda e: e["start_naive"])
        out[a] = dict(edfs=L, xlsx_days=sorted(xlsx[FOLDER[a]]))
    return out


def edf_utc_span(e):
    """UTC epoch pieces [(u0, u1, naive_origin_utc_offset_h)] of an EDF's complete records."""
    s = dt.datetime.fromisoformat(e["start_naive"])
    end = s + dt.timedelta(seconds=e["records"] * e["rdur"])
    if s < FALL_NAIVE < end:
        fall5 = (FALL_NAIVE + dt.timedelta(hours=5)).replace(tzinfo=UTC).timestamp()     # 06:00 UTC
        return [(naive_to_utc_s(s), fall5, 5), (naive_to_utc_s(FALL_NAIVE), naive_to_utc_s(end), 6)]
    return [(naive_to_utc_s(s), naive_to_utc_s(end), 5 if s < FALL_NAIVE else 6)]


def minute_to_naive(m):
    """UTC epoch minute start -> EDF naive local datetime, or None in the DST hour (no naive time maps there)."""
    u = dt.datetime.fromtimestamp(m, tz=UTC).replace(tzinfo=None)
    n5 = u - dt.timedelta(hours=5)
    if n5 < FALL_NAIVE:
        return n5
    n6 = u - dt.timedelta(hours=6)
    if n6 >= FALL_NAIVE:
        return n6
    return None


def cmd_tables(a):
    import numpy as np
    rows = load_file_rows()
    cams, problems = camera_files(rows)
    run_json("s0", dict(sub="tables"))
    log(f"camera files: " + ", ".join(f"{c} {len(L)}" for c, L in cams.items()) + f"; copy problems: {problems}")
    todo = [v for L in cams.values() for v in L if v["kind"] != "unreadable"]
    with ThreadPoolExecutor(a.workers) as ex:
        for msg in ex.map(probe_packets, todo):
            if "cached" not in msg:
                log(msg)
    edfs = edf_table(rows)
    # per-file frame timeline summary
    finfo = {}
    for cam, L in cams.items():
        for v in L:
            if v["kind"] == "unreadable":
                continue
            pts, key, tb = load_packets(v["name"])
            t = pts * tb
            gaps = np.diff(t)
            finfo[v["name"]] = dict(n=len(pts), n_key=int(key.sum()), first=float(t[0]), last=float(t[-1]),
                                    max_gap=float(gaps.max()) if len(gaps) else 0.0,
                                    n_gap_gt1=int((gaps > MAX_GAP_S).sum()), nonmono=int((gaps <= 0).sum()),
                                    nb_frames_audit=v["nb_frames"], kf_spacing=np.unique(np.diff(np.flatnonzero(key)),
                                                                                         return_counts=True))
    tables = {}
    summary = {}
    for cam, L in cams.items():
        for animal in (CAMS[cam]["left"], CAMS[cam]["right"]):
            tab, sm = minute_table(animal, cam, L, edfs[animal])
            tables[animal] = tab
            summary[animal] = sm
            atomic_npz(out_path("s0", "windows", f"{animal}.npz"), **tab)
            log(f"{animal}: {sm['n_minutes']} minutes; candidates {sm['n_candidates']} "
                f"({sm['n_candidates'] / 60:.1f} h); " + ", ".join(f"{k} {v}" for k, v in sm["reasons"].items() if v))
    atomic_json(out_path("s0", "tables.json"), dict(
        cameras={c: [dict((k, v) for k, v in f.items()) for f in L] for c, L in cams.items()},
        copy_problems=problems, edfs=edfs, files=finfo, summary=summary, created=time.strftime("%F %T")))


def minute_table(animal, cam, files, ed):
    """Every whole UTC minute from the animal's first to its last recorded instant, with the first exclusion
    reason that applies among those known before decoding, and the window geometry of the candidates."""
    import numpy as np
    edfs = [e for e in ed["edfs"]]
    # recorded span: union of the camera's frames / file spans and the animal's EDFs
    t_lo, t_hi = [], []
    for v in files:
        if v["t0"] is not None:
            t_lo.append(v["t0"])
        t_hi.append(v["fn_close"])
    for e in edfs:
        for u0, u1, _ in edf_utc_span(e):
            t_lo.append(u0)
            t_hi.append(u1)
    m0 = int(math.floor(min(t_lo) / 60) * 60)
    m1 = int(math.ceil(max(t_hi) / 60) * 60)
    M = np.arange(m0, m1, 60, dtype=np.int64)
    n = len(M)
    vid = np.full(n, -1, np.int32)           # index into files of the file whose frames cover the minute
    vreason = np.full(n, RCODE["no video"], np.int8)
    nfr = np.zeros(n, np.int32)
    first = np.zeros(n, np.int64)
    kf16 = np.zeros((n, T), np.int64)        # file frame indices of the 16 snapped keyframes
    kfpts = np.zeros((n, T), np.int64)
    # 1. video
    spans = []
    for fi, v in enumerate(files):
        if v["kind"] == "unreadable":
            lo = v["t0"] if v["t0"] is not None else v["fn_close"] - 3600
            spans.append((lo, v["fn_close"], RCODE["unreadable file"]))
            continue
        pts, key, tb = load_packets(v["name"])
        tt = v["t0"] + pts * tb
        if v["kind"] == "defect":
            spans.append((min(v["t0"], tt[0]), max(v["fn_close"], tt[-1]), RCODE["timing-defect file"]))
            continue
        kf = np.flatnonzero(key)
        mm0 = int(math.ceil(tt[0] / 60) * 60) - 60
        for m in range(mm0, int(tt[-1]) + 60, 60):
            i = (m - m0) // 60
            if i < 0 or i >= n:
                continue
            i0, i1 = np.searchsorted(tt, [m, m + 60])
            if i1 <= i0:
                continue
            if tt[i0] - m > MAX_GAP_S or (m + 60) - tt[i1 - 1] > MAX_GAP_S:
                continue
            if i1 - i0 > 1 and np.diff(tt[i0:i1]).max() > MAX_GAP_S:
                continue
            if vid[i] >= 0:
                continue                      # already covered by an earlier file (never happens; kept first)
            pos = sample_frame_indices(i1 - i0, T) + i0
            sn = snap_to_keyframes(pos, kf)
            vid[i], vreason[i], nfr[i], first[i] = fi, 0, i1 - i0, i0
            kf16[i] = sn
            kfpts[i] = pts[sn]
    # a minute inside the span of the unreadable file or of a timing-defect file HAS video (of the wrong kind),
    # so 'no video' does not apply to it; it takes that file's reason
    for lo, hi, code in spans:
        sel = (M + 60 > lo) & (M < hi) & (vreason != 0)
        vreason[sel] = code
    # 2. EEG (only for minutes with video)
    eidx = np.full(n, -1, np.int32)
    k0 = np.zeros(n, np.int64)
    ereason = np.zeros(n, np.int8)
    day_of = {}
    for fi, v in enumerate(files):
        day_of[fi] = v["day_folder"].get(animal)
    stub_days = defaultdict(list)
    for e in edfs:
        stub_days[e["day_folder"]].append(e["stub"])
    stub_day = {d: all(x) for d, x in stub_days.items()}
    usable = [e for e in edfs if not e["stub"]]
    naive_start = [dt.datetime.fromisoformat(e["start_naive"]) for e in usable]
    naive_end = [s + dt.timedelta(seconds=e["records"] * e["rdur"]) for s, e in zip(naive_start, usable)]
    unfin = [j for j, e in enumerate(usable) if e["unfinalised"]]
    for i in np.flatnonzero(vreason == 0):
        m = int(M[i])
        nv = minute_to_naive(m)
        if nv is None:
            if stub_day.get(day_of[vid[i]]):
                ereason[i] = RCODE["EDF stub"]
            else:
                ereason[i] = RCODE["DST hour"]
            continue
        n1 = nv + dt.timedelta(seconds=60)
        full = [j for j in range(len(usable)) if naive_start[j] <= nv and n1 <= naive_end[j]]
        part = [j for j in range(len(usable)) if naive_start[j] < n1 and nv < naive_end[j]]
        if not part:
            d = day_of[vid[i]]
            if stub_day.get(d):
                ereason[i] = RCODE["EDF stub"]
            elif unfin and d == UNFINALISED[1] and nv >= naive_end[unfin[0]]:
                ereason[i] = RCODE["unfinalised EDF"]
            else:
                ereason[i] = RCODE["EDF gap"]
            continue
        if stub_day.get(day_of[vid[i]]) and not full:
            ereason[i] = RCODE["EDF stub"]
            continue
        if unfin and not full and part == unfin:
            ereason[i] = RCODE["unfinalised EDF"]
            continue
        if any(not usable[j]["one_khz"] for j in part):
            ereason[i] = RCODE["no 1-kHz channel"]
            continue
        if len(part) > 1:
            ereason[i] = RCODE["EDF ambiguous"]
            continue
        if not full:
            ereason[i] = RCODE["partial window"]
            continue
        j = full[0]
        eidx[i] = edfs.index(usable[j])
        k0[i] = int(round((nv - naive_start[j]).total_seconds() * FS))
    reason = np.where(vreason != 0, vreason, ereason).astype(np.int8)
    cand = reason == 0
    # the day folder that holds the window's video (this animal's copy) and whether it has an xlsx
    day = np.array([day_of[f] if f >= 0 else "" for f in vid], dtype=object)
    xl = set(ed["xlsx_days"])
    has_xlsx = np.array([d in xl for d in day], bool)
    tab = dict(minute=M, reason=reason, fidx=vid, nfr=nfr, first=first, kf16=kf16, kfpts=kfpts, edf=eidx, k0=k0,
               day=np.array([str(d) for d in day]), has_xlsx=has_xlsx,
               files=np.array([v["name"] for v in files]), animal=np.array(animal), camera=np.array(cam))
    sm = dict(n_minutes=int(n), first=epoch_iso(m0), last=epoch_iso(m1), n_candidates=int(cand.sum()),
              reasons={r: int((reason == RCODE[r]).sum()) for r in REASONS})
    return tab, sm


def load_table(animal):
    import numpy as np
    z = np.load(rd(out_path("s0", "windows", f"{animal}.npz")), allow_pickle=False)
    return {k: z[k] for k in z.files}


def load_tables_json():
    return json.load(open(rd(out_path("s0", "tables.json"))))


# ============================================================================ main (sub-commands registered below)

COMMANDS = {}


def command(name):
    def deco(f):
        COMMANDS[name] = f
        return f
    return deco


command("tables")(cmd_tables)


# ============================================================================ S0: gates, reproduction, OOF reference

def gate_features(Pv, pe):
    """joint_gate.Pairing.systems' EGRG features: [logit P_V(sz), logit P_E(sz) pooled], P_V(sz) = 1 - P_V(non)."""
    import numpy as np
    def logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))
    return np.c_[logit(1.0 - Pv[:, 0]), logit(pe)]


def gate_q(coef, Pv, pe):
    import numpy as np
    X = gate_features(Pv, pe)
    a, b, c = coef
    return 1.0 / (1.0 + np.exp(-(a * X[:, 0] + b * X[:, 1] + c)))


def egrg_decide(Pv, q):
    """argmax over [1 - q, q * P_V(g | sz)] (joint_gate.comp), returns the class index (0 = non-seizure)."""
    import numpy as np
    sz = Pv[:, 1:] / np.clip(Pv[:, 1:].sum(1, keepdims=True), 1e-12, None)
    return np.c_[1 - q, q[:, None] * sz].argmax(1)


def cmd_gates(a):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    sys.path.insert(0, GRADER)
    sys.path.insert(0, os.path.join(GRADER, "eeg"))
    import rescore_recut as RR
    JG = RR.import_joint_gate()
    import check_folds as CF
    run_json("s0/gates.json".replace(".json", "_run.json"), dict(sub="gates"))
    folds = CF.expected_folds()
    fold_of = {an: f for f, d in folds.items() for an in d["eeg"]}
    for p, f in PARTNER_FOLD.items():
        if fold_of.get(p) != f:
            raise SystemExit(f"{p}: split_subjects fold {fold_of.get(p)} != registered {f}")
    for n in NOLABEL:
        if n in fold_of:
            raise SystemExit(f"{n} is in a split fold: it should have no labelled clips")
    exc = RR.read_exclude()
    RR.adapt_joint_gate(JG, list(exc), RR.GRID)
    vbs, eb = RR.load_bundles(JG, "x3dfix", "patched", list(exc), folds)
    summ = json.load(open(os.path.join(RECUT, "rescore", "summary.json")))
    out = dict(created=time.strftime("%F %T"), folds={str(f): sorted(d["eeg"]) for f, d in folds.items()},
               partner_fold=PARTNER_FOLD, loads={t: JG.logline(vbs[t]["log"]) for t in TASKS},
               eeg_load=JG.logline(eb["log"]), repro={}, gates={}, oof_reference={})
    ok_all = True
    for t in TASKS:
        P, rows = RR.score_pairs(JG, vbs[t], eb, t)
        ref = summ["recipes"]["x3dfix"][t]["patched"]["pairs"]
        rep = {}
        for (vn, en), r in rows.items():
            k = f"{vn} x {en}"
            want = ref[k]
            got = dict(egrg_hits=r["EGRG"]["hits"], egrg_f1=r["EGRG"]["macro_f1"], video_hits=r["video"]["hits"],
                       video_f1=r["video"]["macro_f1"])
            same = all(got[x] == want[x] for x in got)
            # the same LOAO with this script's feature / fit code must give the same decisions
            Pv = vbs[t]["members"][vn][P.iv]
            pe = eb["members"][en]["stored"][P.ie]
            X = gate_features(Pv, pe)
            q = np.empty(len(P.ys))
            for g in np.unique(P.an):
                tr = P.an != g
                m = LogisticRegression(C=1.0, max_iter=5000).fit(X[tr], P.ys[tr])
                q[~tr] = m.predict_proba(X[~tr])[:, 1]
            mine = egrg_decide(Pv, q)
            jg = r["EGRG"]["pred"]
            same_mine = bool(np.array_equal(mine, jg))
            rep[k] = dict(identical_to_summary=same, own_code_identical_decisions=same_mine, **got)
            ok_all &= same and same_mine
            # deployment gates: G_all (all 20 animals) and G_-P (all but partner P) for this pair
            gates = {}
            m_all = LogisticRegression(C=1.0, max_iter=5000).fit(X, P.ys)
            gates["all"] = np.r_[m_all.coef_[0], m_all.intercept_].tolist()
            for pa in PARTNERS:
                tr = P.an != pa
                if tr.all():
                    raise SystemExit(f"{pa} has no clip in the pairing")
                mp = LogisticRegression(C=1.0, max_iter=5000).fit(X[tr], P.ys[tr])
                gates[f"minus_{pa}"] = np.r_[mp.coef_[0], mp.intercept_].tolist()
            out["gates"][f"{t}|{vn}|{en}"] = gates
            # OOF reference (§6): non-seizure-clip flag share and seizure-clip detection share, all clips and the
            # partners' own clips; EEG alone P_E > 0.5 (one value per EEG member, repeated per pair)
            ns = P.ys == 0
            vflag = Pv.argmax(1) > 0
            eflag = pe > 0.5
            gflag = jg > 0
            ref_row = dict(n_nonseizure=int(ns.sum()), n_seizure=int((~ns).sum()),
                           egrg_flag_nonseizure=int(gflag[ns].sum()), video_flag_nonseizure=int(vflag[ns].sum()),
                           eeg_flag_nonseizure=int(eflag[ns].sum()),
                           egrg_detect_seizure=int(gflag[~ns].sum()), video_detect_seizure=int(vflag[~ns].sum()),
                           eeg_detect_seizure=int(eflag[~ns].sum()), per_partner={})
            for pa in PARTNERS:
                s_ = P.an == pa
                ref_row["per_partner"][pa] = dict(
                    n_nonseizure=int((s_ & ns).sum()), n_seizure=int((s_ & ~ns).sum()),
                    egrg_flag_nonseizure=int(gflag[s_ & ns].sum()), video_flag_nonseizure=int(vflag[s_ & ns].sum()),
                    eeg_flag_nonseizure=int(eflag[s_ & ns].sum()), egrg_detect_seizure=int(gflag[s_ & ~ns].sum()),
                    video_detect_seizure=int(vflag[s_ & ~ns].sum()), eeg_detect_seizure=int(eflag[s_ & ~ns].sum()))
            out["oof_reference"][f"{t}|{vn}|{en}"] = ref_row
        out["repro"][t] = rep
        log(f"[gates] {t}: {sum(v['identical_to_summary'] for v in rep.values())}/{len(rep)} pairs identical to "
            f"rescore/summary.json; own LOAO code identical on {sum(v['own_code_identical_decisions'] for v in rep.values())}"
            f"/{len(rep)}")
    # OOF reference means (§6)
    ref = {}
    for t in TASKS:
        rr = [v for k, v in out["oof_reference"].items() if k.startswith(t + "|")]
        nn = rr[0]["n_nonseizure"]
        ref[t] = dict(n_nonseizure=nn, n_seizure=rr[0]["n_seizure"],
                      egrg_flag_mean=float(np.mean([r["egrg_flag_nonseizure"] for r in rr])),
                      video_flag_mean=float(np.mean([r["video_flag_nonseizure"] for r in rr])),
                      eeg_flag_mean=float(np.mean([r["eeg_flag_nonseizure"] for r in rr])),
                      egrg_detect_mean=float(np.mean([r["egrg_detect_seizure"] for r in rr])),
                      video_detect_mean=float(np.mean([r["video_detect_seizure"] for r in rr])),
                      eeg_detect_mean=float(np.mean([r["eeg_detect_seizure"] for r in rr])))
        for k in ("egrg", "video", "eeg"):
            ref[t][f"{k}_flags_per_hour_if_like_oof_nonseizure"] = ref[t][f"{k}_flag_mean"] / nn * 60
    out["oof_reference_mean"] = ref
    out["repro_ok"] = bool(ok_all)
    atomic_json(out_path("s0", "gates.json"), out)
    for t in TASKS:
        r = ref[t]
        log(f"[gates] OOF reference {t}: EGRG {r['egrg_flag_mean']:.1f}/{r['n_nonseizure']} = "
            f"{r['egrg_flags_per_hour_if_like_oof_nonseizure']:.2f}/h; video {r['video_flag_mean']:.1f} = "
            f"{r['video_flags_per_hour_if_like_oof_nonseizure']:.2f}/h; EEG {r['eeg_flag_mean']:.1f} = "
            f"{r['eeg_flags_per_hour_if_like_oof_nonseizure']:.2f}/h")
    if not ok_all:
        raise SystemExit("REPRODUCTION GATE FAILED: the work stops (§5)")
    log("[gates] reproduction gate passed")


command("gates")(cmd_gates)


# ============================================================================ S1: crop boxes (§8)

LABELLED_BOX = {}          # filled from raw_audit/align/crop_boxes.csv


def labelled_boxes():
    if not LABELLED_BOX:
        for r in read_csv(rd(os.path.join(AUDIT, "align", "crop_boxes.csv"))):
            if r["camera"] in CAMS:
                LABELLED_BOX[r["animal"]] = tuple(int(r[k]) for k in ("x", "y", "w", "h"))
    return LABELLED_BOX


CANDIDATE = {"RN201": (416, 180, 384, 360), "RN203": (0, 0, 480, 480)}   # RN243: from the median image (§8)
PARENT_CROP_RN243 = (400, 270, 320, 222)   # recut_clips.CROP_FRACS['RN243'] (the parent's _crop_roomd.sh): shown only


def decode_keyframes_at(path, want_pts, seek=True):
    """{pts: BGR frame} for keyframe packets with these pts (demux, decode only those packets: I-frames decode on
    their own; bit-identical to PyAV skip_frame='NONKEY' decoding, checked 400/400 on RN203-RN204.20231011024800)."""
    import av
    want = set(int(p) for p in want_pts)
    out = {}
    c = av.open(path)
    try:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        cc = s.codec_context
        last = max(want) if want else -1
        if seek and want:
            c.seek(min(want), stream=s, backward=True, any_frame=False)   # the mp4 index: exact, demux from there
        for pk in c.demux(s):
            if pk.pts is None:
                continue
            if pk.pts in want:
                for fr in cc.decode(pk):
                    out[int(fr.pts)] = fr.to_ndarray(format="bgr24")
            if pk.pts > last:
                break
        for fr in cc.decode(None):
            if int(fr.pts) in want:
                out[int(fr.pts)] = fr.to_ndarray(format="bgr24")
    finally:
        c.close()
    return out


def box_days(animal):
    """The first, middle and last day folder (the animal's own copies) that hold video of its camera."""
    tj = load_tables_json()
    files = [v for v in tj["cameras"][CAM_OF[animal]] if v["kind"] == "ok"]
    days = []
    for v in files:
        d = v["day_folder"][animal]
        if d not in days:
            days.append(d)
    return [days[0], days[len(days) // 2], days[-1]], files


def cmd_boxes(a):
    """Median image and consecutive-keyframe motion map on 3 fixed days per no-label animal's camera (§8)."""
    import numpy as np
    import cv2
    run_json("boxes", dict(sub="boxes"))
    lb = labelled_boxes()
    for animal in NOLABEL:
        days, files = box_days(animal)
        res = {}
        for day in days:
            fl = [v for v in files if v["day_folder"][animal] == day]
            t_first = min(v["t0"] + load_packets(v["name"])[0][0] * load_packets(v["name"])[2] for v in fl)
            targets = [t_first + 600 * i for i in range(144)]
            per_file = defaultdict(list)
            for tt in targets:
                for v in fl:
                    pts, key, tb = load_packets(v["name"])
                    tk = v["t0"] + pts[key] * tb
                    j = np.searchsorted(tk, tt)
                    if j + 1 < len(tk) and tk[0] <= tt:
                        kp = pts[key]
                        per_file[v["name"]].append((int(kp[j]), int(kp[j + 1])))
                        break
            meds, mot, n = [], None, 0
            for v in fl:
                pr = per_file.get(v["name"])
                if not pr:
                    continue
                fr = decode_keyframes_at(raw_path(v["relpath"]), [p for pair in pr for p in pair])
                for p1, p2 in pr:
                    if p1 not in fr or p2 not in fr:
                        continue
                    g1 = cv2.cvtColor(fr[p1], cv2.COLOR_BGR2GRAY).astype(np.float32)
                    g2 = cv2.cvtColor(fr[p2], cv2.COLOR_BGR2GRAY).astype(np.float32)
                    d = np.abs(g1 - g2)
                    mot = d if mot is None else mot + d
                    meds.append(fr[p1])
                    n += 1
            med = np.median(np.stack(meds), 0).astype(np.uint8)
            mot = mot / max(n, 1)
            np.savez_compressed(out_path("boxes", f"{animal}_{day.replace(' ', '_')}.npz"), median=med, motion=mot,
                                n_pairs=n)
            res[day] = dict(n_pairs=n)
            log(f"[boxes] {animal} {day}: {n} keyframe pairs")
        atomic_json(out_path("boxes", f"{animal}_days.json"), dict(days=days, res=res, partner=PARTNER_OF[animal],
                                                                     partner_box=lb[PARTNER_OF[animal]]))


def box_check(motion, cand, partner):
    """Share of the motion energy outside the partner's box that lies inside the candidate (§8)."""
    import numpy as np
    H, W = motion.shape
    outside = np.ones((H, W), bool)
    px, py, pw, ph = partner
    outside[py:py + ph, px:px + pw] = False
    x, y, w, h = cand
    inside = np.zeros((H, W), bool)
    inside[y:y + h, x:x + w] = True
    e_out = float(motion[outside].sum())
    e_in = float(motion[inside & outside].sum())
    overlap = not (x + w <= px or px + pw <= x or y + h <= py or py + ph <= y)
    return e_in / e_out if e_out > 0 else float("nan"), overlap


def grow_box(motions, cand, partner, W=800, H=600, thr=0.9):
    """The smallest even-pixel box that contains the candidate, does not overlap the partner's box and passes on
    every day; exhaustive over even edges with integral images."""
    import numpy as np
    x, y, w, h = cand
    best = None
    ii = []
    outs = []
    for m in motions:
        mm = m.copy()
        px, py, pw, ph = partner
        mm[py:py + ph, px:px + pw] = 0.0
        outs.append(float(mm.sum()))
        ii.append(np.pad(mm, ((1, 0), (1, 0))).cumsum(0).cumsum(1))
    for x0 in range(x, -1, -2):
        for x1 in range(x + w, W + 1, 2):
            for y0 in range(y, -1, -2):
                for y1 in range(y + h, H + 1, 2):
                    area = (x1 - x0) * (y1 - y0)
                    if best is not None and area >= best[0]:
                        break
                    c = (x0, y0, x1 - x0, y1 - y0)
                    px, py, pw, ph = partner
                    if not (x1 <= px or px + pw <= x0 or y1 <= py or py + ph <= y0):
                        break
                    ok = all((I[y1, x1] - I[y0, x1] - I[y1, x0] + I[y0, x0]) / o >= thr for I, o in zip(ii, outs))
                    if ok:
                        best = (area, c)
                        break
    return None if best is None else best[1]


def maximal_box(cand, partner, W=800, H=600):
    """The largest box that contains the candidate and does not overlap the partner's box. Every candidate here
    overlaps the partner's box in y, so it may grow in y to the full frame but in x only up to the partner."""
    x, y, w, h = cand
    px, py, pw, ph = partner
    if not (y + h <= py or py + ph <= y):
        if x >= px + pw:
            return (px + pw, 0, W - (px + pw), H)
        return (0, 0, px, H)
    raise SystemExit("candidate does not overlap the partner's y range: maximal box not implemented")


def draw_boxes(img, boxes, path):
    import cv2
    im = img.copy()
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h), color, label in boxes:
        cv2.rectangle(im, (x, y), (x + w - 1, y + h - 1), color, 2)
        cv2.putText(im, label, (x + 4, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.imwrite(path, im)


def cmd_boxes_freeze(a):
    """Apply the §8 check to the candidates (RN243's given with --rn243 x,w), grow if needed, write the PNGs and
    boxes.json. The boxes are frozen here, before any network sees a raw window."""
    import numpy as np
    import cv2
    if os.path.exists(out_path("boxes", "boxes.json")) and not a.overwrite:
        raise SystemExit("boxes.json exists (frozen)")
    lb = labelled_boxes()
    cand = dict(CANDIDATE)
    x, w = (int(v) for v in a.rn243.split(","))
    py, ph = lb["RN242"][1], lb["RN242"][3]
    cand["RN243"] = (x, py, w, ph)
    out = dict(created=time.strftime("%F %T"), rule="§8", labelled={a_: lb[a_] for a_ in PARTNERS}, boxes={},
               checks={}, rn243_walls=dict(x=x, w=w, note=a.rn243_note), parent_crop_rn243=PARENT_CROP_RN243)
    for animal in NOLABEL:
        info = json.load(open(out_path("boxes", f"{animal}_days.json")))
        partner = tuple(info["partner_box"])
        mots, meds = [], []
        for day in info["days"]:
            z = np.load(out_path("boxes", f"{animal}_{day.replace(' ', '_')}.npz"))
            mots.append(z["motion"].astype(np.float64))
            meds.append(z["median"])
        c = cand[animal]
        chk = [box_check(m, c, partner) for m in mots]
        passed = all(f >= 0.9 and not ov for f, ov in chk)
        final = c
        grown = None
        mx = maximal_box(c, partner)
        chk_max = [box_check(m, mx, partner) for m in mots]
        max_passes = all(f >= 0.9 and not ov for f, ov in chk_max)
        if not passed and max_passes:
            grown = grow_box(mots, c, partner)
            final = grown if grown is not None else c
        chk_final = [box_check(m, final, partner) for m in mots]
        out["boxes"][animal] = list(final)
        out["checks"][animal] = dict(candidate=list(c), candidate_share=[round(f, 4) for f, _ in chk],
                                     candidate_overlap=[ov for _, ov in chk], candidate_passes=passed,
                                     maximal_box=list(mx), maximal_box_share=[round(f, 4) for f, _ in chk_max],
                                     growth_possible=max_passes,
                                     median_pixel_motion=[round(float(np.median(m)), 3) for m in mots],
                                     grown_to=list(grown) if grown else None,
                                     final_share=[round(f, 4) for f, _ in chk_final], days=info["days"],
                                     partner=PARTNER_OF[animal], partner_box=list(partner))
        for day, med, mot in zip(info["days"], meds, mots):
            tag = day.replace(" ", "_")
            bx = [(partner, (0, 200, 255), PARTNER_OF[animal]), (tuple(final), (0, 255, 0), animal)]
            if tuple(final) != tuple(c):
                bx.append((tuple(c), (0, 0, 255), f"{animal} candidate"))
            if animal == "RN243":
                bx.append((PARENT_CROP_RN243, (255, 0, 255), "parent crop (not used)"))
            draw_boxes(med, bx, out_path("boxes", f"{animal}_{tag}_median.png"))
            mv = np.clip(mot / np.percentile(mot, 99.5) * 255, 0, 255).astype(np.uint8)
            draw_boxes(cv2.applyColorMap(mv, cv2.COLORMAP_INFERNO), bx, out_path("boxes", f"{animal}_{tag}_motion.png"))
        log(f"[boxes] {animal}: candidate {c} shares {[round(f, 3) for f, _ in chk]}; maximal box {mx} shares "
            f"{[round(f, 3) for f, _ in chk_max]} -> "
            f"{'passes' if passed else (f'grown to {grown}' if max_passes else 'NO box passes: candidate kept')}; "
            f"final {final}")
    out["sha256_of_boxes"] = sha_obj(out["boxes"])
    atomic_json(out_path("boxes", "boxes.json"), out)


def load_boxes():
    b = json.load(open(rd(out_path("boxes", "boxes.json"))))
    lb = labelled_boxes()
    d = {a_: tuple(lb[a_]) for a_ in PARTNERS}
    d.update({a_: tuple(v) for a_, v in b["boxes"].items()})
    return d, b["sha256_of_boxes"]


command("boxes")(cmd_boxes)
command("boxes-freeze")(cmd_boxes_freeze)


# ============================================================================ networks (GPU or CPU)

def video_run(seed, fold):
    return f"x3dfix_dual_s{seed}_fold{fold}"


def eeg_run(seed, fold):
    return f"tcn_bin_fold{fold}_s{seed}"


class VideoNets:
    """train_grader.py's validation forward pass (GraderNet, tp.norm_batch, tp._autocast, head_probs) for the
    epoch-12 weights (last.pt, checked to be the state after epoch 12 of 12, as rescore_recut.Predictor)."""

    def __init__(self, runs, dev):
        import numpy as np
        import torch
        sys.path.insert(0, GRADER)
        import ttg_common  # noqa: F401  (EEG_ROOT on sys.path)
        import train_grader as TG
        import train_pooled as tp
        self.torch, self.TG, self.tp, self.dev = torch, TG, tp, dev
        self.models, self.checks = OrderedDict(), {}
        for run in runs:
            d = os.path.join(VSUBJ, run)
            cfg = json.load(open(rd(os.path.join(d, "config.json"))))
            rk = cfg["run_key"]
            ck = torch.load(rd(os.path.join(d, "last.pt")), map_location="cpu", weights_only=False)
            st = ck["state"]
            hist = json.load(open(rd(os.path.join(d, "history.json"))))
            z = np.load(rd(os.path.join(d, f"val_ep{V_EPOCH:02d}.npz")), allow_pickle=True)
            ok = (ck["run_key"] == rk and rk["epochs"] == V_EPOCH and st["next_epoch"] == V_EPOCH + 1 and
                  not st["train_done"] and st["next_step"] == 0 and
                  max(h["epoch"] for h in st["history"]) == V_EPOCH and max(h["epoch"] for h in hist) == V_EPOCH
                  and int(z["epoch"]) == V_EPOCH and rk["arch"] == "x3d" and rk["heads"] == "dual" and rk["fix_x3d"])
            if not ok:
                raise SystemExit(f"{run}: last.pt is not the post-epoch-{V_EPOCH} x3dfix state")
            m = TG.GraderNet("x3d", ["g3", "g5"], rk["fix_x3d"], T, S, pretrained=False, logit_bound=0.0)
            m.load_state_dict(ck["model"], strict=True)
            m.eval().to(dev)
            if bool(z["head_softmax_bug"]) != m.head_softmax_bug:
                raise SystemExit(f"{run}: head_softmax_bug mismatch")
            self.models[run] = m
            self.checks[run] = dict(is_epoch12_state=True, saved_at=ck.get("saved_at"),
                                    last_pt_sha256=sha256(rd(os.path.join(d, "last.pt"))))

    def normed(self, frames):
        """uint8 (B, T, H, W, 3) numpy -> the normalised (B, 3, T, H, W) tensor on the device (FrameCache.fetch
        gives (C, T, H, W) per clip; norm_batch as in train_grader's val loop)."""
        import numpy as np
        torch = self.torch
        x = torch.from_numpy(np.ascontiguousarray(frames.transpose(0, 4, 1, 2, 3)))
        return self.tp.norm_batch(x.to(self.dev, non_blocking=True), self.dev)

    def predict(self, x, run):
        torch, tp, TG = self.torch, self.tp, self.TG
        m = self.models[run]
        with torch.no_grad(), tp._autocast(self.dev):
            o = m(x)
        return {t: TG.head_probs(m, o[t]).float().cpu().numpy().astype("float32") for t in TASKS}


class EEGNets:
    """The TCN detector at epoch 30 (final.pt == last.pt's model, epoch 30 of 30), fp32, window softmax, then
    train_pooled_eeg.aggregate_clip(..., 'logmean') per clip."""

    def __init__(self, runs, dev):
        import torch
        sys.path.insert(0, GRADER)
        import ttg_common  # noqa: F401
        import train_pooled_eeg as tpe
        self.torch, self.tpe, self.dev = torch, tpe, dev
        self.models, self.checks = OrderedDict(), {}
        for run in runs:
            d = os.path.join(ESUBJ, run)
            fin = torch.load(rd(os.path.join(d, "final.pt")), map_location="cpu", weights_only=False)
            last = torch.load(rd(os.path.join(d, "last.pt")), map_location="cpu", weights_only=False)
            st = last["state"]
            same = all(torch.equal(fin["model"][k], last["model"][k]) for k in last["model"]) and \
                len(fin["model"]) == len(last["model"])
            ok = (same and int(fin["epoch"]) == E_EPOCH and fin["run_key"]["epochs"] == E_EPOCH and
                  st["next_epoch"] == E_EPOCH + 1 and fin["run_key"]["arch"] == "tcn" and fin["run_key"]["nc"] == 2)
            if not ok:
                raise SystemExit(f"{run}: final.pt is not last.pt's epoch-{E_EPOCH} model")
            m = tpe.build_model("tcn", 2, fin["run_key"]["hidden"]).to(dev)
            with torch.no_grad():
                m(torch.zeros(2, WIN, 1, device=dev))
            m.load_state_dict(fin["model"], strict=True)
            m.eval()
            self.models[run] = m
            self.checks[run] = dict(final_equals_last=True, epoch=int(fin["epoch"]))

    def predict_windows(self, segs, run, bs=4096):
        """segs float32 (Nw, 750) z-scored -> window P (Nw, 2) (softmax, as predict_windows of train_eeg_det)."""
        import numpy as np
        torch = self.torch
        m = self.models[run]
        out = []
        with torch.no_grad():
            for b in range(0, len(segs), bs):
                x = torch.from_numpy(np.ascontiguousarray(segs[b:b + bs])).unsqueeze(-1).to(self.dev)
                out.append(torch.softmax(m(x), 1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 2), np.float32)

    def clip_scores(self, Pw, offsets):
        """logmean pooling per clip (tpe.aggregate_clip) from windows packed by offsets -> P_E(sz)."""
        import numpy as np
        cid = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
        cp = self.tpe.aggregate_clip(Pw, cid, np.arange(len(offsets) - 1), "logmean")
        return cp[:, 1].astype(np.float64)


def eeg_preprocess(sig):
    """1-kHz physical samples of one clip / window -> z-scored 6-s windows (Nw, 750) float32, exactly as
    build_stage_segments_pooled.one() + train_pooled_eeg's per-window z-score (§4)."""
    import numpy as np
    from scipy.signal import resample_poly
    x = resample_poly(sig, 1, DECIM).astype(np.float32)
    sf = FS / DECIM
    cs, ss = int(6.0 * sf), int(3.0 * sf)
    segs = []
    i = 0
    while i + cs <= len(x):
        segs.append(x[i:i + cs].astype(np.float32))
        i += ss
    if not segs:
        return np.zeros((0, WIN), np.float32)
    segs = np.stack(segs)
    m = segs.mean(1, keepdims=True)
    sd = segs.std(1, keepdims=True)
    sd[sd == 0] = 1.0
    return ((segs - m) / sd).astype(np.float32)


def longest_run(x):
    """Length of the longest run of identical consecutive values."""
    import numpy as np
    if len(x) == 0:
        return 0
    ch = np.flatnonzero(np.diff(x) != 0)
    b = np.r_[0, ch + 1, len(x)]
    return int(np.diff(b).max())


def edf_channel_rule(labels):
    """build_stage_segments_pooled.one(): first label containing EEG, else the first containing ECG."""
    ch = next((i for i, l in enumerate(labels) if "EEG" in l.upper()), None)
    if ch is None:
        ch = next((i for i, l in enumerate(labels) if "ECG" in l.upper()), None)
    return ch


def edf_samples(h, ch, s0, n, digital=False):
    """n samples of channel ch from sample s0 (MNE physical scaling: (d - dmin) * (pmax - pmin) / (dmax - dmin) +
    pmin; the EEG/ECG dimension is 'V', which MNE does not rescale)."""
    x, s0_ = AC.edf_read(h, ch, s0, s0 + n, digital=digital)
    if s0_ != s0 or len(x) != n:
        return None
    return x


# ============================================================================ S2: step A (§7)

def step1_candidates():
    c = {r["key"] for r in read_csv(os.path.join(RECUT, "plan", "candidates.csv"))}
    c |= {r["key"] for r in read_csv(os.path.join(RECUT, "exclude.csv"))}
    c |= {r["clip_key"] for r in read_csv(os.path.join(RECUT, "manifest.csv"))}
    return c


def clip_key_of(animal, session, clip_dir):
    return f"data/Data_{animal}_cropped/{session}/{clip_dir}/video.mp4"


def patched_oof():
    """{clip key: (fold, y5, {run: row})} for every clip in the patched OOF dumps (3 seeds)."""
    import numpy as np
    out = {}
    for f in FOLDS:
        for s in SEEDS:
            run = video_run(s, f)
            z = np.load(rd(os.path.join(PATCHED, run, f"val_ep{V_EPOCH:02d}.npz")), allow_pickle=True)
            for i, p in enumerate(z["path"]):
                p = str(p)
                e = out.setdefault(p, dict(fold=f, y5=int(z["y5"][i]), rows={}))
                if e["fold"] != f:
                    raise SystemExit(f"{p} in two folds")
                e["rows"][run] = i
    return out


def stored_eeg():
    """{clip dir key: {run: p_logmean}} from the epoch-30 dumps."""
    import numpy as np
    out = defaultdict(dict)
    for f in FOLDS:
        for s in SEEDS:
            run = eeg_run(s, f)
            z = np.load(rd(os.path.join(ESUBJ, run, f"val_clip_ep{E_EPOCH:02d}.npz")), allow_pickle=True)
            for p, v in zip(z["path"], z["p_logmean"]):
                out[str(p)][run] = float(v)
    return out


def clip_frame0_gray(path):
    import cv2
    cap = cv2.VideoCapture(path)
    ok, fr = cap.read()
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return (cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if ok else None), n


def ncc2(a, b):
    import numpy as np
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    return float(a @ b / (np.sqrt((a @ a) * (b @ b)) + 1e-12))


def crop_resize(img, box):
    """BGR frame -> crop [y:y+h, x:x+w] -> RGB -> 224 x 224 INTER_AREA (the f16s224 rule, §4)."""
    import cv2
    x, y, w, h = box
    c = img[y:y + h, x:x + w]
    c = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
    return cv2.resize(c, (S, S), interpolation=cv2.INTER_AREA)


def decode_exact(path, idxs, pts_sorted):
    """{frame index: BGR} for exact frames (sequential decode from the keyframe at or before the first one)."""
    import av
    want = {int(pts_sorted[i]): int(i) for i in idxs}
    out = {}
    c = av.open(path)
    try:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        first = min(want)
        c.seek(first, stream=s, backward=True, any_frame=False)
        last = max(want)
        for fr in c.decode(s):
            if fr.pts in want:
                out[want[fr.pts]] = fr.to_ndarray(format="bgr24")
            if fr.pts is not None and fr.pts >= last:
                break
    finally:
        c.close()
    return out


def stepa_packets(name, relpath):
    """ffprobe packets for any raw file (cached under stepA/packets/)."""
    import numpy as np
    dst = out_path("stepA", "packets", name.replace(".mp4", ".npz"))
    if not os.path.exists(dst):
        p = raw_path(relpath)
        r = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=time_base",
                            "-of", "csv=p=0", p], capture_output=True, text=True, timeout=300)
        num, den = (int(x) for x in r.stdout.strip().split("/"))
        r = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts,flags",
                            "-of", "csv=p=0", p], capture_output=True, text=True, timeout=1800)
        pts, key = [], []
        for line in r.stdout.splitlines():
            a = line.split(",")
            if len(a) < 2 or a[0] in ("", "N/A"):
                continue
            pts.append(int(a[0]))
            key.append("K" in a[1])
        pts, key = np.array(pts, np.int64), np.array(key, bool)
        o = np.argsort(pts, kind="stable")
        atomic_npz(dst, pts=pts[o], key=key[o], tb=np.array([num, den], np.int64))
    z = np.load(dst)
    return z["pts"], z["key"], z["tb"][0] / z["tb"][1]


def cmd_stepa_prep(a):
    import numpy as np
    run_json("stepA", dict(sub="stepa-prep"))
    clips = {clip_key_of(r["animal"], r["session"], r["clip_dir"]): r
             for r in read_csv(os.path.join(AUDIT, "align", "clips.csv"))}
    cand1 = step1_candidates()
    oof = patched_oof()
    rows_files = {r["name"]: r for r in load_file_rows() if r["kind"] == "video"}
    boxes = {}
    for r in read_csv(os.path.join(AUDIT, "align", "crop_boxes.csv")):
        boxes[(r["animal"], r["camera"])] = tuple(int(r[k]) for k in ("x", "y", "w", "h"))
    defect = {s for L in DEFECT_STAMPS.values() for s in L}
    elig = defaultdict(lambda: defaultdict(list))
    why = Counter()
    for k, e in sorted(oof.items()):
        c = clips.get(k)
        if c is None:
            why["not in clips.csv"] += 1
            continue
        if k in cand1:
            why["step-1 candidate / exclusion"] += 1
            continue
        d = fnum(c["clip_dur_s"])
        if d is None or not (50 <= d <= 75):
            why["duration outside 50-75 s"] += 1
            continue
        rv = os.path.basename(c["raw_video"])          # clips.csv holds the relpath of the clip animal's copy
        fr = rows_files.get(rv)
        if (not rv or c["clip_in_file"] != "yes" or fr is None or fr["status"] == "error" or rv == UNREADABLE
                or rv.split(".")[-2] in defect or c["raw_status"] in ("error", "timing", "")):
            why["not inside one readable non-defect raw file"] += 1
            continue
        if (c["animal"], c["raw_camera"]) not in boxes:
            why["no crop box for the camera"] += 1
            continue
        elig[e["y5"]][c["animal"]].append(k)
    rng = random.Random(STEPA_SEED)
    quota = {0: 120, 1: 60, 2: 60, 3: 60, 4: 60}
    n_s5 = sum(len(v) for v in elig[4].values())
    if n_s5 < 60:
        quota[4] = n_s5
        quota[3] += 60 - n_s5
    order = {}
    for y in sorted(elig):
        for an in sorted(elig[y]):
            L = sorted(elig[y][an])
            rng.shuffle(L)
            elig[y][an] = L
        ans = sorted(elig[y])
        rng.shuffle(ans)
        order[y] = ans
    redrawn = []

    def validate(k):
        """r0 by NCC with clip frame 0 (>= 0.99), and the clip frame count; None -> redraw."""
        c = clips[k]
        full = os.path.join(FULL, k[len("data/"):])
        g0, n = clip_frame0_gray(full)
        if g0 is None or n <= 0:
            return None, "clip frame 0 unreadable"
        rv = os.path.basename(c["raw_video"])
        rel = c["raw_video"]
        pts, key, tb = stepa_packets(rv, rel)
        seek = fnum(c["info_seek_s"]) if fnum(c["info_seek_s"]) is not None else fnum(c["seek_pred_s"])
        kf = np.flatnonzero(key)
        tk = pts[kf] * tb
        j = int(np.searchsorted(tk, seek + 1e-9, "right") - 1)
        cands = [kf[x] for x in (j, j - 1) if x >= 0]
        box = boxes[(c["animal"], c["raw_camera"])]
        imgs = decode_keyframes_at(raw_path(rel), [pts[i] for i in cands])
        import cv2
        best = None
        for i in cands:
            im = imgs.get(int(pts[i]))
            if im is None:
                continue
            x, y, w, h = box
            g = cv2.cvtColor(im[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
            if g.shape != g0.shape:
                return None, f"crop {g.shape} != clip frame {g0.shape}"
            v = ncc2(g, g0)
            if best is None or v > best[1]:
                best = (int(i), v)
        if best is None or best[1] < 0.99:
            return None, f"NCC {best[1] if best else None} < 0.99"
        return dict(key=k, animal=c["animal"], fold=oof[k]["fold"], y5=oof[k]["y5"], raw_video=rv,
                    relpath=rel, camera=c["raw_camera"], box=box, seek=seek, r0=best[0], ncc=best[1],
                    r0_is_before_seek_kf=bool(best[0] != cands[0]), n=n, dur=fnum(c["clip_dur_s"]),
                    clip_start=c["info_clip_start"], raw_edf=c["raw_edf"], edf_start_local=c["edf_start_local"]), None

    picked = {y: [] for y in quota}
    for y in sorted(quota):
        ptr = {an: 0 for an in order[y]}
        while len(picked[y]) < quota[y]:
            progressed = False
            for an in order[y]:
                if len(picked[y]) >= quota[y]:
                    break
                while ptr[an] < len(elig[y][an]):
                    k = elig[y][an][ptr[an]]
                    ptr[an] += 1
                    item, err = validate(k)
                    if item is None:
                        redrawn.append(dict(key=k, reason=err))
                        continue
                    picked[y].append(item)
                    progressed = True
                    if len(picked[y]) % 20 == 0:
                        log(f"[stepA] y5={y}: {len(picked[y])}/{quota[y]} validated")
                    break
            if not progressed:
                break
        if len(picked[y]) < quota[y]:
            log(f"[stepA] stratum y5={y}: only {len(picked[y])} of {quota[y]} valid clips")
        # remember the pointers for the partner top-up
        order[y] = (order[y], ptr)
    # at least 15 clips each from RN199, RN204, RN242: top up by replacing the last-drawn non-partner clip of the
    # same stratum (quotas kept), strata in order non-seizure, S2..S5, one clip per stratum per round
    topups = []
    for pa in PARTNERS:
        rounds = 0
        while sum(it["animal"] == pa for y in picked for it in picked[y]) < 15 and rounds < 50:
            rounds += 1
            for y in sorted(quota):
                if sum(it["animal"] == pa for yy in picked for it in picked[yy]) >= 15:
                    break
                ans, ptr = order[y]
                if pa not in ptr:
                    continue
                item = None
                while ptr[pa] < len(elig[y][pa]):
                    k = elig[y][pa][ptr[pa]]
                    ptr[pa] += 1
                    item, err = validate(k)
                    if item is not None:
                        break
                    redrawn.append(dict(key=k, reason=err))
                if item is None:
                    continue
                for j in range(len(picked[y]) - 1, -1, -1):
                    if picked[y][j]["animal"] not in PARTNERS:
                        gone = picked[y].pop(j)
                        picked[y].append(item)
                        topups.append(dict(stratum=y, added=item["key"], removed=gone["key"]))
                        break
    items = [it for y in sorted(picked) for it in picked[y]]
    log(f"[stepA] {len(items)} clips: " + ", ".join(f"y5={y} {len(picked[y])}" for y in sorted(picked)) +
        f"; partners " + ", ".join(f"{p} {sum(it['animal'] == p for it in items)}" for p in PARTNERS) +
        f"; redrawn {len(redrawn)}; top-up swaps {len(topups)}")
    # arms K and E frames
    N = len(items)
    fk = np.lib.format.open_memmap(out_path("stepA", "frames_K.npy"), "w+", np.uint8, (N, T, S, S, 3))
    fe = np.lib.format.open_memmap(out_path("stepA", "frames_E.npy"), "w+", np.uint8, (N, T, S, S, 3))
    for i, it in enumerate(items):
        pts, key, tb = stepa_packets(it["raw_video"], it["relpath"])
        kf = np.flatnonzero(key)
        pos = it["r0"] + sample_frame_indices(it["n"], T)
        it["pos_exact"] = pos.tolist()
        sn = snap_to_keyframes(pos, kf)
        it["pos_key"] = sn.tolist()
        it["max_snap_frames"] = int(np.abs(sn - pos).max())
        kimg = decode_keyframes_at(raw_path(it["relpath"]), [pts[j] for j in sn])
        eimg = decode_exact(raw_path(it["relpath"]), pos, pts)
        miss_k = [int(j) for j in sn if int(pts[j]) not in kimg]
        miss_e = [int(j) for j in pos if int(j) not in eimg]
        if miss_k or miss_e:
            raise SystemExit(f"{it['key']}: frames not decoded K {miss_k} E {miss_e}")
        for t_, j in enumerate(sn):
            fk[i, t_] = crop_resize(kimg[int(pts[j])], it["box"])
        for t_, j in enumerate(pos):
            fe[i, t_] = crop_resize(eimg[int(j)], it["box"])
        if i % 40 == 0:
            log(f"[stepA] frames {i + 1}/{N}")
    fk.flush()
    fe.flush()
    # EEG arm R: the clip's samples read from the raw EDF at its Clip start on the EDF clock (lag 0 samples), with
    # the clip EDF's sample count
    segs, offs, eeg_rows = [], [0], []
    for it in items:
        c = clips[it["key"]]
        cdir = os.path.join(FULL, it["key"][len("data/"):-len("/video.mp4")])
        hc = AC.edf_header(os.path.join(cdir, "eeg.edf"))
        chc = edf_channel_rule(hc["labels"])
        nclip = hc["nrec_file"] * hc["nsamp"][chc]
        rawedf = raw_path(c["raw_edf"])
        hr = AC.edf_header(rawedf)
        chr_ = edf_channel_rule(hr["labels"])
        cs = AC.parse_dt(c["info_clip_start"])
        t0 = max(0.0, (cs - hr["start"]).total_seconds())
        s0 = int(round(t0 * FS))
        sig = edf_samples(hr, chr_, s0, nclip)
        dig = edf_samples(hr, chr_, s0, nclip, digital=True)
        csig = edf_samples(hc, chc, 0, nclip)
        row = dict(key=it["key"], raw_edf=c["raw_edf"], raw_label=hr["labels"][chr_], clip_label=hc["labels"][chc],
                   s0=s0, n=nclip, raw_fs=hr["nsamp"][chr_] / hr["rdur"], clip_fs=hc["nsamp"][chc] / hc["rdur"])
        if sig is None or csig is None:
            row["error"] = "samples not readable"
            eeg_rows.append(row)
            segs.append(np.zeros((0, WIN), np.float32))
            offs.append(offs[-1])
            continue
        cc = np.corrcoef(sig, csig)[0, 1] if sig.std() > 0 and csig.std() > 0 else float("nan")
        # best lag of the clip samples against the raw ones within +-5 samples (information only)
        lags = {}
        for L in range(-5, 6):
            rs = edf_samples(hr, chr_, s0 + L, nclip)
            if rs is not None and rs.std() > 0 and csig.std() > 0:
                lags[L] = float(np.corrcoef(rs, csig)[0, 1])
        row.update(corr_raw_vs_clip=float(cc), max_abs_diff_rel=float(np.abs(sig - csig).max() / (np.ptp(sig) + 1e-30)),
                   best_lag=max(lags, key=lags.get) if lags else None, longest_run=longest_run(dig),
                   dropout=bool(longest_run(dig) >= DROPOUT_RUN))
        eeg_rows.append(row)
        sg = eeg_preprocess(sig)
        segs.append(sg)
        offs.append(offs[-1] + len(sg))
    atomic_npz(out_path("stepA", "eeg_R.npz"), segs=np.concatenate(segs), offsets=np.array(offs, np.int64))
    atomic_json(out_path("stepA", "items.json"), dict(
        created=time.strftime("%F %T"), seed=STEPA_SEED, quota=quota, eligible_by_stratum={
            str(y): {an: len(v) for an, v in elig[y].items()} for y in elig}, ineligible=dict(why),
        redrawn=redrawn, topups=topups, items=items, eeg=eeg_rows))
    log(f"[stepA] prep done: {N} clips; EEG raw-vs-clip corr median "
        f"{np.nanmedian([r.get('corr_raw_vs_clip', np.nan) for r in eeg_rows]):.6f}; dropout clips "
        f"{sum(r.get('dropout', False) for r in eeg_rows)}")


command("stepa-prep")(cmd_stepa_prep)


# ============================================================================ GPU jobs: ledger, stop signal

STOP = {"flag": False}


def _on_signal(sig, _frm):
    if not STOP["flag"]:
        log(f"signal {signal.Signals(sig).name}: finishing the current batch, then exit 3")
    STOP["flag"] = True


def gpu_ledger(event, **kw):
    rec = dict(event=event, job=os.environ.get("SLURM_JOB_ID"), host=os.uname().nodename, at=time.time(),
               at_iso=time.strftime("%F %T"), **kw)
    with open(out_path("budget", "gpu.jsonl"), "a") as f:
        f.write(json.dumps(rec, default=_json_default) + "\n")
        f.flush()
        os.fsync(f.fileno())


def gpu_used_hours(exclude_attempt=None):
    """GPU time used so far from budget/gpu.jsonl: per attempt (a 'start' record), up to its 'end' record or its
    last heartbeat ('beat'); an attempt killed without a heartbeat counts from start to its first later record of
    any job (or now, if it is the running one)."""
    p = out_path("budget", "gpu.jsonl")
    if not os.path.exists(p):
        return 0.0
    recs = [json.loads(l) for l in open(p) if l.strip()]
    tot = 0.0
    for i, r in enumerate(recs):
        if r["event"] != "start" or r.get("at") == exclude_attempt:
            continue
        end = None
        for q in recs[i + 1:]:
            if q.get("job") == r.get("job") and q["event"] == "start":
                break
            if q.get("job") == r.get("job") and q["event"] in ("end", "beat"):
                end = q["at"]
        if end is None:
            end = r["at"]
        tot += max(0.0, end - r["at"] + (r.get("import_s") or 0.0))
    return tot / 3600


def gpu_device():
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    dev = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    if cap[0] * 10 + cap[1] > 90:
        raise SystemExit(f"GPU {name} sm_{cap[0]}{cap[1]} is not supported by torch 2.4.1+cu121")
    return dev, name


def cmd_stepa_gpu(a, nets=None):
    """Step A arms C / K / E / R through the clips' OOF networks. nets = (V, E, dev, gname) when called from the
    fused scan-gpu job (models already loaded, ledger kept by the caller)."""
    import numpy as np
    t_imp = time.time()
    import torch
    test = bool(os.environ.get("SCAN_TEST_CPU"))          # code test on CPU: 12 clips, no ledger, test outputs
    if nets is not None:
        V, E, dev, gname = nets
    elif test:
        dev, gname = torch.device("cpu"), "cpu-test"
        torch.set_num_threads(8)
    else:
        signal.signal(signal.SIGTERM, _on_signal)
        signal.signal(signal.SIGUSR1, _on_signal)
        dev, gname = gpu_device()
        gpu_ledger("start", stage="stepa-gpu", gpu=gname,
                   import_s=round(time.time() - a.t_launch, 1) if a.t_launch else None)
    t0 = time.time()
    it = json.load(open(rd(out_path("stepA", "items.json"))))
    items = it["items"][:12] if test else it["items"]
    fk = np.load(rd(out_path("stepA", "frames_K.npy")), mmap_mode="r")
    fe = np.load(rd(out_path("stepA", "frames_E.npy")), mmap_mode="r")
    if os.path.exists(rd(out_path("stepA", "frames_C.npy"))):
        fc = np.load(rd(out_path("stepA", "frames_C.npy")), mmap_mode="r")      # the same rows, copied
    else:
        meta = json.load(open(os.path.join(F16, "index.json")))
        row16 = {p: i for i, p in enumerate(meta["paths"])}
        mm16 = np.memmap(os.path.join(F16, "frames.u8"), dtype=np.uint8, mode="r",
                         shape=(meta["n"], meta["frames"], meta["size"], meta["size"], 3))
        fc = np.stack([mm16[row16[x["key"]]] for x in items])
    er = np.load(rd(out_path("stepA", "eeg_R.npz")))
    vruns = [video_run(s, f) for f in FOLDS for s in SEEDS]
    eruns = [eeg_run(s, f) for f in FOLDS for s in SEEDS]
    if nets is None:
        V = VideoNets(vruns, dev)
        E = EEGNets(eruns, dev)
    t_load = time.time() - t0
    out = {}
    by_fold = defaultdict(list)
    for i, x in enumerate(items):
        by_fold[x["fold"]].append(i)
    for arm, F in (("C", fc), ("K", fk), ("E", fe)):
        for f, idx in by_fold.items():
            for b in range(0, len(idx), 32):
                ii = idx[b:b + 32]
                xx = V.normed(np.asarray(F[ii]))
                for s in SEEDS:
                    pr = V.predict(xx, video_run(s, f))
                    for t in TASKS:
                        arr = out.setdefault(f"{arm}|{t}|s{s}", np.full((len(items), 5 if t == "g5" else 3), np.nan,
                                                                          np.float32))
                        arr[ii] = pr[t]
    # EEG arm R: every clip through its 3 OOF TCNs
    segs, offs = er["segs"], er["offsets"]
    pe = {}
    for s in SEEDS:
        p = np.full(len(items), np.nan)
        for f, idx in by_fold.items():
            for i in idx:
                sg = segs[offs[i]:offs[i + 1]]
                if len(sg) == 0:
                    continue
                Pw = E.predict_windows(sg, eeg_run(s, f))
                p[i] = E.clip_scores(Pw, np.array([0, len(Pw)]))[0]
        pe[f"R|s{s}"] = p
    if test:
        atomic_npz(out_path("stepA", "test_preds_cpu.npz"), **{k: v[:12] for k, v in out.items()},
                   **{k: v[:12] for k, v in pe.items()})
        log("[stepA-gpu] CPU test done")
        return
    # throughput benchmark (information for the pilot): X3D-M forwards/s at batch 32 and 64, and TCN windows/s
    bench = {}
    torch.cuda.synchronize()
    for bs in (32, 64):
        xb = V.normed(np.asarray(fk[:bs]))
        run = vruns[0]
        V.predict(xb, run)
        torch.cuda.synchronize()
        tb = time.time()
        nrep = 0
        while time.time() - tb < 15:
            V.predict(xb, run)
            nrep += 1
        torch.cuda.synchronize()
        bench[f"x3d_fwd_per_s_bs{bs}"] = nrep * bs / (time.time() - tb)
    tb = time.time()
    xb = np.asarray(fk[:64])
    nrep = 0
    while time.time() - tb < 15:
        xx = V.normed(xb)
        V.predict(xx, vruns[0])
        nrep += 1
    torch.cuda.synchronize()
    bench["x3d_fwd_per_s_bs64_incl_upload_norm"] = nrep * 64 / (time.time() - tb)
    # the same with cudnn.benchmark (information only: the step-A predictions above use the defaults)
    torch.backends.cudnn.benchmark = True
    xb = V.normed(np.asarray(fk[:64]))
    for _ in range(3):
        V.predict(xb, vruns[0])
    torch.cuda.synchronize()
    tb = time.time()
    nrep = 0
    while time.time() - tb < 15:
        V.predict(xb, vruns[0])
        nrep += 1
    torch.cuda.synchronize()
    bench["x3d_fwd_per_s_bs64_cudnn_benchmark"] = nrep * 64 / (time.time() - tb)
    torch.backends.cudnn.benchmark = False
    sg = np.random.default_rng(0).standard_normal((19 * 512, WIN)).astype(np.float32)
    E.predict_windows(sg, eruns[0])
    torch.cuda.synchronize()
    tb = time.time()
    nrep = 0
    while time.time() - tb < 10:
        E.predict_windows(sg, eruns[0])
        nrep += 1
    torch.cuda.synchronize()
    bench["tcn_windows_per_s"] = nrep * len(sg) / (time.time() - tb)
    atomic_npz(out_path("stepA", "preds.npz"), **out, **pe)
    info = dict(gpu=gname, secs_load=round(t_load, 1), secs_total=round(time.time() - t0, 1), bench=bench,
                torch=torch.__version__, cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
                allow_tf32_cudnn=bool(torch.backends.cudnn.allow_tf32),
                allow_tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32),
                video_checks=V.checks, eeg_checks=E.checks, import_s=round(t0 - t_imp, 1))
    atomic_json(out_path("stepA", "gpu_run.json"), info)
    if nets is None:
        gpu_ledger("end", stage="stepa-gpu", gpu=gname, bench=bench)
    log(f"[stepA-gpu] done on {gname}: {json.dumps(bench)}")


def cmd_stepa_eval(a):
    import numpy as np
    test = bool(os.environ.get("SCAN_TEST_CPU"))
    it = json.load(open(rd(out_path("stepA", "items.json"))))
    items = it["items"][:12] if test else it["items"]
    if test:
        it["eeg"] = it["eeg"][:12]
    P = np.load(out_path("stepA", "test_preds_cpu.npz" if test else "preds.npz"))
    gates = json.load(open(rd(out_path("s0", "gates.json"))))
    st_eeg = stored_eeg()
    N = len(items)
    stored = {}
    for f in FOLDS:
        for s in SEEDS:
            run = video_run(s, f)
            z = np.load(rd(os.path.join(PATCHED, run, f"val_ep{V_EPOCH:02d}.npz")), allow_pickle=True)
            pos = {str(p): i for i, p in enumerate(z["path"])}
            for t in TASKS:
                stored[(run, t)] = (pos, z[f"probs_{t}"])
    y5 = np.array([x["y5"] for x in items])
    res = dict(n_clips=N, arms={}, eeg={}, created=time.strftime("%F %T"))
    for arm in ("C", "K", "E"):
        R = {}
        for t in TASKS:
            d_all, agree, dsz, flag_agree, flag_n = [], [], [], 0, 0
            f1s = []
            for s in SEEDS:
                pa = P[f"{arm}|{t}|s{s}"]
                st = np.stack([stored[(video_run(s, x["fold"]), t)][1][stored[(video_run(s, x["fold"]), t)][0][x["key"]]]
                               for x in items]).astype(np.float64)
                d = np.abs(pa.astype(np.float64) - st).max(1)
                d_all.append(d)
                agree.append(pa.argmax(1) == st.argmax(1))
                dsz.append((1 - pa[:, 0]) - (1 - st[:, 0]))
                yy = y5 if t == "g5" else np.array([0, 1, 1, 2, 2])[y5]
                f1s.append(macro_f1(yy, pa.argmax(1), 5 if t == "g5" else 3))
                # EGRG flag agreement with the stored EEG and G_all of each (video seed, EEG seed) pair
                for se in SEEDS:
                    pe = np.array([st_eeg[x["key"][:-len("/video.mp4")]][eeg_run(se, x["fold"])] for x in items])
                    g = gates["gates"][f"{t}|{video_run(s, 'F')}|{eeg_run(se, 'F')}"]["all"]
                    fa = egrg_decide(pa.astype(np.float64), gate_q(g, pa.astype(np.float64), pe)) > 0
                    fs = egrg_decide(st, gate_q(g, st, pe)) > 0
                    flag_agree += int((fa == fs).sum())
                    flag_n += len(fa)
            d_all = np.concatenate(d_all)
            agree = np.concatenate(agree)
            R[t] = dict(rows=int(len(d_all)), argmax_agree=float(agree.mean()), median_abs_dp=float(np.median(d_all)),
                        p95_abs_dp=float(np.percentile(d_all, 95)), mean_signed_dPsz=float(np.concatenate(dsz).mean()),
                        egrg_flag_agree=flag_agree / flag_n, macro_f1_by_seed=f1s,
                        passes=bool(agree.mean() >= 0.97 and np.median(d_all) <= 0.05))
        R["passes"] = bool(all(R[t]["passes"] for t in TASKS))
        res["arms"][arm] = R
    # stored macro-F1 on the same clips (reference for the arms' macro-F1)
    ref = {}
    for t in TASKS:
        yy = y5 if t == "g5" else np.array([0, 1, 1, 2, 2])[y5]
        ref[t] = [macro_f1(yy, np.stack([stored[(video_run(s, x["fold"]), t)][1][stored[(video_run(s, x["fold"]), t)][0][x["key"]]]
                                         for x in items]).argmax(1), 5 if t == "g5" else 3) for s in SEEDS]
    res["stored_macro_f1_by_seed"] = ref
    # EEG arm R
    d, ag = [], []
    for s in SEEDS:
        pr = P[f"R|s{s}"]
        ps = np.array([st_eeg[x["key"][:-len("/video.mp4")]][eeg_run(s, x["fold"])] for x in items])
        ok = np.isfinite(pr)
        d.append(np.abs(pr[ok] - ps[ok]))
        ag.append((pr[ok] > 0.5) == (ps[ok] > 0.5))
    d, ag = np.concatenate(d), np.concatenate(ag)
    er = it["eeg"]
    res["eeg"] = dict(rows=int(len(d)), missing=int(sum(1 for r in er if "error" in r)),
                      argmax_agree=float(ag.mean()), median_abs_dp=float(np.median(d)), p95_abs_dp=float(np.percentile(d, 95)),
                      max_abs_dp=float(d.max()), passes=bool(ag.mean() >= 0.99 and np.median(d) <= 0.01),
                      raw_vs_clip_corr=dict(min=float(np.nanmin([r.get("corr_raw_vs_clip", np.nan) for r in er])),
                                            median=float(np.nanmedian([r.get("corr_raw_vs_clip", np.nan) for r in er]))),
                      best_lag_counts=dict(Counter(str(r.get("best_lag")) for r in er)),
                      dropout_rule_would_remove=int(sum(r.get("dropout", False) for r in er)),
                      raw_labels=dict(Counter(r.get("raw_label") for r in er)),
                      clip_labels=dict(Counter(r.get("clip_label") for r in er)))
    c_ok = res["arms"]["C"]["g3"]["median_abs_dp"] <= 0.01 and res["arms"]["C"]["g5"]["median_abs_dp"] <= 0.01
    res["C_noise_floor_ok"] = bool(c_ok)
    K, Ea, Rr = res["arms"]["K"]["passes"], res["arms"]["E"]["passes"], res["eeg"]["passes"]
    if not c_ok:
        decision = "STOP: arm C is not within median |dp| <= 0.01 (the inference code is wrong)"
        mode = None
    elif not Rr:
        decision = "STOP: EEG arm R fails"
        mode = None
    elif K:
        decision, mode = "scan in mode K (keyframe-snapped)", "K"
    elif Ea:
        decision, mode = "K fails, E passes: scan in mode E (exact frames, sequential decode)", "E"
    else:
        decision, mode = "STOP: K and E both fail", None
    res["decision"] = decision
    res["mode"] = mode
    if test:
        atomic_json(out_path("stepA", "test_stepA_cpu.json"), res)
        log(f"[stepA] CPU test evaluation: {json.dumps({k: v['passes'] for k, v in res['arms'].items()})}")
        return
    atomic_json(out_path("stepA", "stepA.json"), res)
    atomic_json(out_path("stepA", "decision.json"), dict(decision=decision, mode=mode, at=time.strftime("%F %T"),
                                                          K=res["arms"]["K"], E=res["arms"]["E"], C=res["arms"]["C"],
                                                          R=res["eeg"]))
    for arm in ("C", "K", "E"):
        for t in TASKS:
            r = res["arms"][arm][t]
            log(f"[stepA] {arm} {t}: argmax {r['argmax_agree']:.4f}  median|dp| {r['median_abs_dp']:.4f}  "
                f"p95 {r['p95_abs_dp']:.4f}  mean dPsz {r['mean_signed_dPsz']:+.4f}  EGRG flag agree "
                f"{r['egrg_flag_agree']:.4f}  F1 {np.round(r['macro_f1_by_seed'], 3).tolist()}")
    log(f"[stepA] stored F1 {json.dumps({t: np.round(v, 3).tolist() for t, v in ref.items()})}")
    log(f"[stepA] R: argmax {res['eeg']['argmax_agree']:.4f} median|dp| {res['eeg']['median_abs_dp']:.5f} "
        f"p95 {res['eeg']['p95_abs_dp']:.4f}; dropout would remove {res['eeg']['dropout_rule_would_remove']}")
    log(f"[stepA] DECISION: {decision}")


def macro_f1(y, p, K):
    import numpy as np
    f1 = []
    for k in range(K):
        tp_ = int(((y == k) & (p == k)).sum())
        fp_ = int(((y != k) & (p == k)).sum())
        fn_ = int(((y == k) & (p != k)).sum())
        if tp_ + fp_ + fn_ == 0:
            continue
        f1.append(2 * tp_ / (2 * tp_ + fp_ + fn_))
    return float(np.mean(f1))


command("stepa-gpu")(cmd_stepa_gpu)
command("stepa-eval")(cmd_stepa_eval)


# ============================================================================ S4: decode (CPU)

DECODE_VERSION = 1


def decode_units(pilot=False):
    """Camera files with at least one candidate window of either animal, in time order per camera; the S3 pilot is
    the first RN203-RN204 day folder with >= 20 h of candidate windows for both RN203 and RN204."""
    import numpy as np
    tj = load_tables_json()
    units = []
    per_day = defaultdict(lambda: defaultdict(int))
    tabs = {a_: load_table(a_) for a_ in ANIMALS}
    for cam, files in tj["cameras"].items():
        d = CAMS[cam]
        for fi, v in enumerate(files):
            n = {a_: int(((tabs[a_]["fidx"] == fi) & (tabs[a_]["reason"] == 0)).sum()) for a_ in (d["left"], d["right"])}
            if sum(n.values()) == 0:
                continue
            units.append(dict(camera=cam, fidx=fi, name=v["name"], n=n, day=v["day_folder"][d["left"]]))
            if cam == "RN203-RN204":
                for a_ in n:
                    per_day[v["day_folder"]["RN203"]][a_] += n[a_]
    pilot_day = next(dday for dday in sorted(per_day, key=lambda x: min(u["fidx"] for u in units
                                                                          if u["camera"] == "RN203-RN204"
                                                                          and u["day"] == x))
                     if per_day[dday]["RN203"] >= 1200 and per_day[dday]["RN204"] >= 1200)
    pil = [u for u in units if u["camera"] == "RN203-RN204" and u["day"] == pilot_day]
    if pilot:
        return pil, pilot_day
    rest = [u for u in units if u not in pil]
    return pil + rest, pilot_day


def shard_bytes():
    tot = 0
    root = out_path("shards", "x")
    for dp, _, fs in os.walk(os.path.dirname(root)):
        for f in fs:
            if f.endswith(".u8") or f.endswith(".eeg.npy") or ".partial" in f:
                try:
                    tot += os.path.getsize(os.path.join(dp, f))
                except OSError:
                    pass
    return tot


def stem_of(name):
    return name.replace(".mp4", "")


def edf_cache():
    tj = load_tables_json()
    return {a_: tj["edfs"][a_]["edfs"] for a_ in ANIMALS}


def decode_unit(u, mode, boxes, box_sha, edfs, tabs, threads, frame_root=None):
    """One camera file: EEG windows (dropout rule), keyframes (motion of every keyframe per box, the windows'
    16 frames per animal), then per animal a frame shard + EEG array under frame_root (default ttg_scan/shards/;
    node-local /dev/shm in the fused GPU job) and its meta json under ttg_scan/shards/ (kept)."""
    import numpy as np
    import av
    import cv2
    cv2.setNumThreads(1)
    cam = u["camera"]
    d = CAMS[cam]
    tj_files = load_tables_json()["cameras"][cam]
    v = tj_files[u["fidx"]]
    stem = stem_of(v["name"])
    done_p = out_path("decode", f"{stem}.json")
    key = dict(version=DECODE_VERSION, mode=mode, boxes=box_sha, name=v["name"], size=v["size"],
               prereg=sha256(PREREG)[:16])
    if os.path.exists(done_p):
        dj = json.load(open(done_p))
        if dj.get("key") == key:
            return f"{stem}: done"
    t0 = time.time()
    pts, key_, tb = load_packets(v["name"])
    kf = np.flatnonzero(key_)
    tt = v["t0"] + pts * tb
    animals = (d["left"], d["right"])
    # ---- EEG windows (and the dropout rule) per animal
    wins, eeg, dropout = {}, {}, {}
    hdr = {}
    for a_ in animals:
        tab = tabs[a_]
        idx = np.flatnonzero((tab["fidx"] == u["fidx"]) & (tab["reason"] == 0))
        segs, keep, drop = [], [], []
        for i in idx:
            e = edfs[a_][int(tab["edf"][i])]
            if e["relpath"] not in hdr:
                hdr[e["relpath"]] = AC.edf_header(raw_path(e["relpath"]))
            h = hdr[e["relpath"]]
            dig = edf_samples(h, e["ch"], int(tab["k0"][i]), 60 * FS, digital=True)
            if dig is None:
                raise SystemExit(f"{a_} minute {epoch_iso(tab['minute'][i])}: EDF samples not readable")
            if longest_run(dig) >= DROPOUT_RUN:
                drop.append(int(i))
                continue
            g = (h["pmax"][e["ch"]] - h["pmin"][e["ch"]]) / (h["dmax"][e["ch"]] - h["dmin"][e["ch"]])
            sig = (dig.astype(np.float64) - h["dmin"][e["ch"]]) * g + h["pmin"][e["ch"]]
            sg = eeg_preprocess(sig)
            if sg.shape != (NWIN, WIN):
                raise SystemExit(f"{a_}: {sg.shape} EEG windows")
            segs.append(sg)
            keep.append(int(i))
        wins[a_] = np.array(keep, np.int64)
        eeg[a_] = np.stack(segs) if segs else np.zeros((0, NWIN, WIN), np.float32)
        dropout[a_] = drop
    # ---- frames: every keyframe (motion for the clock check) + the windows' 16 frames per animal
    need = {}
    for a_ in animals:
        tab = tabs[a_]
        if mode == "K":
            need[a_] = tab["kf16"][wins[a_]]                  # file frame indices (snapped, from S0)
        else:
            need[a_] = np.stack([tab["first"][i] + sample_frame_indices(int(tab["nfr"][i]), T) for i in wins[a_]]) \
                if len(wins[a_]) else np.zeros((0, T), np.int64)
    need_set = {a_: set(int(x) for x in need[a_].ravel()) for a_ in animals}
    crops = {a_: {} for a_ in animals}
    # motion of every keyframe is needed for the clock check; when the `motion` stage has already written it
    # for this file, only the windows' keyframes are decoded (each keyframe decodes on its own, so the frames
    # are the same either way) and the motion file is left as it is
    motion_done = os.path.exists(out_path("motion", f"{stem}.npz"))
    kf_set = set(int(x) for x in kf) if not motion_done else set()
    mot_pts, mot = [], []
    prev = None
    pos_of_pts = {int(p): i for i, p in enumerate(pts)} if mode == "E" else None
    c = av.open(raw_path(v["relpath"]))
    n_dec_err = 0
    try:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        s.thread_count = threads
        cc = s.codec_context

        def handle(fr, fidx):
            nonlocal prev
            img = fr.to_ndarray(format="bgr24")
            if fidx in kf_set:
                gs = []
                for a_ in animals:
                    x, y, w, h = boxes[a_]
                    g = cv2.cvtColor(img[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
                    gs.append(cv2.resize(g, (w // 2, h // 2), interpolation=cv2.INTER_AREA).astype(np.int16))
                if prev is not None:
                    mot.append([float(np.abs(g1 - g0).mean()) for g1, g0 in zip(gs, prev)])
                else:
                    mot.append([np.nan] * len(animals))
                mot_pts.append(int(pts[fidx]))
                prev = gs
            for a_ in animals:
                if fidx in need_set[a_]:
                    crops[a_][fidx] = crop_resize(img, boxes[a_])

        if mode == "K":
            kpts = {int(pts[i]): int(i) for i in kf
                    if not motion_done or any(int(i) in need_set[a_] for a_ in animals)}
            for pk in c.demux(s):
                if pk.pts is None or int(pk.pts) not in kpts:
                    continue
                try:
                    for fr in cc.decode(pk):
                        if int(fr.pts) in kpts:
                            handle(fr, kpts[int(fr.pts)])
                except Exception:          # an undecodable keyframe: its windows get 'decode error'
                    n_dec_err += 1
            for fr in cc.decode(None):
                if int(fr.pts) in kpts:
                    handle(fr, kpts[int(fr.pts)])
        else:
            for fr in c.decode(s):
                i = pos_of_pts.get(int(fr.pts))
                if i is not None and (i in kf_set or any(i in need_set[a_] for a_ in animals)):
                    handle(fr, i)
    finally:
        c.close()
    # ---- motion (all keyframes, per box) for the clock check
    if not motion_done:
        atomic_npz(out_path("motion", f"{stem}.npz"), kf_pts=np.array(mot_pts, np.int64),
               kf_time=v["t0"] + np.array(mot_pts, np.float64) * tb, motion=np.array(mot, np.float32).T.reshape(len(animals), -1) if mot else np.zeros((len(animals), 0), np.float32),
                   animals=np.array(animals), boxes=np.array([boxes[a_] for a_ in animals]))
    # ---- shards: one per animal (frames uint8 (n, 16, 224, 224, 3) + EEG (n, 19, 750) float32 + meta)
    rec = dict(key=key, camera=cam, file=v["name"], secs=None, animals={})
    for a_ in animals:
        tab = tabs[a_]
        ok_rows, err_rows = [], []
        for j, i in enumerate(wins[a_]):
            if all(int(x) in crops[a_] for x in need[a_][j]):
                ok_rows.append(j)
            else:
                err_rows.append(int(i))
        n = len(ok_rows)
        nbytes = n * T * S * S * 3 + n * NWIN * WIN * 4
        waited = 0
        while frame_root is None and n and shard_bytes() + nbytes > SHARD_CAP_BYTES:
            if waited % 600 == 0:
                log(f"{stem} {a_}: shard cap {SHARD_CAP_BYTES / 1e9:.0f} GB reached; waiting")
            time.sleep(30)
            waited += 30
        meta_base = out_path("shards", a_, stem)
        if frame_root is None:
            base = meta_base
        else:
            base = os.path.join(frame_root, a_, stem)
            os.makedirs(os.path.dirname(base), exist_ok=True)
        fr_p = base + ".u8"
        if n:
            mm = np.memmap(fr_p + ".partial", dtype=np.uint8, mode="w+", shape=(n, T, S, S, 3))
            for r, j in enumerate(ok_rows):
                for t_, x in enumerate(need[a_][j]):
                    mm[r, t_] = crops[a_][int(x)]
            mm.flush()
            del mm
            os.replace(fr_p + ".partial", fr_p)
            np.save(base + ".eeg.partial.npy", eeg[a_][ok_rows])
            os.replace(base + ".eeg.partial.npy", base + ".eeg.npy")
        rows = wins[a_][ok_rows]
        meta = dict(animal=a_, camera=cam, file=v["name"], mode=mode, n=n, rows=rows.tolist(),
                    minutes=tab["minute"][rows].tolist(), shape=[n, T, S, S, 3], eeg_shape=[n, NWIN, WIN],
                    frames_sha1=None, dropout_rows=dropout[a_], decode_error_rows=err_rows, key=key,
                    frame_root=frame_root or "ttg_scan/shards", created=time.strftime("%F %T"))
        atomic_json(meta_base + ".json", meta)
        rec["animals"][a_] = dict(n=n, dropout=len(dropout[a_]), decode_error=len(err_rows))
    rec["secs"] = round(time.time() - t0, 1)
    rec["n_keyframes"] = len(kf)
    rec["n_keyframes_decoded"] = len(mot_pts) if not motion_done else None
    rec["motion_from_motion_stage"] = motion_done
    rec["n_decode_exceptions"] = n_dec_err
    atomic_json(done_p, rec)
    return (f"{stem}: " + ", ".join(f"{a_} {r['n']} (dropout {r['dropout']}, decode err {r['decode_error']})"
                                    for a_, r in rec["animals"].items()) +
            f"; keyframes {'motion stage' if motion_done else f'{len(mot_pts)}/{len(kf)}'}; {rec['secs']} s")


def cmd_decode(a):
    import numpy as np
    dp = out_path("stepA", "decision.json")
    if os.path.exists(dp):
        dec = json.load(open(dp))
        mode = dec.get("mode")
        if mode not in ("K", "E"):
            raise SystemExit(f"step A decision is {dec.get('decision')}: no scan")
    elif a.speculative_mode in ("K", "E"):
        # decoding only (no network sees a frame before the decision; infer refuses shards of the other mode)
        mode = a.speculative_mode
        log(f"step A decision not written yet: decoding speculatively in mode {mode}")
    else:
        raise SystemExit("no step A decision yet (pass --speculative_mode K to decode ahead of it)")
    boxes, box_sha = load_boxes()
    units, pilot_day = decode_units(pilot=a.pilot)
    if a.files:
        want = set(a.files.split(","))
        units = [u for u in units if u["name"] in want]
    if a.index is not None:
        units = units[a.index::a.stride]
    run_json(f"decode/run_{a.index}_{os.environ.get('SLURM_JOB_ID', 'local')}.json",
             dict(sub="decode", mode=mode, speculative=not os.path.exists(dp), n_units=len(units), pilot=a.pilot,
                  pilot_day=pilot_day, index=a.index, stride=a.stride))
    edfs = edf_cache()
    tabs = {a_: load_table(a_) for a_ in ANIMALS}
    signal.signal(signal.SIGTERM, _on_signal)
    for u in units:
        if STOP["flag"]:
            log("stopping on signal")
            sys.exit(3)
        try:
            log(decode_unit(u, mode, boxes, box_sha, edfs, tabs, a.threads))
        except SystemExit:
            raise
        except Exception:
            log(f"{u['name']}: FAILED\n{traceback.format_exc()}")
            raise


command("decode")(cmd_decode)


# ============================================================================ S4: clock check (CPU, motion only)

CLOCK_BLOCK_S = 3600
CLOCK_MIN_BLOCK_S = 600
STRONG = dict(r=0.10, z=4.0, margin=0.04, sharp=0.02)       # recut_clips.STRONG (is_strong)


def read_activity_utc(e, h):
    """(utc seconds of each 1-Hz Activity sample, values) of one EDF's complete records."""
    import numpy as np
    ia = e["act"]
    na = h["nsamp"][ia]
    if abs(na / h["rdur"] - 1.0) > 1e-9:
        raise SystemExit(f"{e['relpath']}: Activity is {na / h['rdur']} Hz")
    g = (h["pmax"][ia] - h["pmin"][ia]) / (h["dmax"][ia] - h["dmin"][ia])
    R = e["records"]
    vals = np.empty(R * na, np.float64)
    with open(raw_path(e["relpath"]), "rb") as fh:
        for rec in range(R):
            fh.seek(h["hdr"] + rec * h["recbytes"] + 2 * h["off"][ia])
            x = np.frombuffer(fh.read(2 * na), dtype="<i2").astype(np.float64)
            vals[rec * na:(rec + 1) * na] = (x - h["dmin"][ia]) * g + h["pmin"][ia]
    s = dt.datetime.fromisoformat(e["start_naive"])
    k = np.arange(R * na, dtype=np.float64)
    naive_s = (s - dt.datetime(1970, 1, 1)).total_seconds() + k
    fall = (FALL_NAIVE - dt.datetime(1970, 1, 1)).total_seconds()
    utc = naive_s + np.where(naive_s < fall, 5 * 3600.0, 6 * 3600.0)
    return utc, vals


def activity_series(animal):
    """{int utc second: log1p(Activity)} over every non-stub EDF of the animal (cached in clock/)."""
    import numpy as np
    p = out_path("clock", f"activity_{animal}.npz")
    if os.path.exists(p):
        z = np.load(p)
        return z["utc"], z["val"]
    U, Vv = [], []
    for e in edf_cache()[animal]:
        if e["stub"] or e["act"] is None:
            continue
        h = AC.edf_header(raw_path(e["relpath"]))
        u, v = read_activity_utc(e, h)
        U.append(u)
        Vv.append(np.log1p(np.maximum(v, 0)))
    u = np.concatenate(U).astype(np.int64)
    v = np.concatenate(Vv)
    o = np.argsort(u, kind="stable")
    atomic_npz(p, utc=u[o], val=v[o])
    return u[o], v[o]


def masked_ncc(a, m):
    """recut_clips.masked_ncc: Pearson r of template a against every window of m (NaN = missing)."""
    import numpy as np
    from scipy.signal import fftconvolve
    va, vm = np.isfinite(a).astype(float), np.isfinite(m).astype(float)
    a0, m0 = np.where(va > 0, a, 0.0), np.where(vm > 0, m, 0.0)

    def xc(x, y):
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


def clock_block(act_u, act_v, kt, km, b0, b1, dst_day):
    """One block: Activity (1 Hz, UTC) template over [b0, b1) against rank-transformed keyframe motion
    (interpolated to 1 Hz) at lags +-120 s (and +-3600 +- 120 s on 2023-11-05)."""
    import numpy as np
    from scipy.stats import rankdata
    N = int(b1 - b0)
    a = np.full(N, np.nan)
    i0, i1 = np.searchsorted(act_u, [b0, b1])
    a[act_u[i0:i1] - int(b0)] = act_v[i0:i1]
    fa = np.isfinite(a)
    res = dict(b0=epoch_iso(b0), secs=N, act_valid=int(fa.sum()), act_nonzero=int((np.nan_to_num(a) >= math.log1p(0.5)).sum()))
    if fa.sum() < 120 or np.nanstd(a) == 0:
        res["err"] = "activity_flat_or_missing"
        return res
    Lmax = 3600 + CLOCK_LAG_S if dst_day else CLOCK_LAG_S
    grid = np.arange(int(b0) - Lmax, int(b1) + Lmax, dtype=np.float64)
    ok = np.isfinite(km)
    m = np.full(len(grid), np.nan)
    inside = (grid >= kt[ok][0]) & (grid <= kt[ok][-1]) if ok.sum() > 1 else np.zeros(len(grid), bool)
    if inside.any():
        m[inside] = np.interp(grid[inside], kt[ok], km[ok])
    g = np.isfinite(m)
    if g.sum() < 120:
        res["err"] = "no_motion"
        return res
    mr = np.full(len(m), np.nan)
    mr[g] = rankdata(m[g]) / g.sum()
    r, n = masked_ncc(a, mr)                              # index s <-> lag s - Lmax
    lags = np.arange(len(r)) - Lmax
    allowed = np.abs(lags) <= CLOCK_LAG_S
    if dst_day:
        allowed |= np.abs(np.abs(lags) - 3600) <= CLOCK_LAG_S
    okk = allowed & (n >= 0.6 * fa.sum()) & np.isfinite(r)
    if okk.sum() < 60:
        res["err"] = "no_video_overlap"
        return res
    rr = np.where(okk, r, np.nan)
    k = int(np.nanargmax(rr))
    vals = rr[okk]
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) * 1.4826 + 1e-9
    r2 = rr.copy()
    r2[max(0, k - 15): k + 16] = np.nan
    second = float(np.nanmax(r2)) if np.isfinite(r2).any() else float("nan")
    side = np.concatenate([rr[max(0, k - 15): max(0, k - 3)], rr[k + 4: k + 16]])
    side = side[np.isfinite(side)]
    sharp = float(rr[k] - side.max()) if len(side) else float("nan")
    L = int(lags[k])
    z = (float(rr[k]) - med) / mad
    strong = bool(rr[k] >= STRONG["r"] and z >= STRONG["z"] and np.isfinite(second) and rr[k] - second >= STRONG["margin"]
                  and np.isfinite(sharp) and sharp >= STRONG["sharp"])
    res.update(lag=L, r=float(rr[k]), z=z, margin=float(rr[k] - second) if np.isfinite(second) else None, sharp=sharp,
               strong=strong, consistent=bool(abs(L - ACT_LAG_S) <= CLOCK_TOL_S),
               r_at_expected=float(rr[Lmax - 1]) if np.isfinite(rr[Lmax - 1]) else None)
    res["excluded"] = bool(strong and not res["consistent"])
    return res


def cmd_clock(a):
    """Per camera file and box, 1-h blocks from the file's first frame (§3). Uses motion and Activity only."""
    import numpy as np
    run_json("clock", dict(sub="clock"))
    tj = load_tables_json()
    out = dict(created=time.strftime("%F %T"), rule=dict(block_s=CLOCK_BLOCK_S, lags_s=CLOCK_LAG_S, tol_s=CLOCK_TOL_S,
                                                           expected_lag_s=ACT_LAG_S, strong=STRONG,
                                                           min_block_s=CLOCK_MIN_BLOCK_S),
               blocks=[], missing_motion=[])
    acts = {}
    for cam, files in tj["cameras"].items():
        d = CAMS[cam]
        for fi, v in enumerate(files):
            if v["kind"] != "ok":
                continue
            mp = out_path("motion", f"{stem_of(v['name'])}.npz")
            if not os.path.exists(mp):
                out["missing_motion"].append(v["name"])
                continue
            z = np.load(mp)
            kt = z["kf_time"]
            for ai, an in enumerate([str(x) for x in z["animals"]]):
                if an not in acts:
                    acts[an] = activity_series(an)
                au, av_ = acts[an]
                km = z["motion"][ai].astype(np.float64)
                t_first, t_last = float(kt[0]), float(kt[-1])
                b = t_first
                bi = 0
                while b < t_last:
                    e = min(b + CLOCK_BLOCK_S, t_last)
                    rec = dict(camera=cam, file=v["name"], fidx=fi, animal=an, block=bi, u0=b, u1=e)
                    if e - b < CLOCK_MIN_BLOCK_S:
                        rec.update(err="short_block", excluded=False)
                    else:
                        dday = dt.datetime.fromtimestamp(b, tz=UTC).date() == dt.date(2023, 11, 5) or \
                            dt.datetime.fromtimestamp(e, tz=UTC).date() == dt.date(2023, 11, 5)
                        rec.update(clock_block(au, av_, kt, km, math.floor(b), math.floor(e), dday))
                        rec.setdefault("excluded", False)
                    out["blocks"].append(rec)
                    b = e
                    bi += 1
    B = out["blocks"]
    summ = {}
    for an in ANIMALS:
        bb = [r for r in B if r["animal"] == an]
        summ[an] = dict(blocks=len(bb), measured=sum(1 for r in bb if "lag" in r),
                        strong=sum(1 for r in bb if r.get("strong")),
                        strong_consistent=sum(1 for r in bb if r.get("strong") and r.get("consistent")),
                        excluded=sum(1 for r in bb if r.get("excluded")),
                        excluded_lags=sorted(r["lag"] for r in bb if r.get("excluded")))
    out["summary"] = summ
    atomic_json(out_path("clock", "clock.json"), out)
    for an, sm in summ.items():
        log(f"[clock] {an}: blocks {sm['blocks']}, measured {sm['measured']}, strong {sm['strong']}, strong and "
            f"consistent {sm['strong_consistent']}, excluded {sm['excluded']} {sm['excluded_lags'][:10]}")
    if out["missing_motion"]:
        log(f"[clock] WARNING: {len(out['missing_motion'])} files without motion (not decoded yet)")


command("clock")(cmd_clock)


# ============================================================================ S4: inference (GPU)

def infer_runs(animal):
    if animal in PARTNERS:
        f = PARTNER_FOLD[animal]
        return [video_run(s, f) for s in SEEDS], [eeg_run(s, f) for s in SEEDS]
    return [video_run(s, f) for f in FOLDS for s in SEEDS], [eeg_run(s, f) for f in FOLDS for s in SEEDS]


def ready_shards(pilot_first=True):
    """Shards written by decode (meta json with n > 0, frames present) that have no predictions yet."""
    out = []
    root = os.path.dirname(out_path("shards", "x"))
    for a_ in ANIMALS:
        dd = os.path.join(root, a_)
        if not os.path.isdir(dd):
            continue
        for f in sorted(os.listdir(dd)):
            if not f.endswith(".json"):
                continue
            stem = f[:-5]
            if os.path.exists(out_path("preds", a_, f"{stem}.npz")):
                continue
            if not os.path.exists(os.path.join(dd, stem + ".u8")):
                continue
            if json.load(open(os.path.join(dd, f))).get("mode") != decided_mode():
                continue                                      # decoded in the other mode (speculative)
            out.append((a_, stem))
    return out


def decided_mode():
    dec = json.load(open(out_path("stepA", "decision.json")))
    if dec.get("mode") not in ("K", "E"):
        raise SystemExit(f"step A decision: {dec.get('decision')}")
    return dec["mode"]


def infer_shard(a_, stem, V, E, bs, loaded=None, frame_root=None):
    import numpy as np
    meta = json.load(open(out_path("shards", a_, stem + ".json")))
    base = out_path("shards", a_, stem) if frame_root is None else os.path.join(frame_root, a_, stem)
    n = meta["n"]
    vr, er = infer_runs(a_)
    t0 = time.time()
    if loaded is None:
        fr = np.fromfile(base + ".u8", dtype=np.uint8).reshape(n, T, S, S, 3)
        eg = np.load(base + ".eeg.npy")
    else:
        fr, eg = loaded
    if fr.shape[0] != n or eg.shape != (n, NWIN, WIN):
        raise SystemExit(f"{a_}/{stem}: shard shape {fr.shape} / {eg.shape} != n {n}")
    t_load = time.time() - t0
    out = {f"{r}|{t}": np.zeros((n, 3 if t == "g3" else 5), np.float32) for r in vr for t in TASKS}
    for b in range(0, n, bs):
        if STOP["flag"]:
            return None
        xx = V.normed(fr[b:b + bs])
        for r in vr:
            p = V.predict(xx, r)
            for t in TASKS:
                out[f"{r}|{t}"][b:b + bs] = p[t]
    flat = eg.reshape(n * NWIN, WIN)
    offs = np.arange(0, (n + 1) * NWIN, NWIN, dtype=np.int64)
    for r in er:
        Pw = E.predict_windows(flat, r)
        out[f"{r}|win"] = Pw[:, 1].reshape(n, NWIN).astype(np.float32)
        out[f"{r}|pe"] = E.clip_scores(Pw, offs)
    for k, v in out.items():
        if len(v) != n or not np.all(np.isfinite(v)):
            raise SystemExit(f"{a_}/{stem}: {k} has {len(v)} rows / non-finite values")
    secs = time.time() - t0
    atomic_npz(out_path("preds", a_, f"{stem}.npz"), minute=np.array(meta["minutes"], np.int64),
               row=np.array(meta["rows"], np.int64), **out, video_runs=np.array(vr), eeg_runs=np.array(er))
    # the shard is deleted only after the predictions of all its networks are written and their rows checked
    z = np.load(out_path("preds", a_, f"{stem}.npz"))
    if not all(len(z[k]) == n for k in out):
        raise SystemExit(f"{a_}/{stem}: written predictions do not have {n} rows")
    os.remove(base + ".u8")
    os.remove(base + ".eeg.npy")
    return dict(animal=a_, stem=stem, n=n, video_forwards=n * len(vr), eeg_windows=n * NWIN * len(er),
                secs=round(secs, 2), secs_load=round(t_load, 2))


def cmd_infer(a):
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor as TPE
    t_start = a.t_launch or time.time()
    import torch
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGUSR1, _on_signal)
    dev, gname = gpu_device()
    t_ready = time.time()
    used_before = gpu_used_hours()
    gpu_ledger("start", stage="infer", gpu=gname, import_s=round(t_ready - t_start, 1), pilot=a.pilot,
               used_before_h=round(used_before, 4))
    run_json("infer", dict(sub="infer", gpu=gname, pilot=a.pilot, used_before_h=used_before))
    # never run past the 6.0 GPU-hour cap, whatever the time limit (a requeued attempt re-reads the ledger)
    budget_end = t_start + (GPU_CAP_H - used_before) * 3600 - 240
    log(f"[infer] GPU hours used before this attempt {used_before:.3f}; budget ends in "
        f"{(budget_end - time.time()) / 3600:.2f} h")
    if budget_end - time.time() < 600:
        gpu_ledger("end", stage="infer", gpu=gname, shards=0, note="budget exhausted at start")
        raise SystemExit("GPU budget exhausted: not starting")
    need = sorted({a_ for a_ in ANIMALS})
    vr = sorted({r for a_ in need for r in infer_runs(a_)[0]})
    er = sorted({r for a_ in need for r in infer_runs(a_)[1]})
    V = VideoNets(vr, dev)
    E = EEGNets(er, dev)
    t_models = time.time()
    log(f"[infer] {gname}: import+prewarm {t_ready - t_start:.0f} s, models {t_models - t_ready:.0f} s")
    deadline = min(t_start + (a.deadline_s or 1e12), budget_end)
    done, idle_since = 0, None
    pool = TPE(1)
    pending = None
    pil, _ = decode_units(pilot=True)
    pstems = {stem_of(u["name"]) for u in pil}
    proj_p = out_path("budget", "pilot_projection.json")

    def load(item):
        a_, stem = item
        base = out_path("shards", a_, stem)
        meta = json.load(open(base + ".json"))
        n = meta["n"]
        return item, (np.fromfile(base + ".u8", dtype=np.uint8).reshape(n, T, S, S, 3), np.load(base + ".eeg.npy"))

    while not STOP["flag"]:
        if time.time() > deadline:
            log("[infer] deadline reached: not starting another shard")
            break
        todo = ready_shards()
        pilot_left = [x for x in todo if x[1] in pstems]
        if not os.path.exists(proj_p):
            if pilot_left:
                todo = pilot_left
            elif pilot_complete(pil):
                pj = pilot_projection(t_start, t_models, used_before)
                log(f"[infer] PILOT: {json.dumps(pj)}")
                if pj["stop"]:
                    log("[infer] STOP: the pilot projects more than the GPU budget (§10); no further shards")
                    break
                if a.pilot:
                    break
            else:
                todo = []                                   # wait for the pilot shards first
        elif a.pilot:
            break
        if pending is None and todo:
            pending = pool.submit(load, todo[0])
        if pending is None:
            if idle_since is None:
                idle_since = time.time()
            if time.time() - idle_since > a.wait_s:
                log(f"[infer] no shard ready for {a.wait_s:.0f} s: exiting")
                break
            time.sleep(15)
            continue
        idle_since = None
        item, data = pending.result()
        rest = [x for x in todo if x != item]
        pending = pool.submit(load, rest[0]) if rest else None
        r = infer_shard(item[0], item[1], V, E, a.bs, loaded=data)
        if r is None:
            break
        r.update(gpu=gname, job=os.environ.get("SLURM_JOB_ID"), at=time.strftime("%F %T"),
                 pilot=item[1] in pstems)
        with open(out_path("budget", "throughput.jsonl"), "a") as f:
            f.write(json.dumps(r) + "\n")
        gpu_ledger("beat", stage="infer", shard=f"{item[0]}/{item[1]}")
        done += 1
        log(f"[infer] {item[0]}/{item[1]}: {r['n']} windows, {r['video_forwards']} X3D forwards in {r['secs']} s "
            f"({r['video_forwards'] / max(r['secs'], 1e-6):.0f}/s)")
    gpu_ledger("end", stage="infer", gpu=gname, shards=done, wall_s=round(time.time() - t_start, 1))
    if STOP["flag"]:
        sys.exit(3)


def pilot_complete(pil):
    """Every pilot unit decoded and every one of its shards (n > 0) has predictions."""
    for u in pil:
        stem = stem_of(u["name"])
        if not os.path.exists(out_path("decode", f"{stem}.json")):
            return False
        for a_ in (CAMS[u["camera"]]["left"], CAMS[u["camera"]]["right"]):
            m = json.load(open(out_path("shards", a_, stem + ".json")))
            if m["n"] and not os.path.exists(out_path("preds", a_, f"{stem}.npz")):
                return False
    return True


def pilot_projection(t_start, t_models, used_before):
    """S3: integrity of the pilot predictions and the projected GPU hours of the whole scan (§10)."""
    import numpy as np
    pil, pday = decode_units(pilot=True)
    pst = {stem_of(u["name"]) for u in pil}
    rr = [json.loads(l) for l in open(out_path("budget", "throughput.jsonl"))]
    rp = [r for r in rr if r["stem"] in pst]
    integ = []
    for r in rp:
        z = np.load(out_path("preds", r["animal"], r["stem"] + ".npz"))
        m = json.load(open(out_path("shards", r["animal"], r["stem"] + ".json")))
        ok = len(z["minute"]) == m["n"] and all(np.all(np.isfinite(z[k])) for k in z.files if "|" in k)
        ok &= all(np.allclose(z[k].sum(1), 1, atol=1e-3) for k in z.files if k.endswith("|g3") or k.endswith("|g5"))
        ok &= all(((z[k] >= 0) & (z[k] <= 1)).all() for k in z.files if k.endswith("|pe") or k.endswith("|win"))
        integ.append(dict(shard=f"{r['animal']}/{r['stem']}", n=m["n"], ok=bool(ok)))
    fw = sum(r["video_forwards"] for r in rp)
    sec = sum(r["secs"] for r in rp)
    rate = fw / sec if sec else float("nan")
    units, _ = decode_units()
    total_fw = sum(u["n"][a_] * (3 if a_ in PARTNERS else 15) for u in units for a_ in u["n"])
    done_fw = sum(r["video_forwards"] for r in rr)
    overhead_h = (t_models - t_start) / 3600                  # import + prewarm + model load of one job
    remaining_h = (total_fw - done_fw) / rate / 3600
    projected_total_h = used_before + (time.time() - t_start) / 3600 + remaining_h + overhead_h
    pj = dict(pilot_day=pday, pilot_shards=len(rp), integrity=integ, integrity_ok=all(x["ok"] for x in integ),
              pilot_forwards=fw, pilot_secs=round(sec, 1), forwards_per_s_incl_eeg_and_io=round(rate, 1),
              total_forwards=total_fw, forwards_done=done_fw, remaining_h=round(remaining_h, 3),
              job_overhead_h=round(overhead_h, 3), used_before_this_job_h=round(used_before, 3),
              projected_total_h=round(projected_total_h, 3), cap_h=GPU_CAP_H,
              stop=bool(projected_total_h > GPU_CAP_H or not all(x["ok"] for x in integ)),
              at=time.strftime("%F %T"))
    atomic_json(out_path("budget", "pilot_projection.json"), pj)
    return pj


command("infer")(cmd_infer)


# ============================================================================ S5: analysis (§6)

GRADES = {"g3": ["mild (S2-3)", "severe (S4-5)"], "g5": ["S2", "S3", "S4", "S5"]}
STAGE_OF = {"Stage 2": 1, "Stage 3": 2, "Stage 4": 3, "Stage 5": 4}
CENTRAL = None


def load_events(animal):
    rows = [r for r in read_csv(os.path.join(AUDIT, "events.csv")) if r["animal"] == animal]
    ev = []
    for r in rows:
        s, e = AC.parse_dt(r["start"]), AC.parse_dt(r["end"])
        if s is None:
            continue
        if e is None:
            e = s + dt.timedelta(seconds=fnum(r["duration_s"]) or 0)
        ev.append(dict(label=r["label"], start=naive_to_utc_s(s), end=naive_to_utc_s(e), start_local=r["start"],
                       day_folder=r["day_folder"], clipped=bool(r["labelled_clip"]), video=r["video"],
                       duration_s=fnum(r["duration_s"])))
    ev.sort(key=lambda x: x["start"])
    return ev


def far_from_events(minutes, events, guard):
    """True where the window [m, m + 60) is >= guard s from every event interval."""
    import numpy as np
    ok = np.ones(len(minutes), bool)
    if not events:
        return ok
    s = np.array([e["start"] for e in events])
    e_ = np.array([e["end"] for e in events])
    o = np.argsort(s)
    s, e_ = s[o], e_[o]
    lo = np.searchsorted(s, minutes + 60 + guard)          # events starting before the window end + guard
    emax = np.maximum.accumulate(e_)
    for i, m in enumerate(minutes):
        j = lo[i]
        if j > 0 and emax[j - 1] > m - guard:
            ok[i] = False
    return ok


def overlap_events(minutes, events, pad):
    """True where the window overlaps any [start - pad, end + pad]."""
    import numpy as np
    return ~far_from_events(minutes, events, pad) if events else np.zeros(len(minutes), bool)


def episodes(minutes, flag):
    """Consecutive flagged windows merged; an episode starting <= 300 s after the previous one's end is merged."""
    import numpy as np
    m = np.asarray(minutes)[flag]
    if len(m) == 0:
        return []
    ep = [[int(m[0]), int(m[0]) + 60]]
    for x in m[1:]:
        if x <= ep[-1][1] + EPISODE_MERGE_S:
            ep[-1][1] = int(x) + 60
        else:
            ep.append([int(x), int(x) + 60])
    return ep


def window_status(animal, tables_json, clock):
    """Final status of every minute: the S0 reason, then EEG dropout / clock / decode error from S4, and whether
    predictions exist. Returns (tab, status codes, {minute: (stem, row in preds)})."""
    import numpy as np
    tab = load_table(animal)
    st = tab["reason"].astype(np.int16).copy()
    cam = CAM_OF[animal]
    files = tables_json["cameras"][cam]
    blocks = defaultdict(list)
    for b in clock["blocks"]:
        if b["animal"] == animal:
            blocks[b["fidx"]].append(b)
    pred_at = {}
    not_decoded = 0
    for fi, v in enumerate(files):
        idx = np.flatnonzero((tab["fidx"] == fi) & (tab["reason"] == 0))
        if not len(idx):
            continue
        stem = stem_of(v["name"])
        mp = out_path("shards", animal, stem + ".json")
        if not os.path.exists(mp):
            st[idx] = 99                                      # not decoded (reported)
            not_decoded += len(idx)
            continue
        meta = json.load(open(mp))
        for i in meta["dropout_rows"]:
            st[i] = RCODE["EEG dropout"]
        exc = [b for b in blocks.get(fi, []) if b.get("excluded")]
        for i in idx:
            if st[i] != 0:
                continue
            mid = tab["minute"][i] + 30
            if any(b["u0"] <= mid < b["u1"] for b in exc):
                st[i] = RCODE["clock"]
        for i in meta["decode_error_rows"]:
            if st[i] == 0:
                st[i] = RCODE["decode error"]
        pp = out_path("preds", animal, stem + ".npz")
        if os.path.exists(pp):
            for j, i in enumerate(meta["rows"]):
                pred_at[int(i)] = (stem, j)
        for i in meta["rows"]:
            if st[i] == 0 and int(i) not in pred_at:
                st[i] = 98                                    # decoded, not inferred (GPU budget)
    return tab, st, pred_at


def gather_preds(animal, rows, pred_at):
    """{key: array over rows} of every prediction field for these table rows."""
    import numpy as np
    by_stem = defaultdict(list)
    for k, i in enumerate(rows):
        stem, j = pred_at[int(i)]
        by_stem[stem].append((k, j))
    out = {}
    for stem, kj in by_stem.items():
        z = np.load(out_path("preds", animal, stem + ".npz"))
        ks = np.array([x[0] for x in kj])
        js = np.array([x[1] for x in kj])
        for key in z.files:
            if "|" not in key:
                continue
            a = z[key]
            if key not in out:
                out[key] = np.zeros((len(rows),) + a.shape[1:], a.dtype)
            out[key][ks] = a[js]
    return out


def configs(animal, system, matched_partner=None):
    """(label, video run, EEG run, fold, gate key) per configuration (§2): partners use their OOF fold (9 EGRG /
    3 video / 3 EEG); no-label animals all 5 folds (45 / 15 / 15); matched: the partner's fold and G_-P."""
    if matched_partner is not None:
        fl = [PARTNER_FOLD[matched_partner]]
        gate = f"minus_{matched_partner}"
    elif animal in PARTNERS:
        fl = [PARTNER_FOLD[animal]]
        gate = f"minus_{animal}"
    else:
        fl = list(FOLDS)
        gate = "all"
    out = []
    for f in fl:
        if system == "EGRG":
            out += [dict(label=f"v{v}e{e}f{f}", v=v, e=e, f=f, gate=gate) for v in SEEDS for e in SEEDS]
        elif system == "video":
            out += [dict(label=f"v{v}f{f}", v=v, e=None, f=f, gate=None) for v in SEEDS]
        else:
            out += [dict(label=f"e{e}f{f}", v=None, e=e, f=f, gate=None) for e in SEEDS]
    return out


def decide(P, cfg, system, task, gates):
    """(class, flag, q) of every window for one configuration."""
    import numpy as np
    if system == "EEG":
        pe = P[f"{eeg_run(cfg['e'], cfg['f'])}|pe"]
        fl = pe > 0.5
        return fl.astype(int), fl, pe
    Pv = P[f"{video_run(cfg['v'], cfg['f'])}|{task}"].astype(np.float64)
    if system == "video":
        c = Pv.argmax(1)
        return c, c > 0, 1.0 - Pv[:, 0]
    pe = P[f"{eeg_run(cfg['e'], cfg['f'])}|pe"]
    coef = gates["gates"][f"{task}|{video_run(cfg['v'], 'F')}|{eeg_run(cfg['e'], 'F')}"][cfg["gate"]]
    q = gate_q(coef, Pv, pe)
    c = egrg_decide(Pv, q)
    return c, c > 0, q


def rng_stats(vals):
    import numpy as np
    v = np.asarray(vals, float)
    return dict(mean=float(v.mean()), min=float(v.min()), max=float(v.max()), n=int(len(v)))


def rate_block(minutes, P, cfgs, system, task, gates, hours, with_days=False):
    """Flags per scanned hour (any grade and per grade), episodes per scanned day, per configuration."""
    import numpy as np
    per = []
    flags_all = []
    for c in cfgs:
        cls, fl, q = decide(P, c, system, task, gates)
        ep = episodes(minutes, fl)
        r = dict(config=c["label"], fold=c["f"], flags=int(fl.sum()), flags_per_h=fl.sum() / hours if hours else float("nan"),
                 episodes=len(ep), episodes_per_day=len(ep) / (hours / 24) if hours else float("nan"))
        if system != "EEG":
            ng = 3 if task == "g3" else 5
            for g in range(1, ng):
                r[f"per_h_{GRADES[task][g - 1]}"] = int((cls == g).sum()) / hours if hours else float("nan")
        per.append(r)
        flags_all.append(fl)
    return per, np.array(flags_all)


def summarise(per, keys):
    return {k: rng_stats([r[k] for r in per]) for k in keys}


def cmd_analyse(a):
    import numpy as np
    from zoneinfo import ZoneInfo
    central = ZoneInfo("America/Chicago")
    tj = load_tables_json()
    clock = json.load(open(out_path("clock", "clock.json")))
    gates = json.load(open(rd(out_path("s0", "gates.json"))))
    dec = json.load(open(out_path("stepA", "decision.json")))
    boxes = json.load(open(out_path("boxes", "boxes.json")))
    run_json("results", dict(sub="analyse"))
    R = dict(created=time.strftime("%F %T"), prereg=os.path.relpath(PREREG, REPO), prereg_sha256=sha256(PREREG),
             stepA_decision=dec["decision"], mode=dec["mode"], boxes=boxes["boxes"], animals={}, matched={},
             sensitivity={}, gate_2x2={}, cross_talk={}, oof_reference=gates["oof_reference_mean"],
             clock_summary=clock["summary"], coverage={})
    ev = {an: load_events(an) for an in ANIMALS}
    data = {}
    for an in ANIMALS:
        tab, st, pred_at = window_status(an, tj, clock)
        M = tab["minute"]
        scanned = np.flatnonzero(st == 0)
        P = gather_preds(an, scanned, pred_at)
        excl = {r: int((st == RCODE[r]).sum()) / 60 for r in REASONS}
        excl["not decoded"] = int((st == 99).sum()) / 60
        excl["decoded, not inferred (GPU budget)"] = int((st == 98).sum()) / 60
        data[an] = dict(tab=tab, st=st, P=P, rows=scanned, M=M[scanned])
        R["coverage"][an] = dict(first=epoch_iso(M[0]), last=epoch_iso(M[-1]) if len(M) else None,
                                 minutes_total=int(len(M)), hours_total=len(M) / 60, scanned_h=len(scanned) / 60,
                                 excluded_h_by_reason=excl, candidates_after_s0_h=int((tab["reason"] == 0).sum()) / 60)
    live = [an for an in ANIMALS if len(data[an]["M"])]
    R["animals_without_scanned_windows"] = [an for an in ANIMALS if an not in live]
    # ---- per animal, system, task
    for an in live:
        D = data[an]
        Mw, P = D["M"], D["P"]
        hours = len(Mw) / 60
        sf = far_from_events(Mw, ev[an], EVENT_GUARD_S)
        if an in PARTNERS:
            xl = D["tab"]["has_xlsx"][D["rows"]]
            sf = sf & xl
        subsets = {"all time": np.ones(len(Mw), bool)}
        if an in PARTNERS:
            subsets["seizure-free (>= 600 s from any event, day folder with xlsx)"] = sf
        elif ev[an]:
            subsets[">= 600 s from its own events"] = sf
        A = dict(scanned_h=hours, n_events=len(ev[an]), systems={})
        for system in ("EGRG", "video", "EEG"):
            for task in (TASKS if system != "EEG" else ("g5",)):
                cf = configs(an, system)
                key = f"{system}|{task}" if system != "EEG" else "EEG"
                blk = {}
                for sname, sel in subsets.items():
                    h = sel.sum() / 60
                    Psub = {k: v[sel] for k, v in P.items()}
                    per, flags = rate_block(Mw[sel], Psub, cf, system, task, gates, h)
                    ks = ["flags_per_h", "episodes_per_day"] + [k for k in per[0] if k.startswith("per_h_")]
                    b = dict(hours=h, configs=per, summary=summarise(per, ks))
                    if an in NOLABEL:
                        b["per_fold_mean_flags_per_h"] = {str(f): float(np.mean([r["flags_per_h"] for r in per if r["fold"] == f]))
                                                          for f in FOLDS}
                    if sname == "all time":
                        # day-to-day spread (UTC days with >= 1 h scanned) and hour-of-day profile (Central time)
                        days = np.array([dt.datetime.fromtimestamp(int(m), tz=UTC).strftime("%Y-%m-%d") for m in Mw[sel]])
                        mean_flag = flags.mean(0)
                        dr = {}
                        for dday in np.unique(days):
                            s_ = days == dday
                            if s_.sum() >= 60:
                                dr[str(dday)] = float(mean_flag[s_].sum() / (s_.sum() / 60))
                        vals = np.array(list(dr.values()))
                        b["day_to_day"] = dict(n_days=len(vals), median=float(np.median(vals)) if len(vals) else None,
                                               iqr=[float(np.percentile(vals, 25)), float(np.percentile(vals, 75))] if len(vals) else None,
                                               per_day=dr)
                        hod = np.array([dt.datetime.fromtimestamp(int(m) + 30, tz=UTC).astimezone(central).hour for m in Mw[sel]])
                        b["hour_of_day_flags_per_h"] = {int(hh): float(mean_flag[hod == hh].sum() / ((hod == hh).sum() / 60))
                                                        for hh in range(24) if (hod == hh).sum()}
                        if system == "EGRG":
                            qs = np.concatenate([decide(Psub, c, system, task, gates)[2] for c in cf])
                            b["q_quantiles"] = {str(p): float(np.quantile(qs, p)) for p in (0.5, 0.9, 0.99, 0.999, 0.9999)}
                            b["q_quantiles"]["max"] = float(qs.max())
                    blk[sname] = b
                A["systems"][key] = blk
        if an == "RN243":
            A["own_events"] = [dict(label=e["label"], start_local=e["start_local"], duration_s=e["duration_s"],
                                    windows=int(overlap_events(Mw, [e], 0).sum()),
                                    egrg_g5_flag_share=float(np.mean([decide({k: v[overlap_events(Mw, [e], 0)] for k, v in P.items()},
                                                                              c, "EGRG", "g5", gates)[1].any()
                                                                       for c in configs(an, "EGRG")])) if overlap_events(Mw, [e], 0).any() else None,
                                    status_of_windows=[REASONS[int(x) - 1] if 0 < x <= len(REASONS) else ("scanned" if x == 0 else str(int(x)))
                                                       for x in D["st"][(D["tab"]["minute"] < e["end"]) & (D["tab"]["minute"] + 60 > e["start"])]])
                                for e in ev[an]]
        R["animals"][an] = A
        log(f"[analyse] {an}: {hours:.1f} h scanned; EGRG g5 all-time "
            f"{A['systems']['EGRG|g5']['all time']['summary']['flags_per_h']['mean']:.3f}/h")
    # ---- primary comparison (matched), per camera
    for cam, d in CAMS.items():
        pa, nl = d["partner"], d["nolabel"]
        if pa not in live or nl not in live:
            continue
        Dp, Dn = data[pa], data[nl]
        common = np.intersect1d(Dp["M"], Dn["M"])
        far = far_from_events(common, ev[pa] + ev[nl], EVENT_GUARD_S)
        mm = common[far]
        ip = np.searchsorted(Dp["M"], mm)
        inn = np.searchsorted(Dn["M"], mm)
        Pp = {k: v[ip] for k, v in Dp["P"].items()}
        Pn = {k: v[inn] for k, v in Dn["P"].items()}
        h = len(mm) / 60
        blk = dict(matched_h=h, n_windows=int(len(mm)), common_scanned_h=len(common) / 60, systems={})
        for system in ("EGRG", "video", "EEG"):
            for task in (TASKS if system != "EEG" else ("g5",)):
                cf = configs(nl, system, matched_partner=pa)
                rows = []
                for c in cf:
                    fn_ = decide(Pn, c, system, task, gates)[1]
                    fp_ = decide(Pp, c, system, task, gates)[1]
                    rn, rp = fn_.sum() / h, fp_.sum() / h
                    rows.append(dict(config=c["label"], r_nolabel=rn, r_partner=rp,
                                     ratio=(rn / rp) if rp > 0 else None, diff=rn - rp,
                                     episodes_nolabel_per_day=len(episodes(mm, fn_)) / (h / 24),
                                     episodes_partner_per_day=len(episodes(mm, fp_)) / (h / 24)))
                key = f"{system}|{task}" if system != "EEG" else "EEG"
                ratios = [r["ratio"] for r in rows if r["ratio"] is not None]
                blk["systems"][key] = dict(configs=rows, r_nolabel=rng_stats([r["r_nolabel"] for r in rows]),
                                           r_partner=rng_stats([r["r_partner"] for r in rows]),
                                           ratio=rng_stats(ratios) if ratios else None,
                                           diff=rng_stats([r["diff"] for r in rows]),
                                           sign=dict(nolabel_higher=sum(r["diff"] > 0 for r in rows),
                                                     partner_higher=sum(r["diff"] < 0 for r in rows),
                                                     equal=sum(r["diff"] == 0 for r in rows)))
        R["matched"][cam] = blk
        # ---- cross-talk: the no-label animal's windows that overlap the partner's events +-60 s
        xt = overlap_events(Dn["M"], ev[pa], 60)
        cfm = configs(nl, "EGRG", matched_partner=pa)
        xrows = []
        for task in TASKS:
            for c in cfm:
                fl = decide(Dn["P"], c, "EGRG", task, gates)[1]
                r_x = fl[xt].sum() / (xt.sum() / 60) if xt.sum() else float("nan")
                r_m = blk["systems"][f"EGRG|{task}"]["configs"][[x["config"] for x in blk["systems"][f"EGRG|{task}"]["configs"]].index(c["label"])]["r_nolabel"]
                xrows.append(dict(task=task, config=c["label"], r_near_partner_events=r_x, r_matched=r_m))
        vx = []
        for c in configs(nl, "video", matched_partner=pa):
            fl = decide(Dn["P"], c, "video", "g5", gates)[1]
            vx.append(fl[xt].sum() / (xt.sum() / 60) if xt.sum() else float("nan"))
        R["cross_talk"][cam] = dict(nolabel=nl, partner=pa, windows_near_partner_events=int(xt.sum()),
                                    hours=xt.sum() / 60, rows=xrows,
                                    egrg_g5=dict(near=rng_stats([r["r_near_partner_events"] for r in xrows if r["task"] == "g5"]),
                                                 matched=rng_stats([r["r_matched"] for r in xrows if r["task"] == "g5"])),
                                    egrg_g3=dict(near=rng_stats([r["r_near_partner_events"] for r in xrows if r["task"] == "g3"]),
                                                 matched=rng_stats([r["r_matched"] for r in xrows if r["task"] == "g3"])),
                                    video_g5_near=rng_stats(vx) if vx else None)
    # ---- sensitivity (partners, same operating point)
    for pa in [x for x in PARTNERS if x in live]:
        D = data[pa]
        tab, st = D["tab"], D["st"]
        Mall = tab["minute"]
        pos = {int(m): k for k, m in enumerate(D["M"])}
        sf = far_from_events(D["M"], ev[pa], EVENT_GUARD_S) & tab["has_xlsx"][D["rows"]]
        evs = [e for e in ev[pa] if e["label"] in STAGE_OF]
        scor, notscor = [], Counter()
        for e in evs:
            ws = np.flatnonzero((Mall < e["end"]) & (Mall + 60 > e["start"]))
            if len(ws) == 0:
                notscor["no window (outside the recording)"] += 1
                continue
            bad = [int(st[i]) for i in ws if st[i] != 0]
            if bad:
                b = min(bad)
                notscor[REASONS[b - 1] if b <= len(REASONS) else ("not decoded" if b == 99 else "not inferred")] += 1
                continue
            scor.append((e, np.array([pos[int(Mall[i])] for i in ws])))
        S_ = dict(n_events_stage2_5=len(evs), scorable=len(scor), not_scorable_by_reason=dict(notscor), systems={})
        for system in ("EGRG", "video", "EEG"):
            for task in (TASKS if system != "EEG" else ("g5",)):
                cf = configs(pa, system)
                key = f"{system}|{task}" if system != "EEG" else "EEG"
                by = defaultdict(lambda: dict(n=0, detected=[], chance=[]))
                ncol = 2 if system == "EEG" else (5 if task == "g5" else 3)
                hi_tab = np.zeros((4, ncol))                   # stage x highest flagged grade (0 = none)
                for c in cf:
                    cls, fl, q = decide(D["P"], c, system, task, gates)
                    f_sf = fl[sf].mean() if sf.any() else float("nan")
                    for (e, ix) in scor:
                        stg = STAGE_OF[e["label"]]
                        for grp in (f"S{stg + 1}", "all", "clipped" if e["clipped"] else "never clipped",
                                    f"S{stg + 1} {'clipped' if e['clipped'] else 'never clipped'}"):
                            by[(grp, c["label"])]["detected"].append(bool(fl[ix].any()))
                            by[(grp, c["label"])]["chance"].append(1 - (1 - f_sf) ** len(ix))
                        hg = int(cls[ix].max()) if system != "EEG" else int(fl[ix].any())
                        hi_tab[stg - 1, min(hg, hi_tab.shape[1] - 1)] += 1.0 / len(cf)
                groups = sorted({g for (g, _) in by})
                out = {}
                for g in groups:
                    det = [np.sum(by[(g, c['label'])]["detected"]) for c in cf]
                    ch = [np.sum(by[(g, c['label'])]["chance"]) for c in cf]
                    n = len(by[(g, cf[0]['label'])]["detected"])
                    out[g] = dict(n=n, detected=rng_stats(det), chance_expected=rng_stats(ch),
                                  sensitivity=rng_stats([x / n for x in det]) if n else None)
                S_["systems"][key] = dict(by_group=out, stage_x_highest_grade_mean=hi_tab.round(3).tolist(),
                                          highest_grade_columns=(["none"] + GRADES[task]) if system != "EEG" else ["no", "yes"])
        R["sensitivity"][pa] = S_
    # ---- what the gate does: video-alone flag x EEG-alone flag, EGRG flag share per cell (per configuration)
    for an in live:
        D = data[an]
        rows = []
        for task in TASKS:
            for c in configs(an, "EGRG"):
                ge = decide(D["P"], c, "EGRG", task, gates)[1]
                vf = decide(D["P"], c, "video", task, gates)[1]
                ef = decide(D["P"], c, "EEG", task, gates)[1]
                cell = {}
                for vv in (0, 1):
                    for ee in (0, 1):
                        s_ = (vf == vv) & (ef == ee)
                        cell[f"video{vv}_eeg{ee}"] = dict(n=int(s_.sum()), egrg_share=float(ge[s_].mean()) if s_.any() else None)
                rows.append(dict(task=task, config=c["label"], cells=cell))
        agg = {}
        for task in TASKS:
            rr = [r for r in rows if r["task"] == task]
            agg[task] = {k: dict(n_mean=float(np.mean([r["cells"][k]["n"] for r in rr])),
                                 egrg_share_mean=float(np.nanmean([r["cells"][k]["egrg_share"] if r["cells"][k]["egrg_share"] is not None else np.nan for r in rr])))
                         for k in rr[0]["cells"]}
        R["gate_2x2"][an] = dict(summary=agg, configs=rows)
    # ---- per-window predictions regrouped per animal and UTC day (§6)
    for an in live:
        D = data[an]
        days = np.array([dt.datetime.fromtimestamp(int(m), tz=UTC).strftime("%Y-%m-%d") for m in D["M"]])
        for dday in np.unique(days):
            s_ = days == dday
            atomic_npz(out_path("preds", an, f"day_{dday}.npz"), minute=D["M"][s_],
                       **{k: v[s_] for k, v in D["P"].items()})
    R["deviations"] = deviations_list()
    atomic_json(out_path("results", "summary.json"), R)
    atomic_json(out_path("scan_results.json"), R)
    txt = render_results(R)
    for p in (out_path("results", "summary.txt"), out_path("scan_results.txt")):
        with open(p, "w") as f:
            f.write(txt)
    print(txt)


command("analyse")(cmd_analyse)


DEVIATIONS_FILE = "deviations.json"


def deviations_list():
    """The deviations recorded during the work (ttg_scan/deviations.json, appended by the operator)."""
    p = out_path(DEVIATIONS_FILE)
    return json.load(open(p)) if os.path.exists(p) else []


def _mr(s, nd=3):
    if s is None:
        return "n/a"
    if s["n"] == 1:
        return f"{s['mean']:.{nd}f}"
    return f"{s['mean']:.{nd}f} [{s['min']:.{nd}f}, {s['max']:.{nd}f}]"


def render_results(R):
    L = []
    w = L.append
    w("EEG-GATED RACINE GRADER ON THE CONTINUOUS RECORDINGS: FLAG RATES AT THE FIXED OPERATING POINT")
    w(f"created {R['created']}; pre-registration {R['prereg']} sha256 {R['prereg_sha256'][:16]}")
    w(f"step A: {R['stepA_decision']}")
    w("Numbers are flag rates at the grader's argmax operating point, not seizure counts. Every summary is the mean")
    w("[min, max] over model configurations: partners 9 EGRG / 3 video / 3 EEG (their OOF fold, gate G_-P);")
    w("no-label animals 45 / 15 / 15 (all folds, gate G_all). Whether RN201 / RN203 / RN243 are controls is unknown,")
    w("so no rate here is called a false-alarm rate.")
    w("")
    w("COVERAGE (hours)")
    for an, c in R["coverage"].items():
        ex = ", ".join(f"{k} {v:.1f}" for k, v in c["excluded_h_by_reason"].items() if v)
        w(f"  {an}: span {c['first']} .. {c['last']} = {c['hours_total']:.1f} h; scanned {c['scanned_h']:.1f} h; excluded: {ex}")
    w("")
    w("FLAG RATES (flags per scanned hour; episodes per scanned day)")
    ref = R["oof_reference"]
    w(f"  OOF reference (flags/h if every window behaved like an OOF non-seizure clip): g5 EGRG "
      f"{ref['g5']['egrg_flags_per_hour_if_like_oof_nonseizure']:.2f}, video {ref['g5']['video_flags_per_hour_if_like_oof_nonseizure']:.2f}, "
      f"EEG {ref['g5']['eeg_flags_per_hour_if_like_oof_nonseizure']:.2f}; g3 EGRG {ref['g3']['egrg_flags_per_hour_if_like_oof_nonseizure']:.2f}, "
      f"video {ref['g3']['video_flags_per_hour_if_like_oof_nonseizure']:.2f}")
    for an, A in R["animals"].items():
        w(f"  {an} ({'partner' if an in PARTNERS else 'no labelled seizures'}; {A['scanned_h']:.1f} h scanned; "
          f"{A['n_events']} annotated events)")
        for key, blk in A["systems"].items():
            for sname, b in blk.items():
                sm = b["summary"]
                grades = "  ".join(f"{k[6:]} {_mr(v)}" for k, v in sm.items() if k.startswith("per_h_"))
                w(f"    {key:<9} {sname[:24]:<24} {b['hours']:8.1f} h  flags/h {_mr(sm['flags_per_h'])}  "
                  f"episodes/day {_mr(sm['episodes_per_day'], 2)}")
                if grades:
                    w(f"              by grade (/h): {grades}")
                if "per_fold_mean_flags_per_h" in b:
                    w("              per-fold mean flags/h: " + ", ".join(f"f{f} {v:.3f}" for f, v in b["per_fold_mean_flags_per_h"].items()))
                if "day_to_day" in b and b["day_to_day"]["n_days"]:
                    dd = b["day_to_day"]
                    w(f"              day-to-day (config-mean, {dd['n_days']} days): median {dd['median']:.3f}/h, IQR "
                      f"[{dd['iqr'][0]:.3f}, {dd['iqr'][1]:.3f}]")
                if "q_quantiles" in b:
                    w("              q quantiles: " + ", ".join(f"{k} {v:.4f}" for k, v in b["q_quantiles"].items()))
                if "hour_of_day_flags_per_h" in b and key in ("EGRG|g5", "video|g5", "EEG"):
                    w("              by Central hour of day (/h): " + " ".join(
                        f"{h}:{v:.2f}" for h, v in sorted(b["hour_of_day_flags_per_h"].items(), key=lambda x: int(x[0]))))
    w("")
    w("PRIMARY COMPARISON (matched): both animals of a camera scored with the partner's fold networks and G_-P, on")
    w("windows scanned for both and >= 600 s from any event of either")
    for cam, m in R["matched"].items():
        w(f"  {cam}: {m['matched_h']:.1f} h matched (of {m['common_scanned_h']:.1f} h scanned for both)")
        for key, s in m["systems"].items():
            w(f"    {key:<9} no-label {_mr(s['r_nolabel'])}/h  partner {_mr(s['r_partner'])}/h  ratio "
              f"{_mr(s['ratio'], 2) if s['ratio'] else 'n/a'}  diff {_mr(s['diff'])}  sign (no-label higher / partner higher / "
              f"equal) {s['sign']['nolabel_higher']}/{s['sign']['partner_higher']}/{s['sign']['equal']}")
    w("")
    w("SENSITIVITY (partners' annotated Stage 2-5 events, same operating point; chance = sum 1 - (1 - f)^k)")
    for pa, S_ in R["sensitivity"].items():
        w(f"  {pa}: {S_['n_events_stage2_5']} events, {S_['scorable']} scorable; not scorable: {S_['not_scorable_by_reason']}")
        for key, s in S_["systems"].items():
            for g in ("all", "S2", "S3", "S4", "S5", "clipped", "never clipped"):
                if g in s["by_group"]:
                    b = s["by_group"][g]
                    w(f"    {key:<9} {g:<14} n {b['n']:5d}  detected {_mr(b['detected'], 1)}  sensitivity "
                      f"{_mr(b['sensitivity'])}  chance-expected {_mr(b['chance_expected'], 1)}")
            w(f"    {key:<9} stage x highest flagged grade (config mean), columns {s['highest_grade_columns']}:")
            for i, row in enumerate(s["stage_x_highest_grade_mean"]):
                w(f"              S{i + 2}: " + " ".join(f"{x:8.2f}" for x in row))
    w("")
    w("WHAT THE GATE DOES (mean over configurations: windows per cell, EGRG flag share in the cell)")
    for an, g in R["gate_2x2"].items():
        for task, cells in g["summary"].items():
            w(f"  {an} {task}: " + "  ".join(f"{k}: n {v['n_mean']:.0f}, EGRG {v['egrg_share_mean']:.3f}"
                                             if v['egrg_share_mean'] == v['egrg_share_mean'] else f"{k}: n {v['n_mean']:.0f}"
                                             for k, v in cells.items()))
    w("")
    w("CROSS-TALK: no-label animal's EGRG flag rate in windows overlapping the partner's events +-60 s vs matched time")
    for cam, x in R["cross_talk"].items():
        w(f"  {cam} ({x['nolabel']} near {x['partner']}'s events): {x['hours']:.1f} h; g5 near {_mr(x['egrg_g5']['near'])}/h vs "
          f"matched {_mr(x['egrg_g5']['matched'])}/h; g3 near {_mr(x['egrg_g3']['near'])}/h vs matched {_mr(x['egrg_g3']['matched'])}/h; "
          f"video g5 near {_mr(x['video_g5_near'])}/h")
    if "RN243" in R["animals"] and "own_events" in R["animals"]["RN243"]:
        w("")
        w("RN243's own annotated events")
        for e in R["animals"]["RN243"]["own_events"]:
            w(f"  {e['label']:<8} {e['start_local']} ({e['duration_s']} s): {e['windows']} scanned windows; share of the 45 "
              f"EGRG g5 configurations flagging any of them {e['egrg_g5_flag_share']}; window status {e['status_of_windows']}")
    w("")
    w("CLOCK CHECK (1-h blocks per camera file and box)")
    for an, c in R["clock_summary"].items():
        w(f"  {an}: blocks {c['blocks']}, measured {c['measured']}, strong {c['strong']}, strong and consistent "
          f"{c['strong_consistent']}, excluded {c['excluded']}")
    w("")
    w("DEVIATIONS")
    for i, d in enumerate(R.get("deviations", []), 1):
        w(f"  {i}. {d if isinstance(d, str) else d.get('text')}")
    return "\n".join(L) + "\n"


# ============================================================================ S6: gallery (§9)

def consensus(animal, D, gates, sel=None):
    """Per window: share of EGRG g5 configurations that flag it, q-bar (config-mean q), video's modal g5 grade."""
    import numpy as np
    P = D["P"] if sel is None else {k: v[sel] for k, v in D["P"].items()}
    cf = configs(animal, "EGRG")
    fl, qs = [], []
    for c in cf:
        _, f_, q = decide(P, c, "EGRG", "g5", gates)
        fl.append(f_)
        qs.append(q)
    fl, qs = np.array(fl), np.array(qs)
    g3f = np.array([decide(P, c, "EGRG", "g3", gates)[1] for c in cf])
    vr = sorted({video_run(c["v"], c["f"]) for c in cf})
    grades = np.stack([P[f"{r}|g5"].argmax(1) for r in vr])
    modal = np.array([np.bincount(grades[:, i], minlength=5).argmax() for i in range(grades.shape[1])])
    return fl.mean(0), qs.mean(0), modal, (fl.any(0) | g3f.any(0))


def gallery_items(R_data, gates, ev, rng):
    import numpy as np
    items = []
    for an in ANIMALS:
        D = R_data[an]
        M = D["M"]
        share, qbar, modal, anyflag = consensus(an, D, gates)
        if an in PARTNERS:
            sf = far_from_events(M, ev[an], EVENT_GUARD_S) & D["tab"]["has_xlsx"][D["rows"]]
            cons = (share >= 0.5) & sf
            cat = "partner seizure-free, consensus flagged"
        else:
            cons = share >= 0.5
            cat = "no-label, consensus flagged"
        eps = episodes(M, cons)
        ranked = []
        for s0, e0 in eps:
            ix = np.flatnonzero((M >= s0) & (M < e0) & cons)
            k = ix[np.argmax(qbar[ix])]
            ranked.append((float(qbar[k]), s0, e0, int(M[k])))
        ranked.sort(key=lambda x: -x[0])
        for rank, (qm, s0, e0, mk) in enumerate(ranked[:10], 1):
            items.append(dict(animal=an, category=cat, rank=rank, qbar_max=qm, ep_start=s0, ep_end=e0, peak_minute=mk,
                              n_episodes_total=len(ranked)))
        if an in NOLABEL:
            ctrl = np.flatnonzero(~anyflag)
            for k in sorted(rng.choice(ctrl, size=min(5, len(ctrl)), replace=False).tolist()):
                items.append(dict(animal=an, category="blinded control: flagged by no EGRG configuration", rank=None,
                                  qbar_max=float(qbar[k]), ep_start=int(M[k]), ep_end=int(M[k]) + 60, peak_minute=int(M[k]),
                                  n_episodes_total=None))
    return items


def cmd_gallery(a):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import resample_poly
    tj = load_tables_json()
    clock = json.load(open(out_path("clock", "clock.json")))
    gates = json.load(open(rd(out_path("s0", "gates.json"))))
    boxes, _ = load_boxes()
    run_json("gallery", dict(sub="gallery"))
    ev = {an: load_events(an) for an in ANIMALS}
    data = {}
    for an in ANIMALS:
        tab, st, pred_at = window_status(an, tj, clock)
        rows = np.flatnonzero(st == 0)
        data[an] = dict(tab=tab, rows=rows, M=tab["minute"][rows], P=gather_preds(an, rows, pred_at))
    rng = np.random.default_rng(20260930)
    items = gallery_items(data, gates, ev, rng)
    ids = rng.permutation(len(items))
    key = {}
    edfs = edf_cache()
    for it, rid in zip(items, ids):
        iid = f"item_{int(rid):03d}"
        an = it["animal"]
        cam = CAM_OF[an]
        D = data[an]
        s0, e0 = it["ep_start"] - 60, it["ep_end"] + 60
        if e0 - s0 > 360:
            c = it["peak_minute"] + 30
            s0, e0 = max(s0, c - 180), min(e0, c + 180)
        # the camera file that holds the peak window
        i_pk = int(np.flatnonzero(D["tab"]["minute"] == it["peak_minute"])[0])
        fi = int(D["tab"]["fidx"][i_pk])
        v = tj["cameras"][cam][fi]
        pts, key_, tb = load_packets(v["name"])
        t_first, t_last = v["t0"] + pts[0] * tb, v["t0"] + pts[-1] * tb
        s0, e0 = max(s0, t_first), min(e0, t_last)
        ss, du = s0 - v["t0"], e0 - s0
        d_item = out_path("gallery", iid, "x")[:-2]
        x, y, w, h = boxes[an]
        for tag, vf in (("crop", f"crop={w}:{h}:{x}:{y}"), ("full_400x300", "scale=400:300:flags=area")):
            subprocess.run([FFMPEG, "-nostdin", "-y", "-loglevel", "error", "-ss", f"{ss:.3f}", "-i", raw_path(v["relpath"]),
                            "-t", f"{du:.3f}", "-an", "-vf", vf, "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                            "-pix_fmt", "yuv420p", os.path.join(d_item, f"{tag}.mp4")], check=True, timeout=1800)
        # EEG panel: 1-kHz trace, 125-Hz trace, per-6-s P_E(sz) (mean over the animal's EEG networks), per-window
        # q-bar and video's modal g5 grade
        e = edfs[an][int(D["tab"]["edf"][i_pk])]
        hh = AC.edf_header(raw_path(e["relpath"]))
        naive0 = minute_to_naive(int(math.floor(s0)))
        st_ = dt.datetime.fromisoformat(e["start_naive"])
        k0 = int(round(((naive0 - st_).total_seconds() + (s0 - math.floor(s0))) * FS)) if naive0 else 0
        sig = edf_samples(hh, e["ch"], max(k0, 0), int(du * FS)) if naive0 else None
        sel = (D["M"] + 60 > s0) & (D["M"] < e0)
        share, qbar, modal, _ = consensus(an, D, gates, sel)
        Ms = D["M"][sel]
        er = sorted({eeg_run(c["e"], c["f"]) for c in configs(an, "EGRG")})
        win = np.mean([D["P"][f"{r}|win"][sel] for r in er], 0)
        fig, ax = plt.subplots(4, 1, figsize=(14, 9), sharex=True)
        if sig is not None:
            tt = np.arange(len(sig)) / FS
            ax[0].plot(tt, sig * 1e3, lw=0.3)
            ax[0].set_ylabel("1 kHz (mV)")
            s125 = resample_poly(sig, 1, DECIM)
            ax[1].plot(np.arange(len(s125)) / (FS / DECIM), s125 * 1e3, lw=0.4)
            ax[1].set_ylabel("125 Hz (mV)")
        for j, m in enumerate(Ms):
            for k in range(NWIN):
                t_a = m + 3 * k - s0
                ax[2].plot([t_a, t_a + 6], [win[j, k], win[j, k]], color="C3", lw=1)
            ax[3].plot([m - s0, m - s0 + 60], [qbar[j], qbar[j]], color="C2", lw=2)
            ax[3].text(m - s0 + 2, min(qbar[j] + 0.05, 1.0), f"grade {['non', 'S2', 'S3', 'S4', 'S5'][modal[j]]}", fontsize=7)
        ax[2].set_ylabel("P_E(sz) per 6 s")
        ax[2].set_ylim(-0.02, 1.02)
        ax[3].set_ylabel("q-bar per window")
        ax[3].set_ylim(-0.02, 1.1)
        ax[3].set_xlabel("seconds from item start")
        fig.suptitle(f"{iid}: {epoch_iso(s0)} UTC, {du:.0f} s")
        fig.tight_layout()
        fig.savefig(os.path.join(d_item, "eeg.png"), dpi=110)
        plt.close(fig)
        atomic_json(os.path.join(d_item, "item.json"), dict(
            id=iid, camera=cam, start_utc=epoch_iso(s0), duration_s=round(du, 2), raw_file=v["name"],
            seek_s=round(ss, 3), crop_box=[x, y, w, h], windows=[epoch_iso(m) for m in Ms],
            labels_to_fill=dict(verdict="seizure / not seizure / unsure", stage="2-5 if seizure", notes="")))
        key[iid] = dict(it, start_utc=epoch_iso(s0), duration_s=du, raw_file=v["name"])
        log(f"[gallery] {iid}: {an} {it['category']} rank {it['rank']} q-bar {it['qbar_max']:.3f}")
    atomic_json(out_path("gallery", "key.json"), dict(created=time.strftime("%F %T"), seed=20260930, items=key,
                                                        note="open only after every item has been labelled"))
    with open(out_path("gallery", "README.txt"), "w") as f:
        f.write("Blinded review items (scan_prereg.md section 9). Label each item_XXX as seizure (with Racine stage), "
                "not seizure, or unsure, from crop.mp4 (native-resolution crop), full_400x300.mp4 (whole camera frame, "
                "for cross-talk) and eeg.png, BEFORE opening key.json.\n")


command("gallery")(cmd_gallery)


def cmd_status(a):
    import numpy as np
    units, pday = decode_units()
    nd = sum(1 for u in units if os.path.exists(out_path("decode", f"{stem_of(u['name'])}.json")))
    npred = sum(len([f for f in os.listdir(os.path.dirname(out_path("preds", a_, "x")))
                     if f.endswith(".npz") and not f.startswith("day_")])
                if os.path.isdir(os.path.dirname(out_path("preds", a_, "x"))) else 0 for a_ in ANIMALS)
    print(f"decode units done {nd}/{len(units)} (pilot day {pday}); prediction files {npred}; shard bytes "
          f"{shard_bytes() / 1e9:.1f} GB")
    tp = out_path("budget", "throughput.jsonl")
    if os.path.exists(tp):
        rr = [json.loads(l) for l in open(tp)]
        fw = sum(r["video_forwards"] for r in rr)
        sec = sum(r["secs"] for r in rr)
        print(f"inference: {len(rr)} shards, {fw} X3D forwards in {sec:.0f} s of shard time ({fw / max(sec, 1):.0f}/s)")


command("status")(cmd_status)


# ============================================================================ fused GPU job: step A, S3 pilot, S4

_W = {}


def _worker_init(mode, threads, frame_root):
    """Decode worker (forked before CUDA is initialised): ignores SIGTERM / SIGUSR1 (the parent stops cleanly)."""
    for sg in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
        signal.signal(sg, signal.SIG_IGN)
    boxes, box_sha = load_boxes()
    _W.update(mode=mode, threads=threads, frame_root=frame_root, boxes=boxes, box_sha=box_sha, edfs=edf_cache(),
              tabs={a_: load_table(a_) for a_ in ANIMALS})


def _worker_decode(u):
    t0 = time.time()
    try:
        msg = decode_unit(u, _W["mode"], _W["boxes"], _W["box_sha"], _W["edfs"], _W["tabs"], _W["threads"],
                          frame_root=_W["frame_root"])
        return u, msg, None, time.time() - t0
    except BaseException:
        return u, None, traceback.format_exc(), time.time() - t0


def unit_done(u):
    """Every shard of the unit has its meta and, when it holds windows, its predictions."""
    stem = stem_of(u["name"])
    for a_ in (CAMS[u["camera"]]["left"], CAMS[u["camera"]]["right"]):
        mp = out_path("shards", a_, stem + ".json")
        if not os.path.exists(mp):
            return False
        m = json.load(open(mp))
        if m.get("mode") != "K" or (m["n"] and not os.path.exists(out_path("preds", a_, f"{stem}.npz"))):
            return False
    return True


def cmd_scan_gpu(a):
    """One GPU job: step A (if not decided yet), then the S3 pilot and the S4 scan with decoding on the job's own
    CPUs (keyframes of the raw files, which live on a fast file system) into node-local /dev/shm. Stops at the
    deadline, at the 6.0 GPU-hour cap, or when the pilot projects more than the cap."""
    import numpy as np
    import multiprocessing as mp
    t_start = a.t_launch or time.time()
    frame_root = os.path.join("/dev/shm", f"scan_{os.environ.get('SLURM_JOB_ID', os.getpid())}")
    os.makedirs(frame_root, exist_ok=True)
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGUSR1, _on_signal)
    import torch                                             # imported, CUDA not initialised before the fork
    dp = out_path("stepA", "decision.json")
    decided_before = os.path.exists(dp)
    mode0 = json.load(open(dp)).get("mode") if decided_before else "K"
    if decided_before and mode0 != "K":
        raise SystemExit(f"step A decided {mode0}: this job only scans in mode K")
    ctx = mp.get_context("fork")
    pool = ctx.Pool(a.workers, initializer=_worker_init, initargs=("K", a.threads, frame_root))
    dev, gname = gpu_device()
    t_ready = time.time()
    used_before = gpu_used_hours()
    gpu_ledger("start", stage="scan-gpu", gpu=gname, import_s=round(t_ready - t_start, 1),
               used_before_h=round(used_before, 4))
    run_json(f"infer/run_{os.environ.get('SLURM_JOB_ID', 'local')}.json",
             dict(sub="scan-gpu", gpu=gname, used_before_h=used_before, workers=a.workers, threads=a.threads,
                  stage=STAGE, frame_root=frame_root))
    budget_end = t_start + (GPU_CAP_H - used_before) * 3600 - 240
    deadline = min(t_start + (a.deadline_s or 1e12), budget_end)
    log(f"[scan] {gname}: start-up (staging + import) {t_ready - t_start:.0f} s; GPU hours used before "
        f"{used_before:.3f}; stop starting units in {(deadline - time.time()) / 3600:.2f} h")
    if budget_end - time.time() < 600:
        gpu_ledger("end", stage="scan-gpu", gpu=gname, note="budget exhausted at start")
        raise SystemExit("GPU budget exhausted")
    vr = [video_run(s, f) for f in FOLDS for s in SEEDS]
    er = [eeg_run(s, f) for f in FOLDS for s in SEEDS]
    V = VideoNets(vr, dev)
    E = EEGNets(er, dev)
    t_models = time.time()
    log(f"[scan] models loaded in {t_models - t_ready:.0f} s")
    # ---- step A (S2) with the same loaded networks, then its registered decision
    if not decided_before:
        units_all, _ = decode_units()
        pil0, _ = decode_units(pilot=True)
        # decoding starts at once (CPU only); no network sees a raw window before the decision is written
        pending = {}
        for u in pil0:                                        # the pilot units only
            if not unit_done(u):
                pending[u["name"]] = pool.apply_async(_worker_decode, (u,))
        cmd_stepa_gpu(a, nets=(V, E, dev, gname))
        cmd_stepa_eval(a)
        gpu_ledger("beat", stage="scan-gpu", note="step A done")
        mode = json.load(open(dp)).get("mode")
        if mode != "K":
            log(f"[scan] step A decision: {json.load(open(dp))['decision']}: no mode-K scan; stopping")
            pool.terminate()
            shutil.rmtree(frame_root, ignore_errors=True)
            gpu_ledger("end", stage="scan-gpu", gpu=gname, note=f"step A mode {mode}")
            return
    else:
        pending = {}
    # ---- S3 pilot, then S4
    units, _ = decode_units()
    pil, pday = decode_units(pilot=True)
    pstems = {stem_of(u["name"]) for u in pil}
    order = pil + [u for u in units if u not in pil]
    todo = [u for u in order if not unit_done(u) and u["name"] not in pending]
    proj_p = out_path("budget", "pilot_projection.json")
    max_inflight = a.workers + 3
    done_units, stop_reason = 0, None
    ready = []                                               # decoded units waiting for the GPU

    pil_names = {u["name"] for u in pil}

    def submit():
        """Keep the decode workers busy; before the pilot is projected only pilot units are started."""
        while len(pending) < max_inflight and time.time() < deadline and not STOP["flag"]:
            cand = [u for u in todo if os.path.exists(proj_p) or u["name"] in pil_names]
            if not cand:
                return
            u = cand[0]
            todo.remove(u)
            pending[u["name"]] = pool.apply_async(_worker_decode, (u,))

    t_idle = 0.0
    while not STOP["flag"]:
        submit()
        for name in list(pending):
            if pending[name].ready():
                u, msg, err, secs = pending.pop(name).get()
                if err:
                    log(f"[scan] decode FAILED {name}\n{err}")
                    stop_reason = f"decode failure {name}"
                    break
                ready.append((u, secs))
                log(f"[scan] decoded {msg}")
        if stop_reason:
            break
        if ready:
            u, dsecs = ready.pop(0)
            stem = stem_of(u["name"])
            for a_ in (CAMS[u["camera"]]["left"], CAMS[u["camera"]]["right"]):
                m = json.load(open(out_path("shards", a_, stem + ".json")))
                if not m["n"] or os.path.exists(out_path("preds", a_, f"{stem}.npz")):
                    continue
                r = infer_shard(a_, stem, V, E, a.bs, frame_root=frame_root)
                if r is None:
                    break
                r.update(gpu=gname, job=os.environ.get("SLURM_JOB_ID"), at=time.strftime("%F %T"),
                         pilot=stem in pstems, decode_secs=round(dsecs, 1), gpu_idle_s_so_far=round(t_idle, 1))
                with open(out_path("budget", "throughput.jsonl"), "a") as f:
                    f.write(json.dumps(r) + "\n")
                log(f"[scan] {a_}/{stem}: {r['n']} windows, {r['video_forwards']} X3D forwards in {r['secs']} s "
                    f"({r['video_forwards'] / max(r['secs'], 1e-6):.0f}/s); GPU idle so far {t_idle:.0f} s")
            gpu_ledger("beat", stage="scan-gpu", unit=u["name"])
            done_units += 1
        else:
            if not pending and not todo:
                stop_reason = "all units done"
                break
            if not pending and time.time() >= deadline:
                stop_reason = "deadline / GPU budget"
                break
            time.sleep(2)
            t_idle += 2
        if not os.path.exists(proj_p) and pilot_complete(pil):
            pj = pilot_projection(t_start, t_models, used_before)
            log(f"[scan] PILOT ({pday}): {json.dumps({k: v for k, v in pj.items() if k != 'integrity'})}")
            if pj["stop"]:
                stop_reason = "pilot projection above the cap (or integrity failure): STOP (§10)"
                break
    pool.terminate()
    shutil.rmtree(frame_root, ignore_errors=True)
    if STOP["flag"]:
        stop_reason = stop_reason or "signal"
    gpu_ledger("end", stage="scan-gpu", gpu=gname, units=done_units, reason=stop_reason, idle_s=round(t_idle, 1),
               wall_s=round(time.time() - t_start, 1))
    log(f"[scan] stopping: {stop_reason}; {done_units} units this job")
    if STOP["flag"]:
        sys.exit(3)


command("scan-gpu")(cmd_scan_gpu)


def cmd_motion(a):
    """CPU: keyframe motion per box for every camera file (the clock check's only video input), written to
    motion/<file>.npz exactly as `decode` writes it (the fused GPU job's decode rewrites the same values)."""
    import numpy as np
    import av
    import cv2
    cv2.setNumThreads(1)
    boxes, _ = load_boxes()
    units, _ = decode_units()
    if a.index is not None:
        units = units[a.index::a.stride]
    tj = load_tables_json()
    for u in units:
        v = tj["cameras"][u["camera"]][u["fidx"]]
        stem = stem_of(v["name"])
        dst = out_path("motion", f"{stem}.npz")
        if os.path.exists(dst):
            continue
        t0 = time.time()
        pts, key_, tb = load_packets(v["name"])
        kf = np.flatnonzero(key_)
        kpts = {int(pts[i]): int(i) for i in kf}
        animals = (CAMS[u["camera"]]["left"], CAMS[u["camera"]]["right"])
        mot_pts, mot, prev = [], [], None
        c = av.open(raw_path(v["relpath"]))
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        s.thread_count = a.threads
        cc = s.codec_context

        def handle(fr):
            nonlocal prev
            img = fr.to_ndarray(format="bgr24")
            gs = []
            for a_ in animals:
                x, y, w, h = boxes[a_]
                g = cv2.cvtColor(img[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
                gs.append(cv2.resize(g, (w // 2, h // 2), interpolation=cv2.INTER_AREA).astype(np.int16))
            mot.append([float(np.abs(g1 - g0).mean()) for g1, g0 in zip(gs, prev)] if prev is not None
                       else [np.nan] * len(animals))
            mot_pts.append(int(fr.pts))
            prev = gs

        for pk in c.demux(s):
            if pk.pts is None or int(pk.pts) not in kpts:
                continue
            try:
                for fr in cc.decode(pk):
                    if int(fr.pts) in kpts:
                        handle(fr)
            except Exception:
                pass
        for fr in cc.decode(None):
            if int(fr.pts) in kpts:
                handle(fr)
        c.close()
        atomic_npz(dst, kf_pts=np.array(mot_pts, np.int64), kf_time=v["t0"] + np.array(mot_pts, np.float64) * tb,
                   motion=np.array(mot, np.float32).T.reshape(len(animals), -1) if mot else np.zeros((len(animals), 0), np.float32),
                   animals=np.array(animals), boxes=np.array([boxes[a_] for a_ in animals]))
        log(f"[motion] {stem}: {len(mot_pts)}/{len(kf)} keyframes, {time.time() - t0:.0f} s")


command("motion")(cmd_motion)


# ============================================================================ the registered STOP outcome (§7)

def stepa_breakdown():
    """Information only: step-A disagreements by stratum, video-flag agreement, and the pixel differences of the
    K / E arms from the cached clip frames (C)."""
    import numpy as np
    it = json.load(open(out_path("stepA", "items.json")))
    items = it["items"]
    P = np.load(out_path("stepA", "preds.npz"))
    y5 = np.array([x["y5"] for x in items])
    stored = {}
    for f in FOLDS:
        for s in SEEDS:
            run = video_run(s, f)
            z = np.load(rd(os.path.join(PATCHED, run, f"val_ep{V_EPOCH:02d}.npz")), allow_pickle=True)
            pos = {str(p): i for i, p in enumerate(z["path"])}
            for t in TASKS:
                stored[(run, t)] = (pos, z[f"probs_{t}"])
    out = {}
    for t in TASKS:
        st = {s: np.stack([stored[(video_run(s, x["fold"]), t)][1][stored[(video_run(s, x["fold"]), t)][0][x["key"]]]
                           for x in items]) for s in SEEDS}
        for arm in "CKE":
            ag, fa, yy, mg = [], [], [], []
            for s in SEEDS:
                pa, ps = P[f"{arm}|{t}|s{s}"], st[s]
                ag.append(pa.argmax(1) == ps.argmax(1))
                fa.append((pa.argmax(1) > 0) == (ps.argmax(1) > 0))
                yy.append(y5)
                srt = np.sort(ps, 1)
                mg.append(srt[:, -1] - srt[:, -2])
            ag, fa, yy, mg = map(np.concatenate, (ag, fa, yy, mg))
            out[f"{arm}|{t}"] = dict(
                rows=int(len(ag)), argmax_agree=float(ag.mean()), video_flag_agree=float(fa.mean()),
                argmax_agree_by_stage={["non-seizure", "S2", "S3", "S4", "S5"][k]: [int(ag[yy == k].sum()), int((yy == k).sum())]
                                       for k in range(5)},
                disagreements_changing_the_video_flag=int((~ag & ~fa).sum()),
                disagreements_within_seizure_grades=int((~ag & fa).sum()),
                median_stored_top2_margin_of_disagreeing_rows=float(np.median(mg[~ag])) if (~ag).any() else None)
    pix = {}
    try:
        fc = np.load(rd(out_path("stepA", "frames_C.npy")), mmap_mode="r")
        for arm in "KE":
            fx = np.load(rd(out_path("stepA", f"frames_{arm}.npy")), mmap_mode="r")
            mae = np.array([np.abs(fx[i].astype(np.int16) - fc[i].astype(np.int16)).mean() for i in range(len(fc))])
            lvl = np.array([fx[i].astype(np.float32).mean() - fc[i].astype(np.float32).mean() for i in range(len(fc))])
            pix[arm] = dict(mae_vs_C_median=float(np.median(mae)), mae_vs_C_p95=float(np.percentile(mae, 95)),
                            mae_vs_C_max=float(mae.max()), mean_level_minus_C_median=float(np.median(lvl)))
    except FileNotFoundError:
        pix = None
    return out, pix


def cmd_report_stop(a):
    """scan_results.{json,txt} for the registered STOP of step A: everything that exists, and no flag rate."""
    import numpy as np
    dec = json.load(open(out_path("stepA", "decision.json")))
    sa = json.load(open(out_path("stepA", "stepA.json")))
    gr = json.load(open(out_path("stepA", "gpu_run.json")))
    it = json.load(open(out_path("stepA", "items.json")))
    gates = json.load(open(out_path("s0", "gates.json")))
    tj = load_tables_json()
    boxes = json.load(open(out_path("boxes", "boxes.json")))
    clock = json.load(open(out_path("clock", "clock.json")))
    brk, pix = stepa_breakdown()
    items = it["items"]
    cov = {}
    for an in ANIMALS:
        tab = load_table(an)
        r = tab["reason"]
        cov[an] = dict(span=[epoch_iso(tab["minute"][0]), epoch_iso(tab["minute"][-1] + 60)], hours=len(r) / 60,
                       candidate_hours_after_S0=float((r == 0).sum() / 60),
                       excluded_hours_S0={k: float((r == RCODE[k]).sum() / 60) for k in REASONS if (r == RCODE[k]).any()})
    ex_blocks = [b for b in clock["blocks"] if b.get("excluded")]
    shifted = sorted({b["file"] for b in ex_blocks})
    same_file = [b for b in clock["blocks"] if b["file"] in shifted and "lag" in b]
    ledger = [json.loads(l) for l in open(out_path("budget", "gpu.jsonl"))]
    R = dict(
        created=time.strftime("%F %T"), prereg=os.path.relpath(PREREG, REPO), prereg_sha256=sha256(PREREG),
        outcome="STOP (§7): step A arms K and E both fail the registered tolerance, so no raw window was scanned. "
                "Per the pre-registration the result is reported and an amendment is needed before any scan.",
        flag_rates=None,
        stepA=dict(decision=dec["decision"], mode=dec["mode"], tolerance=dict(
            video="per head: argmax agreement >= 0.97 of (clip, network) rows AND median |dp| <= 0.05",
            eeg="argmax agreement >= 0.99 AND median |dp| <= 0.01", noise_floor="arm C median |dp| <= 0.01"),
            arms={arm: {t: sa["arms"][arm][t] for t in TASKS} | {"passes": sa["arms"][arm]["passes"]} for arm in "CKE"},
            stored_macro_f1_by_seed=sa["stored_macro_f1_by_seed"], eeg_R=sa["eeg"], breakdown=brk,
            pixels_vs_C=pix, sample=dict(n=len(items), by_stratum=dict(Counter(["non-seizure", "S2", "S3", "S4", "S5"][x["y5"]] for x in items)),
                                         partners={p: sum(x["animal"] == p for x in items) for p in PARTNERS},
                                         animals=len({x["animal"] for x in items}), redrawn=it["redrawn"],
                                         topups=it["topups"], ineligible=it["ineligible"],
                                         r0_one_gop_earlier=sum(x["r0_is_before_seek_kf"] for x in items),
                                         ncc_min=min(x["ncc"] for x in items),
                                         max_keyframe_snap_frames=max(x["max_snap_frames"] for x in items)),
            gpu=gr["gpu"], throughput=gr["bench"],
            egrg_flag_breakdown=json.load(open(out_path("stepA", "egrg_flag_breakdown.json")))
            if os.path.exists(out_path("stepA", "egrg_flag_breakdown.json")) else None),
        reproduction_gate=dict(ok=gates["repro_ok"], pairs={t: f"{sum(v['identical_to_summary'] for v in gates['repro'][t].values())}/9" for t in TASKS}),
        oof_reference=gates["oof_reference_mean"],
        coverage_before_scanning=cov,
        boxes=dict(final=boxes["boxes"], checks=boxes["checks"], labelled={k: v for k, v in boxes["labelled"].items()}),
        clock_check=dict(summary=clock["summary"], excluded_blocks=len(ex_blocks), files_with_excluded_blocks=shifted,
                         lags_in_those_files=sorted({b["lag"] for b in same_file}),
                         blocks_in_those_files=len(same_file), strong_blocks_in_those_files=sum(bool(b.get("strong")) for b in same_file),
                         lag_histogram_strong={str(k): v for k, v in sorted(Counter(b["lag"] for b in clock["blocks"] if b.get("strong")).items())}),
        gpu_budget=dict(cap_h=GPU_CAP_H, used_h=round(gpu_used_hours(), 3), ledger=ledger),
        deviations=deviations_list())
    for p in (out_path("scan_results.json"), out_path("results", "summary.json")):
        atomic_json(p, R)
    txt = render_stop(R)
    for p in (out_path("scan_results.txt"), out_path("results", "summary.txt")):
        with open(p, "w") as f:
            f.write(txt)
    print(txt)


def render_stop(R):
    L = []
    w = L.append
    sa = R["stepA"]
    w("EEG-GATED RACINE GRADER ON THE CONTINUOUS RECORDINGS: OUTCOME OF THE PRE-REGISTERED STEP A")
    w(f"created {R['created']}; pre-registration {R['prereg']} sha256 {R['prereg_sha256'][:16]}")
    w("")
    w("OUTCOME: STOP (section 7). Arms K and E both fail the registered tolerance, so the scan was not run. No flag")
    w("rate, episode rate, matched comparison, sensitivity or gallery exists. An amendment is needed before scanning.")
    w("")
    w(f"STEP A: {sa['sample']['n']} labelled clips from {sa['sample']['animals']} animals, strata {sa['sample']['by_stratum']}, partners "
      f"{sa['sample']['partners']}; each (clip, network) row = one of the clip's 3 OOF x3dfix networks; GPU {sa['gpu']}")
    w("  tolerance per head: argmax agreement >= 0.97 and median |dp| <= 0.05 vs the stored val_ep12 predictions")
    w("  arm  head  rows  argmax  median|dp|  p95|dp|  mean dP(sz)  video-flag  EGRG-flag  macro-F1 by seed (stored: g3 "
      f"{sa['stored_macro_f1_by_seed']['g3']}, g5 {sa['stored_macro_f1_by_seed']['g5']})")
    for arm in "CKE":
        for t in TASKS:
            r = sa["arms"][arm][t]
            b = sa["breakdown"][f"{arm}|{t}"]
            w(f"  {arm}    {t}   {r['rows']:5d}  {r['argmax_agree']:.4f}  {r['median_abs_dp']:.4f}      {r['p95_abs_dp']:.4f}   "
              f"{r['mean_signed_dPsz']:+.4f}     {b['video_flag_agree']:.4f}      {r['egrg_flag_agree']:.4f}     "
              f"{[round(x, 3) for x in r['macro_f1_by_seed']]}  {'PASS' if r['passes'] else 'FAIL'}")
    w("  argmax agreement by stratum (agreeing / rows):")
    for arm in "KE":
        for t in TASKS:
            b = sa["breakdown"][f"{arm}|{t}"]
            w(f"    {arm} {t}: " + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in b["argmax_agree_by_stage"].items()) +
              f"; disagreements that change the video flag {b['disagreements_changing_the_video_flag']}, within seizure grades "
              f"{b['disagreements_within_seizure_grades']}; median stored top-2 margin of the disagreeing rows "
              f"{b['median_stored_top2_margin_of_disagreeing_rows']:.3f}")
    if sa["pixels_vs_C"]:
        for arm, p in sa["pixels_vs_C"].items():
            w(f"  arm {arm} pixels vs the cached clip frames (C): mean |difference| per clip median {p['mae_vs_C_median']:.2f}, p95 "
              f"{p['mae_vs_C_p95']:.2f}, max {p['mae_vs_C_max']:.2f} grey levels; mean level {p['mean_level_minus_C_median']:+.2f}")
    fb = sa.get("egrg_flag_breakdown")
    if fb:
        w("  EGRG flags with the stored EEG and G_all, 9 (video, EEG) pairs per clip, on the 120 non-seizure clips (1080 rows):")
        for k, c in fb.items():
            w(f"    {k.replace('|', ' ')}: stored {c['ns_stored_flag']}, arm {c['ns_arm_flag']} (new {c['ns_01']}, lost {c['ns_10']}); "
              f"on the 240 seizure clips (2160 rows) stored {c['sz_stored']}, arm {c['sz_arm']}")
    e = sa["eeg_R"]
    w(f"  EEG arm R (raw EDF): argmax {e['argmax_agree']:.4f}, median |dp| {e['median_abs_dp']:.6f}, p95 {e['p95_abs_dp']:.4f}, max "
      f"{e['max_abs_dp']:.4f} -> {'PASS' if e['passes'] else 'FAIL'}; raw vs clip EDF samples corr median {e['raw_vs_clip_corr']['median']:.6f}, "
      f"min {e['raw_vs_clip_corr']['min']:.4f}; best lag (samples) {e['best_lag_counts']}; the dropout rule would remove "
      f"{e['dropout_rule_would_remove']} of 360 clips")
    w(f"  decision: {sa['decision']}")
    w(f"  sample notes: r0 one GOP before the seek keyframe in {sa['sample']['r0_one_gop_earlier']} clips; min NCC "
      f"{sa['sample']['ncc_min']:.4f}; max keyframe snap {sa['sample']['max_keyframe_snap_frames']} frames; redrawn "
      f"{len(sa['sample']['redrawn'])}; ineligible {sa['sample']['ineligible']}")
    tp = sa["throughput"]
    w(f"  throughput on this GPU: X3D-M {tp['x3d_fwd_per_s_bs64']:.0f} forwards/s at batch 64 ({tp['x3d_fwd_per_s_bs64_incl_upload_norm']:.0f} "
      f"with upload + normalisation per network), cudnn.benchmark {tp['x3d_fwd_per_s_bs64_cudnn_benchmark']:.0f}; TCN "
      f"{tp['tcn_windows_per_s']:.0f} windows/s")
    w("")
    w("WHAT EXISTS BEFORE SCANNING")
    rg = R["reproduction_gate"]
    w(f"  reproduction gate (section 5): {'PASSED' if rg['ok'] else 'FAILED'}; per-pair EGRG hits and macro-F1 identical to "
      f"ttg_recut/rescore/summary.json for {rg['pairs']}")
    ref = R["oof_reference"]
    w(f"  OOF reference (flags per hour if every window behaved like an OOF non-seizure clip): g5 EGRG "
      f"{ref['g5']['egrg_flags_per_hour_if_like_oof_nonseizure']:.2f}, video {ref['g5']['video_flags_per_hour_if_like_oof_nonseizure']:.2f}, "
      f"EEG {ref['g5']['eeg_flags_per_hour_if_like_oof_nonseizure']:.2f}; g3 EGRG {ref['g3']['egrg_flags_per_hour_if_like_oof_nonseizure']:.2f}, "
      f"video {ref['g3']['video_flags_per_hour_if_like_oof_nonseizure']:.2f}")
    w("  window accounting from S0 (hours; decode-time reasons - EEG dropout, clock, decode error - not applied):")
    for an, c in R["coverage_before_scanning"].items():
        w(f"    {an}: {c['span'][0]} .. {c['span'][1]} = {c['hours']:.1f} h; scannable after S0 {c['candidate_hours_after_S0']:.1f} h; "
          + ", ".join(f"{k} {v:.1f}" for k, v in c["excluded_hours_S0"].items()))
    w("  crop boxes (frozen): " + ", ".join(f"{k} {v}" for k, v in R["boxes"]["final"].items()) +
      "; the section-8 motion-share check fails for every box, including the largest allowed one (see deviations)")
    ck = R["clock_check"]
    w("  clock check (1-h blocks per camera file and box; motion and Activity only):")
    for an, c in ck["summary"].items():
        w(f"    {an}: blocks {c['blocks']}, measured {c['measured']}, strong {c['strong']}, strong and consistent "
          f"{c['strong_consistent']}, excluded {c['excluded']}")
    w(f"    all {ck['excluded_blocks']} excluded blocks are in {ck['files_with_excluded_blocks']}: every measured block of that "
      f"room-C file ({ck['blocks_in_those_files']}, {ck['strong_blocks_in_those_files']} strong) sits at lag {ck['lags_in_those_files']} s, "
      f"i.e. its video is ~13 s off the EDF clock; this file is not in the audit's defect list")
    w(f"    lags of strong blocks (s): {ck['lag_histogram_strong']} (keyframe motion is assigned to the later keyframe, ~1 s late)")
    g = R["gpu_budget"]
    w(f"  GPU hours used: {g['used_h']:.3f} of {g['cap_h']:.1f}")
    w("")
    w("DEVIATIONS")
    for i, d in enumerate(R.get("deviations", []), 1):
        w(f"  {i}. [{d.get('section')}] {d.get('text')}")
    return "\n".join(L) + "\n"


command("report-stop")(cmd_report_stop)


# ============================================================================ CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=sorted(COMMANDS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--rn243", default=None, help="boxes-freeze: RN243 box x,w (from the median image)")
    ap.add_argument("--rn243_note", default="")
    ap.add_argument("--t_launch", type=float, default=None, help="GPU jobs: epoch time the job started (import time)")
    ap.add_argument("--camera", default=None)
    ap.add_argument("--files", default=None, help="decode: comma-separated camera file names")
    ap.add_argument("--index", type=int, default=None, help="decode: index into the decode unit list")
    ap.add_argument("--stride", type=int, default=1, help="decode: process units index, index+stride, ...")
    ap.add_argument("--pilot", action="store_true", help="decode/infer: the S3 pilot units only")
    ap.add_argument("--deadline_s", type=float, default=None, help="infer: stop starting new shards after this")
    ap.add_argument("--wait_s", type=float, default=0, help="infer: wait this long for new shards when idle")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--speculative_mode", default=None, help="decode: mode to decode in before step A decides")
    a = ap.parse_args()
    return COMMANDS[a.command](a)


if __name__ == "__main__":
    sys.exit(main())
