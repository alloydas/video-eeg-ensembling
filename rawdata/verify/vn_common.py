"""Shared helpers for the independent verification of ttg_newclips (vn_*.py).

Read-only everywhere except VOUT = $EEG_ROOT/output/ttg_newclips/verify/, and there only files named vn_*.
Own EDF reader (header + records, numpy), own XML/clock helpers; nothing is imported from cut_new_clips.py,
recut_clips.py or align_clips.py, so the checks do not share code with the pipeline they check.
"""
import csv
import datetime as dt
import json
import os
import re
import subprocess
from zoneinfo import ZoneInfo

import numpy as np

EEG_ROOT = os.environ.get("EEG_ROOT", "/work/mech-ai-scratch/alloy/EEG")
RAW = "/work/mech-ai/alloydas/EEG/Data"
NEW = os.path.join(EEG_ROOT, "output", "ttg_newclips")
NEWFULL = os.path.join(NEW, "data_full")
FULL = os.path.join(EEG_ROOT, "data_full")
DATA = os.path.join(EEG_ROOT, "data")
AUDIT = os.path.join(EEG_ROOT, "output", "raw_audit")
VOUT = os.path.join(NEW, "verify")
FFMPEG = "/work/mech-ai/alloy/miniconda3/bin/ffmpeg"
FFPROBE = "/work/mech-ai/alloy/miniconda3/bin/ffprobe"
CENTRAL = ZoneInfo("America/Chicago")
UTC = dt.timezone.utc


def vout(name):
    """The only write location: VOUT/vn_* (a file or a vn_* subdirectory)."""
    root = os.path.realpath(VOUT)
    p = os.path.realpath(os.path.join(root, name))
    rel = os.path.relpath(p, root)
    if rel.startswith("..") or not rel.split(os.sep)[0].startswith("vn_"):
        raise SystemExit(f"refusing to write {p}")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def write_csv(name, rows, cols=None):
    cols = cols or sorted({k for r in rows for k in r})
    p = vout(name)
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    return p


def write_json(name, obj):
    p = vout(name)
    with open(p, "w") as fh:
        json.dump(obj, fh, indent=1, default=str)
    return p


def read_csv(p):
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def manifest():
    return read_csv(os.path.join(NEW, "manifest.csv"))


def parse_dt(s):
    s = (s or "").strip()
    for f in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return dt.datetime.strptime(s, f)
        except ValueError:
            pass
    return None


def parse_info(path):
    d, order = {}, []
    with open(path) as fh:
        for line in fh.read().splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                d[k.strip()] = v.strip()
                order.append(k.rstrip())
    return d, order


def central_naive(epoch):
    """Central wall time (naive) of a UTC epoch: what datetime.fromtimestamp gives on a Central-time box."""
    return dt.datetime.fromtimestamp(epoch, UTC).astimezone(CENTRAL).replace(tzinfo=None)


def naive_local_to_utc(t, fold=0):
    return t.replace(tzinfo=CENTRAL, fold=fold).astimezone(UTC).replace(tzinfo=None)


def xml_of(mp4):
    base = os.path.splitext(mp4)[0]
    for ext in (".XML", ".xml"):
        if os.path.exists(base + ext):
            txt = open(base + ext, errors="replace").read()
            out = {}
            for tag in ("DSI_utc_start_time", "DSI_sync_offset", "DSI_local_start_time"):
                m = re.search(rf"<{tag}>([^<]*)</{tag}>", txt)
                out[tag] = m.group(1).strip() if m else None
            return out
    return None


# ------------------------------------------------------------------ EDF (own reader)
class EDF:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as fh:
            h = fh.read(256)
            self.fixed = h
            self.version = h[0:8].decode("ascii", "replace")
            self.patient = h[8:88].decode("ascii", "replace")
            self.recording = h[88:168].decode("ascii", "replace")
            self.startdate = h[168:176].decode("ascii", "replace")
            self.starttime = h[176:184].decode("ascii", "replace")
            self.header_bytes = int(h[184:192])
            self.reserved = h[192:236].decode("ascii", "replace")
            self.nrec = int(h[236:244])
            self.rec_dur = float(h[244:252])
            self.ns = int(h[252:256])
            ns = self.ns
            sh = fh.read(ns * 256)
            self.sig_header = sh

            # EDF: label16 transducer80 dim8 pmin8 pmax8 dmin8 dmax8 prefilt80 nsamp8 res32)
            offs, widths = [], [16, 80, 8, 8, 8, 8, 8, 80, 8, 32]
            acc = 0
            for w in widths:
                offs.append(acc); acc += w
            vals = []
            for off, w in zip(offs, widths):
                vals.append([sh[off * ns + i * w: off * ns + (i + 1) * w].decode("ascii", "replace") for i in range(ns)])
            (self.labels_raw, self.transducer, self.phys_dim, pmin, pmax, dmin, dmax, self.prefilter, nsamp,
             self.sig_reserved) = vals
            self.labels = [x.strip() for x in self.labels_raw]
            self.pmin_raw, self.pmax_raw, self.dmin_raw, self.dmax_raw = pmin, pmax, dmin, dmax
            self.pmin = np.array([float(x) for x in pmin]); self.pmax = np.array([float(x) for x in pmax])
            self.dmin = np.array([float(x) for x in dmin]); self.dmax = np.array([float(x) for x in dmax])
            self.nsamp = np.array([int(x) for x in nsamp])
        self.fs = self.nsamp / self.rec_dur
        self.rec_samples = int(self.nsamp.sum())
        self.rec_bytes = 2 * self.rec_samples
        self.sig_off = np.concatenate([[0], np.cumsum(self.nsamp)[:-1]])
        self.gain = (self.pmax - self.pmin) / (self.dmax - self.dmin)
        self.size = os.path.getsize(path)
        self.nrec_by_size = (self.size - self.header_bytes) // self.rec_bytes if self.rec_bytes else 0

    def start(self):
        d, t = self.startdate, self.starttime
        dd, mm, yy = [int(x) for x in d.split(".")]
        yy = 1900 + yy if yy >= 85 else 2000 + yy
        H, M, S = [int(x) for x in t.split(".")]
        return dt.datetime(yy, mm, dd, H, M, S)

    def channel(self, name_or_idx):
        if isinstance(name_or_idx, int):
            return name_or_idx
        return self.labels.index(name_or_idx)

    def read_digital(self, ch, rec0, nrec):
        ch = self.channel(ch)
        nrec = max(0, min(nrec, self.nrec_by_size - rec0))
        if nrec <= 0:
            return np.zeros(0, np.int16)
        with open(self.path, "rb") as fh:
            fh.seek(self.header_bytes + rec0 * self.rec_bytes)
            buf = np.frombuffer(fh.read(nrec * self.rec_bytes), dtype="<i2")
        buf = buf[: (buf.size // self.rec_samples) * self.rec_samples].reshape(-1, self.rec_samples)
        o, n = self.sig_off[ch], self.nsamp[ch]
        return buf[:, o:o + n].reshape(-1)

    def read_phys(self, ch, t0, t1):
        """Physical samples of channel ch for [t0, t1) seconds from file start (sample-exact)."""
        ch = self.channel(ch)
        fs = self.fs[ch]
        s0, s1 = int(round(t0 * fs)), int(round(t1 * fs))
        spr = self.nsamp[ch]
        r0, r1 = s0 // spr, -(-s1 // spr)
        dig = self.read_digital(ch, r0, r1 - r0).astype(np.float64)
        x = (dig - self.dmin[ch]) * self.gain[ch] + self.pmin[ch]
        a = s0 - r0 * spr
        return x[a:a + (s1 - s0)], dig[a:a + (s1 - s0)]

    def read_all_phys(self, ch):
        ch = self.channel(ch)
        dig = self.read_digital(ch, 0, self.nrec).astype(np.float64)
        return (dig - self.dmin[ch]) * self.gain[ch] + self.pmin[ch], dig


def eeg_channel(labels):
    """The downstream rule: first label containing EEG, else first containing ECG."""
    for key in ("EEG", "ECG"):
        for i, l in enumerate(labels):
            if key in l.upper():
                return i
    return None


def run(cmd, timeout=3600, text=True):
    return subprocess.run(cmd, capture_output=True, text=text, timeout=timeout)


def files_csv():
    return {r["relpath"]: r for r in read_csv(os.path.join(AUDIT, "files.csv"))}


def events_csv():
    return read_csv(os.path.join(AUDIT, "events.csv"))


# ------------------------------------------------------------------ xlsx (own zip/XML reader; openpyxl semantics)
def _col_index(ref):
    m = re.match(r"([A-Z]+)(\d+)", ref)
    c = 0
    for ch in m.group(1):
        c = c * 26 + (ord(ch) - 64)
    return c - 1, int(m.group(2))


def from_excel(v):
    """openpyxl.utils.datetime.from_excel: serial -> datetime rounded to the millisecond."""
    day, frac = divmod(v, 1)
    diff = dt.timedelta(milliseconds=round(frac * 86400 * 1000))
    if 0 < v < 60:
        day += 1
    return dt.datetime(1899, 12, 30) + dt.timedelta(days=day) + diff


def xlsx_events(path):
    """Rows of the active (first) sheet as the parent's parse_annotations sees them.

    Returns [dict(sheet_row, idx=sheet_row-1, label, duration, start, end, spikes)] for rows with a datetime start,
    plus the header list."""
    import zipfile
    import xml.etree.ElementTree as ET
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    z = zipfile.ZipFile(path)
    names = z.namelist()
    shared = []
    if "xl/sharedStrings.xml" in names:
        root = ET.fromstring(z.read("xl/sharedStrings.xml"))
        for si in root.findall("x:si", ns):
            shared.append("".join(t.text or "" for t in si.iter("{%s}t" % ns["x"])))
    sheet = "xl/worksheets/sheet1.xml"
    root = ET.fromstring(z.read(sheet))
    rows = {}
    for row in root.iter("{%s}row" % ns["x"]):
        r = int(row.get("r"))
        vals = {}
        for c in row.findall("x:c", ns):
            ci, _ = _col_index(c.get("r"))
            t = c.get("t", "n")
            v = c.find("x:v", ns)
            if v is None or v.text is None:
                isel = c.find("x:is", ns)
                vals[ci] = "".join(tt.text or "" for tt in isel.iter("{%s}t" % ns["x"])) if isel is not None else None
                continue
            if t == "s":
                vals[ci] = shared[int(v.text)]
            elif t in ("str", "inlineStr"):
                vals[ci] = v.text
            else:
                vals[ci] = float(v.text) if any(ch in v.text for ch in ".eE") else int(v.text)
        rows[r] = vals
    headers = [str(rows.get(1, {}).get(i, "") or "").strip().lower() for i in range(max(rows.get(1, {0: 0})) + 1)]

    def col(*keys):
        for k in keys:
            for i, h in enumerate(headers):
                if k in h:
                    return i
        return None
    ix = dict(label=col("label", "stage", "type"), dur=col("dur", "length"), start=col("start"), end=col("end"),
              spike=col("spike"))
    out = []
    for r in sorted(rows):
        if r < 2:
            continue
        v = rows[r]
        s = v.get(ix["start"])
        if not isinstance(s, (int, float)):
            continue            # the unit row ('[s]', text) and empty rows
        start = from_excel(s)
        e = v.get(ix["end"]) if ix["end"] is not None else None
        end = from_excel(e) if isinstance(e, (int, float)) else None
        lab = v.get(ix["label"]) if ix["label"] is not None else None
        lab = str(lab).strip() if lab is not None else "Unknown"
        d = v.get(ix["dur"]) if ix["dur"] is not None else None
        try:
            dur = float(d) if d is not None else 30.0
        except (TypeError, ValueError):
            dur = 30.0
        sp = v.get(ix["spike"]) if ix["spike"] is not None else 0
        out.append(dict(sheet_row=r, idx=r - 1, label=lab, duration=dur, start=start, end=end, spikes=sp))
    return out, headers
