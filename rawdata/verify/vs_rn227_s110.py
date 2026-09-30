#!/usr/bin/env python3
"""Where in the raw EDF is the EEG of RN227 seizure_110 (raw-vs-clip corr 0.13 at its stated start)? Exact search of
the clip's digital samples in the raw EDF(s) of that animal within +-12 h. Read-only."""
import csv, datetime as dt, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vs_raw_recompute import edf_hdr, edf_read, pick_channel
E = "/work/mech-ai-scratch/alloy/EEG"
RAW = "/work/mech-ai/alloydas/EEG/Data"
it = json.load(open(f"{E}/output/ttg_scan/stepA/items.json"))
x = next(v for v in it["items"] if "RN227" in v["key"] and "seizure_110" in v["key"])
c = next(r for r in csv.DictReader(open(f"{E}/output/raw_audit/align/clips.csv"))
         if f"data/Data_{r['animal']}_cropped/{r['session']}/{r['clip_dir']}/video.mp4" == x["key"])
print({k: c[k] for k in c if any(s in k for s in ("clip_start", "edf", "seek", "raw_video", "status", "lag", "corr"))})
hc = edf_hdr(os.path.join(E, "data_full", x["key"][5:-10], "eeg.edf"))
chc = pick_channel(hc["labels"])
n = hc["nrec_file"] * hc["nsamp"][chc]
_, cd = edf_read(hc, chc, 0, n)
print("clip EDF start", hc["start"], "n", n, "label", hc["labels"][chc])
hr = edf_hdr(os.path.join(RAW, c["raw_edf"]))
ch = pick_channel(hr["labels"])
cs = dt.datetime.fromisoformat(c["info_clip_start"])
s0 = int(round((cs - hr["start"]).total_seconds() * 1000))
tot = hr["nrec_file"] * hr["nsamp"][ch]
lo, hi = max(0, s0 - 12 * 3600000), min(tot, s0 + 12 * 3600000 + n)
_, rd_ = edf_read(hr, ch, lo, hi - lo)
pat = cd[:200]
# exact match of the first 200 digital samples
cand = np.flatnonzero(rd_[:len(rd_) - 200] == pat[0])
hits = [int(i) for i in cand if np.array_equal(rd_[i:i + 200], pat)]
print("raw EDF", c["raw_edf"], "start", hr["start"], "stated s0", s0, "hits (offset from stated start, s):",
      [round((lo + h - s0) / 1000, 3) for h in hits][:10])
for h in hits[:3]:
    seg = rd_[h:h + n]
    print("  full-length identical:", bool(len(seg) == n and np.array_equal(seg, cd)), "at raw naive time",
          hr["start"] + dt.timedelta(milliseconds=lo + h))
# ---- normalised cross-correlation search (the clip EDF may be rescaled, so exact digital matching can miss)
from scipy.signal import fftconvolve, resample_poly
print("scaling clip", hc["dmin"][chc], hc["dmax"][chc], hc["pmin"][chc], hc["pmax"][chc], "raw", hr["dmin"][ch], hr["dmax"][ch], hr["pmin"][ch], hr["pmax"][ch])
lo2, hi2 = max(0, s0 - 48 * 3600000), min(tot, s0 + 48 * 3600000 + n)
xr, _ = edf_read(hr, ch, lo2, hi2 - lo2)
xc, _ = edf_read(hc, chc, 0, n)
a = resample_poly(xr, 1, 8); b = resample_poly(xc, 1, 8)
b = (b - b.mean()) / b.std()
num = fftconvolve(a, b[::-1], mode="valid")
w = np.ones(len(b))
s1 = fftconvolve(a, w, mode="valid"); s2 = fftconvolve(a * a, w, mode="valid")
sd = np.sqrt(np.maximum(s2 / len(b) - (s1 / len(b)) ** 2, 1e-30))
r = num / len(b) / sd
k = int(np.nanargmax(r))
off = (lo2 + k * 8 - s0) / 1000
print("best NCC within +-48 h:", round(float(r[k]), 4), "at offset", round(off, 2), "s from the stated start =",
      hr["start"] + dt.timedelta(milliseconds=lo2 + k * 8))
top = np.argsort(r)[-5:][::-1]
print("top 5:", [(round(float(r[i]), 3), round((lo2 + i * 8 - s0) / 1000, 1)) for i in top])
