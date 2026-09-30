#!/usr/bin/env python3
"""Is the animal's cage in view in the last days of RN203-RN204 (cages seen absent at 12-14 16:48 UTC)?
Thumbnails every 2 h from 12-12 16:00 to 12-14 23:00 UTC, and per-hour EEG statistics of RN203 / RN204 from their
last EDF (std, longest identical run, share of |x| at the rails). Read-only; writes to ttg_scan/verify/."""
import datetime as dt
import json
import os
import sys

import av
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vs_raw_recompute import edf_hdr, edf_read, pick_channel  # noqa: E402

E = "/work/mech-ai-scratch/alloy/EEG/output"
RAW = "/work/mech-ai/alloydas/EEG/Data"
OUT = f"{E}/ttg_scan/verify"
cam = sys.argv[1] if len(sys.argv) > 1 else "RN203-RN204"
t_lo = dt.datetime.fromisoformat(sys.argv[2] if len(sys.argv) > 2 else "2023-12-12T16:00").replace(tzinfo=dt.timezone.utc).timestamp()
t_hi = dt.datetime.fromisoformat(sys.argv[3] if len(sys.argv) > 3 else "2023-12-14T23:00").replace(tzinfo=dt.timezone.utc).timestamp()
step = float(sys.argv[4]) if len(sys.argv) > 4 else 7200
tj = json.load(open(f"{E}/ttg_scan/s0/tables.json"))
files = [v for v in tj["cameras"][cam] if v["kind"] == "ok"]
thumbs = []
for t in np.arange(t_lo, t_hi, step):
    v = next((v for v in files if v["t0"] <= t <= v["fn_close"]), None)
    if v is None:
        continue
    c = av.open(os.path.join(RAW, v["relpath"]))
    s = c.streams.video[0]
    c.seek(int((t - v["t0"]) / s.time_base), stream=s, backward=True, any_frame=False)
    img = None
    for fr in c.decode(s):
        img = fr.to_ndarray(format="bgr24")
        break
    c.close()
    if img is None:
        continue
    im = cv2.resize(img, (320, 240), interpolation=cv2.INTER_AREA)
    cv2.putText(im, dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%m-%d %H:%M"), (4, 232),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    thumbs.append(im)
while len(thumbs) % 6:
    thumbs.append(np.zeros_like(thumbs[0]))
cv2.imwrite(f"{OUT}/cage_presence_{cam}.png", np.vstack([np.hstack(thumbs[i:i + 6]) for i in range(0, len(thumbs), 6)]))
# EEG per hour for the two animals of the camera
res = {}
for a in cam.split("-"):
    ed = [e for e in tj["edfs"][a]["edfs"] if not e["stub"]]
    rows = []
    for e in ed:
        s = dt.datetime.fromisoformat(e["start_naive"])
        u0 = (s + dt.timedelta(hours=6 if s >= dt.datetime(2023, 11, 5, 1) else 5)).replace(tzinfo=dt.timezone.utc).timestamp()
        u1 = u0 + e["records"] * e["rdur"]
        if u1 < t_lo or u0 > t_hi:
            continue
        h = edf_hdr(os.path.join(RAW, e["relpath"]))
        ch = pick_channel(h["labels"])
        for t in np.arange(max(u0, t_lo), min(u1, t_hi) - 600, 3600):
            k0 = int((t - u0) * 1000)
            x, d = edf_read(h, ch, k0, 600000)
            if x is None:
                continue
            runs = np.diff(np.flatnonzero(np.r_[True, np.diff(d) != 0, True])).max()
            rows.append(dict(utc=dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%m-%d %H:%M"),
                             std=round(float(x.std()), 6), longest_run=int(runs),
                             rail=float(np.mean((d <= h["dmin"][ch]) | (d >= h["dmax"][ch])))))
    res[a] = rows
json.dump(res, open(f"{OUT}/cage_presence_{cam}_eeg.json", "w"), indent=1)
for a, rows in res.items():
    print(a)
    for r in rows:
        print("  ", r)
