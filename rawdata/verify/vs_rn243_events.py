#!/usr/bin/env python3
"""RN243's two annotated seizures (Stage 2 10-12 08:08:36 local, Stage 5 10-16 13:07:02 local): frames of the
shared camera (both boxes) and the 1-kHz EEG of RN243 and RN242 around each. Read-only; PNGs to
ttg_scan/verify/rn243/."""
import datetime as dt
import json
import os
import sys

import av
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vs_raw_recompute import edf_hdr, edf_read, pick_channel  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

E = "/work/mech-ai-scratch/alloy/EEG/output"
RAW = "/work/mech-ai/alloydas/EEG/Data"
OUT = f"{E}/ttg_scan/verify/rn243"
os.makedirs(OUT, exist_ok=True)
tj = json.load(open(f"{E}/ttg_scan/s0/tables.json"))
bj = json.load(open(f"{E}/ttg_scan/boxes/boxes.json"))
EV = [("stage2_1012", "2023-10-12 08:08:36.153", 22.5), ("stage5_1016", "2023-10-16 13:07:02.231", 15.849)]
files = [v for v in tj["cameras"]["RN242-RN243"] if v["kind"] == "ok"]
for tag, st, dur in EV:
    naive = dt.datetime.fromisoformat(st)
    u = (naive + dt.timedelta(hours=5)).replace(tzinfo=dt.timezone.utc).timestamp()
    v = next(v for v in files if v["t0"] <= u <= v["fn_close"])
    c = av.open(os.path.join(RAW, v["relpath"]))
    s = c.streams.video[0]
    t_rel0 = u - v["t0"] - 12
    c.seek(int(t_rel0 / s.time_base), stream=s, backward=True, any_frame=False)
    want = list(np.arange(-10, dur + 20, 3.0))
    got, k = [], 0
    for fr in c.decode(s):
        tt = float(fr.pts * s.time_base) + v["t0"] - u
        if k < len(want) and tt >= want[k]:
            img = fr.to_ndarray(format="bgr24")
            for (x, y, w, h), col in ((bj["labelled"]["RN242"], (0, 200, 255)), (bj["boxes"]["RN243"], (0, 255, 0))):
                cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), col, 2)
            cv2.putText(img, f"{tt:+.1f} s", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            got.append(cv2.resize(img[200:520], (400, 160), interpolation=cv2.INTER_AREA))
            k += 1
        if k >= len(want):
            break
    c.close()
    while len(got) % 4:
        got.append(np.zeros_like(got[0]))
    cv2.imwrite(f"{OUT}/{tag}_frames.png", np.vstack([np.hstack(got[i:i + 4]) for i in range(0, len(got), 4)]))
    fig, ax = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    for a_, axx in zip(("RN243", "RN242"), ax):
        ed = [e for e in tj["edfs"][a_]["edfs"] if not e["stub"]]
        e = next(e for e in ed if dt.datetime.fromisoformat(e["start_naive"]) <= naive <=
                 dt.datetime.fromisoformat(e["start_naive"]) + dt.timedelta(seconds=e["records"] * e["rdur"]))
        h = edf_hdr(os.path.join(RAW, e["relpath"]))
        ch = pick_channel(h["labels"])
        k0 = int(((naive - dt.timedelta(seconds=60)) - h["start"]).total_seconds() * 1000)
        x, _ = edf_read(h, ch, k0, 150000)
        tt = np.arange(len(x)) / 1000 - 60
        axx.plot(tt, x * 1e3, lw=0.3)
        axx.axvspan(0, dur, color="orange", alpha=0.3)
        axx.set_ylabel(f"{a_} {h['labels'][ch]} (mV)")
    ax[1].set_xlabel("s from RN243's annotated start")
    fig.suptitle(f"RN243 {tag}: {st} local, {dur} s")
    fig.tight_layout()
    fig.savefig(f"{OUT}/{tag}_eeg.png", dpi=90)
    plt.close(fig)
    print(tag, v["name"], len(got))
