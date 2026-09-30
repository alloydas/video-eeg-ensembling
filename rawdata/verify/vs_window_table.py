#!/usr/bin/env python3
"""Spot-check the S0 window table (s0/windows/<animal>.npz) from the raw files with independent code: for random
scannable minutes of every animal, the video file whose frames cover [m, m+60) (own XML parse + own PyAV packet
index), the frame count, the 16 keyframe-snapped positions, and the EDF placement (own header parse, naive local
time = UTC-5 before 2023-11-05 01:00 naive, UTC-6 after). Also a few excluded minutes per reason. Read-only."""
import datetime as dt
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vs_raw_recompute import edf_hdr, packets, pick_channel  # noqa: E402

E = "/work/mech-ai-scratch/alloy/EEG/output"
RAW = "/work/mech-ai/alloydas/EEG/Data"
FOLDER = {"RN199": "RN199", "RN201": "DONE_RN201", "RN203": "DONE_RN203", "RN204": "RN204", "RN242": "RN242",
          "RN243": "RN243"}
R = ["scannable", "no video", "unreadable file", "timing-defect file", "EDF stub", "DST hour", "unfinalised EDF",
     "EDF gap", "no 1-kHz channel", "EDF ambiguous", "partial window", "EEG dropout", "clock", "decode error"]
FALL = dt.datetime(2023, 11, 5, 1)
rng = np.random.default_rng(1)
tj = json.load(open(f"{E}/ttg_scan/s0/tables.json"))
out = []
pk = {}
for a in FOLDER:
    z = np.load(f"{E}/ttg_scan/s0/windows/{a}.npz")
    M, reason, fidx = z["minute"], z["reason"], z["fidx"]
    files = [str(x) for x in z["files"]]
    cand = np.flatnonzero(reason == 0)
    # own list of this animal's EDFs (header start, records, 1-kHz channel)
    edfs = []
    for p in sorted(glob.glob(f"{RAW}/{FOLDER[a]}/*/*.edf")):
        h = edf_hdr(p)
        ch = pick_channel(h["labels"])
        edfs.append((h, ch, h["nrec_file"] * h["rdur"]))
    for i in rng.choice(cand, 4, replace=False):
        m = int(M[i])
        name = files[fidx[i]]
        v = next(x for x in tj["cameras"][str(z["camera"])] if x["name"] == name)
        xml = open(os.path.join(RAW, v["relpath"][:-4] + ".XML")).read()
        t0 = int(re.search(r"<DSI_utc_start_time>(\d+)<", xml).group(1)) + \
            int(re.search(r"<DSI_sync_offset>(-?\d+)<", xml).group(1)) / 1e7
        if v["relpath"] not in pk:
            pk[v["relpath"]] = packets(os.path.join(RAW, v["relpath"]))
        pts, key, tb = pk[v["relpath"]]
        tt = t0 + pts * tb
        i0, i1 = np.searchsorted(tt, [m, m + 60])
        n = i1 - i0
        gaps_ok = (tt[i0] - m <= 1) and (m + 60 - tt[i1 - 1] <= 1) and np.diff(tt[i0:i1]).max() <= 1
        pos = i0 + np.linspace(0, n - 1, 16).round().astype(int)
        kf = np.flatnonzero(key)
        kf = np.r_[0, kf] if 0 not in kf else kf          # ffprobe also flags packet 0 as K
        sn = np.array([int(kf[np.abs(kf - p) == np.abs(kf - p).min()].min()) for p in pos])
        u = dt.datetime.fromtimestamp(m, dt.timezone.utc).replace(tzinfo=None)
        nv = u - dt.timedelta(hours=5)
        if nv >= FALL:
            nv = u - dt.timedelta(hours=6)
            if nv < FALL:
                nv = None
        cov = [(h, ch) for h, ch, dur in edfs if nv is not None and dur >= 3600 and h["start"] <= nv and
               nv + dt.timedelta(seconds=60) <= h["start"] + dt.timedelta(seconds=dur)]
        k0 = int(round((nv - cov[0][0]["start"]).total_seconds() * 1000)) if len(cov) == 1 else None
        e_tab = tj["edfs"][a]["edfs"][int(z["edf"][i])]
        rec = dict(animal=a, minute=dt.datetime.fromtimestamp(m, dt.timezone.utc).isoformat(), file=name,
                   t0_matches=abs(t0 - v["t0"]) < 1e-6, gaps_ok=bool(gaps_ok), n_own=int(n), n_tab=int(z["nfr"][i]),
                   first_match=int(i0) == int(z["first"][i]), kf16_match=sn.tolist() == z["kf16"][i].tolist(),
                   n_edf_cover=len(cov), edf_match=(len(cov) == 1 and os.path.basename(cov[0][0]["path"]) ==
                                                    os.path.basename(e_tab["relpath"])),
                   k0_own=k0, k0_tab=int(z["k0"][i]), channel_label=cov[0][0]["labels"][cov[0][1]] if cov else None,
                   fs=cov[0][0]["nsamp"][cov[0][1]] / cov[0][0]["rdur"] if cov else None)
        out.append(rec)
        print(rec, flush=True)
    # excluded minutes: one per reason present
    for code in sorted(set(int(x) for x in reason) - {0}):
        i = int(rng.choice(np.flatnonzero(reason == code)))
        m = int(M[i])
        u = dt.datetime.fromtimestamp(m, dt.timezone.utc)
        print(f"   {a} {u:%Y-%m-%d %H:%M} UTC: {R[code]} (file {files[fidx[i]] if fidx[i] >= 0 else '-'}, "
              f"day {z['day'][i]})", flush=True)
json.dump(out, open(f"{E}/ttg_scan/verify/window_table_check.json", "w"), indent=1, default=str)
ok = all(r["t0_matches"] and r["gaps_ok"] and r["n_own"] == r["n_tab"] and r["first_match"] and r["kf16_match"]
         and r["edf_match"] and r["k0_own"] == r["k0_tab"] and r["fs"] == 1000 for r in out)
print("ALL MATCH" if ok else "MISMATCH", len(out))
