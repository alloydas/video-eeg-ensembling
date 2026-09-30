#!/usr/bin/env python3
"""Look at the RN201 / RN203 / RN243 crop boxes on raw frames (not medians): for each no-label camera, 8 files
spread over the recording, the keyframe nearest the middle of each and one 2 min later; full frames with both
boxes drawn, and the no-label crop at native resolution. Read-only; writes PNGs to ttg_scan/verify/boxes/."""
import json
import os
import sys

import av
import cv2
import numpy as np

E = "/work/mech-ai-scratch/alloy/EEG/output"
RAW = "/work/mech-ai/alloydas/EEG/Data"
OUT = f"{E}/ttg_scan/verify/boxes"
os.makedirs(OUT, exist_ok=True)
tj = json.load(open(f"{E}/ttg_scan/s0/tables.json"))
bj = json.load(open(f"{E}/ttg_scan/boxes/boxes.json"))
CAM = {"RN201": ("RN199-RN201", "RN199"), "RN203": ("RN203-RN204", "RN204"), "RN243": ("RN242-RN243", "RN242")}


def key_at(path, frac):
    c = av.open(path)
    s = c.streams.video[0]
    dur = float(s.duration * s.time_base) if s.duration else float(c.duration / 1e6)
    t = dur * frac
    c.seek(int(t / s.time_base), stream=s, backward=True, any_frame=False)
    for fr in c.decode(s):
        img = fr.to_ndarray(format="bgr24")
        ts = float(fr.pts * s.time_base)
        c.close()
        return img, ts
    c.close()
    return None, None


for an, (cam, pa) in CAM.items():
    files = [v for v in tj["cameras"][cam] if v["kind"] == "ok"]
    pick = [files[int(round(i))] for i in np.linspace(1, len(files) - 2, 8)]
    fulls, crops = [], []
    bx = bj["boxes"][an]
    pbx = bj["labelled"][pa]
    for v in pick:
        for frac in (0.5, 0.5 + 120 / 27000):
            img, ts = key_at(os.path.join(RAW, v["relpath"]), frac)
            if img is None:
                continue
            import datetime as dt
            utc = dt.datetime.utcfromtimestamp(v["t0"] + ts).strftime("%m-%d %H:%M")
            im = img.copy()
            for (x, y, w, h), col, lab in ((pbx, (0, 200, 255), pa), (bx, (0, 255, 0), an)):
                cv2.rectangle(im, (x, y), (x + w - 1, y + h - 1), col, 2)
                cv2.putText(im, lab, (x + 4, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            cv2.putText(im, utc + " UTC", (8, 590), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            fulls.append(cv2.resize(im, (400, 300), interpolation=cv2.INTER_AREA))
            x, y, w, h = bx
            cr = img[y:y + h, x:x + w].copy()
            cr = cv2.resize(cr, (int(w * 300 / h), 300)) if h != 300 else cr
            cv2.putText(cr, utc, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            crops.append(cr)
    rows = [np.hstack(fulls[i:i + 4]) for i in range(0, 16, 4)]
    cv2.imwrite(f"{OUT}/{an}_full.png", np.vstack(rows))
    W = max(c.shape[1] for c in crops)
    crops = [np.pad(c, ((0, 0), (0, W - c.shape[1]), (0, 0))) for c in crops]
    rows = [np.hstack(crops[i:i + 4]) for i in range(0, 16, 4)]
    cv2.imwrite(f"{OUT}/{an}_crops.png", np.vstack(rows))
    print(an, [v["name"] for v in pick], flush=True)
