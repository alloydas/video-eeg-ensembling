#!/usr/bin/env python3
"""E-vs-C frame alignment margins: for step-A clips, MAE of raw frame r0+j+o (o in -3..3) against the cached
f16s224 frame, per sampled frame; is offset 0 clearly best where the scene moves? Read-only."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vs_raw_recompute import packets, decode_range, crop224
E = "/work/mech-ai-scratch/alloy/EEG"
R = json.load(open(f"{E}/output/ttg_scan/verify/raw_recompute.json"))
it = json.load(open(f"{E}/output/ttg_scan/stepA/items.json"))["items"]
fC = np.load(f"{E}/output/ttg_scan/stepA/frames_C.npy", mmap_mode="r")
out = []
for r in R:
    x = it[r["i"]]
    p, k, tb = packets("/work/mech-ai/alloydas/EEG/Data/" + x["relpath"])
    pos = np.array(x["pos_exact"])
    fr = decode_range("/work/mech-ai/alloydas/EEG/Data/" + x["relpath"], p, int(pos.min()) - 3, int(pos.max()) + 3)
    for t_, j in enumerate(pos):
        m = [float(np.abs(crop224(fr[j + o], x["box"]).astype(np.int16) - fC[r["i"]][t_].astype(np.int16)).mean()) for o in range(-3, 4)]
        out.append(dict(i=r["i"], t=t_, mae=m))
json.dump(out, open(f"{E}/output/ttg_scan/verify/align_margin.json", "w"))
m = np.array([o["mae"] for o in out])
best = m.argmin(1) - 3
gap = m[:, 3] - m.min(1)                  # how much worse offset 0 is than the best offset
spread = m.max(1) - m.min(1)              # how much the frames change over +-3
print("frames", len(m), "best offset 0:", int((best == 0).sum()))
mv = spread > 0.5                         # frames with visible motion over +-3 frames
print("moving frames (spread > 0.5 grey)", int(mv.sum()), "best offset 0 among them", int((best[mv] == 0).sum()))
print("offset-0 excess over best: median", float(np.median(gap)), "p99", float(np.percentile(gap, 99)), "max", float(gap.max()))
print("among moving frames: excess max", float(gap[mv].max()) if mv.any() else None, "spread median", float(np.median(spread[mv])) if mv.any() else None)
bad = np.flatnonzero(mv & (best != 0))
for b in bad[:10]:
    print(out[b]["i"], out[b]["t"], np.round(m[b], 3))
