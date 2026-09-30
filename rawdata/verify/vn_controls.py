#!/usr/bin/env python3
"""The 5 reproduction controls, rechecked: eeg (parent labels) and info.txt bytes vs data_full, decoded frames vs
data_full (full-size gray MAE per frame), and the x264 core / encoder tags of new clips vs existing data_full clips.
Writes verify/vn_controls.json."""
import glob
import hashlib
import json
import os
import re
import sys
from collections import Counter

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vn_common as C  # noqa: E402


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def frames(p):
    w, h = [int(v) for v in C.run([C.FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
                                   "stream=width,height", "-of", "csv=p=0", p]).stdout.strip().split(",")]
    r = C.run([C.FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-i", p, "-vsync", "0", "-f", "rawvideo",
               "-pix_fmt", "gray", "pipe:1"], text=False)
    f = np.frombuffer(r.stdout, np.uint8)
    return f[: (f.size // (w * h)) * w * h].reshape(-1, h, w)


def x264_core(p):
    with open(p, "rb") as fh:
        b = fh.read(4 << 20)
    m = re.search(rb"x264 - core (\d+)", b)
    return int(m.group(1)) if m else None


def main():
    out = {"controls": []}
    for c in C.read_jsonl(os.path.join(C.NEW, "plan", "controls.jsonl")) if hasattr(C, "read_jsonl") else \
            [json.loads(x) for x in open(os.path.join(C.NEW, "plan", "controls.jsonl"))]:
        rel = c["rel"]
        rd = os.path.join(C.NEW, "repro", "data_full", rel)
        fd = os.path.join(C.FULL, rel)
        rec = dict(rel=rel, files=sorted(os.listdir(rd)))
        rec["info_identical"] = sha(os.path.join(rd, "info.txt")) == sha(os.path.join(fd, "info.txt"))
        pl = os.path.join(rd, "eeg_parentlabels.edf")
        rec["eeg_parentlabels_identical"] = os.path.exists(pl) and sha(pl) == sha(os.path.join(fd, "eeg.edf"))
        a, b = open(os.path.join(rd, "eeg.edf"), "rb").read(), open(os.path.join(fd, "eeg.edf"), "rb").read()
        rec["eeg_sourcelabels_bytes_differing"] = sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))
        fa, fb = frames(os.path.join(rd, "video.mp4")), frames(os.path.join(fd, "video.mp4"))
        rec["frames"] = [len(fa), len(fb)]
        n = min(len(fa), len(fb))
        mae = np.abs(fa[:n].astype(np.int16) - fb[:n].astype(np.int16)).mean(axis=(1, 2))
        rec["frames_differing"] = int((mae > 0).sum())
        rec["mae_max"] = float(mae.max())
        rec["x264_core"] = [x264_core(os.path.join(rd, "video.mp4")), x264_core(os.path.join(fd, "video.mp4"))]
        out["controls"].append(rec)
    M = [r for r in C.manifest() if r["status"] != "excluded"]
    out["x264_core_new"] = dict(Counter(x264_core(os.path.join(C.NEWFULL, f"Data_{r['animal']}_cropped", r["session"],
                                                                r["clip_dir"], "video.mp4")) for r in M))
    ex = []
    for a, s in sorted({(r["animal"], r["session"]) for r in M}):
        ex += sorted(glob.glob(os.path.join(glob.escape(os.path.join(C.FULL, f"Data_{a}_cropped", s)), "*", "video.mp4")))[:20]
    out["x264_core_existing_same_sessions"] = dict(Counter(x264_core(p) for p in ex))
    C.write_json("vn_controls.json", out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
