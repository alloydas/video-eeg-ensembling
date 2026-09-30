#!/usr/bin/env python3
"""Telemetry dropouts inside clip EEG: longest run of identical samples in the picked channel (downstream rule), for
every new clip and for existing clips of the same sessions. Writes verify/vn_flat.json."""
import glob
import json
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vn_common as C  # noqa: E402


def longest_flat(path):
    E = C.EDF(path)
    k = C.eeg_channel(E.labels)
    _, d = E.read_all_phys(k)
    ch = np.flatnonzero(np.diff(d) != 0)
    if len(ch) == 0:
        return len(d) / 1000.0
    edges = np.concatenate([[-1], ch, [len(d) - 1]])
    return float(np.diff(edges).max() / 1000.0)


def main():
    M = [r for r in C.manifest() if r["status"] != "excluded"]
    new = {}
    for r in M:
        p = os.path.join(C.NEWFULL, f"Data_{r['animal']}_cropped", r["session"], r["clip_dir"], "eeg.edf")
        new[r["key"]] = longest_flat(p)
    ex = {}
    for a, s in sorted({(r["animal"], r["session"]) for r in M}):
        for p in sorted(glob.glob(os.path.join(glob.escape(os.path.join(C.FULL, f"Data_{a}_cropped", s)), "*", "eeg.edf"))):
            ex[os.path.relpath(p, C.FULL)] = longest_flat(p)
    def summ(d):
        v = np.array(list(d.values()))
        return dict(n=len(v), ge_1s=int((v >= 1).sum()), ge_5s=int((v >= 5).sum()), max=float(v.max()))
    out = dict(new=summ(new), existing_same_sessions=summ(ex),
               new_ge_1s=sorted([(k, v) for k, v in new.items() if v >= 1], key=lambda x: -x[1]))
    C.write_json("vn_flat.json", out)
    print(json.dumps(out, indent=1)[:3000])


if __name__ == "__main__":
    main()
