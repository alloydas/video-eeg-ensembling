#!/usr/bin/env python3
"""Zoomed raw-EEG panels (same y scale per clip) for label-sanity inspection: baseline, onset, mid-span, end.
Usage: vn_zoom.py <new|existing> <clip rel under data_full> [...]; writes verify/vn_sanity_fig/zoom_*.png"""
import os
import re
import sys

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vn_common as C  # noqa: E402
import vn_sanity as S  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

man = {f"Data_{r['animal']}_cropped/{r['session']}/{r['clip_dir']}": r for r in C.manifest()}


def raw_edf_for(which, rel):
    if which == "new":
        return os.path.join(C.RAW, man[rel]["raw_edf"])
    info = C.parse_info(os.path.join(C.FULL, rel, "info.txt"))[0]
    a, sess = rel.split("/")[0][5:-8], rel.split("/")[1]
    import glob
    g = glob.glob(os.path.join(glob.escape(os.path.join(C.RAW, a, sess)), "**", glob.escape(info["EDF file"])), recursive=True)
    return g[0]


def zoom(which, rel):
    root = C.NEWFULL if which == "new" else C.FULL
    info = C.parse_info(os.path.join(root, rel, "info.txt"))[0]
    if "Seizure start" in info:
        a, b = C.parse_dt(info["Seizure start"]), C.parse_dt(info["Seizure end"])
    else:
        a, b = C.parse_dt(info["Clip start"]), C.parse_dt(info["Clip end"])
    E = C.EDF(raw_edf_for(which, rel))
    k = S.chan(E)
    t0 = (a - E.start()).total_seconds()
    span = (b - a).total_seconds()
    wins = [(-40, -25, "baseline -40..-25 s"), (-5, 10, "onset -5..+10 s"), (span / 2 - 7.5, span / 2 + 7.5, "mid-span"),
            (span - 10, span + 5, "end -10..+5 s"), (span + 30, span + 45, "post +30..+45 s")]
    segs = [E.read_phys(k, t0 + w0, t0 + w1)[0] for w0, w1, _ in wins]
    lim = max(float(np.percentile(np.abs(np.concatenate(segs)), 99.9)), 1e-6) * 1.1
    fig, axs = plt.subplots(len(wins), 1, figsize=(14, 9))
    for ax, (w0, w1, lab), x in zip(axs, wins, segs):
        tt = np.arange(len(x)) / E.fs[k] + w0
        ax.plot(tt, x, lw=0.4, color="k")
        ax.set_ylim(-lim, lim)
        ax.axvspan(max(w0, 0), min(w1, span), color="tab:red", alpha=0.12)
        ll = np.abs(np.diff(x)).sum() / (len(x) / E.fs[k])
        ax.set_title(f"{lab}   LL/s={ll:.3g}", fontsize=8)
    fig.suptitle(f"{which}: {rel}  ({E.labels[k]}, labelled span {span:.1f} s)", fontsize=9)
    fig.tight_layout()
    name = re.sub(r"[^\w]+", "_", rel)[:120]
    fig.savefig(C.vout(f"vn_sanity_fig/zoom_{which}_{name}.png"), dpi=75)
    plt.close(fig)


if __name__ == "__main__":
    S.load_events()
    args = sys.argv[1:]
    for i in range(0, len(args), 2):
        zoom(args[i], args[i + 1])
