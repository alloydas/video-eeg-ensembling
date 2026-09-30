#!/usr/bin/env python3
"""Priority 6: are the label-sanity flags right? Own EEG evidence (no detector), own motion, and pictures.

EEG evidence, from the RAW EDF (not the clip), FIR-HP 5 Hz channel (else the ECG/EEG channel), 1 kHz:
  line length per second, divided by the median of the +-600 s context outside every annotated event of the animal
  (any xlsx, +-30 s); a second is 'high' when that ratio >= THR. Per clip:
    seizure: fraction of high seconds inside the labelled span, longest high run touching [onset - 5, end + 5]
    negative: longest high run inside the clip window
  THR and the run length that counts as 'seizure evidence' are calibrated on existing clips of the same animals
  (existing seizures vs existing negatives), before looking at the new clips' flags.
Video: per-second motion of the clip itself (1/4 size gray), labelled span vs pre-buffer.
Figures for a sample (flagged + unflagged + references): EEG trace, LL context, clip motion, 6-frame montage.
Writes verify/vn_sanity.csv, vn_sanity.json, verify/vn_sanity_fig/*.png.
"""
import datetime as dt
import glob
import json
import os
import random
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vn_common as C  # noqa: E402

CTX = 600.0
N_REF = 250
TD = dt.timedelta
_E = {}


def edf(path):
    if path not in _E:
        _E[path] = C.EDF(path)
    return _E[path]


def chan(E):
    for i, l in enumerate(E.labels):
        if "FIR-HP" in l:
            return i
    return C.eeg_channel(E.labels)


EVENTS = defaultdict(list)


def load_events():
    for e in C.events_csv():
        st, en = C.parse_dt(e["start"]), C.parse_dt(e["end"])
        if st is None:
            continue
        EVENTS[e["animal"]].append((st, en or st + TD(seconds=float(e["duration_s"] or 30))))


def ll_context(animal, raw_edf, t_a, t_b):
    """LL per second (ratio to the context baseline) for [t_a - CTX, t_b + CTX] on the EDF clock."""
    E = edf(raw_edf)
    k = chan(E)
    s0 = max(0.0, (t_a - E.start()).total_seconds() - CTX)
    s1 = min(E.nrec * E.rec_dur, (t_b - E.start()).total_seconds() + CTX)
    s0, s1 = np.floor(s0), np.floor(s1)
    x, _ = E.read_phys(k, s0, s1)
    fs = int(E.fs[k])
    n = len(x) // fs
    x = x[: n * fs].reshape(n, fs)
    ll = np.abs(np.diff(x, axis=1)).sum(axis=1)
    t = np.array([E.start() + TD(seconds=s0 + i) for i in range(n)])
    # baseline: outside every annotated event of the animal (+-30 s) and outside [t_a, t_b] (+-30 s)
    ok = np.ones(n, bool)
    rel = np.array([(ti - E.start()).total_seconds() for ti in t])
    spans = [(a, b) for a, b in EVENTS[animal] if b >= t[0] - TD(seconds=60) and a <= t[-1] + TD(seconds=60)]
    spans.append((t_a, t_b))
    for a, b in spans:
        ra, rb = (a - E.start()).total_seconds() - 30, (b - E.start()).total_seconds() + 30
        ok &= ~((rel >= ra) & (rel < rb))
    base = np.median(ll[ok]) if ok.sum() > 60 else np.median(ll)
    return t, ll / (base + 1e-30), x, fs, s0, E


def runs(mask):
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def evidence(t, z, a, b, thr, pad=5):
    """seizure-type evidence for [a, b]: frac of high seconds inside, longest high run touching [a - pad, b + pad]."""
    inside = np.array([(a <= ti < b) for ti in t])
    hi = z >= thr
    frac = float(hi[inside].mean()) if inside.any() else float("nan")
    lo_t, hi_t = a - TD(seconds=pad), b + TD(seconds=pad)
    best = 0
    for i, j in runs(hi):
        if t[j] >= lo_t and t[i] < hi_t:
            best = max(best, j - i + 1)
    return frac, best


def clip_motion(d):
    p = os.path.join(d, "video.mp4")
    w, h = [int(v) for v in C.run([C.FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
                                   "stream=width,height", "-of", "csv=p=0", p]).stdout.strip().split(",")]
    ow, oh = w // 4, h // 4
    r = C.run([C.FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-i", p, "-vf",
               f"scale={ow}:{oh}:flags=area,format=gray", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"], text=False)
    fr = np.frombuffer(r.stdout, np.uint8)
    n = fr.size // (ow * oh)
    fr = fr[: n * ow * oh].reshape(n, oh, ow).astype(np.float32)
    m = np.zeros(n); m[1:] = np.abs(np.diff(fr, axis=0)).mean(axis=(1, 2))
    return m


def clip_list():
    M = C.manifest()
    out = []
    for r in M:
        if r["status"] == "excluded":
            continue
        d = os.path.join(C.NEWFULL, f"Data_{r['animal']}_cropped", r["session"], r["clip_dir"])
        out.append(dict(set="new", animal=r["animal"], kind=r["kind"], dir=d, raw_edf=os.path.join(C.RAW, r["raw_edf"]),
                        key=r["key"], flags=r["label_flags"], status=r["status"], p_oof=r["p_eeg_oof"],
                        ll_ratio=r["ll_ratio"], stage=r["stage"]))
    return out


def ref_list(animals):
    rng = random.Random(7)
    out = []
    for a in animals:
        dirs = glob.glob(os.path.join(glob.escape(os.path.join(C.FULL, f"Data_{a}_cropped")), "*", "*"))
        dirs = [d for d in dirs if os.path.exists(os.path.join(d, "info.txt"))]
        rng.shuffle(dirs)
        n_sz = n_ng = 0
        for d in dirs:
            b = os.path.basename(d)
            kind = "seizure" if b.startswith("seizure_") else "nonseizure"
            if kind == "seizure" and not re.search(r"Stage_[2-5]", b):
                continue
            if (kind == "seizure" and n_sz >= N_REF // len(animals) + 1) or (kind == "nonseizure" and n_ng >= N_REF // len(animals) + 1):
                continue
            info = C.parse_info(os.path.join(d, "info.txt"))[0]
            ef = info.get("EDF file")
            if not ef or ef == "N/A":
                continue
            sess = os.path.basename(os.path.dirname(d))
            cand = glob.glob(os.path.join(glob.escape(os.path.join(C.RAW, a, sess)), glob.escape(ef))) + \
                glob.glob(os.path.join(glob.escape(os.path.join(C.RAW, a, sess)), "*", glob.escape(ef)))
            if not cand:
                continue
            out.append(dict(set="existing", animal=a, kind=kind, dir=d, raw_edf=cand[0], key=os.path.relpath(d, C.FULL),
                            flags="", status="", stage=re.search(r"Stage_\d", b).group().replace("_", " ") if kind == "seizure" else "non-seizure"))
            if kind == "seizure":
                n_sz += 1
            else:
                n_ng += 1
    return out


def measure(c):
    info = C.parse_info(os.path.join(c["dir"], "info.txt"))[0]
    cs, ce = C.parse_dt(info["Clip start"]), C.parse_dt(info["Clip end"])
    if c["kind"] == "seizure":
        a, b = C.parse_dt(info["Seizure start"]), C.parse_dt(info["Seizure end"])
    else:
        a, b = cs, ce
    try:
        t, z, x, fs, s0, E = ll_context(c["animal"], c["raw_edf"], a, b)
    except Exception as e:
        return dict(c, error=str(e))
    return dict(c, a=a, b=b, cs=cs, ce=ce, t=t, z=z)


def main():
    load_events()
    new = clip_list()
    refs = ref_list(sorted({c["animal"] for c in new}))
    with ThreadPoolExecutor(8) as ex:
        allm = list(ex.map(measure, new + refs))
    ok = [m for m in allm if "error" not in m]
    # calibration on existing clips only: threshold and run length
    ref_sz = [m for m in ok if m["set"] == "existing" and m["kind"] == "seizure"]
    ref_ng = [m for m in ok if m["set"] == "existing" and m["kind"] == "nonseizure"]
    calib = {}
    for thr in (1.5, 2.0, 2.5, 3.0):
        rs = [evidence(m["t"], m["z"], m["a"], m["b"], thr)[1] for m in ref_sz]
        rn = [evidence(m["t"], m["z"], m["a"], m["b"], thr, pad=0)[1] for m in ref_ng]
        for L in (5, 8, 10, 15):
            sens = float(np.mean([x >= L for x in rs])); fpr = float(np.mean([x >= L for x in rn]))
            calib[f"{thr}|{L}"] = dict(sens=sens, fpr=fpr, youden=sens - fpr)
    best = max(calib, key=lambda k: calib[k]["youden"])
    THR, RUN = float(best.split("|")[0]), int(best.split("|")[1])
    rows = []
    for m in ok:
        pad = 5 if m["kind"] == "seizure" else 0
        frac, run = evidence(m["t"], m["z"], m["a"], m["b"], THR, pad)
        rows.append({k: m.get(k) for k in ("set", "animal", "kind", "stage", "key", "flags", "status", "p_oof", "ll_ratio")} |
                    dict(frac_high=frac, longest_run=run, evidence=run >= RUN,
                         span_s=(m["b"] - m["a"]).total_seconds()))
    # reference rates
    ref_rate_sz = float(np.mean([r["evidence"] for r in rows if r["set"] == "existing" and r["kind"] == "seizure"]))
    ref_rate_ng = float(np.mean([r["evidence"] for r in rows if r["set"] == "existing" and r["kind"] == "nonseizure"]))
    new_sz = [r for r in rows if r["set"] == "new" and r["kind"] == "seizure"]
    new_ng = [r for r in rows if r["set"] == "new" and r["kind"] == "nonseizure"]
    out = dict(THR=THR, RUN=RUN, calibration=calib, n_ref_seizure=len(ref_sz), n_ref_negative=len(ref_ng),
               ref_seizure_evidence_rate=ref_rate_sz, ref_negative_evidence_rate=ref_rate_ng,
               new_seizure_evidence_rate=float(np.mean([r["evidence"] for r in new_sz])),
               new_negative_evidence_rate=float(np.mean([r["evidence"] for r in new_ng])),
               new_seizures_without_evidence=[(r["key"], r["stage"], r["longest_run"], round(r["frac_high"], 2), r["flags"][:60])
                                              for r in new_sz if not r["evidence"]],
               new_negatives_with_evidence=[(r["key"], r["longest_run"], r["flags"][:60]) for r in new_ng if r["evidence"]],
               errors=[(m["key"], m["error"]) for m in allm if "error" in m])
    C.write_csv("vn_sanity.csv", rows)
    C.write_json("vn_sanity.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "calibration"}, indent=1, default=str))
    print("best", best, calib[best])
    # figures for a sample
    want = ["seizure_92_", "clip_19_vs_seizure_20_", "clip_26_vs_seizure_27_", "11-20-2023/seizure_28_",
            "11-30-2023 to 12-04-2023/seizure_52_", "11-30-2023 to 12-04-2023/seizure_91_", "10-26-2023/seizure_56_",
            "10-26-2023/seizure_75_", "10-26-2023/seizure_76_", "RN199_cropped/10-26-2023/seizure_71_",
            "RN213_cropped/11-30-2023 to 12-04-2023/seizure_64_", "11-20-2023/seizure_31_",
            "RN242_cropped/10-12-2023/seizure_10_", "RN242_cropped/10-12-2023/seizure_08_"]
    extra = [r["key"] for r in new_sz if not r["evidence"]][:6] + [r["key"] for r in new_ng if r["evidence"]][:6]
    pick = []
    for m in ok:
        if m["set"] != "new":
            continue
        if any(w in m["key"] for w in want) or m["key"] in extra:
            pick.append(m)
    # two unflagged new seizures and two existing references as visual anchors
    pick += [m for m in ok if m["set"] == "new" and m["kind"] == "seizure" and not m["flags"]][:2]
    pick += [m for m in ok if m["set"] == "existing" and m["kind"] == "seizure"][:2]
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(lambda m: figure(m, THR), pick))
    print("figures:", len(pick))


def figure(m, thr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    E = edf(m["raw_edf"])
    k = chan(E)
    a, b = m["a"], m["b"]
    s_from = (a - E.start()).total_seconds() - 60
    s_to = (b - E.start()).total_seconds() + 60
    x, _ = E.read_phys(k, max(0, s_from), s_to)
    tt = np.arange(len(x)) / E.fs[k] + max(0, s_from) - (a - E.start()).total_seconds()
    mot = clip_motion(m["dir"])
    info = C.parse_info(os.path.join(m["dir"], "info.txt"))[0]
    cs = C.parse_dt(info["Clip start"])
    fig = plt.figure(figsize=(14, 10))
    ax1 = fig.add_subplot(4, 1, 1)
    ax1.plot(tt, x * 1e3, lw=0.3, color="k")
    ax1.axvspan(0, (b - a).total_seconds(), color="tab:red", alpha=0.15)
    ax1.set_title(f"{m['key']}  [{m['kind']} {m.get('stage')}]  flags: {(m.get('flags') or '')[:110]}", fontsize=8)
    ax1.set_xlabel("s from labelled onset / clip start (raw EDF, " + E.labels[k] + ")", fontsize=8)
    ax2 = fig.add_subplot(4, 1, 2)
    tz = np.array([(ti - a).total_seconds() for ti in m["t"]])
    ax2.plot(tz, m["z"], lw=0.6)
    ax2.axhline(thr, color="tab:orange", lw=0.8)
    ax2.axvspan(0, (b - a).total_seconds(), color="tab:red", alpha=0.15)
    for (a2, b2) in EVENTS[m["animal"]]:
        r0, r1 = (a2 - a).total_seconds(), (b2 - a).total_seconds()
        if -CTX - 60 < r0 < CTX + 60 and not (abs(r0) < 1e-3):
            ax2.axvspan(r0, r1, color="tab:green", alpha=0.25)
    ax2.set_ylim(0, max(6, float(np.nanpercentile(m["z"], 99.5)) + 1))
    ax2.set_xlabel("s from labelled onset: line length / context median (green = other annotated events)", fontsize=8)
    ax3 = fig.add_subplot(4, 1, 3)
    tm = np.arange(len(mot)) / 15.0 + (cs - a).total_seconds()
    ax3.plot(tm, mot, lw=0.5)
    ax3.axvspan(0, (b - a).total_seconds(), color="tab:red", alpha=0.15)
    ax3.set_xlabel("s from labelled onset: clip video motion (mean |frame diff|)", fontsize=8)
    # montage
    dur = len(mot) / 15.0
    span0 = (a - cs).total_seconds()
    times = [max(0.5, span0 - 5), span0 + 2, span0 + (b - a).total_seconds() / 3, span0 + 2 * (b - a).total_seconds() / 3,
             min(dur - 0.5, span0 + (b - a).total_seconds() - 1), min(dur - 0.2, span0 + (b - a).total_seconds() + 6)]
    for i, ts in enumerate(times):
        r = C.run([C.FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-ss", f"{max(0, ts):.2f}", "-i",
                   os.path.join(m["dir"], "video.mp4"), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"],
                  text=False)
        w, h = [int(v) for v in C.run([C.FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
                                       "stream=width,height", "-of", "csv=p=0", os.path.join(m["dir"], "video.mp4")]).stdout.strip().split(",")]
        fr = np.frombuffer(r.stdout, np.uint8)
        ax = fig.add_subplot(4, 6, 19 + i)
        if fr.size >= w * h:
            ax.imshow(fr[: w * h].reshape(h, w), cmap="gray")
        ax.set_title(f"{ts - span0:+.1f} s", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    name = re.sub(r"[^\w]+", "_", m["key"].replace("data/", "").replace("/video.mp4", ""))[:120]
    fig.savefig(C.vout(f"vn_sanity_fig/{m['set']}_{name}.png"), dpi=80)
    plt.close(fig)


if __name__ == "__main__":
    main()
