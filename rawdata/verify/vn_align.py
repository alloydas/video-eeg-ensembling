#!/usr/bin/env python3
"""Priority 2 (video half): does each sampled new seizure clip show the labelled time?

Independent of the pipeline's measure/decide code. Per sampled clip (new, plus existing controls on the same
kind of files):
  1. timing model of the raw file: no known defect stamp, no DST span, |wall - stream| small (files.csv)
  2. decode the raw file over the clip +-W s in the animal's crop box (crop_boxes.csv; equals the clip's crop), gray,
     1/4 size, with per-frame pts (showinfo); per-frame motion = mean |frame difference|
  3. decode the clip's own video.mp4 the same way; find clip frame 0 in the raw decode by the motion sequence (NCC),
     confirm with pixel NCC; -> raw nominal pts of clip frame 0, compared with the cutter's seek (keyframe at/before)
  4. EDF Activity (1 Hz, raw EDF) against the raw motion: lag L at 0.1 s over +-30 s (Spearman on the per-second
     motion binned at each lag); strength = peak r and its lead over the best r >= 5 s away. Weak -> the partner
     animal on the same camera (partner box, partner EDF), then a wider window.
  5. shown-time error of clip frame 0 on the EDF clock: err0 = (T0 + pts_clip0 + L - lambda) - Clip start, with
     lambda = the median (L - sync) of the controls (own baseline; the pipeline's 0.739 s is reported beside it).
     For an aligned clip err0 = keyframe lead (0 .. -2.13 s) + sync (0..1 s).
Writes verify/vn_align.csv, vn_align.json; per-clip lag curves in verify/vn_align_curves/.
"""
import datetime as dt
import glob
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vn_common as C  # noqa: E402

DS = 4
W = 900.0
W_WIDE = 2700.0
LAM_PIPE = 0.739
DEFECT_STAMPS = {"20231014004100", "20231103122100", "20231205222300", "20231109055600"}
DST = dt.datetime(2023, 11, 5, 7, 0, 0)   # 02:00 CDT = 07:00 UTC
EXTRA_BOX = {"RN243": (400, 240, 360, 222)}   # _crop_roomd.sh fractions 0.50 0.40 0.45 0.37 of 800x600

NEW_SAMPLE = [
    ("RN197", "10-26-2023", "seizure_41_"), ("RN197", "10-26-2023", "seizure_65_"),
    ("RN197", "10-26-2023", "seizure_75_"), ("RN197", "10-26-2023", "seizure_91_"),
    ("RN199", "10-26-2023", "seizure_71_"), ("RN199", "10-26-2023", "seizure_89_"),
    ("RN199", "10-26-2023", "seizure_80_"),
    ("RN213", "11-30-2023 to 12-04-2023", "seizure_64_"), ("RN213", "11-30-2023 to 12-04-2023", "seizure_56_"),
    ("RN213", "11-30-2023 to 12-04-2023", "seizure_99_"),
    ("RN235", "11-30-2023 to 12-04-2023", "seizure_20_"), ("RN235", "11-30-2023 to 12-04-2023", "seizure_65_"),
    ("RN235", "11-30-2023 to 12-04-2023", "seizure_92_"), ("RN235", "11-30-2023 to 12-04-2023", "seizure_52_"),
    ("RN235", "11-18-2023", "seizure_08_"), ("RN235", "11-18-2023", "seizure_27_"),
    ("RN235", "11-20-2023", "seizure_28_"), ("RN235", "11-20-2023", "seizure_31_"),
    ("RN235", "10-20-2023 to 10-22-2023", "seizure_07_"), ("RN235", "10-26-2023", "seizure_06_"),
    ("RN242", "10-12-2023", "seizure_02_"), ("RN242", "10-12-2023", "seizure_07_"),
    ("RN242", "10-12-2023", "seizure_08_"), ("RN242", "10-12-2023", "seizure_09_"),
    ("RN219", "10-12-2023", "seizure_02_"), ("RN219", "10-12-2023", "seizure_03_"),
    ("RN224", "10-12-2023", "seizure_02_"), ("RN227", "10-12-2023", "seizure_03_"),
    ("RN216", "11-30-2023 to 12-04-2023", "seizure_04_"), ("RN216", "12-12-2023", "seizure_03_"),
]
# existing clips (controls): same cameras / sessions, files without a known defect
CTRL_SESSIONS = [("RN235", "11-30-2023 to 12-04-2023"), ("RN197", "10-26-2023"), ("RN199", "10-26-2023"),
                 ("RN213", "11-30-2023 to 12-04-2023"), ("RN235", "11-20-2023"), ("RN242", "10-11-2023(2)"),
                 ("RN235", "10-26-2023"), ("RN235", "11-18-2023"), ("RN235", "10-20-2023 to 10-22-2023")]


def ctrl_sample():
    """The last existing seizure clip with info.txt and eeg.edf in each control session (nearest the new tail)."""
    out = []
    for a, s in CTRL_SESSIONS:
        ds = sorted(glob.glob(os.path.join(glob.escape(os.path.join(C.FULL, f"Data_{a}_cropped", s)), "seizure_*")),
                    key=lambda d: int(os.path.basename(d).split("_")[1]))
        ds = [d for d in ds if os.path.exists(os.path.join(d, "info.txt"))]
        if ds:
            out.append((a, s, "_".join(os.path.basename(ds[-1]).split("_")[:2]) + "_"))
    return out


def boxes():
    b = {}
    for r in C.read_csv(os.path.join(C.AUDIT, "align", "crop_boxes.csv")):
        b.setdefault(r["animal"], (int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])))
    b.update(EXTRA_BOX)
    return b


BOX = boxes()
DECODE_LOG = []


def decode(path, t0, dur, box=None):
    vf = []
    if box:
        x, y, w, h = box
        vf.append(f"crop={w}:{h}:{x}:{y}")
    else:
        w, h = [int(v) for v in C.run([C.FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
                                       "stream=width,height", "-of", "csv=p=0", path]).stdout.strip().split(",")]
    ow, oh = w // DS, h // DS
    vf += [f"scale={ow}:{oh}:flags=area", "format=gray", "showinfo"]
    cmd = [C.FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "info", "-threads", "2"]
    if t0 > 0:
        cmd += ["-ss", f"{t0:.3f}"]
    cmd += ["-i", path]
    if dur:
        cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-vf", ",".join(vf), "-an", "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
    r = C.run(cmd, text=False, timeout=7200)
    fr = np.frombuffer(r.stdout, np.uint8)
    n = fr.size // (ow * oh)
    fr = fr[: n * ow * oh].reshape(n, oh, ow)
    pts = np.array([float(m.group(1)) for m in re.finditer(rb"pts_time:\s*([-0-9.e]+)", r.stderr)])
    k = min(n, len(pts))
    DECODE_LOG.append((os.path.basename(path), n, len(pts)))
    return fr[:k], pts[:k]


def motion(fr):
    f = fr.astype(np.float32)
    m = np.zeros(len(f), np.float32)
    m[1:] = np.abs(np.diff(f, axis=0)).mean(axis=(1, 2))
    return m


def rank(x):
    o = np.argsort(x, kind="stable")
    r = np.empty(len(x)); r[o] = np.arange(len(x))
    # average ties
    xs = x[o]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def lag_curve(t_rel, mot, act, act_t0, lags):
    """Spearman r between Activity (1 Hz, sample i covers [act_t0 + i, +1)) and motion binned per EDF second after
    shifting frame times by L. t_rel: frame times on the EDF-relative nominal timeline (s)."""
    out = []
    for L in lags:
        idx = np.floor(t_rel + L - act_t0).astype(np.int64)
        ok = (idx >= 0) & (idx < len(act))
        s = np.bincount(idx[ok], weights=mot[ok], minlength=len(act))
        c = np.bincount(idx[ok], minlength=len(act))
        good = c >= 10
        if good.sum() < 120:
            out.append(np.nan); continue
        a, m = act[good], s[good] / c[good]
        if a.std() == 0 or m.std() == 0:
            out.append(np.nan); continue
        out.append(float(np.corrcoef(rank(a), rank(m))[0, 1]))
    return np.array(out)


def strength(lags, r):
    if np.all(np.isnan(r)):
        return dict(L=None, r=None, lead=None, z=None, strong=False)
    k = int(np.nanargmax(r))
    L = float(lags[k])
    far = np.abs(lags - L) >= 5
    second = float(np.nanmax(r[far])) if np.any(far & ~np.isnan(r)) else np.nan
    z = float((r[k] - np.nanmean(r)) / (np.nanstd(r) + 1e-12))
    # parabolic refinement on the 0.1-s grid
    if 0 < k < len(r) - 1 and not np.isnan(r[k - 1]) and not np.isnan(r[k + 1]):
        den = r[k - 1] - 2 * r[k] + r[k + 1]
        if den < 0:
            L += 0.1 * 0.5 * (r[k - 1] - r[k + 1]) / den
    lead = float(r[k] - second) if not np.isnan(second) else None
    return dict(L=L, r=float(r[k]), lead=lead, z=z, strong=bool(r[k] >= 0.10 and lead is not None and lead >= 0.03))


_EDFS = {}


def edf_covering(animal, t):
    """The animal's raw EDF whose header span holds naive-local time t (own header reader)."""
    if animal not in _EDFS:
        L = []
        for f in glob.glob(os.path.join(glob.escape(os.path.join(C.RAW, animal)), "**", "*.edf"), recursive=True):
            try:
                E = C.EDF(f)
                L.append((E.start(), E.start() + dt.timedelta(seconds=E.nrec * E.rec_dur), f))
            except Exception:
                pass
        _EDFS[animal] = L
    c = [x for x in _EDFS[animal] if x[0] <= t < x[1]]
    return max(c, key=lambda x: x[1] - x[0])[2] if c else None


def activity(edf_path, t_from, t_to):
    E = C.EDF(edf_path)
    k = E.labels.index("Activity")
    s0 = (t_from - E.start()).total_seconds()
    s1 = (t_to - E.start()).total_seconds()
    s0 = max(0.0, np.floor(s0)); s1 = min(E.nrec * E.rec_dur, np.ceil(s1))
    x, _ = E.read_phys(k, s0, s1)
    return x, E.start() + dt.timedelta(seconds=s0)


def find_clip(animal, session, prefix, new=True):
    root = C.NEWFULL if new else C.FULL
    g = glob.glob(os.path.join(glob.escape(os.path.join(root, f"Data_{animal}_cropped", session)), prefix + "*"))
    g = [x for x in g if os.path.basename(x).startswith(prefix)]
    return g[0] if len(g) == 1 else None


def raw_video_of(animal, session, fname):
    g = glob.glob(os.path.join(glob.escape(os.path.join(C.RAW, animal)), "*", fname)) + \
        glob.glob(os.path.join(glob.escape(os.path.join(C.RAW, animal)), "*", "*", fname))
    g = [x for x in g if session.split(" ")[0] in x] or g
    return g[0] if g else None


def camera_partner(fname, animal):
    cam = fname.split(".")[0]
    parts = cam.split("-")
    others = [p for p in parts if p != animal]
    return others[0] if others else None


def one(item):
    try:
        return one_(item)
    except Exception as e:
        import traceback
        return dict(animal=item[0], session=item[1], set="new" if item[3] else "control", clip=item[2],
                    error=f"{e!r} {traceback.format_exc()[-300:]}"), None


def one_(item):
    animal, session, prefix, new = item
    d = find_clip(animal, session, prefix, new)
    rec = dict(animal=animal, session=session, set="new" if new else "control", clip=os.path.basename(d) if d else prefix)
    if d is None:
        rec["error"] = "clip not found"
        return rec, None
    info = C.parse_info(os.path.join(d, "info.txt"))[0]
    cs, ce = C.parse_dt(info["Clip start"]), C.parse_dt(info["Clip end"])
    onset = C.parse_dt(info.get("Seizure start"))
    vpath = MANIFEST_VIDEO.get(os.path.relpath(d, C.NEWFULL)) if new else None
    vpath = os.path.join(C.RAW, vpath) if vpath else raw_video_of(animal, session, info["Video file"])
    rec["video"] = os.path.relpath(vpath, C.RAW)
    X = C.xml_of(vpath)
    U, S = int(X["DSI_utc_start_time"]), float(X["DSI_sync_offset"]) / 1e7
    T0 = C.central_naive(U)
    rec["sync_s"] = S
    # 1. timing model of the file
    stamp = re.search(r"\.(\d{14})\.mp4$", vpath).group(1)
    F = FILES.get(rec["video"], {})
    start_utc = dt.datetime.fromtimestamp(U, dt.timezone.utc).replace(tzinfo=None)
    end_utc = C.parse_dt(F.get("end_utc", "")) or start_utc
    rec["defect_stamp"] = stamp in DEFECT_STAMPS
    rec["dst_span"] = start_utc < DST <= end_utc
    rec["wall_minus_stream_s"] = F.get("wall_minus_stream_s")
    rec["file_problems"] = F.get("problems")
    seek = max(0.0, (onset - T0).total_seconds() - 10.0)
    rec["seek_parent"] = seek
    # 2-3. raw decode around the clip and the clip decode
    clip_dur = (ce - cs).total_seconds()
    w0 = max(0.0, seek - W)
    fr_raw, pts = decode(vpath, w0, clip_dur + 2 * W, BOX[animal])
    pts = pts + w0
    m_raw = motion(fr_raw)
    fr_c, pts_c = decode(os.path.join(d, "video.mp4"), 0, None, None)
    m_c = motion(fr_c)
    rec["n_raw"], rec["n_clip"] = len(fr_raw), len(fr_c)
    # locate the clip in the raw decode by the motion sequence, then pixel NCC around it
    n = len(m_c)
    lo = max(0, int(np.searchsorted(pts, seek - 40)))
    hi = min(len(m_raw) - n, int(np.searchsorted(pts, seek + 40)))
    best = (-2, None)
    mc = (m_c[1:] - m_c[1:].mean()) / (m_c[1:].std() + 1e-9)
    for j in range(lo, max(lo, hi) + 1):
        seg = m_raw[j + 1: j + n]
        if len(seg) != n - 1:
            continue
        sz = (seg - seg.mean()) / (seg.std() + 1e-9)
        c = float((mc * sz).mean())
        if c > best[0]:
            best = (c, j)
    j0 = best[1]
    rec["motion_ncc"] = best[0]
    if j0 is not None:
        ks = np.linspace(0, n - 1, 25).astype(int)
        nc = []
        for k in ks:
            a = fr_c[k].astype(np.float64).ravel(); b = fr_raw[j0 + k].astype(np.float64).ravel()
            a -= a.mean(); b -= b.mean()
            nc.append(float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9)))
        rec["pixel_ncc_min"] = min(nc)
        rec["pts_clip0"] = float(pts[j0])
        rec["clip0_minus_seek_s"] = float(pts[j0] - seek)
        rec["keyframe_rule_ok"] = bool(-2.2 <= pts[j0] - seek <= 0.07)
    # 4. Activity vs motion (own animal, then partner, then wider)
    lags = np.round(np.arange(-30, 30.0001, 0.1), 3)
    edf_rel0 = None
    curves = {}

    def measure(an, box, fr_m, pts_m, width):
        ep = edf_covering(an, cs)
        if ep is None:
            return None
        if box is not None:
            fr2, p2 = decode(vpath, max(0.0, seek - width), clip_dur + 2 * width, box)
            p2 = p2 + max(0.0, seek - width)
            mm = motion(fr2)
        else:
            mm, p2 = fr_m, pts_m
        t_abs0 = T0
        act, a0 = activity(ep, t_abs0 + dt.timedelta(seconds=float(p2[0]) - 40),
                           t_abs0 + dt.timedelta(seconds=float(p2[-1]) + 40))
        t_rel = (T0 - a0).total_seconds() + p2          # frame times relative to the Activity series start
        r = lag_curve(t_rel, mm, act, 0.0, lags)
        st = strength(lags, r)
        st.update(edf=os.path.relpath(ep, C.RAW), animal=an, width=width, act_nonzero=int((act >= 0.5).sum()),
                  act_n=len(act))
        curves[f"{an}_{int(width)}"] = r
        return st

    tries = [measure(animal, None, m_raw, pts, W)]
    partner = camera_partner(info["Video file"], animal)
    if not tries[-1] or not tries[-1]["strong"]:
        if partner in BOX:
            tries.append(measure(partner, BOX[partner], None, None, W))
    if not any(t and t["strong"] for t in tries):
        tries.append(measure(animal, BOX[animal], None, None, W_WIDE))
        if partner in BOX:
            tries.append(measure(partner, BOX[partner], None, None, W_WIDE))
    good = [t for t in tries if t and t["strong"]]
    use = good[0] if good else (tries[0] or {})
    rec["tries"] = json.dumps([t for t in tries if t])
    rec["lag_L"] = use.get("L")
    rec["lag_r"] = use.get("r")
    rec["lag_lead"] = use.get("lead")
    rec["lag_strong"] = bool(use.get("strong"))
    rec["lag_from"] = f"{use.get('animal')} +-{use.get('width')}" if use else ""
    rec["act_nonzero"] = use.get("act_nonzero")
    return rec, curves


FILES = {}
MANIFEST_VIDEO = {f"Data_{r['animal']}_cropped/{r['session']}/{r['clip_dir']}": r["raw_video"] for r in C.manifest()}


def main():
    global FILES
    FILES = C.files_csv()
    items = [(a, s, p, True) for a, s, p in NEW_SAMPLE] + [(a, s, p, False) for a, s, p in ctrl_sample()]
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(one, items))
    rows = [r for r, _ in res]
    for (r, cv) in res:
        if cv:
            p = C.vout(f"vn_align_curves/{r['set']}_{r['animal']}_{r['clip'][:30]}.json")
            with open(p, "w") as fh:
                json.dump({k: [None if np.isnan(x) else round(float(x), 4) for x in v] for k, v in cv.items()}, fh)
    # own baseline lambda from strong controls: L - sync
    ctrl = [r for r in rows if r["set"] == "control" and r.get("lag_strong")]
    lam = float(np.median([r["lag_L"] - r["sync_s"] for r in ctrl])) if ctrl else LAM_PIPE
    for r in rows:
        if r.get("lag_L") is None or r.get("pts_clip0") is None:
            continue
        T0 = C.central_naive(int(C.xml_of(os.path.join(C.RAW, r["video"]))["DSI_utc_start_time"]))
        root = C.NEWFULL if r["set"] == "new" else C.FULL
        info = C.parse_info(os.path.join(root, f"Data_{r['animal']}_cropped", r["session"], r["clip"], "info.txt"))[0]
        cs = C.parse_dt(info["Clip start"])
        base = (T0 - cs).total_seconds() + r["pts_clip0"]
        r["residual_s"] = r["lag_L"] - r["sync_s"] - lam
        r["err0_s"] = base + r["lag_L"] - lam
        r["err0_pipe_lambda_s"] = base + r["lag_L"] - LAM_PIPE
    out = dict(lambda_own=lam, lambda_pipeline=LAM_PIPE, n_controls_strong=len(ctrl),
               n_new=sum(r["set"] == "new" for r in rows), n_new_strong=sum(r["set"] == "new" and r.get("lag_strong") for r in rows),
               animals_new=sorted({r["animal"] for r in rows if r["set"] == "new"}),
               defect_or_dst_files=[r["video"] for r in rows if r.get("defect_stamp") or r.get("dst_span")],
               decode_frames_vs_pts_mismatch=[x for x in DECODE_LOG if x[1] != x[2]],
               errors=[(r["clip"], r.get("error")) for r in rows if r.get("error")],
               keyframe_rule_ok=sum(bool(r.get("keyframe_rule_ok")) for r in rows),
               pixel_ncc_min=min(r.get("pixel_ncc_min", 1) for r in rows),
               rows=[{k: r.get(k) for k in ("set", "animal", "clip", "video", "sync_s", "clip0_minus_seek_s", "pixel_ncc_min",
                                             "lag_L", "lag_r", "lag_lead", "lag_strong", "lag_from", "act_nonzero",
                                             "residual_s", "err0_s", "err0_pipe_lambda_s", "wall_minus_stream_s")}
                     for r in rows])
    C.write_csv("vn_align.csv", rows)
    C.write_json("vn_align.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1, default=str))
    for r in out["rows"]:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
