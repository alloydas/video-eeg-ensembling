# Independent check that a re-cut clip shows its labelled time and the original did not.
# Written for the ttg_recut verification; shares no code with rawdata/recut_clips.py or align_clips.py.
#
# For each clip and each version (new = ttg_recut/data_full, old = EEG/data_full, or the parent's own clip for
# unaffected controls):
#   1. seek: the parent cutter's seek recomputed here from the raw XML + info.txt
#      (clip start - datetime.fromtimestamp(int(DSI_utc_start_time)) on a Central clock), compared with the
#      manifest's old_seek_pts; the keyframe at or before each seek found with ffprobe.
#   2. frame identity: the clip's first frames matched against the raw file decoded around the keyframe, cropped
#      with raw_audit/align/crop_boxes.csv; best time shift (frames) and best pixel shift of the crop (+-6 px).
#   3. Activity lag: the raw video from (seek - W - L) to (seek + dur + W + L), motion energy in the animal's box
#      binned per second on the clip's IMPLIED labelled time axis (labelled time t <-> raw pts seek + (t - clip
#      start)), correlated (Spearman) with the EDF Activity channel on the EDF's own naive axis, over lags
#      -L..L s. A clip that shows its labelled time peaks near the method baseline (-(sync) - ~0.7 s).
#   4. seizure burst: motion in the labelled seizure window [sz start, sz end] against every same-length window
#      within +-W (percentile), on the raw decode at the implied mapping; and the EEG line length the same way.
#   5. for the clips claimed to be EEG-disputed, the EEG line length at the label +-3600 s and +60 s.
# Output: $EEG_ROOT/output/ttg_recut/verify/indep/align.json (+ .csv). numpy + scipy; ffmpeg/ffprobe from miniconda3.
import os, re, sys, csv, json, subprocess, datetime as dt
from concurrent.futures import ThreadPoolExecutor
from zoneinfo import ZoneInfo
import numpy as np
from scipy.stats import rankdata
csv.field_size_limit(1 << 30)
FF = "/work/mech-ai/alloy/miniconda3/bin"
RAW = "/work/mech-ai/alloydas/EEG/Data"
EEG = "/work/mech-ai-scratch/alloy/EEG"
R = f"{EEG}/output/ttg_recut"
OUT = f"{R}/verify/indep"
CT = ZoneInfo("America/Chicago")
W = 600      # context each side of the clip (s)
L = 150      # lag search (s)
DS = 4       # spatial downscale for motion

SELECT = [  # (animal, clip_dir) : corrected clips across the 4 causes, then unaffected controls
    ("RN213", "seizure_41_Stage_3_20231103_050307"), ("RN213", "seizure_42_Stage_2_20231103_050406"),
    ("RN222", "seizure_17_Stage_3_20231103_041003"), ("RN216", "seizure_02_Stage_3_20231103_063808"),
    ("RN238", "seizure_06_Stage_3_20231205_142543"), ("RN197", "clip_06_vs_seizure_07_Stage_3_20231205_133211"),
    ("RN199", "seizure_27_Stage_2_20231103_045255"),
    ("RN210", "seizure_22_Stage_3_20231105_031600"), ("RN210", "seizure_23_Stage_3_20231105_054757"),
    ("RN219", "seizure_15_Stage_3_20231105_044051"),
    ("RN242", "seizure_33_Stage_3_20231105_044116"), ("RN242", "seizure_34_Stage_3_20231105_055439"),
    ("RN229", "seizure_41_Stage_2_20231105_051400"), ("RN229", "seizure_42_Stage_4_20231105_061409"),
    ("RN229", "seizure_39_Stage_4_20231105_012024"), ("RN224", "seizure_32_Stage_3_20231105_013058"),
    ("RN224", "seizure_34_Stage_2_20231105_030653"),
    ("RN242", "seizure_04_Stage_2_20231013_153338"), ("RN242", "seizure_03_Stage_2_20231013_135058"),
    ("RN242", "clip_04_vs_seizure_05_Stage_2_20231013_171433"),
]
CONTROLS = [("RN213", "seizure_40_Stage_3_20231103_021846"), ("RN197", "seizure_19_Stage_3_20231103_031504"),
            ("RN229", "seizure_38_Stage_4_20231105_003153"), ("RN242", "seizure_31_Stage_3_20231104_235534")]


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, **kw)


def ffprobe_json(path, extra):
    r = run([f"{FF}/ffprobe", "-v", "error", "-of", "json"] + extra + [path])
    return json.loads(r.stdout or b"{}")


def xml_times(mp4):
    x = open(mp4[:-4] + ".XML", encoding="latin1").read()
    u = int(re.search(r"<DSI_utc_start_time>\s*(\d+)", x).group(1))
    s = int(re.search(r"<DSI_sync_offset>\s*(-?\d+)", x).group(1)) / 1e7
    return u, s


def cutter_seek(clip_start, mp4):
    u, s = xml_times(mp4)
    t0 = dt.datetime.fromtimestamp(u, dt.timezone.utc).astimezone(CT).replace(tzinfo=None)
    return (clip_start - t0).total_seconds(), s


def keyframe_le(mp4, t):
    j = ffprobe_json(mp4, ["-select_streams", "v:0", "-skip_frame", "nokey", "-show_entries", "frame=best_effort_timestamp_time",
                           "-read_intervals", f"{max(0, t - 10):.3f}%{t + 1:.3f}"])
    ks = sorted(float(f["best_effort_timestamp_time"]) for f in j.get("frames", []) if "best_effort_timestamp_time" in f)
    ks = [k for k in ks if k <= t + 1e-6]
    return ks[-1] if ks else None


def parse_info(path):
    d = {}
    for ln in open(path, encoding="latin1"):
        if ":" in ln:
            k, v = ln.split(":", 1); d[k.strip()] = v.strip()
    f = lambda k: dt.datetime.fromisoformat(d[k]) if k in d else None  # noqa: E731
    return dict(clip_start=f("Clip start"), clip_end=f("Clip end"), sz_start=f("Seizure start"),
                sz_end=f("Seizure end"), video=d.get("Video file"), label=d.get("Label"))


class EDF:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            h = f.read(256); ns = int(h[252:256]); self.hb = int(h[184:192]); self.nrec = int(h[236:244])
            self.rd = float(h[244:252]); s = f.read(ns * 256)
        self.lab = [s[i * 16:(i + 1) * 16].decode("latin1").strip() for i in range(ns)]
        o = ns * 16 + ns * 80 + ns * 8     # label, transducer, physical dimension
        self.pmin = [float(s[o + i * 8:o + (i + 1) * 8]) for i in range(ns)]; o += ns * 8
        self.pmax = [float(s[o + i * 8:o + (i + 1) * 8]) for i in range(ns)]; o += ns * 8
        self.dmin = [float(s[o + i * 8:o + (i + 1) * 8]) for i in range(ns)]; o += ns * 8
        self.dmax = [float(s[o + i * 8:o + (i + 1) * 8]) for i in range(ns)]; o += ns * 8
        o += ns * 80
        self.nsamp = [int(s[o + i * 8:o + (i + 1) * 8]) for i in range(ns)]
        dd, mm, yy = h[168:176].decode().split("."); H, M, S = h[176:184].decode().split(".")
        self.start = dt.datetime(2000 + int(yy), int(mm), int(dd), int(H), int(M), int(S))
        self.recb = 2 * sum(self.nsamp)

    def channel(self, label, t0, t1):
        """Samples of `label` covering naive [t0, t1), physical units; returns (t_rel_to_t0 seconds, x, fs)."""
        ci = self.lab.index(label); off = 2 * sum(self.nsamp[:ci]); n = self.nsamp[ci]; fs = n / self.rd
        a = (t0 - self.start).total_seconds(); b = (t1 - self.start).total_seconds()
        r0 = max(0, int(a // self.rd)); r1 = min(self.nrec, int(b // self.rd) + 1)
        xs = []
        with open(self.path, "rb") as f:
            for r in range(r0, r1):
                f.seek(self.hb + r * self.recb + off); xs.append(np.frombuffer(f.read(2 * n), "<i2"))
        if not xs:
            return np.array([]), np.array([]), fs
        x = np.concatenate(xs).astype(float)
        g = (self.pmax[ci] - self.pmin[ci]) / (self.dmax[ci] - self.dmin[ci])
        x = (x - self.dmin[ci]) * g + self.pmin[ci]
        t = r0 * self.rd + np.arange(len(x)) / fs - a
        return t, x, fs

    def eeg_label(self):
        for c in ("EEG", "ECG"):
            if c in self.lab:
                return c
        raise KeyError(self.lab)


def decode_motion(mp4, box, p0, dur):
    """Per-frame motion energy (mean |diff| in the box, 1/DS scale, grey) and frame pts, from raw pts p0."""
    x, y, w, h = box
    p0c = max(0.0, p0)
    w2, h2 = (w // DS) // 2 * 2, (h // DS) // 2 * 2
    cmd = [f"{FF}/ffmpeg", "-nostdin", "-v", "info", "-ss", f"{p0c:.3f}", "-i", mp4, "-t", f"{dur - (p0c - p0):.3f}",
           "-vf", f"crop={w}:{h}:{x}:{y},scale={w2}:{h2}:flags=area,format=gray,showinfo", "-f", "rawvideo", "pipe:1"]
    r = run(cmd)
    fr = np.frombuffer(r.stdout, np.uint8); n = fr.size // (w2 * h2)
    fr = fr[:n * w2 * h2].reshape(n, h2, w2).astype(np.float32)
    pts = [float(m) for m in re.findall(rb"pts_time:\s*([-0-9.]+)", r.stderr)][:n]
    pts = np.array(pts) + p0c          # showinfo pts are relative to the seek point (output timeline)
    mot = np.r_[np.nan, np.abs(np.diff(fr, axis=0)).mean(axis=(1, 2))]
    k = min(len(pts), len(mot))              # showinfo can log a few lines fewer than frames piped
    return pts[:k], mot[:k], n


def bin_series(t, v, lo, n):
    """Sum of v in 1-s bins [lo + k, lo + k + 1), k < n; NaN where no sample."""
    k = np.floor(t - lo).astype(int); ok = (k >= 0) & (k < n) & np.isfinite(v)
    s = np.zeros(n); c = np.zeros(n)
    np.add.at(s, k[ok], v[ok]); np.add.at(c, k[ok], 1)
    out = s.copy(); out[c == 0] = np.nan
    return out, c


def spearman_lags(mot, act, lags):
    """mot has len(act) + 2L bins (mot[j + L] <-> act[j] at lag 0); r(lag) = corr(mot(t + lag), act(t))."""
    n = len(act); res = []
    for lag in lags:
        m = mot[L + lag: L + lag + n]
        ok = np.isfinite(m) & np.isfinite(act)
        if ok.sum() < float(os.environ.get("MINCOV", "0.6")) * n or np.nanstd(act[ok]) == 0 or np.nanstd(m[ok]) == 0:
            res.append(np.nan); continue
        a, b = rankdata(m[ok]), rankdata(act[ok])
        res.append(np.corrcoef(a, b)[0, 1])
    return np.array(res)


def window_percentile(series, t_lo, a, b):
    """Percentile of mean(series over [a, b)) among all same-length windows of the 1-s series (bins from t_lo)."""
    n = int(round(b - a)); i0 = int(np.floor(a - t_lo))
    s = series.copy()
    if n <= 0 or i0 < 0 or i0 + n > len(s):
        return None, None
    ok = np.isfinite(s); s[~ok] = np.nan
    cs = np.convolve(np.nan_to_num(s), np.ones(n), "valid") / n
    cnt = np.convolve(ok.astype(float), np.ones(n), "valid")
    valid = cnt >= 0.8 * n
    v = cs[i0]
    if not valid[i0]:
        return None, float(v)
    return float((cs[valid] < v).mean() * 100), float(v)


def analyse_version(tag, mp4, seek, clip_start, clip_end, sz, box, edf, sync, act, eegll, t_lo):
    dur = (clip_end - clip_start).total_seconds()
    n = len(act)
    p0 = seek - W - L
    pts, mot, nfr = decode_motion(mp4, box, p0, dur + 2 * W + 2 * L + 2)
    # implied labelled time of raw pts p: clip_start + (p - seek); bins over labelled [t_lo - L, t_lo + n + L)
    lab_rel = pts - seek - (-W - L)          # seconds from (clip_start - W - L)
    mb, cnt = bin_series(lab_rel, mot, 0.0, n + 2 * L)
    lags = np.arange(-L, L + 1)
    r = spearman_lags(mb, act, lags)
    k = int(np.nanargmax(r)) if np.isfinite(r).any() else None
    med = np.nanmedian(r); mad = np.nanmedian(np.abs(r - med)) * 1.4826
    base = -sync - 0.74
    kb = int(np.argmin(np.abs(lags - base)))
    near = np.abs(lags - base) <= 3
    out = dict(version=tag, video=os.path.relpath(mp4, RAW), seek=round(seek, 3), frames_decoded=int(nfr),
               video_cover=float(np.isfinite(mb[L:L + n]).mean()),
               best_lag=int(lags[k]) if k is not None else None, r_best=float(r[k]) if k is not None else None,
               z_best=float((r[k] - med) / (mad + 1e-9)) if k is not None else None,
               baseline_lag=round(base, 2), r_at_baseline=float(r[kb]) if np.isfinite(r[kb]) else None,
               r_max_within3_of_baseline=float(np.nanmax(r[near])) if np.isfinite(r[near]).any() else None,
               best_within3_of_baseline=bool(k is not None and abs(lags[k] - base) <= 3))
    if sz is not None:
        mot0 = mb[L:L + n]
        pct, val = window_percentile(mot0, 0.0, (sz[0] - clip_start).total_seconds() + W,
                                     (sz[1] - clip_start).total_seconds() + W)
        out["motion_sz_percentile"] = pct
        pre = mot0[W - 60 + 10: W + 10]                   # the 60 s before onset (clip start + 10 s)
        dur_sz = (sz[1] - sz[0]).total_seconds()
        dur_i = int(max(1, round(dur_sz)))
        during = mot0[W + 10: W + 10 + dur_i]
        out["motion_sz_over_pre60"] = float(np.nanmean(during) / (np.nanmean(pre) + 1e-9)) if np.isfinite(pre).any() else None
    return out


def eeg_ll_series(edf, t0, n):
    lab = edf.eeg_label()
    t, x, fs = edf.channel(lab, t0, t0 + dt.timedelta(seconds=n))
    if len(x) == 0:
        return np.full(n, np.nan)
    d = np.abs(np.diff(x)); td = t[1:]
    s, c = bin_series(td, d, 0.0, n)
    return s


def clip_frame_check(clip_mp4, raw_mp4, kf, box):
    """Match the clip's first 30 frames against raw frames around the keyframe; crop pixel-shift search."""
    x, y, w, h = box
    # clip: first 45 frames, half scale grey
    w2, h2 = (w // 2) // 2 * 2, (h // 2) // 2 * 2
    rc = run([f"{FF}/ffmpeg", "-nostdin", "-v", "error", "-i", clip_mp4, "-frames:v", "45", "-vf",
              f"scale={w2}:{h2}:flags=area,format=gray", "-f", "rawvideo", "pipe:1"])
    C = np.frombuffer(rc.stdout, np.uint8); nc = C.size // (w2 * h2); C = C[:nc * w2 * h2].reshape(nc, h2, w2).astype(np.float32)
    t0 = max(0.0, kf - 1.0)
    rr = run([f"{FF}/ffmpeg", "-nostdin", "-v", "info", "-ss", f"{t0:.3f}", "-i", raw_mp4, "-t", "6", "-vf",
              f"crop={w}:{h}:{x}:{y},scale={w2}:{h2}:flags=area,format=gray,showinfo", "-f", "rawvideo", "pipe:1"])
    Rr = np.frombuffer(rr.stdout, np.uint8); nr = Rr.size // (w2 * h2); Rr = Rr[:nr * w2 * h2].reshape(nr, h2, w2).astype(np.float32)
    pts = np.array([float(m) for m in re.findall(rb"pts_time:\s*([-0-9.]+)", rr.stderr)][:nr]) + t0
    m = min(30, nc)
    best = None
    for i in range(0, max(0, nr - m) + 1):
        e = np.abs(Rr[i:i + m] - C[:m]).mean()
        if best is None or e < best[0]:
            best = (e, i)
    # full-res crop geometry on frame 0: raw full frame at the matched pts, clip frame 0
    res = dict(n_clip_frames_read=int(nc), best_mae=float(best[0]) if best else None,
               matched_raw_pts=float(pts[best[1]]) if best and len(pts) > best[1] else None)
    if res["matched_raw_pts"] is not None:
        res["matched_minus_keyframe_s"] = round(res["matched_raw_pts"] - kf, 3)
        # contrast: NCC of the frame-DIFFERENCE images (motion), clip vs raw, at every time shift; the true shift
        # must win clearly (static frames make plain MAE nearly flat across shifts)
        Dc = np.diff(C[:m], axis=0); Dr = np.diff(Rr, axis=0)
        def ncc(a, b):
            a = a - a.mean(); b = b - b.mean(); return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))
        sc = [ncc(Dr[i:i + m - 1], Dc) for i in range(0, max(0, nr - m) + 1)]
        ib = int(np.argmax(sc))
        res["diff_ncc_best"] = sc[ib]; res["diff_ncc_best_shift_frames"] = ib - best[1]
        others = [v for i, v in enumerate(sc) if abs(i - ib) >= 2]
        res["diff_ncc_next_best_2plus_frames_away"] = max(others) if others else None
        tp = res["matched_raw_pts"]
        rf = run([f"{FF}/ffmpeg", "-nostdin", "-v", "error", "-ss", f"{max(0, tp - 0.02):.3f}", "-i", raw_mp4,
                  "-frames:v", "1", "-vf", "format=gray", "-f", "rawvideo", "pipe:1"])
        F = np.frombuffer(rf.stdout, np.uint8)
        cf = run([f"{FF}/ffmpeg", "-nostdin", "-v", "error", "-i", clip_mp4, "-frames:v", "1", "-vf", "format=gray",
                  "-f", "rawvideo", "pipe:1"])
        G = np.frombuffer(cf.stdout, np.uint8)
        if F.size == 800 * 600 and G.size == w * h:
            F = F.reshape(600, 800).astype(np.float32); G = G.reshape(h, w).astype(np.float32)
            grid = {}
            for dy in range(-6, 7):
                for dx in range(-6, 7):
                    yy, xx = y + dy, x + dx
                    if yy < 0 or xx < 0 or yy + h > 600 or xx + w > 800:
                        continue
                    grid[(dx, dy)] = float(np.abs(F[yy:yy + h, xx:xx + w] - G).mean())
            b = min(grid, key=grid.get)
            res["crop_best_shift_px"] = list(b); res["crop_mae_at_box"] = grid.get((0, 0)); res["crop_mae_best"] = grid[b]
        else:
            res["crop_note"] = f"frame sizes raw {F.size} clip {G.size} (box {w}x{h})"
    return res


def one(item):
    animal, cdir, kind, man, crow, boxes = item
    sess = crow["session"]
    info = parse_info(f"{EEG}/data_full/Data_{animal}_cropped/{sess}/{cdir}/info.txt")
    cs, ce = info["clip_start"], info["clip_end"]
    sz = (info["sz_start"], info["sz_end"]) if cdir.startswith("seizure_") else None
    box = boxes[animal]
    edf = EDF(os.path.join(RAW, crow["raw_edf"]))
    t_lo = cs - dt.timedelta(seconds=W)
    n = int(round((ce - cs).total_seconds())) + 2 * W
    ta, xa, fsa = edf.channel("Activity", t_lo, t_lo + dt.timedelta(seconds=n))
    act, _ = bin_series(ta, xa, 0.0, n)
    ll = eeg_ll_series(edf, t_lo, n)
    res = dict(animal=animal, clip_dir=cdir, session=sess, kind=kind, label=info["label"],
               clip_start=str(cs), dur=round((ce - cs).total_seconds(), 3), box=box,
               act_nonzero_bins=int(np.nansum(act > 0)))
    if sz is not None:
        pct, v = window_percentile(ll, 0.0, (sz[0] - cs).total_seconds() + W, (sz[1] - cs).total_seconds() + W)
        res["eeg_ll_sz_percentile"] = pct
        # EEG line length at shifted labels (duplicate-label / wrong-clock hypotheses)
        shifts = {}
        for sh in (-3600, -60, 60, 3600):
            t2 = t_lo + dt.timedelta(seconds=sh)
            ll2 = eeg_ll_series(edf, t2, n)
            p2, _ = window_percentile(ll2, 0.0, (sz[0] - cs).total_seconds() + W, (sz[1] - cs).total_seconds() + W)
            shifts[str(sh)] = p2
        res["eeg_ll_sz_percentile_label_shifted"] = shifts
    old_mp4 = os.path.join(RAW, os.path.dirname(crow["raw_video"]), info["video"]) if info["video"] else None
    if (old_mp4 is None or not os.path.exists(old_mp4)) and man is not None:
        old_mp4 = os.path.join(RAW, man["old_raw_video"])
    old_seek, sync = cutter_seek(cs, old_mp4)
    res["old_video_from_info"] = os.path.relpath(old_mp4, RAW)
    res["old_seek_recomputed"] = round(old_seek, 4)
    old_clip = f"{EEG}/data_full/Data_{animal}_cropped/{sess}/{cdir}/video.mp4"
    kf_old = keyframe_le(old_mp4, old_seek)
    res["old_keyframe"] = kf_old
    versions = [("old", old_mp4, old_seek, old_clip, kf_old)]
    if man is not None:
        res["manifest_old_seek"] = float(man["old_seek_pts"]); res["manifest_old_kf"] = float(man["old_keyframe_pts"])
        res["old_seek_diff_vs_manifest"] = round(old_seek - float(man["old_seek_pts"]), 4)
        res["old_video_matches_manifest"] = os.path.basename(man["old_raw_video"]) == os.path.basename(old_mp4)
        new_mp4 = os.path.join(RAW, man["new_raw_video"])
        new_seek = float(man["new_seek_pts"])
        kf_new = keyframe_le(new_mp4, new_seek)
        res.update(new_video=man["new_raw_video"], new_seek=new_seek, new_keyframe=kf_new,
                   new_kf_diff_vs_manifest=None if kf_new is None else round(kf_new - float(man["new_keyframe_pts"]), 4),
                   offset_used=float(man["offset_used_s"]), offset_source=man["offset_source"], cause=man["sub"])
        new_clip = f"{R}/data_full/Data_{animal}_cropped/{sess}/{cdir}/video.mp4"
        versions.append(("new", new_mp4, new_seek, new_clip, kf_new))
    sync_by_file = {}
    for tag, mp4, seek, clip, kf in versions:
        u, s = xml_times(mp4)
        a = analyse_version(tag, mp4, seek, cs, ce, sz, box, edf, s, act, ll, t_lo)
        if kf is not None and os.path.exists(clip):
            a["frames"] = clip_frame_check(clip, mp4, kf, box)
        res[tag] = a
    return res


def main():
    man = {}
    for r in csv.DictReader(open(f"{R}/manifest.csv")):
        man[(r["animal"], r["clip_dir"])] = r
    clips = {}
    for r in csv.DictReader(open(f"{EEG}/output/raw_audit/align/clips.csv")):
        clips[(r["animal"], r["clip_dir"])] = r
    boxes = {}
    for r in csv.DictReader(open(f"{EEG}/output/raw_audit/align/crop_boxes.csv")):
        boxes.setdefault(r["animal"], tuple(int(r[k]) for k in ("x", "y", "w", "h")))
    sel = SELECT + CONTROLS
    if len(sys.argv) > 1:
        sel = [s for s in sel if any(a in s[1] for a in sys.argv[1:])]
    items = []
    for a, c in sel:
        m = man.get((a, c))
        kind = "corrected" if m is not None else "control"
        items.append((a, c, kind, m, clips[(a, c)], boxes))
    def safe(it):
        try:
            return one(it)
        except Exception as e:  # noqa: BLE001  (report, never drop silently)
            import traceback
            return dict(animal=it[0], clip_dir=it[1], kind=it[2], error=repr(e), tb=traceback.format_exc()[-800:])
    with ThreadPoolExecutor(int(os.environ.get("NPAR", "8"))) as ex:
        res = list(ex.map(safe, items))
    os.makedirs(OUT, exist_ok=True)
    tag = os.environ.get("TAG", "align")
    json.dump(res, open(f"{OUT}/{tag}.json", "w"), indent=1, default=str)
    cols = ["animal", "clip_dir", "kind", "cause", "offset_source", "offset_used", "old_seek_diff_vs_manifest",
            "new_kf_diff_vs_manifest", "act_nonzero_bins", "eeg_ll_sz_percentile"]
    vcols = ["best_lag", "r_best", "z_best", "baseline_lag", "r_max_within3_of_baseline", "best_within3_of_baseline",
             "video_cover", "motion_sz_percentile", "motion_sz_over_pre60"]
    fcols = ["matched_minus_keyframe_s", "best_mae", "diff_ncc_best", "diff_ncc_best_shift_frames", "diff_ncc_next_best_2plus_frames_away", "crop_best_shift_px", "crop_mae_at_box"]
    with open(f"{OUT}/{tag}.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(cols + [f"{v}_{c}" for v in ("old", "new") for c in vcols + fcols])
        for r in res:
            row = [r.get(c) for c in cols]
            for v in ("old", "new"):
                d = r.get(v, {})
                row += [d.get(c) for c in vcols] + [d.get("frames", {}).get(c) for c in fcols]
            wr.writerow(row)
    for r in res:
        o, nw = r.get("old", {}), r.get("new", {})
        print(f"{r['kind'][:4]} {r['animal']} {r['clip_dir'][:40]:40s} {str(r.get('cause'))[:10]:10s} "
              f"old lag {o.get('best_lag')} r {o.get('r_best') and round(o['r_best'], 3)} base {o.get('baseline_lag')} "
              f"r@b {o.get('r_max_within3_of_baseline') and round(o['r_max_within3_of_baseline'], 3)} "
              f"szpct {o.get('motion_sz_percentile') and round(o['motion_sz_percentile'])} | "
              f"new lag {nw.get('best_lag')} r {nw.get('r_best') and round(nw['r_best'], 3)} "
              f"r@b {nw.get('r_max_within3_of_baseline') and round(nw['r_max_within3_of_baseline'], 3)} "
              f"szpct {nw.get('motion_sz_percentile') and round(nw['motion_sz_percentile'])} | "
              f"eegLL {r.get('eeg_ll_sz_percentile') and round(r['eeg_ll_sz_percentile'])} "
              f"shift {r.get('eeg_ll_sz_percentile_label_shifted')}", flush=True)


if __name__ == "__main__":
    main()
