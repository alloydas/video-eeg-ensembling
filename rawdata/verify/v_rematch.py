# Independent re-match of labelled clips against raw video + raw EDF (read-only).
# Usage: python v_rematch.py <animal> <day> <clipdir>
# Writes one JSON line to OUTD/rematch.jsonl.
import os, sys, re, json, struct, subprocess, datetime as dt
import numpy as np
from scipy.signal import fftconvolve
from zoneinfo import ZoneInfo

FF = "/work/mech-ai/alloy/miniconda3/bin"
RAW = "/work/mech-ai/alloydas/EEG/Data"
DF = "/work/mech-ai-scratch/alloy/EEG/data_full"
OUTD = "/work/mech-ai-scratch/alloy/EEG/output/raw_audit/verify"
assert OUTD.startswith("/work/mech-ai-scratch/alloy/EEG/output/raw_audit/")
CT = ZoneInfo("America/Chicago"); UTC = dt.timezone.utc


def info(path):
    d = {}
    for line in open(path):
        if ':' in line:
            k, v = line.split(':', 1); d[k.strip()] = v.strip()
    return d


def pts_list(path, a, b, key_only=False):
    cmd = [f"{FF}/ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time,flags",
           "-read_intervals", f"{a}%{b}", "-of", "csv=p=0", path]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout.split()
    r = []
    for l in out:
        p, f = l.split(',')[:2]
        if key_only and 'K' not in f: continue
        r.append(float(p))
    return sorted(set(r))


def decode(path, ss, t, w, h, nmax=None):
    """Decode gray frames with original pts (copyts). Returns pts array, frames array."""
    cmd = [f"{FF}/ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "info"]
    if ss is not None: cmd += ["-ss", f"{ss:.3f}"]
    if t is not None: cmd += ["-t", f"{t:.3f}"]
    cmd += ["-copyts", "-i", path]
    if nmax: cmd += ["-frames:v", str(nmax)]
    cmd += ["-map", "0:v:0", "-vf", "showinfo,format=gray", "-vsync", "passthrough", "-f", "rawvideo", "pipe:1"]
    p = subprocess.run(cmd, capture_output=True)
    tb = re.search(rb"config in time_base: (\d+)/(\d+)", p.stderr)
    tbv = int(tb.group(1)) / int(tb.group(2))
    pts = [int(m) * tbv for m in re.findall(rb" n: *\d+ pts: *(-?\d+) ", p.stderr)]
    fr = np.frombuffer(p.stdout, np.uint8)
    n = fr.size // (w * h)
    fr = fr[: n * w * h].reshape(n, h, w).astype(np.float32)
    return np.array(pts[:n]), fr


def ncc_map(img, tpl):
    th, tw = tpl.shape
    t = tpl - tpl.mean(); tn = np.sqrt((t ** 2).sum())
    num = fftconvolve(img, t[::-1, ::-1], mode='valid')
    ii = np.pad(img.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    ii2 = np.pad((img ** 2).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    def box(I):
        return I[th:, tw:] - I[:-th, tw:] - I[th:, :-tw] + I[:-th, :-tw]
    s = box(ii); s2 = box(ii2); n = th * tw
    var = np.maximum(s2 - s * s / n, 1e-6)
    return num / (tn * np.sqrt(var))


def edf_read(path, chan_pred, rec_from=None, rec_to=None):
    with open(path, 'rb') as f:
        h = f.read(256)
        ns = int(h[252:256]); hb = int(h[184:192]); nrec = int(h[236:244]); rd = float(h[244:252])
        sd = h[168:176].decode(); stt = h[176:184].decode()
        s = f.read(ns * 256)
        lab = [s[i*16:(i+1)*16].decode('latin1').strip() for i in range(ns)]
        o = ns*16 + ns*80 + ns*8
        pmin = [float(s[o+i*8:o+(i+1)*8]) for i in range(ns)]; o += ns*8
        pmax = [float(s[o+i*8:o+(i+1)*8]) for i in range(ns)]; o += ns*8
        dmin = [float(s[o+i*8:o+(i+1)*8]) for i in range(ns)]; o += ns*8
        dmax = [float(s[o+i*8:o+(i+1)*8]) for i in range(ns)]; o += ns*8
        o += ns*80
        nsamp = [int(s[o+i*8:o+(i+1)*8]) for i in range(ns)]
        ci = next(i for i in range(ns) if chan_pred(lab[i], nsamp[i], rd))
        recb = 2 * sum(nsamp); off = 2 * sum(nsamp[:ci])
        if nrec < 0:
            nrec = (os.path.getsize(path) - hb) // recb
        r0 = 0 if rec_from is None else max(0, rec_from)
        r1 = nrec if rec_to is None else min(nrec, rec_to)
        out = []
        for r in range(r0, r1):
            f.seek(hb + r * recb + off)
            out.append(np.frombuffer(f.read(2 * nsamp[ci]), '<i2'))
        x = np.concatenate(out).astype(np.float64) if out else np.zeros(0)
        g = (pmax[ci] - pmin[ci]) / (dmax[ci] - dmin[ci])
        x = (x - dmin[ci]) * g + pmin[ci]
        dd, mm, yy = sd.split('.'); H, M, S = stt.split('.')
        start = dt.datetime(2000 + int(yy), int(mm), int(dd), int(H), int(M), int(S))
        return dict(label=lab[ci], fs=nsamp[ci] / rd, start=start, x=x, r0=r0, rd=rd, labels=lab, nrec=nrec)


def is1k(l, n, rd):
    return n / rd == 1000 and (l in ('ECG', 'EEG') or l.startswith('Channel'))


def main(animal, day, clip):
    cd = f"{DF}/Data_{animal}_cropped/{day}/{clip}"
    inf = info(f"{cd}/info.txt")
    out = dict(animal=animal, day=day, clip=clip)
    clip_start = dt.datetime.fromisoformat(inf['Clip start'])
    vname = inf['Video file']
    # find raw video in the animal's own folder, any day folder
    cands = [os.path.join(RAW, animal, d, vname) for d in os.listdir(os.path.join(RAW, animal))
             if os.path.exists(os.path.join(RAW, animal, d, vname))]
    out['raw_candidates'] = [os.path.relpath(c, RAW) for c in cands]
    raw = cands[0]
    xml = open(raw[:-4] + '.XML', encoding='latin1').read()
    utc0 = int(re.search(r'<DSI_utc_start_time>(\d+)<', xml).group(1))
    sync = int(re.search(r'<DSI_sync_offset>(\d+)<', xml).group(1)) / 1e7
    loc0 = dt.datetime.fromtimestamp(utc0, UTC).astimezone(CT).replace(tzinfo=None)
    seek = (clip_start - loc0).total_seconds()
    out['seek_mine'] = seek
    if 'Video seek' in inf:
        m = re.search(r'\(([\d.]+)s\)', inf['Video seek']); out['seek_info'] = float(m.group(1))
    # clip video
    pr = json.loads(subprocess.run([f"{FF}/ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                    "stream=width,height,nb_frames", "-of", "json", f"{cd}/video.mp4"],
                                   capture_output=True, text=True).stdout)['streams'][0]
    cw, ch, cn = int(pr['width']), int(pr['height']), int(pr.get('nb_frames', 0))
    out['clip_wh'] = (cw, ch); out['clip_nb_frames'] = cn
    cpts, cfr = decode(f"{cd}/video.mp4", None, None, cw, ch)
    out['clip_decoded'] = len(cfr)
    # raw window around seek
    keys = pts_list(raw, max(0, seek - 20), "+40", key_only=True)
    kb = [k for k in keys if k <= seek + 1e-6]
    out['keyframe_before_seek'] = kb[-1] if kb else None
    out['gop_s'] = float(np.median(np.diff(keys))) if len(keys) > 2 else None
    ss = max(0.0, seek - 8.0)
    rpts, rfr = decode(raw, ss, 16.0, 800, 600)
    out['raw_decoded'] = len(rfr); out['raw_pts_range'] = (float(rpts[0]), float(rpts[-1]))
    # crop box: NCC of clip frame 5 against each of a few raw frames; take best
    tpl = cfr[5]
    best = (-9, None, None)
    for i in range(0, len(rfr), 3):
        m_ = ncc_map(rfr[i], tpl)
        k = np.unravel_index(np.argmax(m_), m_.shape)
        if m_[k] > best[0]: best = (float(m_[k]), (int(k[1]), int(k[0])), i)
    out['crop_ncc'] = best[0]; out['crop_xy'] = best[1]; out['crop_box'] = (best[1][0], best[1][1], cw, ch)
    x0, y0 = best[1]
    crops = rfr[:, y0:y0 + ch, x0:x0 + cw]
    # frame alignment by frame-difference correlation: clip j <-> raw L+j
    def dcorr(A, B):
        A = A - A.mean(axis=(1, 2), keepdims=True); B = B - B.mean(axis=(1, 2), keepdims=True)
        return float(((A * B).sum(axis=(1, 2)) / np.sqrt((A * A).sum(axis=(1, 2)) * (B * B).sum(axis=(1, 2)) + 1e-9)).mean())
    J = min(60, len(cfr) - 1)
    dc = np.diff(cfr[:J + 1], axis=0); dr = np.diff(crops, axis=0)
    sc = np.array([dcorr(dc, dr[L:L + J]) for L in range(0, len(dr) - J)])
    L = int(np.argmax(sc)); srt = np.sort(sc)[::-1]
    mad = np.median(np.abs(sc - np.median(sc))) + 1e-9
    out['dcorr_best'] = float(srt[0]); out['dcorr_second'] = float(srt[1]); out['dcorr_z'] = float((srt[0] - np.median(sc)) / (1.4826 * mad))
    out['dcorr_neighbors'] = [round(float(sc[k]), 3) if 0 <= k < len(sc) else None for k in (L - 1, L + 1)]
    out['mae_at_L'] = float(np.abs(cfr[:J] - crops[L:L + J]).mean())
    out['raw_pts_of_clip_frame0'] = float(rpts[L])
    # end-of-clip check: decode raw around predicted pts of clip frames n-40..n-1
    n = len(cfr); j0 = n - 41
    pred = rpts[L] + j0 / 14.9975
    rp2, rf2 = decode(raw, max(0, pred - 4), 8.0 + 40 / 15, 800, 600)
    c2 = rf2[:, y0:y0 + ch, x0:x0 + cw]
    dc2 = np.diff(cfr[j0:j0 + 41], axis=0); dr2 = np.diff(c2, axis=0)
    sc2 = np.array([dcorr(dc2, dr2[k:k + 40]) for k in range(0, len(dr2) - 40)])
    k2 = int(np.argmax(sc2))
    out['end_check_pts'] = float(rp2[k2]); out['end_check_pred'] = float(pred)
    out['end_check_frames_off'] = round((rp2[k2] - pred) * 14.9975, 2); out['end_check_dcorr'] = float(sc2[k2])
    # frames between clip frame0 and end block in the raw: should equal j0 exactly if no drop/dup
    out['end_block_raw_frame_count'] = int(np.searchsorted(rp2, rp2[k2]) + 0)
    # raw-clock time of clip frame 0 vs info clip start (local)
    raw_local0 = loc0 + dt.timedelta(seconds=sync + rpts[L])
    out['clip_frame0_minus_info_start_s'] = (raw_local0 - clip_start).total_seconds()
    out['pts0_minus_keyframe_before_seek'] = (rpts[L] - out['keyframe_before_seek']) if kb else None
    # EEG
    try:
        c = edf_read(f"{cd}/eeg.edf", is1k)
        edfname = inf['EDF file']
        ecands = [os.path.join(RAW, animal, d, edfname) for d in os.listdir(os.path.join(RAW, animal))
                  if os.path.exists(os.path.join(RAW, animal, d, edfname))]
        out['raw_edf'] = [os.path.relpath(e_, RAW) for e_ in ecands]
        hdr = edf_read(ecands[0], is1k, 0, 0)
        implied = (clip_start - hdr['start']).total_seconds()
        r0 = int((implied - 30) // hdr['rd']); r1 = int((implied + 30 + 30) // hdr['rd']) + 1
        rr = edf_read(ecands[0], is1k, r0, r1)
        seg = c['x'][: 20000]
        base = rr['r0'] * rr['rd']
        a = rr['x']
        # NCC over lags
        n = len(seg); s0 = seg - seg.mean(); sn = np.sqrt((s0 ** 2).sum())
        num = fftconvolve(a, s0[::-1], mode='valid')
        cs = np.concatenate([[0], np.cumsum(a)]); cs2 = np.concatenate([[0], np.cumsum(a * a)])
        su = cs[n:] - cs[:-n]; su2 = cs2[n:] - cs2[:-n]
        den = sn * np.sqrt(np.maximum(su2 - su * su / n, 1e-12))
        ncc = num / den
        k = int(np.argmax(ncc))
        found = base + k / 1000.0
        out['clip_edf_label'] = c['label']; out['clip_edf_start'] = str(c['start']); out['raw_edf_label'] = rr['label']
        out['eeg_ncc'] = float(ncc[k]); out['eeg_lag_ms_vs_implied'] = round((found - implied) * 1000, 1)
        out['eeg_scale'] = float(np.std(seg) / np.std(a[k:k + n])) if np.std(a[k:k + n]) > 0 else None
        out['edf_start'] = str(hdr['start'])
        # EEG sample0 minus video frame0 on the local clock
        out['eeg0_minus_video0_s'] = (clip_start - raw_local0).total_seconds()
    except Exception as ex:
        out['eeg_error'] = repr(ex)
    with open(f"{OUTD}/rematch.jsonl", "a") as fo:
        fo.write(json.dumps(out, default=str) + "\n")
    print(json.dumps(out, default=str, indent=0))


if __name__ == '__main__':
    main(*sys.argv[1:4])
