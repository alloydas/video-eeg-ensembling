# Independent clock check: EDF 'Activity' (1 Hz) vs video motion in an animal's crop, in windows along one raw file.
# Usage: python v_activity.py <animal> <video relpath> <x,y,w,h> <window_s> <center_pts1,center_pts2,...>
# Video axis: Central(DSI_utc_start_time)+sync+pts; EDF axis: header start + elapsed. Lag L maximises corr(motion(t+L), Activity(t)).
import os, sys, re, json, subprocess, datetime as dt
import numpy as np
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor
FF = "/work/mech-ai/alloy/miniconda3/bin"; RAW = "/work/mech-ai/alloydas/EEG/Data"
OUTD = "/work/mech-ai-scratch/alloy/EEG/output/raw_audit/verify"
CT = ZoneInfo("America/Chicago")

def edf_channel(path, label, t0, t1):
    with open(path, 'rb') as f:
        h = f.read(256); ns = int(h[252:256]); hb = int(h[184:192]); nrec = int(h[236:244]); rd = float(h[244:252])
        s = f.read(ns * 256)
        lab = [s[i*16:(i+1)*16].decode('latin1').strip() for i in range(ns)]
        o = ns*16 + ns*80 + ns*8 + ns*8*4 + ns*80
        nsamp = [int(s[o+i*8:o+(i+1)*8]) for i in range(ns)]
        ci = lab.index(label); recb = 2 * sum(nsamp); off = 2 * sum(nsamp[:ci])
        dd, mm, yy = h[168:176].decode().split('.'); H, M, S = h[176:184].decode().split('.')
        start = dt.datetime(2000 + int(yy), int(mm), int(dd), int(H), int(M), int(S))
        a = (t0 - start).total_seconds(); b = (t1 - start).total_seconds()
        r0 = max(0, int(a // rd)); r1 = min(nrec if nrec > 0 else 10**9, int(b // rd) + 1)
        xs = []
        for r in range(r0, r1):
            f.seek(hb + r * recb + off); buf = f.read(2 * nsamp[ci])
            if len(buf) < 2 * nsamp[ci]: break
            xs.append(np.frombuffer(buf, '<i2'))
        x = np.concatenate(xs).astype(float)
        fs = nsamp[ci] / rd
        t = np.array([start + dt.timedelta(seconds=r0 * rd + k / fs) for k in range(len(x))])
        return t, x, fs

def window(animal, rel, box, W, c, loc0, edfp, eshift=0.0):
    x, y, w, h = map(int, box.split(','))
    ss = max(0.0, c - W / 2)
    Wd, Hd = 160, 120
    cmd = [f"{FF}/ffmpeg", "-nostdin", "-v", "error", "-ss", f"{ss:.3f}", "-t", f"{W:.1f}", "-i", f"{RAW}/{rel}",
           "-vf", f"crop={w}:{h}:{x}:{y},fps=5,scale={Wd}:{Hd},format=gray", "-f", "rawvideo", "pipe:1"]
    p = subprocess.run(cmd, capture_output=True)
    fr = np.frombuffer(p.stdout, np.uint8); n = fr.size // (Wd * Hd)
    fr = fr[:n * Wd * Hd].reshape(n, Hd, Wd).astype(np.float32)
    mot = np.abs(np.diff(fr, axis=0)).mean(axis=(1, 2))
    tv = ss + (np.arange(len(mot)) + 1) / 5.0                     # pts seconds
    sec0 = int(np.floor(ss)); nsec = int(W)
    mb = np.zeros(nsec)
    for k in range(nsec):
        m = (tv >= sec0 + k) & (tv < sec0 + k + 1)
        mb[k] = mot[m].sum() if m.any() else 0
    # video second k covers local time loc0 + sec0 + k
    vt0 = loc0 + dt.timedelta(seconds=sec0)
    vt0e = vt0 - dt.timedelta(seconds=eshift)   # EDF window start (EDF naive time assumed = video naive - eshift)
    L = 150
    te, act, fs = edf_channel(edfp, 'Activity', vt0e - dt.timedelta(seconds=L + 5), vt0e + dt.timedelta(seconds=nsec + L + 5))
    # resample activity to 1-s grid aligned to vt0 (nearest)
    rel_e = np.array([(t - vt0e).total_seconds() for t in te])
    grid = np.arange(-L, nsec + L)
    ag = np.interp(grid, rel_e, act)
    res = []
    for lag in range(-L, L + 1):
        # motion(t+lag) vs activity(t): motion bin k <-> activity at t = k - lag
        idx = np.arange(nsec) - lag + L
        a = ag[idx]
        if a.std() == 0 or mb.std() == 0: res.append(0); continue
        res.append(np.corrcoef(mb, a)[0, 1])
    res = np.array(res); k = int(np.argmax(res)); med = np.median(res); sd = np.std(res)
    top = [int(v) - L for v in np.argsort(res)[::-1][:3]]
    return dict(eshift=eshift, center_pts=c, local=(loc0 + dt.timedelta(seconds=c)).strftime('%m-%d %H:%M:%S'),
                lag=int(k - L), r=round(float(res[k]), 3), r0=round(float(res[L]), 3), r_m1=round(float(res[L - 1]), 3),
                r_m61=round(float(res[L - 61]), 3), z=round(float((res[k] - med) / (sd + 1e-9)), 1), top3=top,
                act_nonzero=int((ag != 0).sum()))

def main(animal, rel, box, W, centers, eshift='0'):
    W = float(W); centers = [float(v) for v in centers.split(',')]
    xml = open(f"{RAW}/{rel}"[:-4] + ".XML", encoding="latin1").read()
    u = int(re.search(r"<DSI_utc_start_time>(\d+)<", xml).group(1)); s = int(re.search(r"<DSI_sync_offset>(\d+)<", xml).group(1)) / 1e7
    loc0 = dt.datetime.fromtimestamp(u + s, dt.timezone.utc).astimezone(CT).replace(tzinfo=None)
    day = rel.split('/')[1]
    edfp = next(os.path.join(RAW, animal, day, f) for f in os.listdir(os.path.join(RAW, animal, day)) if f.lower().endswith('.edf'))
    with ThreadPoolExecutor(6) as ex:
        out = list(ex.map(lambda c: window(animal, rel, box, W, c, loc0, edfp, float(eshift)), centers))
    for o in out:
        o.update(animal=animal, video=rel)
        with open(f"{OUTD}/activity_scan.jsonl", "a") as fo: fo.write(json.dumps(o) + "\n")
        print(json.dumps(o))

if __name__ == "__main__":
    main(*sys.argv[1:7])
