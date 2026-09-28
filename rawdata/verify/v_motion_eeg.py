# Independent clock check: raw-video motion in the animal's crop vs raw-EDF EEG line length around an annotated seizure.
# Usage: python v_motion_eeg.py <animal> <day> <clipdir> <x,y,w,h>
# Video time axis = Central(DSI_utc_start_time) + sync + pts (i.e. assumes no frames missing);
# EEG axis = EDF header start + sample/fs. Reports lag (s) maximising corr(motion(t+lag), LL(t)).
import os, sys, re, json, subprocess, datetime as dt
import numpy as np
from zoneinfo import ZoneInfo
sys.path.insert(0, os.path.dirname(__file__))
from v_rematch import info, edf_read, is1k, FF, RAW, DF, OUTD, CT, UTC

def main(animal, day, clip, box, shift='0', half=150.0):
    x, y, w, h = map(int, box.split(','))
    cd = f"{DF}/Data_{animal}_cropped/{day}/{clip}"
    inf = info(f"{cd}/info.txt")
    onset = dt.datetime.fromisoformat(inf.get('Seizure start') or inf.get('Matched seizure start'))
    send = dt.datetime.fromisoformat(inf.get('Seizure end') or inf.get('Matched seizure end'))
    vname = inf['Video file']
    raw = next(os.path.join(RAW, animal, d, vname) for d in os.listdir(os.path.join(RAW, animal))
               if os.path.exists(os.path.join(RAW, animal, d, vname)))
    xml = open(raw[:-4] + '.XML', encoding='latin1').read()
    utc0 = int(re.search(r'<DSI_utc_start_time>(\d+)<', xml).group(1))
    sync = int(re.search(r'<DSI_sync_offset>(\d+)<', xml).group(1)) / 1e7
    loc0 = dt.datetime.fromtimestamp(utc0, UTC).astimezone(CT).replace(tzinfo=None) + dt.timedelta(seconds=sync)
    t_on = (onset - loc0).total_seconds() + float(shift)   # nominal pts of onset (+ optional shift)
    ss = max(0.0, t_on - half)
    W, H = 200, 150
    cmd = [f"{FF}/ffmpeg", "-nostdin", "-v", "error", "-ss", f"{ss:.3f}", "-t", f"{2*half:.1f}", "-i", raw,
           "-vf", f"crop={w}:{h}:{x}:{y},fps=5,scale={W}:{H},format=gray", "-f", "rawvideo", "pipe:1"]
    p = subprocess.run(cmd, capture_output=True)
    fr = np.frombuffer(p.stdout, np.uint8); n = fr.size // (W * H)
    fr = fr[:n * W * H].reshape(n, H, W).astype(np.float32)
    mot = np.abs(np.diff(fr, axis=0)).mean(axis=(1, 2))           # 5 Hz, t = ss + (i+1)/5
    tm = ss + (np.arange(len(mot)) + 1) / 5.0 - t_on                 # seconds relative to nominal onset
    # 1-s bins
    bins = np.arange(-half, half, 1.0)
    mb = np.array([mot[(tm >= b) & (tm < b + 1)].mean() if np.any((tm >= b) & (tm < b + 1)) else np.nan for b in bins])
    # EEG
    edfname = inf['EDF file']
    ep = next(os.path.join(RAW, animal, d, edfname) for d in os.listdir(os.path.join(RAW, animal))
              if os.path.exists(os.path.join(RAW, animal, d, edfname)))
    hdr = edf_read(ep, is1k, 0, 0)
    e_on = (onset - hdr['start']).total_seconds()
    r0 = int((e_on - half - 20) // 10); r1 = int((e_on + half + 20) // 10) + 1
    rr = edf_read(ep, is1k, r0, r1)
    te = rr['r0'] * 10 + np.arange(len(rr['x'])) / 1000.0 - e_on
    ll = np.abs(np.diff(rr['x']))
    lb = np.array([ll[(te[1:] >= b) & (te[1:] < b + 1)].sum() for b in bins])
    def z(a):
        a = np.where(np.isnan(a), np.nanmedian(a), a)
        return (a - np.median(a)) / (1.4826 * np.median(np.abs(a - np.median(a))) + 1e-9)
    mz, lz = z(mb), z(np.log(lb + 1e-9))
    lags = np.arange(-90, 91)
    cor = []
    for L in lags:  # corr(motion(t+L), LL(t))
        a = mz[max(0, L):len(mz) + min(0, L)]; b = lz[max(0, -L):len(lz) + min(0, -L)]
        cor.append(np.corrcoef(a, b)[0, 1])
    cor = np.array(cor); k = int(np.argmax(cor))
    # motion onset: first 1-s bin at/after -80 s where 3 consecutive bins exceed z 4
    def first_run(zz, thr=4.0):
        for i in range(len(zz) - 2):
            if bins[i] >= -80 and zz[i] > thr and zz[i + 1] > thr and zz[i + 2] > thr: return float(bins[i])
        return None
    out = dict(shift=float(shift), animal=animal, clip=clip, video=os.path.relpath(raw, RAW), nominal_onset_pts=round(t_on, 1),
               label_dur=round((send - onset).total_seconds(), 1),
               lag_best=int(lags[k]), r_best=round(float(cor[k]), 3), r_at0=round(float(cor[90]), 3),
               r_at_m61=round(float(cor[90 - 61]), 3), r_at_p61=round(float(cor[90 + 61]), 3),
               motion_onset_rel=first_run(mz), eeg_onset_rel=first_run(lz, 3.0),
               motion_z_in_label=round(float(np.nanmax(mz[(bins >= 0) & (bins < 60)])), 1),
               motion_z_at_label_minus61=round(float(np.nanmax(mz[(bins >= -61) & (bins < -1)])), 1))
    with open(f"{OUTD}/motion_eeg.jsonl", "a") as fo: fo.write(json.dumps(out) + "\n")
    print(json.dumps(out))

if __name__ == '__main__':
    main(*sys.argv[1:6])
