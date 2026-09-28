# Independent clock check via room light transitions: find brightness steps near a target local time (read-only).
# Usage: python v_lights.py HH:MM:SS <before_s> <after_s> <step_s> <thr> rel1 [rel2 ...]
# Local time axis = Central(DSI_utc_start_time) + DSI_sync_offset + pts.
import sys, re, json, subprocess, datetime as dt
import numpy as np
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor
FF = "/work/mech-ai/alloy/miniconda3/bin"; RAW = "/work/mech-ai/alloydas/EEG/Data"
CT = ZoneInfo("America/Chicago")

def bright(rel, t):
    p = subprocess.run([f"{FF}/ffmpeg", "-nostdin", "-v", "error", "-ss", f"{t:.2f}", "-i", f"{RAW}/{rel}", "-frames:v", "1",
                        "-vf", "scale=80:60,format=gray", "-f", "rawvideo", "pipe:1"], capture_output=True)
    a = np.frombuffer(p.stdout, np.uint8)
    return float(a.mean()) if a.size else None

def run(target, before, after, step, thr, rel):
    x = open(f"{RAW}/{rel}"[:-4] + ".XML", encoding="latin1").read()
    u = int(re.search(r"<DSI_utc_start_time>(\d+)<", x).group(1)); s = int(re.search(r"<DSI_sync_offset>(\d+)<", x).group(1)) / 1e7
    st = dt.datetime.fromtimestamp(u + s, dt.timezone.utc).astimezone(CT).replace(tzinfo=None)
    hh, mm, ss = map(int, target.split(":"))
    tgt = st.replace(hour=hh, minute=mm, second=ss, microsecond=0)
    if tgt < st: tgt += dt.timedelta(days=1)
    p0 = (tgt - st).total_seconds()
    ts = np.arange(max(0, p0 - before), p0 + after, step)
    with ThreadPoolExecutor(8) as ex: b = list(ex.map(lambda t: bright(rel, t), ts))
    steps = []
    for i in range(1, len(ts)):
        if b[i] is not None and b[i - 1] is not None and abs(b[i] - b[i - 1]) > thr:
            steps.append(((st + dt.timedelta(seconds=float(ts[i - 1]))).strftime("%H:%M:%S.%f")[:-4],
                          (st + dt.timedelta(seconds=float(ts[i]))).strftime("%H:%M:%S.%f")[:-4], round(b[i - 1], 1), round(b[i], 1), float(ts[i])))
    return dict(rel=rel, start_local=st.isoformat(), steps=steps, base=round(np.median([v for v in b if v is not None]), 1))

if __name__ == "__main__":
    target, before, after, step, thr = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
    for rel in sys.argv[6:]:
        print(json.dumps(run(target, before, after, step, thr, rel)))
