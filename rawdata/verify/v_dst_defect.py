# Independent: count clips affected by DST fall-back (video started CDT, label after change) and by timing-defect files.
import os, re, json, datetime as dt, collections
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor
DF = "/work/mech-ai-scratch/alloy/EEG/data_full"; RAW = "/work/mech-ai/alloydas/EEG/Data"
OUTD = "/work/mech-ai-scratch/alloy/EEG/output/raw_audit/verify"
CT = ZoneInfo("America/Chicago"); UTC = dt.timezone.utc
cd = json.load(open(OUTD + "/clipdirs.json"))
data = set(tuple(x) for x in cd["/work/mech-ai-scratch/alloy/EEG/data"])
full = cd[DF]
tim = {os.path.basename(json.loads(l)['rel']): json.loads(l) for l in open(OUTD + "/timing.jsonl")}
want = [x for x in full if x[1] in ('11-03-2023 to 11-05-2023', '11-2-2023', '12-05-2023', '10-13-2023', '10-12-2023(2)', '10-13-2023-10-14-2023')]
def rd(x):
    ad, day, c = x; p = f"{DF}/{ad}/{day}/{c}/info.txt"
    try:
        d = {}
        for line in open(p):
            if ':' in line: k, v = line.split(':', 1); d[k.strip()] = v.strip()
        return x, d
    except FileNotFoundError:
        return x, None
with ThreadPoolExecutor(32) as ex: res = list(ex.map(rd, want))
print("clips read", len(res), "no info", sum(1 for _, d in res if d is None))
FALL = dt.datetime(2023, 11, 5, 7, 0, tzinfo=UTC)
dst = collections.Counter(); dst_amb = []; dst_ok = collections.Counter(); defect = collections.Counter(); defect_rows = []
FLAT = {'20231103122100': (dt.datetime(2023, 11, 3, 3, 36, 6), dt.datetime(2023, 11, 3, 3, 52, 23)),
        '20231205222300': (dt.datetime(2023, 12, 5, 13, 37, 13), dt.datetime(2023, 12, 5, 13, 54, 3))}
for (ad, day, c), d in res:
    if d is None: continue
    a = ad[5:-8]; indisc = (ad, day, c) in data
    v = d.get('Video file'); cs = dt.datetime.fromisoformat(d['Clip start'])
    o = tim.get(v)
    if o is None: continue
    u0 = dt.datetime.fromtimestamp(int(o['DSI_utc_start_time']), UTC)
    if day == '11-03-2023 to 11-05-2023' and indisc:
        if u0 < FALL:
            if dt.datetime(2023, 11, 5, 1) <= cs < dt.datetime(2023, 11, 5, 2): dst_amb.append((a, c))
            elif cs >= dt.datetime(2023, 11, 5, 2): dst[a] += 1
        elif u0 >= FALL:
            dst_ok[a] += 1
    stamp = v.split('.')[1]
    if stamp in FLAT and indisc:
        f0, f1 = FLAT[stamp]
        ce = cs + dt.timedelta(seconds=float(d.get('Clip duration (s)') or 0) if d.get('Clip duration (s)') else 0)
        k = 'before' if cs < f0 else ('after' if cs >= f1 else 'in_flat')
        defect[(stamp, k)] += 1; defect_rows.append((a, c, stamp, k))
    if stamp == '20231014004100' and indisc:
        defect[(stamp, a)] += 1; defect_rows.append((a, c, stamp, ''))
print("DST: video started before fall-back, clip start >= 02:00 11-05 (naive):", sum(dst.values()), dict(dst))
print("DST: clip start in repeated hour 01:00-02:00:", len(dst_amb), dst_amb)
print("DST: clips in the folder whose video started after fall-back:", sum(dst_ok.values()))
print("defect:", dict(defect))
json.dump(defect_rows, open(OUTD + "/defect_clips.json", "w"))
