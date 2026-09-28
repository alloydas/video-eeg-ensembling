# Independent: do any non-seizure clips (discover set) overlap an annotated xlsx event of the same animal? (read-only)
import os, json, datetime as dt, bisect, collections
from concurrent.futures import ThreadPoolExecutor
OUTD = "/work/mech-ai-scratch/alloy/EEG/output/raw_audit/verify"; DF = "/work/mech-ai-scratch/alloy/EEG/data_full"
cd = json.load(open(OUTD + "/clipdirs.json"))
ns = [x for x in cd["/work/mech-ai-scratch/alloy/EEG/data"] if x[2].startswith('clip_')]
def rd(x):
    a, d, c = x
    try:
        out = {}
        for line in open(f"{DF}/{a}/{d}/{c}/info.txt"):
            if ':' in line:
                k, v = line.split(':', 1); out[k.strip()] = v.strip()
        return x, out
    except FileNotFoundError:
        return x, None
with ThreadPoolExecutor(48) as ex: res = list(ex.map(rd, ns))
xl = json.load(open(OUTD + "/xlsx_events.json"))
ev = collections.defaultdict(list)
for x in xl:
    a = x['rel'].split('/')[0].replace('DONE_', '')
    for e in x['events']:
        ev[a].append((dt.datetime.fromisoformat(e['start']), dt.datetime.fromisoformat(e['end']), e['label'], x['rel']))
for a in ev: ev[a].sort()
noinfo = 0; hits = []; margin_viol = []
for (a, d, c), inf in res:
    if inf is None or 'Clip start' not in inf: noinfo += 1; continue
    an = a[5:-8]; s = dt.datetime.fromisoformat(inf['Clip start']); e = dt.datetime.fromisoformat(inf['Clip end'])
    for es, ee, lab, rel in ev[an]:
        if es < e and ee > s: hits.append((an, d, c, lab, es.isoformat(), ee.isoformat(), rel))
        # 30 s safety margin check
        elif es < e + dt.timedelta(seconds=30) and ee > s - dt.timedelta(seconds=30): margin_viol.append((an, d, c, lab, es.isoformat(), rel))
print("nonseizure clips", len(ns), "no info", noinfo, "overlapping an annotated event:", len(hits))
for h in hits: print("  ", h)
print("within 30 s of an event (not overlapping):", len(margin_viol)); [print("  ", m) for m in margin_viol[:10]]
