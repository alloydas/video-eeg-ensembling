# Independent re-count of xlsx annotations vs labelled clip dirs (read-only).
import os, re, json, zipfile, datetime as dt, collections, csv
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor
RAW = "/work/mech-ai/alloydas/EEG/Data"; DF = "/work/mech-ai-scratch/alloy/EEG/data_full"
OUTD = "/work/mech-ai-scratch/alloy/EEG/output/raw_audit/verify"
NS = {'x': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
CT = ZoneInfo("America/Chicago"); UTC = dt.timezone.utc
EPOCH = dt.datetime(1899, 12, 30)

def col(ref):
    return re.match(r'[A-Z]+', ref).group(0)

def read_xlsx(path):
    z = zipfile.ZipFile(path)
    names = z.namelist()
    sst = []
    if 'xl/sharedStrings.xml' in names:
        for si in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('x:si', NS):
            sst.append(''.join(t.text or '' for t in si.iter('{%s}t' % NS['x'])))
    sheets = sorted(n for n in names if n.startswith('xl/worksheets/sheet'))
    rows = []
    for sh in sheets:
        root = ET.fromstring(z.read(sh))
        for r in root.iter('{%s}row' % NS['x']):
            d = {}
            for c in r.findall('x:c', NS):
                v = c.find('x:v', NS)
                if v is None: continue
                val = sst[int(v.text)] if c.get('t') == 's' else v.text
                d[col(c.get('r'))] = val
            rows.append((sh, int(r.get('r')), d))
    return len(sheets), rows

def parse(rel):
    n_sheets, rows = read_xlsx(os.path.join(RAW, rel))
    ev = []
    for sh, rn, d in rows:
        if rn <= 2: continue
        if not d.get('A'): continue
        st = EPOCH + dt.timedelta(days=float(d['C'])); en = EPOCH + dt.timedelta(days=float(d['D']))
        ev.append(dict(xlsx=rel, sheet=sh, row=rn, idx=rn - 2, label=d['A'], dur=float(d['B']),
                       start=st.isoformat(), end=en.isoformat(), spikes=d.get('E'), eegave=d.get('F')))
    hdr = [d for sh, rn, d in rows if rn == 1]
    return rel, n_sheets, hdr[0] if hdr else None, ev

walk = [json.loads(l) for l in open(OUTD + "/walk.jsonl")]
xl = sorted(r['rel'] for r in walk if r['rel'].lower().endswith('.xlsx'))
with ThreadPoolExecutor(16) as ex: res = list(ex.map(parse, xl))
json.dump([dict(rel=r, n_sheets=n, header=h, events=e) for r, n, h, e in res], open(OUTD + "/xlsx_events.json", "w"))
print("xlsx", len(res), "events", sum(len(e) for _, _, _, e in res))
