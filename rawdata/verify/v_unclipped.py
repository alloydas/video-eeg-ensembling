# Independent: match xlsx events to seizure clip dirs; classify unclipped Stage 2-5 events by video/EDF coverage.
import os, re, json, csv, datetime as dt, collections
from zoneinfo import ZoneInfo
OUTD = "/work/mech-ai-scratch/alloy/EEG/output/raw_audit/verify"
CT = ZoneInfo("America/Chicago"); UTC = dt.timezone.utc
xl = json.load(open(OUTD + "/xlsx_events.json"))
cd = json.load(open(OUTD + "/clipdirs.json"))
full = cd["/work/mech-ai-scratch/alloy/EEG/data_full"]; data = set(tuple(x) for x in cd["/work/mech-ai-scratch/alloy/EEG/data"])
# seizure dirs
sz = collections.defaultdict(list)   # (animal, name-time) -> dirs
szdirs = []
for ad, day, c in full:
    m = re.match(r'seizure_(\d+)_(.+)_(\d{8}_\d{6})$', c)
    if m:
        a = ad[5:-8]
        szdirs.append((a, day, c, int(m.group(1)), m.group(2), m.group(3)))
print("seizure dirs full", len(szdirs), "in data", sum(1 for s in szdirs if ('Data_%s_cropped' % s[0], s[1], s[2]) in data))
key = collections.defaultdict(list)
for s in szdirs: key[(s[0], s[5])].append(s)
# events
events = []
for x in xl:
    a = x['rel'].split('/')[0].replace('DONE_', '')
    day = x['rel'].split('/')[1]
    for e in x['events']:
        st = dt.datetime.fromisoformat(e['start']); st = (st + dt.timedelta(microseconds=500)).replace(microsecond=((st + dt.timedelta(microseconds=500)).microsecond // 1000) * 1000)
        # name time: truncated to the second (check both floor and round)
        e2 = dict(e, animal=a, day=day, st=st, en=dt.datetime.fromisoformat(e['end']))
        events.append(e2)
matched = 0; used = set(); exact_idx = 0; label_ok = 0; day_ok = 0
for e in events:
    nt = e['st'].strftime('%Y%m%d_%H%M%S')
    cands = key.get((e['animal'], nt), [])
    e['dirs'] = cands
    if cands:
        matched += 1
        for s in cands:
            used.add(s)
            if s[3] == e['idx'] + 1: exact_idx += 1
            if s[4] == e['label'].replace(' ', '_'): label_ok += 1
            if s[1] == e['day']: day_ok += 1
print("events", len(events), "with a same-second seizure dir", matched, "dirs used", len(used), "of", len(szdirs),
      "idx==row", exact_idx, "label ok", label_ok, "day ok", day_ok)
multi = [e for e in events if len(e['dirs']) > 1]; print("events with >1 dir", len(multi))
unused = [s for s in szdirs if s not in used]; print("seizure dirs without event", len(unused), unused[:5])
# video spans per animal (own folder), local naive
dur = {}
for r in csv.DictReader(open("/work/mech-ai-scratch/alloy/EEG/output/raw_audit/files.csv")):
    if r['kind'] == 'video' and r['stream_duration']: dur[r['relpath']] = float(r['stream_duration'])
vids = collections.defaultdict(list)
for l in open(OUTD + "/timing.jsonl"):
    o = json.loads(l)
    a = o['rel'].split('/')[0].replace('DONE_', '')
    if o['rel'] not in dur: continue
    s = dt.datetime.fromtimestamp(int(o['DSI_utc_start_time']) + int(o['DSI_sync_offset']) / 1e7, UTC).astimezone(CT).replace(tzinfo=None)
    vids[a].append((s, s + dt.timedelta(seconds=dur[o['rel']]), o['rel']))
edf = {}
for l in open(OUTD + "/edfhdr.jsonl"):
    o = json.loads(l)
    if '/seizure_clips/' in o['rel']: continue
    a = o['rel'].split('/')[0].replace('DONE_', ''); day = o['rel'].split('/')[1]
    dd, mm, yy = o['startdate'].split('.'); H, M, S = o['starttime'].split('.')
    s = dt.datetime(2000 + int(yy), int(mm), int(dd), int(H), int(M), int(S))
    n = o['nrec'] if o['nrec'] >= 0 else o['nrec_from_size']
    edf[(a, day)] = (s, s + dt.timedelta(seconds=n * o['recdur']))
LAB = {'RN197','RN199','RN204','RN208','RN210','RN213','RN215','RN216','RN219','RN222','RN223','RN224','RN227','RN229','RN235','RN237','RN238','RN242','RN244','RN245'}
cnt = collections.defaultdict(collections.Counter); lab_by = collections.defaultdict(collections.Counter)
rows = []
for e in events:
    if e['dirs'] or not re.match(r'Stage [2-5]$', e['label']): continue
    inv = any(s <= e['st'] and e['en'] <= en for s, en, _ in vids[e['animal']])
    inv_start = any(s <= e['st'] < en for s, en, _ in vids[e['animal']])
    ed = edf.get((e['animal'], e['day']))
    ine = bool(ed and ed[0] <= e['st'] and e['en'] <= ed[1])
    k = 'vid+edf' if (inv and ine) else ('vid_start_only+edf' if inv_start and ine else ('vid_no_edf' if inv else ('no_vid')))
    cnt[e['animal']][k] += 1
    if k == 'vid+edf': lab_by['vid+edf'][e['label']] += 1
    rows.append((e['animal'], e['day'], e['idx'], e['label'], e['start'], k))
tot = collections.Counter()
for a in sorted(cnt): print(a, dict(cnt[a])); tot.update(cnt[a])
print("TOTAL", dict(tot), "sum", sum(tot.values()))
print("labelled-animal vid+edf", sum(cnt[a]['vid+edf'] for a in cnt if a in LAB), dict(lab_by['vid+edf']))
json.dump(rows, open(OUTD + "/unclipped.json", "w"))
# mislabelled negative check
for ad, day, c in full:
    if ad == 'Data_RN242_cropped' and c.startswith('clip_03_vs_seizure_04_Stage_4_20231012_234446'):
        print("found", day, c)
