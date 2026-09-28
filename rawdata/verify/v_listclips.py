# List clip dir names under data_full and data (names only; read-only).
import os, json
from concurrent.futures import ThreadPoolExecutor
OUTD = "/work/mech-ai-scratch/alloy/EEG/output/raw_audit/verify"
out = {}
for root in ("/work/mech-ai-scratch/alloy/EEG/data_full", "/work/mech-ai-scratch/alloy/EEG/data"):
    tasks = []
    for ad in sorted(os.listdir(root)):
        if not ad.startswith('Data_'): continue
        for day in sorted(os.listdir(os.path.join(root, ad))):
            if os.path.isdir(os.path.join(root, ad, day)): tasks.append((ad, day))
    def ls(t):
        ad, day = t
        return [(ad, day, c) for c in os.listdir(os.path.join(root, ad, day))]
    with ThreadPoolExecutor(32) as ex: res = [x for r in ex.map(ls, tasks) for x in r]
    out[root] = res
    print(root, len(tasks), len(res))
json.dump(out, open(OUTD + "/clipdirs.json", "w"))
