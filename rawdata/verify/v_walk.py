# Independent verification: list raw tree files with size (read-only; stat only).
import os, sys, json
from concurrent.futures import ThreadPoolExecutor
RAW = "/work/mech-ai/alloydas/EEG/Data"
OUT = "/work/mech-ai-scratch/alloy/EEG/output/raw_audit/verify/walk.jsonl"
assert OUT.startswith("/work/mech-ai-scratch/alloy/EEG/output/raw_audit/")
def walk_animal(a):
    rows = []
    for root, dirs, files in os.walk(os.path.join(RAW, a)):
        for f in files:
            p = os.path.join(root, f)
            st = os.lstat(p)
            rows.append({"rel": os.path.relpath(p, RAW), "size": st.st_size, "mtime": st.st_mtime})
    return rows
animals = sorted(os.listdir(RAW))
with ThreadPoolExecutor(23) as ex:
    res = list(ex.map(walk_animal, animals))
with open(OUT, "w") as fo:
    for rows in res:
        for r in rows:
            fo.write(json.dumps(r) + "\n")
print(sum(len(r) for r in res))
