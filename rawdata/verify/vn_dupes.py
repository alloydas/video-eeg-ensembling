#!/usr/bin/env python3
"""Is any new seizure already represented elsewhere? For every new seizure clip:
  - existing seizure clips of the animal (any session, data_full info.txt) whose labelled span overlaps it, or whose
    onset is within 120 s
  - annotated events of the animal in OTHER xlsx files (audit events.csv) overlapping it or within 120 s
  - other new seizure clips of the animal overlapping it
Writes verify/vn_dupes.json."""
import glob
import json
import os
import sys
from collections import defaultdict

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vn_common as C  # noqa: E402

NEAR = 120.0


def main():
    M = [r for r in C.manifest() if r["status"] != "excluded" and r["kind"] == "seizure"]
    animals = sorted({r["animal"] for r in M})
    ex = defaultdict(list)
    for a in animals:
        for p in glob.glob(os.path.join(glob.escape(os.path.join(C.FULL, f"Data_{a}_cropped")), "*", "seizure_*", "info.txt")):
            i = C.parse_info(p)[0]
            s, e = C.parse_dt(i.get("Seizure start")), C.parse_dt(i.get("Seizure end"))
            if s and e:
                ex[a].append((s, e, os.path.relpath(os.path.dirname(p), C.FULL)))
    ev = defaultdict(list)
    for e in C.events_csv():
        s, en = C.parse_dt(e["start"]), C.parse_dt(e["end"])
        if s:
            ev[e["animal"]].append((s, en or s, e["xlsx"], e["row"], e["label"]))
    hits_ex, hits_ev, hits_new = [], [], []
    for r in M:
        a = r["animal"]
        s, e = C.parse_dt(r["onset"]), C.parse_dt(r["seizure_end"])
        for s2, e2, rel in ex[a]:
            if (s < e2 and e > s2) or abs((s - s2).total_seconds()) < NEAR:
                hits_ex.append((r["clip_dir"], rel, (s2 - s).total_seconds()))
        for s2, e2, x, row, lab in ev[a]:
            if x == r["xlsx"] and s2 == s:
                continue
            if (s < e2 and e > s2) or abs((s - s2).total_seconds()) < NEAR:
                hits_ev.append((r["clip_dir"], f"{x} row {row} {lab}", round((s2 - s).total_seconds(), 1)))
        for q in M:
            if q is r or q["animal"] != a:
                continue
            s2, e2 = C.parse_dt(q["onset"]), C.parse_dt(q["seizure_end"])
            if s < e2 and e > s2:
                hits_new.append((r["clip_dir"], q["clip_dir"]))
    out = dict(n=len(M), existing_seizure_clip_overlap_or_near=hits_ex, other_xlsx_event_overlap_or_near=hits_ev,
               new_new_span_overlap=hits_new, n_existing_seizure_clips_scanned={a: len(v) for a, v in ex.items()})
    C.write_json("vn_dupes.json", out)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
