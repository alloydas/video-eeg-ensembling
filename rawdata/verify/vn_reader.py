#!/usr/bin/env python3
"""Priority 5 (downstream reader) and part of 7 (EEG window counts), in the eeg env.

Runs the parent's own build_stage_segments_pooled.one() (imported read-only from $EEG_ROOT, bytecode off) on every
new clip, exactly as `--root data_full` would after the merge, and on existing clips of the same sessions:
  - MNE opens every new eeg.edf; the channel the builder picks (first EEG, else first ECG) is the same channel index
    as in the existing clips of that session, and its samples equal the raw EDF channel read by vn_common.EDF
  - the window count and window labels per class (to compare with integration.txt: +12,518 windows)
Writes verify/vn_reader.json and vn_reader.csv.
"""
import glob
import os
import re
import sys
import time
import warnings
from collections import Counter

sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vn_common as C  # noqa: E402

t0 = time.time()
sys.path.insert(0, C.EEG_ROOT)
import numpy as np  # noqa: E402
import mne  # noqa: E402
import build_stage_segments_pooled as B  # noqa: E402

mne.set_log_level("ERROR")
print(f"imports {time.time() - t0:.0f} s", flush=True)


def pick_info(edf):
    raw = mne.io.read_raw_edf(edf, preload=True, verbose="ERROR")
    eeg = [c for c in raw.ch_names if "EEG" in c.upper()]
    ecg = [c for c in raw.ch_names if "ECG" in c.upper()]
    chans = eeg or ecg
    k = raw.ch_names.index(chans[0])
    return raw.ch_names, k, raw.get_data(picks=[k])[0], raw.info["sfreq"]


def main():
    M = [r for r in C.manifest() if r["status"] != "excluded"]
    rows = []
    wl = Counter()
    per_kind = Counter()
    for i, r in enumerate(M):
        d = os.path.join(C.NEWFULL, f"Data_{r['animal']}_cropped", r["session"], r["clip_dir"])
        b = os.path.basename(d)
        if b.startswith("seizure_"):
            y = B.STAGE[re.search(r"Stage_[0-9]+", b).group()]
        else:
            y = 0
        res = B.one((d, y, f"{r['animal']}/{r['session']}"))
        names, k, sig, sf = pick_info(os.path.join(d, "eeg.edf"))
        E = C.EDF(os.path.join(d, "eeg.edf"))
        mine, _ = E.read_all_phys(C.eeg_channel(E.labels))
        rec = dict(key=r["key"], y=y, ok=res is not None, n_win=len(res[1]) if res is not None else 0,
                   labels=dict(Counter(res[1].tolist())) if res is not None else {}, mne_names="|".join(names),
                   pick_idx=k, pick_name=names[k], own_pick_idx=C.eeg_channel(E.labels),
                   max_abs_vs_own=float(np.abs(sig - mine).max()), sfreq=sf)
        if res is not None:
            wl.update(res[1].tolist())
        per_kind[(r["kind"], rec["ok"])] += 1
        rows.append(rec)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(M)} {time.time() - t0:.0f} s", flush=True)
    # existing clips of the same sessions: pick index
    ex = []
    for a, s in sorted({(r["animal"], r["session"]) for r in M}):
        dirs = sorted(glob.glob(os.path.join(glob.escape(os.path.join(C.FULL, f"Data_{a}_cropped", s)), "*", "eeg.edf")))
        for e in dirs[:: max(1, len(dirs) // 4)][:4]:
            names, k, sig, sf = pick_info(e)
            ex.append(dict(key=os.path.relpath(os.path.dirname(e), C.FULL), mne_names="|".join(names), pick_idx=k,
                           pick_name=names[k], sfreq=sf))
    ex_pick = {}
    for x in ex:
        ex_pick.setdefault(x["key"].split("/")[0] + "/" + x["key"].split("/")[1], set()).add((x["pick_idx"], x["pick_name"]))
    mism = []
    for r_ in rows:
        a_s = "/".join(r_["key"].split("/")[1:3])
        ep = ex_pick.get(a_s, set())
        if ep and r_["pick_idx"] not in {p[0] for p in ep}:
            mism.append((r_["key"], r_["pick_idx"], sorted(ep)))
    out = dict(n=len(rows), read_ok=Counter(x["ok"] for x in rows), per_kind={f"{k[0]}|{k[1]}": v for k, v in per_kind.items()},
               windows_total=int(sum(wl.values())), windows_by_label={int(k): v for k, v in sorted(wl.items())},
               pick_names_new=Counter(x["pick_name"] for x in rows),
               mne_names_new=Counter(x["mne_names"] for x in rows),
               pick_names_existing=Counter(x["pick_name"] for x in ex),
               mne_names_existing=Counter(x["mne_names"] for x in ex),
               pick_index_mismatch_vs_existing=mism,
               max_abs_mne_vs_own=max(x["max_abs_vs_own"] for x in rows),
               seizure_clips_without_positive_window=[x["key"] for x in rows if x["y"] > 0 and not any(
                   int(k) > 0 for k in x["labels"])],
               secs=time.time() - t0)
    C.write_csv("vn_reader.csv", rows)
    C.write_json("vn_reader.json", out)
    import json
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
