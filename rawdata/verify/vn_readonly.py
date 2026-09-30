#!/usr/bin/env python3
"""Priority 1: nothing read-only changed.

- raw tree: every file recorded by the 2026-09-28 audit (files.csv) still has its recorded size and mtime; no entry
  of the tree has mtime or ctime after the audit's upload window; no file was added or removed.
- data/, data_full/, cache_frames/: no entry with mtime or ctime after T0 (the step-2 commit, before any
  ttg_newclips file exists); entry counts reported.
- output/: every directory except ttg_newclips: entries with mtime/ctime after T0 (listed).
- none of the 505 new clip dirs exists under data/ or data_full/ (nothing was merged).
Writes verify/vn_readonly.json.
"""
import datetime as dt
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vn_common as C  # noqa: E402

T0 = dt.datetime(2026, 9, 29, 23, 23, 26).timestamp()   # step-2 commit d87bc43; first ttg_newclips entry 23:44:27


def scan(root, prune=None):
    n, late, newest_m, newest_c = 0, [], 0.0, 0.0
    for dp, dns, fns in os.walk(root):
        if prune:
            dns[:] = [d for d in dns if os.path.join(dp, d) not in prune]
        for name in dns + fns:
            p = os.path.join(dp, name)
            try:
                st = os.lstat(p)
            except FileNotFoundError:
                continue
            n += 1
            newest_m, newest_c = max(newest_m, st.st_mtime), max(newest_c, st.st_ctime)
            if st.st_mtime > T0 or st.st_ctime > T0:
                late.append((p, dt.datetime.fromtimestamp(st.st_mtime).isoformat(),
                             dt.datetime.fromtimestamp(st.st_ctime).isoformat()))
    st = os.lstat(root)
    return dict(root=root, entries=n, changed_after_T0=len(late), changed=late[:50],
                newest_mtime=dt.datetime.fromtimestamp(newest_m).isoformat(),
                newest_ctime=dt.datetime.fromtimestamp(newest_c).isoformat(),
                root_mtime=dt.datetime.fromtimestamp(st.st_mtime).isoformat())


def main():
    out = {"T0": dt.datetime.fromtimestamp(T0).isoformat()}
    # raw tree against the audit
    F = C.files_csv()
    size_bad, mtime_bad, missing = [], [], []
    for rel, r in F.items():
        p = os.path.join(C.RAW, rel)
        if not os.path.exists(p):
            missing.append(rel); continue
        st = os.stat(p)
        if str(st.st_size) != r["size"]:
            size_bad.append((rel, r["size"], st.st_size))
        rec = C.parse_dt(r["mtime"])
        if rec is None or abs(dt.datetime.fromtimestamp(st.st_mtime) - rec).total_seconds() > 1e-3:
            mtime_bad.append((rel, r["mtime"], dt.datetime.fromtimestamp(st.st_mtime).isoformat()))
    now_files = set()
    for dp, dns, fns in os.walk(C.RAW):
        for f in fns:
            now_files.add(os.path.relpath(os.path.join(dp, f), C.RAW))
    added = sorted(now_files - set(F))
    out["raw_vs_audit"] = dict(audited_files=len(F), now_files=len(now_files), missing=missing[:20],
                               n_missing=len(missing), added=added[:20], n_added=len(added),
                               size_changed=size_bad[:20], n_size_changed=len(size_bad),
                               mtime_changed=mtime_bad[:20], n_mtime_changed=len(mtime_bad))
    out["raw_scan"] = scan(C.RAW)
    for name in ("data", "data_full", "cache_frames"):
        out[name] = scan(os.path.join(C.EEG_ROOT, name))
    # output/: every existing directory except ttg_newclips
    od = os.path.join(C.EEG_ROOT, "output")
    outs = {}
    for d in sorted(os.listdir(od)):
        p = os.path.join(od, d)
        if d == "ttg_newclips":
            continue
        if os.path.isdir(p):
            s = scan(p)
        else:
            st = os.lstat(p)
            s = dict(entries=1, changed_after_T0=int(st.st_mtime > T0 or st.st_ctime > T0), changed=[])
        if s["changed_after_T0"]:
            outs[d] = s
    out["output_dirs_changed_after_T0"] = outs
    # the rest of the EEG checkout (code etc.), excluding output/ and the big trees already scanned
    prune = {os.path.join(C.EEG_ROOT, x) for x in ("output", "data", "data_full", "cache_frames", ".git")}
    out["eeg_repo_other"] = scan(C.EEG_ROOT, prune=prune)
    # nothing merged: no new clip dir exists in data/ or data_full/
    M = [r for r in C.manifest() if r["status"] != "excluded"]
    merged = []
    for r in M:
        rel = f"Data_{r['animal']}_cropped/{r['session']}/{r['clip_dir']}"
        for root in (C.DATA, C.FULL):
            if os.path.exists(os.path.join(root, rel)):
                merged.append(os.path.join(root, rel))
    out["new_clips_present_in_data_or_data_full"] = merged
    out["n_new_clips_checked"] = len(M)
    ok = (not missing and not added and not size_bad and not mtime_bad and out["raw_scan"]["changed_after_T0"] == 0
          and all(out[n]["changed_after_T0"] == 0 for n in ("data", "data_full", "cache_frames")) and not merged)
    out["ok_readonly_trees"] = ok
    C.write_json("vn_readonly.json", out)
    for k, v in out.items():
        if isinstance(v, dict):
            print(k, {kk: vv for kk, vv in v.items() if kk not in ("changed", "size_changed", "mtime_changed")})
        else:
            print(k, v)


if __name__ == "__main__":
    main()
