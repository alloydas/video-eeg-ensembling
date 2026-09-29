# Independent read-only check for the ttg_recut step: nothing under the protected paths changed.
# Reads only; writes one JSON under $EEG_ROOT/output/ttg_recut/verify/indep/.
#  - raw recordings: every file listed in raw_audit/files.csv still has its audited size and mtime;
#    the raw videos / EDFs the re-cut used also keep their audited sha1 of the first and last MiB
#  - data_full clips: every clip in align/clips.csv keeps its audited video byte size
#  - data/ placeholders, data_full, cache_frames, output/<everything but ttg_recut>: no mtime or ctime after T0
import os, sys, csv, json, hashlib, datetime as dt
csv.field_size_limit(1 << 30)
RAW = "/work/mech-ai/alloydas/EEG/Data"
EEG = "/work/mech-ai-scratch/alloy/EEG"
AUD = f"{EEG}/output/raw_audit"
OUT = f"{EEG}/output/ttg_recut/verify/indep"
T0 = dt.datetime(2026, 9, 29, 0, 0).timestamp()   # the re-cut work began 2026-09-29 08:53
MIB = 1 << 20


def sha1_ht(p, size):
    with open(p, "rb") as f:
        h = f.read(MIB)
        if size > MIB:
            f.seek(size - MIB); t = f.read(MIB)
        else:
            t = h
    return hashlib.sha1(h).hexdigest(), hashlib.sha1(t).hexdigest()


def main():
    res = {}
    rows = list(csv.DictReader(open(f"{AUD}/files.csv")))
    n = bad_size = bad_mtime = missing = 0; ex = []
    by_rel = {}
    for r in rows:
        rel = r["relpath"]; p = os.path.join(RAW, rel); by_rel[rel] = r
        n += 1
        try:
            st = os.stat(p)
        except FileNotFoundError:
            missing += 1; ex.append(("missing", rel)); continue
        if str(st.st_size) != r["size"]:
            bad_size += 1; ex.append(("size", rel, st.st_size, r["size"]))
        mt = dt.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S.%f")
        if r["mtime"] and mt[:19] != r["mtime"][:19]:
            bad_mtime += 1; ex.append(("mtime", rel, mt, r["mtime"]))
        if st.st_mtime >= T0 or st.st_ctime >= T0:
            ex.append(("changed_after_T0", rel))
    res["raw_files"] = dict(n=n, missing=missing, size_mismatch=bad_size, mtime_mismatch=bad_mtime, examples=ex[:20])

    # sha1 head/tail of the raw files the re-cut read
    man = list(csv.DictReader(open(f"{EEG}/output/ttg_recut/manifest.csv")))
    used = set()
    for m in man:
        used.add(m["old_raw_video"]); used.add(m["new_raw_video"])
    clips = {}
    for r in csv.DictReader(open(f"{AUD}/align/clips.csv")):
        clips[(r["animal"], r["session"], r["clip_dir"])] = r
    for m in man:
        c = clips[(m["animal"], m["session"], m["clip_dir"])]
        if c.get("raw_edf"):
            used.add(c["raw_edf"])
    hash_ok = hash_bad = hash_na = 0; hb = []
    for rel in sorted(used):
        r = by_rel.get(rel)
        if r is None or not r.get("hash_head"):
            hash_na += 1; hb.append(("no audited hash", rel)); continue
        h, t = sha1_ht(os.path.join(RAW, rel), int(r["size"]))
        if (h, t) == (r["hash_head"], r["hash_tail"]):
            hash_ok += 1
        else:
            hash_bad += 1; hb.append(("hash differs", rel))
    res["raw_used_sha1_head_tail"] = dict(n=len(used), ok=hash_ok, differs=hash_bad, no_audit_hash=hash_na, examples=hb[:20])

    # data_full clip sizes vs the audit
    nf = okf = badf = 0; bf = []
    for (a, s, d), r in clips.items():
        if r.get("in_full") != "True" or not r.get("full_video_bytes"):
            continue
        p = f"{EEG}/data_full/Data_{a}_cropped/{s}/{d}/video.mp4"
        nf += 1
        try:
            sz = os.path.getsize(p)
        except OSError:
            badf += 1; bf.append(("missing", p)); continue
        if str(sz) == r["full_video_bytes"].split(".")[0]:
            okf += 1
        else:
            badf += 1; bf.append(("size", p, sz, r["full_video_bytes"]))
    res["data_full_video_sizes_vs_audit"] = dict(n=nf, ok=okf, bad=badf, examples=bf[:20])

    # no mtime/ctime after T0 anywhere under the protected trees
    trees = [f"{EEG}/data", f"{EEG}/data_full", f"{EEG}/cache_frames"]
    trees += [os.path.join(f"{EEG}/output", d) for d in sorted(os.listdir(f"{EEG}/output")) if d != "ttg_recut"]
    newer = []; counted = 0
    for t in trees:
        if os.path.isfile(t):
            st = os.stat(t); counted += 1
            if st.st_mtime >= T0 or st.st_ctime >= T0:
                newer.append(t)
            continue
        for root, dirs, files in os.walk(t):
            for x in dirs + files:
                p = os.path.join(root, x)
                try:
                    st = os.lstat(p)
                except OSError:
                    continue
                counted += 1
                if st.st_mtime >= T0 or st.st_ctime >= T0:
                    newer.append(p)
    res["protected_trees_changed_after_T0"] = dict(T0="2026-09-29 00:00 local", entries_checked=counted,
                                                   n_newer=len(newer), examples=newer[:30])
    # the stored video dumps the patch read: sha1 of the whole file, for the record
    vs = {}
    for d in sorted(os.listdir(f"{EEG}/output/ttg_vsubj")):
        p = f"{EEG}/output/ttg_vsubj/{d}/val_ep12.npz"
        if os.path.isfile(p):
            vs[d] = hashlib.sha1(open(p, "rb").read()).hexdigest()
    res["ttg_vsubj_val_ep12_sha1"] = vs
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/readonly.json", "w"), indent=1)
    print(json.dumps({k: (v if k != "ttg_vsubj_val_ep12_sha1" else len(v)) for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
