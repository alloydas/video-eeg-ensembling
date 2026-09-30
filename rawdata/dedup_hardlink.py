#!/usr/bin/env python3
"""Replace byte-identical duplicate camera files in the raw upload with hard links.

Each camera shows two animals, and the upload stores every camera file under both animals'
folders (raw_audit/duplicates.csv: 2,472 mp4 + 2,472 XML groups of exactly 2 copies). This
keeps both paths and stores the bytes once.

Per group, in order, and only for groups the audit marked identical with exactly 2 copies:
  1. both paths exist under RAW_ROOT, on one filesystem, with the audited size;
  2. skip if they already share an inode (resumable);
  3. full byte-for-byte comparison of the two files (sha256 of each is logged);
  4. os.link(keep, dup + '.dedup_tmp'), then os.replace() it over dup (atomic rename);
  5. check that dup now has keep's inode and size.
Any failed check leaves both files untouched and is logged as an error. `keep` is the first
path in the audit's order. One JSON line per group goes to <out>/log_<task>.jsonl, with dup's
original inode, mtime and mode.

Usage (stdlib only):
  python rawdata/dedup_hardlink.py --dry_run [--limit N]        # compare only, change nothing
  python rawdata/dedup_hardlink.py --task I --ntasks N           # link groups with index % N == I
  python rawdata/dedup_hardlink.py --summary                     # totals over all logs
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import sys
import time

RAW_ROOT = os.path.realpath(os.environ.get("RAW_ROOT", "/work/mech-ai/alloydas/EEG/Data"))
EEG_ROOT = os.environ.get("EEG_ROOT", "/work/mech-ai-scratch/alloy/EEG")
DUPS = os.path.join(EEG_ROOT, "output/raw_audit/duplicates.csv")
OUT = os.path.join(EEG_ROOT, "output/raw_audit/dedup")
CHUNK = 8 << 20


def under_raw(p):
    rp = os.path.realpath(p)
    return rp == RAW_ROOT or rp.startswith(RAW_ROOT + os.sep)


def compare(a, b):
    """True iff the files are byte-identical; also returns both sha256 digests."""
    ha, hb = hashlib.sha256(), hashlib.sha256()
    same = True
    with open(a, "rb") as fa, open(b, "rb") as fb:
        while True:
            x, y = fa.read(CHUNK), fb.read(CHUNK)
            ha.update(x)
            hb.update(y)
            if x != y:
                same = False
            if not x and not y:
                break
    return same, ha.hexdigest(), hb.hexdigest()


def groups():
    with open(DUPS, newline="") as f:
        for r in csv.DictReader(f):
            yield int(r["group"]), r


def process(r, dry):
    paths = [os.path.join(RAW_ROOT, p) for p in r["paths"].split("|")]
    rec = dict(group=int(r["group"]), kind=r["kind"], name=r["name"], size=int(r["size"]), paths=paths)
    if r["identical"] != "True" or int(r["n_copies"]) != 2 or len(paths) != 2:
        rec["action"] = "skip_not_eligible"
        return rec
    keep, dup = paths
    if not (under_raw(keep) and under_raw(dup)):
        rec["action"] = "error_outside_raw_root"
        return rec
    try:
        sk, sd = os.stat(keep), os.stat(dup)
    except FileNotFoundError as e:
        rec["action"] = f"error_missing: {e}"
        return rec
    rec.update(dup_ino=sd.st_ino, dup_mtime=sd.st_mtime, dup_mode=oct(sd.st_mode), keep_ino=sk.st_ino)
    if sk.st_ino == sd.st_ino and sk.st_dev == sd.st_dev:
        rec["action"] = "already_linked"
        return rec
    if sk.st_dev != sd.st_dev:
        rec["action"] = "error_different_filesystem"
        return rec
    if not (sk.st_size == sd.st_size == rec["size"]):
        rec["action"] = f"error_size keep={sk.st_size} dup={sd.st_size} audited={rec['size']}"
        return rec
    t0 = time.time()
    same, h_keep, h_dup = compare(keep, dup)
    rec.update(sha256_keep=h_keep, sha256_dup=h_dup, compare_s=round(time.time() - t0, 1))
    if not same or h_keep != h_dup:
        rec["action"] = "error_content_differs"
        return rec
    if dry:
        rec["action"] = "would_link"
        return rec
    tmp = dup + ".dedup_tmp"
    if os.path.lexists(tmp):
        os.unlink(tmp)
    os.link(keep, tmp)
    os.replace(tmp, dup)
    s2 = os.stat(dup)
    if s2.st_ino != sk.st_ino or s2.st_size != sk.st_size:
        rec["action"] = "error_post_check"
        return rec
    rec["action"] = "linked"
    rec["freed_bytes"] = rec["size"]
    return rec


def summary():
    tot = {}
    freed = 0
    for fn in sorted(glob.glob(os.path.join(OUT, "log_*.jsonl"))):
        last = {}
        for line in open(fn):
            r = json.loads(line)
            last[r["group"]] = r
        for r in last.values():
            a = r["action"].split(":")[0].split(" ")[0]
            tot[a] = tot.get(a, 0) + 1
            freed += r.get("freed_bytes", 0)
    print(json.dumps(dict(actions=tot, freed_GB=round(freed / 1e9, 1)), indent=1))
    return 0 if not any(k.startswith("error") for k in tot) else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--ntasks", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--kind", choices=["video", "xml", "all"], default="all")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        return summary()
    os.makedirs(OUT, exist_ok=True)
    tag = ("dry_" if a.dry_run else "") + f"{a.task:02d}"
    log = open(os.path.join(OUT, f"log_{tag}.jsonl"), "a")
    n = 0
    for gi, r in groups():
        if gi % a.ntasks != a.task or (a.kind != "all" and r["kind"] != a.kind):
            continue
        rec = process(r, a.dry_run)
        rec["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"[{rec['time']}] group {gi} {rec['kind']} {rec['name']}: {rec['action']}", flush=True)
        n += 1
        if a.limit and n >= a.limit:
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
