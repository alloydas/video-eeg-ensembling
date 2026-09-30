#!/usr/bin/env python3
"""Priority 4: the paired negatives follow the parent rule and contain no annotated event.

1. Own replay of cut_non_seizure_clips.py's sampler (random.seed(42) once per session process, every xlsx row in
   order, window = 10 + duration + 10 s drawn uniformly from the free time of the seizure's own video, falling back to
   the other session videos; free = outside every row of the xlsx widened by pre+safety = 40 s before and post+safety
   = 40 s after). Videos: XML DSI_utc_start_time as Central wall time, duration nb_frames / avg_frame_rate (what cv2
   reports), every *.mp4 in the folder and one level down. Validated on every existing negative with an info.txt in
   the same sessions, then compared with every new negative.
2. Every new negative: distance to every annotated event of the animal in ANY xlsx (audit events.csv, all labels);
   inside-the-window events and gaps under 40 s are failures.
3. Pairing: every new seizure has exactly one negative (new, or an existing dir whose name carries the seizure's own
   index, stage and onset stamp); the manifest's existing_negative agrees.
4. Existing negatives that contain a new seizure's labelled span.
Writes verify/vn_negatives.csv and vn_negatives.json.
"""
import datetime as dt
import glob
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from fractions import Fraction

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vn_common as C  # noqa: E402

TD = dt.timedelta
_V = {}


def vinfo(path):
    if path not in _V:
        x = C.xml_of(path)
        st = C.central_naive(int(x["DSI_utc_start_time"]))
        r = C.run([C.FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
                   "stream=avg_frame_rate,nb_frames", "-of", "json", path])
        s = json.loads(r.stdout)["streams"][0]
        dur = int(s["nb_frames"]) / float(Fraction(s["avg_frame_rate"]))
        _V[path] = dict(start=st, end=st + TD(seconds=dur), dur=dur, fname=os.path.basename(path), path=path)
    return _V[path]


def session_videos(folder):
    fs = sorted(set(glob.glob(os.path.join(glob.escape(folder), "*.mp4")) +
                    glob.glob(os.path.join(glob.escape(folder), "*", "*.mp4"))))
    vs = [vinfo(f) for f in fs if C.xml_of(f)]
    return sorted(vs, key=lambda v: v["start"])


def session_edfs(folder):
    fs = sorted(set(glob.glob(os.path.join(glob.escape(folder), "*.edf")) +
                    glob.glob(os.path.join(glob.escape(folder), "*.EDF")) +
                    glob.glob(os.path.join(glob.escape(folder), "*", "*.edf"))))
    out = []
    for f in fs:
        E = C.EDF(f)
        s = E.start()
        out.append(dict(start=s, end=s + TD(seconds=E.nrec * E.rec_dur), fname=os.path.basename(f)))
    return sorted(out, key=lambda e: e["start"])


def blocked_of(rows):
    b = sorted((r["start"] - TD(seconds=40), r["end_f"] + TD(seconds=40)) for r in rows)
    merged = []
    for s, e in b:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def free_windows(v, blocked, need):
    free, cur = [], v["start"]
    for bs, be in blocked:
        bs, be = max(bs, v["start"]), min(be, v["end"])
        if bs >= v["end"] or be <= v["start"]:
            continue
        if cur < bs:
            free.append((cur, bs))
        cur = max(cur, be)
    if cur < v["end"]:
        free.append((cur, v["end"]))
    return [(s, e) for s, e in free if (e - s).total_seconds() >= need]


def draw(rng, fw, need):
    ws, ss = [], []
    for s, e in fw:
        ue = e - TD(seconds=need)
        if ue > s:
            ws.append((ue - s).total_seconds()); ss.append(s)
    if not ws:
        return None
    r = rng.uniform(0, sum(ws))
    cum = 0.0
    for w, s in zip(ws, ss):
        cum += w
        if r <= cum:
            return s + TD(seconds=rng.uniform(0, w))
    return ss[-1]


def replay(targets, blocked_rows, videos):
    rng = random.Random(42)
    blocked = blocked_of(blocked_rows)
    out = {}
    for n, sz in enumerate(targets, 1):
        need = 10 + sz["duration"] + 10
        src = next((v for v in videos if v["start"] <= sz["start"] < v["end"]), None)
        cands = videos if src is None else [src] + [v for v in videos if v is not src]
        got = None
        for v in cands:
            fw = free_windows(v, blocked, need)
            if not fw:
                continue
            s = draw(rng, fw, need)
            if s is not None:
                got = (s, v)
                break
        out[n] = dict(clip_n=n, sz=sz, start=got[0] if got else None, video=got[1]["fname"] if got else None,
                      src=src["fname"] if src else None, need=need)
    return out


def rows_of(xlsx):
    ev = C.xlsx_events(xlsx)[0]
    for e in ev:
        e["end_f"] = e["end"] if e["end"] else e["start"] + TD(seconds=e["duration"])
    return ev


def main():
    M = C.manifest()
    new_negs = [r for r in M if r["kind"] == "nonseizure"]
    new_sz = [r for r in M if r["kind"] == "seizure" and r["status"] != "excluded"]
    sessions = sorted({(r["animal"], r["session"]) for r in new_negs + new_sz})
    out, rows = {"sessions": {}}, []
    # ---------------------------------------------------------------- 1. replay
    replays = {}
    for a, s in sessions:
        folder = os.path.join(C.RAW, a, s)
        xl = sorted(glob.glob(os.path.join(glob.escape(folder), "*.xlsx")))
        videos = session_videos(folder)
        full = os.path.join(C.FULL, f"Data_{a}_cropped", s)
        ex_negs = {}
        for d in glob.glob(os.path.join(glob.escape(full), "clip_*_vs_seizure_*")):
            p = os.path.join(d, "info.txt")
            if os.path.exists(p):
                ex_negs[os.path.basename(d)] = C.parse_info(p)[0]
        parsed = {x: rows_of(x) for x in xl}
        # the xlsx the parent read: the one whose (clip_n, idx, stamp) reproduce the existing negatives' names
        def score(x):
            names = {f"clip_{k:02d}_vs_seizure_{e['idx']:02d}_{re.sub(r'[^\w]', '_', e['label'])}_"
                     f"{e['start'].strftime('%Y%m%d_%H%M%S')}" for k, e in enumerate(parsed[x], 1)}
            ex_all = {os.path.basename(d) for d in glob.glob(os.path.join(glob.escape(full), "*"))}
            return len(names & ex_all)
        sc = {x: score(x) for x in xl}
        px = max(sc, key=sc.get)
        rp = replay(parsed[px], parsed[px], videos)
        # validation on existing negatives
        dev = []
        for name, info in ex_negs.items():
            cn = int(name.split("_")[1])
            r_ = rp.get(cn)
            if r_ is None or r_["start"] is None:
                dev.append((name, None)); continue
            d_ = abs((r_["start"] - C.parse_dt(info["Clip start"])).total_seconds())
            same_v = r_["video"] == info["Video file"]
            dev.append((name, d_ if same_v else f"video {r_['video']} != {info['Video file']}"))
        num = [d for _, d in dev if isinstance(d, float)]
        out["sessions"][f"{a}/{s}"] = dict(xlsx=[os.path.basename(x) for x in xl], parent_xlsx=os.path.basename(px),
                                           xlsx_scores={os.path.basename(k): v for k, v in sc.items()},
                                           n_videos=len(videos), existing_negatives_with_info=len(ex_negs),
                                           replay_max_abs_dev_s=max(num) if num else None,
                                           replay_n_within_1ms=sum(d <= 1e-3 for d in num),
                                           replay_mismatch=[x for x in dev if not isinstance(x[1], float) or x[1] > 1e-3][:10])
        replays[(a, s)] = (px, parsed, videos, rp)
    # ---------------------------------------------------------------- per new negative
    ev_all = defaultdict(list)
    for e in C.events_csv():
        st, en = C.parse_dt(e["start"]), C.parse_dt(e["end"])
        if st is None:
            continue
        if en is None:
            en = st + TD(seconds=float(e["duration_s"] or 30))
        ev_all[e["animal"]].append((st, en, e["label"], e["xlsx"], e["row"]))
    for r in new_negs:
        a, s = r["animal"], r["session"]
        d = os.path.join(C.NEWFULL, f"Data_{a}_cropped", s, r["clip_dir"])
        info = C.parse_info(os.path.join(d, "info.txt"))[0]
        cs, ce = C.parse_dt(info["Clip start"]), C.parse_dt(info["Clip end"])
        px, parsed, videos, rp = replays[(a, s)]
        cn = int(r["clip_dir"].split("_")[1])
        rec = dict(key=r["key"], animal=a, session=s, clip=r["clip_dir"], clip_start=cs, clip_end=ce,
                   video=info["Video file"], deviations=r["deviations"])
        xrel = os.path.join(C.RAW, r["xlsx"])
        if xrel == px:
            r_ = rp[cn]
            rec["rule"] = "parent xlsx stream"
        else:
            # second xlsx: fresh seed-42 stream over that xlsx, both xlsx blocked; the event's own position
            union = [e for x in parsed.values() for e in x]
            rp2 = replay(parsed[xrel], union, videos)
            msd = C.parse_dt(info["Matched seizure start"])
            pos = next(k for k, e in enumerate(parsed[xrel], 1) if e["start"] == msd)
            r_ = rp2[pos]
            rec["rule"] = f"second xlsx stream, position {pos}"
        rec["replay_start"] = r_["start"]
        rec["replay_video"] = r_["video"]
        rec["replay_dev_s"] = abs((r_["start"] - cs).total_seconds()) if r_["start"] else None
        rec["replay_same_video"] = r_["video"] == info["Video file"]
        rec["seizure_video"] = r_["src"]
        rec["from_seizure_video"] = r_["src"] == info["Video file"]
        # events of the animal in ANY xlsx
        gaps = []
        inside = []
        for st, en, lab, xl, row in ev_all[a]:
            if st < ce and en > cs:
                inside.append(f"{lab} {st} ({os.path.basename(xl)} row {row})")
            gap = max((st - ce).total_seconds(), (cs - en).total_seconds())
            gaps.append((gap, lab, st, os.path.basename(xl)))
        g = min(gaps) if gaps else (None,)
        rec["events_inside"] = " | ".join(inside)
        rec["min_gap_s"] = g[0]
        rec["nearest_event"] = f"{g[1]} {g[2]} {g[3]}" if gaps else ""
        rec["n_events_within_40s"] = sum(1 for x in gaps if x[0] < 40)
        # inside the chosen video and EDF window (parent: seek = clip start - video start)
        rows.append(rec)
    # ---------------------------------------------------------------- pairing
    ex_names = {}
    pairing = []
    new_neg_by = defaultdict(list)
    for r in new_negs:
        m = re.match(r"clip_\d+_vs_(seizure_\d+_Stage_\d_\d{8}_\d{6})$", r["clip_dir"])
        new_neg_by[(r["animal"], r["session"], m.group(1))].append(r["clip_dir"])
    for r in new_sz:
        a, s = r["animal"], r["session"]
        if (a, s) not in ex_names:
            ex_names[(a, s)] = {os.path.basename(p) for root in (C.DATA, C.FULL)
                                for p in glob.glob(os.path.join(glob.escape(os.path.join(root, f"Data_{a}_cropped", s)),
                                                                "clip_*_vs_seizure_*"))}
        tail = r["clip_dir"]
        exist = sorted(n for n in ex_names[(a, s)] if n.endswith("_vs_" + tail))
        new = new_neg_by.get((a, s, tail), [])
        man = r["existing_negative"].split("/")[-1] if r["existing_negative"] else ""
        pairing.append(dict(key=r["key"], existing=exist, new=new, manifest_existing=man,
                            ok=(len(exist) + len(new) == 1) and ((man == (exist[0] if exist else "")) or
                                                                (not exist and not man))))
    # ---------------------------------------------------------------- existing negatives containing a new seizure
    contain = []
    for r in new_sz:
        a = r["animal"]
        on, en = C.parse_dt(r["onset"]), C.parse_dt(r["seizure_end"])
        for root in (C.FULL,):
            for p in glob.glob(os.path.join(glob.escape(os.path.join(root, f"Data_{a}_cropped")), "*",
                                            "clip_*_vs_seizure_*", "info.txt")):
                info = C.parse_info(p)[0]
                cs, ce = C.parse_dt(info.get("Clip start")), C.parse_dt(info.get("Clip end"))
                if cs and ce and on < ce and en > cs:
                    contain.append((r["clip_dir"], os.path.relpath(os.path.dirname(p), C.FULL)))
    # also: new negatives vs new seizure windows (any session of the animal)
    neg_vs_newsz = []
    for n in rows:
        for r in new_sz:
            if r["animal"] != n["animal"]:
                continue
            cs, ce = C.parse_dt(r["clip_start"]), C.parse_dt(r["clip_end"])
            if cs < n["clip_end"] and ce > n["clip_start"]:
                neg_vs_newsz.append((n["clip"], r["clip_dir"]))
    devs = [x["replay_dev_s"] for x in rows if x["replay_dev_s"] is not None]
    out.update(
        n_new_negatives=len(rows),
        replay_reproduces_new=sum(1 for x in rows if x["replay_dev_s"] is not None and x["replay_dev_s"] <= 1e-3
                                  and x["replay_same_video"]),
        replay_max_dev_s=max(devs) if devs else None,
        not_reproduced=[(x["clip"], x["replay_dev_s"], x["replay_video"], x["video"]) for x in rows
                        if not (x["replay_dev_s"] is not None and x["replay_dev_s"] <= 1e-3 and x["replay_same_video"])],
        from_seizure_video=Counter(x["from_seizure_video"] for x in rows),
        negatives_with_event_inside=[(x["clip"], x["events_inside"]) for x in rows if x["events_inside"]],
        negatives_with_event_within_40s=[(x["clip"], x["min_gap_s"], x["nearest_event"]) for x in rows
                                         if x["n_events_within_40s"]],
        min_gap_overall_s=min(x["min_gap_s"] for x in rows),
        pairing_ok=Counter(p["ok"] for p in pairing),
        pairing_bad=[p for p in pairing if not p["ok"]],
        pairing_counts=dict(existing=sum(1 for p in pairing if p["existing"]), new=sum(1 for p in pairing if p["new"])),
        existing_negatives_containing_new_seizure=contain,
        new_negatives_overlapping_new_seizure_clips=neg_vs_newsz,
    )
    C.write_csv("vn_negatives.csv", rows)
    C.write_json("vn_negatives.json", out)
    print(json.dumps(out, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
