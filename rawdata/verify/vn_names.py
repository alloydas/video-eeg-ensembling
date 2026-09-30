#!/usr/bin/env python3
"""Priority 3: naming, numbering and info.txt of the new clips, against the parent's rules and the existing tree.

For every new clip (status != excluded):
  - the directory holds exactly video.mp4, eeg.edf, info.txt
  - the name matches the parent's pattern; the stage and time stamp match the xlsx event (own xlsx reader)
  - seizure index = xlsx sheet row - 1 (cut_seizure_clips: enumerate(iter_rows(min_row=2)) + 1); negative clip_n =
    position of the matched event among the xlsx's timed rows (cut_non_seizure_clips: enumerate(targets, 1))
  - info.txt: same keys in the same order as an existing info.txt of the same kind in the same session; every value
    recomputed here from the xlsx, the raw video (XML start, nb_frames / avg_frame_rate) and the raw EDF header
  - collisions: no new name, seizure index or clip number repeats one in data/ or data_full/ of that session, or
    another new clip; no new key is in either frame cache index
Writes verify/vn_names.csv and vn_names.json.
"""
import datetime as dt
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict
from fractions import Fraction

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vn_common as C  # noqa: E402

SZ_RE = re.compile(r"^seizure_(\d{2,})_(Stage_[2-5])_(\d{8}_\d{6})$")
NS_RE = re.compile(r"^clip_(\d{2,})_vs_seizure_(\d{2,})_(Stage_[2-5])_(\d{8}_\d{6})$")
_VINFO = {}


def video_info(path):
    """(Central naive start, duration by nb_frames / avg_frame_rate, fps, w, h) of a raw mp4."""
    if path not in _VINFO:
        x = C.xml_of(path)
        start = C.central_naive(int(x["DSI_utc_start_time"]))
        r = C.run([C.FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
                   "stream=avg_frame_rate,nb_frames,width,height", "-of", "json", path])
        s = json.loads(r.stdout)["streams"][0]
        fps = float(Fraction(s["avg_frame_rate"]))
        _VINFO[path] = dict(start=start, dur=int(s["nb_frames"]) / fps, fps=fps, w=s["width"], h=s["height"],
                            sync=float(x["DSI_sync_offset"]) / 1e7, utc=int(x["DSI_utc_start_time"]))
    return _VINFO[path]


def hms(s):
    s = max(0, int(s))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def existing_names(animal, session):
    out = set()
    for root in (C.DATA, C.FULL):
        for d in glob.glob(os.path.join(glob.escape(os.path.join(root, f"Data_{animal}_cropped", session)), "*")):
            out.add(os.path.basename(d))
    return out


def template(animal, session, kind):
    """Key order of an existing info.txt of this kind: same session, else same animal (any session)."""
    pat = "seizure_*" if kind == "seizure" else "clip_*_vs_seizure_*"
    for root in (os.path.join(glob.escape(os.path.join(C.FULL, f"Data_{animal}_cropped", session)), pat),
                 os.path.join(glob.escape(os.path.join(C.FULL, f"Data_{animal}_cropped")), "*", pat)):
        for d in sorted(glob.glob(root)):
            p = os.path.join(d, "info.txt")
            if os.path.exists(p):
                return C.parse_info(p)[1], os.path.relpath(d, C.FULL)
    return None, None


def main():
    M = [r for r in C.manifest() if r["status"] != "excluded"]
    rows, problems = [], []
    xl_cache = {}
    edf_cache = {}
    new_names = Counter()
    new_idx = defaultdict(Counter)       # (animal, session) -> seizure idx counter
    new_cn = defaultdict(Counter)        # (animal, session) -> clip_n counter
    keys_f16 = set(json.load(open(os.path.join(C.EEG_ROOT, "cache_frames/f16s224/index.json")))["paths"])
    keys_f32 = set(json.load(open(os.path.join(C.EEG_ROOT, "cache_frames/f32s224/index.json")))["paths"])
    for r in M:
        a, s, name, kind = r["animal"], r["session"], r["clip_dir"], r["kind"]
        d = os.path.join(C.NEWFULL, f"Data_{a}_cropped", s, name)
        pr = []
        files = sorted(os.listdir(d))
        if files != ["eeg.edf", "info.txt", "video.mp4"]:
            pr.append(f"files {files}")
        info, order = C.parse_info(os.path.join(d, "info.txt"))
        raw_text = open(os.path.join(d, "info.txt")).read()
        new_names[(a, s, name)] += 1
        m = (SZ_RE if kind == "seizure" else NS_RE).match(name)
        if not m:
            pr.append("name pattern")
            rows.append(dict(key=r["key"], kind=kind, problems="; ".join(pr)))
            problems.append((r["key"], pr))
            continue
        dev = r["deviations"]
        second_xlsx = "second xlsx" in dev
        # the xlsx the event comes from
        xrel = r["xlsx"]
        if xrel not in xl_cache:
            xl_cache[xrel] = C.xlsx_events(os.path.join(C.RAW, xrel))[0]
        ev = xl_cache[xrel]
        sz_start = C.parse_dt(info["Seizure start"] if kind == "seizure" else info["Matched seizure start"])
        hits = [(k, e) for k, e in enumerate(ev, 1) if e["start"] == sz_start]
        if len(hits) != 1:
            pr.append(f"xlsx event match {len(hits)}")
            rows.append(dict(key=r["key"], kind=kind, problems="; ".join(pr)))
            problems.append((r["key"], pr))
            continue
        pos, e = hits[0]
        sz_end = e["end"] if e["end"] else e["start"] + dt.timedelta(seconds=e["duration"])
        stage = re.sub(r"[^\w]", "_", e["label"])
        stamp = e["start"].strftime("%Y%m%d_%H%M%S")
        if kind == "seizure":
            idx, st, ts = int(m.group(1)), m.group(2), m.group(3)
            if not second_xlsx and idx != e["idx"]:
                pr.append(f"seizure idx {idx} != xlsx row-1 {e['idx']}")
            new_idx[(a, s)][idx] += 1
        else:
            cn, idx, st, ts = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)
            if not second_xlsx and cn != pos:
                pr.append(f"clip_n {cn} != event position {pos}")
            if not second_xlsx and idx != e["idx"]:
                pr.append(f"matched idx {idx} != xlsx row-1 {e['idx']}")
            new_cn[(a, s)][cn] += 1
        if st != stage:
            pr.append(f"stage {st} != {stage}")
        if ts != stamp:
            pr.append(f"stamp {ts} != {stamp}")
        # template key order
        tmpl, tname = template(a, s, kind)
        if tmpl is None:
            pr.append("no template in session")
        elif order != tmpl:
            pr.append(f"info keys differ from {tname}")
        # raw video and EDF
        vpath = os.path.join(C.RAW, r["raw_video"])
        if os.path.basename(vpath) != info["Video file"]:
            pr.append("video file != manifest raw_video")
        vi = video_info(vpath)
        epath = os.path.join(C.RAW, r["raw_edf"])
        if epath not in edf_cache:
            E = C.EDF(epath)
            edf_cache[epath] = dict(start=E.start(), dur=E.nrec * E.rec_dur, ns=E.ns, sf=float(E.fs.max()))
        ei = edf_cache[epath]
        if os.path.basename(epath) != info["EDF file"]:
            pr.append("EDF file != manifest raw_edf")
        if info["EEG channels"] != str(ei["ns"]):
            pr.append(f"EEG channels {info['EEG channels']} != {ei['ns']}")
        if info["EEG sfreq (Hz)"] != str(ei["sf"]):
            pr.append(f"sfreq {info['EEG sfreq (Hz)']} != {ei['sf']}")
        v_end = vi["start"] + dt.timedelta(seconds=vi["dur"])
        e_end = ei["start"] + dt.timedelta(seconds=ei["dur"])
        rec = dict(key=r["key"], animal=a, session=s, kind=kind, name=name, xlsx_row=e["sheet_row"], xlsx_idx=e["idx"],
                   event_pos=pos, video=info["Video file"], video_start=vi["start"], video_end=v_end, edf_start=ei["start"])
        if kind == "seizure":
            cs = e["start"] - dt.timedelta(seconds=10)
            ce = sz_end + dt.timedelta(seconds=10)
            exp = {"Seizure index": str(idx if second_xlsx else e["idx"]), "Label": e["label"],
                   "Seizure start": str(e["start"]), "Seizure end": str(sz_end), "Duration (s)": f"{e['duration']:.2f}",
                   "Spikes": str(e["spikes"]), "Clip start": str(cs), "Clip end": str(ce), "Pre-buffer (s)": "10.0",
                   "Post-buffer(s)": "10.0", "EDF date src": "header", "Video FPS": str(round(vi["fps"], 3)),
                   "Video res": f"{vi['w']}x{vi['h']}"}
            for k, v in exp.items():
                if info.get(k) != v:
                    pr.append(f"{k}: {info.get(k)!r} != {v!r}")
            # the parent's video rule: onset in [start, start + dur); seek = max(0, onset - start - 10)
            if not (vi["start"] <= e["start"] < v_end):
                pr.append("onset not inside the named video")
            seek = max(0.0, (e["start"] - vi["start"]).total_seconds() - 10)
            rec["seek"] = seek
            if abs(seek - float(r["seek_s"])) > 1e-3:
                pr.append(f"seek {seek:.3f} != manifest {r['seek_s']}")
            if not (ei["start"] <= e["start"] < e_end):
                pr.append("onset not inside the EDF")
            vdur = min((ce - cs).total_seconds(), vi["dur"] - seek)
            rec["v_clip_dur"] = vdur
            if abs(vdur - float(r["v_clip_dur_s"])) > 0.01:
                pr.append(f"v_clip_dur {vdur:.3f} != manifest {r['v_clip_dur_s']}")
        else:
            cs, ce = C.parse_dt(info["Clip start"]), C.parse_dt(info["Clip end"])
            cdur = 10 + e["duration"] + 10
            exp = {"Non-seizure clip index": str(cn), "Matched seizure index": str(idx),
                   "Matched seizure label": e["label"], "Matched seizure end": str(sz_end),
                   "Matched seizure dur (s)": f"{e['duration']:.2f}", "Clip duration (s)": f"{cdur:.2f}",
                   "Pre-buffer (s)": "10.0", "Post-buffer (s)": "10.0", "Safety margin (s)": "30.0"}
            for k, v in exp.items():
                if info.get(k) != v:
                    pr.append(f"{k}: {info.get(k)!r} != {v!r}")
            if abs((ce - cs).total_seconds() - cdur) > 1e-3:
                pr.append("clip end - start != 20 + duration")
            seek = (cs - vi["start"]).total_seconds()
            rec["seek"] = seek
            if info["Video seek"] != f"{hms(seek)} ({seek:.2f}s)":
                pr.append(f"Video seek {info['Video seek']!r} != {hms(seek)} ({seek:.2f}s)")
            if not (vi["start"] <= cs and ce <= v_end):
                pr.append("window not inside the named video")
            if not (ei["start"] <= cs < e_end):
                pr.append("clip start not inside the EDF")
            if abs(seek - float(r["seek_s"])) > 1e-3:
                pr.append(f"seek {seek:.3f} != manifest {r['seek_s']}")
        # manifest agrees with info.txt
        for mk, ik in (("clip_start", "Clip start"), ("clip_end", "Clip end")):
            if C.parse_dt(r[mk]) != C.parse_dt(info[ik]):
                pr.append(f"manifest {mk} != info")
        rec["problems"] = "; ".join(pr)
        rows.append(rec)
        if pr:
            problems.append((r["key"], pr))
        if r["key"] in keys_f16 or r["key"] in keys_f32:
            problems.append((r["key"], ["key already in a frame cache index"]))
    # collisions against the existing tree
    coll = []
    for (a, s), cnt in list(new_idx.items()) + list(new_cn.items()):
        pass
    sessions = {(r["animal"], r["session"]) for r in M}
    for a, s in sorted(sessions):
        ex = existing_names(a, s)
        ex_sz = Counter(int(SZ_RE.match(n).group(1)) for n in ex if SZ_RE.match(n))
        ex_sz_any = Counter(int(re.match(r"seizure_(\d+)_", n).group(1)) for n in ex if re.match(r"seizure_(\d+)_", n))
        ex_cn = Counter(int(re.match(r"clip_(\d+)_", n).group(1)) for n in ex if re.match(r"clip_(\d+)_", n))
        for n in [x for (aa, ss, x) in new_names if (aa, ss) == (a, s)]:
            if n in ex:
                coll.append((a, s, n, "name exists"))
        for i, c in new_idx[(a, s)].items():
            if c > 1:
                coll.append((a, s, f"seizure_{i:02d}", "index repeated among new clips"))
            if i in ex_sz_any:
                coll.append((a, s, f"seizure_{i:02d}", "index exists in data/ or data_full/"))
        for i, c in new_cn[(a, s)].items():
            if c > 1:
                coll.append((a, s, f"clip_{i:02d}", "clip number repeated among new clips"))
            if i in ex_cn:
                coll.append((a, s, f"clip_{i:02d}", "clip number exists in data/ or data_full/"))
    dup_new = [k for k, c in new_names.items() if c > 1]
    out = dict(n_checked=len(M), n_with_problems=len(problems), problems=problems[:60], collisions=coll,
               duplicate_new_names=dup_new,
               n_by_kind=Counter(r["kind"] for r in M))
    C.write_csv("vn_names.csv", rows)
    C.write_json("vn_names.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "problems"}, default=str, indent=1))
    for k, p in problems[:40]:
        print(k, p)


if __name__ == "__main__":
    main()
