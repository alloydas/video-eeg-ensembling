# Independent check that the 81 re-cut clips were encoded like the parent's data_full clips.
# For every manifest row: ffprobe the new clip and the original data_full clip (codec, profile, pix_fmt, size,
# r/avg frame rate, time base, frame count, duration), read the x264 SEI option string from both files, and
# compare the size with raw_audit/align/crop_boxes.csv. Frame count is compared with the copy-cut expectation
# (new seek - new keyframe + v_clip_dur) x 15 and with the original clip's count. Read-only; writes encode.json/.csv.
import os, re, csv, json, subprocess
from concurrent.futures import ThreadPoolExecutor
FF = "/work/mech-ai/alloy/miniconda3/bin"
EEG = "/work/mech-ai-scratch/alloy/EEG"
R = f"{EEG}/output/ttg_recut"
OUT = f"{R}/verify/indep"
csv.field_size_limit(1 << 30)


def probe(p):
    r = subprocess.run([f"{FF}/ffprobe", "-v", "error", "-of", "json", "-count_frames", "-show_streams", "-show_format",
                        p], capture_output=True)
    j = json.loads(r.stdout or b"{}")
    v = [s for s in j.get("streams", []) if s.get("codec_type") == "video"]
    a = [s for s in j.get("streams", []) if s.get("codec_type") == "audio"]
    s = v[0] if v else {}
    head = open(p, "rb").read(1 << 17)
    m = re.search(rb"x264 - core (\d+)[^\x00]*?options: ([^\x00]*)", head)
    return dict(codec=s.get("codec_name"), profile=s.get("profile"), pix_fmt=s.get("pix_fmt"), w=s.get("width"),
                h=s.get("height"), r_fps=s.get("r_frame_rate"), avg_fps=s.get("avg_frame_rate"),
                tb=s.get("time_base"), nb_frames=int(s.get("nb_frames", -1)), nb_read=int(s.get("nb_read_frames", -1)),
                dur=float(s.get("duration", "nan")), n_audio=len(a), n_video=len(v),
                x264_core=m.group(1).decode() if m else None, x264_opts=m.group(2).decode() if m else None,
                encoder=j.get("format", {}).get("tags", {}).get("encoder"))


def norm_opts(o):
    # threads / lookahead_threads / sliced_threads depend on the machine; everything else must match
    return " ".join(t for t in (o or "").split() if not t.startswith(("threads=", "lookahead_threads=")))


def one(r):
    new = os.path.join(EEG, r["out"])
    old = f"{EEG}/data_full/Data_{r['animal']}_cropped/{r['session']}/{r['clip_dir']}/video.mp4"
    pn, po = probe(new), probe(old)
    exp = (float(r["new_seek_pts"]) - float(r["new_keyframe_pts"]) + float(r["v_clip_dur_s"])) * 15.0
    return dict(key=r["clip_key"], new=pn, old=po, expected_frames=exp, truncated_s=float(r["truncated_s"] or 0),
                clip_dur_s=float(r["clip_dur_s"]), v_clip_dur_s=float(r["v_clip_dur_s"]))


def main():
    man = list(csv.DictReader(open(f"{R}/manifest.csv")))
    boxes = {}
    for b in csv.DictReader(open(f"{EEG}/output/raw_audit/align/crop_boxes.csv")):
        boxes.setdefault(b["animal"], (int(b["w"]), int(b["h"])))
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(one, man))
    agg = dict(n=len(res), problems=[])
    same = lambda f: sum(x["new"][f] == x["old"][f] for x in res)  # noqa: E731
    for f in ("codec", "profile", "pix_fmt", "w", "h", "r_fps", "avg_fps", "tb", "n_audio"):
        agg[f"same_{f}"] = same(f)
    agg["new_values"] = {f: sorted({str(x["new"][f]) for x in res}) for f in
                         ("codec", "profile", "pix_fmt", "r_fps", "avg_fps", "tb", "x264_core", "encoder", "n_audio")}
    agg["old_values"] = {f: sorted({str(x["old"][f]) for x in res}) for f in
                         ("codec", "profile", "pix_fmt", "r_fps", "avg_fps", "tb", "x264_core", "encoder", "n_audio")}
    agg["x264_opts_equal_modulo_threads"] = sum(norm_opts(x["new"]["x264_opts"]) == norm_opts(x["old"]["x264_opts"]) for x in res)
    diffs = set()
    for x in res:
        a = set(norm_opts(x["new"]["x264_opts"]).split()); b = set(norm_opts(x["old"]["x264_opts"]).split())
        diffs |= {("new_only", t) for t in a - b} | {("old_only", t) for t in b - a}
    agg["x264_opt_differences"] = sorted(diffs)
    agg["crf18_preset_fast_markers"] = sum(("crf=18.0" in (x["new"]["x264_opts"] or "")) for x in res)
    agg["size_equals_crop_box"] = sum((x["new"]["w"], x["new"]["h"]) == boxes[x["key"].split("/")[1][5:10]] for x in res)
    agg["nb_frames_header_eq_decoded_new"] = sum(x["new"]["nb_frames"] == x["new"]["nb_read"] for x in res)
    dexp = [x["new"]["nb_read"] - x["expected_frames"] for x in res]
    agg["new_frames_minus_expected"] = dict(min=min(dexp), max=max(dexp), n_within_1=sum(abs(d) <= 1 for d in dexp))
    dold = [x["new"]["nb_read"] - x["old"]["nb_read"] for x in res]
    agg["new_minus_old_frames"] = dict(min=min(dold), max=max(dold), n_within_32=sum(abs(d) <= 32 for d in dold))
    agg["old_truncated_like"] = [(x["key"], x["old"]["nb_read"], x["new"]["nb_read"], round(x["clip_dur_s"] * 15))
                                 for x in res if abs(x["new"]["nb_read"] - x["old"]["nb_read"]) > 32]
    agg["new_short_of_label"] = [(x["key"], x["new"]["nb_read"], round(x["clip_dur_s"] * 15), x["truncated_s"])
                                 for x in res if x["truncated_s"] > 0]
    agg["durations_new_minus_old_s"] = dict(min=min(x["new"]["dur"] - x["old"]["dur"] for x in res),
                                            max=max(x["new"]["dur"] - x["old"]["dur"] for x in res))
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(summary=agg, rows=res), open(f"{OUT}/encode.json", "w"), indent=1)
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
