#!/usr/bin/env python3
"""Every new clip: where does its first frame come from in the raw file? (own decode, own matching)

Decode the clip's first 3 s and the raw file over [seek - 4, seek + 4] in the animal's crop box (gray, 1/4 size,
passthrough timing, pts from showinfo); match the clip's first 30 frames as a block against every raw offset (mean
pixel NCC). Checks: best offset's raw pts lies in [seek - 2.2, seek + 0.07] (the copy-cut starts at the keyframe at or
before the parent's seek), block NCC high and far above the next-best offset > 1 s away; the clip's frame count
against v_clip_dur * 15; the clip's last frames match the raw at the same offset. Writes verify/vn_frames.csv/json.
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vn_common as C  # noqa: E402
import vn_align as A  # noqa: E402


def ncc_block(a, b):
    a = a.reshape(len(a), -1).astype(np.float64); b = b.reshape(len(b), -1).astype(np.float64)
    a -= a.mean(axis=1, keepdims=True); b -= b.mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1); den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1)) + 1e-9
    return float((num / den).mean())


def decode_k(path, t0, dur, box):
    """A.decode plus the showinfo iskey flag of every frame."""
    import re
    x, y, w, h = box
    ow, oh = w // A.DS, h // A.DS
    cmd = [C.FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "info", "-threads", "2", "-ss", f"{t0:.3f}", "-i", path,
           "-t", f"{dur:.3f}", "-vf", f"crop={w}:{h}:{x}:{y},scale={ow}:{oh}:flags=area,format=gray,showinfo", "-an",
           "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
    r = C.run(cmd, text=False, timeout=3600)
    fr = np.frombuffer(r.stdout, np.uint8)
    n = fr.size // (ow * oh)
    fr = fr[: n * ow * oh].reshape(n, oh, ow)
    info = [(float(m.group(1)), int(m.group(2))) for m in
            re.finditer(rb"pts_time:\s*([-0-9.e]+).*?iskey:\s*(\d)", r.stderr)]
    k = min(n, len(info))
    return fr[:k], np.array([i[0] for i in info[:k]]), np.array([i[1] for i in info[:k]])


def one(r):
    d = os.path.join(C.NEWFULL, f"Data_{r['animal']}_cropped", r["session"], r["clip_dir"])
    vp = os.path.join(C.RAW, r["raw_video"])
    seek = float(r["seek_s"])
    vdur = float(r["v_clip_dur_s"])
    out = dict(key=r["key"], seek=seek)
    fc, _ = A.decode(os.path.join(d, "video.mp4"), 0, 3.0, None)
    fc_all_n = int(C.run([C.FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                          "stream=nb_read_packets", "-of", "csv=p=0", os.path.join(d, "video.mp4")]).stdout.strip())
    out["clip_frames"] = fc_all_n
    out["expected_frames"] = vdur * 15
    w0 = max(0.0, seek - 4)
    fr, pts, iskey = decode_k(vp, w0, 8.0, A.BOX[r["animal"]])
    pts = pts + w0
    n = min(30, len(fc))
    sc = []
    for j in range(0, len(fr) - n + 1):
        sc.append(ncc_block(fc[:n], fr[j:j + n]))
    sc = np.array(sc)
    j = int(np.argmax(sc))
    far = np.abs(pts[: len(sc)] - pts[j]) > 1.0
    out["best_ncc"] = float(sc[j])
    out["next_ncc_far"] = float(sc[far].max()) if far.any() else None
    out["clip0_raw_pts"] = float(pts[j])
    out["clip0_minus_seek"] = float(pts[j] - seek)
    out["keyframe_rule_ok"] = bool(-2.2 <= pts[j] - seek <= 0.07)
    out["matched_raw_frame_is_keyframe"] = bool(iskey[j])
    keys = pts[iskey == 1]
    before = keys[keys <= seek + 1e-6]
    out["keyframe_at_or_before_seek"] = float(before.max()) if len(before) else None
    out["matched_is_that_keyframe"] = bool(len(before) and abs(before.max() - pts[j]) < 1e-3)
    out["frames_minus_copycut_expectation"] = fc_all_n - (vdur - (pts[j] - seek)) * 15
    # tail: the clip's last 2 s against the raw at the same offset
    t_tail = max(0.0, fc_all_n / 15.0 - 2.0)
    ft, _ = A.decode(os.path.join(d, "video.mp4"), t_tail, 2.0, None)
    k0 = pts[j] + t_tail
    fr2, p2 = A.decode(vp, max(0.0, k0 - 1.0), 4.0, A.BOX[r["animal"]])
    p2 = p2 + max(0.0, k0 - 1.0)
    m = min(15, len(ft))
    sc2 = [ncc_block(ft[:m], fr2[i:i + m]) for i in range(0, len(fr2) - m + 1)]
    i2 = int(np.argmax(sc2)) if sc2 else None
    out["tail_ncc"] = float(sc2[i2]) if sc2 else None
    out["tail_offset_s"] = float(p2[i2] - k0) if sc2 else None
    return out


def main():
    M = [r for r in C.manifest() if r["status"] != "excluded"]
    with ThreadPoolExecutor(12) as ex:
        rows = list(ex.map(one, M))
    ok = [x for x in rows if x["keyframe_rule_ok"] and x["matched_is_that_keyframe"] and x["best_ncc"] > 0.95
          and x["tail_offset_s"] is not None and abs(x["tail_offset_s"]) <= 0.14]
    out = dict(n=len(rows), n_ok=len(ok), min_best_ncc=min(x["best_ncc"] for x in rows),
               max_next_far=max((x["next_ncc_far"] or 0) for x in rows),
               clip0_minus_seek_range=[min(x["clip0_minus_seek"] for x in rows), max(x["clip0_minus_seek"] for x in rows)],
               frames_minus_expected_range=[min(x["clip_frames"] - x["expected_frames"] for x in rows),
                                            max(x["clip_frames"] - x["expected_frames"] for x in rows)],
               tail_offset_range=[min(x["tail_offset_s"] for x in rows), max(x["tail_offset_s"] for x in rows)],
               min_tail_ncc=min(x["tail_ncc"] for x in rows),
               matched_is_keyframe_at_or_before_seek=sum(x["matched_is_that_keyframe"] for x in rows),
               frames_minus_copycut_expectation_range=[min(x["frames_minus_copycut_expectation"] for x in rows),
                                                       max(x["frames_minus_copycut_expectation"] for x in rows)],
               bad=[x for x in rows if x not in ok][:20])
    C.write_csv("vn_frames.csv", rows)
    C.write_json("vn_frames.json", out)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
