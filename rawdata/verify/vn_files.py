#!/usr/bin/env python3
"""Timing situation of every raw camera file the new clips use (audit files.csv + own XML read): known defect stamps
(10-13 room D, 11-03 / 12-05 room-C clock steps, .20231109055600), DST span, wall-clock minus stream duration (frames
missing), and the day folder. Writes verify/vn_files.json."""
import datetime as dt
import json
import os
import sys
from collections import Counter

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vn_common as C  # noqa: E402

DEFECT = {"20231014004100": "10-13 room D", "20231103122100": "11-03 room C step", "20231205222300": "12-05 room C step",
          "20231109055600": "room C 13 s"}
DST = dt.datetime(2023, 11, 5, 7, 0, 0)


def main():
    F = C.files_csv()
    M = [r for r in C.manifest() if r["status"] != "excluded"]
    by = Counter(r["raw_video"] for r in M)
    rows = []
    for v, n in sorted(by.items()):
        f = F[v]
        x = C.xml_of(os.path.join(C.RAW, v))
        u = int(x["DSI_utc_start_time"])
        st = dt.datetime.fromtimestamp(u, dt.timezone.utc).replace(tzinfo=None)
        en = C.parse_dt(f["end_utc"])
        stamp = v[-18:-4]
        rows.append(dict(video=v, n_clips=n, stamp=stamp, defect=DEFECT.get(stamp), dst_span=bool(st < DST <= en),
                         wall_minus_stream_s=float(f["wall_minus_stream_s"]), problems=f["problems"],
                         xml_start_eq_audit=abs(u - int(f["xml_utc_start"])) == 0,
                         camera=os.path.basename(v).split(".")[0], stamp_date=stamp[:8]))
    out = dict(n_files=len(rows), defect_files=[r for r in rows if r["defect"]], dst_span=[r for r in rows if r["dst_span"]],
               wall_minus_stream_abs_gt_1s=[(r["video"], r["wall_minus_stream_s"]) for r in rows if abs(r["wall_minus_stream_s"]) > 1],
               frames_missing=[r["video"] for r in rows if "frames_missing" in (r["problems"] or "")],
               problems=dict(Counter(r["problems"] for r in rows)),
               any_file_on_20231109=[r["video"] for r in rows if r["stamp_date"] == "20231109"],
               xml_start_matches_audit=all(r["xml_start_eq_audit"] for r in rows), rows=rows)
    C.write_json("vn_files.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
