#!/usr/bin/env python3
"""Collect the vn_*.json results into verify/vn_summary.json (one block per priority)."""
import json
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vn_common as C  # noqa: E402


def J(n):
    p = os.path.join(C.VOUT, n)
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    ro, nm, ng, ed, rd, al, fr, ct, sn, fl, du, co = (J(x) for x in (
        "vn_readonly.json", "vn_names.json", "vn_negatives.json", "vn_edf.json", "vn_reader.json", "vn_align.json",
        "vn_frames.json", "vn_counts.json", "vn_sanity.json", "vn_flat.json", "vn_dupes.json", "vn_controls.json"))
    out = {
        "1_readonly": dict(ok=ro["ok_readonly_trees"], raw_vs_audit=ro["raw_vs_audit"],
                           changed_after_T0={k: ro[k]["changed_after_T0"] for k in ("raw_scan", "data", "data_full", "cache_frames", "eeg_repo_other")},
                           output_dirs_changed=list(ro["output_dirs_changed_after_T0"]),
                           new_clips_in_data_or_full=ro["new_clips_present_in_data_or_data_full"]),
        "2_eeg_equals_raw_all_505": dict(new=ed["new"], lag_bad=ed["lag_bad"], q_bad=ed["q_bad"]),
        "2_frames_origin_all_505": {k: v for k, v in (fr or {}).items() if k != "bad"},
        "2_video_timing_sample": {k: v for k, v in (al or {}).items() if k != "rows"},
        "3_names_info": nm,
        "4_negatives": {k: v for k, v in ng.items() if k != "sessions"},
        "4_new_seizures_duplicates": du,
        "5_edf_format": dict(header_field_diffs=ed["header_field_diffs_new_vs_existing"], existing_controls=ed["existing_controls"],
                             reader=rd, flat=fl, controls=co),
        "6_label_flags": {k: v for k, v in (sn or {}).items() if k != "calibration"},
        "7_counts": dict(n_checks=co and ct["n_checks"], n_equal=ct["n_equal"], not_equal=ct["not_equal"]),
    }
    C.write_json("vn_summary.json", out)
    print(json.dumps({k: (v.get("ok") if isinstance(v, dict) and "ok" in v else "see file") for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()
