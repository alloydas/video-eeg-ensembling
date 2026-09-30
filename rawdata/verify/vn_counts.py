#!/usr/bin/env python3
"""Priority 7: every count in summary.txt / summary.json / integration.txt / the cut report, recomputed from
manifest.csv, items.csv and an own replica of train_pooled.discover() / split_subjects() / split_sessions()
(stdlib random, as the parent). EEG window counts come from vn_reader.json (the parent's own builder function).
Writes verify/vn_counts.json (each check: reported, recomputed, equal).
"""
import glob
import json
import os
import random
import re
import statistics
import sys
from collections import Counter, defaultdict

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vn_common as C  # noqa: E402

STAGE = {"Stage_2": 1, "Stage_3": 2, "Stage_4": 3, "Stage_5": 4}
NAMES = ["non-seizure", "Stage2", "Stage3", "Stage4", "Stage5"]


def roots():
    src = open(os.path.join(C.EEG_ROOT, "train_pooled.py")).read()
    block = src[src.index("ROOTS = {"): src.index("}", src.index("ROOTS = {")) + 1]
    return dict(re.findall(r'"(RN\d+)":\s*"([^"]+)"', block))


def discover():
    items = []
    for subj, root in roots().items():
        root = os.path.join(C.EEG_ROOT, root)
        for d in glob.glob(f"{glob.escape(root)}/*/seizure_*") + glob.glob(f"{glob.escape(root)}/*/clip_*_vs_seizure_*"):
            if not os.path.exists(os.path.join(d, "video.mp4")):
                continue
            b, sess = os.path.basename(d), os.path.basename(os.path.dirname(d))
            if b.startswith("seizure_"):
                m = re.search(r"Stage_[0-9]+", b)
                if not m or m.group() not in STAGE:
                    continue
                y = STAGE[m.group()]
            else:
                y = 0
            items.append((os.path.relpath(os.path.join(d, "video.mp4"), C.EEG_ROOT), y, f"{subj}/{sess}", subj))
    return items


def split_subjects(items, seed=49, n_folds=5):
    subs = sorted({i[3] for i in items})
    rng = random.Random(seed); rng.shuffle(subs)
    return [subs[k::n_folds] for k in range(n_folds)]


def split_sessions(items, seed=49, val_frac=0.2, nc=5):
    sessions = sorted({i[2] for i in items})
    for s in range(seed, seed + 500):
        rng = random.Random(s); ss = sessions[:]; rng.shuffle(ss)
        val = set(ss[:max(1, round(len(ss) * val_frac))])
        tr = [i for i in items if i[2] not in val]
        va = [i for i in items if i[2] in val]
        ctr, cva = Counter(i[1] for i in tr), Counter(i[1] for i in va)
        if all(ctr[c] > 0 for c in range(nc)) and all(cva[c] > 0 for c in range(nc)):
            return val, s
    return None, None


def main():
    M = C.manifest()
    S = json.load(open(os.path.join(C.NEW, "summary.json")))
    checks = []

    def chk(name, reported, recomputed):
        checks.append(dict(check=name, reported=reported, recomputed=recomputed, equal=reported == recomputed))

    new = [r for r in M if r["status"] != "excluded"]
    sz = [r for r in new if r["kind"] == "seizure"]
    ng = [r for r in new if r["kind"] == "nonseizure"]
    chk("manifest rows", 506, len(M))
    chk("seizure clips", S["n_seizure"], len(sz))
    chk("negatives", S["n_negative"], len(ng))
    chk("excluded", S["n_excluded"], sum(r["status"] == "excluded" for r in M))
    chk("status", S["status"], dict(Counter(r["status"] for r in M)))
    chk("by class", S["by_class"], dict(Counter(r["stage"] if r["kind"] == "seizure" else "non-seizure" for r in new)))
    chk("timing decisions", S["timing"], dict(Counter(r["alignment"] for r in new)))
    fl = Counter()
    for r in M:
        for f in (r["label_flags"] or "").split(";"):
            f = f.strip()
            if f:
                fl[f.split("(")[0].strip()] += 1
    chk("label flags", S["flags"], dict(fl))
    chk("cut report: 505 clips", 505, len(new))
    chk("cut report: S2/S3/S4/S5", [67, 235, 24, 1], [sum(r["stage"] == f"Stage {k}" for r in sz) for k in (2, 3, 4, 5)])
    ses = Counter(f"{r['animal']}/{r['session']}" for r in new)
    rep_ses = {"RN235/11-30-2023 to 12-04-2023": 188, "RN235/11-18-2023": 74, "RN235/10-26-2023": 30,
               "RN235/10-20-2023 to 10-22-2023": 15, "RN235/11-20-2023": 14, "RN213/11-30-2023 to 12-04-2023": 92,
               "RN197/10-26-2023": 50, "RN199/10-26-2023": 20, "RN242/10-12-2023": 12,
               "RN216/11-30-2023 to 12-04-2023": 2, "RN216/12-12-2023": 2, "RN219/10-12-2023": 3, "RN227/10-12-2023": 2,
               "RN224/10-12-2023": 1}
    chk("by session", rep_ses, dict(ses))
    reuse = sum(1 for r in sz if r["existing_negative"])
    chk("seizures reusing an existing negative (report: 150; of the 327 cut)", 150, reuse)
    chk("... of the 328 candidates incl. the excluded one", 150,
        sum(1 for r in M if r["kind"] == "seizure" and r["existing_negative"]))
    # alignment numbers
    al = [r for r in new if r["alignment"] == "aligned"]
    offs = [float(r["offset_used_s"]) for r in al]
    chk("offset median (report +0.25)", 0.25, round(statistics.median(offs), 2))
    chk("offset range (report -0.94..+5.23)", [-0.94, 5.23], [round(min(offs), 2), round(max(offs), 2)])
    chk("offset sources", {"measured": 167, "partner": 67, "local_consensus": 236, "file_strong_all_agree": 17},
        dict(Counter(r["offset_source"] for r in al)))
    un = [r for r in new if r["alignment"] == "unmeasured"]
    chk("unmeasured seizure/negative (report 12/6)", [12, 6], [sum(r["kind"] == "seizure" for r in un),
                                                             sum(r["kind"] == "nonseizure" for r in un)])
    chk("unmeasured files (report 4)", sorted(["RN199-RN201.20231030143427", "RN197-RN213.20231202234600",
                                               "RN222-RN235.20231201191300", "RN222-RN235.20231203212800"]),
        sorted({os.path.basename(r["raw_video"])[:-4] for r in un}))
    chk("correction applied 0 everywhere", True, all(float(r["correction_applied_s"]) == 0 for r in new))
    chk("timing cause none everywhere", True, all(r["timing_cause"] == "none" for r in new))
    chk("raw videos used (report 80)", 80, len({r["raw_video"] for r in new}))
    # label checks
    p = [float(r["p_eeg_oof"]) for r in sz if r["p_eeg_oof"]]
    chk("seizures with OOF P(sz) < 0.5 (report 16)", 16, sum(x < 0.5 for x in p))
    ll = [float(r["ll_ratio"]) for r in sz if r["ll_ratio"]]
    chk("LL ratio median (report 1.42)", 1.42, round(statistics.median(ll), 2))
    chk("LL ratio < p10 1.0714 (report 27 ll_ratio_low + 3 clip_rule_both_low)", 30,
        sum(x < S["ll"]["ll_p10"] for x in ll))
    chk("hard flags", {"label_flagged": 3},
        {"label_flagged": sum(r["status"] == "label_flagged" for r in M)})
    # items.csv
    items_new = C.read_csv(os.path.join(C.NEW, "items.csv"))
    chk("items.csv rows", len(new), len(items_new))
    ik = {r["key"]: r for r in items_new}
    bad = [r["key"] for r in new if r["key"] not in ik or int(ik[r["key"]]["y"]) != int(r["y"])
           or ik[r["key"]]["sess"] != f"{r['animal']}/{r['session']}" or ik[r["key"]]["subj"] != r["animal"]]
    chk("items.csv agrees with manifest", [], bad)
    # discover() and splits
    items = discover()
    base = Counter(i[1] for i in items)
    chk("existing discover() items (report 24497)", 24497, len(items))
    chk("existing class counts (report 12357/1459/9419/1068/194)", [12357, 1459, 9419, 1068, 194], [base[k] for k in range(5)])
    add = Counter(int(r["y"]) for r in new)
    after = [base[k] + add[k] for k in range(5)]
    chk("after merge (report 12535/1526/9654/1092/195)", [12535, 1526, 9654, 1092, 195], after)
    sev_b = (base[3] + base[4]) / sum(base[k] for k in range(1, 5))
    sev_a = (after[3] + after[4]) / sum(after[k] for k in range(1, 5))
    chk("severe share of seizures (report 10.40% -> 10.32%)", [10.40, 10.32], [round(100 * sev_b, 2), round(100 * sev_a, 2)])
    chk("3-class mild/severe added (report +302/+25)", [302, 25], [add[1] + add[2], add[3] + add[4]])
    folds = split_subjects(items)
    rep_folds = [["RN213", "RN223", "RN235", "RN237"], ["RN197", "RN208", "RN215", "RN238"],
                 ["RN216", "RN229", "RN242", "RN245"], ["RN199", "RN210", "RN219", "RN224"],
                 ["RN204", "RN222", "RN227", "RN244"]]
    chk("subject folds seed 49", rep_folds, [sorted(g) for g in folds])
    fold_of = {a: k for k, g in enumerate(folds) for a in g}
    chk("new clips per fold as val (report 413/50/16/24/2)", [413, 50, 16, 24, 2],
        [sum(fold_of[r["animal"]] == k for r in new) for k in range(5)])
    f0 = Counter(int(r["y"]) for r in new if fold_of[r["animal"]] == 0)
    chk("fold 0 val adds (report non 172, S2 47, S3 183, S4 11)", [172, 47, 183, 11, 0], [f0[k] for k in range(5)])
    # stored OOF folds: the animals in each stored fold's validation dump (ttg_vsubj x3dbug_dual_s1, val_ep12.npz)
    import numpy as np
    stored = {}
    for k in range(5):
        z = np.load(os.path.join(C.EEG_ROOT, "output", "ttg_vsubj", f"x3dbug_dual_s1_fold{k}", "val_ep12.npz"),
                    allow_pickle=True)
        stored[k] = sorted({re.search(r"Data_(RN\d+)_cropped", str(x)).group(1) for x in z["path"]})
    chk("stored OOF folds (ttg_vsubj x3dbug_dual_s1 val_ep12 paths)", rep_folds, [stored[k] for k in range(5)])
    chk("unmeasured clips on RN197-RN213.20231204185800 (not in the report's 4-file list)", 0,
        sum(1 for r in un if "20231204185800" in r["raw_video"]))
    val_b, seed_b = split_sessions(items)
    items_after = items + [(r["key"], int(r["y"]), f"{r['animal']}/{r['session']}", r["animal"]) for r in new]
    val_a, seed_a = split_sessions(items_after)
    chk("session split seed and identical val sessions (report 49, 121, identical)", [49, 49, 121, True],
        [seed_b, seed_a, len(val_a), val_a == val_b])
    nv = [r for r in new if f"{r['animal']}/{r['session']}" in val_a]
    chk("new clips in seed-49 val sessions (report 70: S3 43, S4 7, S5 1, S2 19)", [70, {1: 19, 2: 43, 3: 7, 4: 1}],
        [len(nv), dict(sorted(Counter(int(r["y"]) for r in nv).items()))])
    chk("val sessions holding new clips (report RN197/10-26, RN199/10-26)", ["RN197/10-26-2023", "RN199/10-26-2023"],
        sorted({f"{r['animal']}/{r['session']}" for r in nv}))
    new_sessions = {f"{r['animal']}/{r['session']}" for r in new} - {i[2] for i in items}
    chk("new sessions (report 0)", 0, len(new_sessions))
    # EEG windows (from the parent's builder, vn_reader.json)
    rd = os.path.join(C.VOUT, "vn_reader.json")
    if os.path.exists(rd):
        R = json.load(open(rd))
        chk("EEG windows added (report 12518)", 12518, R["windows_total"])
        chk("EEG windows by label (report 6488/1468/4163/398/1)", {"0": 6488, "1": 1468, "2": 4163, "3": 398, "4": 1},
            {str(k): v for k, v in R["windows_by_label"].items()})
    # eeg.edf with/without in the existing tree (report 45 no eeg.edf, 25 also no info.txt)
    no_edf = no_info = 0
    for key, y, sess, subj in items:
        d = os.path.join(C.FULL, os.path.dirname(key)[len("data/"):])
        if not os.path.exists(os.path.join(d, "eeg.edf")):
            no_edf += 1
            if not os.path.exists(os.path.join(d, "info.txt")):
                no_info += 1
    chk("existing discover() clips without eeg.edf / also without info.txt (report 45 / 25)", [45, 25], [no_edf, no_info])
    chk("existing with eeg.edf (report 24452)", 24452, len(items) - no_edf)
    # reproduction table
    rp = C.read_csv(os.path.join(C.NEW, "plan", "reproduce_existing.csv"))
    rc = Counter((r["kind"], r["name_reproduced"], r["info_identical"]) for r in rp)
    chk("plan reproduction (report 359/207/10/5; 566 byte-identical)", [359, 207, 10, 5, 566],
        [rc[("nonseizure", "True", "True")], rc[("seizure", "True", "True")], rc[("seizure", "True", "")],
         rc[("nonseizure", "True", "")], sum(1 for r in rp if r["info_identical"] == "True")])
    out = dict(n_checks=len(checks), n_equal=sum(c["equal"] for c in checks),
               not_equal=[c for c in checks if not c["equal"]], checks=checks)
    C.write_json("vn_counts.json", out)
    for c in checks:
        print(("OK  " if c["equal"] else "DIFF"), c["check"], "| reported", c["reported"], "| recomputed", c["recomputed"])


if __name__ == "__main__":
    main()
