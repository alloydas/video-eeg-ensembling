#!/usr/bin/env python3
"""Independent verification of the scan's S0 gates and the step-A table (read-only; writes only to
$EEG_ROOT/output/ttg_scan/verify/).

1. Builds the protocol-B OOF pairing from the stored dumps with its own code (patched x3dfix val_ep12 + tcn_bin
   val_clip_ep30, aligned on the clip path, step-1 exclusions removed), derives each animal's fold from the dumps,
   and refits G_all / G_-P / the 20-animal LOAO gate for all 9 (video seed, EEG seed) pairs and both tasks.
   Compares: gates.json coefficients, rescore/summary.json per-pair hits + macro-F1, the OOF reference.
2. Recomputes every step-A number (argmax agreement, median / p95 |dp|, video-flag and EGRG-flag agreement,
   macro-F1, R arm) from stepA/preds.npz and the stored dumps, and the registered decision.
"""
import csv
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix

E = "/work/mech-ai-scratch/alloy/EEG/output"
SCAN = f"{E}/ttg_scan"
OUT = f"{SCAN}/verify"
os.makedirs(OUT, exist_ok=True)
SEEDS, FOLDS = (1, 2, 3), (0, 1, 2, 3, 4)
PARTNERS = ("RN199", "RN204", "RN242")


def animal(p):
    return re.search(r"Data_(RN\d+)_cropped", p).group(1)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def feats(Pv, pe):
    return np.c_[logit(1 - Pv[:, 0]), logit(pe)]


def qgate(coef, Pv, pe):
    X = feats(Pv, pe)
    return 1 / (1 + np.exp(-(X @ np.array(coef[:2]) + coef[2])))


def decide(Pv, q):
    sz = Pv[:, 1:] / np.clip(Pv[:, 1:].sum(1, keepdims=True), 1e-12, None)
    return np.c_[1 - q, q[:, None] * sz].argmax(1)


excl = {r["key"] for r in csv.DictReader(open(f"{E}/ttg_recut/exclude.csv"))}
# ---- video dumps (patched), EEG dumps
V = {t: defaultdict(dict) for t in ("g3", "g5")}   # t -> seed -> path -> probs
Y5, FOLD_OF_CLIP = {}, {}
for f in FOLDS:
    for s in SEEDS:
        z = np.load(f"{E}/ttg_recut/vsubj_patched/x3dfix_dual_s{s}_fold{f}/val_ep12.npz", allow_pickle=True)
        assert int(z["epoch"]) == 12 and int(z["fold"]) == f and int(z["seed"]) == s
        for i, p in enumerate(z["path"]):
            p = str(p)
            if p in excl:
                continue
            Y5[p] = int(z["y5"][i])
            assert FOLD_OF_CLIP.setdefault(p, f) == f
            for t in ("g3", "g5"):
                V[t][s][p] = z[f"probs_{t}"][i].astype(np.float64)
EG = defaultdict(dict)
for f in FOLDS:
    for s in SEEDS:
        z = np.load(f"{E}/ttg_eeg/subject/tcn_bin_fold{f}_s{s}/val_clip_ep30.npz", allow_pickle=True)
        assert int(z["epoch"]) == 30 and int(z["fold"]) == f
        for p, v in zip(z["path"], z["p_logmean"]):
            EG[s][str(p) + "/video.mp4"] = float(v)
vpaths = set(V["g5"][1])
for s in SEEDS:
    assert set(V["g5"][s]) == vpaths
epaths = set(EG[1])
paths = sorted(vpaths & epaths)
an = np.array([animal(p) for p in paths])
y5 = np.array([Y5[p] for p in paths])
ys = (y5 > 0).astype(int)
fold_of_animal = {}
for p in paths:
    a_ = animal(p)
    assert fold_of_animal.setdefault(a_, FOLD_OF_CLIP[p]) == FOLD_OF_CLIP[p]
res = dict(pairing=dict(n_video=len(vpaths), n_eeg=len(epaths), n_scored=len(paths),
                        n_nonseizure=int((y5 == 0).sum()), class_counts=np.bincount(y5, minlength=5).tolist(),
                        animals=len(set(an))),
           fold_of_animal=fold_of_animal)
gj = json.load(open(f"{SCAN}/s0/gates.json"))
summ = json.load(open(f"{E}/ttg_recut/rescore/summary.json"))
res["fold_match_gates_json"] = all(fold_of_animal[a_] == int(f) for f, L in gj["folds"].items() for a_ in L)
res["partner_folds"] = {p: fold_of_animal[p] for p in PARTNERS}
cmp_rows, maxdiff, repro_ok, dec_ok = [], 0.0, True, True
oof = defaultdict(list)
for t in ("g3", "g5"):
    K = 3 if t == "g3" else 5
    yt = y5 if t == "g5" else np.array([0, 1, 1, 2, 2])[y5]
    for vs in SEEDS:
        Pv = np.stack([V[t][vs][p] for p in paths])
        for es in SEEDS:
            pe = np.array([EG[es][p] for p in paths])
            X = feats(Pv, pe)
            key = f"{t}|x3dfix_dual_s{vs}_foldF|tcn_bin_foldF_s{es}"
            G = gj["gates"][key]
            mine = {}
            m = LogisticRegression(C=1.0, max_iter=5000).fit(X, ys)
            mine["all"] = np.r_[m.coef_[0], m.intercept_]
            for pa in PARTNERS:
                tr = an != pa
                m = LogisticRegression(C=1.0, max_iter=5000).fit(X[tr], ys[tr])
                mine[f"minus_{pa}"] = np.r_[m.coef_[0], m.intercept_]
            for k in mine:
                d = float(np.abs(mine[k] - np.array(G[k])).max())
                maxdiff = max(maxdiff, d)
            # LOAO over all 20 animals; the gate for partner P's clips must be G_-P
            q = np.empty(len(paths))
            for g in np.unique(an):
                tr = an != g
                m = LogisticRegression(C=1.0, max_iter=5000).fit(X[tr], ys[tr])
                q[~tr] = m.predict_proba(X[~tr])[:, 1]
                if g in PARTNERS:
                    qq = qgate(G[f"minus_{g}"], Pv[~tr], pe[~tr])
                    dec_ok &= bool(np.array_equal(decide(Pv[~tr], qq), decide(Pv[~tr], q[~tr])))
            pred = decide(Pv, q)
            cm = confusion_matrix(yt, pred, labels=range(K))
            hits = [int(cm[i, i]) for i in range(K)]
            f1 = f1_score(yt, pred, average="macro", labels=range(K))
            vpred = Pv.argmax(1)
            vcm = confusion_matrix(yt, vpred, labels=range(K))
            vhits = [int(vcm[i, i]) for i in range(K)]
            vf1 = f1_score(yt, vpred, average="macro", labels=range(K))
            ref = summ["recipes"]["x3dfix"][t]["patched"]["pairs"][f"x3dfix_dual_s{vs}_foldF x tcn_bin_foldF_s{es}"]
            same = hits == ref["egrg_hits"] and abs(f1 - ref["egrg_f1"]) < 1e-12 and vhits == ref["video_hits"] \
                and abs(vf1 - ref["video_f1"]) < 1e-12
            repro_ok &= same
            ns = y5 == 0
            oof[t].append(dict(egrg=int((pred[ns] > 0).sum()), video=int((vpred[ns] > 0).sum()),
                               eeg=int((pe[ns] > 0.5).sum())))
            cmp_rows.append(dict(task=t, vs=vs, es=es, egrg_hits=hits, egrg_f1=f1, ref_hits=ref["egrg_hits"],
                                 ref_f1=ref["egrg_f1"], identical=same))
res["gates_max_abs_coef_diff_vs_gates_json"] = maxdiff
res["loao_reproduces_summary_all_18"] = bool(repro_ok)
res["partner_loao_gate_equals_G_minus_P_decisions"] = bool(dec_ok)
res["repro_rows"] = cmp_rows
nn = int((y5 == 0).sum())
res["oof_reference"] = {t: dict(egrg_mean=float(np.mean([r["egrg"] for r in oof[t]])),
                                video_mean=float(np.mean([r["video"] for r in oof[t]])),
                                eeg_mean=float(np.mean([r["eeg"] for r in oof[t]])),
                                egrg_per_h=float(np.mean([r["egrg"] for r in oof[t]])) / nn * 60,
                                video_per_h=float(np.mean([r["video"] for r in oof[t]])) / nn * 60,
                                eeg_per_h=float(np.mean([r["eeg"] for r in oof[t]])) / nn * 60) for t in oof}

# ---- step A table from preds.npz
it = json.load(open(f"{SCAN}/stepA/items.json"))
items = it["items"]
P = np.load(f"{SCAN}/stepA/preds.npz")
ky = [x["key"] for x in items]
ya = np.array([Y5[k] for k in ky])
assert np.array_equal(ya, np.array([x["y5"] for x in items]))
fold_items_ok = all(fold_of_animal[animal(k)] == x["fold"] for k, x in zip(ky, items))
sample = dict(n=len(items), strata=np.bincount(ya, minlength=5).tolist(), animals=len({animal(k) for k in ky}),
              partners={p: sum(animal(k) == p for k in ky) for p in PARTNERS}, fold_items_ok=bool(fold_items_ok),
              unique=len(set(ky)) == len(ky), in_exclusions=sum(k in excl for k in ky),
              dur_ok=all(50 <= x["dur"] <= 75 for x in items))
A = {}
for arm in "CKE":
    for t in ("g3", "g5"):
        K = 3 if t == "g3" else 5
        yt = ya if t == "g5" else np.array([0, 1, 1, 2, 2])[ya]
        d, ag, vfa, f1s, dsz = [], [], [], [], []
        fl = dict(ns_st=0, ns_arm=0, ns_01=0, ns_10=0, sz_st=0, sz_arm=0, agree=0, n=0)
        for vs in SEEDS:
            pa = P[f"{arm}|{t}|s{vs}"].astype(np.float64)
            st = np.stack([V[t][vs][k] for k in ky])
            d.append(np.abs(pa - st).max(1))
            ag.append(pa.argmax(1) == st.argmax(1))
            vfa.append((pa.argmax(1) > 0) == (st.argmax(1) > 0))
            dsz.append((1 - pa[:, 0]) - (1 - st[:, 0]))
            f1s.append(f1_score(yt, pa.argmax(1), average="macro", labels=sorted(set(yt) | set(pa.argmax(1)))))
            for es in SEEDS:
                pe = np.array([EG[es][k] for k in ky])
                G = gj["gates"][f"{t}|x3dfix_dual_s{vs}_foldF|tcn_bin_foldF_s{es}"]["all"]
                fa = decide(pa, qgate(G, pa, pe)) > 0
                fs = decide(st, qgate(G, st, pe)) > 0
                ns = ya == 0
                fl["ns_st"] += int(fs[ns].sum()); fl["ns_arm"] += int(fa[ns].sum())
                fl["ns_01"] += int((~fs[ns] & fa[ns]).sum()); fl["ns_10"] += int((fs[ns] & ~fa[ns]).sum())
                fl["sz_st"] += int(fs[~ns].sum()); fl["sz_arm"] += int(fa[~ns].sum())
                fl["agree"] += int((fa == fs).sum()); fl["n"] += len(fa)
        d, ag, vfa = map(np.concatenate, (d, ag, vfa))
        A[f"{arm}|{t}"] = dict(rows=len(d), argmax=float(ag.mean()), median_dp=float(np.median(d)),
                               p95_dp=float(np.percentile(d, 95)), mean_dPsz=float(np.concatenate(dsz).mean()),
                               video_flag_agree=float(vfa.mean()), egrg_flag_agree=fl["agree"] / fl["n"],
                               egrg_flags=fl, macro_f1=f1s,
                               passes=bool(ag.mean() >= 0.97 and np.median(d) <= 0.05))
# R arm
dR, agR = [], []
for s in SEEDS:
    pr = P[f"R|s{s}"]
    ps = np.array([EG[s][k] for k in ky])
    dR.append(np.abs(pr - ps))
    agR.append((pr > 0.5) == (ps > 0.5))
dR, agR = np.concatenate(dR), np.concatenate(agR)
A["R"] = dict(rows=len(dR), argmax=float(agR.mean()), median_dp=float(np.median(dR)), max_dp=float(dR.max()),
              passes=bool(agR.mean() >= 0.99 and np.median(dR) <= 0.01))
Kp = A["K|g3"]["passes"] and A["K|g5"]["passes"]
Ep = A["E|g3"]["passes"] and A["E|g5"]["passes"]
cp = A["C|g3"]["median_dp"] <= 0.01 and A["C|g5"]["median_dp"] <= 0.01
decision = ("STOP C" if not cp else "STOP R" if not A["R"]["passes"] else "K" if Kp else "E" if Ep else
            "STOP: K and E both fail")
res["stepA_sample"] = sample
res["stepA"] = A
res["stepA_decision_recomputed"] = decision
dj = json.load(open(f"{SCAN}/stepA/decision.json"))
res["stepA_decision_file"] = dj["decision"]
diffs = {}
for arm in "CKE":
    for t in ("g3", "g5"):
        r = dj[arm][t]
        a = A[f"{arm}|{t}"]
        diffs[f"{arm}|{t}"] = dict(argmax=a["argmax"] - r["argmax_agree"], median=a["median_dp"] - r["median_abs_dp"],
                                   p95=a["p95_dp"] - r["p95_abs_dp"], egrg=a["egrg_flag_agree"] - r["egrg_flag_agree"],
                                   f1=float(np.max(np.abs(np.array(a["macro_f1"]) - np.array(r["macro_f1_by_seed"])))))
res["stepA_diff_vs_decision_json"] = diffs
json.dump(res, open(f"{OUT}/vs_gates_stepa.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in res.items() if k not in ("repro_rows", "stepA")}, indent=1, default=float))
for k, v in A.items():
    print(k, {kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in v.items()})
