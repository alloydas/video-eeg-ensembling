# Independent check of the patched video dumps (ttg_recut/vsubj_patched, vsubj_exclonly) against the stored
# ones (output/ttg_vsubj): the only differences must be the 11 excluded rows (removed) and the 81 re-cut rows
# (probabilities replaced by the CPU fp32 predictions in ttg_recut/infer). Also re-derives the equivalence
# statistics from the stored predictions (not from equivalence.json). numpy only; read-only except the JSON.
import os, csv, json
import numpy as np
EEG = "/work/mech-ai-scratch/alloy/EEG"
R = f"{EEG}/output/ttg_recut"
OUT = f"{R}/verify/indep"
RUNS = [f"x3d{r}_dual_s{s}_fold{f}" for r in ("fix", "bug") for s in (1, 2, 3) for f in range(5)]
ROWF = ("path", "y5", "probs_g3", "probs_g5")


def main():
    man = {r["clip_key"]: r for r in csv.DictReader(open(f"{R}/manifest.csv"))}
    exc = {r["key"]: r for r in csv.DictReader(open(f"{R}/exclude.csv"))}
    assert len(man) == 81 and len(exc) == 11 and not (set(man) & set(exc))
    res = dict(runs={}, problems=[])
    seen_recut = {k: 0 for k in man}; seen_exc = {k: 0 for k in exc}
    agg = dict(ctl=[], orig=[], ctl_argmax=[0, 0], orig_argmax=[0, 0], changed={"g3": 0, "g5": 0}, replaced=0,
               removed=0)
    for run in RUNS:
        zo = np.load(f"{EEG}/output/ttg_vsubj/{run}/val_ep12.npz", allow_pickle=True)
        zp = np.load(f"{R}/vsubj_patched/{run}/val_ep12.npz", allow_pickle=True)
        ze = np.load(f"{R}/vsubj_exclonly/{run}/val_ep12.npz", allow_pickle=True)
        zi = np.load(f"{R}/infer/{run}_fp32.npz", allow_pickle=True)
        pr = []
        # scalar / non-row fields identical, same field list, same dtypes
        if set(zo.files) != set(zp.files) or set(zo.files) != set(ze.files):
            pr.append(f"field sets differ {sorted(zo.files)} {sorted(zp.files)}")
        for k in zo.files:
            for tag, z in (("patched", zp), ("exclonly", ze)):
                if z[k].dtype != zo[k].dtype:
                    pr.append(f"{tag} {k} dtype {z[k].dtype} != {zo[k].dtype}")
                if k not in ROWF and not np.array_equal(z[k], zo[k]):
                    pr.append(f"{tag} scalar field {k} differs")
        po = [str(p) for p in zo["path"]]
        keep = np.array([p not in exc for p in po])
        removed = [p for p in po if p in exc]
        for p in removed:
            seen_exc[p] += 1
        # exclusion-only: exactly the stored rows minus the excluded ones, same order, bit-identical
        for k in ROWF:
            if not np.array_equal(ze[k], zo[k][keep]):
                pr.append(f"exclonly {k} != stored minus excluded")
        # patched: same rows/order as exclonly; path, y5 identical; probs identical except re-cut rows
        pp = [str(p) for p in zp["path"]]
        if pp != [p for p, m in zip(po, keep) if m]:
            pr.append("patched path list != stored minus excluded (order or content)")
        if not np.array_equal(zp["y5"], zo["y5"][keep]):
            pr.append("patched y5 differs")
        rec = np.array([p in man for p in pp])
        for p in np.array(pp)[rec]:
            seen_recut[p] += 1
        ia = {str(k): i for i, k in enumerate(zi["keys_aff"])}
        if set(ia) != set(np.array(pp)[rec].tolist()):
            pr.append(f"infer keys_aff != re-cut rows in this fold ({len(ia)} vs {int(rec.sum())})")
        for t in ("g3", "g5"):
            a, b = zp[f"probs_{t}"], zo[f"probs_{t}"][keep]
            same = np.all(a == b, axis=1)
            if not np.all(same[~rec]):
                pr.append(f"{t}: {int((~same[~rec]).sum())} non-re-cut rows changed")
            nochg = int(same[rec].sum())
            if nochg:
                pr.append(f"{t}: {nochg} re-cut rows NOT replaced")
            # replaced rows == infer aff_new rows exactly
            for j in np.where(rec)[0]:
                if not np.array_equal(a[j], zi[f"aff_new_{t}"][ia[pp[j]]]):
                    pr.append(f"{t}: patched row {pp[j]} != infer aff_new"); break
            agg["changed"][t] += int((a[rec].argmax(1) != b[rec].argmax(1)).sum())
            # equivalence re-derived: stored row vs CPU re-prediction, controls and re-cut clips on their old frames
            pos = {p: i for i, p in enumerate(po)}
            ctl = np.array([pos[str(k)] for k in zi["keys_ctl"]])
            aff = np.array([pos[str(k)] for k in zi["keys_aff"]])
            if not np.array_equal(zi[f"ctl_stored_{t}"], zo[f"probs_{t}"][ctl]):
                pr.append(f"{t}: infer ctl_stored != stored dump rows")
            if not np.array_equal(zi[f"aff_stored_{t}"], zo[f"probs_{t}"][aff]):
                pr.append(f"{t}: infer aff_stored != stored dump rows")
            d1 = np.abs(zi[f"ctl_pred_{t}"] - zo[f"probs_{t}"][ctl]).max()
            d2 = np.abs(zi[f"aff_orig_{t}"] - zo[f"probs_{t}"][aff]).max()
            agg["ctl"].append(float(d1)); agg["orig"].append(float(d2))
            agg["ctl_argmax"][0] += int((zi[f"ctl_pred_{t}"].argmax(1) == zo[f"probs_{t}"][ctl].argmax(1)).sum())
            agg["ctl_argmax"][1] += len(ctl)
            agg["orig_argmax"][0] += int((zi[f"aff_orig_{t}"].argmax(1) == zo[f"probs_{t}"][aff].argmax(1)).sum())
            agg["orig_argmax"][1] += len(aff)
            # control keys must be unaffected clips (not candidates, not excluded)
        cand = {r["key"] for r in csv.DictReader(open(f"{R}/plan/candidates.csv"))}
        bad_ctl = [str(k) for k in zi["keys_ctl"] if str(k) in cand or str(k) in exc]
        if bad_ctl:
            pr.append(f"control clips that are candidates: {bad_ctl}")
        agg["replaced"] += int(rec.sum()); agg["removed"] += len(removed)
        res["runs"][run] = dict(n_stored=len(po), n_patched=len(pp), removed=len(removed), replaced=int(rec.sum()),
                                problems=pr)
        res["problems"] += [f"{run}: {x}" for x in pr]
    res["recut_rows_per_key"] = sorted(set(seen_recut.values()))
    res["excluded_rows_per_key"] = sorted(set(seen_exc.values()))
    res["recut_keys_missing"] = [k for k, v in seen_recut.items() if v != 6]
    res["excluded_keys_missing"] = [k for k, v in seen_exc.items() if v != 6]
    res["totals"] = dict(replaced=agg["replaced"], removed=agg["removed"], argmax_changed=agg["changed"])
    res["equivalence_rederived"] = dict(
        controls_max_abs_dp=max(agg["ctl"]), recut_on_old_frames_max_abs_dp=max(agg["orig"]),
        controls_argmax_agree=agg["ctl_argmax"], recut_old_frames_argmax_agree=agg["orig_argmax"],
        runs_over_0p02=[RUNS[i // 2] + ("/g3" if i % 2 == 0 else "/g5") for i, v in enumerate(agg["orig"]) if v > 0.02])
    # per-fold count of re-cut / excluded rows (x3dfix s1)
    json.dump(res, open(f"{OUT}/dumps.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "runs"}, indent=1))
    print({r: (v["removed"], v["replaced"]) for r, v in res["runs"].items() if r.startswith("x3dfix_dual_s1")})


if __name__ == "__main__":
    main()
