#!/usr/bin/env python3
"""
Stage 2 of TT-X3D (CPU): linear probes on frozen X3D-M features and the PRE-REGISTERED
kill test for the dense (native-rate) pathway.

PURPOSE
  1. Probes. For each input of grader/probe_extract.py -- sparse (historical 16 linspace
     frames), dense (mean of 8 native-rate stride-2 snippets) and shuffled (the same
     snippet frames with their temporal order permuted: appearance kept, motion destroyed)
     -- fit a standardised, class-balanced, L2 logistic regression on the TRAIN clips of
     train_pooled.split_sessions(seed 49) for
        (i)  severe vs mild  (g3 2 vs 1, seizure clips only),
        (ii) Stage 4 vs Stage 3.
     C is chosen by 5-fold cross-validation over TRAINING ANIMALS (folds are whole
     animals, balanced by clip count), maximising the mean held-out AUROC. The val set is
     never used for any choice. Val clips are then scored once. Reported per input: val
     pooled AUROC and within-session AUROC (pairs inside sessions that hold both classes),
     each with an animal-clustered bootstrap CI (2,000 reps), plus paired CIs for
     dense - shuffled and dense - sparse.

  2. Kill test (pre-registered; val seizure clips, severe vs mild). Leave-one-animal-out
     logistic stacks, fit exactly like the temporal-sampling analysis (fit_predict: median
     impute, standardise, class-balanced L2 with lam=1):
        base (2 inputs) = [top-3 x 5 ensemble severe logit log(p_severe / p_mild)
                           (decision-headroom/ens_g3_top3x5.npz -- a SELECTED MEMBER SET:
                           members chosen on this val set and best-epoch selected, so
                           absolute stack AUROCs are upper bounds; the output JSON carries
                           this label. The delta is valid: both stacks share it),
                           native-motion score]
        plus (3 inputs) = base + dense probe score
     The native-motion score is the temporal-sampling "native" model: the 20 native-rate
     columns of temporal-sampling/out/features.pkl (every key starting with n_ except
     n_decoded / n_header -- listed in the output JSON), trained on TRAIN-session seizure
     clips (severe = stage >= 4) and scored on val. The same plus-stack is built with the
     sparse and shuffled probe scores as controls. Delta within-session AUROC (plus - base)
     gets a paired animal-bootstrap CI (the same animal resamples for both stacks).
     DECISION: KILL the dense work if
         delta_within(dense) < +0.005 AND its upper 95% bound < +0.015,
       OR the dense probe's val within-session AUROC is not above the shuffled probe's;
     otherwise GO.

  Alignment: every source is joined on the clip path with '/video.mp4' stripped, and labels
  are asserted to agree across the probe features, features.pkl and the ensemble file.
  Clips missing from any source, or with NaN probe features, are dropped and COUNTED.

OUTPUT  <out> (default $EEG_ROOT/output/ttg_probe/probe_results.json)

  Inputs are data under EEG_ROOT: the probe features, and the reference files in
  output/ttg_ref/decision-headroom (ens_g3_top3x5.npz, val_paths_split49.npy) and
  output/ttg_ref/temporal-sampling/out/features.pkl. Nothing is imported from output/ttg_ref.

USAGE (from the video-eeg-ensembling repo, any cwd; PYTHONDONTWRITEBYTECODE=1; numpy/scipy
       only, no GPU; EEG_ROOT env var, default /work/mech-ai-scratch/alloy/EEG. The script
       chdirs to EEG_ROOT, so relative --features / --ens / --motion / --out are EEG_ROOT's.)
  python grader/probe_analysis.py
  python grader/probe_analysis.py --features <npz> --out <json> --dry_run    # 200 reps, 5 C values
  --limit N subsamples N training clips per probe (seeded) for quick tests.
"""
import argparse
import os
import pickle
import random
import sys
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ttg_common as C                                               # noqa: E402  (puts EEG_ROOT on sys.path)

REF = os.path.join(C.EEG_ROOT, "output", "ttg_ref")
DEF_FEAT = os.path.join(C.EEG_ROOT, "output", "ttg_probe", "features.npz")
DEF_OUT = os.path.join(C.EEG_ROOT, "output", "ttg_probe", "probe_results.json")
DEF_ENS = os.path.join(REF, "decision-headroom", "ens_g3_top3x5.npz")
DEF_MOTION = os.path.join(REF, "temporal-sampling", "out", "features.pkl")
INPUTS = {"sparse": "sparse", "dense": "dense_mean", "shuffled": "shuffled_mean"}
ENS_LABEL = ("SELECTED MEMBER SET: top-3 backbones x 5 seeds (ens_g3_top3x5), chosen on this "
             "val set, each member a best-epoch (val-selected) checkpoint. Absolute AUROCs of "
             "every stack that contains it are upper bounds. The plus-minus-base deltas are "
             "valid comparisons because both stacks contain the same ensemble score.")
NATIVE_EXPECTED = ['n_area_range', 'n_burst1s', 'n_cy_hf_frac', 'n_cy_jump02', 'n_cy_range',
                   'n_me_centroid', 'n_me_hf_frac', 'n_me_lf_frac', 'n_me_mf_frac', 'n_me_p95_rel',
                   'n_me_peak_f', 'n_me_peak_prom', 'n_me_rel', 'n_me_slope', 'n_metop_rel',
                   'n_pc1_hf_frac', 'n_pc1_mf_frac', 'n_pc1_peak_f', 'n_pc1_peak_prom', 'n_rough']


# ----------------------------------------------------------------------------- models

def fit_logreg(X, y, Cr, w0=None, maxiter=2000, return_info=False):
    """sklearn-equivalent LogisticRegression(C=Cr, penalty='l2', class_weight='balanced'):
    minimise Cr * sum_i s_i * logloss_i + 0.5 * ||w||^2 (intercept unpenalised).

    return_info=True (added for the Step 0 analysis, backward compatible: the default path and
    its numbers are unchanged) also returns dict(nit, success, grad_inf, maxiter): the L-BFGS-B
    iteration count, its success flag and the inf-norm of the objective's gradient at the
    returned solution."""
    n, d = X.shape
    sw = np.where(y > 0, n / (2.0 * max(y.sum(), 1)), n / (2.0 * max((1 - y).sum(), 1)))

    def f(w):
        z = X @ w[:-1] + w[-1]
        loss = Cr * np.sum(sw * (np.logaddexp(0, z) - y * z)) + 0.5 * w[:-1] @ w[:-1]
        gz = Cr * sw * (expit(z) - y)
        return loss, np.r_[X.T @ gz + w[:-1], gz.sum()]

    res = minimize(f, np.zeros(d + 1) if w0 is None else w0, jac=True, method="L-BFGS-B",
                   options=dict(maxiter=maxiter))
    w = res.x
    if return_info:
        return w, dict(nit=int(res.nit), success=bool(res.success), maxiter=int(maxiter),
                       grad_inf=float(np.max(np.abs(f(w)[1]))))
    return w


def standardise(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    return (Xtr - mu) / sd, (Xte - mu) / sd


def animal_folds(animals, k=5):
    """Whole-animal folds balanced by clip count (largest first -> emptiest fold)."""
    ua, cnt = np.unique(animals, return_counts=True)
    load, fold_of = np.zeros(k), {}
    for i in np.argsort(-cnt, kind="stable"):
        f = int(np.argmin(load))
        fold_of[ua[i]] = f
        load[f] += cnt[i]
    return np.array([fold_of[a] for a in animals])


def probe(Xtr, ytr, atr, Xva, grid):
    """C by 5-fold CV over training animals (mean held-out AUROC); then refit on all train."""
    folds = animal_folds(atr)
    cv = {}
    for f in range(5):
        tr, te = folds != f, folds == f
        A, B = standardise(Xtr[tr], Xtr[te])
        w = None
        for Cr in grid:                                  # increasing C, warm-started
            w = fit_logreg(A, ytr[tr], Cr, w0=w)
            cv.setdefault(Cr, []).append(C.auroc(B @ w[:-1] + w[-1], ytr[te] > 0))
    mean = {Cr: float(np.nanmean(v)) for Cr, v in cv.items()}
    best = max(grid, key=lambda Cr: (round(mean[Cr], 6), -Cr))  # ties -> stronger penalty
    A, B = standardise(Xtr, Xva)
    w = fit_logreg(A, ytr, best)
    return B @ w[:-1] + w[-1], dict(C=best, cv_auroc={f"{k:g}": round(v, 4) for k, v in mean.items()},
                                    fold_sizes=np.bincount(folds, minlength=5).tolist())


def fit_predict(Xtr, ytr, Xte, lam=1.0):
    """temporal-sampling/motion_analysis.py fit_predict, verbatim in behaviour: median
    impute, standardise, class-balanced logistic with an L2 penalty lam on the weights."""
    med = np.nanmedian(Xtr, 0)
    Xtr = np.where(np.isnan(Xtr), med, Xtr)
    Xte = np.where(np.isnan(Xte), med, Xte)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    Xtr = np.c_[np.ones(len(Xtr)), Xtr]
    Xte = np.c_[np.ones(len(Xte)), Xte]
    wpos = 0.5 / max(ytr.mean(), 1e-6)
    wneg = 0.5 / max(1 - ytr.mean(), 1e-6)
    sw = np.where(ytr > 0, wpos, wneg)

    def f(w):
        z = Xtr @ w
        l = np.sum(sw * (np.logaddexp(0, z) - ytr * z)) + lam * w[1:] @ w[1:]
        g = Xtr.T @ (sw * (1 / (1 + np.exp(-z)) - ytr))
        g[1:] += 2 * lam * w[1:]
        return l, g

    w = minimize(f, np.zeros(Xtr.shape[1]), jac=True, method="L-BFGS-B").x
    return Xte @ w


def loao(X, y, animals):
    out = np.zeros(len(y))
    for q in np.unique(animals):
        t = animals != q
        out[~t] = fit_predict(X[t], y[t], X[~t])
    return out


def auc_summary(score, pos, sess, an, ua, picks):
    d = C.boot_auc(score, pos, sess, an, ua, picks)
    return dict(pooled=round(C.auroc(score, pos), 4), ci_pooled=np.round(C.ci95(d[:, 0]), 4).tolist(),
                within=round(C.within_auroc(score, pos, sess, an), 4),
                ci_within=np.round(C.ci95(d[:, 1]), 4).tolist()), d


def delta(da, db, pa, pb):
    D = da - db
    return dict(pooled=round(pa["pooled"] - pb["pooled"], 4), ci_pooled=np.round(C.ci95(D[:, 0]), 4).tolist(),
                within=round(pa["within"] - pb["within"], 4), ci_within=np.round(C.ci95(D[:, 1]), 4).tolist())


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default=DEF_FEAT)
    ap.add_argument("--ens", default=DEF_ENS)
    ap.add_argument("--motion", default=DEF_MOTION)
    ap.add_argument("--out", default=DEF_OUT)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--grid", default="1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1,3,10",
                    help="C values for the animal-fold CV")
    ap.add_argument("--limit", type=int, default=0, help="tests: N training clips per probe (seeded)")
    ap.add_argument("--dry_run", action="store_true", help="tests: 200 reps, 5 C values")
    a = ap.parse_args()
    C.enter_eeg_root()                                   # relative paths below are EEG_ROOT's
    out = C.check_output_dir(os.path.dirname(os.path.abspath(a.out)))
    grid = sorted(float(g) for g in a.grid.split(","))
    if a.dry_run:
        a.reps = min(a.reps, 200)
        grid = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    t0 = time.time()
    res = dict(created=time.strftime("%F %T"), features=a.features, ens=a.ens, ens_label=ENS_LABEL,
               motion=a.motion,
               reps=a.reps, grid=grid, dry_run=a.dry_run, limit=a.limit)

    # ---- split (session-disjoint, seed 49, train_pooled's own function)
    import train_pooled as tp
    items = C.items_from_f32_index()
    tr, va, valsess, sseed = tp.split_sessions(items, 49)
    val_keys = {C.key_of(i[0]) for i in va}
    dh = {C.key_of(p) for p in np.load(os.path.join(REF, "decision-headroom", "val_paths_split49.npy"),
                                       allow_pickle=True)}
    assert len(va) == 5289 and val_keys == dh, "seed-49 val set does not match decision-headroom"
    res["split"] = dict(seed=sseed, n_train=len(tr), n_val=len(va), n_val_sessions=len(valsess))

    # ---- probe features
    Z = np.load(a.features, allow_pickle=False)
    keys = np.array([C.key_of(p) for p in Z["path"]])
    y5 = Z["y5"].astype(int)
    lab = np.array([C.y5_of(p) for p in Z["path"]])
    assert np.array_equal(lab, y5), "features.npz y5 disagrees with the labels in the paths"
    X = {k: Z[v].astype(np.float64) for k, v in INPUTS.items()}
    ok = np.all([np.isfinite(X[k]).all(1) for k in X], axis=0)
    isval = np.array([k in val_keys for k in keys])
    an = np.array([C.animal_of(p) for p in keys])
    se = np.array([C.session_of(p) for p in keys])
    res["features_file"] = dict(n=len(keys), n_nan_rows_dropped=int((~ok).sum()),
                                n_failed_listed=int(len(Z["failed"])) if "failed" in Z.files else None,
                                n_val=int(isval.sum()), n_val_expected=5289,
                                feature_dim=int(X["dense"].shape[1]))
    print(f"features: {len(keys)} clips, {int((~ok).sum())} NaN rows dropped, {int(isval.sum())} val", flush=True)

    # ---- (1) probes
    contrasts = {"severe_vs_mild": (lambda v: np.isin(v, [3, 4]), lambda v: np.isin(v, [1, 2])),
                 "S4_vs_S3": (lambda v: v == 3, lambda v: v == 2)}
    res["probes"] = {}
    val_scores = {}
    for cname, (fpos, fneg) in contrasts.items():
        m = ok & (fpos(y5) | fneg(y5))
        trm, vam = m & ~isval, m & isval
        if a.limit:
            idx = np.flatnonzero(trm)
            keep = np.array(sorted(random.Random(0).sample(list(idx), min(a.limit, len(idx)))))
            trm = np.zeros_like(trm)
            trm[keep] = True
        ytr = fpos(y5[trm]).astype(float)
        pos_va = fpos(y5[vam])
        ua, picks = C.animal_picks(an[vam], a.reps, seed=11)
        block = dict(n_train_pos=int(ytr.sum()), n_train_neg=int((1 - ytr).sum()),
                     n_train_animals=int(len(np.unique(an[trm]))), n_val_pos=int(pos_va.sum()),
                     n_val_neg=int((~pos_va).sum()), n_val_animals=int(len(ua)), inputs={})
        draws = {}
        for inp in INPUTS:
            t1 = time.time()
            sc, info = probe(X[inp][trm], ytr, an[trm], X[inp][vam], grid)
            summ, draws[inp] = auc_summary(sc, pos_va, se[vam], an[vam], ua, picks)
            block["inputs"][inp] = dict(**summ, **info)
            val_scores[(cname, inp)] = dict(zip(keys[vam], sc))
            print(f"[{cname}] {inp:8s} C={info['C']:g}  val pooled {summ['pooled']:.4f} {summ['ci_pooled']}  "
                  f"within {summ['within']:.4f} {summ['ci_within']}  ({time.time() - t1:.0f}s)", flush=True)
        P = block["inputs"]
        block["dense_minus_shuffled"] = delta(draws["dense"], draws["shuffled"], P["dense"], P["shuffled"])
        block["dense_minus_sparse"] = delta(draws["dense"], draws["sparse"], P["dense"], P["sparse"])
        res["probes"][cname] = block

    # ---- (2) kill test on val seizure clips, severe vs mild
    E = np.load(a.ens, allow_pickle=True)
    ekey = np.array([C.key_of(k) for k in E["keys"]])
    epos = {k: i for i, k in enumerate(ekey)}
    Pe = E["probs"].astype(np.float64)
    lg_all = np.log(np.clip(Pe[:, 2], 1e-9, None)) - np.log(np.clip(Pe[:, 1], 1e-9, None))
    rows = pickle.load(open(a.motion, "rb"))
    mkeys = sorted({k for r in rows[:500] for k in r if k.startswith(("n_", "s16", "s32", "s64", "k2_", "k4_",
                    "k8_", "k16_")) and not k.endswith("_n_in") and k not in ("n_decoded", "n_header")})
    NATIVE = [k for k in mkeys if k.startswith("n_")]
    assert NATIVE == NATIVE_EXPECTED, f"native column set changed: {NATIVE}"
    mk = np.array([C.key_of(r["path"]) for r in rows])
    mst = np.array([r["stage"] for r in rows])
    mval = np.array([bool(r["val"]) for r in rows])
    assert all((k in val_keys) == v for k, v in zip(mk, mval)), "features.pkl val flag != seed-49 split"
    MX = np.array([[r.get(c, np.nan) for c in NATIVE] for r in rows], float)
    trn = (~mval) & (mst > 0)
    van = mval & (mst > 0)
    nat_val = dict(zip(mk[van], fit_predict(MX[trn], (mst[trn] >= 4).astype(float), MX[van])))

    probe_val = {inp: val_scores[("severe_vs_mild", inp)] for inp in INPUTS}
    val_sz = sorted(set(keys[isval & (y5 > 0)]))          # incl. NaN rows, so they are counted
    have = [k for k in val_sz if k in epos and k in nat_val and all(k in probe_val[i] for i in INPUTS)]
    kidx = np.array([epos[k] for k in have])
    ky5 = np.array([C.y5_of(k + "/video.mp4") for k in have])
    # label agreement across the three sources
    assert np.array_equal(E["y5"][kidx], ky5) and np.array_equal(E["y"][kidx], C.GROUP3[ky5])
    mstage = dict(zip(mk, mst))
    assert all(mstage[k] == v + 1 for k, v in zip(have, ky5)), "features.pkl stage != path label"
    y = (ky5 >= 3).astype(float)
    kan = np.array([C.animal_of(k) for k in have])
    kse = np.array([C.session_of(k) for k in have])
    lg = lg_all[kidx]
    nat = np.array([nat_val[k] for k in have])
    ps = {i: np.array([probe_val[i][k] for k in have]) for i in INPUTS}
    res["kill_test_alignment"] = dict(
        n_val_seizure_in_features=len(val_sz), n_used=len(have),
        dropped_missing_ensemble=int(sum(k not in epos for k in val_sz)),
        dropped_missing_motion=int(sum(k not in nat_val for k in val_sz)),
        dropped_missing_or_nan_probe=int(sum(not all(k in probe_val[i] for i in INPUTS) for k in val_sz)),
        n_severe=int(y.sum()), n_mild=int((1 - y).sum()), n_animals=int(len(np.unique(kan))),
        ensemble_members=int(len(E["members"])),
        ensemble_label=ENS_LABEL, ensemble_member_list=[str(m) for m in E["members"]],
        native_columns=NATIVE, native_model="temporal-sampling fit_predict (median impute, standardise, "
        "class-balanced L2 lam=1) trained on train-session seizure clips of features.pkl, severe = stage>=4",
        n_native_train=int(trn.sum()), labels_agree=True)
    print(f"kill test: {len(have)} val seizure clips ({int(y.sum())} severe / {int((1 - y).sum())} mild, "
          f"{len(np.unique(kan))} animals); dropped {len(val_sz) - len(have)}", flush=True)

    ua, picks = C.animal_picks(kan, a.reps, seed=21)
    pos = y > 0
    stacks = {"ens_only": lg[:, None], "base_ens+native": np.c_[lg, nat]}
    for i in INPUTS:
        stacks[f"plus_{i}"] = np.c_[lg, nat, ps[i]]
    K = {}
    D = {}
    for nm, Xs in stacks.items():
        K[nm], D[nm] = auc_summary(loao(Xs, y, kan), pos, kse, kan, ua, picks)
    for i in INPUTS:
        K[f"{i}_probe_alone_on_kill_set"], _ = auc_summary(ps[i], pos, kse, kan, ua, picks)
        K[f"delta_plus_{i}_minus_base"] = delta(D[f"plus_{i}"], D["base_ens+native"], K[f"plus_{i}"],
                                                 K["base_ens+native"])
    res["kill_test"] = K

    dd = K["delta_plus_dense_minus_base"]
    psev = res["probes"]["severe_vs_mild"]["inputs"]
    w_dense, w_shuf = psev["dense"]["within"], psev["shuffled"]["within"]
    crit1 = dd["within"] < 0.005 and dd["ci_within"][1] < 0.015
    crit2 = not (w_dense > w_shuf)
    decision = "KILL" if (crit1 or crit2) else "GO"
    res["decision"] = dict(
        decision=decision,
        rule="KILL if (delta_within(dense) < +0.005 AND its upper CI < +0.015) OR dense probe "
             "within-session AUROC <= shuffled probe's; else GO",
        delta_within_dense=dd["within"], delta_within_dense_ci=dd["ci_within"],
        criterion_1_small_delta=bool(crit1),
        dense_probe_within=w_dense, shuffled_probe_within=w_shuf,
        criterion_2_not_above_shuffled=bool(crit2),
        controls=dict(sparse=K["delta_plus_sparse_minus_base"]["within"],
                      shuffled=K["delta_plus_shuffled_minus_base"]["within"]),
        caveat=("smoke test: synthetic/limited inputs -- the decision is meaningless"
                if (a.dry_run or a.limit) else "full run"),
        base_ensemble_label=ENS_LABEL)
    res["runtime_s"] = round(time.time() - t0, 1)
    C.atomic_json(a.out, res)
    print(f"\nDECISION: {decision}  (delta_within dense {dd['within']:+.4f} CI {dd['ci_within']}; "
          f"dense probe within {w_dense:.4f} vs shuffled {w_shuf:.4f})")
    print(f"controls: sparse {K['delta_plus_sparse_minus_base']['within']:+.4f} "
          f"{K['delta_plus_sparse_minus_base']['ci_within']}, shuffled "
          f"{K['delta_plus_shuffled_minus_base']['within']:+.4f} {K['delta_plus_shuffled_minus_base']['ci_within']}")
    print("note: the base ensemble ens_g3_top3x5 is a SELECTED member set (val-selected members, "
          "best-epoch checkpoints): absolute stack AUROCs are upper bounds; the deltas are valid")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
