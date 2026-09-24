#!/usr/bin/env python3
"""
CPU tests of grader/eeg/joint_gate.py, including the loaders that NEW runs will go through.

PURPOSE
  joint_gate.py --regression proves the stored-posterior path reproduces the validated
  prototype. New runs arrive through different code: train_grader.py dumps (val_ep{E}.npz with
  path / y5 / probs_g3 / probs_g5) and train_eeg_det.py dumps (val_clip_ep{E}.npz with path /
  y5 / p_logmean / packed windows). This script tests those paths:

  regression   joint_gate.py --regression (46 checks against j10_argmax_rule.py).
  synthetic    Full-size dumps in the EXACT train_grader / train_eeg_det formats are written
               from stored posteriors (float64 kept), and joint_gate must give the SAME numbers
               through the run-directory loaders as through the stored loaders:
                 A  two v3_vidseeds MViT seeds as dual-head train_grader dumps x v3_eegalign
                    tcn_g3_s1 as a train_eeg_det dump  vs  --video stored --video_members ...
                 B  R(2+1)D subject folds as fold dumps x GRU-bin subject folds as fold dumps
                    vs  --video stored --eeg stored (and hence the prototype's OOF numbers).
               Then the hazards: a double-softmaxed video dump is repaired (and logged) with
               unchanged results; a path-less dump is dropped (and logged); an EEG dump from the
               non-aligned 'eeg' session universe is refused for protocol A; a fold dump with the
               wrong animals is refused for protocol B.
  real         Tiny REAL runs of both trainers (CPU): train_grader.py (x3d bugged, dual heads)
               session split and subject folds 0-4; train_eeg_det.py aligned and subject folds
               0-4. Each is scored by joint_gate against the stored other modality
               (--allow_partial), end to end. Numbers are meaningless; the plumbing is not.
  frozen       protocol-B gate.json entries applied frozen to protocol A (--frozen_gate): every
               entry records its EEG input (pooled / top25) and signature; the frozen q is
               checked against sigmoid(coef . [logit P_V, logit <that score>, 1]) in-process and
               the reported macro-F1 / per-class hits against a hand recomputation, for EGRG,
               EGRG-top25 and EGRG-nested; a gate fitted on a different EEG detector / pooling is
               refused without --frozen_allow_mismatch; a protocol-B run with two EEG members
               needs --frozen_eeg_entry.

USAGE (from the video-eeg-ensembling repo, any cwd; PYTHONDONTWRITEBYTECODE=1, eeg env; EEG_ROOT env
       var, default /work/mech-ai-scratch/alloy/EEG. The script and every run it starts work in
       EEG_ROOT, so a relative --out is EEG_ROOT's.)
  python grader/eeg/verify_joint_gate.py all --out /work/mech-ai-scratch/alloy/EEG/output/ttg_eeg_test/jg
Exit 0 only if every check passed; summary in <out>/verify_joint_gate.json.
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import joint_gate as J                                    # noqa: E402
import check_folds as CF                                  # noqa: E402

EEG_ROOT = J.EEG_ROOT
PY = sys.executable
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
RES = []


def check(name, ok, detail=""):
    RES.append(dict(check=name, ok=bool(ok), detail=detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""), flush=True)
    return ok


def jg(args, out, log, expect_fail=False):
    t = time.time()
    cmd = [PY, os.path.join(HERE, "joint_gate.py"), *args, "--out", out]
    with open(log, "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=ENV, cwd=EEG_ROOT).returncode
    print(f"    joint_gate {' '.join(args)[:150]} -> rc {rc} ({time.time() - t:.0f}s)", flush=True)
    if expect_fail:
        return rc, open(log).read()
    return rc, (json.load(open(os.path.join(out, "results.json"))) if rc == 0 else None)


def run(cmd, log):
    t = time.time()
    with open(log, "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=ENV, cwd=EEG_ROOT).returncode
    print(f"    {os.path.basename(cmd[1])} {' '.join(cmd[2:])[:140]} -> rc {rc} ({time.time() - t:.0f}s)", flush=True)
    return rc


def same_rows(ra, rb, tag, tol=1e-12):
    """Every system of two result rows: macro-F1, MCC, per-class hits, CIs, deltas."""
    ok, why = True, []
    for n, sa in ra["systems"].items():
        sb = rb["systems"].get(n)
        if sb is None:
            ok = False; why.append(f"{n} missing"); continue
        if abs(sa["macro_f1"] - sb["macro_f1"]) > tol or abs(sa["mcc"] - sb["mcc"]) > tol:
            ok = False; why.append(f"{n} F1 {sa['macro_f1']:.6f} vs {sb['macro_f1']:.6f}")
        if [v["hit"] for v in sa["per_class"].values()] != [v["hit"] for v in sb["per_class"].values()]:
            ok = False; why.append(f"{n} per-class hits differ")
        for k in ("d_vs_video", "d_vs_posthoc"):
            if k in sa and any(abs(sa[k][m]["lo"] - sb[k][m]["lo"]) > 1e-9 or abs(sa[k][m]["hi"] - sb[k][m]["hi"]) > 1e-9
                               for m in sa[k]):
                ok = False; why.append(f"{n} {k} CI differs")
    ga, gb = ra["gates"]["EGRG"]["fit_all"], rb["gates"]["EGRG"]["fit_all"]
    if np.max(np.abs(np.array(ga) - np.array(gb))) > 1e-3:
        ok = False; why.append(f"gate {ga} vs {gb}")
    f = {n: round(s["macro_f1"], 4) for n, s in ra["systems"].items()}
    return check(tag, ok, "; ".join(why) if why else f"identical: {f}")


# ============================================================================ synthetic dumps

def write_video_dump(d, keys, y5, P3, P5, epoch=12, fold=-1, drop_path=False, ds=False, split="session"):
    os.makedirs(d, exist_ok=True)
    if ds:                                    # softmax(softmax(z)) as the X3D head bug stored it
        P3 = np.exp(P3) / np.exp(P3).sum(1, keepdims=True)
        P5 = np.exp(P5) / np.exp(P5).sum(1, keepdims=True)
    o = np.argsort([k + "/video.mp4" for k in keys], kind="stable")   # train_grader: sorted by path
    dump = dict(path=np.array([k + "/video.mp4" for k in keys])[o], y5=y5[o].astype(np.int64),
                epoch=np.int64(epoch), seed=np.int64(1), arch=np.array("x3d"), heads=np.array("dual"),
                split=np.array(split), fold=np.int64(fold), fix_x3d=np.bool_(False),
                head_softmax_bug=np.bool_(False), probs_g3=P3[o], probs_g5=P5[o])
    if drop_path:
        del dump["path"]
    np.savez(os.path.join(d, f"val_ep{epoch:02d}.npz"), **dump)
    json.dump(dict(run_key=dict(epochs=epoch, synthetic=True)), open(os.path.join(d, "config.json"), "w"))


def write_eeg_dump(d, keys, stored, win_p0, win_off, epoch=30, universe="video", fold=-1):
    os.makedirs(d, exist_ok=True)
    y5 = J.stage5_of(keys)
    wp = np.c_[win_p0, 1.0 - win_p0].astype(np.float32)
    tk = np.array([np.sort(1.0 - wp[win_off[i]:win_off[i + 1], 0].astype(float))[
        -max(1, int(round(0.25 * (win_off[i + 1] - win_off[i])))):].mean() for i in range(len(keys))])
    pl = np.c_[1.0 - stored, stored]
    np.savez(os.path.join(d, f"val_clip_ep{epoch:02d}.npz"), path=np.asarray(keys), y=(y5 > 0).astype(np.int64),
             y5=y5, clip_id=np.arange(len(keys)), sub=J.D.animal_session(keys)[0],
             sess=J.D.animal_session(keys)[1], p_logmean=stored.astype(np.float64),
             p_mean=stored.astype(np.float64), p_top25=tk, probs_logmean=pl.astype(np.float32),
             probs_mean=pl.astype(np.float32), win_probs=wp, win_offsets=win_off.astype(np.int64),
             topk_frac=np.float64(0.25), epoch=np.int64(epoch), seed=np.int64(1), arch=np.array("tcn"),
             split=np.array("session" if fold < 0 else "subject"), fold=np.int64(fold),
             session_universe=np.array(universe if fold < 0 else "None"), last_epoch=np.bool_(True))
    json.dump(dict(run_key=dict(epochs=epoch, synthetic=True)), open(os.path.join(d, "config.json"), "w"))


def stored_eeg_windows(run_dir):
    """(keys, stored P(sz), window P(non-sz) packed per clip, offsets) of a stored EEG run."""
    z = np.load(os.path.join(run_dir, "val_clip_preds.npz"), allow_pickle=True)
    w = np.load(os.path.join(run_dir, "val_window_preds.npz"))
    pos = {c: i for i, c in enumerate(w["clips"])}
    r = np.array([pos[c] for c in w["cid"]])
    o = np.argsort(r, kind="stable")
    off = np.r_[0, np.cumsum(np.bincount(r, minlength=len(z["path"])))]
    P = z["probs"].astype(np.float64)
    return (np.array([J.key_of(p) for p in z["path"]]), J.det_score(P), w["probs"][o, 0], off)


def write_syn(a):
    """Write the synthetic dumps (idempotent; a few seconds). Returns the syn/ directory."""
    S = os.path.join(a.out, "syn")
    # ---- protocol A video: two MViT seeds as dual-head dumps
    va = {t: J.load_video_stored_A(t) for t in ("g3", "g5")}
    for s in (1, 2):
        P3, P5 = va["g3"]["members"][f"mvit_g3_s{s}"], va["g5"]["members"][f"mvit_g5_s{s}"]
        assert np.array_equal(va["g3"]["keys"], va["g5"]["keys"])
        write_video_dump(os.path.join(S, f"A_video/mvit_dual_s{s}"), va["g3"]["keys"], va["g3"]["y5"], P3, P5)
    write_video_dump(os.path.join(S, "A_video/mvit_dual_s1_ds"), va["g3"]["keys"], va["g3"]["y5"],
                     va["g3"]["members"]["mvit_g3_s1"], va["g5"]["members"]["mvit_g5_s1"], ds=True)
    write_video_dump(os.path.join(S, "A_video/mvit_dual_s1_nopath"), va["g3"]["keys"], va["g3"]["y5"],
                     va["g3"]["members"]["mvit_g3_s1"], va["g5"]["members"]["mvit_g5_s1"], drop_path=True)
    # ---- protocol A EEG: v3_eegalign tcn_g3_s1 as a train_eeg_det dump
    k, st, w0, off = stored_eeg_windows(os.path.join(EEG_ROOT, "output/v3_eegalign/tcn_g3_s1"))
    write_eeg_dump(os.path.join(S, "A_eeg/tcn_g3_s1"), k, st, w0, off)
    write_eeg_dump(os.path.join(S, "A_eeg/tcn_g3_s1_eeguniverse"), k, st, w0, off, universe="eeg")
    # ---- protocol B: R(2+1)D folds + GRU-bin folds as fold dumps
    folds = CF.expected_folds()
    vb = {t: J.load_video_stored_B(t) for t in ("g3", "g5")}
    assert np.array_equal(vb["g3"]["keys"], vb["g5"]["keys"])
    for f in range(5):
        m = np.isin(vb["g3"]["keys"], list(folds[f]["video_keys"]))
        write_video_dump(os.path.join(S, f"B_video/r2p1d_dual_fold{f}"), vb["g3"]["keys"][m], vb["g3"]["y5"][m],
                         vb["g3"]["members"]["r2plus1d_g3_subject_folds"][m],
                         vb["g5"]["members"]["r2plus1d_g5_subject_folds"][m], fold=f, split="subject")
        k, st, w0, off = stored_eeg_windows(os.path.join(EEG_ROOT, f"output/v3_subject_cv/eeg_bin_fold{f}"))
        write_eeg_dump(os.path.join(S, f"B_eeg/gru_bin_fold{f}"), k, st, w0, off, fold=f)
    # a fold dump holding the WRONG animals (fold 1's clips saved as fold 0)
    for f in range(5):
        src = os.path.join(S, f"B_video/r2p1d_dual_fold{1 if f == 0 else f}", "val_ep12.npz")
        d = os.path.join(S, f"B_video_bad/r2p1d_dual_fold{f}")
        os.makedirs(d, exist_ok=True)
        z = dict(np.load(src, allow_pickle=True))
        np.savez(os.path.join(d, "val_ep12.npz"), **z)
    # a second copy of the EEG folds under another name: a protocol-B run with two EEG members
    for f in range(5):
        k, st, w0, off = stored_eeg_windows(os.path.join(EEG_ROOT, f"output/v3_subject_cv/eeg_bin_fold{f}"))
        write_eeg_dump(os.path.join(S, f"B_eeg2/gru_bin_copy_fold{f}"), k, st, w0, off, fold=f)
    return S


def synthetic(a):
    print("\n== synthetic full-size dumps through the run-directory loaders", flush=True)
    S = write_syn(a)

    ref, _ = jg(["--protocol", "A", "--video", "stored", "--video_members", "mvit_{task}_s1", "mvit_{task}_s2",
                 "--eeg", "stored", "--eeg_members", "tcn_g3_s1", "--rows", "members"],
                os.path.join(a.out, "A_ref"), os.path.join(a.out, "A_ref.log"))
    new, _ = jg(["--protocol", "A", "--video", f"{S}/A_video/mvit_dual_s1", f"{S}/A_video/mvit_dual_s2",
                 "--eeg", f"{S}/A_eeg/tcn_g3_s1", "--rows", "members"],
                os.path.join(a.out, "A_dirs"), os.path.join(a.out, "A_dirs.log"))
    if check("A: run-directory path completes", ref == 0 and new == 0):
        R = json.load(open(os.path.join(a.out, "A_ref/results.json")))
        N = json.load(open(os.path.join(a.out, "A_dirs/results.json")))
        for bR, bN in zip(R["blocks"], N["blocks"]):
            t = bR["task"]
            same_rows(bR["rows"][0], bN["rows"][0], f"A {t}: 2-seed video ensemble x EEG dump == stored path")
            sR = [r for r in bR["rows"] if r["video_kind"] == "single network"]
            sN = [r for r in bN["rows"] if r["video_kind"] == "single network"]
            for x, y in zip(sR, sN):
                same_rows(x, y, f"A {t}: single {x['video']} == dump {y['video']}")
            check(f"A {t}: dump selection labelled fixed last epoch",
                  "fixed LAST epoch" in str(bN["video_selection"]) and "fixed LAST epoch" in str(bN["eeg_selection"]),
                  f"{bN['video_selection']} | {bN['eeg_selection']}")
            check(f"A {t}: paired 5,279 clips, 10 video clips dropped for no EEG",
                  bN["pairing"]["n_scored"] == 5279 and bN["pairing"]["video_clips_dropped_no_eeg"] == 10)
    rc, _ = jg(["--protocol", "A", "--video", f"{S}/A_video/mvit_dual_s1_ds", "--eeg", f"{S}/A_eeg/tcn_g3_s1",
                "--rows", "headline", "--tasks", "g3"], os.path.join(a.out, "A_ds"), os.path.join(a.out, "A_ds.log"))
    if check("A: double-softmaxed video dump loads", rc == 0):
        D_ = json.load(open(os.path.join(a.out, "A_ds/results.json")))["blocks"][0]
        R1 = json.load(open(os.path.join(a.out, "A_dirs/results.json")))["blocks"][0]
        check("A: double-softmax detected and repaired (logged)",
              len(D_["video_log"]["double_softmax_repaired"]) == 1, D_["video_load"])
        s1 = [r for r in R1["rows"] if r["video"] == "mvit_dual_s1"][0]
        same_rows(D_["rows"][0], s1, "A g3: repaired dump gives the single-run numbers", tol=1e-9)
    rc, _ = jg(["--protocol", "A", "--video", f"{S}/A_video/mvit_dual_s1", f"{S}/A_video/mvit_dual_s1_nopath",
                "--eeg", f"{S}/A_eeg/tcn_g3_s1", "--rows", "headline", "--tasks", "g3"],
               os.path.join(a.out, "A_nopath"), os.path.join(a.out, "A_nopath.log"))
    if check("A: a path-less dump next to a good one still runs", rc == 0):
        b = json.load(open(os.path.join(a.out, "A_nopath/results.json")))["blocks"][0]
        check("A: path-less video dump dropped and logged",
              b["video_log"]["n_loaded"] == 1 and len(b["video_log"]["dropped"]) == 1, b["video_load"])
    rc, txt = jg(["--protocol", "A", "--video", f"{S}/A_video/mvit_dual_s1", "--eeg",
                  f"{S}/A_eeg/tcn_g3_s1_eeguniverse", "--tasks", "g3"],
                 os.path.join(a.out, "A_eegu"), os.path.join(a.out, "A_eegu.log"), expect_fail=True)
    check("A: EEG dump from the non-aligned 'eeg' session universe is refused",
          rc != 0 and "not the aligned" in txt, txt.strip().splitlines()[-1][:160])
    # a dump written with another --topk_frac: the secondary score is still top-25%, recomputed at
    # 0.25 from the packed windows, and the load log says the dump's value was ignored
    z = dict(np.load(os.path.join(S, "A_eeg/tcn_g3_s1/val_clip_ep30.npz"), allow_pickle=True))
    z["topk_frac"] = np.float64(0.5)
    z["p_top25"] = J.topk_packed(1.0 - z["win_probs"][:, 0].astype(float), z["win_offsets"], 0.5)
    dfr = os.path.join(S, "A_eeg/tcn_g3_s1_frac50")
    os.makedirs(dfr, exist_ok=True)
    np.savez(os.path.join(dfr, "val_clip_ep30.npz"), **z)
    e0 = J.load_eeg_dirs([os.path.join(S, "A_eeg/tcn_g3_s1")], 30, "A", "logmean", False)
    e1 = J.load_eeg_dirs([dfr], 30, "A", "logmean", False)
    t0_, t1_ = e0["members"]["tcn_g3_s1"]["top25"], e1["members"]["tcn_g3_s1_frac50"]["top25"]
    check("A: a dump with --topk_frac 0.5 still gives the top-25% score, and the log says so",
          np.array_equal(t0_, t1_) and any("topk_frac=0.5" in x for x in e1["log"]["notes"])
          and not e0["log"]["notes"], "; ".join(e1["log"]["notes"])[:200])

    ref, _ = jg(["--protocol", "B", "--video", "stored", "--eeg", "stored", "--rows", "headline"],
                os.path.join(a.out, "B_ref"), os.path.join(a.out, "B_ref.log"))
    new, _ = jg(["--protocol", "B", "--video", f"{S}/B_video/r2p1d_dual_fold{{f}}", "--eeg",
                 f"{S}/B_eeg/gru_bin_fold{{f}}", "--rows", "headline"],
                os.path.join(a.out, "B_dirs"), os.path.join(a.out, "B_dirs.log"))
    if check("B: fold-template path completes", ref == 0 and new == 0):
        R = json.load(open(os.path.join(a.out, "B_ref/results.json")))
        N = json.load(open(os.path.join(a.out, "B_dirs/results.json")))
        for bR, bN in zip(R["blocks"], N["blocks"]):
            same_rows(bR["rows"][0], bN["rows"][0], f"B {bR['task']}: fold dumps == stored OOF path")
            exp = J.REG[("B", bR["task"])]["f1"]
            got = (bN["rows"][0]["systems"]["video"]["macro_f1"], bN["rows"][0]["systems"]["post-hoc gate"]["macro_f1"],
                   bN["rows"][0]["systems"]["EGRG-top25"]["macro_f1"])
            check(f"B {bR['task']}: fold dumps reproduce the prototype OOF numbers",
                  all(abs(g - e) <= 5e-5 for g, e in zip(got, exp)), f"{np.round(got, 4).tolist()} vs {list(exp)}")
            check(f"B {bR['task']}: 45 video clips dropped for no EEG",
                  bN["pairing"]["video_clips_dropped_no_eeg"] == 45)
    rc, txt = jg(["--protocol", "B", "--video", f"{S}/B_video_bad/r2p1d_dual_fold{{f}}", "--eeg",
                  f"{S}/B_eeg/gru_bin_fold{{f}}", "--tasks", "g3"],
                 os.path.join(a.out, "B_bad"), os.path.join(a.out, "B_bad.log"), expect_fail=True)
    check("B: a fold dump holding another fold's animals is refused", rc != 0 and "animals" in txt,
          txt.strip().splitlines()[-1][:200])


def real(a):
    print("\n== real tiny runs of both trainers, scored end to end (numbers meaningless)", flush=True)
    R = os.path.join(a.out, "real")
    grader = os.path.join(os.path.dirname(HERE), "train_grader.py")      # grader/train_grader.py
    det = os.path.join(HERE, "train_eeg_det.py")
    gcommon = ["--arch", "x3d", "--heads", "dual", "--seed", "1", "--epochs", "1", "--batch_size", "2",
               "--workers", "0", "--allow_cpu", "--only_cached", "--limit_train", "4", "--items", "f32index"]
    dcommon = ["--seed", "1", "--epochs", "1", "--limit_train", "1536", "--allow_cpu", "--threads", "4"]
    os.makedirs(R, exist_ok=True)
    jobs = [([PY, grader, *gcommon, "--split", "session", "--limit_val", "40", "--output",
              os.path.join(R, "vsess/x3dbug_dual_s1")], "vsess"),
            ([PY, det, *dcommon, "--split", "session", "--session_universe", "video", "--limit_val", "60",
              "--output", os.path.join(R, "eal/tcn_bin_s1")], "eal")]
    for f in range(5):
        jobs.append(([PY, grader, *gcommon, "--split", "subject", "--fold", str(f), "--limit_val", "30",
                      "--output", os.path.join(R, f"vsubj/x3dbug_dual_s1_fold{f}")], f"vsubj{f}"))
        jobs.append(([PY, det, *dcommon, "--split", "subject", "--fold", str(f), "--limit_val", "40",
                      "--output", os.path.join(R, f"esubj/tcn_bin_fold{f}_s1")], f"esubj{f}"))
    for cmd, tag in jobs:
        if os.path.exists(os.path.join(cmd[-1], "results.json")):
            continue
        check(f"real run {tag} finishes", run(cmd, os.path.join(R, f"{tag}.log")) == 0)
    z = np.load(os.path.join(R, "vsubj/x3dbug_dual_s1_fold0/val_ep01.npz"), allow_pickle=True)
    zs = np.load(os.path.join(a.out, "syn/B_video/r2p1d_dual_fold0/val_ep12.npz"), allow_pickle=True)
    check("synthetic video dumps carry every field of a real train_grader dump",
          set(z.files) == set(zs.files) and all(z[k].dtype.kind == zs[k].dtype.kind for k in z.files),
          f"{sorted(z.files)}")
    ze = np.load(os.path.join(R, "esubj/tcn_bin_fold0_s1/val_clip_ep01.npz"), allow_pickle=True)
    zse = np.load(os.path.join(a.out, "syn/B_eeg/gru_bin_fold0/val_clip_ep30.npz"), allow_pickle=True)
    check("synthetic EEG dumps carry every field of a real train_eeg_det dump",
          set(ze.files) == set(zse.files), f"{sorted(set(ze.files) ^ set(zse.files))}")
    cases = [("A_realvideo", ["--protocol", "A", "--video", f"{R}/vsess/x3dbug_dual_s1", "--video_epoch", "1",
                              "--eeg", "stored", "--eeg_members", "tcn_g3_s1"]),
             ("A_realeeg", ["--protocol", "A", "--video", "stored", "--video_members", "x3d_{task}_s1",
                            "--eeg", f"{R}/eal/tcn_bin_s1", "--eeg_epoch", "1"]),
             ("B_realvideo", ["--protocol", "B", "--video", f"{R}/vsubj/x3dbug_dual_s1_fold{{f}}", "--video_epoch", "1",
                              "--eeg", "stored"]),
             ("B_realeeg", ["--protocol", "B", "--video", "stored", "--eeg", f"{R}/esubj/tcn_bin_fold{{f}}_s1",
                            "--eeg_epoch", "1"])]
    for tag, args in cases:
        rc, res = jg(args + ["--allow_partial", "--rows", "headline"], os.path.join(a.out, tag),
                     os.path.join(a.out, f"{tag}.log"))
        ok = rc == 0 and res is not None
        if ok:
            b = res["blocks"][0]
            n = b["pairing"]["n_scored"]
            ok = n > 0 and all(t in b["rows"][0]["systems"] for t in ("video", "post-hoc gate", "EGRG"))
            detail = (f"{n} clips scored; video {b['video_load']} | EEG {b['eeg_load']} | "
                      f"EGRG {b['rows'][0]['systems']['EGRG']['macro_f1']:.3f}")
        else:
            detail = open(os.path.join(a.out, f"{tag}.log")).read().strip().splitlines()[-1][:200]
        check(f"{tag}: joint_gate end to end on real dumps", ok, detail)
    rc, txt = jg(["--protocol", "B", "--video", f"{R}/vsubj/x3dbug_dual_s1_fold{{f}}", "--video_epoch", "1",
                  "--eeg", "stored", "--tasks", "g3"], os.path.join(a.out, "B_partial_refused"),
                 os.path.join(a.out, "B_partial_refused.log"), expect_fail=True)
    check("B: partial folds are refused without --allow_partial", rc != 0 and "clips" in txt,
          txt.strip().splitlines()[-1][:200])


def hand_frozen(task, coef, eeg_input, video_members):
    """Recompute a frozen gate by hand on protocol A stored: q = sigmoid(coef . [logit P_V(sz),
    logit P_E(sz)<eeg_input>, 1]) on the ensembles, decision argmax [1-q, q*P_V(g|sz)]."""
    vb = J.filter_members(J.load_video_stored_A(task), video_members, task, "video")
    eb = J.load_eeg_stored_A()
    P = J.Pairing(vb, eb, task, 10, 0)
    Pv = np.mean([M[P.iv] for M in vb["members"].values()], 0)
    e = {k: np.mean([M[k][P.ie] for M in eb["members"].values()], 0) for k in ("stored", "top25")}
    sv = 1.0 - Pv[:, 0]
    out = {}
    for inp, ev in (("pooled", e["stored"]), ("top25", e["top25"])):
        z = coef[0] * J.logit(sv) + coef[1] * J.logit(ev) + coef[2]
        q = 1.0 / (1.0 + np.exp(-z))
        C = J.D.confusion(P.y, J.comp(Pv, q), P.K)
        out[inp] = dict(f1=float(J.D.cm_metrics(C)["macro_f1"]), hits=np.diag(C).tolist())
    return out[eeg_input], out["top25" if eeg_input == "pooled" else "pooled"], (P, Pv, e)


def frozen(a):
    print("\n== frozen protocol-B gates applied to protocol A", flush=True)
    g = os.path.join(a.out, "B_ref", "gate.json")
    if not (os.path.exists(g) and json.load(open(g)).get("format") == J.GATE_FORMAT):
        jg(["--protocol", "B", "--video", "stored", "--eeg", "stored", "--rows", "headline"],
           os.path.join(a.out, "B_ref"), os.path.join(a.out, "B_ref.log"))
    G = json.load(open(g))
    ok = G.get("format") == J.GATE_FORMAT and all(
        x["eeg_input"] == J.GATE_INPUT.get(x["system"], x["eeg_input"]) and
        (x["eeg_input"] == "top25") == any("top25" in f for f in x["features"]) and
        x["eeg_signature"]["score"].startswith("top25" if x["eeg_input"] == "top25" else "pooled")
        for x in G["gates"])
    check("gate.json (format 2): every entry records eeg_input consistent with system / features, and "
          "eeg_signature", ok, f"{len(G['gates'])} entries: " +
          ", ".join(f"{x['task']} {x['system']}={x['eeg_input']}" for x in G["gates"]))

    # ---- unit: the frozen q is computed on the EEG score the entry names (the reviewed bug)
    for sysn in ("EGRG", "EGRG-top25", "EGRG-nested"):
        ent = [x for x in G["gates"] if x["task"] == "g3" and x["system"] == sysn][0]
        fz = J.select_frozen(g, sysn, None, None, ["g3"])[0]
        fz.update(applied_to=None, compatible=False)
        _, _, (P, Pv, e) = hand_frozen("g3", ent["coef"], ent["eeg_input"], ["r2plus1d_{task}_s*"])
        _, _, Q, gates = P.systems(Pv, e["stored"], e["top25"], frozen=fz)
        sv = 1.0 - Pv[:, 0]
        want = {inp: 1.0 / (1.0 + np.exp(-(ent["coef"][0] * J.logit(sv) + ent["coef"][1] * J.logit(ev) +
                                            ent["coef"][2])))
                for inp, ev in (("pooled", e["stored"]), ("top25", e["top25"]))}
        right = np.allclose(Q["EGRG (frozen gate)"], want[ent["eeg_input"]], rtol=0, atol=1e-12)
        wrong = np.allclose(Q["EGRG (frozen gate)"], want["top25" if ent["eeg_input"] == "pooled" else "pooled"])
        check(f"unit g3 {sysn}: frozen q = sigmoid(coef . [logit P_V, logit {ent['eeg_input']}, 1]) and not the "
              f"other EEG score", right and not wrong and gates["EGRG (frozen gate)"]["eeg_input"] == ent["eeg_input"],
              f"right={right} wrong-input-match={wrong}")

    # ---- end to end: stored B gate (GRU-bin, mean pooling) -> stored A (g3 heads, logmean) is a
    #      DIFFERENT EEG input: refused unless --frozen_allow_mismatch, then labelled MISMATCHED
    base = ["--protocol", "A", "--video", "stored", "--video_members", "r2plus1d_{task}_s*", "--eeg", "stored",
            "--frozen_gate", g, "--rows", "headline", "--reps", "200", "--nested", "none"]
    rc, txt = jg(base + ["--frozen_features", "EGRG"], os.path.join(a.out, "A_frozen_refused"),
                 os.path.join(a.out, "A_frozen_refused.log"), expect_fail=True)
    check("frozen gate fitted on another EEG detector / pooling is refused without --frozen_allow_mismatch",
          rc != 0 and "mismatch" in txt, txt.strip().splitlines()[-1][:220])
    for sysn in ("EGRG", "EGRG-top25", "EGRG-nested"):
        tag = sysn.replace("EGRG", "egrg").replace("-", "_")
        rc, res = jg(base + ["--frozen_features", sysn, "--frozen_allow_mismatch"],
                     os.path.join(a.out, f"A_frozen_{tag}"), os.path.join(a.out, f"A_frozen_{tag}.log"))
        if not check(f"frozen {sysn} (allowed mismatch) completes", rc == 0):
            continue
        for b in res["blocks"]:
            t = b["task"]
            ent = [x for x in G["gates"] if x["task"] == t and x["system"] == sysn][0]
            s_ = b["rows"][0]["systems"]["EGRG (frozen gate)"]
            gf = b["rows"][0]["gates"]["EGRG (frozen gate)"]
            mine, other, _ = hand_frozen(t, ent["coef"], ent["eeg_input"], ["r2plus1d_{task}_s*"])
            got_hits = [v["hit"] for v in s_["per_class"].values()]
            check(f"A {t} frozen {sysn}: macro-F1 = hand recomputation on the {ent['eeg_input']} EEG score",
                  abs(s_["macro_f1"] - mine["f1"]) < 1e-12 and got_hits == mine["hits"],
                  f"reported {s_['macro_f1']:.4f} {got_hits}; hand {mine['f1']:.4f} {mine['hits']}; "
                  f"on the other score it would be {other['f1']:.4f}")
            check(f"A {t} frozen {sysn}: labelled MISMATCHED, eeg_input {ent['eeg_input']} recorded",
                  gf["compatible"] is False and gf["eeg_input"] == ent["eeg_input"] and
                  "d_vs_posthoc" in s_, f"fitted_on {gf['fitted_on']} applied_to {gf['applied_to']}")

    # ---- matching EEG input (train_eeg_det-format dumps on both sides): applied without the flag;
    #      a protocol-B run with TWO EEG members needs --frozen_eeg_entry to pick the gate
    S = write_syn(a)
    rc, _ = jg(["--protocol", "B", "--video", f"{S}/B_video/r2p1d_dual_fold{{f}}", "--eeg",
                f"{S}/B_eeg/gru_bin_fold{{f}}", f"{S}/B_eeg2/gru_bin_copy_fold{{f}}", "--rows", "members",
                "--nested", "headline", "--reps", "200"],
               os.path.join(a.out, "B_dirs2"), os.path.join(a.out, "B_dirs2.log"))
    if check("B with two EEG members (dump dirs) completes", rc == 0):
        g2 = os.path.join(a.out, "B_dirs2", "gate.json")
        G2 = json.load(open(g2))
        names = sorted({x["eeg"] for x in G2["gates"]})
        check("gate.json holds one entry per EEG member and for the EEG ensemble", len(names) == 3, str(names))
        argsA = ["--protocol", "A", "--video", f"{S}/A_video/mvit_dual_s1", "--eeg", f"{S}/A_eeg/tcn_g3_s1",
                 "--rows", "headline", "--reps", "200", "--nested", "none", "--frozen_gate", g2]
        rc, txt = jg(argsA, os.path.join(a.out, "A_frozen_amb"), os.path.join(a.out, "A_frozen_amb.log"),
                     expect_fail=True)
        check("ambiguous frozen entry (several EEG members) is refused and the entries listed",
              rc != 0 and "need exactly one" in txt and "eeg_ens2" in txt, txt.strip().splitlines()[-1][:220])
        for sysn in ("EGRG", "EGRG-top25"):
            rc, res = jg(argsA + ["--frozen_eeg_entry", "eeg_ens2", "--frozen_features", sysn],
                         os.path.join(a.out, f"A_frozen_ens2_{sysn}"),
                         os.path.join(a.out, f"A_frozen_ens2_{sysn}.log"))
            ok = rc == 0
            if ok:
                for b in res["blocks"]:
                    gf = b["rows"][0]["gates"]["EGRG (frozen gate)"]
                    ent = [x for x in G2["gates"] if x["task"] == b["task"] and x["system"] == sysn
                           and x["eeg"] == "eeg_ens2"][0]
                    ok = ok and gf["compatible"] is True and "eeg_ens2" in gf["source"] and \
                        gf["coef"] == ent["coef"] and gf["eeg_input"] == ent["eeg_input"]
            check(f"--frozen_eeg_entry eeg_ens2 {sysn}: the matching EEG input is applied without the "
                  f"mismatch flag", ok, "" if ok else f"rc {rc}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("what", choices=["all", "regression", "synthetic", "real", "frozen"])
    ap.add_argument("--out", default=os.path.join(EEG_ROOT, "output", "ttg_eeg_test", "jg"))
    a = ap.parse_args()
    os.chdir(EEG_ROOT)                  # the runs it starts use EEG_ROOT-relative defaults
    if not os.path.realpath(a.out).startswith(os.path.join(EEG_ROOT, "output", "ttg_eeg")):
        raise SystemExit("--out must live under output/ttg_eeg*/")
    os.makedirs(a.out, exist_ok=True)
    t = time.time()
    if a.what in ("all", "regression"):
        print("\n== joint_gate.py --regression", flush=True)
        rc, res = jg(["--regression"], os.path.join(a.out, "regression"), os.path.join(a.out, "regression.log"))
        rg = res["regression"] if res else json.load(open(os.path.join(a.out, "regression/results.json")))["regression"]
        check("prototype regression", rc == 0 and rg["n_passed"] == rg["n_checks"],
              f"{rg['n_passed']}/{rg['n_checks']} checks")
    if a.what in ("all", "synthetic"):
        synthetic(a)
    if a.what in ("all", "real"):
        real(a)
    if a.what in ("all", "frozen"):
        frozen(a)
    nf = sum(not r["ok"] for r in RES)
    json.dump(dict(what=a.what, n_checks=len(RES), n_failed=nf, secs=round(time.time() - t), checks=RES),
              open(os.path.join(a.out, f"verify_joint_gate_{a.what}.json"), "w"), indent=1)
    print(f"\n{len(RES) - nf}/{len(RES)} checks passed ({time.time() - t:.0f}s)")
    return 1 if nf else 0


if __name__ == "__main__":
    sys.exit(main())
