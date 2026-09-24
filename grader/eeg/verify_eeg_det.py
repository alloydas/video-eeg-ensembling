#!/usr/bin/env python3
"""
CPU verification of grader/eeg/train_eeg_det.py against the parent trainer train_pooled_eeg.py.

PURPOSE
  train_eeg_det.py replicates train_pooled_eeg.main()'s training loop (to add fixed-epoch
  dumps and mid-epoch resume). This script proves on CPU, with short runs, that nothing
  the EGRG numbers depend on drifted:

  firstbatch  The ORIGINAL train_pooled_eeg.main() is run in-process (child process) with
              its DataLoaders limited to the first K training batches and one val batch and
              its CrossEntropyLoss instrumented; train_eeg_det.py is run with
              --epochs 1 --debug_max_steps K. Same seed must give: the same split (every val
              clip path, label and window, against the original's own saved
              val_clip_preds / val_window_preds), the same train window count, the same
              class weights, the same batch label sums and the SAME K training losses
              (model init + dummy forward + shuffle + dropout + AdamW all in the same order).
              Settings: aligned (--session_universe video) and subject fold 0.
  ckpt        A stored best.pt (default output/v3_eegalign/tcn_g3_s1, a 3-class TCN) is
              evaluated through train_eeg_det's load_cache / make_split / predict_windows /
              aggregate_clip path on a seeded subset of its val clips; its stored
              val_window_preds and val_clip_preds (GPU) must be reproduced, and its full
              val clip list must equal this split's.
  resume      Tiny runs: uninterrupted vs (in-process SIGTERM mid-epoch -> exit 3 -> resume
              -> SIGTERM during validation -> exit 3 -> resume) vs (external kill -TERM ->
              exit 3 -> resume). Every per-epoch dump and the final weights must be
              BIT-identical.
  errors      --split subject --session_universe video, a session split without a universe,
              and an output outside output/ttg_eeg*/ must all be refused.
  all         everything above.

USAGE (from the video-eeg-ensembling repo, any cwd; PYTHONDONTWRITEBYTECODE=1, eeg env; CPU only,
       ~4 threads each; EEG_ROOT env var, default /work/mech-ai-scratch/alloy/EEG. The script and
       every run it starts work in EEG_ROOT, so a relative --out / --ckpt_run is EEG_ROOT's.)
  python grader/eeg/verify_eeg_det.py all --out /work/mech-ai-scratch/alloy/EEG/output/ttg_eeg_test/verify
  python grader/eeg/verify_eeg_det.py firstbatch --setting aligned --seed 1 --steps 3 --out ...
Exit 0 only if every check passed; a JSON summary is written to <out>/verify_<what>.json.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PY = sys.executable
TRAINER = os.path.join(HERE, "train_eeg_det.py")
SELF = os.path.abspath(__file__)                          # re-exec'd with cwd = EEG_ROOT
EEG_ROOT = os.path.realpath(os.environ.get("EEG_ROOT", "/work/mech-ai-scratch/alloy/EEG"))
sys.path.insert(1, EEG_ROOT)                              # train_pooled_eeg / preds_io
CACHE = "cache_bestcfg/seg_w6.0_s3.0_d8.npz"
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(dict(check=name, ok=bool(ok), detail=detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""), flush=True)
    return ok


def run(cmd, log, expect=(0,), timeout=7200):
    t = time.time()
    with open(log, "w") as fh:
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=ENV, cwd=EEG_ROOT, timeout=timeout)
    print(f"    ran {' '.join(os.path.basename(c) if c.startswith('/') and c.endswith('.py') else c for c in cmd[1:])[:160]} "
          f"-> rc {p.returncode} ({time.time() - t:.0f}s, log {log})", flush=True)
    return p.returncode


def small_cache():
    d = np.load(os.path.join(EEG_ROOT, CACHE), allow_pickle=True)
    return dict(cid=d["clip_id"], clab5=d["clip_lab"], csess=d["clip_sess"], cpath=d["clip_path"])


# ================================================================ original, instrumented

def orig_child(a):
    """Run the ORIGINAL train_pooled_eeg.main() with K train batches and 1 val batch."""
    import torch
    torch.set_num_threads(a.threads)
    import train_pooled_eeg as tpe
    rec = dict(losses=[], y_sums=[], weights=None, n_train=None, exception=None)
    RealDL = tpe.DataLoader

    class LimitedDL:
        def __init__(s, ds, batch_size=1, shuffle=False, drop_last=False):
            s.dl = RealDL(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
            s.train = shuffle
            s.nmax = a.steps if shuffle else 1
            if shuffle:
                rec["n_train"] = len(ds)

        def __iter__(s):
            it = iter(s.dl)                  # draws the iterator base seed, as in the original
            for _ in range(s.nmax):
                try:
                    b = next(it)
                except StopIteration:
                    return
                if s.train:
                    rec["y_sums"].append(int(b[1].sum()))
                yield b

    RealCE = torch.nn.CrossEntropyLoss

    class RecCE(RealCE):
        def __init__(s, *x, **k):
            super().__init__(*x, **k)
            rec["weights"] = s.weight.tolist()

        def forward(s, i, t):
            loss = super().forward(i, t)
            rec["losses"].append(loss.item())
            return loss

    real_agg = tpe.aggregate_clip

    def agg_padded(P, cid_va, val_clips, *x, **k):
        # only the first val batch was evaluated: pad the unevaluated windows with a uniform
        # posterior so the original can still pool and SAVE its val clip list / window map
        if len(P) < len(cid_va):
            P = np.vstack([P, np.full((len(cid_va) - len(P), P.shape[1]), 1.0 / P.shape[1], P.dtype)])
        return real_agg(P, cid_va, val_clips, *x, **k)

    tpe.DataLoader = LimitedDL
    tpe.aggregate_clip = agg_padded
    torch.nn.CrossEntropyLoss = RecCE
    argv = ["train_pooled_eeg.py", "--arch", "tcn", "--group2", "--agg", "logmean", "--cache", CACHE,
            "--seed", str(a.seed), "--split_seed", "49", "--epochs", "1", "--output", a.orig_out]
    argv += (["--session_universe", "video"] if a.setting == "aligned" else ["--split", "subject", "--fold", "0"])
    sys.argv = argv
    try:
        tpe.main()
    except BaseException as e:               # report() may trip on a 3-step model; records are kept
        rec["exception"] = f"{type(e).__name__}: {e}"
    with open(a.rec, "w") as f:
        json.dump(rec, f, indent=1)


def firstbatch(a, setting, seed, steps):
    print(f"\n== firstbatch: setting={setting} seed={seed} steps={steps}", flush=True)
    base = os.path.join(a.out, f"firstbatch_{setting}_s{seed}")
    orig_out, mine_out = os.path.join(base, "orig"), os.path.join(base, "mine")
    os.makedirs(base, exist_ok=True)
    rec_path = os.path.join(base, "orig_record.json")
    run([PY, SELF, "_orig", "--setting", setting, "--seed", str(seed), "--steps", str(steps),
         "--orig_out", orig_out, "--rec", rec_path, "--threads", "4"], os.path.join(base, "orig.log"))
    extra = ["--session_universe", "video"] if setting == "aligned" else ["--split", "subject", "--fold", "0"]
    for f in ("last.pt", "results.json"):     # a re-run must train again, not resume / skip
        p = os.path.join(mine_out, f)
        if os.path.exists(p):
            os.replace(p, p + ".prev")
    rc = run([PY, TRAINER, *extra, "--seed", str(seed), "--epochs", "1", "--debug_max_steps", str(steps),
              "--limit_val", "20", "--allow_cpu", "--threads", "4", "--output", mine_out],
             os.path.join(base, "mine.log"))
    check(f"{setting}: train_eeg_det run completed", rc == 0)
    rec = json.load(open(rec_path))
    res = json.load(open(os.path.join(mine_out, "results.json")))
    cfg = json.load(open(os.path.join(mine_out, "config.json")))
    mine_l = [r["loss"] for r in res["first_losses"]][:steps]
    mine_y = [r["y_sum"] for r in res["first_losses"]][:steps]
    d = float(np.max(np.abs(np.array(mine_l) - np.array(rec["losses"][:steps])))) if len(mine_l) == len(rec["losses"][:steps]) else float("inf")
    check(f"{setting}: same {steps} training losses as the original main()", d == 0.0,
          f"orig {rec['losses'][:steps]} mine {mine_l} max|diff| {d:.3g}")
    check(f"{setting}: same batch label sums", mine_y == rec["y_sums"][:steps], f"{rec['y_sums']} vs {mine_y}")
    check(f"{setting}: same train window count", rec["n_train"] == cfg["n_train_windows"],
          f"{rec['n_train']} vs {cfg['n_train_windows']}")
    dw = float(np.max(np.abs(np.array(rec["weights"]) - np.array(cfg["class_weights"]))))
    check(f"{setting}: same class weights", dw < 1e-6, f"orig {np.round(rec['weights'], 4).tolist()} "
          f"mine {np.round(cfg['class_weights'], 4).tolist()}")
    # full split vs the original's own saved val set
    import train_eeg_det as T
    S = small_cache()
    clab2 = T.tpe.GROUP2[S["clab5"]]
    civ, info = T.make_split("subject" if setting == "subject" else "session",
                             None if setting == "subject" else "video", 0, 5, 49,
                             S["csess"], clab2, S["cpath"], 2)
    win_val = civ[S["cid"]]
    cid_va = S["cid"][win_val]
    vc = np.unique(cid_va)
    zo = np.load(os.path.join(orig_out, "val_clip_preds.npz"), allow_pickle=True)
    wo = np.load(os.path.join(orig_out, "val_window_preds.npz"), allow_pickle=True)
    check(f"{setting}: val clip paths == original's, element for element",
          np.array_equal(S["cpath"][vc], zo["path"]), f"{len(vc)} vs {len(zo['path'])} clips")
    check(f"{setting}: val clip binary labels == original's", np.array_equal(clab2[vc], zo["y"]))
    check(f"{setting}: val window -> clip map == original's", np.array_equal(cid_va, wo["cid"]),
          f"{len(cid_va)} windows")
    import torch
    wo_ = torch.load(os.path.join(orig_out, "best.pt"), map_location="cpu", weights_only=False)["model"]
    wm_ = torch.load(os.path.join(mine_out, "final.pt"), map_location="cpu", weights_only=False)["model"]
    check(f"{setting}: weights after {steps} AdamW steps bit-identical to the original's",
          set(wo_) == set(wm_) and all(torch.equal(wo_[n], wm_[n]) for n in wo_), f"{len(wo_)} tensors")
    zm = np.load(os.path.join(mine_out, "val_clip_ep01.npz"), allow_pickle=True)
    check(f"{setting}: dumped val clips are a subset of the split with the right y / y5",
          set(zm["path"].tolist()) <= set(zo["path"].tolist()) and
          np.array_equal(zm["y"], clab2[zm["clip_id"]]) and np.array_equal(zm["y5"], S["clab5"][zm["clip_id"]]))
    ref = ("output/v3_eegalign/tcn_g3_s1/val_clip_preds.npz" if setting == "aligned"
           else "output/v3_subject_cv/eeg_bin_fold0/val_clip_preds.npz")
    zr = np.load(os.path.join(EEG_ROOT, ref), allow_pickle=True)
    check(f"{setting}: val clip paths == stored run {os.path.dirname(ref)}",
          np.array_equal(zr["path"], S["cpath"][vc]), f"{len(zr['path'])} stored vs {len(vc)}")
    if rec["exception"]:
        print(f"    (original main() ended with {rec['exception']} after the records were taken)")


# ================================================================ stored checkpoint

def ckpt(a):
    print(f"\n== ckpt: {a.ckpt_run} evaluated through train_eeg_det's eval / pooling path", flush=True)
    import torch
    torch.set_num_threads(4)
    from torch.utils.data import DataLoader, TensorDataset
    import train_eeg_det as T
    run_dir = os.path.join(EEG_ROOT, a.ckpt_run)
    ck = torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu", weights_only=False)
    nc = ck["model"]["head.weight"].shape[0]
    arch = "tcn" if any(k.startswith("tcn.") for k in ck["model"]) else "gru"
    ai = json.load(open(os.path.join(run_dir, "align_info.json")))
    D = T.load_cache(CACHE)
    lab = T.GROUP_BY_NC[nc][D["clab5"]]
    wl = T.GROUP_BY_NC[nc][D["wlab5"]]
    civ, info = T.make_split("session", ai["session_universe"], 0, 5, ai["split_seed"],
                             D["csess"], lab, D["cpath"], nc)
    win_val = civ[D["cid"]]
    cid_va = D["cid"][win_val]
    vc = np.unique(cid_va)
    zc = np.load(os.path.join(run_dir, "val_clip_preds.npz"), allow_pickle=True)
    zw = np.load(os.path.join(run_dir, "val_window_preds.npz"), allow_pickle=True)
    check("stored val clip paths == this split's", np.array_equal(zc["path"], D["cpath"][vc]), f"{len(vc)} clips")
    check(f"stored y == this split's {nc}-class labels", np.array_equal(zc["y"], lab[vc]))
    check("stored window->clip map == this split's", np.array_equal(zw["cid"], cid_va))
    rng = np.random.default_rng(7)
    pick = np.sort(rng.choice(vc, min(a.ckpt_clips, len(vc)), replace=False))
    wsel = np.flatnonzero(np.isin(D["cid"], pick))
    X, cidp = D["segs"][wsel], D["cid"][wsel]
    model = T.tpe.build_model(arch, nc, 128)
    with torch.no_grad():
        model(torch.zeros(2, X.shape[1], 1))
    model.load_state_dict(ck["model"])
    dl = DataLoader(TensorDataset(torch.from_numpy(X).unsqueeze(-1), torch.from_numpy(wl[wsel])),
                    batch_size=512, shuffle=False)
    t = time.time()
    P = T.predict_windows(model, dl, torch.device("cpu"))
    cp = T.tpe.aggregate_clip(P, cidp, pick, "logmean")
    stored_w = zw["probs"][np.isin(zw["cid"], pick)]
    ci = np.searchsorted(vc, pick)
    # The stored posteriors were computed on an A100, where cuDNN convolutions run in TF32 by
    # default (10-bit mantissa): window differences of ~1e-3 at worst, ~2e-5 typically, are
    # arithmetic, not pipeline, differences. A pipeline error (z-score, window order, clip map,
    # pooling) moves posteriors by O(0.1); the negative control below (same windows, the
    # checkpoint of ANOTHER seed) shows what a real difference looks like.
    dwa = np.abs(P - stored_w)
    dw, dw50 = float(dwa.max()), float(np.median(dwa))
    dc = float(np.abs(cp - zc["probs"][ci]).max())
    agree = float((cp.argmax(1) == zc["pred"][ci]).mean())
    check(f"{len(pick)} clips / {len(wsel)} windows: window posteriors reproduce the stored GPU ones",
          dw < 5e-3 and dw50 < 1e-4, f"max|diff| {dw:.2e}, median {dw50:.2e} (TF32 on the GPU side; "
          f"{time.time() - t:.0f}s on CPU)")
    check("logmean clip posteriors reproduce the stored val_clip_preds", dc < 1e-3, f"max|diff| {dc:.2e}")
    check("clip argmax agrees with the stored pred on every clip", agree == 1.0, f"{agree:.4f}")
    other = a.ckpt_run.rstrip("/")[:-1] + ("2" if not a.ckpt_run.rstrip("/").endswith("2") else "3")
    if os.path.exists(os.path.join(EEG_ROOT, other, "best.pt")):
        model.load_state_dict(torch.load(os.path.join(EEG_ROOT, other, "best.pt"), map_location="cpu",
                                         weights_only=False)["model"])
        Pn = T.predict_windows(model, dl, torch.device("cpu"))
        dn = np.abs(Pn - stored_w)
        check(f"negative control: {os.path.basename(other)}'s weights on the same windows differ by far more",
              float(np.median(dn)) > 100 * dw50 and float(dn.max()) > 10 * dw,
              f"max|diff| {float(dn.max()):.2e}, median {float(np.median(dn)):.2e}")
    off = T.clip_offsets(cidp, pick)
    check("window offsets pack every clip's windows", int(off[-1]) == len(wsel) and len(off) == len(pick) + 1)
    # the binary P(seizure) the dumps carry, from the same windows
    pt = T.top_frac(1.0 - P[:, 0].astype(float), off, 0.25)
    psz = 1.0 - P[:, 0].astype(float)
    check("top-25% clip score lies within [min, max] of the clip's window P(seizure) and >= its mean",
          all(psz[off[i]:off[i + 1]].min() - 1e-9 <= pt[i] <= psz[off[i]:off[i + 1]].max() + 1e-9 and
              pt[i] >= psz[off[i]:off[i + 1]].mean() - 1e-9 for i in range(len(pick))))


# ================================================================ resume

def resume(a):
    print("\n== resume: uninterrupted vs interrupted+resumed runs must be bit-identical", flush=True)
    base = os.path.join(a.out, "resume")
    common = ["--split", "session", "--session_universe", "video", "--seed", "3", "--epochs", "3",
              "--limit_train", "3072", "--limit_val", "25", "--allow_cpu", "--threads", "4"]
    dirs = {k: os.path.join(base, k) for k in ("ref", "int", "ext")}
    for d in dirs.values():                   # fresh runs every time
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.endswith((".npz", ".pt", ".json", ".tmp")):
                    os.remove(os.path.join(d, f))
        os.makedirs(d, exist_ok=True)
    rc = run([PY, TRAINER, *common, "--output", dirs["ref"]], os.path.join(base, "ref.log"))
    check("uninterrupted run finishes", rc == 0)
    rc1 = run([PY, TRAINER, *common, "--debug_sigterm_at", "2:5", "--output", dirs["int"]],
              os.path.join(base, "int_1.log"))
    check("in-process SIGTERM mid-epoch 2 -> exit 3", rc1 == 3)
    rc2 = run([PY, TRAINER, *common, "--debug_sigterm_in_val", "3", "--output", dirs["int"]],
              os.path.join(base, "int_2.log"))
    check("resume, then SIGTERM before validating epoch 3 -> exit 3", rc2 == 3)
    rc3 = run([PY, TRAINER, *common, "--output", dirs["int"]], os.path.join(base, "int_3.log"))
    check("second resume finishes", rc3 == 0)
    # external kill -TERM once the first batch of epoch 1 is logged
    log = os.path.join(base, "ext_1.log")
    with open(log, "w") as fh:
        p = subprocess.Popen([PY, TRAINER, *common, "--output", dirs["ext"]], stdout=fh,
                             stderr=subprocess.STDOUT, env=ENV, cwd=EEG_ROOT)
    t0 = time.time()
    while time.time() - t0 < 900 and p.poll() is None:
        if "first batch:" in open(log).read():
            p.send_signal(signal.SIGTERM)
            break
        time.sleep(0.5)
    rce = p.wait()
    check("external SIGTERM during training -> exit 3", rce == 3, f"rc {rce}")
    rc4 = run([PY, TRAINER, *common, "--output", dirs["ext"]], os.path.join(base, "ext_2.log"))
    check("resume after external SIGTERM finishes", rc4 == 0)
    import torch
    for k in ("int", "ext"):
        same = True
        for ep in (1, 2, 3):
            A = np.load(os.path.join(dirs["ref"], f"val_clip_ep{ep:02d}.npz"), allow_pickle=True)
            B = np.load(os.path.join(dirs[k], f"val_clip_ep{ep:02d}.npz"), allow_pickle=True)
            for f in A.files:
                if not np.array_equal(A[f], B[f]):
                    same = False
                    print(f"    {k} ep{ep} {f} differs")
        check(f"{k}: every per-epoch dump bit-identical to the uninterrupted run", same)
        wa = torch.load(os.path.join(dirs["ref"], "final.pt"), weights_only=False)["model"]
        wb = torch.load(os.path.join(dirs[k], "final.pt"), weights_only=False)["model"]
        check(f"{k}: final weights bit-identical", all(torch.equal(wa[n], wb[n]) for n in wa))
        ha = json.load(open(os.path.join(dirs["ref"], "history.json")))
        hb = json.load(open(os.path.join(dirs[k], "history.json")))
        check(f"{k}: train losses identical", [h["train_loss"] for h in ha] == [h["train_loss"] for h in hb],
              str([round(h["train_loss"], 6) for h in hb]))
        rb = json.load(open(os.path.join(dirs[k], "results.json")))
        check(f"{k}: results.json reports the LAST epoch and logs the resumes",
              rb["reported_epoch"] == 3 and len(rb["resumes"]) >= 1, f"resumes {len(rb['resumes'])}")
    z = np.load(os.path.join(dirs["ref"], "val_clip_ep03.npz"), allow_pickle=True)
    need = {"path", "y", "y5", "p_logmean", "p_mean", "p_top25", "win_probs", "win_offsets"}
    check("dump carries path, y, y5, p_logmean, p_mean, p_top25, packed windows", need <= set(z.files))
    check("packed windows: offsets consistent", int(z["win_offsets"][-1]) == len(z["win_probs"]) and
          len(z["win_offsets"]) == len(z["path"]) + 1)


# ================================================================ refusals

def errors(a):
    print("\n== errors: refused configurations", flush=True)
    base = os.path.join(a.out, "errors")
    os.makedirs(base, exist_ok=True)
    cases = [("subject + session_universe video",
              ["--split", "subject", "--session_universe", "video", "--allow_cpu", "--output",
               os.path.join(a.out, "errors", "x")], "applies to --split session only"),
             ("session split without an explicit universe",
              ["--split", "session", "--allow_cpu", "--output", os.path.join(a.out, "errors", "x")],
              "--session_universe {eeg,video} chosen explicitly"),
             ("output outside output/ttg_eeg*",
              ["--split", "subject", "--allow_cpu", "--output", "/tmp/not_allowed"], "must live under"),
             ("subject fold outside 0..n_folds-1 (split_subjects would wrap it)",
              ["--split", "subject", "--fold", "5", "--allow_cpu", "--output",
               os.path.join(a.out, "errors", "x")], "0 <= fold < n_folds")]
    for name, args, msg in cases:
        log = os.path.join(base, "".join(c if c.isalnum() else "_" for c in name) + ".log")
        rc = run([PY, TRAINER, *args], log)
        txt = open(log).read()
        check(f"refused: {name}", rc != 0 and msg in txt, txt.strip().splitlines()[-1][:150] if txt.strip() else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("what", choices=["all", "firstbatch", "ckpt", "resume", "errors", "_orig"])
    ap.add_argument("--out", default=os.path.join(EEG_ROOT, "output", "ttg_eeg_test", "verify"))
    ap.add_argument("--setting", choices=["aligned", "subject", "both"], default="both")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--ckpt_run", default="output/v3_eegalign/tcn_g3_s1")
    ap.add_argument("--ckpt_clips", type=int, default=150)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--orig_out", default=None)
    ap.add_argument("--rec", default=None)
    a = ap.parse_args()
    os.chdir(EEG_ROOT)                  # CACHE and the parent trainer's defaults are EEG_ROOT-relative
    if a.what == "_orig":
        return orig_child(a)
    if not os.path.realpath(a.out).startswith(os.path.join(EEG_ROOT, "output", "ttg_eeg")):
        raise SystemExit("--out must live under output/ttg_eeg*/")
    os.makedirs(a.out, exist_ok=True)
    t = time.time()
    if a.what in ("all", "errors"):
        errors(a)
    if a.what in ("all", "firstbatch"):
        for s in (("aligned", "subject") if a.setting == "both" else (a.setting,)):
            firstbatch(a, s, a.seed, a.steps)
    if a.what in ("all", "ckpt"):
        ckpt(a)
    if a.what in ("all", "resume"):
        resume(a)
    n_fail = sum(not r["ok"] for r in RESULTS)
    summ = dict(what=a.what, n_checks=len(RESULTS), n_failed=n_fail, secs=round(time.time() - t),
                checks=RESULTS, finished=time.strftime("%F %T"))
    with open(os.path.join(a.out, f"verify_{a.what}.json"), "w") as f:
        json.dump(summ, f, indent=1)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed ({time.time() - t:.0f}s); "
          f"summary {a.out}/verify_{a.what}.json")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
