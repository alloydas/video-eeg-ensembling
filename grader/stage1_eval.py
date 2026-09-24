#!/usr/bin/env python3
"""
TT-X3D Stage-1 evaluator (CPU): X3D head fix, dual vs dedicated heads, rebuilt-cache control.

PURPOSE
  Reads the Stage-1 run directories written by grader/train_grader.py (listed in the config
  table grader/stage1.tsv) and reports, on the 5,289-clip seed-49 session-split val set:

  * per run and per arm 3-seed ensemble (posteriors averaged after aligning on path), for
    every head the run has (dual -> g3 and g5; dedicated -> one):
      - LAST epoch (the pre-registered number) and BEST epoch by val macro-F1 (the
        historical rule; selected on the scored set, so it is labelled an UPPER BOUND),
      - macro-F1 / precision / recall, MCC, per-class recall WITH counts, severe-group
        recall and precision, Stage-5 recall as x/34,
      - severe-vs-mild AUROC, pooled and within-session (pairs inside sessions holding both
        classes; score = log(P(severe)/P(mild))), last epoch;
  * the train-loss floor check (history.json, last epoch): a fixed run passes if its train
    loss is below half the bugged floor (0.59 g3 / 1.03 g5);
  * GATE (a) X3D fix: Stage-5 recall >= 5/34 in >= 2 of 3 fixed-dual seeds AND g5 macro-F1
    >= 0.63, both under the best-epoch rule (like-for-like with the stored bugged seeds,
    which were best-epoch selected). The 0.63 bar applies to the 3-seed MEAN of the per-seed
    best-epoch g5 macro-F1 values (each seed's value is reported too), not to the ensemble;
  * GATE (b) ranking: fixed-dual 3-seed mean LAST-epoch within-session severe AUROC within
    0.02 of the SlowFast single-run mean, per task. The SlowFast reference is recomputed via
    dhlib (decision-headroom; vendored as grader/dhlib.py) from stored val_preds: the 4
    output/v3_vidseeds/slowfast_{t}_s* runs PLUS vid_slowfast_{t} (seed 42) reproduce the
    pre-registered 0.831 (g3) / 0.811 (g5) and are the primary reference; the
    v3_vidseeds-only mean (4 runs) is reported too;
    Reading: one-sided non-inferiority (X3D >= reference - 0.02); the 5-run reference
    gates (it reproduces the pre-registered values and is the stricter bar);
  * GATE (c) heads: adopt dual heads if, for g3 and g5, the dual 3-seed mean last-epoch
    macro-F1 is >= the dedicated one - 0.01;
  Gates (a), (b) and (c) return None (not evaluable) unless every arm they use has 3
  complete seeds; partial numbers are still reported;
  * the rebuilt-cache shift: bugged-control (new f16s224, dual heads) vs the stored bugged
    X3D runs (old cache, dedicated heads; v3_vidseeds x3d_{t}_s{1,2,3,5} + vid_x3d_{t}),
    best-epoch rule on both sides. Heads differ as well as the cache, so this bounds the
    cache effect rather than isolating it. Stored runs are loaded with dhlib (path
    alignment, double-softmax repair); dropped and repaired counts are reported.

  Loading: every dump is aligned on the clip path ('/video.mp4' stripped) to the dhlib
  reference val order and its labels are asserted against the reference.

OUTPUT  <out> (default $EEG_ROOT/output/ttg_stage1/stage1_results.json)

USAGE (from the video-eeg-ensembling repo, any cwd; PYTHONDONTWRITEBYTECODE=1; EEG_ROOT env
       var, default /work/mech-ai-scratch/alloy/EEG. The script chdirs to EEG_ROOT, so a
       relative --out is resolved against EEG_ROOT; a relative --table is resolved against the
       directory you start it from, then against grader/, since the tables live in this repo.)
  python grader/stage1_eval.py                                  # grader/stage1.tsv, 12 epochs
  python grader/stage1_eval.py --table <tsv> --epochs 3 --out <json>   # e.g. synthetic runs
  python grader/stage1_eval.py --dry_run                        # list runs/epochs found only
"""
import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ttg_common as C                                               # noqa: E402

os.environ.setdefault("EEG_ROOT", C.EEG_ROOT)
if C.DH_DIR not in sys.path:
    sys.path.insert(0, C.DH_DIR)                   # grader/dhlib.py unless DHLIB_DIR is set
import dhlib as D                                                     # noqa: E402

FLOOR = {"g3": 0.59, "g5": 1.03}
DEF_OUT = os.path.join(C.EEG_ROOT, "output", "ttg_stage1", "stage1_results.json")
UB = "best epoch by val macro-F1 on the scored val set (historical rule): UPPER BOUND"


class Ref:
    def __init__(s):
        keys, _, y5 = D.reference_val("g5")
        s.keys, s.y5 = np.array(keys), np.array(y5)
        s.pos = {k: i for i, k in enumerate(s.keys)}
        s.an = np.array([C.animal_of(k) for k in s.keys])
        s.se = np.array([C.session_of(k) for k in s.keys])


def read_table(path):
    rows = []
    for line in open(path):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        f = line.split()
        rows.append(dict(name=f[0], arch=f[1], fix=f[2], heads=f[3], seed=int(f[4]), split=f[5],
                         fold=int(f[6]), out=f[7], extra=" ".join(f[8:])))
    return rows


def metrics(P, task, ref):
    y = ref.y5 if task == "g5" else C.GROUP3[ref.y5]
    pred = P.argmax(1)
    rep = C.cls_report(y, pred, task)
    sv = np.isin(y, C.SEV[task])
    spred = np.isin(pred, C.SEV[task])
    sz = np.isin(y, C.SEV[task] + C.MILD[task])
    sc = C.severity_logit(P, task)
    o = dict(macro_f1=rep["macro_f1"], macro_precision=rep["macro_precision"],
             macro_recall=rep["macro_recall"], mcc=rep["mcc"],
             per_class={k: dict(recall=round(v["recall"], 4), correct=v["n_correct"], n=v["n"])
                        for k, v in rep["per_class"].items()},
             severe_recall=float((sv & spred).sum() / max(sv.sum(), 1)),
             severe_precision=float((sv & spred).sum() / max(spred.sum(), 1)),
             severe_n=int(sv.sum()),
             sev_vs_mild_auroc_pooled=C.auroc(sc[sz], sv[sz]),
             sev_vs_mild_auroc_within_session=C.within_auroc(sc[sz], sv[sz], ref.se[sz], ref.an[sz]))
    if task == "g5":
        o["S5_correct"] = int(((y == 4) & (pred == 4)).sum())
        o["S5_n"] = int((y == 4).sum())
        o["S5_recall_str"] = f"{o['S5_correct']}/{o['S5_n']}"
        m = np.isin(y, [2, 3])
        s43 = np.log(np.clip(P[:, 3], 1e-12, None)) - np.log(np.clip(P[:, 2], 1e-12, None))
        o["S4_vs_S3_auroc_pooled"] = C.auroc(s43[m], y[m] == 3)
    return o


def load_run(row, ref, epochs, dry=False):
    out = row["out"]
    files = sorted(glob.glob(os.path.join(out, "val_ep*.npz")))
    eps = [int(re.search(r"val_ep(\d+)\.npz$", f).group(1)) for f in files]
    tasks = ["g3", "g5"] if row["heads"] == "dual" else [row["heads"]]
    r = dict(name=row["name"], out=out, tasks=tasks, epochs_found=eps,
             complete=bool(eps) and max(eps) >= epochs and os.path.exists(os.path.join(out, "results.json")))
    if dry or not files:
        r["status"] = "missing" if not files else ("complete" if r["complete"] else "incomplete")
        return r, None
    P = {t: {} for t in tasks}
    for e, f in zip(eps, files):
        z = np.load(f, allow_pickle=False)
        keys = np.array([C.key_of(p) for p in z["path"]])
        if len(keys) != len(ref.keys) or set(keys) != set(ref.pos):
            raise SystemExit(f"{f}: val clip set differs from the seed-49 reference (5,289)")
        order = np.array([ref.pos[k] for k in keys])
        y5 = np.empty_like(ref.y5)
        y5[order] = z["y5"]
        if not np.array_equal(y5, ref.y5):
            raise SystemExit(f"{f}: labels disagree with the reference after path alignment")
        for t in tasks:
            p = np.empty((len(keys), C.NCLS[t]))
            p[order] = z[f"probs_{t}"]
            P[t][e] = p
    r["status"] = "complete" if r["complete"] else "incomplete"
    last = epochs if epochs in eps else max(eps)
    r["last_epoch_used"] = last
    hist = {}
    hp = os.path.join(out, "history.json")
    if os.path.exists(hp):
        hist = {h["epoch"]: h for h in json.load(open(hp))}
    r["heads"] = {}
    for t in tasks:
        f1 = {e: metrics(P[t][e], t, ref)["macro_f1"] for e in eps}
        best = max(eps, key=lambda e: (f1[e], -e))              # first maximum (strict > rule)
        tl = hist.get(last, {}).get("train_loss", {}).get(t)
        r["heads"][t] = dict(
            macro_f1_by_epoch={e: round(v, 4) for e, v in f1.items()},
            last=dict(epoch=last, **metrics(P[t][last], t, ref)),
            best_upper_bound=dict(epoch=best, note=UB, **metrics(P[t][best], t, ref)),
            train_loss_last=tl,
            train_loss_below_half_bugged_floor=(None if tl is None else bool(tl < 0.5 * FLOOR[t])),
            bugged_floor=FLOOR[t])
    return r, dict(P=P, last=last, best={t: r["heads"][t]["best_upper_bound"]["epoch"] for t in tasks})


def mean_sd(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    if not v:
        return [None, None, 0]
    sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0       # sample sd, as the parent tables
    return [round(float(np.mean(v)), 4), round(sd, 4), len(v)]


def stored_family(prefix_fmt, task, ref, seeds=(1, 2, 3, 5), seed42_dir=None):
    """Stored runs via dhlib (path alignment, double-softmax repair)."""
    keys, y, _ = D.reference_val(task)
    raw_y = np.load(os.path.join(C.EEG_ROOT, f"output/v3_vidseeds/mvit_{task}_s1/val_preds.npz"))["y"]
    dirs = [prefix_fmt.format(task=task, seed=s) for s in seeds]
    if seed42_dir:
        dirs.append(seed42_dir.format(task=task))
    Pm, names, log = D.load_members(dirs, keys, raw_y, y)
    assert np.array_equal(np.array(keys), ref.keys)
    return Pm, names, {k: len(v) for k, v in log.items()}, log


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", default=os.path.join(C.GRADER_DIR, "stage1.tsv"),
                    help="config table (relative: to the current directory, then to grader/; "
                         "default grader/stage1.tsv)")
    ap.add_argument("--epochs", type=int, default=12, help="the pre-registered last epoch")
    ap.add_argument("--out", default=DEF_OUT)
    ap.add_argument("--dry_run", action="store_true", help="list runs and epochs found; write nothing")
    a = ap.parse_args()
    t = a.table                                    # a table lives in this repo: caller's cwd,
    if not os.path.isabs(t) and not os.path.isfile(t) and os.path.isfile(os.path.join(C.GRADER_DIR, t)):
        t = os.path.join(C.GRADER_DIR, t)          # then grader/ (as the shell drivers do)
    a.table = os.path.abspath(t)
    if not os.path.isfile(a.table):
        raise SystemExit(f"no config table {a.table} (a relative --table is looked up in the "
                         f"current directory, then in {C.GRADER_DIR})")
    C.enter_eeg_root()                             # every other relative path: EEG_ROOT's
    rows = read_table(a.table)
    ref = Ref()
    assert len(ref.keys) == 5289
    s5_n = int((ref.y5 == 4).sum())

    runs, raw = {}, {}
    for row in rows:
        r, rw = load_run(row, ref, a.epochs, dry=a.dry_run)
        runs[row["name"]] = dict(r, arch=row["arch"], fix=row["fix"], head_mode=row["heads"], seed=row["seed"])
        raw[row["name"]] = rw
        print(f"  {row['name']:18s} {r['status']:10s} epochs {r['epochs_found'][:1]}..{r['epochs_found'][-1:]}")
    if a.dry_run:
        print("--dry_run: nothing written")
        return
    C.check_output_dir(os.path.dirname(os.path.abspath(a.out)))

    # ---- arms: 3-seed ensembles and single-run summaries
    arms = {}
    for n, r in runs.items():
        arms.setdefault(f"{r['arch']}-{r['fix']}-{r['head_mode']}", []).append(n)
    arm_res = {}
    for arm, names in arms.items():
        ok = [n for n in names if runs[n]["status"] == "complete"]
        tasks = runs[names[0]]["tasks"]
        A = dict(runs=names, complete_runs=ok, n_complete=len(ok), heads={})
        for t in tasks:
            h = dict()
            for rule, key in (("last", "last"), ("best_upper_bound", "best_upper_bound")):
                vals = [runs[n]["heads"][t][key] for n in ok]
                h[f"single_run_{rule}"] = {
                    m: mean_sd([v[m] for v in vals]) for m in
                    ("macro_f1", "mcc", "severe_recall", "severe_precision",
                     "sev_vs_mild_auroc_within_session", "sev_vs_mild_auroc_pooled")}
                if t == "g5":
                    h[f"single_run_{rule}"]["S5_correct_per_seed"] = [v["S5_correct"] for v in vals]
            if ok:
                Pl = np.mean([raw[n]["P"][t][raw[n]["last"]] for n in ok], 0)
                Pb = np.mean([raw[n]["P"][t][raw[n]["best"][t]] for n in ok], 0)
                h["ensemble_last"] = metrics(Pl, t, ref)
                h["ensemble_best_upper_bound"] = dict(note=UB, **metrics(Pb, t, ref))
            A["heads"][t] = h
        arm_res[arm] = A

    def arm(k):
        return arm_res.get(k, dict(n_complete=0, heads={}))

    gates = {}
    # ---- (a) X3D fix
    fd = arm("x3d-fixed-dual")
    if fd["n_complete"] >= 1 and "g5" in fd["heads"]:
        s5 = fd["heads"]["g5"]["single_run_best_upper_bound"]["S5_correct_per_seed"]
        f1 = fd["heads"]["g5"]["single_run_best_upper_bound"]["macro_f1"][0]
        per = [runs[n]["heads"]["g5"]["best_upper_bound"]["macro_f1"] for n in fd["complete_runs"]]
        gates["a_x3d_fix"] = dict(
            rule="S5 recall >= 5/34 in >= 2 of 3 fixed-dual seeds AND mean g5 macro-F1 >= 0.63 "
                 "(best-epoch rule, like-for-like with the stored bugged seeds)",
            reading="'g5 macro-F1 >= 0.63' is applied to the 3-seed MEAN of the per-seed "
                    "best-epoch macro-F1 values (listed per seed below), not to each seed and "
                    "not to the 3-seed ensemble; evaluable only with 3 complete seeds",
            S5_correct_per_seed=[f"{x}/{s5_n}" for x in s5], seeds_with_S5_ge_5=int(sum(x >= 5 for x in s5)),
            g5_macro_f1_per_seed=[round(v, 4) for v in per], g5_macro_f1_mean=f1,
            n_seeds=fd["n_complete"],
            passed=(bool(sum(x >= 5 for x in s5) >= 2 and f1 >= 0.63) if fd["n_complete"] == 3 else None),
            last_epoch_view=dict(
                S5_correct_per_seed=fd["heads"]["g5"]["single_run_last"]["S5_correct_per_seed"],
                g5_macro_f1_mean=fd["heads"]["g5"]["single_run_last"]["macro_f1"][0]))

    # ---- (b) ranking vs SlowFast (recomputed from stored val_preds via dhlib)
    sfref, bref = {}, {}
    for t in ("g3", "g5"):
        Pm, names, cnt, _ = stored_family("output/v3_vidseeds/slowfast_{task}_s{seed}", t, ref,
                                          seed42_dir="vid_slowfast_{task}")
        w = [metrics(p, t, ref)["sev_vs_mild_auroc_within_session"] for p in Pm]
        v3 = [wi for wi, nm in zip(w, names) if nm.startswith("output/v3_vidseeds/")]
        sfref[t] = dict(runs=names, load_counts=cnt, within_per_run=[round(x, 4) for x in w],
                        mean_5run=round(float(np.mean(w)), 4), mean_v3_vidseeds_only=round(float(np.mean(v3)), 4),
                        preregistered={"g3": 0.831, "g5": 0.811}[t])
    gb = dict(rule="fixed-dual 3-seed mean LAST-epoch within-session severe AUROC >= SlowFast "
                   "single-run mean - 0.02 (primary ref: 5 runs incl. seed 42, reproduces 0.831/0.811)",
              reading="one-sided non-inferiority: 'within 0.02' passes when X3D >= ref - 0.02 "
                      "(X3D above the reference always passes). The GATING reference is the "
                      "5-run SlowFast mean (4 output/v3_vidseeds/slowfast_{t}_s{1,2,3,5} + the "
                      "path-less vid_slowfast_{t} seed-42 run, admitted by dhlib because its "
                      "raw label sequence equals the reference order): it is the set that "
                      "reproduces the pre-registered 0.831 / 0.811 and it is the stricter bar. "
                      "The 4-run v3_vidseeds-only mean the spec names (0.8265 / 0.8078 when "
                      "this was written) is reported as passed_vs_v3only_ref, not gated on. "
                      "Evaluable only with 3 complete fixed-dual seeds",
              slowfast_reference=sfref, per_task={})
    for t in ("g3", "g5"):
        row = {}
        for armk in ("x3d-fixed-dual", f"x3d-fixed-{t}", "x3d-bugged-dual"):
            A = arm(armk)
            if A["n_complete"] and t in A["heads"]:
                m = A["heads"][t]["single_run_last"]["sev_vs_mild_auroc_within_session"][0]
                row[armk] = dict(mean_last_within=m, n=A["n_complete"],
                                 minus_ref_5run=round(m - sfref[t]["mean_5run"], 4),
                                 minus_ref_v3only=round(m - sfref[t]["mean_v3_vidseeds_only"], 4))
        fdm = row.get("x3d-fixed-dual")
        row["passed"] = (bool(fdm["minus_ref_5run"] >= -0.02) if fdm and fdm["n"] == 3 else None)
        row["passed_vs_v3only_ref"] = (bool(fdm["minus_ref_v3only"] >= -0.02) if fdm and fdm["n"] == 3 else None)
        gb["per_task"][t] = row
    gb["passed"] = (all(gb["per_task"][t]["passed"] for t in ("g3", "g5"))
                    if all(gb["per_task"][t]["passed"] is not None for t in ("g3", "g5")) else None)
    gates["b_ranking_vs_slowfast"] = gb

    # ---- (c) dual vs dedicated (numbers shown whenever both arms have a run; the decision
    #      needs 3 complete seeds in BOTH arms, like gates (a) and (b))
    gc = dict(rule="adopt dual if dual 3-seed mean LAST-epoch macro-F1 >= dedicated - 0.01 for both g3 and g5",
              reading="evaluable only with 3 complete seeds in the dual arm and in each dedicated "
                      "arm; with fewer, per-task deltas are informational and adopt_dual is None",
              per_task={})
    for t in ("g3", "g5"):
        du, de = arm("x3d-fixed-dual"), arm(f"x3d-fixed-{t}")
        if du["n_complete"] and de["n_complete"] and t in du["heads"] and t in de["heads"]:
            a1 = du["heads"][t]["single_run_last"]
            b1 = de["heads"][t]["single_run_last"]
            gc["per_task"][t] = dict(
                dual_macro_f1=a1["macro_f1"], dedicated_macro_f1=b1["macro_f1"],
                delta_macro_f1=round(a1["macro_f1"][0] - b1["macro_f1"][0], 4),
                delta_within_auroc=round(a1["sev_vs_mild_auroc_within_session"][0] -
                                         b1["sev_vs_mild_auroc_within_session"][0], 4),
                delta_severe_recall=round(a1["severe_recall"][0] - b1["severe_recall"][0], 4),
                ensemble_last_delta_macro_f1=round(du["heads"][t]["ensemble_last"]["macro_f1"] -
                                                   de["heads"][t]["ensemble_last"]["macro_f1"], 4),
                n=(du["n_complete"], de["n_complete"]),
                evaluable=bool(du["n_complete"] == 3 and de["n_complete"] == 3))
    done = [t for t in ("g3", "g5") if gc["per_task"].get(t, {}).get("evaluable")]
    gc["adopt_dual"] = (all(gc["per_task"][t]["delta_macro_f1"] >= -0.01 for t in done)
                        if len(done) == 2 else None)
    gates["c_dual_vs_dedicated"] = gc

    # ---- (d) rebuilt-cache control vs stored bugged X3D
    gd = dict(note="bugged control = new data_full f16s224 cache AND dual heads; stored = old "
                   "microway cache, dedicated heads. Both best-epoch selected (upper bounds). "
                   "The shift bounds the cache effect; it does not isolate it.", per_task={})
    bc = arm("x3d-bugged-dual")
    for t in ("g3", "g5"):
        Pm, names, cnt, log = stored_family("output/v3_vidseeds/x3d_{task}_s{seed}", t, ref,
                                            seed42_dir="vid_x3d_{task}")
        ms = [metrics(p, t, ref) for p in Pm]
        stored = dict(runs=names, load_counts=cnt,
                      macro_f1=mean_sd([m["macro_f1"] for m in ms]),
                      severe_recall=mean_sd([m["severe_recall"] for m in ms]),
                      within=mean_sd([m["sev_vs_mild_auroc_within_session"] for m in ms]),
                      pooled=mean_sd([m["sev_vs_mild_auroc_pooled"] for m in ms]))
        if t == "g5":
            stored["S5_correct_per_run"] = [m["S5_correct"] for m in ms]
        bref[t] = ms
        row = dict(stored_bugged=stored)
        if bc["n_complete"] and t in bc["heads"]:
            c = bc["heads"][t]["single_run_best_upper_bound"]
            row["control_best_epoch"] = dict(macro_f1=c["macro_f1"], severe_recall=c["severe_recall"],
                                             within=c["sev_vs_mild_auroc_within_session"],
                                             pooled=c["sev_vs_mild_auroc_pooled"])
            row["shift_control_minus_stored"] = dict(
                macro_f1=round(c["macro_f1"][0] - stored["macro_f1"][0], 4),
                severe_recall=round(c["severe_recall"][0] - stored["severe_recall"][0], 4),
                within=round(c["sev_vs_mild_auroc_within_session"][0] - stored["within"][0], 4))
        fdm = arm("x3d-fixed-dual")
        if fdm["n_complete"] and bc["n_complete"]:
            f_ = fdm["heads"][t]["single_run_last"]
            b_ = bc["heads"][t]["single_run_last"]
            row["fixed_minus_bugged_control_last_epoch"] = dict(
                macro_f1=round(f_["macro_f1"][0] - b_["macro_f1"][0], 4),
                within=round(f_["sev_vs_mild_auroc_within_session"][0] - b_["sev_vs_mild_auroc_within_session"][0], 4),
                pooled=round(f_["sev_vs_mild_auroc_pooled"][0] - b_["sev_vs_mild_auroc_pooled"][0], 4))
        gd["per_task"][t] = row
    gates["d_cache_rebuild_shift"] = gd

    # ---- train-loss floor check
    floor = {n: {t: dict(train_loss_last=r["heads"][t]["train_loss_last"],
                         below_half_bugged_floor=r["heads"][t]["train_loss_below_half_bugged_floor"])
                 for t in r.get("heads", {})} for n, r in runs.items()}
    fixed_ok = [v["below_half_bugged_floor"] for n, r in runs.items() if r["fix"] == "fixed"
                for v in floor[n].values()]
    gates["train_loss_floor"] = dict(
        rule=f"fixed runs: last-epoch train loss < 0.5 x bugged floor {FLOOR}",
        per_run=floor, all_fixed_pass=(all(fixed_ok) if fixed_ok and None not in fixed_ok else None))

    res = dict(created=time.strftime("%F %T"), table=a.table, expected_epochs=a.epochs,
               val=dict(n=len(ref.keys), S5_n=s5_n, severe_n=int(np.isin(ref.y5, [3, 4]).sum()),
                        class_counts_g5=np.bincount(ref.y5, minlength=5).tolist()),
               runs=runs, arms=arm_res, gates=gates)
    C.atomic_json(a.out, res)

    # ---- console summary
    print(f"\nval: {len(ref.keys)} clips, severe {res['val']['severe_n']}, S5 {s5_n}")
    for armk, A in arm_res.items():
        for t, h in A["heads"].items():
            L, B = h["single_run_last"], h["single_run_best_upper_bound"]
            s5 = f"  S5 last {L.get('S5_correct_per_seed')} best {B.get('S5_correct_per_seed')}" if t == "g5" else ""
            print(f"{armk:18s} {t} n={A['n_complete']}  macroF1 last {L['macro_f1'][0]} (best-UB "
                  f"{B['macro_f1'][0]})  within {L['sev_vs_mild_auroc_within_session'][0]}  "
                  f"severe recall {L['severe_recall'][0]}{s5}")
    for k, g in gates.items():
        if k == "d_cache_rebuild_shift":
            sh = {t: g["per_task"][t].get("shift_control_minus_stored") for t in g["per_task"]}
            print(f"REPORT {k} (control - stored, best-epoch): {sh}")
            continue
        p = g.get("passed", g.get("adopt_dual", g.get("all_fixed_pass")))
        print(f"GATE {k}: {p}  (None = not evaluable: runs missing/incomplete)")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
