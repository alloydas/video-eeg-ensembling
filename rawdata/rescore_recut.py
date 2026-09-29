#!/usr/bin/env python3
"""Re-score the EEG-Gated Racine Grader with the re-cut clips, WITHOUT retraining.

The video of 81 labelled clips was cut from the wrong time and has been re-cut at the right time
(rawdata/recut_clips.py -> $EEG_ROOT/output/ttg_recut/data_full/, manifest.csv). 11 more clips are
excluded (ttg_recut/exclude.csv: 5 with no video recorded, 5 unresolved, 1 non-seizure clip that holds
an annotated seizure). The EEG and the labels were always right, so only the video posteriors change.

Stages (run in order; each reads the previous stage's output):
  cache     16 x 224 frame cache of the 81 re-cut clips, with the f16s224 rule (ttg_common.decode_linspace:
            header CAP_PROP_FRAME_COUNT -> train_classifier.sample_frame_indices(n, 16) -> sequential
            grab/retrieve -> BGR->RGB -> INTER_AREA 224x224), keyed by the clips' data/ keys, into
            ttg_recut/cache_f16s224 (frames.u8 + index.json, the train_pooled.FrameCache layout).
            Checks, all bit-for-bit: every built row against train_classifier.load_clip on its source;
            3 unaffected clips rebuilt from data_full/ against cache_frames/f16s224; the 5 reproduction
            controls (ttg_recut/repro/, re-cut at their ORIGINAL seek) against cache_frames/f16s224.
  infer     for each of the 30 video runs output/ttg_vsubj/{x3dfix,x3dbug}_dual_s{1,2,3}_fold{0..4}: load
            last.pt (checked to be the state after epoch 12 of 12), rebuild train_grader.GraderNet and run
            the validation forward pass of train_grader.py (eval mode, tp.norm_batch, tp._autocast, both
            heads, train_grader.head_probs, float32) on
              - 5 unaffected validation clips of the run (equivalence controls, vs the stored rows),
              - the run's re-cut validation clips from cache_frames/f16s224 (the ORIGINAL frames: a second
                equivalence check on exactly the rows that get replaced),
              - the same clips from ttg_recut/cache_f16s224 (the CORRECTED frames: the new predictions).
            CPU only. --precision fp32 is train_grader's own CPU path; fp16 is CPU fp16 autocast.
  patch     ttg_recut/vsubj_patched/<run>/val_ep12.npz: the stored dump with the re-cut rows' probs_g3 /
            probs_g5 replaced by the new predictions and the excluded clips' rows removed; every other
            field kept. ttg_recut/vsubj_exclonly/<run>/: the exclusions only (the control that separates
            re-cutting from dropping clips). config.json is copied so joint_gate labels the epoch as the
            pre-registered last one. EEG dumps are not touched.
  grid      grader/eeg/joint_gate.py, run in-process with the stored seed-grid argv
            (ttg_eeg_gate/seeds/B_<recipe>_grid3x3/results.json) but --video pointing at the variant's
            dirs, into ttg_recut/grid/<prefix>B_<recipe>_grid3x3. Two in-process adaptations, both logged:
            the --out prefix is widened to ttg_recut/grid/, and for the variants that drop clips the
            per-fold "video dump holds exactly the fold's clips" check expects the fold MINUS the
            excluded clips (which are named). variant 'orig' reads the stored dumps with no adaptation
            other than --out, to prove the wrapper reproduces the stored grid.
  analyse   before (stored dumps) / exclusion-only / after (patched), per recipe: every single video x single
            EEG pair's video and EGRG decisions (the joint_gate code: Pairing.systems, LOAO gate), per-class
            hits with counts, severe hits, within-session severe-vs-mild AUROC, macro-F1, and how many
            clip predictions changed class; the per-pair macro-F1 is asserted equal to the grids'.
            Also: the re-cut clips' video predictions before -> after, and how many affected clips sit in
            the TRAINING folds of each run. -> ttg_recut/rescore/summary.{json,txt}
  compare   ttg_recut/grid/repro_B_<recipe>_grid3x3 against the stored grid: every field equal except the
            run's timestamps, argv, run time and the --out-dependent strings.

Writes ONLY under $EEG_ROOT/output/ttg_recut/ (out_path() refuses anything else). Reads the stored runs, the
f16s224 cache and data_full/ read-only. No GPU, no training, no git.

Environment: the eeg conda env (torch, cv2, sklearn): /work/mech-ai-scratch/alloy/.conda/envs/eeg/bin/python,
PYTHONDONTWRITEBYTECODE=1. rawdata/submit_rescore.sh runs the stages on the CPU scavenger QoS.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import sys
import time
import zlib
from collections import Counter, OrderedDict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
GRADER = os.path.join(REPO, "grader")
sys.path.insert(0, GRADER)
import ttg_common as C  # noqa: E402  (puts EEG_ROOT on sys.path; decode rule; atomic io)

EEG_ROOT = C.EEG_ROOT
RECUT = os.path.join(EEG_ROOT, "output", "ttg_recut")
CACHE = os.path.join(RECUT, "cache_f16s224")
INFER = os.path.join(RECUT, "infer")
PATCHED = os.path.join(RECUT, "vsubj_patched")
EXCLONLY = os.path.join(RECUT, "vsubj_exclonly")
GRID = os.path.join(RECUT, "grid")
RESCORE = os.path.join(RECUT, "rescore")
VSUBJ = os.path.join(EEG_ROOT, "output", "ttg_vsubj")
F16 = os.path.join(EEG_ROOT, "cache_frames", "f16s224")
STORED_GRID = os.path.join(EEG_ROOT, "output", "ttg_eeg_gate", "seeds")
RECIPES = ("x3dfix", "x3dbug")
SEEDS = (1, 2, 3)
FOLDS = (0, 1, 2, 3, 4)
EPOCH = 12
T, S = 16, 224
TASKS = ("g3", "g5")
VARIANTS = {"orig": VSUBJ, "exclonly": EXCLONLY, "patched": PATCHED}
GRID_PREFIX = {"orig": "repro_", "exclonly": "exclonly_", "patched": ""}


def runs():
    return [f"{r}_dual_s{s}_fold{f}" for r in RECIPES for s in SEEDS for f in FOLDS]


def _under(p, root):
    p, root = os.path.realpath(p), os.path.realpath(root)
    return p == root or p.startswith(root + os.sep)


def out_path(*parts):
    """Every write goes through here: under ttg_recut/ only, and never into its re-cut clips or repro/."""
    p = os.path.join(RECUT, *parts)
    if not _under(p, RECUT) or _under(p, os.path.join(RECUT, "data_full")) or \
            _under(p, os.path.join(RECUT, "repro")) or _under(p, os.path.join(EEG_ROOT, "cache_frames")):
        raise SystemExit(f"refusing to write outside {RECUT} (or into its clips): {p}")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def sha256(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ============================================================================ inputs

def read_manifest():
    """The 81 re-cut clips: data/ key -> absolute re-cut source (status ok only; anything else refuses)."""
    rows = list(csv.DictReader(open(os.path.join(RECUT, "manifest.csv"))))
    bad = [r["clip_key"] for r in rows if r["status"] != "ok"]
    if bad:
        raise SystemExit(f"manifest rows not ok: {bad[:3]}")
    out = OrderedDict()
    for r in sorted(rows, key=lambda r: r["clip_key"]):
        src = os.path.join(EEG_ROOT, r["out"])
        if not _under(src, os.path.join(RECUT, "data_full")):
            raise SystemExit(f"re-cut source outside ttg_recut/data_full: {src}")
        out[r["clip_key"]] = dict(src=src, y=int(r["y"]), cause=r["cause"], sub=r["sub"],
                                  offset_s=float(r["offset_used_s"]), source=r["offset_source"],
                                  truncated_s=float(r["truncated_s"] or 0))
    return out


def read_exclude():
    rows = list(csv.DictReader(open(os.path.join(RECUT, "exclude.csv"))))
    return OrderedDict((r["key"], dict(y=int(r["y"]), status=r["status"], cause=r["cause"], reason=r["reason"]))
                       for r in sorted(rows, key=lambda r: r["key"]))


def read_candidates():
    """Every clip the re-cut step looked at (172) plus the excluded ones: never used as an unaffected control."""
    c = {r["key"] for r in csv.DictReader(open(os.path.join(RECUT, "plan", "candidates.csv")))}
    return c | set(read_exclude()) | set(read_manifest())


def f16_index():
    meta = json.load(open(os.path.join(F16, "index.json")))
    mm = np.memmap(os.path.join(F16, "frames.u8"), dtype=np.uint8, mode="r",
                   shape=(meta["n"], meta["frames"], meta["size"], meta["size"], 3))
    return meta, {p: i for i, p in enumerate(meta["paths"])}, mm


# ============================================================================ stage: cache

def stage_cache(a):
    man = read_manifest()
    keys = list(man)
    meta16, row16, mm16 = f16_index()
    known = set(meta16["paths"])
    unknown = [k for k in keys if k not in known]
    if unknown:
        raise SystemExit(f"{len(unknown)} re-cut keys are not f16s224 / discover() clips: {unknown[:3]}")
    bad16 = set(meta16.get("unreadable", []))
    print(f"[cache] {len(keys)} re-cut clips; all are discover() / f16s224 keys "
          f"({sum(k in bad16 for k in keys)} marked unreadable there)", flush=True)
    idx_path = out_path("cache_f16s224", "index.json")
    if os.path.exists(idx_path) and not a.overwrite:
        raise SystemExit(f"{idx_path} exists; pass --overwrite to rebuild")
    n, rowbytes = len(keys), T * S * S * 3
    part = out_path("cache_f16s224", "frames.u8.partial")
    info, t0 = {}, time.time()
    with open(part, "wb") as f:
        for i, k in enumerate(keys):
            arr, inf = C.decode_linspace(man[k]["src"], T, S, allow_substitution=False)
            b = np.ascontiguousarray(arr).tobytes()
            assert len(b) == rowbytes
            f.write(b)
            info[k] = dict(n_header=inf["n_header"], n_substituted=inf["n_substituted"],
                           src=os.path.relpath(man[k]["src"], EEG_ROOT))
        f.flush()
        os.fsync(f.fileno())
    print(f"[cache] decoded {n} clips in {time.time() - t0:.0f}s", flush=True)

    # check 1: every built row == train_classifier.load_clip(source) (the historical decoder)
    from train_classifier import load_clip
    mm = np.memmap(part, dtype=np.uint8, mode="r", shape=(n, T, S, S, 3))
    mism = []
    for i, k in enumerate(keys):
        ref = load_clip(man[k]["src"], T, S, raw=True)
        if ref is None or not np.array_equal(ref.transpose(1, 2, 3, 0), mm[i]):
            mism.append(k)
    print(f"[cache] check 1: {n - len(mism)}/{n} rows bit-identical to train_classifier.load_clip(re-cut source)",
          flush=True)
    if mism:
        raise SystemExit(f"ABORT: {len(mism)} rows differ from load_clip, e.g. {mism[0]}")
    # check 1b: how different the corrected frames are from the old ones (informational)
    diff = {k: float(np.abs(mm[i].astype(np.int16) - mm16[row16[k]].astype(np.int16)).mean())
            for i, k in enumerate(keys)}
    same = [k for k, v in diff.items() if v == 0.0]
    del mm

    # check 2: the rule reproduces cache_frames/f16s224 on 3 unaffected clips decoded from data_full/
    cand = read_candidates()
    pool = sorted(p for p in meta16["paths"] if p not in cand and p not in bad16)
    rule = []
    for k in random.Random(20260929).sample(pool, 3):
        arr, inf = C.decode_linspace(os.path.join(EEG_ROOT, C.to_full(k)), T, S, allow_substitution=False)
        rule.append(dict(key=k, n_header=inf["n_header"],
                         bit_identical=bool(np.array_equal(arr, mm16[row16[k]])),
                         max_abs_diff=int(np.abs(arr.astype(np.int16) - mm16[row16[k]].astype(np.int16)).max())))
    # check 3: the whole chain (re-cut code at the ORIGINAL seek + this rule) reproduces f16s224
    chain = []
    for r in csv.DictReader(open(os.path.join(RECUT, "verify", "repro.csv"))):
        k = r["key"]
        src = os.path.join(RECUT, "repro", C.to_full(k))
        arr, inf = C.decode_linspace(src, T, S, allow_substitution=False)
        chain.append(dict(key=k, n_header=inf["n_header"], bit_identical=bool(np.array_equal(arr, mm16[row16[k]])),
                          max_abs_diff=int(np.abs(arr.astype(np.int16) - mm16[row16[k]].astype(np.int16)).max())))
    for tag, rr in (("check 2 (rule, data_full/ source)", rule), ("check 3 (re-cut chain, repro/ source)", chain)):
        print(f"[cache] {tag}: {sum(x['bit_identical'] for x in rr)}/{len(rr)} bit-identical to "
              f"cache_frames/f16s224", flush=True)
        for x in rr:
            print(f"     {x['bit_identical']}  max|diff| {x['max_abs_diff']}  n={x['n_header']}  {x['key']}")
    if not all(x["bit_identical"] for x in rule + chain):
        raise SystemExit("ABORT: the rule does not reproduce cache_frames/f16s224; no index written")

    os.replace(part, out_path("cache_f16s224", "frames.u8"))
    C.atomic_json(idx_path, dict(
        paths=keys, frames=T, size=S, n=n, unreadable=[], substituted={},
        source="ttg_recut/data_full/<rel> (re-cut clips) decoded, keys kept as data/<rel>",
        index_rule="sample_frame_indices(header CAP_PROP_FRAME_COUNT, T); sequential grab/retrieve; BGR->RGB; "
                   "INTER_AREA (ttg_common.decode_linspace, as grader/build_f16_cache.py)",
        built_by="video-eeg-ensembling/rawdata/rescore_recut.py cache", built_at=time.strftime("%F %T"),
        clips=info, fingerprint=C.fingerprint(keys) + f":{T}x{S}",
        checks=dict(load_clip_bit_identical=f"{n}/{n}", rule_vs_f16s224=rule, recut_chain_vs_f16s224=chain,
                    mean_abs_diff_vs_old_rows=dict(min=min(diff.values()), median=float(np.median(list(diff.values()))),
                                                   max=max(diff.values()), n_identical_to_old=len(same)))))
    print(f"[cache] wrote {idx_path}; mean |new - old| grey levels per clip: median "
          f"{np.median(list(diff.values())):.2f}, min {min(diff.values()):.2f}; {len(same)} identical to the old row",
          flush=True)


# ============================================================================ stage: infer

class Predictor:
    """train_grader.py's validation forward pass for one run's epoch-12 weights, on CPU."""

    def __init__(s, run, precision, threads):
        import torch
        import train_grader as TG                      # grader/train_grader.py (main() is not run)
        import train_pooled as tp
        s.torch, s.TG, s.tp = torch, TG, tp
        torch.set_num_threads(threads)
        d = os.path.join(VSUBJ, run)
        cfg = json.load(open(os.path.join(d, "config.json")))
        rk = cfg["run_key"]
        ck = torch.load(os.path.join(d, "last.pt"), map_location="cpu", weights_only=False)
        st = ck["state"]
        hist = json.load(open(os.path.join(d, "history.json")))
        z = np.load(os.path.join(d, f"val_ep{EPOCH:02d}.npz"), allow_pickle=True)
        chk = dict(run_key_equal=ck["run_key"] == rk, epochs=rk["epochs"], next_epoch=st["next_epoch"],
                   train_done=st["train_done"], next_step=st["next_step"],
                   history_epochs=[h["epoch"] for h in st["history"]],
                   history_json_last=max(h["epoch"] for h in hist), dump_epoch=int(z["epoch"]),
                   saved_at=ck.get("saved_at"), arch=rk["arch"], fix_x3d=rk["fix_x3d"], heads=rk["heads"],
                   cache_dir=rk["cache_dir"])
        ok = (chk["run_key_equal"] and rk["epochs"] == EPOCH and st["next_epoch"] == EPOCH + 1 and
              not st["train_done"] and st["next_step"] == 0 and max(chk["history_epochs"]) == EPOCH and
              chk["history_json_last"] == EPOCH and chk["dump_epoch"] == EPOCH and rk["arch"] == "x3d" and
              rk["heads"] == "dual" and os.path.realpath(rk["cache_dir"]) == os.path.realpath(F16))
        chk["is_epoch12_state"] = bool(ok)
        if not ok:
            raise SystemExit(f"{run}: last.pt is not the post-epoch-{EPOCH} state: {chk}")
        s.check = chk
        s.dev = torch.device("cpu")
        m = TG.GraderNet("x3d", ["g3", "g5"], rk["fix_x3d"], T, S, pretrained=False, logit_bound=0.0)
        m.load_state_dict(ck["model"], strict=True)
        m.eval()
        if bool(z["head_softmax_bug"]) != m.head_softmax_bug:
            raise SystemExit(f"{run}: head_softmax_bug mismatch between dump and model")
        s.model = m
        s.precision = precision

    def predict(s, cache_dir, keys, bs=8):
        """The val loop of train_grader.py: FrameCache.fetch -> stack -> norm_batch -> model under autocast ->
        head_probs -> float32. precision fp32 = tp._autocast(cpu) (disabled), fp16 = CPU fp16 autocast."""
        torch, tp, TG = s.torch, s.tp, s.TG
        fc = tp.FrameCache.get(cache_dir, T, S)
        out = {t: [] for t in TASKS}
        with torch.no_grad():
            for b in range(0, len(keys), bs):
                xs = []
                for k in keys[b:b + bs]:
                    a = fc.fetch(k)
                    if a is None:
                        raise SystemExit(f"{k} is not readable in {cache_dir}")
                    xs.append(torch.from_numpy(np.ascontiguousarray(a)))
                x = tp.norm_batch(torch.stack(xs).to(s.dev), s.dev)
                ac = tp._autocast(s.dev) if s.precision == "fp32" else torch.autocast("cpu", dtype=torch.float16)
                with ac:
                    o = s.model(x)
                for t in TASKS:
                    out[t].append(TG.head_probs(s.model, o[t]).cpu().numpy())
        return {t: np.concatenate(out[t]).astype(np.float32) for t in TASKS}


def controls_for(run, paths, cand, n=5):
    pool = sorted(p for p in paths if p not in cand)
    return sorted(random.Random(zlib.crc32(run.encode())).sample(pool, n))


TOL_P = 0.02          # equivalence tolerance on any posterior entry (fp16 GPU dump vs CPU re-prediction),
#                       fixed before the check was run; argmax must also agree on every checked row.
# On 2026-09-29 the fp32 check exceeded TOL_P on ONE row of one run (0.0205: x3dfix_dual_s3_fold1, g5, a re-cut
# clip's original-frame row) while argmax agreed on every row. The dump was written under CUDA fp16 autocast and
# the re-prediction is fp32 on CPU, so mid-range posteriors move by up to ~0.02. --accept_precision_noise lets
# patch proceed on the pre-set argmax criterion alone; the override and the measured noise are recorded in
# infer/equivalence.json and in every patch.json, and the new predictions whose top-2 margin is within the
# measured noise ("fragile") are listed.


def infer_run(run, precision, man, cand, a):
    t0 = time.time()
    z = np.load(os.path.join(VSUBJ, run, f"val_ep{EPOCH:02d}.npz"), allow_pickle=True)
    paths = [str(p) for p in z["path"]]
    pos = {p: i for i, p in enumerate(paths)}
    aff = [k for k in man if k in pos]
    ctl = controls_for(run, paths, cand, a.n_controls)
    P = Predictor(run, precision, a.threads)
    empty = {t: np.zeros((0, C.NCLS[t]), np.float32) for t in TASKS}
    ctl_pred = P.predict(F16, ctl)
    orig = P.predict(F16, aff) if aff else empty
    new = P.predict(CACHE, aff) if aff else empty
    rec = dict(run=np.array(run), precision=np.array(precision), keys_aff=np.array(aff, dtype=str),
               keys_ctl=np.array(ctl, dtype=str), check=np.array(json.dumps(P.check)))
    stats = dict(run=run, precision=precision, n_aff=len(aff), n_ctl=len(ctl), secs=None)
    for t in TASKS:
        st_c = z[f"probs_{t}"][[pos[k] for k in ctl]]
        st_a = z[f"probs_{t}"][[pos[k] for k in aff]] if aff else empty[t]
        rec.update({f"ctl_stored_{t}": st_c, f"ctl_pred_{t}": ctl_pred[t], f"aff_stored_{t}": st_a,
                    f"aff_orig_{t}": orig[t], f"aff_new_{t}": new[t]})
        for part, s_, p_ in (("controls", st_c, ctl_pred[t]), ("recut_original_frames", st_a, orig[t])):
            if len(s_) == 0:
                stats[f"{t}_{part}"] = dict(n=0)
                continue
            d = np.abs(s_.astype(np.float64) - p_.astype(np.float64))
            lg = np.abs(np.log(np.clip(s_, 1e-30, None)) - np.log(np.clip(p_, 1e-30, None)))
            big = s_ > 1e-3
            stats[f"{t}_{part}"] = dict(n=len(s_), max_abs_dp=float(d.max()), mean_abs_dp=float(d.mean()),
                                        max_abs_dlogp_where_p_gt_1e3=float(lg[big].max()),
                                        argmax_agree=int((s_.argmax(1) == p_.argmax(1)).sum()))
        stats[f"{t}_new_vs_stored_argmax_changed"] = int((st_a.argmax(1) != new[t].argmax(1)).sum()) if aff else 0
    stats["secs"] = round(time.time() - t0, 1)
    stats["check"] = P.check
    rec["stats"] = np.array(json.dumps(stats))
    C.atomic_npz(out_path("infer", f"{run}_{precision}.npz"), **rec)
    print(f"[infer] {run} ({precision}): {len(aff)} re-cut val clips, {len(ctl)} controls; re-predicted vs stored: "
          + "  ".join(f"{t}/{part[:4]} max|dp| {v['max_abs_dp']:.1e} argmax {v['argmax_agree']}/{v['n']}"
                      for t in TASKS for part in ("controls", "recut_original_frames")
                      for v in [stats[f'{t}_{part}']] if v["n"]) + f"  ({stats['secs']}s)", flush=True)
    return stats


def stage_infer(a):
    man = read_manifest()
    cand = read_candidates()
    if not os.path.exists(os.path.join(CACHE, "index.json")):
        raise SystemExit("run the cache stage first")
    sel = [r for r in runs() if not a.runs or r in a.runs]
    todo = [(r, "fp32") for r in sel] + [(r, "fp16") for r in (a.fp16_runs or [])]
    allstats = []
    for run, prec in todo:
        fn = os.path.join(INFER, f"{run}_{prec}.npz")
        if os.path.exists(fn) and not a.overwrite:
            allstats.append(json.loads(str(np.load(fn)["stats"])))
            print(f"[infer] {run} ({prec}): exists, reused", flush=True)
            continue
        allstats.append(infer_run(run, prec, man, cand, a))
    # the equivalence gate: every checked row (5 controls + the re-cut rows on their ORIGINAL frames) of every
    # fp32 run must match the stored dump within TOL_P and on argmax
    fails = []
    for st in allstats:
        if st["precision"] != "fp32":
            continue
        for t in TASKS:
            for part in ("controls", "recut_original_frames"):
                v = st[f"{t}_{part}"]
                if v["n"] and (v["max_abs_dp"] > TOL_P or v["argmax_agree"] != v["n"]):
                    fails.append((st["run"], t, part, v))
    argmax_fails = [f for f in fails if f[3]["argmax_agree"] != f[3]["n"]]
    summ = dict(tolerance_abs_dp=TOL_P, n_runs=len({s["run"] for s in allstats}), fails=fails,
                argmax_fails=argmax_fails, runs=allstats)
    # measured precision noise (fp32 CPU vs the fp16 GPU dump) and the new predictions it could flip
    eps = {}
    for t in TASKS:
        v = [s[f"{t}_{part}"]["max_abs_dlogp_where_p_gt_1e3"] for s in allstats if s["precision"] == "fp32"
             for part in ("controls", "recut_original_frames") if s[f"{t}_{part}"]["n"]]
        eps[t] = max(v) if v else None
    fragile = []
    for st in allstats:
        if st["precision"] != "fp32":
            continue
        z = np.load(os.path.join(INFER, f"{st['run']}_fp32.npz"))
        for t in TASKS:
            L = np.log(np.clip(z[f"aff_new_{t}"].astype(np.float64), 1e-30, None))
            if not len(L):
                continue
            srt = np.sort(L, 1)
            marg = srt[:, -1] - srt[:, -2]
            for j in np.where(marg < 2 * eps[t])[0]:
                fragile.append(dict(run=st["run"], head=t, key=str(z["keys_aff"][j]), margin_logp=float(marg[j]),
                                    new_argmax=int(L[j].argmax()), stored_argmax=int(z[f"aff_stored_{t}"][j].argmax())))
    summ["noise_max_abs_dlogp_where_p_gt_1e3"] = eps
    summ["fragile_rule"] = "a new prediction whose top-2 log-posterior margin is < 2 x the largest |d log p| seen on the checked rows"
    summ["fragile"] = fragile
    print(f"[infer] measured precision noise max|d log p| (p > 1e-3): {eps}; fragile new predictions: "
          f"{len(fragile)} (of which argmax changed vs stored: {sum(f['new_argmax'] != f['stored_argmax'] for f in fragile)})",
          flush=True)
    for prec in ("fp32", "fp16"):
        ss = [s for s in allstats if s["precision"] == prec]
        if not ss:
            continue
        agg = {}
        for t in TASKS:
            for part in ("controls", "recut_original_frames"):
                vs = [s[f"{t}_{part}"] for s in ss if s[f"{t}_{part}"]["n"]]
                agg[f"{t}_{part}"] = dict(n=sum(v["n"] for v in vs), max_abs_dp=max(v["max_abs_dp"] for v in vs),
                                          mean_abs_dp=float(np.average([v["mean_abs_dp"] for v in vs],
                                                                       weights=[v["n"] for v in vs])),
                                          argmax_agree=sum(v["argmax_agree"] for v in vs))
        summ[f"aggregate_{prec}"] = dict(runs=[s["run"] for s in ss], **agg)
        print(f"[infer] {prec} over {len(ss)} runs: " + "; ".join(
            f"{k} n={v['n']} max|dp| {v['max_abs_dp']:.2e} mean {v['mean_abs_dp']:.1e} argmax {v['argmax_agree']}/{v['n']}"
            for k, v in agg.items()), flush=True)
    if fails and a.accept_precision_noise and not argmax_fails:
        summ["override"] = dict(flag="--accept_precision_noise", at=time.strftime("%F %T"),
                                reason=f"{len(fails)} (run, head, part) exceed the pre-set |dp| <= {TOL_P} while argmax "
                                       f"agrees on every checked row; fp16-GPU vs fp32-CPU precision noise")
    C.atomic_json(out_path("infer", "equivalence.json"), summ)
    if fails and "override" not in summ:
        raise SystemExit(f"EQUIVALENCE FAILED on {len(fails)} (run, head, part): {fails[:3]}")
    if fails:
        print(f"[infer] pre-set |dp| <= {TOL_P} EXCEEDED on {len(fails)}: " + "; ".join(
            f"{f[0]} {f[1]} {f[2]} max|dp| {f[3]['max_abs_dp']:.4f} argmax {f[3]['argmax_agree']}/{f[3]['n']}"
            for f in fails) + "; argmax identical everywhere -> proceeding under --accept_precision_noise", flush=True)
    else:
        print(f"[infer] equivalence PASSED: every fp32 run within {TOL_P} and argmax-identical", flush=True)


# ============================================================================ stage: patch

def load_infer(run, precision):
    z = np.load(os.path.join(INFER, f"{run}_{precision}.npz"), allow_pickle=True)
    return {k: z[k] for k in z.files}


def stage_patch(a):
    import dhlib as D
    eq = json.load(open(os.path.join(INFER, "equivalence.json")))
    if (eq["fails"] and "override" not in eq) or eq.get("argmax_fails") or eq.get(f"aggregate_{a.precision}") is None:
        raise SystemExit(f"equivalence not established for {a.precision}: {eq['fails'][:3]}")
    man, exc = read_manifest(), read_exclude()
    tot = Counter()
    for run in runs():
        src = os.path.join(VSUBJ, run, f"val_ep{EPOCH:02d}.npz")
        z = np.load(src, allow_pickle=True)
        f = {k: z[k] for k in z.files}
        n0 = len(f["path"])
        rowf = [k for k in f if getattr(f[k], "ndim", 0) >= 1 and len(f[k]) == n0]
        if sorted(rowf) != ["path", "probs_g3", "probs_g5", "y5"]:
            raise SystemExit(f"{run}: unexpected per-row fields {rowf}")
        paths = [str(p) for p in f["path"]]
        pos = {p: i for i, p in enumerate(paths)}
        inf = load_infer(run, a.precision)
        aff = [str(k) for k in inf["keys_aff"]]
        if aff != [k for k in man if k in pos]:
            raise SystemExit(f"{run}: infer file does not cover exactly the run's re-cut val clips")
        drop = [k for k in exc if k in pos]
        if set(drop) & set(aff):
            raise SystemExit(f"{run}: a clip is both re-cut and excluded")
        keep = np.array([p not in set(drop) for p in paths])
        for variant, root in (("patched", PATCHED), ("exclonly", EXCLONLY)):
            g = dict(f)
            if variant == "patched":
                for t in TASKS:
                    P = f[f"probs_{t}"].copy()
                    for j, k in enumerate(aff):
                        P[pos[k]] = inf[f"aff_new_{t}"][j]
                    g[f"probs_{t}"] = P
            for k in rowf:
                g[k] = g[k][keep]
            for t in TASKS:                                   # the contract joint_gate / dhlib rely on
                P = g[f"probs_{t}"]
                assert P.dtype == np.float32 and np.all(np.isfinite(P)) and np.allclose(P.sum(1), 1, atol=1e-4)
                assert not D.is_double_softmax(P.astype(np.float64)), f"{run} {t}: looks double-softmaxed"
            assert g["path"].dtype == f["path"].dtype and len(set(g["path"].tolist())) == len(g["path"])
            assert np.array_equal(g["y5"], f["y5"][keep])
            d = out_path(os.path.basename(root), run)
            os.makedirs(os.path.dirname(d + "/x"), exist_ok=True)
            C.atomic_npz(os.path.join(d, f"val_ep{EPOCH:02d}.npz"), **g)
            shutil.copyfile(os.path.join(VSUBJ, run, "config.json"), os.path.join(d, "config.json"))
            nrep = len(aff) if variant == "patched" else 0
            newpos = np.cumsum(keep) - 1                     # row index after the removals (re-cut rows are kept)
            changed = {t: int((f[f"probs_{t}"][[pos[k] for k in aff]].argmax(1) !=
                               g[f"probs_{t}"][[int(newpos[pos[k]]) for k in aff]].argmax(1)).sum())
                       if (variant == "patched" and aff) else 0 for t in TASKS}
            C.atomic_json(os.path.join(d, "patch.json"), dict(
                variant=variant, source=os.path.relpath(src, EEG_ROOT), source_sha256=sha256(src),
                n_rows_before=n0, n_rows_after=int(keep.sum()), n_removed=len(drop), removed=drop,
                n_replaced=nrep, replaced=aff if variant == "patched" else [],
                predictions_from=os.path.relpath(os.path.join(INFER, f"{run}_{a.precision}.npz"), EEG_ROOT)
                if variant == "patched" else None, precision=a.precision if variant == "patched" else None,
                argmax_changed_on_replaced=changed, fields_kept=sorted(g), created=time.strftime("%F %T"),
                equivalence_override=eq.get("override") if variant == "patched" else None))
            tot[(variant, "removed")] += len(drop)
            tot[(variant, "replaced")] += nrep
            for t in TASKS:
                tot[(variant, f"changed_{t}")] += changed[t]
        print(f"[patch] {run}: rows {n0} -> {int(keep.sum())} (removed {len(drop)}), replaced {len(aff)}", flush=True)
    print("[patch] totals over 30 runs: " + ", ".join(f"{k[0]} {k[1]} {v}" for k, v in sorted(tot.items())),
          flush=True)
    C.atomic_json(out_path("vsubj_patched", "patch_summary.json"),
                  dict(created=time.strftime("%F %T"), precision=a.precision, n_runs=len(runs()),
                       totals={f"{k[0]}_{k[1]}": v for k, v in sorted(tot.items())}))


# ============================================================================ joint_gate in-process

def import_joint_gate():
    sys.path.insert(0, os.path.join(GRADER, "eeg"))
    import joint_gate as JG
    return JG


def adapt_joint_gate(JG, exclude_keys, out_prefix):
    """The two adaptations (see the module docstring). Returns a description for the provenance file."""
    JG.OUT_PREFIX = out_prefix
    excl = {JG.key_of(k) for k in exclude_keys}
    if excl:
        orig = JG._check_fold_animals

        def check(keys, f, log, what, folds, side=None, allow_partial=False):
            if side == "video":
                gone = folds[f]["video_keys"] & excl
                folds = {k: (dict(v, video_keys=v["video_keys"] - excl) if k == f else v) for k, v in folds.items()}
                if gone and not any(n.startswith(f"{what}: fold {f}:") for n in log["notes"]):
                    log["notes"].append(f"{what}: fold {f}: expected clip set = split fold minus {len(gone)} "
                                        f"excluded clips (ttg_recut/exclude.csv)")
            return orig(keys, f, log, what, folds, side, allow_partial)

        JG._check_fold_animals = check
    return dict(out_prefix=out_prefix, n_excluded_from_video_fold_check=len(excl), excluded=sorted(excl))


def stored_grid_argv(recipe):
    r = json.load(open(os.path.join(STORED_GRID, f"B_{recipe}_grid3x3", "results.json")))
    return r["argv"]


def grid_argv(recipe, variant):
    """The stored argv with --video pointing at the variant's dirs and --out under ttg_recut/grid/."""
    argv = list(stored_grid_argv(recipe))
    i, j = argv.index("--video"), argv.index("--video_epoch")
    vids = argv[i + 1:j]
    exp = [f"output/ttg_vsubj/{recipe}_dual_s{s}_fold{{f}}" for s in SEEDS]
    if vids != exp:
        raise SystemExit(f"stored argv --video {vids} != {exp}")
    argv[i + 1:j] = [os.path.join(VARIANTS[variant], os.path.basename(v)) for v in vids]
    k = argv.index("--out")
    argv[k + 1] = os.path.join(GRID, f"{GRID_PREFIX[variant]}B_{recipe}_grid3x3")
    return argv


def _grid_one(recipe, variant):
    """One joint_gate run (in a forked child: joint_gate is reloaded, adapted, and main() called)."""
    import importlib
    JG = importlib.reload(import_joint_gate())
    exc = list(read_exclude()) if variant != "orig" else []
    argv = grid_argv(recipe, variant)
    out = argv[argv.index("--out") + 1]
    out_path(os.path.relpath(out, RECUT), "wrapper.json")
    prov = adapt_joint_gate(JG, exc, GRID)
    C.atomic_json(os.path.join(out, "wrapper.json"), dict(
        stage="grid", recipe=recipe, variant=variant, joint_gate=os.path.join(GRADER, "eeg", "joint_gate.py"),
        joint_gate_sha256=sha256(os.path.join(GRADER, "eeg", "joint_gate.py")), argv=argv,
        stored_argv=stored_grid_argv(recipe), adaptations=prov, started=time.strftime("%F %T")))
    sys.argv = argv
    print(f"[grid] {recipe} {variant}: joint_gate {' '.join(argv[1:])}", flush=True)
    return JG.main()


def _grid_child(recipe, variant, log):
    fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o664)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    sys.stdout = os.fdopen(1, "w", buffering=1)
    sys.stderr = os.fdopen(2, "w", buffering=1)
    try:
        rc = _grid_one(recipe, variant)
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else 1
        print(f"SystemExit: {e}", flush=True)
    except BaseException:
        import traceback
        traceback.print_exc()
        rc = 1
    sys.stdout.flush()
    os._exit(int(rc or 0))


def stage_grid(a):
    """Every requested (recipe, variant) in its own forked child (the eeg env takes minutes to import, so the
    children inherit this process's modules), at most --jobs at a time; logs in ttg_recut/logs/."""
    import_joint_gate()                                  # import once in the parent (sklearn, dhlib, check_folds)
    todo = [(r, v) for r in a.recipes for v in a.variants]
    running, results = {}, {}
    while todo or running:
        while todo and len(running) < a.jobs:
            r, v = todo.pop(0)
            log = out_path("logs", f"grid_{r}_{v}.out")
            pid = os.fork()
            if pid == 0:
                _grid_child(r, v, log)
            running[pid] = (r, v, log, time.time())
            print(f"[grid] started {r} {v} (pid {pid}) -> {log}", flush=True)
        pid, status = os.wait()
        if pid in running:
            r, v, log, t0 = running.pop(pid)
            rc = os.waitstatus_to_exitcode(status)
            results[(r, v)] = rc
            print(f"[grid] {r} {v}: exit {rc} after {time.time() - t0:.0f}s", flush=True)
    bad = {k: v for k, v in results.items() if v != 0}
    if bad:
        raise SystemExit(f"grid runs failed: {bad}")


# ============================================================================ stage: compare (wrapper faithfulness)

VOLATILE = {"created", "argv", "secs"}


def _diff(x, y, path, out, tol):
    if isinstance(x, dict) and isinstance(y, dict):
        for k in sorted(set(x) | set(y)):
            if k in VOLATILE:
                continue
            if k not in x or k not in y:
                out.append((f"{path}/{k}", "missing on one side"))
                continue
            _diff(x[k], y[k], f"{path}/{k}", out, tol)
    elif isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            out.append((path, f"len {len(x)} != {len(y)}"))
        for i, (u, v) in enumerate(zip(x, y)):
            _diff(u, v, f"{path}[{i}]", out, tol)
    elif isinstance(x, float) and isinstance(y, float):
        if not (x == y or (np.isnan(x) and np.isnan(y)) or abs(x - y) <= tol):
            out.append((path, f"{x!r} != {y!r}"))
    elif x != y:
        out.append((path, f"{str(x)[:80]!r} != {str(y)[:80]!r}"))


def stage_compare(a):
    res = {}
    for recipe in RECIPES:
        mine = os.path.join(GRID, f"repro_B_{recipe}_grid3x3")
        if not os.path.exists(os.path.join(mine, "results.json")):
            print(f"[compare] {recipe}: no repro grid, skipped")
            continue
        st = os.path.join(STORED_GRID, f"B_{recipe}_grid3x3")
        out = []
        _diff(json.load(open(os.path.join(st, "results.json"))), json.load(open(os.path.join(mine, "results.json"))),
              "", out, 0.0)
        g = []
        _diff(json.load(open(os.path.join(st, "gate.json"))), json.load(open(os.path.join(mine, "gate.json"))),
              "", g, 0.0)
        tx = [l for l in open(os.path.join(st, "results.txt")).read().splitlines()[1:]]
        tm = [l for l in open(os.path.join(mine, "results.txt")).read().splitlines()[1:]]
        res[recipe] = dict(results_json_differences=out, gate_json_differences=g,
                           results_txt_identical_after_header=tx == tm)
        print(f"[compare] {recipe}: results.json {len(out)} differing fields (outside created/argv/secs); "
              f"gate.json {len(g)}; results.txt identical after the timestamp line: {tx == tm}", flush=True)
        for p, m in out[:10]:
            print(f"     {p}: {m}")
    C.atomic_json(out_path("rescore", "compare_repro.json"), res)


# ============================================================================ stage: analyse

def load_bundles(JG, recipe, variant, excl, folds):
    argv = grid_argv(recipe, variant)
    i, j = argv.index("--video"), argv.index("--video_epoch")
    e0, e1 = argv.index("--eeg"), argv.index("--eeg_epoch")
    vspec, espec = argv[i + 1:j], argv[e0 + 1:e1]
    eb = JG.load_eeg_dirs(espec, int(argv[e1 + 1]), "B", "logmean", False, folds)
    vbs = {t: JG.load_video_dirs(vspec, int(argv[j + 1]), t, "B", False, folds) for t in TASKS}
    for t, vb in vbs.items():
        if len(vb["members"]) != 3 or vb["log"]["dropped"]:
            raise SystemExit(f"{recipe} {variant} {t}: video load {JG.logline(vb['log'])}")
    if len(eb["members"]) != 3 or eb["log"]["dropped"]:
        raise SystemExit(f"{recipe} {variant}: EEG load {JG.logline(eb['log'])}")
    return vbs, eb


def score_pairs(JG, vb, eb, task):
    """Every single video x single EEG pair: video argmax and EGRG decisions (joint_gate's own code), and
    the numbers reported below. Returns (pairing, {(v, e): row})."""
    D = JG.D
    P = JG.Pairing(vb, eb, task, 1, 29)                  # reps only size the unused bootstrap weights
    out = {}
    for vn, M in vb["members"].items():
        Pv = M[P.iv]
        wss = P.within_session_severe(Pv)
        for en, E in eb["members"].items():
            preds, _, Q, _ = P.systems(Pv, E["stored"][P.ie], E["top25"][P.ie], top25=False)
            r = {}
            for sname in ("video", "EGRG"):
                p = preds[sname]
                Cm = D.confusion(P.y, p, P.K)
                m = D.cm_metrics(Cm)
                r[sname] = dict(macro_f1=float(m["macro_f1"]), mcc=float(m["mcc"]),
                                hits=[int(Cm[i, i]) for i in range(P.K)], n=[int(Cm[i].sum()) for i in range(P.K)],
                                sev_hit=int(Cm[np.ix_(P.sev, P.sev)].sum()), pred=p)
            r["within_session_severe_auroc"] = wss
            out[(vn, en)] = r
    return P, out


def drop_keys(bundle, keys, kind):
    """A copy of a joint_gate video / EEG bundle without these clip keys (sensitivity analysis only)."""
    m = ~np.isin(bundle["keys"], list(keys))
    b = dict(bundle, keys=bundle["keys"][m], y5=bundle["y5"][m])
    if kind == "video":
        b["members"] = {n: M[m] for n, M in bundle["members"].items()}
    else:
        b["members"] = {n: dict(stored=M["stored"][m], top25=M["top25"][m]) for n, M in bundle["members"].items()}
    return b


def stage_analyse(a):
    JG = import_joint_gate()
    import check_folds as CF
    man, exc = read_manifest(), read_exclude()
    folds = CF.expected_folds()
    summary = dict(created=time.strftime("%F %T"), precision=a.precision, recipes={})
    recut = set(JG.key_of(k) for k in man)
    lines = []
    for recipe in RECIPES:
        R = {}
        per = {}
        for variant in ("orig", "exclonly", "patched"):
            JG2 = import_joint_gate()
            import importlib
            JG2 = importlib.reload(JG2)
            adapt_joint_gate(JG2, list(exc) if variant != "orig" else [], GRID)
            vbs, eb = load_bundles(JG2, recipe, variant, list(exc), folds)
            per[variant] = {"_eb": eb, "_vbs": vbs, "_JG": JG2}
            for t in TASKS:
                P, rows = score_pairs(JG2, vbs[t], eb, t)
                per[variant][t] = (P, rows, vbs[t])
            print(f"[analyse] {recipe} {variant}: scored "
                  + ", ".join(f"{t} {per[variant][t][0].log['n_scored']} clips" for t in TASKS), flush=True)
        # the grid runs must agree with this re-scoring (per-pair macro-F1 and severe hits)
        agree = {}
        for variant in ("orig", "exclonly", "patched"):
            gd = os.path.join(GRID, f"{GRID_PREFIX[variant]}B_{recipe}_grid3x3", "results.json")
            if variant == "orig":
                gd = os.path.join(STORED_GRID, f"B_{recipe}_grid3x3", "results.json")
            if not os.path.exists(gd):
                agree[variant] = "grid missing"
                continue
            g = json.load(open(gd))
            ok = True
            for blk in g["blocks"]:
                rows = per[variant][blk["task"]][1]
                for pr in blk["single_pairs"]["pairs"]:
                    r = rows[(pr["video_run"], pr["eeg_run"])]
                    for sname in ("video", "EGRG"):
                        ok &= abs(pr["systems"][sname]["f1"] - r[sname]["macro_f1"]) < 1e-12
                        ok &= pr["systems"][sname]["sev_hit"] == r[sname]["sev_hit"]
            agree[variant] = "per-pair macro-F1 and severe hits identical" if ok else "MISMATCH"
            if not ok:
                raise SystemExit(f"{recipe} {variant}: re-scoring disagrees with {gd}")
        R["grid_agreement"] = agree
        R["grid_rows"] = {}                              # every row of each grid: point + CI, from results.json
        for variant in ("orig", "exclonly", "patched"):
            gd = os.path.join(GRID, f"{GRID_PREFIX[variant]}B_{recipe}_grid3x3", "results.json")
            if variant == "orig":
                gd = os.path.join(STORED_GRID, f"B_{recipe}_grid3x3", "results.json")
            if not os.path.exists(gd):
                continue
            g = json.load(open(gd))
            R["grid_rows"][variant] = {blk["task"]: dict(
                source=os.path.relpath(gd, EEG_ROOT), pairing=blk["pairing"], video_load=blk["video_load"],
                rows=[dict(video_name=r["video"], eeg_name=r["eeg"], n=r["n_clips"], ws_auroc=r["within_session_severe_auroc"],
                           **{sn: dict(macro_f1=r["systems"][sn]["macro_f1"], ci=r["systems"][sn]["ci"]["macro_f1"],
                                       d_vs_video=r["systems"][sn].get("d_vs_video", {}).get("macro_f1"),
                                       per_class={k: [v["hit"], v["n"]] for k, v in r["systems"][sn]["per_class"].items()},
                                       severe=[r["systems"][sn]["severe"]["hit"], r["systems"][sn]["severe"]["n"]])
                              for sn in ("video", "EGRG")}) for r in blk["rows"]],
                single_pairs={k: v for k, v in blk["single_pairs"].items() if k != "pairs"})
                for blk in g["blocks"]}
        # per re-cut clip: the EEG detector at the label (3-seed mean pooled P(sz), OOF) and the video calls
        ebo = per["orig"]["_eb"]
        e_mean = np.mean([M["stored"] for M in ebo["members"].values()], 0)
        epos = {k: i for i, k in enumerate(ebo["keys"].tolist())}
        table = []
        for k0, m in man.items():
            k = JG.key_of(k0)
            row = dict(key=k, y5=m["y"], cause=m["cause"], sub=m["sub"], offset_s=m["offset_s"],
                       offset_source=m["source"], eeg_p_sz=float(e_mean[epos[k]]) if k in epos else None)
            for variant, tag in (("orig", "old"), ("patched", "new")):
                vb5 = per[variant]["_vbs"]["g5"]
                vpos = {kk: i for i, kk in enumerate(vb5["keys"].tolist())}
                am = [int(M[vpos[k]].argmax()) for M in vb5["members"].values()]
                row[f"video_g5_argmax_{tag}"] = am
                row[f"video_calls_seizure_{tag}"] = int(sum(x > 0 for x in am))
            table.append(row)
        # EEG disagrees with the label at the label time: a seizure label whose EEG P(sz) < 0.5, or a non-seizure
        # label whose EEG P(sz) > 0.5. For those clips the premise "label and EEG are right, only the video is
        # wrong" fails, so the re-cut video need not be the right one. Rule uses the EEG only, never the video.
        disputed = sorted(r["key"] for r in table if r["eeg_p_sz"] is not None and
                          ((r["y5"] > 0 and r["eeg_p_sz"] < 0.5) or (r["y5"] == 0 and r["eeg_p_sz"] > 0.5)))
        ys_all = ebo["y5"] > 0
        R["recut_clip_table"] = table
        R["eeg_disputed"] = dict(
            rule="re-cut clip whose 3-seed mean EEG P(sz) at the label disagrees with the label at 0.5",
            keys=disputed, n=len(disputed),
            base_rate_all_seizure_clips_eeg_lt_0p5=[int((e_mean[ys_all] < 0.5).sum()), int(ys_all.sum())],
            base_rate_all_nonseizure_clips_eeg_gt_0p5=[int((e_mean[~ys_all] > 0.5).sum()), int((~ys_all).sum())],
            recut_seizure_eeg_lt_0p5=[sum(1 for r in table if r["y5"] > 0 and r["eeg_p_sz"] < 0.5),
                                      sum(1 for r in table if r["y5"] > 0)],
            recut_seizure_video_majority_calls_seizure=dict(
                old=sum(1 for r in table if r["y5"] > 0 and r["video_calls_seizure_old"] >= 2),
                new=sum(1 for r in table if r["y5"] > 0 and r["video_calls_seizure_new"] >= 2),
                n=sum(1 for r in table if r["y5"] > 0)),
            recut_nonseizure_video_majority_calls_seizure=dict(
                old=sum(1 for r in table if r["y5"] == 0 and r["video_calls_seizure_old"] >= 2),
                new=sum(1 for r in table if r["y5"] == 0 and r["video_calls_seizure_new"] >= 2),
                n=sum(1 for r in table if r["y5"] == 0)),
            disputed_detail=[r for r in table if r["key"] in disputed])
        for t in TASKS:
            names = JG.D.NAMES[t]
            blk = {}
            variants = [("orig", per["orig"][t]), ("exclonly", per["exclonly"][t]), ("patched", per["patched"][t])]
            if disputed:                                    # SENSITIVITY: the disputed clips removed, before and after
                for variant in ("orig", "patched"):
                    JGv, ebv, vbv = per[variant]["_JG"], per[variant]["_eb"], per[variant]["_vbs"][t]
                    Pd, rowsd = score_pairs(JGv, drop_keys(vbv, disputed, "video"), drop_keys(ebv, disputed, "eeg"), t)
                    variants.append((f"{variant}_minus_eeg_disputed", (Pd, rowsd, vbv)))
            for variant, (P, rows, vb) in variants:
                pairs = sorted(rows)
                f = lambda s, k: np.array([rows[p][s][k] for p in pairs], float)
                blk[variant] = dict(
                    n_clips=int(P.log["n_scored"]), class_counts=np.bincount(P.y, minlength=P.K).tolist(),
                    pairing=P.log,
                    video_f1_mean=float(f("video", "macro_f1").mean()), egrg_f1_mean=float(f("EGRG", "macro_f1").mean()),
                    gain_mean=float((f("EGRG", "macro_f1") - f("video", "macro_f1")).mean()),
                    gain_sd=float((f("EGRG", "macro_f1") - f("video", "macro_f1")).std(ddof=1)),
                    video_f1_by_seed={vn: float(np.mean([rows[p]["video"]["macro_f1"] for p in pairs if p[0] == vn]))
                                      for vn in vb["members"]},
                    egrg_f1_by_seed={vn: float(np.mean([rows[p]["EGRG"]["macro_f1"] for p in pairs if p[0] == vn]))
                                     for vn in vb["members"]},
                    video_hits_mean=np.mean([rows[p]["video"]["hits"] for p in pairs], 0).round(2).tolist(),
                    egrg_hits_mean=np.mean([rows[p]["EGRG"]["hits"] for p in pairs], 0).round(2).tolist(),
                    video_sev_hit_mean=float(f("video", "sev_hit").mean()),
                    egrg_sev_hit_mean=float(f("EGRG", "sev_hit").mean()),
                    sev_hit_range=[int(f("EGRG", "sev_hit").min()), int(f("EGRG", "sev_hit").max())],
                    within_session_severe_auroc={vn: rows[(vn, sorted(eb_n for (v_, eb_n) in pairs if v_ == vn)[0])]
                                                 ["within_session_severe_auroc"] for vn in vb["members"]},
                    pairs={f"{p[0]} x {p[1]}": dict(video_f1=rows[p]["video"]["macro_f1"], egrg_f1=rows[p]["EGRG"]["macro_f1"],
                                                    video_hits=rows[p]["video"]["hits"], egrg_hits=rows[p]["EGRG"]["hits"],
                                                    video_sev=rows[p]["video"]["sev_hit"], egrg_sev=rows[p]["EGRG"]["sev_hit"])
                           for p in pairs})
                blk[variant]["within_session_severe_auroc_mean"] = float(
                    np.mean(list(blk[variant]["within_session_severe_auroc"].values())))
            # predictions that changed class, on the clips both sets score
            P0, rows0, _ = per["orig"][t]
            P1, rows1, _ = per["patched"][t]
            Px, rowsx, _ = per["exclonly"][t]
            common = sorted(set(P0.keys.tolist()) & set(P1.keys.tolist()))
            i0 = {k: i for i, k in enumerate(P0.keys.tolist())}
            i1 = {k: i for i, k in enumerate(P1.keys.tolist())}
            ix = {k: i for i, k in enumerate(Px.keys.tolist())}
            a0 = np.array([i0[k] for k in common]); a1 = np.array([i1[k] for k in common])
            ax = np.array([ix[k] for k in common])
            isrec = np.array([k in recut for k in common])
            ch = dict(n_common=len(common), n_recut_scored=int(isrec.sum()), per_pair={})
            for p in sorted(rows0):
                d = {}
                for sname in ("video", "EGRG"):
                    c = rows0[p][sname]["pred"][a0] != rows1[p][sname]["pred"][a1]
                    cx = rows0[p][sname]["pred"][a0] != rowsx[p][sname]["pred"][ax]   # exclusion alone
                    d[sname] = dict(recut=int((c & isrec).sum()), other=int((c & ~isrec).sum()),
                                    other_by_exclusion_alone=int((cx & ~isrec).sum()))
                ch["per_pair"][f"{p[0]} x {p[1]}"] = d
            for sname in ("video", "EGRG"):
                for part in ("recut", "other", "other_by_exclusion_alone"):
                    v = [d[sname][part] for d in ch["per_pair"].values()]
                    ch[f"{sname}_{part}_mean"] = float(np.mean(v))
                    ch[f"{sname}_{part}_range"] = [int(min(v)), int(max(v))]
            # the re-cut clips themselves: video (per seed) and EGRG (per pair) correct before -> after
            y1 = P1.y[a1]
            rc = dict(n=int(isrec.sum()), by_class=np.bincount(y1[isrec], minlength=P1.K).tolist(), seeds={})
            for vn in per["orig"][t][2]["members"]:
                en0 = sorted(e for (v_, e) in rows0 if v_ == vn)[0]
                b = rows0[(vn, en0)]["video"]["pred"][a0][isrec]
                aft = rows1[(vn, en0)]["video"]["pred"][a1][isrec]
                yy = y1[isrec]
                rc["seeds"][vn] = dict(
                    correct_before=np.bincount(yy[b == yy], minlength=P1.K).tolist(),
                    correct_after=np.bincount(yy[aft == yy], minlength=P1.K).tolist(),
                    called_seizure_before=np.bincount(yy[b > 0], minlength=P1.K).tolist(),
                    called_seizure_after=np.bincount(yy[aft > 0], minlength=P1.K).tolist(),
                    changed=int((b != aft).sum()))
            eg = []
            K = P1.K
            conf = {sn: {w: np.zeros((K, K), int) for w in ("before", "after")} for sn in ("video", "EGRG")}
            for p in sorted(rows0):
                b = rows0[p]["EGRG"]["pred"][a0][isrec]; aft = rows1[p]["EGRG"]["pred"][a1][isrec]
                eg.append(dict(pair=f"{p[0]} x {p[1]}", correct_before=int((b == y1[isrec]).sum()),
                               correct_after=int((aft == y1[isrec]).sum())))
                for sn in ("video", "EGRG"):
                    for w, rr, aa in (("before", rows0, a0), ("after", rows1, a1)):
                        conf[sn][w] += np.bincount(y1[isrec] * K + rr[p][sn]["pred"][aa][isrec],
                                                   minlength=K * K).reshape(K, K)
            rc["egrg_pairs"] = eg
            # true class x predicted class on the re-cut clips, SUMMED over the 9 pairs (divide by 9 for a pair)
            rc["confusion_sum_over_pairs"] = {sn: {w: c.tolist() for w, c in d.items()} for sn, d in conf.items()}
            blk["changed"] = ch
            blk["recut_clips"] = rc
            blk["class_names"] = names
            R[t] = blk
        summary["recipes"][recipe] = R
    summary["training_exposure"] = exposure(JG, man, exc, folds)
    C.atomic_json(out_path("rescore", "summary.json"), summary)
    txt = render(summary)
    with open(out_path("rescore", "summary.txt"), "w") as fh:
        fh.write(txt)
    print(txt)


def exposure(JG, man, exc, folds):
    """How many affected clips sit in each fold's TRAINING set (the same for every seed and recipe: all runs
    share split_subjects seed 49), by class, against the training class counts."""
    items = sorted(C.items_from_f32_index(), key=lambda it: it[0])
    import train_pooled as tp
    y5 = {JG.key_of(i[0]): i[1] for i in items}
    groups = OrderedDict([
        ("recut (wrong-time video, corrected here)", [JG.key_of(k) for k in man]),
        ("no_video (wrong-time video, no right video exists)",
         [JG.key_of(k) for k, v in exc.items() if v["status"] == "no_video"]),
        ("unresolved (4 in the orchestrator's 90 + 1 other)",
         [JG.key_of(k) for k, v in exc.items() if v["status"] == "unresolved"]),
        ("mislabelled negative (class 0 holding a Stage 3 seizure)",
         [JG.key_of(k) for k, v in exc.items() if v["status"] == "exclude_from_class0"]),
    ])
    out = dict(note="all 30 runs share split_subjects(seed 49) folds; a clip is in the training set of the 4 "
                    "folds that do not hold its animal", folds={})
    for f in FOLDS:
        tr, va, vg, _ = tp.split_subjects(items, 49, f, 5)
        trk = {JG.key_of(i[0]) for i in tr}
        ytr = np.bincount([i[1] for i in tr], minlength=5)
        g = {}
        for name, ks in groups.items():
            inn = [k for k in ks if k in trk]
            g[name] = dict(n=len(inn), by_class5=np.bincount([y5[k] for k in inn], minlength=5).tolist())
        wrong = [k for name in list(groups)[:2] for k in groups[name]]
        w_in = np.bincount([y5[k] for k in wrong if k in trk], minlength=5)
        out["folds"][f] = dict(val_animals=sorted(vg), n_train=len(tr), train_class5=ytr.tolist(), groups=g,
                               wrong_time_in_train=int(w_in.sum()), wrong_time_in_train_class5=w_in.tolist(),
                               wrong_time_share_of_train_class5=[round(float(w_in[c] / max(ytr[c], 1)), 4)
                                                                 for c in range(5)])
    return out


def render(s):
    L = [f"ttg_recut rescore -- {s['created']}  (new video predictions: CPU {s['precision']})", ""]
    for recipe, R in s["recipes"].items():
        L.append("=" * 110)
        L.append(f"{recipe}: 3 video seeds x 3 EEG seeds (9 single pairs), protocol B, video epoch 12, EEG epoch 30")
        L.append(f"  grid agreement: {R['grid_agreement']}")
        d = R.get("eeg_disputed")
        if d:
            L.append(f"  re-cut seizure clips the video calls seizure (majority of 3 seeds): old "
                     f"{d['recut_seizure_video_majority_calls_seizure']['old']} -> new "
                     f"{d['recut_seizure_video_majority_calls_seizure']['new']} of "
                     f"{d['recut_seizure_video_majority_calls_seizure']['n']}; non-seizure clips called seizure: "
                     f"{d['recut_nonseizure_video_majority_calls_seizure']}")
            L.append(f"  EEG-disputed re-cut clips ({d['rule']}): {d['n']}; base rate over all seizure clips "
                     f"{d['base_rate_all_seizure_clips_eeg_lt_0p5']}, non-seizure {d['base_rate_all_nonseizure_clips_eeg_gt_0p5']}")
            for r in d["disputed_detail"]:
                L.append(f"     y5={r['y5']} EEG {r['eeg_p_sz']:.3f} {r['sub']:<20} off {r['offset_s']:+.1f} ({r['offset_source']}) "
                         f"video g5 old {r['video_g5_argmax_old']} new {r['video_g5_argmax_new']}  {r['key']}")
            L.append("  (-D rows: the EEG-disputed clips removed from before AND after; a SENSITIVITY analysis, the clips "
                     "were chosen by the EEG score on the scored clips)")
        for v, gr in R.get("grid_rows", {}).items():
            for t, gb in gr.items():
                for r in gb["rows"]:
                    if r["video_name"].startswith("video_ens") and r["eeg_name"].startswith("eeg_ens"):
                        e = r["EGRG"]
                        L.append(f"  grid {v:<8} {t} headline {r['video_name']} x {r['eeg_name']} n={r['n']}: video "
                                 f"{r['video']['macro_f1']:.4f} EGRG {e['macro_f1']:.4f} [{e['ci'][0]:.4f},{e['ci'][1]:.4f}]"
                                 f"  EGRG-video {e['d_vs_video']['mean']:+.4f} [{e['d_vs_video']['lo']:+.4f},"
                                 f"{e['d_vs_video']['hi']:+.4f}]  severe {r['video']['severe']} -> {e['severe']}")
        for t in TASKS:
            blk = R[t]
            nm = blk["class_names"]
            L.append(f"  -- {t}  (before = stored dumps; excl = the 11 excluded clips removed; after = excl + the "
                     f"81 re-cut clips re-predicted)")
            for v, lab in (("orig", "before"), ("exclonly", "excl"), ("patched", "after"),
                           ("orig_minus_eeg_disputed", "before-D"), ("patched_minus_eeg_disputed", "after-D")):
                if v not in blk:
                    continue
                b = blk[v]
                L.append(f"   {lab:<6} n={b['n_clips']} counts={b['class_counts']}  video {b['video_f1_mean']:.4f}  "
                         f"EGRG {b['egrg_f1_mean']:.4f}  gain {b['gain_mean']:+.4f} (sd {b['gain_sd']:.4f})  "
                         f"sev hits video {b['video_sev_hit_mean']:.1f} EGRG {b['egrg_sev_hit_mean']:.1f} "
                         f"{b['sev_hit_range']}  ws-AUROC {b['within_session_severe_auroc_mean']:.4f}")
                L.append(f"          video hits " + " ".join(f"{n} {h:.1f}/{c}" for n, h, c in
                                                            zip(nm, b['video_hits_mean'], b['class_counts'])))
                L.append(f"          EGRG  hits " + " ".join(f"{n} {h:.1f}/{c}" for n, h, c in
                                                            zip(nm, b['egrg_hits_mean'], b['class_counts'])))
            ch = blk["changed"]
            L.append(f"   predictions changed, before -> after, on {ch['n_common']} common clips ({ch['n_recut_scored']} "
                     f"re-cut), mean per pair [range]: video re-cut {ch['video_recut_mean']:.1f} "
                     f"{ch['video_recut_range']}, other {ch['video_other_mean']:.1f}; EGRG re-cut "
                     f"{ch['EGRG_recut_mean']:.1f} {ch['EGRG_recut_range']}, other {ch['EGRG_other_mean']:.1f} "
                     f"{ch['EGRG_other_range']} (exclusion alone: {ch['EGRG_other_by_exclusion_alone_mean']:.1f})")
            rc = blk["recut_clips"]
            L.append(f"   re-cut clips scored: {rc['n']} by class {rc['by_class']}")
            for vn, d in rc["seeds"].items():
                L.append(f"     {vn}: video correct {d['correct_before']} -> {d['correct_after']}; called seizure "
                         f"{d['called_seizure_before']} -> {d['called_seizure_after']}; changed {d['changed']}")
            eb = np.array([[e["correct_before"], e["correct_after"]] for e in rc["egrg_pairs"]])
            L.append(f"     EGRG correct on re-cut clips, mean over 9 pairs: {eb[:, 0].mean():.1f} -> {eb[:, 1].mean():.1f}")
            for sn in ("video", "EGRG"):
                for w in ("before", "after"):
                    cm = np.array(rc["confusion_sum_over_pairs"][sn][w]) / 9.0
                    L.append(f"     {sn:<5} {w:<6} re-cut confusion (rows true {nm}, mean per pair): "
                             + " | ".join(" ".join(f"{v:5.1f}" for v in r) for r in cm))
    L.append("=" * 110)
    L.append("training exposure (affected clips in each fold's TRAINING set; identical for every seed / recipe)")
    for f, d in s["training_exposure"]["folds"].items():
        L.append(f"  fold {f} (val {','.join(d['val_animals'])}): n_train {d['n_train']}  wrong-time in train "
                 f"{d['wrong_time_in_train']} by class {d['wrong_time_in_train_class5']} = share "
                 f"{d['wrong_time_share_of_train_class5']}")
        for name, g in d["groups"].items():
            L.append(f"      {name}: {g['n']} {g['by_class5']}")
    return "\n".join(L) + "\n"


# ============================================================================ main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("stages", nargs="+",
                    choices=["cache", "infer", "patch", "grid", "compare", "analyse"])
    ap.add_argument("--precision", choices=["fp32", "fp16"], default="fp32",
                    help="patch / analyse: which predictions to use (infer always runs fp32)")
    ap.add_argument("--fp16_runs", nargs="*", default=["x3dfix_dual_s1_fold0"],
                    help="infer: also predict these runs with CPU fp16 autocast (precision comparison)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--n_controls", type=int, default=5)
    ap.add_argument("--runs", nargs="*", default=None, help="infer: only these runs")
    ap.add_argument("--recipes", nargs="+", choices=RECIPES, default=list(RECIPES))
    ap.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=["orig", "exclonly", "patched"])
    ap.add_argument("--jobs", type=int, default=6, help="grid: joint_gate runs in parallel (forked)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--accept_precision_noise", action="store_true",
                    help="infer/patch: proceed when the pre-set |dp| tolerance is exceeded but argmax agrees on every "
                         "checked row (recorded in equivalence.json and patch.json)")
    ap.add_argument("--wait_for", nargs="+", default=None,
                    help="import the heavy modules first, then wait until this file exists before running the "
                         "stages (lets a second process pay the slow eeg-env import while the first one runs)")
    ap.add_argument("--wait_hours", type=float, default=8.0)
    a = ap.parse_args()
    C.enter_eeg_root()
    if a.wait_for:
        t0 = time.time()
        import_joint_gate()
        import check_folds  # noqa: F401
        import train_pooled  # noqa: F401
        print(f"[wait] modules imported in {time.time() - t0:.0f}s; waiting for {a.wait_for}", flush=True)
        while not all(os.path.exists(w) for w in a.wait_for):
            if time.time() - t0 > 3600 * a.wait_hours:
                raise SystemExit(f"gave up waiting for {a.wait_for}")
            time.sleep(15)
        print(f"[wait] {a.wait_for} exists after {time.time() - t0:.0f}s", flush=True)
    for st in a.stages:
        t0 = time.time()
        {"cache": stage_cache, "infer": stage_infer, "patch": stage_patch, "grid": stage_grid,
         "compare": stage_compare, "analyse": stage_analyse}[st](a)
        print(f"[{st}] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
