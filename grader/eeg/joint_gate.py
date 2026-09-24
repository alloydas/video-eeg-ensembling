#!/usr/bin/env python3
r"""
EGRG joint gate: fit the EEG-Gated Racine Grader and score it against video alone and the
post-hoc gate, on the aligned split (protocol A) and on subject-disjoint OOF (protocol B).

MODEL (video grades, video AND EEG detect)
  P_V(sz)    = 1 - P_V(non-seizure) from the video head of the task (g3 or g5)
  P_V(g|sz)  = video's renormalised within-seizure split (g3: mild / severe; g5: S2..S5)
  P_E(sz)    = the EEG run's pooled clip P(seizure): PRIMARY = the trainer's usual pooling
               (logmean for train_eeg_det.py and v3_eegalign; the stored v3_subject_cv GRU folds
               used mean); SECONDARY = mean of the top 25% window P(seizure) ("top25")
  systems    video             argmax of the video posterior (reference)
             post-hoc gate     q = mean(P_V(sz), P_E(sz))                     (the verified rule)
             EGRG              q = sigmoid(a logit P_V(sz) + b logit P_E(sz) + c): sklearn
                               LogisticRegression(C=1) on seizure-vs-not, fitted LEAVE-ONE-
                               ANIMAL-OUT for scoring (refit for every video/EEG pair: the
                               coefficients depend on the video model)
             EGRG-top25        the same gate on the top25 EEG score (secondary; reviewers found
                               it costs S5 recall at g5 OOF, and frac 0.25 was chosen after looking)
             EGRG-nested       the EEG input (pooled vs top25) chosen by NESTED leave-one-animal-
                               out: for each held-out animal an inner LOAO on the other animals
                               compares the two gates by task macro-F1 (tie -> pooled), and the
                               winner, fitted on the other animals, scores the held-out one
                               (design review required change #3; secondary; --nested)
             EGRG (frozen gate) protocol A only: a protocol-B gate.json entry applied unchanged
             oracle detection  q = the true seizure label: the ceiling of any gate, NOT a system
  decision   argmax over [1 - q, q * P_V(g|sz)] for every gate. Video's grading is untouched:
             every clip a gate calls seizure gets video's within-seizure argmax (checked per row:
             grading_untouched), so the within-session severe-vs-mild AUROC is video's, computed
             once per row from the video posterior (within_session_severe_auroc).
  CIs        animal-clustered bootstrap with the LOAO gate outputs q held fixed: gate-fitting
             variability is NOT propagated into the EGRG CIs.
  The deployment gates (fitted on ALL protocol-B OOF clips) are written to gate.json (format 2):
  per entry the coefficients, eeg_input (pooled | top25: the score b multiplies) and
  eeg_signature (detector arch, head, pooling). --frozen_gate refuses an entry whose signature
  differs from the protocol-A EEG input unless --frozen_allow_mismatch (then labelled MISMATCHED).

PROTOCOLS
  A  aligned 5,279 clips (EEG --session_universe video, a strict subset of the video session
     split seed 49 val set of 5,289; the 10 video clips with no EEG are dropped and counted).
     Gates are LOAO-refit on the scored split -- labelled as such. --frozen_gate applies a
     gate.json from protocol B instead (nothing fitted on the scored split): that is the honest
     protocol-A comparison once protocol-B runs of the same video / EEG models exist. Select the
     entry with --frozen_entry (video name) and --frozen_eeg_entry (EEG name, e.g. eeg_ens2).
  B  subject-disjoint 5-fold OOF (split_subjects seed 49; EEG and video folds are verified to
     hold identical animals by check_folds.py before anything is scored; 45 video clips have
     no EEG and are dropped and counted).

INPUTS  (--video / --eeg take 'stored' or one or more run directories; relative paths are
        resolved against EEG_ROOT; for protocol B a directory is a TEMPLATE containing {f})
  video stored, A   the 40 output/v3_vidseeds/*_<task>_s* runs through dhlib.load_members
                    (path alignment, double-softmax repair, path-less runs admitted only by
                    label sequence): ens40, the 10 per-backbone 4-seed ensembles and all 40
                    single networks. Best-epoch checkpoints: UPPER BOUND.
  video stored, B   output/subject_cv/vid_<task>_fold{0-4} (R(2+1)D; path-less, recovered from
                    idx + dhlib.split_subjects and admitted only if the labels match). UPPER BOUND.
  video dirs        grader/train_grader.py runs: val_ep{E:02d}.npz (path, y5, probs_g3/probs_g5) at
                    --video_epoch (default 12 = its last epoch); one directory = one network,
                    several = an ensemble plus singles.
  eeg stored, A     output/v3_eegalign/*_g3_s* (GRU+TCN g3 heads, 10 runs; P_E(sz) = 1 - P(non-sz)
                    of the stored logmean clip vector; top25 from val_window_preds). UPPER BOUND.
  eeg stored, B     output/v3_subject_cv/eeg_bin_fold{0-4} (GRU binary, mean pooling). UPPER BOUND.
  eeg dirs          grader/eeg/train_eeg_det.py runs: val_clip_ep{E:02d}.npz at --eeg_epoch
                    (default 30 = last): p_logmean (or --eeg_score mean) and p_top25.
  Every load logs: runs requested / loaded, dropped, double-softmax repaired, path-less
  admitted, reordered; every pairing logs the clips dropped on either side.

REPORTED (per task, per video x EEG pairing)
  macro-F1, MCC, balanced accuracy, macro precision / recall, per-class recall WITH counts,
  severe recall (g3: severe; g5: S4+S5 predicted as S4 or S5) with counts, detection macro-F1
  with false alarms / misses / misses by stage, OvR macro AUROC of the composed vector,
  within-session severe-vs-mild AUROC of video's grading (one per row, shared by every gate;
  grading_untouched is checked per row and printed for every block), and animal-clustered
  bootstrap CIs (2,000 reps, gate outputs held fixed) for every system plus paired deltas vs
  video and vs the post-hoc gate. Rows: ensemble x
  ensemble (the headline for stored A is ens40 -- a 40-network confound, a lower bound on the
  gate's value), per-backbone ensembles, single video networks, single EEG networks, and the
  distribution over every single-video x single-EEG pair (point estimates).

REGRESSION (--regression; mandatory before trusting new numbers)
  Re-derives the validated prototype (joint-gated j10_argmax_rule.py): video ens40 + v3_eegalign
  aligned, and R(2+1)D + GRU-bin OOF, with the prototype's gate features (top25), and asserts
    aligned g3 video 0.8105 / post-hoc 0.8220 / EGRG 0.8231, g5 0.6969 / 0.7137 / 0.7163
    OOF     g3 0.7327 / 0.7541 / 0.7669,           g5 0.4780 / 0.4964 / 0.5178
  plus every per-class count, the paired bootstrap CIs and the load counts. Exit 1 on mismatch.

USAGE (from the video-eeg-ensembling repo, cwd anywhere; EEG_ROOT env, default
       /work/mech-ai-scratch/alloy/EEG, holds every run read and written; DHLIB_DIR env, default
       grader/ = where the vendored dhlib.py lives; numpy/scipy + sklearn. Relative --video /
       --eeg run directories are resolved against EEG_ROOT; --out and --frozen_gate against the cwd.)
  python grader/eeg/joint_gate.py --regression --out /work/mech-ai-scratch/alloy/EEG/output/ttg_eeg_gate/regression
  python grader/eeg/joint_gate.py --protocol A --video stored --eeg stored --out .../ttg_eeg_gate/A_stored
  python grader/eeg/joint_gate.py --protocol B --video stored --eeg stored --out .../ttg_eeg_gate/B_stored
  python grader/eeg/joint_gate.py --protocol A --video output/ttg_stage1/x3dbug_dual_s1 --video_epoch 12 \
      --eeg output/ttg_eeg/aligned/tcn_bin_s1 output/ttg_eeg/aligned/tcn_bin_s2 output/ttg_eeg/aligned/tcn_bin_s3 \
      --out .../ttg_eeg_gate/A_x3dbug_tcnbin
  python grader/eeg/joint_gate.py --protocol B --video 'output/ttg_vsubj/x3dfix_dual_s1_fold{f}' \
      --eeg 'output/ttg_eeg/subject/tcn_bin_fold{f}_s1' 'output/ttg_eeg/subject/tcn_bin_fold{f}_s2' \
      --out .../ttg_eeg_gate/B_x3dfix_tcnbin
  # the honest protocol-A number: that protocol-B deployment gate, frozen
  python grader/eeg/joint_gate.py --protocol A --video output/ttg_stage1/x3dfix_dual_s1 \
      --eeg output/ttg_eeg/aligned/tcn_bin_s1 output/ttg_eeg/aligned/tcn_bin_s2 --rows headline \
      --frozen_gate .../ttg_eeg_gate/B_x3dfix_tcnbin/gate.json --frozen_eeg_entry eeg_ens2 \
      --frozen_features EGRG --out .../ttg_eeg_gate/A_x3dfix_tcnbin_frozen
  Writes <out>/results.json, <out>/results.txt (readable tables) and, for B, <out>/gate.json.
"""
import argparse
import glob
import json
import os
import re
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
EEG_ROOT = os.path.realpath(os.environ.get("EEG_ROOT", "/work/mech-ai-scratch/alloy/EEG"))
# dhlib.py (validated loaders: path alignment, double-softmax repair, split_subjects, metrics)
# is vendored into grader/ (this file's parent directory); DHLIB_DIR overrides.
DH_DIR = os.path.realpath(os.environ.get("DHLIB_DIR", os.path.dirname(HERE)))
if not os.path.isfile(os.path.join(DH_DIR, "dhlib.py")):
    raise SystemExit(f"dhlib.py not found in {DH_DIR}: joint_gate.py needs the validated decision-headroom "
                     f"loaders. Set DHLIB_DIR to the directory that holds dhlib.py.")
for _p in (HERE, DH_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import dhlib as D                                          # noqa: E402  (validated loaders + metrics)
import check_folds as CF                                   # noqa: E402

OUT_PREFIX = os.path.join(EEG_ROOT, "output", "ttg_eeg")
TASKK = {"g3": 3, "g5": 5}
G3 = np.array([0, 1, 1, 2, 2])
N_FOLDS, SPLIT_SEED = 5, 49
UB = "best-epoch checkpoint selected on the scored clips: UPPER BOUND"
TOPK_FRAC = 0.25                                           # the SECONDARY EEG score is top-25%, fixed
SYS_ORDER = ["video", "post-hoc gate", "EGRG", "EGRG-top25", "EGRG-nested", "EGRG (frozen gate)",
             "oracle detection"]
GATE_INPUT = {"EGRG": "pooled", "EGRG-top25": "top25"}      # EGRG-nested: chosen per fit

# prototype numbers (output/ttg_ref/eeg/joint-gated/j10_log.txt): (video, post-hoc, EGRG[top25])
REG = {
    ("A", "g3"): dict(f1=(0.8105, 0.8220, 0.8231),
                      hits=([2652, 2225, 99], [2702, 2257, 99], [2707, 2266, 99]),
                      d_vs_video=(0.0127, 0.0083, 0.0189), d_vs_posthoc=(0.0012, -0.0005, 0.0032),
                      posthoc_vs_video=(0.0115, 0.0078, 0.0162)),
    ("A", "g5"): dict(f1=(0.6969, 0.7137, 0.7163),
                      hits=([2660, 157, 2019, 64, 9], [2706, 173, 2030, 64, 9], [2707, 178, 2034, 64, 9]),
                      d_vs_video=(0.0193, 0.0089, 0.0419), d_vs_posthoc=(0.0042, 0.0005, 0.0171),
                      posthoc_vs_video=(0.0150, 0.0021, 0.0245)),
    ("B", "g3"): dict(f1=(0.7327, 0.7541, 0.7669),
                      hits=([11712, 8782, 677], [12160, 9040, 675], [12236, 9389, 674]),
                      d_vs_video=(0.0340, 0.0264, 0.0424), d_vs_posthoc=(0.0129, 0.0077, 0.0197),
                      posthoc_vs_video=(0.0212, 0.0157, 0.0272)),
    ("B", "g5"): dict(f1=(0.4780, 0.4964, 0.5178),
                      hits=([11573, 203, 7332, 490, 20], [12138, 230, 7411, 489, 20],
                            [12247, 347, 7624, 490, 18]),
                      d_vs_video=(0.0396, 0.0280, 0.0508), d_vs_posthoc=(0.0213, 0.0131, 0.0288),
                      posthoc_vs_video=(0.0184, 0.0137, 0.0235)),
}
REG_LOADS = {"A_video": dict(n_loaded=40, dropped=0, double_softmax_repaired=4, pathless_admitted=8,
                             reordered=9),
             "A_eeg": dict(n_loaded=10, dropped=0, double_softmax_repaired=0, pathless_admitted=0),
             "B_video_clips_without_eeg": 45, "A_video_clips_without_eeg": 10}


# ============================================================================ small helpers

def key_of(p):
    p = str(p)
    if p.endswith("/video.mp4"):
        p = p[: -len("/video.mp4")]
    return p.rstrip("/")


def resolve(d):
    return d if os.path.isabs(d) else os.path.join(EEG_ROOT, d)


def rel(d):
    return os.path.relpath(d, EEG_ROOT) if d.startswith(EEG_ROOT) else d


def stage5_of(keys):
    """5-class label from the clip directory name (seizure_*_Stage_N -> N-1, else 0)."""
    out = []
    for k in keys:
        b = os.path.basename(k)
        out.append(int(re.search(r"Stage_(\d)", b).group(1)) - 1 if b.startswith("seizure_") else 0)
    return np.array(out)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def det_score(P):
    """P(seizure) of a posterior matrix: column 1 of a binary head, else 1 - P(non-seizure)."""
    return P[:, 1].astype(float) if P.shape[1] == 2 else 1.0 - P[:, 0]


def topk_packed(pw, off, frac):
    """Mean of the top max(1, round(frac*n)) window scores per clip, windows packed by offsets."""
    return np.array([np.sort(pw[off[i]:off[i + 1]])[-max(1, int(round(frac * (off[i + 1] - off[i])))):].mean()
                     for i in range(len(off) - 1)])


def topk_pool(pw, r, n, frac=0.25):
    """Mean of the top max(1, round(frac*n)) window scores per clip (jlib.topk_pool, verbatim)."""
    order = np.argsort(r, kind="stable"); ps = pw[order]; st = np.r_[0, np.cumsum(n)[:-1]]
    s = np.empty(len(n))
    for i in range(len(n)):
        v = ps[st[i]:st[i] + n[i]]; k = max(1, int(round(frac * len(v)))); s[i] = np.sort(v)[-k:].mean()
    return s


def comp(Pv, q):
    """argmax over [1 - q, q * P_V(g | sz)] (the verified gate's decision rule)."""
    sz = Pv[:, 1:] / np.clip(Pv[:, 1:].sum(1, keepdims=True), 1e-12, None)
    return np.c_[1 - q, q[:, None] * sz].argmax(1)


def comp_probs(Pv, q):
    sz = Pv[:, 1:] / np.clip(Pv[:, 1:].sum(1, keepdims=True), 1e-12, None)
    return np.c_[1 - q, q[:, None] * sz]


def loao_lr(X, ys, an, C=1.0):
    """Leave-one-animal-out logistic gate: q for each clip from a fit without its animal."""
    q = np.empty(len(ys)); coefs = []
    for g in np.unique(an):
        tr = an != g
        m = LogisticRegression(C=C, max_iter=5000).fit(X[tr], ys[tr])
        q[~tr] = m.predict_proba(X[~tr])[:, 1]; coefs.append(np.r_[m.coef_[0], m.intercept_])
    return q, np.array(coefs)


def gate_features(eeg_input):
    return ["logit P_V(sz)", f"logit P_E(sz) {eeg_input}", "intercept"]


NESTED_RULE = ("EEG input (pooled vs top25) chosen by nested leave-one-animal-out: inner LOAO on the "
               "other animals, task macro-F1 of the decision, tie -> pooled; deployment gate = the same "
               "choice on all animals, fitted on all")


def fit_all(X, ys, C=1.0):
    m = LogisticRegression(C=C, max_iter=5000).fit(X, ys)
    return np.r_[m.coef_[0], m.intercept_]


def newlog(source, selection):
    return dict(source=source, selection=selection, n_requested=0, n_loaded=0, dropped=[],
                double_softmax_repaired=[], pathless_admitted=[], reordered=[], notes=[])


def logline(lg):
    s = (f"{lg['n_loaded']}/{lg['n_requested']} loaded; dropped={len(lg['dropped'])} "
         f"double-softmax repaired={len(lg['double_softmax_repaired'])} "
         f"path-less admitted={len(lg['pathless_admitted'])} reordered={len(lg['reordered'])}")
    if lg["dropped"]:
        s += f" DROPPED: {lg['dropped']}"
    if lg.get("notes"):
        s += " | " + "; ".join(lg["notes"])
    return s


def filter_members(b, patterns, task, what):
    """Keep only members matching any fnmatch pattern; logged. {task} is substituted for video
    (task=g3/g5); the EEG bundle is shared by both tasks, so task=None leaves patterns as given."""
    if not patterns:
        return b
    import fnmatch
    pats = [p.replace("{task}", task) if task else p for p in patterns]
    keep = {n: v for n, v in b["members"].items() if any(fnmatch.fnmatch(n, p) for p in pats)}
    if not keep:
        raise SystemExit(f"--{what}_members {pats} matches none of {sorted(b['members'])}")
    b = dict(b, members=keep)
    b["log"] = dict(b["log"], notes=b["log"]["notes"] + [f"member filter {pats}: {len(keep)} of "
                                                         f"{b['log']['n_loaded']} loaded members kept"])
    b["ens_name"] = f"{what}_ens{len(keep)}(filtered)"
    return b


def ensemble(members, kind):
    names = list(members)
    if kind == "video":
        return np.mean([members[n] for n in names], 0)
    return dict(stored=np.mean([members[n]["stored"] for n in names], 0),
                top25=np.mean([members[n]["top25"] for n in names], 0))


def epoch_selection(run_dir, epoch, what):
    """Label a per-epoch dump: fixed LAST epoch (no selection) or an epoch picked by the caller."""
    ep_total = None
    for f in ("config.json", "results.json"):
        p = os.path.join(run_dir, f)
        if os.path.exists(p):
            try:
                rk = json.load(open(p)).get("run_key", {})
                ep_total = rk.get("epochs", ep_total)
            except Exception:
                pass
    if ep_total is not None and int(ep_total) == int(epoch):
        return f"{what}: fixed LAST epoch {epoch} of {ep_total} (pre-registered, no selection)"
    return (f"{what}: epoch {epoch} of {ep_total if ep_total is not None else '?'} chosen by the caller "
            f"(if chosen by looking at these val clips it is selection on the scored set)")


# ============================================================================ video loaders

def load_video_stored_A(task):
    keys, y, y5 = D.reference_val(task)              # canonical 5,289 order, asserted vs mvit s1
    dirs = sorted(os.path.relpath(d, D.EEG_ROOT) for d in
                  glob.glob(os.path.join(D.EEG_ROOT, f"output/v3_vidseeds/*_{task}_s*")))
    ref_raw_y = np.load(os.path.join(D.EEG_ROOT, f"output/v3_vidseeds/mvit_{task}_s1/val_preds.npz"))["y"]
    P, names, lg = D.load_members(dirs, keys, ref_raw_y, y)
    log = newlog(f"output/v3_vidseeds/*_{task}_s* via dhlib.load_members", UB)
    log.update(n_requested=len(dirs), n_loaded=len(names), dropped=lg["dropped"],
               double_softmax_repaired=lg["double_softmax_repaired"],
               pathless_admitted=lg["pathless_admitted"], reordered=lg["reordered"])
    o = np.argsort(keys, kind="stable")
    members = {os.path.basename(n): P[i][o] for i, n in enumerate(names)}
    return dict(keys=keys[o], y5=y5[o], members=members, log=log, selection=UB,
                ens_name=f"ens{len(names)}", backbones=True)


def load_video_stored_B(task):
    items = D.items_from_cache_index()
    log = newlog(f"output/subject_cv/vid_{task}_fold{{0-4}} (R(2+1)D; path-less, idx + split_subjects)", UB)
    K, Y, P = [], [], []
    for f in range(N_FOLDS):
        d = f"output/subject_cv/vid_{task}_fold{f}"
        log["n_requested"] += 1
        fn = os.path.join(EEG_ROOT, d, "val_preds.npz")
        if not os.path.exists(fn):
            log["dropped"].append((d, "missing val_preds.npz")); continue
        z = np.load(fn, allow_pickle=True)
        _, va, _, _ = D.split_subjects(items, SPLIT_SEED, fold=f, n_folds=N_FOLDS)
        if "path" in z.files:
            keys = np.array([key_of(p) for p in z["path"]])
            y5 = np.array([dict((key_of(i[0]), i[1]) for i in va)[k] for k in keys])
        else:
            if len(z["idx"]) != len(va) or not np.array_equal(np.sort(z["idx"]), np.arange(len(va))):
                log["dropped"].append((d, "no path and idx is not a permutation of the fold")); continue
            sel = [va[i] for i in z["idx"]]
            y5 = np.array([i[1] for i in sel])
            ylab = y5 if task == "g5" else G3[y5]
            if not np.array_equal(ylab, z["y"]):
                log["dropped"].append((d, "no path and label order differs")); continue
            keys = np.array([key_of(i[0]) for i in sel])
            log["pathless_admitted"].append(d)
        p = z["probs"].astype(np.float64)
        if D.is_double_softmax(p):
            p = D.unsquash(p); log["double_softmax_repaired"].append(d)
        K.append(keys); Y.append(y5); P.append(p); log["n_loaded"] += 1
    if log["n_loaded"] != N_FOLDS:
        raise SystemExit(f"stored OOF video {task}: only {log['n_loaded']}/5 folds usable: {log['dropped']}")
    keys, y5, P = np.concatenate(K), np.concatenate(Y), np.concatenate(P)
    o = np.argsort(keys, kind="stable")
    return dict(keys=keys[o], y5=y5[o], members={f"r2plus1d_{task}_subject_folds": P[o]}, log=log,
                selection=UB, ens_name=f"r2plus1d_{task}_subject_folds", backbones=False)


def _fold_dirs(spec, protocol):
    if protocol == "B":
        if "{f}" not in spec:
            raise SystemExit(f"protocol B needs a fold TEMPLATE containing {{f}}: {spec}")
        return [resolve(spec.format(f=f)) for f in range(N_FOLDS)]
    return [resolve(spec)]


def _check_fold_animals(keys, f, log, what, folds, side=None, allow_partial=False):
    """Fold f of a protocol-B run must hold exactly the split_subjects animals of fold f and,
    for new dumps (side = 'video' / 'eeg'), exactly that fold's clips."""
    an = set(D.animal_session(keys)[0].tolist())
    exp = folds[f]["eeg"]
    if an != exp and not (allow_partial and an <= exp):
        log["dropped"].append((what, f"fold {f} animals {sorted(an)} != split_subjects fold {sorted(exp)}"))
        return False
    if side is not None:
        want = folds[f]["video_keys" if side == "video" else "eeg_keys"]
        got = set(keys.tolist())
        if got != want:
            if allow_partial and got <= want:
                log["notes"].append(f"{what}: PARTIAL fold {f}: {len(got)} of {len(want)} clips (smoke test)")
            else:
                log["dropped"].append((what, f"fold {f} holds {len(got)} clips, the split has {len(want)} "
                                             f"({len(got - want)} outside it)"))
                return False
    return True


def load_video_dirs(specs, epoch, task, protocol, allow_partial, folds=None):
    log = newlog(f"grader/train_grader.py dumps val_ep{epoch:02d}.npz", None)
    members, keys0, y50, sels = {}, None, None, []
    for spec in specs:
        log["n_requested"] += 1
        name = os.path.basename(spec.rstrip("/")).replace("{f}", "F")
        K, Y, P, ok = [], [], [], True
        for f, d in enumerate(_fold_dirs(spec, protocol)):
            fn = os.path.join(d, f"val_ep{epoch:02d}.npz")
            if not os.path.exists(fn):
                log["dropped"].append((rel(d), f"missing val_ep{epoch:02d}.npz")); ok = False; break
            z = np.load(fn, allow_pickle=True)
            if f"probs_{task}" not in z.files:
                log["dropped"].append((rel(d), f"no {task} head")); ok = False; break
            if "path" not in z.files:           # train_grader always stores path; refuse otherwise
                log["dropped"].append((rel(d), "no path (cannot align)")); ok = False; break
            keys = np.array([key_of(p) for p in z["path"]])
            if protocol == "B" and not _check_fold_animals(keys, f, log, rel(d), folds, "video", allow_partial):
                ok = False; break
            p = z[f"probs_{task}"].astype(np.float64)
            if D.is_double_softmax(p):
                p = D.unsquash(p); log["double_softmax_repaired"].append(rel(d))
            K.append(keys); Y.append(z["y5"].astype(int)); P.append(p)
            sels.append(epoch_selection(d, epoch, "video"))
        if not ok:
            continue
        keys, y5, P = np.concatenate(K), np.concatenate(Y), np.concatenate(P)
        if len(set(keys.tolist())) != len(keys):
            log["dropped"].append((name, "duplicate clip keys")); continue
        o = np.argsort(keys, kind="stable")
        keys, y5, P = keys[o], y5[o], P[o]
        if keys0 is None:
            keys0, y50 = keys, y5
        elif not (np.array_equal(keys, keys0) and np.array_equal(y5, y50)):
            log["dropped"].append((name, "different clip list / labels from the first member")); continue
        if name in members:
            name = f"{name}#{log['n_loaded']}"
        members[name] = P; log["n_loaded"] += 1
    if not members:
        raise SystemExit(f"no usable video run: {log['dropped']}")
    if protocol == "A":
        ref, _, ref5 = D.reference_val(task)
        full = set(ref.tolist())
        if set(keys0.tolist()) != full:
            if not (set(keys0.tolist()) <= full and allow_partial):
                raise SystemExit(f"video clip set ({len(keys0)}) is not the 5,289-clip seed-49 val set "
                                 f"(pass --allow_partial only for smoke tests)")
            log["notes"].append(f"PARTIAL clip set: {len(keys0)} of {len(full)} (smoke test)")
        rp = dict(zip(ref.tolist(), ref5.tolist()))
        if not np.array_equal(y50, np.array([rp[k] for k in keys0])):
            raise SystemExit("video y5 disagrees with the reconstructed seed-49 labels")
    sel = sorted(set(sels))
    log["selection"] = sel[0] if len(sel) == 1 else sel
    return dict(keys=keys0, y5=y50, members=members, log=log, selection=log["selection"],
                ens_name=f"video_ens{len(members)}" if len(members) > 1 else next(iter(members)),
                backbones=False)


# ============================================================================ EEG loaders

def _eeg_stored_run(d):
    """One stored train_pooled_eeg run: keys, y (task labels), P, stored P(sz), top25, window ok."""
    z = np.load(os.path.join(d, "val_clip_preds.npz"), allow_pickle=True)
    P = z["probs"].astype(np.float64)
    sq = D.is_double_softmax(P)
    if sq:
        P = D.unsquash(P)
    keys = np.array([key_of(p) for p in z["path"]]) if "path" in z.files else None
    top = None
    wf = os.path.join(d, "val_window_preds.npz")
    if os.path.exists(wf):
        w = np.load(wf)
        pos = {c: i for i, c in enumerate(w["clips"])}
        r = np.array([pos[c] for c in w["cid"]])
        pw = 1.0 - w["probs"].astype(float)[:, 0]
        top = topk_pool(pw, r, np.bincount(r, minlength=len(z["y"])))
    return dict(y=z["y"].astype(int), P=P, sq=sq, keys=keys, stored=det_score(P), top25=top,
                sub=np.asarray(z["sub"]) if "sub" in z.files else None)


def load_eeg_stored_A():
    runs = sorted(glob.glob(os.path.join(EEG_ROOT, "output/v3_eegalign/*_g3_s*")))
    log = newlog("output/v3_eegalign/*_g3_s* (GRU+TCN g3 heads, logmean; P_E(sz) = 1 - P(non-sz))", UB)
    raw = {}
    for d in runs:
        log["n_requested"] += 1
        if not os.path.exists(os.path.join(d, "val_clip_preds.npz")):
            log["dropped"].append((rel(d), "missing val_clip_preds.npz")); continue
        raw[os.path.basename(d)] = _eeg_stored_run(d)
    ref = next(n for n in sorted(raw) if raw[n]["keys"] is not None)
    o = np.argsort(raw[ref]["keys"], kind="stable")
    rk, ry = raw[ref]["keys"][o], raw[ref]["y"]
    members = {}
    for n, r in raw.items():
        if r["sq"]:
            log["double_softmax_repaired"].append(n)
        if r["top25"] is None:
            log["dropped"].append((n, "no val_window_preds.npz (top25 impossible)")); continue
        if r["keys"] is not None:
            q = np.argsort(r["keys"], kind="stable")
            if not (np.array_equal(r["keys"][q], rk) and np.array_equal(r["y"][q], ry[o])):
                log["dropped"].append((n, "different clip list / labels")); continue
            if not np.array_equal(r["keys"], raw[ref]["keys"]):
                log["reordered"].append(n)
        elif len(r["y"]) == len(ry) and np.array_equal(r["y"], ry):
            q = o; log["pathless_admitted"].append(n)
        else:
            log["dropped"].append((n, "no path and label order differs")); continue
        members[n] = dict(stored=r["stored"][q], top25=r["top25"][q]); log["n_loaded"] += 1
    y5 = stage5_of(rk)
    if not np.array_equal(G3[y5], ry[o]):
        raise SystemExit("v3_eegalign labels disagree with the clip names")
    return dict(keys=rk, y5=y5, members=members, log=log, selection=UB,
                ens_name=f"eegalign_ens{len(members)}", score_name="stored logmean",
                head="3-class g3", pooling="logmean", member_arch={n: n.split("_")[0] for n in members})


def load_eeg_stored_B(folds):
    log = newlog("output/v3_subject_cv/eeg_bin_fold{0-4} (GRU binary, mean pooling)", UB)
    K, S, T = [], [], []
    for f in range(N_FOLDS):
        d = os.path.join(EEG_ROOT, f"output/v3_subject_cv/eeg_bin_fold{f}")
        log["n_requested"] += 1
        r = _eeg_stored_run(d)
        if r["sq"]:
            log["double_softmax_repaired"].append(rel(d))
        if r["keys"] is None:
            raise SystemExit(f"{d}: stored EEG fold without path")
        if not _check_fold_animals(r["keys"], f, log, rel(d), folds):
            raise SystemExit(f"EEG fold {f} animals do not match the split: {log['dropped']}")
        K.append(r["keys"]); S.append(r["stored"]); T.append(r["top25"]); log["n_loaded"] += 1
    keys, s, t = np.concatenate(K), np.concatenate(S), np.concatenate(T)
    o = np.argsort(keys, kind="stable")
    return dict(keys=keys[o], y5=stage5_of(keys[o]), log=log, selection=UB,
                members={"gru_bin_subject_folds": dict(stored=s[o], top25=t[o])},
                ens_name="gru_bin_subject_folds", score_name="stored mean-pooled",
                head="binary", pooling="mean", member_arch={"gru_bin_subject_folds": "gru"})


def load_eeg_dirs(specs, epoch, protocol, score, allow_partial, folds=None):
    fld = "p_logmean" if score == "logmean" else "p_mean"
    log = newlog(f"grader/eeg/train_eeg_det.py dumps val_clip_ep{epoch:02d}.npz ({fld} / p_top25)", None)
    members, keys0, y50, sels, arch = {}, None, None, [], {}
    for spec in specs:
        log["n_requested"] += 1
        name = os.path.basename(spec.rstrip("/")).replace("{f}", "F")
        K, Y, S, T, A, ok = [], [], [], [], set(), True
        for f, d in enumerate(_fold_dirs(spec, protocol)):
            fn = os.path.join(d, f"val_clip_ep{epoch:02d}.npz")
            if not os.path.exists(fn):
                log["dropped"].append((rel(d), f"missing val_clip_ep{epoch:02d}.npz")); ok = False; break
            z = np.load(fn, allow_pickle=True)
            if "path" not in z.files:
                log["dropped"].append((rel(d), "no path (cannot align)")); ok = False; break
            keys = np.array([key_of(p) for p in z["path"]])
            if protocol == "B" and not _check_fold_animals(keys, f, log, rel(d), folds, "eeg", allow_partial):
                ok = False; break
            if protocol == "A" and str(z["session_universe"]) != "video":
                log["dropped"].append((rel(d), f"session_universe={z['session_universe']}: not the aligned "
                                              f"split, cannot be paired with video")); ok = False; break
            P2 = z["probs_logmean"] if score == "logmean" else z["probs_mean"]
            if D.is_double_softmax(P2.astype(float)):
                log["dropped"].append((rel(d), "double-softmaxed EEG clip posterior (unexpected)")); ok = False; break
            # the SECONDARY score is DEFINED as the top 25% (frac 0.25 was chosen after looking, so it
            # stays fixed): always recomputed at TOPK_FRAC from the packed windows; the dump's own
            # p_top25 is only cross-checked when the trainer used the same fraction
            off, wp = z["win_offsets"], z["win_probs"].astype(float)
            tk = topk_packed(1.0 - wp[:, 0], off, TOPK_FRAC)
            dfrac = float(z["topk_frac"]) if "topk_frac" in z.files else None
            if dfrac is not None and abs(dfrac - TOPK_FRAC) > 1e-12:
                log["notes"].append(f"{rel(d)}: dump p_top25 used topk_frac={dfrac}; top25 recomputed at "
                                    f"{TOPK_FRAC} from the packed windows (dump value ignored)")
            elif not np.allclose(tk, z["p_top25"], atol=1e-9):
                log["notes"].append(f"{rel(d)}: dump p_top25 differs from the recomputation at the same "
                                    f"topk_frac={TOPK_FRAC} (max |diff| {np.abs(tk - z['p_top25']).max():.2e}); "
                                    f"using recomputed")
            A.add(str(z["arch"]) if "arch" in z.files else "unknown")
            K.append(keys); Y.append(z["y5"].astype(int)); S.append(z[fld].astype(float)); T.append(tk)
            sels.append(epoch_selection(d, epoch, "EEG"))
        if not ok:
            continue
        keys, y5, s, t = np.concatenate(K), np.concatenate(Y), np.concatenate(S), np.concatenate(T)
        o = np.argsort(keys, kind="stable")
        keys, y5, s, t = keys[o], y5[o], s[o], t[o]
        if keys0 is None:
            keys0, y50 = keys, y5
        elif not (np.array_equal(keys, keys0) and np.array_equal(y5, y50)):
            log["dropped"].append((name, "different clip list / labels from the first member")); continue
        if not np.array_equal(y5, stage5_of(keys)):
            raise SystemExit(f"{name}: dumped y5 disagrees with the clip names")
        members[name] = dict(stored=s, top25=t); log["n_loaded"] += 1
        arch[name] = "+".join(sorted(A))
    if not members:
        raise SystemExit(f"no usable EEG run: {log['dropped']}")
    if protocol == "A" and len(keys0) != 5279:
        if not allow_partial:
            raise SystemExit(f"EEG clip set has {len(keys0)} clips, not the aligned 5,279 (--allow_partial "
                             f"only for smoke tests)")
        log["notes"].append(f"PARTIAL clip set: {len(keys0)} of 5,279 (smoke test)")
    sel = sorted(set(sels))
    log["selection"] = sel[0] if len(sel) == 1 else sel
    return dict(keys=keys0, y5=y50, members=members, log=log, selection=log["selection"],
                ens_name=f"eeg_ens{len(members)}" if len(members) > 1 else next(iter(members)),
                score_name=f"pooled {score}", head="binary", pooling=score, member_arch=arch)


def eeg_detector(eb):
    """What a gate's EEG input came from: the architectures of the (kept) members, the head and the
    clip pooling. Written into gate.json and compared when a gate is applied frozen."""
    return dict(arch="+".join(sorted({eb["member_arch"].get(n, "unknown") for n in eb["members"]})),
                head=eb["head"], pooling=eb["pooling"])


def gate_input_signature(det, eeg_input):
    """The EEG score that a gate's coefficient b multiplies: detector arch + head + score."""
    if eeg_input not in ("pooled", "top25"):
        raise ValueError(f"eeg_input must be 'pooled' or 'top25', got {eeg_input!r}")
    return dict(arch=det["arch"], head=det["head"],
                score="top25 (mean of top 25% window P(sz))" if eeg_input == "top25" else f"pooled {det['pooling']}")


# ============================================================================ scoring

class Pairing:
    """Common clips of a video and an EEG bundle, the animal bootstrap, and the scorers."""

    def __init__(s, vb, eb, task, reps, seed):
        vpos = {k: i for i, k in enumerate(vb["keys"])}
        epos = {k: i for i, k in enumerate(eb["keys"])}
        common = sorted(set(vpos) & set(epos))
        if not common:
            raise SystemExit("video and EEG share no clip")
        s.iv = np.array([vpos[k] for k in common]); s.ie = np.array([epos[k] for k in common])
        s.keys = np.array(common)
        y5v, y5e = vb["y5"][s.iv], eb["y5"][s.ie]
        if not np.array_equal(y5v, y5e):
            raise SystemExit(f"video and EEG disagree on {int((y5v != y5e).sum())} clip labels")
        s.log = dict(n_video=len(vb["keys"]), n_eeg=len(eb["keys"]), n_scored=len(common),
                     video_clips_dropped_no_eeg=len(vb["keys"]) - len(common),
                     eeg_clips_dropped_no_video=len(eb["keys"]) - len(common))
        s.task, s.K = task, TASKK[task]
        s.y5 = y5v
        s.y = y5v if task == "g5" else G3[y5v]
        s.ys = (y5v > 0).astype(int)
        s.an, s.se = D.animal_session(s.keys)
        s.ug = np.unique(s.an)
        s.W = D.boot_weights(len(s.ug), reps, seed)
        s.names = D.NAMES[task]
        s.sev = D.SEV[task]; s.mild = D.MILD[task]

    # ---- the gates
    def systems(s, Pv, e_st, e_tk, frozen=None, top25=True, nested=False):
        sv = det_score(Pv) if Pv.shape[1] == 2 else 1.0 - Pv[:, 0]
        Q = {"video": None, "post-hoc gate": 0.5 * (sv + e_st)}
        gates = {}
        E = {"pooled": e_st, "top25": e_tk}
        X = {k: np.c_[logit(sv), logit(v)] for k, v in E.items()}
        for sname in (("EGRG", "EGRG-top25") if top25 else ("EGRG",)):
            inp = GATE_INPUT[sname]
            q, co = loao_lr(X[inp], s.ys, s.an)
            Q[sname] = q
            gates[sname] = dict(eeg_input=inp, features=gate_features(inp),
                                loao_coef_mean=co.mean(0).round(4).tolist(), loao_coef_sd=co.std(0).round(4).tolist(),
                                fit_all=fit_all(X[inp], s.ys).round(4).tolist())
        if nested and top25:
            Q["EGRG-nested"], picks = s.nested_gate(Pv, X["pooled"], X["top25"])
            # deployment analogue of the nested rule: the same choice made on ALL animals (the LOAO
            # task macro-F1 of the two gates above; tie -> pooled), then fitted on all of them
            f1 = {k: float(D.cm_metrics(D.confusion(s.y, comp(Pv, Q[k]), s.K))["macro_f1"])
                  for k in ("EGRG", "EGRG-top25")}
            inp = "top25" if f1["EGRG-top25"] > f1["EGRG"] else "pooled"
            n_t = sum(v["pick"] == "top25" for v in picks.values())
            gates["EGRG-nested"] = dict(eeg_input=inp, features=gate_features(inp),
                                        fit_all=fit_all(X[inp], s.ys).round(4).tolist(), rule=NESTED_RULE,
                                        outer_picks_top25=n_t, outer_picks_pooled=len(picks) - n_t,
                                        all_animals_loao_macro_f1={k: round(v, 4) for k, v in f1.items()},
                                        outer=picks)
        if frozen is not None:
            inp = frozen["eeg_input"]
            if inp not in E:                                     # never guess the EEG input
                raise SystemExit(f"frozen gate eeg_input={inp!r}: must be 'pooled' or 'top25'")
            a_, b_, c_ = frozen["coef"]
            Q["EGRG (frozen gate)"] = 1.0 / (1.0 + np.exp(-(a_ * logit(sv) + b_ * logit(E[inp]) + c_)))
            gates["EGRG (frozen gate)"] = dict(coef=[a_, b_, c_], eeg_input=inp, features=frozen["features"],
                                               system=frozen["system"], source=frozen["source"],
                                               fitted_on=frozen["signature"], applied_to=frozen["applied_to"],
                                               compatible=frozen["compatible"])
        Q["oracle detection"] = s.ys.astype(float)
        preds = {n: (Pv.argmax(1) if q is None else comp(Pv, q)) for n, q in Q.items()}
        probs = {n: (Pv if q is None else comp_probs(Pv, q)) for n, q in Q.items() if n != "oracle detection"}
        return preds, probs, Q, gates

    def nested_gate(s, Pv, Xp, Xt):
        """EGRG with its EEG input (pooled vs top25) chosen by NESTED leave-one-animal-out: for each
        held-out animal, an inner LOAO on the remaining animals scores both gates by task macro-F1
        of the decision (tie -> pooled, the primary); the winner is fitted on the remaining animals
        and predicts the held-out one. Nothing about the held-out animal enters its choice."""
        q = np.empty(len(s.ys)); picks = {}
        for g in s.ug:
            tr = s.an != g
            f1 = []
            for X in (Xp, Xt):
                qi, _ = loao_lr(X[tr], s.ys[tr], s.an[tr])
                f1.append(float(D.cm_metrics(D.confusion(s.y[tr], comp(Pv[tr], qi), s.K))["macro_f1"]))
            inp = "top25" if f1[1] > f1[0] else "pooled"
            X = Xt if inp == "top25" else Xp
            m = LogisticRegression(C=1.0, max_iter=5000).fit(X[tr], s.ys[tr])
            q[~tr] = m.predict_proba(X[~tr])[:, 1]
            picks[str(g)] = dict(pick=inp, inner_f1_pooled=round(f1[0], 4), inner_f1_top25=round(f1[1], 4))
        return q, picks

    # ---- metrics
    def point(s, pred, prob=None, q=None):
        C = D.confusion(s.y, pred, s.K); m = D.cm_metrics(C)
        present = (C.sum(1) > 0) | (C.sum(0) > 0)
        hit = int(C[np.ix_(s.sev, s.sev)].sum()); nsev = int(C[s.sev].sum())
        pd_ = (pred > 0).astype(int)
        Cd = D.confusion(s.ys, pd_, 2); md = D.cm_metrics(Cd)
        r = dict(macro_f1=float(m["macro_f1"]), mcc=float(m["mcc"]), bal_acc=float(m["bal_acc"]),
                 macro_precision=float(m["precision"][present].mean()),
                 macro_recall=float(m["recall"][present].mean()),
                 per_class={s.names[i]: dict(hit=int(C[i, i]), n=int(C[i].sum()),
                                             recall=round(float(m["recall"][i]), 4),
                                             precision=round(float(m["precision"][i]), 4))
                            for i in range(s.K)},
                 severe=dict(hit=hit, n=nsev, recall=hit / nsev if nsev else float("nan")),
                 det=dict(macro_f1=float(md["macro_f1"]), seizure_f1=float(md["f1"][1]),
                          false_alarms=int(Cd[0, 1]), misses=int(Cd[1, 0]),
                          misses_by_stage={f"S{st + 1}": [int(((s.y5 == st) & (pd_ == 0)).sum()),
                                                          int((s.y5 == st).sum())] for st in (1, 2, 3, 4)}),
                 confusion=C.tolist())
        if prob is not None:
            try:
                pres = sorted(set(s.y.tolist())); pp = prob[:, pres] / prob[:, pres].sum(1, keepdims=True)
                r["auroc_ovr_macro"] = float(roc_auc_score(s.y, pp, multi_class="ovr", average="macro",
                                                           labels=pres))
            except ValueError:
                r["auroc_ovr_macro"] = float("nan")
        if q is not None:
            r["det"]["auroc_q"] = float(D.auroc(q[s.ys == 1], q[s.ys == 0]))
        return r

    def boot(s, pred):
        C, _ = D.per_group_cm(s.y, pred, s.an, s.K, s.ug)
        m = D.boot_cm_metrics(C, s.W, s.task)
        Cd, _ = D.per_group_cm(s.ys, (pred > 0).astype(int), s.an, 2, s.ug)
        md = D.cm_metrics(np.einsum("rg,gij->rij", s.W, Cd))
        return dict(macro_f1=m["macro_f1"], mcc=m["mcc"], sev_recall=m["sev_recall"], det_f1=md["macro_f1"])

    def within_session_severe(s, Pv):
        """Severe-vs-mild AUROC within session of VIDEO's grading: log P(severe) - log P(mild) of the
        video posterior. Every gate grades with video's within-seizure split (argmax over
        [1-q, q*P_V(g|sz)]), and q cancels in the ratio, so this is every system's value by
        construction. It is NOT recomputed from a composed vector: repaired double-softmax runs
        hold exact zeros, and log(clip(q*0)) would let q break ties that video leaves tied."""
        sc = np.log(np.clip(Pv[:, s.sev].sum(1), 1e-300, None)) - np.log(np.clip(Pv[:, s.mild].sum(1), 1e-300, None))
        is_sev, is_mild = np.isin(s.y, s.sev), np.isin(s.y, s.mild)
        return float(D.within_session_auc(sc, is_sev, is_mild, s.se)[0])

    def row(s, Pv, e_st, e_tk, frozen=None, bootstrap=True, nested=False):
        preds, probs, Q, gates = s.systems(Pv, e_st, e_tk, frozen, nested=nested)
        out = dict(n_clips=int(len(s.y)), n_animals=int(len(s.ug)),
                   class_counts=np.bincount(s.y, minlength=s.K).tolist(), gates=gates, systems={})
        B = {n: s.boot(p) for n, p in preds.items()} if bootstrap else {}
        for n in [x for x in SYS_ORDER if x in preds]:
            r = s.point(preds[n], probs.get(n), None if n in ("video",) else Q[n])
            if bootstrap:
                r["ci"] = {k: D.ci(v) for k, v in B[n].items()}
                if n != "video":
                    r["d_vs_video"] = {k: D.paired_summary(B[n][k] - B["video"][k]) for k in B[n]}
                if n not in ("video", "post-hoc gate"):
                    r["d_vs_posthoc"] = {k: D.paired_summary(B[n][k] - B["post-hoc gate"][k]) for k in B[n]}
            out["systems"][n] = r
        out["within_session_severe_auroc"] = s.within_session_severe(Pv)
        # the invariant behind "within-session severity is video's by construction": every clip a
        # gate calls seizure gets video's within-seizure argmax grade (checked, not assumed)
        vg = 1 + Pv[:, 1:].argmax(1)
        out["grading_untouched"] = {n: bool(np.array_equal(p[p > 0], vg[p > 0])) for n, p in preds.items()
                                    if n != "video"}
        out["grading_untouched_all"] = bool(all(out["grading_untouched"].values()))
        return out

    def quick(s, Pv, e_st, e_tk):
        """Point estimates only (single x single pairs)."""
        preds, _, _, gates = s.systems(Pv, e_st, e_tk)
        r = {}
        for n, p in preds.items():
            C = D.confusion(s.y, p, s.K)
            r[n] = dict(f1=float(D.cm_metrics(C)["macro_f1"]), sev_hit=int(C[np.ix_(s.sev, s.sev)].sum()))
        return r, gates["EGRG"]["loao_coef_mean"]


def dist(v):
    v = np.asarray(v, float)
    return dict(n=int(len(v)), mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                p5=float(np.percentile(v, 5)), p95=float(np.percentile(v, 95)),
                min=float(v.min()), max=float(v.max()), share_gt0=float((v > 0).mean()))


def backbone_of(name, task):
    return name.split(f"_{task}_s")[0]


def analyse(protocol, task, vb, eb, a, frozen=None, pairs=True):
    """Every row for one (protocol, task, video bundle, EEG bundle)."""
    P = Pairing(vb, eb, task, a.reps, a.boot_seed)
    Vm = {n: M[P.iv] for n, M in vb["members"].items()}
    Em = {n: dict(stored=M["stored"][P.ie], top25=M["top25"][P.ie]) for n, M in eb["members"].items()}
    v_ens, e_ens = ensemble(Vm, "video"), ensemble(Em, "eeg")
    fz = None
    if frozen:
        fz = [g for g in frozen if g["task"] == task]
        fz = fz[0] if fz else None
    res = dict(protocol=protocol, task=task, pairing=P.log, n_video_members=len(Vm), n_eeg_members=len(Em),
               video_selection=vb["selection"], eeg_selection=eb["selection"], eeg_score=eb["score_name"],
               rows=[])
    t0 = time.time()

    def add(vname, vkind, Pv, ename, ekind, e):
        nested = a.nested == "rows" or (a.nested == "headline" and not res["rows"])
        r = P.row(Pv, e["stored"], e["top25"], frozen=fz, nested=nested)
        r.update(video=vname, video_kind=vkind, eeg=ename, eeg_kind=ekind)
        if "EGRG (frozen gate)" in r["gates"]:           # a gate is video-model specific: say which
            gf = r["gates"]["EGRG (frozen gate)"]
            gf["applied_to_row"] = f"{vname} x {ename}"
            gf["applied_to"] = gate_input_signature(row_detector(eb, ename), gf["eeg_input"])
            gf["compatible"] = gf["fitted_on"] == gf["applied_to"]
        res["rows"].append(r)

    vk = f"ensemble of {len(Vm)}" if len(Vm) > 1 else "single network"
    ek = f"ensemble of {len(Em)}" if len(Em) > 1 else "single network"
    add(vb["ens_name"] if len(Vm) > 1 else next(iter(Vm)), vk, v_ens,
        eb["ens_name"] if len(Em) > 1 else next(iter(Em)), ek, e_ens)
    if len(Vm) > 1 and a.rows != "headline":
        if vb.get("backbones"):
            groups = {}
            for n in Vm:
                groups.setdefault(backbone_of(n, task), []).append(n)
            for g, ns in sorted(groups.items()):
                add(f"{g} ({len(ns)} seeds)", "backbone ensemble", np.mean([Vm[n] for n in ns], 0),
                    eb["ens_name"], ek, e_ens)
        for n in Vm:
            add(n, "single network", Vm[n], eb["ens_name"], ek, e_ens)
    if len(Em) > 1 and a.rows != "headline":
        for n in Em:
            add(vb["ens_name"] if len(Vm) > 1 else next(iter(Vm)), vk, v_ens, n, "single network", Em[n])
    if pairs and len(Vm) * len(Em) > 1 and a.rows == "all":
        pr = []
        for vn, Pv in Vm.items():
            for en, e in Em.items():
                q, co = P.quick(Pv, e["stored"], e["top25"])
                pr.append(dict(video_run=vn, eeg_run=en, systems=q, egrg_loao_coef_mean=co))
        f = lambda k: np.array([r["systems"][k]["f1"] for r in pr])
        sh = lambda k: np.array([r["systems"][k]["sev_hit"] for r in pr])
        res["single_pairs"] = dict(
            n_pairs=len(pr),
            posthoc_minus_video=dist(f("post-hoc gate") - f("video")),
            egrg_minus_video=dist(f("EGRG") - f("video")),
            egrg_minus_posthoc=dist(f("EGRG") - f("post-hoc gate")),
            top25_minus_egrg=dist(f("EGRG-top25") - f("EGRG")),
            egrg_top25_minus_posthoc=dist(f("EGRG-top25") - f("post-hoc gate")),
            severe_hits_egrg_minus_video=dist(sh("EGRG") - sh("video")),
            severe_hits_posthoc_minus_video=dist(sh("post-hoc gate") - sh("video")),
            pairs=pr)
    res["secs"] = round(time.time() - t0, 1)
    return res


# ============================================================================ text output

def fmt_ci(s):
    return f"{s['mean']:+.4f} [{s['lo']:+.4f},{s['hi']:+.4f}]"


def render(results):
    L = []
    w = L.append
    w(f"EGRG joint gate -- {results['created']}  ({results['mode']})")
    w("decision: argmax over [1-q, q*P_V(g|sz)]; gates LOAO-fitted (LogisticRegression C=1); animal "
      f"bootstrap {results['reps']} reps (seed {results['boot_seed']}); macro-F1 unless stated")
    w("CIs: animals are resampled with every gate's LOAO outputs q held FIXED, so the variability of "
      "fitting the gate is not propagated into the EGRG CIs (they are narrower than a refit-per-replicate CI).")
    w("PRIMARY gate: EGRG = [logit P_V(sz), logit pooled P_E(sz)]. SECONDARY: EGRG-top25 (frac 0.25 chosen "
      "after looking; costs S5 recall at g5 OOF) and EGRG-nested (pooled vs top25 chosen by nested LOAO).")
    for blk in results["blocks"]:
        w("")
        w("=" * 118)
        w(f"PROTOCOL {blk['protocol']} {'(aligned 5,279; gates LOAO-refit on the scored split)' if blk['protocol'] == 'A' else '(subject-disjoint OOF)'}"
          f" -- task {blk['task']}")
        if blk["protocol"] == "A":
            has_fz = any("EGRG (frozen gate)" in r["systems"] for r in blk["rows"])
            w("NOTE: EGRG / EGRG-top25 / EGRG-nested here are LOAO-refit on the scored clips, beside an UNFITTED "
              "post-hoc gate. The honest protocol-A comparison is a protocol-B gate applied frozen "
              + ("('EGRG (frozen gate)' below)." if has_fz else "(--frozen_gate <protocol-B gate.json>; not applied "
                                                                "in this run)."))
        w(f"video: {blk['video_source']}  [{blk['video_selection']}]")
        w(f"  load: {blk['video_load']}")
        w(f"EEG:   {blk['eeg_source']}  [{blk['eeg_selection']}]  score={blk['eeg_score']}")
        w(f"  load: {blk['eeg_load']}")
        pl = blk["pairing"]
        w(f"  paired: {pl['n_scored']} clips; video clips dropped (no EEG) {pl['video_clips_dropped_no_eeg']}, "
          f"EEG clips dropped (no video) {pl['eeg_clips_dropped_no_video']}")
        for i, r in enumerate(blk["rows"]):
            head = i == 0
            if head:
                w(f"\n  HEADLINE  video {r['video']} ({r['video_kind']}) x EEG {r['eeg']} ({r['eeg_kind']}); "
                  f"{r['n_clips']} clips, {r['n_animals']} animals, class counts {r['class_counts']}")
                if blk["n_video_members"] > 1:
                    w(f"  NOTE: the headline video is an ensemble of {blk['n_video_members']} networks, which "
                      f"detects better than any deployable single network and so UNDERSTATES the gate; the "
                      f"deployable unit is one video network (rows and pair distribution below).")
                if "UPPER BOUND" in str(blk["video_selection"]) or "UPPER BOUND" in str(blk["eeg_selection"]):
                    w("  NOTE: at least one modality is a best-epoch checkpoint selected on these clips: "
                      "every number in this block is an UPPER BOUND.")
                g = r["gates"]["EGRG"]
                w(f"  EGRG gate [a, b, c] LOAO mean {g['loao_coef_mean']} sd {g['loao_coef_sd']}; fit-all {g['fit_all']}")
                gn = r["gates"].get("EGRG-nested")
                if gn:
                    w(f"  EGRG-nested: outer folds picked top25 {gn['outer_picks_top25']}/"
                      f"{gn['outer_picks_top25'] + gn['outer_picks_pooled']}; on all animals it picks "
                      f"{gn['eeg_input']} (LOAO macro-F1 {gn['all_animals_loao_macro_f1']})")
                gf = r["gates"].get("EGRG (frozen gate)")
                if gf:
                    w(f"  EGRG (frozen gate): {gf['system']} coef {gf['coef']} on the {gf['eeg_input']} EEG score, "
                      f"from {gf['source']}")
                    w(f"    fitted on EEG {gf['fitted_on']}; applied to EEG {gf['applied_to']}"
                      + ("" if gf["compatible"] else "  *** MISMATCHED EEG INPUT (--frozen_allow_mismatch): "
                                                     "the coefficients were fitted on a different score ***"))
                    w(f"    applied to {gf['applied_to_row']} (a gate is video-model specific; it was fitted "
                      f"for the video named in its source)")
                w(f"  {'system':<20}{'macro-F1 [95% CI]':<28}{'vs video':<27}{'vs post-hoc':<27}"
                  f"{'MCC':>7}{'detF1':>8}  severe      per-class hit/n")
                for n, sr in r["systems"].items():
                    ci = sr.get("ci", {}).get("macro_f1", [np.nan, np.nan])
                    dv = fmt_ci(sr["d_vs_video"]["macro_f1"]) if "d_vs_video" in sr else ""
                    dp = fmt_ci(sr["d_vs_posthoc"]["macro_f1"]) if "d_vs_posthoc" in sr else ""
                    pc = " ".join(f"{v['hit']}/{v['n']}" for v in sr["per_class"].values())
                    w(f"  {n:<20}{sr['macro_f1']:.4f} [{ci[0]:.4f},{ci[1]:.4f}]    {dv:<27}{dp:<27}"
                      f"{sr['mcc']:>7.4f}{sr['det']['macro_f1']:>8.4f}  {sr['severe']['hit']:>4}/{sr['severe']['n']:<5} {pc}")
                for n in ("EGRG", "post-hoc gate"):
                    if n in r["systems"] and "d_vs_video" in r["systems"][n]:
                        d = r["systems"][n]["d_vs_video"]
                        w(f"  {n} vs video: MCC {fmt_ci(d['mcc'])}  severe recall {fmt_ci(d['sev_recall'])}  "
                          f"detection F1 {fmt_ci(d['det_f1'])}")
                bad = [f"{x['video']} x {x['eeg']}" for x in blk["rows"] if not x["grading_untouched_all"]]
                w(f"  within-session severe-vs-mild AUROC (video's grading, every gate's by construction): "
                  f"{r['within_session_severe_auroc']:.4f}; grading untouched (every gate-called seizure clip "
                  f"gets video's within-seizure argmax) in {len(blk['rows']) - len(bad)}/{len(blk['rows'])} rows"
                  + (f"  *** VIOLATED in: {bad} ***" if bad else ""))
                if len(blk["rows"]) > 1:
                    w(f"\n  {'video':<28}{'EEG':<26}{'video F1':>9}{'EGRG':>8}  {'EGRG - video':<27}"
                      f"{'EGRG - post-hoc':<27}{'sev hit v->E':>13}")
            else:
                ev, ep = r["systems"]["EGRG"]["d_vs_video"]["macro_f1"], r["systems"]["EGRG"]["d_vs_posthoc"]["macro_f1"]
                w(f"  {r['video'][:27]:<28}{r['eeg'][:25]:<26}{r['systems']['video']['macro_f1']:>9.4f}"
                  f"{r['systems']['EGRG']['macro_f1']:>8.4f}  {fmt_ci(ev):<27}{fmt_ci(ep):<27}"
                  f"{r['systems']['video']['severe']['hit']:>6}->{r['systems']['EGRG']['severe']['hit']:<5}")
        sp = blk.get("single_pairs")
        if sp:
            w(f"\n  single video network x single EEG network ({sp['n_pairs']} pairs; point estimates):")
            for k in ("posthoc_minus_video", "egrg_minus_video", "egrg_minus_posthoc", "top25_minus_egrg",
                      "severe_hits_egrg_minus_video"):
                d = sp[k]
                w(f"    {k:<30} mean {d['mean']:+.4f} sd {d['sd']:.4f} [p5 {d['p5']:+.4f}, p95 {d['p95']:+.4f}] "
                  f"min {d['min']:+.4f} share>0 {d['share_gt0']:.2f}")
    if results.get("regression"):
        w("")
        w("=" * 118)
        w("REGRESSION vs the joint-gated prototype (j10_argmax_rule.py):")
        for c in results["regression"]["checks"]:
            w(f"  [{'PASS' if c['ok'] else 'FAIL'}] {c['check']}: {c['detail']}")
        w(f"  {results['regression']['n_passed']}/{results['regression']['n_checks']} passed")
    return "\n".join(L) + "\n"


# ============================================================================ main

GATE_FORMAT = 2        # 2: every entry carries eeg_input + eeg_signature (1: features only)


def row_detector(eb, ename):
    """The detector of one scored row's EEG side: a single member's own arch, else the bundle's."""
    d = dict(eb["detector"])
    if ename in eb["member_arch"] and ename in eb["members"]:
        d["arch"] = eb["member_arch"][ename]
    return d


def select_frozen(path, system, video_sel, eeg_sel, tasks):
    """One gate.json entry per task, chosen on (system, video, eeg); refuses ambiguity."""
    G = json.load(open(path))["gates"]
    out = []
    for t in tasks:
        c = [g for g in G if g["task"] == t and g["system"] == system]
        for sel, fld in ((video_sel, "video"), (eeg_sel, "eeg")):
            if sel is not None:
                exact = [g for g in c if g[fld] == sel]
                c = exact if exact else [g for g in c if sel in g[fld]]
        if len(c) != 1:
            avail = sorted({f"{g['video']} x {g['eeg']}" for g in G if g["task"] == t and g["system"] == system})
            raise SystemExit(f"--frozen_gate: {len(c)} {system} entries for {t} match --frozen_entry={video_sel!r} "
                             f"--frozen_eeg_entry={eeg_sel!r}; need exactly one. Entries (video x eeg): {avail}")
        g = c[0]
        feat_inp = "top25" if any("top25" in f for f in g["features"]) else "pooled"
        inp = g.get("eeg_input", feat_inp)                  # format-1 files: from the feature names
        if inp != feat_inp or (system in GATE_INPUT and inp != GATE_INPUT[system]) or len(g["coef"]) != 3:
            raise SystemExit(f"--frozen_gate: inconsistent {system} entry for {t}: eeg_input={inp!r}, "
                             f"features={g['features']}, coef={g['coef']}")
        out.append(dict(task=t, system=system, coef=[float(x) for x in g["coef"]], features=g["features"],
                        eeg_input=inp, signature=g.get("eeg_signature"),
                        source=f"{path}: {g['video']} x {g['eeg']} ({g['fit_set']})"))
    return out


def bind_frozen(frozen, eb, allow_mismatch):
    """A frozen gate's coefficient b belongs to one EEG score: refuse to apply it to another."""
    for fz in frozen:
        cur = gate_input_signature(eb["detector"], fz["eeg_input"])
        fz["applied_to"], fz["compatible"] = cur, fz["signature"] == cur
        if fz["compatible"]:
            continue
        why = ("the entry records no eeg_signature (gate.json format 1): its EEG detector is unknown"
               if fz["signature"] is None else f"fitted on {fz['signature']}, applied to {cur}")
        if not allow_mismatch:
            raise SystemExit(f"--frozen_gate {fz['system']} {fz['task']}: EEG input mismatch: {why}. The gate's "
                             f"coefficients are specific to that EEG score: refit, or pass "
                             f"--frozen_allow_mismatch to apply it anyway (labelled MISMATCHED)")
        print(f"[frozen] WARNING {fz['system']} {fz['task']}: MISMATCHED EEG input ({why}); applied because "
              f"--frozen_allow_mismatch", flush=True)


def check_out(p):
    rp = os.path.realpath(p)
    if not rp.startswith(OUT_PREFIX):
        raise SystemExit(f"--out must live under {OUT_PREFIX}*/ (got {rp})")
    return rp


def regression_checks(blocks):
    out = []

    def ck(name, ok, detail):
        out.append(dict(check=name, ok=bool(ok), detail=detail))

    for blk in blocks:
        exp = REG[(blk["protocol"], blk["task"])]
        r = blk["rows"][0]["systems"]
        tag = f"{blk['protocol']} {blk['task']}"
        for n, e, h in zip(("video", "post-hoc gate", "EGRG-top25"), exp["f1"], exp["hits"]):
            got = r[n]["macro_f1"]
            ck(f"{tag} {n} macro-F1", abs(got - e) <= 5e-5, f"{got:.4f} vs prototype {e:.4f}")
            hits = [v["hit"] for v in r[n]["per_class"].values()]
            ck(f"{tag} {n} per-class hits", hits == list(h), f"{hits} vs {list(h)}")
        for lab, sysn, key in (("EGRG-top25 vs video", "EGRG-top25", "d_vs_video"),
                               ("EGRG-top25 vs post-hoc", "EGRG-top25", "d_vs_posthoc"),
                               ("post-hoc vs video", "post-hoc gate", "d_vs_video")):
            e = exp[{"d_vs_video": "d_vs_video", "d_vs_posthoc": "d_vs_posthoc"}[key]] if sysn == "EGRG-top25" \
                else exp["posthoc_vs_video"]
            s = r[sysn][key]["macro_f1"]
            got = (s["mean"], s["lo"], s["hi"])
            ck(f"{tag} {lab} paired CI", all(abs(g - x) <= 6e-5 for g, x in zip(got, e)),
               f"{fmt_ci(s)} vs prototype {e[0]:+.4f} [{e[1]:+.4f},{e[2]:+.4f}]")
        pl = blk["pairing"]
        if blk["protocol"] == "A":
            ck(f"{tag} video clips without EEG", pl["video_clips_dropped_no_eeg"] == REG_LOADS["A_video_clips_without_eeg"],
               str(pl["video_clips_dropped_no_eeg"]))
            for side, lg in (("video", blk["video_log"]), ("eeg", blk["eeg_log"])):
                ex = REG_LOADS[f"A_{side}"]
                got = {k: (lg[k] if k == "n_loaded" else len(lg[k])) for k in ex}
                ck(f"{tag} {side} load counts", got == ex, f"{got} vs {ex}")
        else:
            ck(f"{tag} video clips without EEG", pl["video_clips_dropped_no_eeg"] == REG_LOADS["B_video_clips_without_eeg"],
               str(pl["video_clips_dropped_no_eeg"]))
            ck(f"{tag} video double-softmax repaired folds", len(blk["video_log"]["double_softmax_repaired"]) == 0,
               str(len(blk["video_log"]["double_softmax_repaired"])))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--protocol", choices=["A", "B"], default=None)
    ap.add_argument("--video", nargs="+", default=["stored"],
                    help="'stored' or train_grader.py run dirs (protocol B: templates with {f})")
    ap.add_argument("--video_epoch", type=int, default=12)
    ap.add_argument("--eeg", nargs="+", default=["stored"],
                    help="'stored' or train_eeg_det.py run dirs (protocol B: templates with {f})")
    ap.add_argument("--video_members", nargs="+", default=None,
                    help="keep only video members matching these fnmatch patterns ({task} is "
                         "substituted), e.g. 'x3d_{task}_s*'")
    ap.add_argument("--eeg_epoch", type=int, default=30)
    ap.add_argument("--eeg_members", nargs="+", default=None,
                    help="keep only EEG members matching these fnmatch patterns, e.g. 'tcn_g3_s*'")
    ap.add_argument("--eeg_score", choices=["logmean", "mean"], default="logmean",
                    help="PRIMARY pooled EEG score of train_eeg_det dumps (stored runs use their own)")
    ap.add_argument("--tasks", nargs="+", choices=["g3", "g5"], default=["g3", "g5"])
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--boot_seed", type=int, default=29, help="29 = the prototype's j10 seed")
    ap.add_argument("--rows", choices=["all", "members", "headline"], default="all",
                    help="headline: ensemble x ensemble only; members: + per-member rows; all: + "
                         "every single x single pair (point estimates)")
    ap.add_argument("--frozen_gate", default=None,
                    help="protocol A: apply a protocol-B gate.json frozen (one entry per task; pick it "
                         "with --frozen_entry / --frozen_eeg_entry if several)")
    ap.add_argument("--frozen_entry", default=None,
                    help="the gate.json entry's video name (exact match preferred, else substring)")
    ap.add_argument("--frozen_eeg_entry", default=None,
                    help="the gate.json entry's EEG name (exact match preferred, else substring), e.g. "
                         "eeg_ens2 for the EEG-ensemble gate of a two-seed protocol-B run")
    ap.add_argument("--frozen_features", choices=["EGRG", "EGRG-top25", "EGRG-nested"], default="EGRG",
                    help="which gate to freeze; its EEG input (pooled / top25) is read from the entry")
    ap.add_argument("--frozen_allow_mismatch", action="store_true",
                    help="apply a frozen gate even if it was fitted on a different EEG score (detector "
                         "arch / head / pooling) than this run's; the row is labelled MISMATCHED")
    ap.add_argument("--nested", choices=["rows", "headline", "none"], default="rows",
                    help="compute EGRG-nested (nested-LOAO choice of pooled vs top25; ~10 s per row at "
                         "24k clips) for every scored row, the headline only, or not at all")
    ap.add_argument("--allow_partial", action="store_true", help="smoke tests on --limit_val dumps only")
    ap.add_argument("--regression", action="store_true",
                    help="reproduce the prototype numbers (stored A + stored B) and assert them")
    ap.add_argument("--out", required=True, help=f"output dir under {OUT_PREFIX}*/")
    a = ap.parse_args()
    out = check_out(a.out)
    if a.regression:
        runs = [("A", ["stored"], ["stored"]), ("B", ["stored"], ["stored"])]
        a.rows = "headline"
    else:
        if a.protocol is None:
            raise SystemExit("--protocol A or B is required (or --regression)")
        runs = [(a.protocol, a.video, a.eeg)]
    frozen = None
    if a.frozen_gate:
        if a.protocol != "A":
            raise SystemExit("--frozen_gate applies a protocol-B gate to protocol A")
        frozen = select_frozen(a.frozen_gate, a.frozen_features, a.frozen_entry, a.frozen_eeg_entry, a.tasks)
    t0 = time.time()
    folds = None
    fold_rows = None
    blocks, gates_out = [], []
    for protocol, vspec, espec in runs:
        if protocol == "B":
            print("[folds] EEG vs video subject-disjoint folds (split_subjects seed 49):", flush=True)
            ok, fold_rows = CF.fold_table(verbose=True)
            if not ok:
                raise SystemExit("EEG and video subject folds differ: OOF scoring would leak")
            folds = CF.expected_folds()
        if espec == ["stored"]:
            eb = load_eeg_stored_A() if protocol == "A" else load_eeg_stored_B(folds)
        else:
            eb = load_eeg_dirs(espec, a.eeg_epoch, protocol, a.eeg_score, a.allow_partial, folds)
        eb = filter_members(eb, a.eeg_members, None, "eeg")
        eb["detector"] = eeg_detector(eb)
        print(f"[load] EEG {protocol}: {eb['log']['source']}: {logline(eb['log'])}", flush=True)
        if frozen:
            bind_frozen(frozen, eb, a.frozen_allow_mismatch)
        for task in a.tasks:
            if vspec == ["stored"]:
                vb = load_video_stored_A(task) if protocol == "A" else load_video_stored_B(task)
            else:
                vb = load_video_dirs(vspec, a.video_epoch, task, protocol, a.allow_partial, folds)
            vb = filter_members(vb, a.video_members, task, "video")
            print(f"[load] video {protocol} {task}: {vb['log']['source']}: {logline(vb['log'])}", flush=True)
            res = analyse(protocol, task, vb, eb, a, frozen, pairs=True)
            pl = res["pairing"]
            print(f"[pair] {protocol} {task}: {pl['n_scored']} clips scored; video clips dropped (no EEG) "
                  f"{pl['video_clips_dropped_no_eeg']}, EEG clips dropped (no video) "
                  f"{pl['eeg_clips_dropped_no_video']} ({res['secs']}s, {len(res['rows'])} rows)", flush=True)
            h = res["rows"][0]["systems"]
            print("       " + "  ".join(f"{n} {h[n]['macro_f1']:.4f}" for n in h), flush=True)
            res.update(video_source=vb["log"]["source"], eeg_source=eb["log"]["source"],
                       video_load=logline(vb["log"]), eeg_load=logline(eb["log"]),
                       video_log=vb["log"], eeg_log=eb["log"])
            blocks.append(res)
            if protocol == "B":
                for r in res["rows"]:
                    for sname in ("EGRG", "EGRG-top25", "EGRG-nested"):
                        g = r["gates"].get(sname)
                        if g is None:
                            continue
                        gates_out.append(dict(task=task, system=sname, video=r["video"], eeg=r["eeg"],
                                              eeg_input=g["eeg_input"], features=g["features"], coef=g["fit_all"],
                                              eeg_signature=gate_input_signature(row_detector(eb, r["eeg"]),
                                                                                 g["eeg_input"]),
                                              loao_coef_mean=g.get("loao_coef_mean"), rule=g.get("rule"),
                                              n=r["n_clips"], n_animals=r["n_animals"],
                                              fit_set="all protocol-B OOF clips (deployment gate)",
                                              video_source=vb["log"]["source"], eeg_source=eb["log"]["source"],
                                              video_selection=vb["selection"], eeg_selection=eb["selection"]))
    results = dict(created=time.strftime("%F %T"), mode="regression" if a.regression else f"protocol {a.protocol}",
                   argv=sys.argv, reps=a.reps, boot_seed=a.boot_seed, eeg_root=EEG_ROOT,
                   dhlib=os.path.join(DH_DIR, "dhlib.py"), folds=fold_rows, blocks=blocks,
                   secs=round(time.time() - t0, 1))
    if a.regression:
        cks = regression_checks(blocks)
        results["regression"] = dict(checks=cks, n_checks=len(cks), n_passed=sum(c["ok"] for c in cks))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(results, f, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else
                  (o.item() if isinstance(o, np.generic) else str(o)))
    txt = render(results)
    with open(os.path.join(out, "results.txt"), "w") as f:
        f.write(txt)
    if gates_out:
        with open(os.path.join(out, "gate.json"), "w") as f:
            json.dump(dict(created=results["created"], format=GATE_FORMAT,
                           note="3-parameter EGRG gates fitted on ALL protocol-B OOF clips: the deployment "
                                "gates. q = sigmoid(coef . [logit P_V(sz), logit P_E(sz)<eeg_input>, 1]); "
                                "eeg_input says which EEG score (pooled or top25) b multiplies and "
                                "eeg_signature which detector / pooling it was fitted on. Refit for every "
                                "video model.", gates=gates_out), f, indent=1)
    print("\n" + txt)
    print(f"wrote {out}/results.json, results.txt" + (", gate.json" if gates_out else "") +
          f" ({time.time() - t0:.0f}s)")
    if a.regression:
        rg = results["regression"]
        print(f"REGRESSION: {rg['n_passed']}/{rg['n_checks']} checks passed")
        return 0 if rg["n_passed"] == rg["n_checks"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
