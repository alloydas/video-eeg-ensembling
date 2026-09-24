"""decision-headroom: shared loading, metrics, bias fitting and animal-clustered bootstrap.

numpy + scipy only (no sklearn in the base interpreter). EEG_ROOT is read-only.

Vendored into video-eeg-ensembling/grader/ from EEG_ROOT/output/ttg_ref/decision-headroom/dhlib.py
(2026-09-23 10:55 version); the library code below is unchanged, only this note and the
`python grader/dhlib.py` entry point at the end were added. Imported by grader/stage1_eval.py,
stage1_ensemble.py, stage1b_eval.py, eeg/joint_gate.py, eeg/check_folds.py and
eeg/verify_joint_gate.py (via joint_gate); DHLIB_DIR overrides where they look for it. Every path
it reads is absolute under EEG_ROOT (env var, default /work/mech-ai-scratch/alloy/EEG), so it does
not depend on the cwd.
"""
from __future__ import annotations

import json
import os
import random
import re
from collections import Counter

import numpy as np
from scipy.stats import rankdata

EEG_ROOT = os.environ.get("EEG_ROOT", "/work/mech-ai-scratch/alloy/EEG")
HERE = os.path.dirname(os.path.abspath(__file__))

SEV = {"g3": [2], "g5": [3, 4], "s4": [2, 3]}
MILD = {"g3": [1], "g5": [1, 2], "s4": [0, 1]}
K = {"g3": 3, "g5": 5, "s4": 4}
NAMES = {"g3": ["non-seizure", "mild(S2-3)", "severe(S4-5)"],
         "g5": ["non-seizure", "Stage2", "Stage3", "Stage4", "Stage5"],
         "s4": ["Stage2", "Stage3", "Stage4", "Stage5"]}
STAGE = {"Stage_2": 1, "Stage_3": 2, "Stage_4": 3, "Stage_5": 4}
GROUP3 = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}


# ------------------------------------------------------------------ keys / paths

def key_of(p: str) -> str:
    p = str(p)
    return p[: -len("/video.mp4")] if p.endswith("/video.mp4") else p


def animal_session(keys):
    an, se = [], []
    for k in keys:
        a = re.search(r"Data_(RN\d+)_", k).group(1)
        parts = [c for c in k.split("/") if c]
        an.append(a)
        se.append(f"{a}/{parts[-2]}")
    return np.array(an), np.array(se)


_items = None


def items_from_cache_index():
    """discover() item list in original microway order (5-class labels)."""
    global _items
    if _items is None:
        meta = json.load(open(os.path.join(EEG_ROOT, "cache_frames/f32s224/index.json")))
        out = []
        for mp4 in meta["paths"]:
            d = os.path.dirname(mp4)
            b, sess = os.path.basename(d), os.path.basename(os.path.dirname(d))
            subj = re.search(r"Data_(RN\d+)_", mp4).group(1)
            if b.startswith("seizure_"):
                m = re.search(r"Stage_[0-9]+", b)
                if not m or m.group() not in STAGE:
                    continue
                y = STAGE[m.group()]
            else:
                y = 0
            out.append((mp4, y, f"{subj}/{sess}", subj))
        _items = out
    return _items


def split_sessions(items, seed=49, val_frac=0.2, NC=5):
    sessions = sorted({i[2] for i in items})
    for s in range(seed, seed + 500):
        rng = random.Random(s); ss = sessions[:]; rng.shuffle(ss)
        val = set(ss[:max(1, round(len(ss) * val_frac))])
        tr = [i for i in items if i[2] not in val]
        va = [i for i in items if i[2] in val]
        ctr, cva = Counter(i[1] for i in tr), Counter(i[1] for i in va)
        if all(ctr[c] > 0 for c in range(NC)) and all(cva[c] > 0 for c in range(NC)):
            return tr, va, val, s
    raise SystemExit("no split")


def split_subjects(items, seed, fold=0, n_folds=5):
    subs = sorted({i[3] for i in items})
    rng = random.Random(seed); rng.shuffle(subs)
    groups = [subs[k::n_folds] for k in range(n_folds)]
    val = set(groups[fold % n_folds])
    tr = [i for i in items if i[3] not in val]
    va = [i for i in items if i[3] in val]
    return tr, va, val, seed


# ------------------------------------------------------------------ x3d repair

def squash_bounds(k):
    e = np.e
    return 1.0 / (k - 1 + e), e / (k - 1 + e)


def is_double_softmax(p, tol=2e-3):
    lo, hi = squash_bounds(p.shape[1])
    return abs(p.min() - lo) < tol and abs(p.max() - hi) < tol


def unsquash(p):
    k = p.shape[1]
    lp = np.log(np.clip(p, 1e-12, None))
    q = lp + (1.0 - lp.sum(1, keepdims=True)) / k
    q = np.clip(q, 0.0, None)
    return q / q.sum(1, keepdims=True)


# ------------------------------------------------------------------ run loading

def load_members(dirs, ref_keys, ref_y_raw_order, ref_y_sorted):
    """Load runs aligned on clip key to ref_keys order.

    ref_keys: canonical key order (reference run's raw order, /video.mp4 stripped).
    ref_y_raw_order: the reference run's raw y (for admitting path-less runs).
    Returns P (M,N,K), names, log dict.
    """
    pos = {k: i for i, k in enumerate(ref_keys)}
    P, names, log = [], [], {"dropped": [], "double_softmax_repaired": [],
                             "pathless_admitted": [], "reordered": []}
    for d in dirs:
        f = os.path.join(EEG_ROOT, d, "val_preds.npz")
        z = np.load(f, allow_pickle=True)
        probs = z["probs"].astype(np.float64)
        y = z["y"].astype(int)
        if "path" in z.files:
            keys = np.array([key_of(p) for p in z["path"]])
            if not np.array_equal(keys, ref_keys):
                log["reordered"].append(d)
        else:
            idx = z["idx"]
            if np.array_equal(y, ref_y_raw_order) and np.array_equal(idx, np.arange(len(y))):
                keys = ref_keys
                log["pathless_admitted"].append(d)
            else:
                log["dropped"].append(d)
                continue
        if set(keys) != set(ref_keys) or len(keys) != len(ref_keys):
            log["dropped"].append(d)
            continue
        if is_double_softmax(probs):
            probs = unsquash(probs)
            log["double_softmax_repaired"].append(d)
        order = np.array([pos[k] for k in keys])
        Pa = np.empty_like(probs); Pa[order] = probs
        ya = np.empty_like(y); ya[order] = y
        if not np.array_equal(ya, ref_y_sorted):
            log["dropped"].append(d + " (label mismatch)")
            continue
        P.append(Pa); names.append(d)
    return np.array(P), names, log


def reference_val(task):
    """Canonical val order = reconstructed split_sessions(seed=49) order, verified
    against a path-bearing run."""
    items = items_from_cache_index()
    _, va, _, _ = split_sessions(items, 49)
    keys = np.array([key_of(i[0]) for i in va])
    y5 = np.array([i[1] for i in va])
    y = y5 if task == "g5" else np.array([GROUP3[v] for v in y5])
    z = np.load(os.path.join(EEG_ROOT, f"output/v3_vidseeds/mvit_{task}_s1/val_preds.npz"),
                allow_pickle=True)
    assert np.array_equal(np.array([key_of(p) for p in z["path"]]), keys)
    assert np.array_equal(z["y"], y)
    return keys, y, y5


# ------------------------------------------------------------------ metrics

def confusion(y, pred, k):
    return np.bincount(y * k + pred, minlength=k * k).reshape(k, k)


def cm_metrics(C):
    """Metrics from a (..., k, k) confusion array (rows true, cols pred)."""
    C = np.asarray(C, dtype=np.float64)
    tp = np.diagonal(C, axis1=-2, axis2=-1)
    rows = C.sum(-1)
    cols = C.sum(-2)
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = np.where(rows > 0, tp / rows, 0.0)
        prec = np.where(cols > 0, tp / cols, 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    present = rows > 0
    # sklearn macro-F1 averages over labels present in y OR pred; here all labels
    # always present in full sets; in resamples classes absent from y and pred give 0
    labels = present | (cols > 0)
    macro_f1 = (f1 * labels).sum(-1) / labels.sum(-1)
    bal = (rec * present).sum(-1) / present.sum(-1)
    s = C.sum((-2, -1))
    c = tp.sum(-1)
    num = c * s - (cols * rows).sum(-1)
    den = np.sqrt((s ** 2 - (cols ** 2).sum(-1)) * (s ** 2 - (rows ** 2).sum(-1)))
    with np.errstate(invalid="ignore", divide="ignore"):
        mcc = np.where(den > 0, num / den, 0.0)
    return dict(macro_f1=macro_f1, bal_acc=bal, mcc=mcc, recall=rec, precision=prec,
                f1=f1, support=rows)


def sev_group(C, task):
    """Severe-group recall / precision / F1 from a confusion matrix (pred in SEV)."""
    sv = SEV[task]
    C = np.asarray(C, dtype=np.float64)
    tp = C[..., sv, :][..., :, sv].sum((-2, -1))
    t = C[..., sv, :].sum((-2, -1))
    p = C[..., :, sv].sum((-2, -1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(t > 0, tp / t, 0.0)
        pr = np.where(p > 0, tp / p, 0.0)
        f = np.where(r + pr > 0, 2 * r * pr / (r + pr), 0.0)
    return r, pr, f


def auroc(pos_scores, neg_scores):
    n1, n0 = len(pos_scores), len(neg_scores)
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(np.concatenate([pos_scores, neg_scores]))
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def pair_counts(pos, neg):
    """(concordant + 0.5 ties, total pairs)."""
    if len(pos) == 0 or len(neg) == 0:
        return 0.0, 0.0
    r = rankdata(np.concatenate([pos, neg]))
    u = r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u), float(len(pos) * len(neg))


def within_session_auc(score, is_sev, is_mild, sess):
    """Pooled within-session AUROC (pairs only inside sessions with both classes),
    plus the unweighted mean of per-session AUROCs, plus per-session pair counts."""
    per = {}
    for s in np.unique(sess):
        m = sess == s
        a, b = score[m & is_sev], score[m & is_mild]
        if len(a) and len(b):
            per[s] = pair_counts(a, b)
    U = sum(v[0] for v in per.values()); T = sum(v[1] for v in per.values())
    mean_per = float(np.mean([v[0] / v[1] for v in per.values()])) if per else np.nan
    return (U / T if T else np.nan), mean_per, per


# ------------------------------------------------------------------ bias fitting

def _coord_sweep(L, y, b, k, K_, lim, objective="macro_f1"):
    """Exact 1-D optimisation of macro-F1 over b[k] (others fixed).

    Returns (best_value, best_obj). Piecewise-constant objective; breakpoints at
    t_i = max_{j!=k}(L_ij + b_j) - L_ik. Choose the maximising interval closest
    to the current value, set b[k] to its midpoint (clipped to [-lim, lim])."""
    S = L + b
    Sk = S.copy(); Sk[:, k] = -np.inf
    alt = Sk.argmax(1)
    M = Sk[np.arange(len(y)), alt]
    t = M - L[:, k]
    order = np.argsort(t, kind="stable")
    ts = t[order]
    # base confusion: all predicted alt (v -> -inf)
    C0 = confusion(y, alt, K_).astype(np.int64)
    # moving clip i: (y_i, alt_i) -1 ; (y_i, k) +1
    delta = np.zeros((len(y), K_ * K_), dtype=np.int64)
    yo, ao = y[order], alt[order]
    np.add.at(delta, (np.arange(len(y)), yo * K_ + ao), -1)
    np.add.at(delta, (np.arange(len(y)), yo * K_ + k), 1)
    cum = np.concatenate([np.zeros((1, K_ * K_), np.int64), np.cumsum(delta, 0)])
    Cs = (C0.reshape(1, -1) + cum).reshape(-1, K_, K_)
    obj = cm_metrics(Cs)[objective]
    # position j means first j clips (smallest t) switched: v in (ts[j-1], ts[j])
    lo = np.concatenate([[-np.inf], ts])
    hi = np.concatenate([ts, [np.inf]])
    feasible = (hi > -lim) & (lo < lim) & (hi > lo)
    obj = np.where(feasible, obj, -1)
    best = obj.max()
    cand = np.flatnonzero(obj >= best - 1e-12)
    mids = []
    for j in cand:
        a, c = max(lo[j], -lim), min(hi[j], lim)
        mids.append(0.5 * (a + c))
    mids = np.array(mids)
    cur = b[k]
    pick = mids[np.argmin(np.abs(mids - cur))]
    return pick, best


def fit_bias(P, y, K_, lim=6.0, passes=12, restarts=0, seed=0, objective="macro_f1"):
    """Per-class additive logit bias (b[0]=0) maximising macro-F1 by exact coordinate
    ascent. Returns (b, in-sample macro-F1)."""
    L = np.log(np.clip(P, 1e-9, None))
    rng = np.random.default_rng(seed)
    starts = [np.zeros(K_)] + [np.concatenate([[0], rng.uniform(-1.5, 1.5, K_ - 1)])
                              for _ in range(restarts)]
    best_b, best_o = None, -1
    for b in starts:
        b = b.copy()
        o_prev = -1
        for _ in range(passes):
            for k in range(1, K_):
                v, o = _coord_sweep(L, y, b, k, K_, lim, objective)
                b[k] = v
            if o <= o_prev + 1e-12:
                break
            o_prev = o
        pred = (L + b).argmax(1)
        o = cm_metrics(confusion(y, pred, K_))[objective]
        if o > best_o + 1e-12 or (abs(o - best_o) <= 1e-12 and np.abs(b).sum() < np.abs(best_b).sum()):
            best_b, best_o = b.copy(), o
    return best_b, best_o


def apply_bias(P, b):
    L = np.log(np.clip(P, 1e-9, None))
    return (L + b).argmax(1)


def cv_bias(P, y, groups, K_, restarts=4, objective="macro_f1"):
    """Fit on all groups but g, predict g. Returns OOF preds and per-group biases."""
    pred = np.empty(len(y), dtype=int)
    bs = {}
    for g in np.unique(groups):
        te = groups == g
        b, _ = fit_bias(P[~te], y[~te], K_, restarts=restarts, objective=objective)
        pred[te] = apply_bias(P[te], b)
        bs[str(g)] = b.tolist()
    return pred, bs


# ------------------------------------------------------------------ bootstrap

def per_group_cm(y, pred, groups, K_, ug=None):
    ug = np.unique(groups) if ug is None else ug
    gi = {g: i for i, g in enumerate(ug)}
    gidx = np.array([gi[g] for g in groups])
    C = np.zeros((len(ug), K_, K_), dtype=np.int64)
    np.add.at(C, (gidx, y, pred), 1)
    return C, ug


def boot_weights(n_groups, reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    W = np.zeros((reps, n_groups), dtype=np.int64)
    for r in range(reps):
        pick = rng.integers(0, n_groups, n_groups)
        W[r] = np.bincount(pick, minlength=n_groups)
    return W


def boot_cm_metrics(Cg, W, task):
    """Bootstrap distributions of confusion-derived metrics. Cg (G,K,K), W (R,G)."""
    Cb = np.einsum("rg,gij->rij", W, Cg)
    m = cm_metrics(Cb)
    r, p, f = sev_group(Cb, task)
    m["sev_recall"], m["sev_precision"], m["sev_f1"] = r, p, f
    return m


def ci(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def paired_summary(da):
    da = np.asarray(da, dtype=float)
    da = da[np.isfinite(da)]
    return dict(mean=float(da.mean()), lo=float(np.percentile(da, 2.5)),
                hi=float(np.percentile(da, 97.5)), p_gt0=float((da > 0).mean()))


def full_report(y, pred, task):
    K_ = K[task]
    C = confusion(y, pred, K_)
    m = cm_metrics(C)
    r, p, f = sev_group(C, task)
    return dict(macro_f1=float(m["macro_f1"]), bal_acc=float(m["bal_acc"]),
                mcc=float(m["mcc"]),
                per_class={NAMES[task][i]: dict(recall=float(m["recall"][i]),
                                                precision=float(m["precision"][i]),
                                                f1=float(m["f1"][i]),
                                                n=int(m["support"][i]))
                           for i in range(K_)},
                sev_group=dict(recall=float(r), precision=float(p), f1=float(f)),
                confusion=C.tolist())


def boot_auc(score, pos, neg, animals, sess, W, ug):
    """Animal-bootstrap distributions of pooled AUROC (pos vs neg) and pooled
    within-session AUROC. W (R,G) animal multiplicities over ug."""
    gi = {g: i for i, g in enumerate(ug)}
    ai = np.array([gi[a] for a in animals])
    sp, sn = score[pos], score[neg]
    ap, an_ = ai[pos], ai[neg]
    # cnt[g, i]: neg clips of animal g below pos i (+0.5 ties)
    G = len(ug)
    cnt = np.zeros((G, len(sp)))
    for g in range(G):
        s = np.sort(sn[an_ == g])
        if len(s) == 0:
            continue
        lo = np.searchsorted(s, sp, "left"); hi = np.searchsorted(s, sp, "right")
        cnt[g] = lo + 0.5 * (hi - lo)
    negc = np.bincount(an_, minlength=G).astype(float)
    posc = np.bincount(ap, minlength=G).astype(float)
    U = ((W @ cnt) * W[:, ap]).sum(1)
    T = (W @ posc) * (W @ negc)
    pooled = np.where(T > 0, U / np.where(T > 0, T, 1), np.nan)
    # within-session
    ss = np.unique(sess)
    Us = np.zeros(G); Ts = np.zeros(G)
    for s in ss:
        m = sess == s
        a, b = score[m & pos], score[m & neg]
        if len(a) and len(b):
            u, t = pair_counts(a, b)
            g = gi[animals[m][0]]
            Us[g] += u; Ts[g] += t
    Tw = W @ Ts
    within = np.where(Tw > 0, (W @ Us) / np.where(Tw > 0, Tw, 1), np.nan)
    return pooled, within


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    print("dhlib is a library (imported by the grader scripts); EEG_ROOT =", EEG_ROOT)
