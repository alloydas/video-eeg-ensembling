# Independent recomputation of video-alone and EGRG macro-F1 (3- and 5-class) from the video dumps, before
# (output/ttg_vsubj) and after (ttg_recut/vsubj_patched), for every single video x single EEG pair, with a
# logistic regression written here (scipy L-BFGS, L2 C=1, intercept unpenalised: sklearn's lbfgs objective),
# fitted leave-one-animal-out on seizure-vs-not over [logit P_V(sz), logit P_E(sz) pooled logmean].
# Decision: argmax over [1 - q, q * P_V(g | sz)]. Macro-F1 over the task's classes. Shares no code with
# grader/eeg/joint_gate.py or rawdata/rescore_recut.py. numpy + scipy only.
import os, re, sys, json, csv
import numpy as np
from scipy.optimize import minimize
EEG = "/work/mech-ai-scratch/alloy/EEG"
R = f"{EEG}/output/ttg_recut"
OUT = f"{R}/verify/indep"
G3 = np.array([0, 1, 1, 2, 2])


def key(p):
    p = str(p)
    return p[:-len("/video.mp4")] if p.endswith("/video.mp4") else p.rstrip("/")


def load_video(tmpl):
    K, Y, P3, P5 = [], [], [], []
    for f in range(5):
        z = np.load(tmpl.format(f=f), allow_pickle=True)
        K += [key(p) for p in z["path"]]; Y.append(z["y5"]); P3.append(z["probs_g3"]); P5.append(z["probs_g5"])
    P3, P5 = np.concatenate(P3).astype(np.float64), np.concatenate(P5).astype(np.float64)
    for P in (P3, P5):          # double-softmax guard (rows of softmax(prob vector) sit in a narrow band)
        k = P.shape[1]
        if P.max() <= np.e / (np.e + k - 1) + 2e-3:
            raise SystemExit(f"{tmpl}: looks double-softmaxed")
    return np.array(K), np.concatenate(Y).astype(int), P3, P5


def load_eeg(tmpl):
    K, Y, S = [], [], []
    for f in range(5):
        z = np.load(tmpl.format(f=f), allow_pickle=True)
        K += [key(p) for p in z["path"]]; Y.append(z["y5"]); S.append(z["p_logmean"])
    return np.array(K), np.concatenate(Y).astype(int), np.concatenate(S).astype(np.float64)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def fit_lr(X, y, C=1.0):
    s = 2.0 * y - 1.0
    d = X.shape[1]

    def f(th):
        w, b = th[:d], th[d]
        m = s * (X @ w + b)
        loss = np.logaddexp(0, -m).sum()
        sig = np.exp(-np.logaddexp(0, m))          # sigmoid(-m)
        g = -(s * sig)
        return 0.5 * w @ w + C * loss, np.r_[w + C * (X.T @ g), C * g.sum()]
    r = minimize(f, np.zeros(d + 1), jac=True, method="L-BFGS-B", options=dict(maxiter=10000, gtol=1e-10, ftol=1e-15))
    return r.x


def loao(X, y, an):
    q = np.empty(len(y))
    for a in np.unique(an):
        tr = an != a
        th = fit_lr(X[tr], y[tr])
        z = X[~tr] @ th[:-1] + th[-1]
        q[~tr] = 1.0 / (1.0 + np.exp(-z))
    return q


def macro_f1(y, p, k):
    C = np.zeros((k, k)); np.add.at(C, (y, p), 1)
    tp = np.diag(C); rows = C.sum(1); cols = C.sum(0)
    rec = np.divide(tp, rows, out=np.zeros(k), where=rows > 0)
    prec = np.divide(tp, cols, out=np.zeros(k), where=cols > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(k), where=(prec + rec) > 0)
    lab = (rows > 0) | (cols > 0)
    return float(f1[lab].mean()), C


def score_pair(vk, vy, P3, P5, ek, ey, es):
    common = np.array(sorted(set(vk) & set(ek)))
    iv = {k: i for i, k in enumerate(vk)}; ie = {k: i for i, k in enumerate(ek)}
    a = np.array([iv[k] for k in common]); b = np.array([ie[k] for k in common])
    y5 = vy[a]
    if not np.array_equal(y5, ey[b]):
        raise SystemExit("label mismatch")
    an = np.array([re.search(r"Data_(RN\d+)_cropped", k).group(1) for k in common])
    out = dict(n=len(common), n_video_only=len(vk) - len(common), n_eeg_only=len(ek) - len(common),
               n_animals=len(set(an)))
    for task, P in (("g3", P3[a]), ("g5", P5[a])):
        k = P.shape[1]
        y = G3[y5] if task == "g3" else y5
        pv = 1.0 - P[:, 0]
        X = np.c_[logit(pv), logit(es[b])]
        q = loao(X, (y5 > 0).astype(float), an)
        sz = P[:, 1:] / np.clip(P[:, 1:].sum(1, keepdims=True), 1e-12, None)
        pred_e = np.c_[1 - q, q[:, None] * sz].argmax(1)
        pred_v = P.argmax(1)
        fv, Cv = macro_f1(y, pred_v, k); fe, Ce = macro_f1(y, pred_e, k)
        sev = (y5 >= 3)
        sev_hit = lambda pr: int((sev & ((pr >= 3) if task == "g5" else (pr == 2))).sum())  # noqa: E731
        out[task] = dict(video_f1=fv, egrg_f1=fe, video_hits=np.diag(Cv).astype(int).tolist(),
                         egrg_hits=np.diag(Ce).astype(int).tolist(), counts=Cv.sum(1).astype(int).tolist(),
                         video_sev_hit=sev_hit(pred_v), egrg_sev_hit=sev_hit(pred_e))
    return out, common


def main():
    recipes = sys.argv[1:] or ["x3dfix", "x3dbug"]
    res = {}
    for rc in recipes:
        stored = json.load(open(f"{EEG}/output/ttg_eeg_gate/seeds/B_{rc}_grid3x3/results.json"))
        patched = json.load(open(f"{R}/grid/B_{rc}_grid3x3/results.json"))
        ref = {}
        for tag, j in (("before", stored), ("after", patched)):
            for blk in j["blocks"]:
                for p in blk["single_pairs"]["pairs"]:
                    ref[(tag, blk["task"], p["video_run"], p["eeg_run"])] = (p["systems"]["video"]["f1"], p["systems"]["EGRG"]["f1"],
                                                                         p["systems"]["video"]["sev_hit"], p["systems"]["EGRG"]["sev_hit"])
        for tag, vt in (("before", f"{EEG}/output/ttg_vsubj/{rc}_dual_s{{s}}_fold{{f}}/val_ep12.npz"),
                        ("after", f"{R}/vsubj_patched/{rc}_dual_s{{s}}_fold{{f}}/val_ep12.npz")):
            for vs in (1, 2, 3):
                V = load_video(vt.replace("{s}", str(vs)))
                for es in (1, 2, 3):
                    E = load_eeg(f"{EEG}/output/ttg_eeg/subject/tcn_bin_fold{{f}}_s{es}/val_clip_ep30.npz")
                    o, _ = score_pair(*V, *E)
                    for task in ("g3", "g5"):
                        r = ref.get((tag, task, f"{rc}_dual_s{vs}_foldF", f"tcn_bin_foldF_s{es}"))
                        o[task]["stored"] = r
                        if r:
                            o[task]["d_video"] = o[task]["video_f1"] - r[0]; o[task]["d_egrg"] = o[task]["egrg_f1"] - r[1]
                            o[task]["d_sev"] = [o[task]["video_sev_hit"] - r[2], o[task]["egrg_sev_hit"] - r[3]]
                    res[f"{rc}|{tag}|v{vs}|e{es}"] = o
                    print(rc, tag, vs, es, o["n"], {t: (round(o[t]["video_f1"], 4), round(o[t]["egrg_f1"], 4),
                                                        f"{o[t].get('d_video', float('nan')):+.1e}",
                                                        f"{o[t].get('d_egrg', float('nan')):+.1e}", o[t].get("d_sev"))
                                                    for t in ("g3", "g5")}, flush=True)
        # 9-pair means
        for tag in ("before", "after"):
            for t in ("g3", "g5"):
                vv = [res[f"{rc}|{tag}|v{v}|e{e}"][t]["video_f1"] for v in (1, 2, 3) for e in (1, 2, 3)]
                ee = [res[f"{rc}|{tag}|v{v}|e{e}"][t]["egrg_f1"] for v in (1, 2, 3) for e in (1, 2, 3)]
                gg = np.array(ee) - np.array(vv)
                res[f"{rc}|{tag}|{t}|mean9"] = dict(video=float(np.mean(vv)), egrg=float(np.mean(ee)),
                                                    gain=float(gg.mean()), gain_sd=float(gg.std(ddof=1)))
                print(rc, tag, t, "9-pair mean video %.4f EGRG %.4f gain %+.4f (sd %.4f)" % (
                    np.mean(vv), np.mean(ee), gg.mean(), gg.std(ddof=1)), flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/grid_{'_'.join(recipes)}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
