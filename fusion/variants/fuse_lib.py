import os
"""Fusion of the video and EEG ensembles on the 2,830-clip intersection.

Conventions, all deliberate:
  * A "score matrix" S is (N,K). argmax(1) is the prediction; column k is the
    one-vs-rest score for AUROC. Row-normalisation is NOT applied, because it
    reorders clips within a column and would corrupt AUROC for rank averaging.
  * Macro AUROC = mean over present classes of the binary OvR AUROC. This is what
    sklearn's average='macro', multi_class='ovr' computes, but done by hand so that
    non-probability scores (ranks) are admissible.
  * Unit of inference is the ANIMAL. Bootstrap resamples animals, never clips.
"""
import re
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = '.'
NAMES = {'bin': ['non-seizure', 'seizure'],
         'g3':  ['non-seizure', 'mild (S2-3)', 'severe (S4-5)'],
         'g5':  ['non-seizure', 'Stage2', 'Stage3', 'Stage4', 'Stage5']}
SEVERE = {'bin': [1], 'g3': [2], 'g5': [3, 4]}     # "severe" class indices per task


def load(task):
    z = np.load('%s/aligned_%s.npz' % (HERE, task), allow_pickle=True)
    clips = z['clips'].astype(str)
    subj = np.array([re.search(r'Data_(RN\d+)', c).group(1) for c in clips])
    day = np.array([c.split('/')[2] for c in clips])
    sess = np.array(['%s|%s' % (a, d) for a, d in zip(subj, day)])
    return dict(clips=clips, y=z['y'].astype(int), V=z['V'].astype(np.float64),
                E=z['E'].astype(np.float64), vnames=z['vnames'].astype(str),
                enames=z['enames'].astype(str), subj=subj, sess=sess,
                K=z['V'].shape[2], task=task)


# ---------------------------------------------------------------- metrics
def confusion(y, pred, K):
    cm = np.zeros((K, K), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    return cm


def prf(y, pred, K):
    cm = confusion(y, pred, K)
    tp = np.diag(cm).astype(float)
    sup = cm.sum(1).astype(float)
    pp = cm.sum(0).astype(float)
    rec = np.divide(tp, sup, out=np.zeros(K), where=sup > 0)
    pre = np.divide(tp, pp, out=np.zeros(K), where=pp > 0)
    f1 = np.divide(2 * pre * rec, pre + rec, out=np.zeros(K), where=(pre + rec) > 0)
    present = sup > 0
    return (float(pre[present].mean()), float(rec[present].mean()),
            float(f1[present].mean()), rec, sup.astype(int), cm)


def macro_auc(y, S, K):
    aucs = []
    for k in range(K):
        pos = (y == k)
        if pos.sum() == 0 or pos.sum() == y.size:
            continue
        aucs.append(roc_auc_score(pos.astype(int), S[:, k]))
    return float(np.mean(aucs)) if aucs else float('nan')


def metrics(y, S, K):
    pred = S.argmax(1)
    P, R, F1, rec, sup, cm = prf(y, pred, K)
    auc = roc_auc_score(y, S[:, 1]) if K == 2 else macro_auc(y, S, K)
    return dict(P=P, R=R, F1=F1, AUC=float(auc), MCC=float(matthews_corrcoef(y, pred)),
                rec=rec.tolist(), sup=sup.tolist(), cm=cm.tolist())


def fast_f1_rec(y, pred, K):
    """macro-F1 and the per-class recall vector, no sklearn (for the bootstrap)."""
    cm = np.zeros((K, K), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    tp = np.diag(cm).astype(float)
    sup = cm.sum(1).astype(float)
    pp = cm.sum(0).astype(float)
    rec = np.divide(tp, sup, out=np.zeros(K), where=sup > 0)
    pre = np.divide(tp, pp, out=np.zeros(K), where=pp > 0)
    f1 = np.divide(2 * pre * rec, pre + rec, out=np.zeros(K), where=(pre + rec) > 0)
    present = sup > 0
    return float(f1[present].mean()), rec, sup


# ---------------------------------------------------------------- fusion rules
def norm(S):
    return S / S.sum(1, keepdims=True)


def geometric(A, B, eps=1e-12):
    G = np.sqrt(np.clip(A, eps, None) * np.clip(B, eps, None))
    return norm(G)


def colrank(S):
    """Rank each class column across clips, scaled to [0,1]."""
    return np.column_stack([rankdata(S[:, k]) for k in range(S.shape[1])]) / S.shape[0]


def rank_avg(A, B):
    return 0.5 * (colrank(A) + colrank(B))


def maxconf(A, B):
    take_a = A.max(1) >= B.max(1)
    return np.where(take_a[:, None], A, B)


def stack_loao(X, y, groups, K, balanced=False, seed=0):
    """Leave-one-ANIMAL-out multinomial logistic regression. Returns out-of-fold scores."""
    S = np.zeros((X.shape[0], K))
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        ytr = y[tr]
        cls = np.unique(ytr)
        if cls.size < 2:
            S[te, 0] = 1.0
            continue
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(max_iter=5000, C=1.0, random_state=seed,
                                class_weight='balanced' if balanced else None)
        lr.fit(sc.transform(X[tr]), ytr)
        S[np.ix_(te, cls)] = lr.predict_proba(sc.transform(X[te]))
    return S
