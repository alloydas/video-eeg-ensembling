import os
import numpy as np, re, os, json, collections
D='.'

def load(task):
    z=np.load(os.path.join(D,'aligned_%s.npz'%task),allow_pickle=True)
    clips=np.array([str(c) for c in z['clips']])
    parts=[c.split('/') for c in clips]
    subj=np.array([re.search(r'Data_(RN\d+)',c).group(1) for c in clips])
    sess=np.array(['/'.join(p[1:3]) for p in parts])
    return dict(clips=clips, y=z['y'].astype(int), V=z['V'].astype(np.float64),
                E=z['E'].astype(np.float64),
                vnames=np.array([str(x) for x in z['vnames']]),
                enames=np.array([str(x) for x in z['enames']]),
                subj=subj, sess=sess)

def prf(y,p,K):
    """macro precision/recall/f1 + per-class arrays. Classes with 0 support excluded from macro."""
    P=np.zeros(K); R=np.zeros(K); F=np.zeros(K); sup=np.zeros(K,int)
    for c in range(K):
        tp=int(((p==c)&(y==c)).sum()); fp=int(((p==c)&(y!=c)).sum()); fn=int(((p!=c)&(y==c)).sum())
        sup[c]=tp+fn
        P[c]=tp/(tp+fp) if tp+fp else 0.0
        R[c]=tp/(tp+fn) if tp+fn else 0.0
        F[c]=2*P[c]*R[c]/(P[c]+R[c]) if (P[c]+R[c]) else 0.0
    m=sup>0
    return dict(mP=float(P[m].mean()), mR=float(R[m].mean()), mF=float(F[m].mean()),
                P=P.tolist(), R=R.tolist(), F=F.tolist(), sup=sup.tolist(),
                acc=float((y==p).mean()))

def gated(gate_pos, grad_prob):
    """gate_pos: bool (N,) pass=seizure.  grad_prob: (N,K). Rejected -> 0; passed -> argmax over c>=1."""
    K=grad_prob.shape[1]
    pred=np.zeros(len(gate_pos),int)
    if K>1:
        sub=grad_prob[:,1:].argmax(1)+1
        pred[gate_pos]=sub[gate_pos]
    return pred
