import os
import numpy as np, re, json, collections
D='./'

def load(task):
    z=np.load(D+'aligned_%s.npz'%task,allow_pickle=True)
    clips=np.array([str(c) for c in z['clips']])
    subj=np.array([re.search(r'Data_(RN\d+)',c).group(1) for c in clips])
    sess=np.array(['/'.join(c.split('/')[:3]) for c in clips])
    return dict(clips=clips,y=z['y'].astype(int),V=z['V'].astype(np.float64),
                E=z['E'].astype(np.float64),vnames=np.array([str(x) for x in z['vnames']]),
                enames=np.array([str(x) for x in z['enames']]),subj=subj,sess=sess)

def norm(P): return P/np.clip(P.sum(-1,keepdims=True),1e-300,None)
def amean(Ps,w=None):
    Ps=np.asarray(Ps)
    if w is None: return norm(Ps.mean(0))
    w=np.asarray(w,float); w=w/w.sum()
    return norm(np.tensordot(w,Ps,axes=(0,0)))
def gmean(Ps,w=None):
    Ps=np.clip(np.asarray(Ps),1e-12,None)
    if w is None: L=np.log(Ps).mean(0)
    else:
        w=np.asarray(w,float); w=w/w.sum(); L=np.tensordot(w,np.log(Ps),axes=(0,0))
    L=L-L.max(-1,keepdims=True); return norm(np.exp(L))

def macro_f1(y,p,K):
    f=[]
    for k in range(K):
        tp=np.sum((p==k)&(y==k)); fp=np.sum((p==k)&(y!=k)); fn=np.sum((p!=k)&(y==k))
        f.append(0.0 if tp==0 else 2*tp/(2*tp+fp+fn))
    return float(np.mean(f)),[float(x) for x in f]

def sev_recall(y,p,sev_lo):
    """fraction of severe-labelled clips predicted into the severe band"""
    m=y>=sev_lo
    if m.sum()==0: return np.nan,0
    return float((p[m]>=sev_lo).mean()), int(m.sum())

def auroc(pos,neg):
    pos=np.asarray(pos,float); neg=np.asarray(neg,float)
    if len(pos)==0 or len(neg)==0: return np.nan
    allv=np.concatenate([pos,neg]); r=np.argsort(np.argsort(allv))+1.0
    # ties -> average ranks
    order=np.argsort(allv,kind='mergesort'); sv=allv[order]; rk=np.empty(len(allv))
    i=0
    while i<len(sv):
        j=i
        while j+1<len(sv) and sv[j+1]==sv[i]: j+=1
        rk[order[i:j+1]]=(i+j)/2.0+1.0; i=j+1
    R=rk[:len(pos)].sum()
    return float((R-len(pos)*(len(pos)+1)/2.0)/(len(pos)*len(neg)))
