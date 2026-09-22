import os
"""Selective ensembling: does curating the member pool rescue the EEG ensemble?

Member selection must not peek at the test labels, so every rule below is applied on
TRAINING-time information only (which architecture family, and each run's own recorded
validation history), never on the val F1 we are reporting. The 'oracle top-k' row is
printed ONLY as an upper bound and is explicitly labelled as cheating.
"""
import glob, os, re, json
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef
import importlib.util
spec = importlib.util.spec_from_file_location('eb', os.path.join(os.path.dirname(__file__), 'ens_both.py'))

import os
_DEFAULT_ROOT = os.environ.get('EEG_ROOT', '/work/mech-ai-scratch/alloy/EEG')
ROOT = os.environ.get('EEG_ROOT', _DEFAULT_ROOT)  # override for another machine
_TOL=1e-3
def is_ds(P):
    k=P.shape[1]; e=np.e
    return abs(P.min()-1.0/(k-1+e))<_TOL and abs(P.max()-e/(k-1+e))<_TOL
def unsq(P):
    k=P.shape[1]; lp=np.log(np.clip(P,1e-12,None))
    q=np.clip(lp+(1.0-lp.sum(1,keepdims=True))/k,0.0,None); return q/q.sum(1,keepdims=True)
def F1(y,P): return f1_score(y,P.argmax(1),average='macro',zero_division=0)

def load(root,fname,task):
    raw={}
    for d in sorted(glob.glob(os.path.join(root,'*_%s_s*'%task))):
        f=os.path.join(d,fname)
        if not os.path.exists(f): continue
        z=np.load(f,allow_pickle=True); P=z['probs'].astype(np.float64)
        if is_ds(P): P=unsq(P)
        raw[os.path.basename(d)]=dict(y=z['y'].astype(int),P=P,
            path=z['path'].astype(str) if 'path' in z.files else None, dir=d)
    ref=next((n for n in sorted(raw) if raw[n]['path'] is not None),None)
    order=np.argsort(raw[ref]['path']); rp=raw[ref]['path'][order]; ry=raw[ref]['y'][order]
    rr=raw[ref]['y']; out={}
    for n,r in raw.items():
        if r['path'] is not None:
            o=np.argsort(r['path'])
            if np.array_equal(r['path'][o],rp) and np.array_equal(r['y'][o],ry): out[n]=(r['P'][o],r['dir'])
        elif np.array_equal(r['y'],rr): out[n]=(r['P'][order],r['dir'])
    return out,ry

def best_val_from_history(d):
    """Each run's own recorded best validation macro-F1 -- available at training time."""
    for fn in ('history.json','results.json'):
        p=os.path.join(d,fn)
        if not os.path.exists(p): continue
        try: j=json.load(open(p))
        except Exception: continue
        if isinstance(j,dict):
            for k in ('best_macro_f1','macro_f1','best_val_macro_f1','val_macro_f1'):
                if k in j and isinstance(j[k],(int,float)): return float(j[k])
            h=j.get('val_macro_f1') or j.get('val_f1')
            if isinstance(h,list) and h: return float(max(h))
    return None

DEEP={'gru','lstm','eegnet','conformer','tcn'}
for mod,root,fname in (('EEG',f'{ROOT}/output/v3_bestcfg','val_clip_preds.npz'),
                       ('VIDEO',f'{ROOT}/output/v3_vidseeds','val_preds.npz')):
    print('\n'+'='*92); print('###',mod)
    for task in ('bin','g3','g5'):
        runs,y=load(root,fname,task)
        if not runs: continue
        names=sorted(runs)
        arch_of={n:re.sub(r'_%s_s\d+$'%task,'',n) for n in names}
        singles={n:F1(y,runs[n][0]) for n in names}
        hist={n:best_val_from_history(runs[n][1]) for n in names}
        def ens(ms,label):
            if not ms: return
            f=F1(y,np.mean([runs[m][0] for m in ms],axis=0))
            print('     %-34s n=%-3d F1 %.4f   %+0.4f vs best single'%(label,len(ms),f,f-max(singles.values())))
        print('\n-- task=%s   best single %.4f (%s)   all-member ensemble:'%(task,max(singles.values()),max(names,key=singles.get)))
        ens(names,'all members')
        ens([n for n in names if arch_of[n] in DEEP],'deep architectures only')
        # drop runs whose OWN recorded validation history says they collapsed
        ok=[n for n in names if hist[n] is None or hist[n]>=0.5]
        ens(ok,'drop self-reported collapsed runs')
        ens([n for n in ok if arch_of[n] in DEEP],'deep + drop collapsed')
        # per-architecture family ensembles, top families by seed-mean
        fam={}
        for n in names: fam.setdefault(arch_of[n],[]).append(n)
        rank=sorted(fam,key=lambda a:-np.mean([singles[m] for m in fam[a]]))
        for k in (2,3):
            ens([m for a in rank[:k] for m in fam[a]],'top-%d families (ORACLE, cheats)'%k)
