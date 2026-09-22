"""Is the error variance-limited (members disagree -> averaging helps) or
information-limited (members fail on the SAME clips -> averaging cannot help)?"""
import glob, os, re
import numpy as np
import os
ROOT=os.environ.get('EEG_ROOT',os.environ.get('EEG_ROOT', '/work/mech-ai-scratch/alloy/EEG')); _TOL=1e-3
def is_ds(P):
    k=P.shape[1]; e=np.e
    return abs(P.min()-1.0/(k-1+e))<_TOL and abs(P.max()-e/(k-1+e))<_TOL
def unsq(P):
    k=P.shape[1]; lp=np.log(np.clip(P,1e-12,None))
    q=np.clip(lp+(1.0-lp.sum(1,keepdims=True))/k,0.0,None); return q/q.sum(1,keepdims=True)
def load(root,fname,task):
    raw={}
    for d in sorted(glob.glob(os.path.join(root,'*_%s_s*'%task))):
        f=os.path.join(d,fname)
        if not os.path.exists(f): continue
        z=np.load(f,allow_pickle=True); P=z['probs'].astype(np.float64)
        if is_ds(P): P=unsq(P)
        raw[os.path.basename(d)]=dict(y=z['y'].astype(int),P=P,
            path=z['path'].astype(str) if 'path' in z.files else None)
    ref=next((n for n in sorted(raw) if raw[n]['path'] is not None),None)
    o=np.argsort(raw[ref]['path']); rp=raw[ref]['path'][o]; ry=raw[ref]['y'][o]; rr=raw[ref]['y']
    out={}
    for n,r in raw.items():
        if r['path'] is not None:
            q=np.argsort(r['path'])
            if np.array_equal(r['path'][q],rp) and np.array_equal(r['y'][q],ry): out[n]=r['P'][q]
        elif np.array_equal(r['y'],rr): out[n]=r['P'][o]
    return out,ry
print('%-6s %-4s %6s %9s %9s %9s %9s'%('mod','task','K','n_clips','all-wrong','some-wrong','irreducible'))
for mod,root,fname in (('VIDEO',f'{ROOT}/output/v3_vidseeds','val_preds.npz'),
                       ('EEG',f'{ROOT}/output/v3_bestcfg','val_clip_preds.npz')):
    for task in ('bin','g3','g5'):
        runs,y=load(root,fname,task)
        if not runs: continue
        M=np.stack([runs[n].argmax(1) for n in sorted(runs)])       # members x clips
        wrong=(M!=y[None,:])
        allw=wrong.all(0).sum(); somew=wrong.any(0).sum()
        print('%-6s %-4s %6d %9d %9d %9d %8.1f%%'%(mod,task,int(y.max())+1,y.size,allw,somew,100*allw/max(somew,1)))
        # severe classes only (the ones that matter)
        if task=='g5':
            sev=np.isin(y,[3,4])
            aw=wrong[:,sev].all(0).sum(); sw=wrong[:,sev].any(0).sum()
            print('   %-3s severe (S4+S5) only: n=%d  all-wrong=%d  some-wrong=%d  irreducible=%.1f%%'
                  %(mod,sev.sum(),aw,sw,100*aw/max(sw,1)))
