import os
"""Build the clip intersection between the video and EEG validation sets.

Video paths look like  data/Data_RN197_cropped/<day>/<clip>/video.mp4
EEG paths   look like  data/Data_RN197_cropped/<day>/<clip>
so the clip directory is the join key.
"""
import glob, os, re, json
import numpy as np
import os
ROOT=os.environ.get('EEG_ROOT',os.environ.get('EEG_ROOT', '/work/mech-ai-scratch/alloy/EEG')); _TOL=1e-3
def is_ds(P):
    k=P.shape[1]; e=np.e
    return abs(P.min()-1.0/(k-1+e))<_TOL and abs(P.max()-e/(k-1+e))<_TOL
def unsq(P):
    k=P.shape[1]; lp=np.log(np.clip(P,1e-12,None))
    q=np.clip(lp+(1.0-lp.sum(1,keepdims=True))/k,0.0,None); return q/q.sum(1,keepdims=True)
def clipkey(p):
    p=str(p)
    if p.endswith('/video.mp4'): p=p[:-len('/video.mp4')]
    return p.rstrip('/')

def load(root,fname,task):
    raw={}
    for d in sorted(glob.glob(os.path.join(root,'*_%s_s*'%task))):
        f=os.path.join(d,fname)
        if not os.path.exists(f): continue
        z=np.load(f,allow_pickle=True); P=z['probs'].astype(np.float64)
        if is_ds(P): P=unsq(P)
        raw[os.path.basename(d)]=dict(y=z['y'].astype(int),P=P,
            path=np.array([clipkey(x) for x in z['path']]) if 'path' in z.files else None)
    ref=next((n for n in sorted(raw) if raw[n]['path'] is not None),None)
    o=np.argsort(raw[ref]['path']); rp=raw[ref]['path'][o]; ry=raw[ref]['y'][o]; rr=raw[ref]['y']
    out={}
    for n,r in raw.items():
        if r['path'] is not None:
            q=np.argsort(r['path'])
            if np.array_equal(r['path'][q],rp) and np.array_equal(r['y'][q],ry): out[n]=r['P'][q]
        elif np.array_equal(r['y'],rr): out[n]=r['P'][o]
    return out,ry,rp

for task in ('bin','g3','g5'):
    V,vy,vp = load(f'{ROOT}/output/v3_vidseeds','val_preds.npz',task)
    E,ey,ep = load(f'{ROOT}/output/v3_bestcfg','val_clip_preds.npz',task)
    sv,se = set(vp), set(ep)
    inter = sorted(sv & se)
    print('\n== task=%s  video=%d clips  eeg=%d clips  intersection=%d  (video-only %d, eeg-only %d)'
          %(task,len(sv),len(se),len(inter),len(sv-se),len(se-sv)))
    vi={p:i for i,p in enumerate(vp)}; ei={p:i for i,p in enumerate(ep)}
    iv=np.array([vi[p] for p in inter]); ie=np.array([ei[p] for p in inter])
    agree = int((vy[iv]==ey[ie]).sum())
    print('   labels agree on intersection: %d/%d (%.4f)'%(agree,len(inter),agree/len(inter)))
    if agree!=len(inter):
        bad=np.where(vy[iv]!=ey[ie])[0][:5]
        for b in bad: print('     MISMATCH', inter[b], 'video y=',vy[iv][b],'eeg y=',ey[ie][b])
    np.savez(os.path.join(os.path.dirname(__file__),'aligned_%s.npz'%task),
             clips=np.array(inter), y=vy[iv],
             V=np.stack([V[n][iv] for n in sorted(V)]), vnames=np.array(sorted(V)),
             E=np.stack([E[n][ie] for n in sorted(E)]), enames=np.array(sorted(E)))
    print('   saved aligned_%s.npz  video members=%d  eeg members=%d  class counts=%s'
          %(task,len(V),len(E),np.bincount(vy[iv]).tolist()))
