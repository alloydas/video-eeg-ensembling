import os
import numpy as np,glob,os,sys
sys.path.insert(0,'.')
from fz_core import macro_f1,norm
ROOT=os.environ.get('EEG_ROOT','/work/mech-ai-scratch/alloy/EEG')
_TOL=1e-3
def is_ds(P):
    k=P.shape[1]; e=np.e
    return abs(P.min()-1.0/(k-1+e))<_TOL and abs(P.max()-e/(k-1+e))<_TOL
def unsq(P):
    k=P.shape[1]; lp=np.log(np.clip(P,1e-12,None))
    q=np.clip(lp+(1.0-lp.sum(1,keepdims=True))/k,0.0,None); return q/q.sum(1,keepdims=True)
def ck(p):
    p=str(p)
    if p.endswith('/video.mp4'): p=p[:-len('/video.mp4')]
    return p.rstrip('/')
for task,K,lo in (('g5',5,3),('g3',3,2)):
    raw={}
    for d in sorted(glob.glob(os.path.join(ROOT,'output/v3_vidseeds','*_%s_s*'%task))):
        f=os.path.join(d,'val_preds.npz')
        if not os.path.exists(f): continue
        z=np.load(f,allow_pickle=True); P=z['probs'].astype(np.float64)
        if is_ds(P): P=unsq(P)
        pa=np.array([ck(x) for x in z['path']]) if 'path' in z.files else None
        raw[os.path.basename(d)]=(z['y'].astype(int),P,pa)
    ref=next(n for n in sorted(raw) if raw[n][2] is not None)
    rp=raw[ref][2]; o=np.argsort(rp); rp=rp[o]; ry=raw[ref][0][o]
    Ps=[]
    for n,(yy,P,pa) in raw.items():
        if pa is not None:
            q=np.argsort(pa)
            if np.array_equal(pa[q],rp) and np.array_equal(yy[q],ry): Ps.append(P[q])
        elif np.array_equal(yy,raw[ref][0]): Ps.append(P[o])
    Vfull=norm(np.mean(Ps,0)); pf=Vfull.argmax(1)
    inter=set(str(c) for c in np.load('aligned_%s.npz'%task,allow_pickle=True)['clips'])
    m=np.array([p in inter for p in rp])
    f_all,_=macro_f1(ry,pf,K); f_in,_=macro_f1(ry[m],pf[m],K); f_out,_=macro_f1(ry[~m],pf[~m],K)
    sr=lambda y,p:(float((p[y>=lo]>=lo).mean()),int((y>=lo).sum()))
    print('%s FULL video val n=%d (members=%d)  macroF1=%.4f  sevRec=%.4f n=%d'%(task,len(ry),len(Ps),f_all,*sr(ry,pf)))
    print('   on intersection  n=%d  macroF1=%.4f  sevRec=%.4f n=%d'%(m.sum(),f_in,*sr(ry[m],pf[m])))
    print('   dropped clips    n=%d  macroF1=%.4f  sevRec=%.4f n=%d'%((~m).sum(),f_out,*sr(ry[~m],pf[~m])))
    print('   class counts full',np.bincount(ry,minlength=K).tolist(),'inter',np.bincount(ry[m],minlength=K).tolist(),'dropped',np.bincount(ry[~m],minlength=K).tolist())
