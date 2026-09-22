import os
import sys, numpy as np, collections
sys.path.insert(0,'./gate')
from common import *
db=load('bin'); subj=db['subj']; us=np.array(sorted(set(subj))); ybin=db['y']
Ea=db['E']; Ebin=Ea.mean(0); Vbin=db['V'].mean(0)
gl=np.zeros(len(ybin),bool)
for s in us:
    te=subj==s; tr=~te
    sc=[prf(ybin[tr],Ea[i][tr].argmax(1),2)['mF'] for i in range(Ea.shape[0])]
    gl[te]=Ea[int(np.argmax(sc))][te,1]>=0.5
for task in ('g3','g5'):
    g=load(task); y=g['y']; K=g['V'].shape[2]; Vb=g['V'].mean(0)
    sev=[2] if task=='g3' else [3,4]
    pv=Vb.argmax(1); pg=gated(gl,Vb)
    print('== %s  drop-one-animal jackknife of  d(macroF1) = gate_eegLOAO - video_only'%task)
    full=prf(y,pg,K)['mF']-prf(y,pv,K)['mF']
    print('   ALL 15 animals: d = %+.4f'%full)
    rows=[]
    for s in us:
        m=subj!=s
        d=prf(y[m],pg[m],K)['mF']-prf(y[m],pv[m],K)['mF']
        rows.append((s,int((subj==s).sum()),d))
    for s,n,d in sorted(rows,key=lambda t:t[2]):
        print('   drop %-7s (n=%4d): d = %+.4f %s'%(s,n,d,'  <-- sign flips' if d<=0 else ''))
    print()
