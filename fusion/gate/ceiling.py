import os
import sys, numpy as np, json
sys.path.insert(0,'./gate')
from common import *
db=load('bin'); d5=load('g5'); d3=load('g3')
Ebin=db['E'].mean(0); Vbin=db['V'].mean(0)
es=[prf(db['y'],db['E'][i].argmax(1),2)['mF'] for i in range(db['E'].shape[0])]
bi=int(np.argmax(es)); Ebest=db['E'][bi]
print('EEG best-single binary member:',db['enames'][bi],'macroF1 %.4f'%es[bi])
LAB5=['non-seizure','Stage1','Stage2','Stage3','Stage5']
LAB3=['non-seizure','mild(S1+S2)','severe(S3+S5)']
OUT={}
for nm,g,LAB in (('g5',d5,LAB5),('g3',d3,LAB3)):
    y=g['y']; K=g['V'].shape[2]; Vb=g['V'].mean(0)
    rows=[]
    for c in range(K):
        m=y==c; n=int(m.sum())
        r={}
        for gname,gp in (('eeg_ens',Ebin[:,1]>=0.5),('eeg_best',Ebest[:,1]>=0.5),('video_ens',Vbin[:,1]>=0.5)):
            passed=int(gp[m].sum())
            r[gname]=dict(n=n,passed=passed,rejected=n-passed,
                          pass_rate=passed/n, ceiling=(passed/n) if c>0 else None)
        # video-alone realised recall for that class
        r['video_alone_recall']=float((Vb[m].argmax(1)==c).mean())
        # video confusion into class 0 among true c
        r['video_pred0_frac']=float((Vb[m].argmax(1)==0).mean())
        rows.append((LAB[c],c,n,r))
    OUT[nm]=rows
    print('\n== %s : per-class gate pass rate (= ceiling on stage-2 recall) ==' % nm)
    print('%-14s %5s | %-28s | %-28s | %-28s | %8s %8s'%('class','n','EEG-ens gate @0.5','EEG-best gate @0.5','VIDEO-ens gate @0.5','vidAlnR','vid->0'))
    for lab,c,n,r in rows:
        f=lambda k:'pass %3d/%3d = %.4f'%(r[k]['passed'],n,r[k]['pass_rate'])
        print('%-14s %5d | %-28s | %-28s | %-28s | %8.4f %8.4f'%(lab,n,f('eeg_ens'),f('eeg_best'),f('video_ens'),r['video_alone_recall'],r['video_pred0_frac']))
json.dump(OUT,open('ceiling.json','w'),indent=1,default=float)

# confusion matrices video_only
for nm,g,LAB in (('g3',d3,LAB3),('g5',d5,LAB5)):
    y=g['y']; Vb=g['V'].mean(0); p=Vb.argmax(1); K=len(LAB)
    C=np.zeros((K,K),int)
    for a,b in zip(y,p): C[a,b]+=1
    print('\n%s video_only confusion (rows=true):'%nm)
    print('%-14s'%'', ' '.join('%6s'%l[:6] for l in LAB))
    for i,l in enumerate(LAB): print('%-14s'%l, ' '.join('%6d'%v for v in C[i]))
