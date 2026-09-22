import os
import sys, numpy as np, json
sys.path.insert(0,'./gate')
from common import *
db=load('bin')
Ebin=db['E'].mean(0); Vbin=db['V'].mean(0)
es=[prf(db['y'],db['E'][i].argmax(1),2)['mF'] for i in range(db['E'].shape[0])]
Ebest=db['E'][int(np.argmax(es))]
TAUS=np.concatenate([[0.0,0.001,0.005],np.arange(0.01,1.00,0.01),[0.995,0.999,1.0001]])
OUT={}
for task in ('g3','g5'):
    g=load(task); y=g['y']; K=g['V'].shape[2]; Vb=g['V'].mean(0); Eb=g['E'].mean(0)
    sev=[2] if task=='g3' else [3,4]; sm=np.isin(y,sev); nsev=int(sm.sum())
    base=prf(y,Vb.argmax(1),K); baseF=base['mF']; baseS=float(np.isin(Vb.argmax(1)[sm],sev).mean())
    rows={}
    for gname,score,grad in (('eegEns->video',Ebin[:,1],Vb),('eegBest->video',Ebest[:,1],Vb),
                             ('video->video',Vbin[:,1],Vb),('video->eeg',Vbin[:,1],Eb),
                             ('eegEns->eeg',Ebin[:,1],Eb)):
        rr=[]
        for t in TAUS:
            p=gated(score>=t,grad); r=prf(y,p,K)
            s=float(np.isin(p[sm],sev).mean())
            gp=score>=t
            rr.append(dict(tau=float(t),mF=r['mF'],mP=r['mP'],mR=r['mR'],sevR=s,acc=r['acc'],
                           R=r['R'],pass_rate=float(gp.mean()),
                           sev_pass=int(gp[sm].sum()), sev_ceiling=float(gp[sm].mean()),
                           spec=float((~gp[y==0]).mean())))
        rows[gname]=rr
        bf=max(rr,key=lambda x:x['mF'])
        both=[x for x in rr if x['mF']>baseF+1e-12 and x['sevR']>baseS+1e-12]
        beq =[x for x in rr if x['mF']>baseF+1e-12 and x['sevR']>=baseS-1e-12]
        print('%-6s %-16s bestF1 tau=%.3f mF=%.4f sevR=%.4f | video-alone mF=%.4f sevR=%.4f | taus beating BOTH strictly: %d | beating F1 w/ sevR>=: %d (tau range %s)'%(
            task,gname,bf['tau'],bf['mF'],bf['sevR'],baseF,baseS,len(both),len(beq),
            ('%.3f-%.3f'%(min(x['tau'] for x in beq),max(x['tau'] for x in beq))) if beq else '-'))
    OUT[task]=dict(rows=rows,baseF=baseF,baseS=baseS,nsev=nsev,counts=np.bincount(y,minlength=K).tolist())
json.dump(OUT,open('sweep.json','w'),indent=1)
