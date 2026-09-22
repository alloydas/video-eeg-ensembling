import os
import sys, numpy as np, json, collections
sys.path.insert(0,'./gate')
from common import *
db=load('bin'); subj=db['subj']; sess=db['sess']; us=np.array(sorted(set(subj)))
Ebin_all=db['E']; ybin=db['y']; Ebin=Ebin_all.mean(0); Vbin=db['V'].mean(0)
gate_loao=np.zeros(len(ybin),bool)
for s in us:
    te=subj==s; tr=~te
    sc=[prf(ybin[tr],Ebin_all[i][tr].argmax(1),2)['mF'] for i in range(Ebin_all.shape[0])]
    gate_loao[te]=Ebin_all[int(np.argmax(sc))][te,1]>=0.5

g3=load('g3'); g5=load('g5')
# severe distribution
for nm,g,sev in (('g3',g3,[2]),('g5',g5,[3,4])):
    y=g['y']; sm=np.isin(y,sev)
    print('%s severe n=%d  animals=%s'%(nm,sm.sum(),dict(collections.Counter(subj[sm]))))
    print('   sessions carrying >=1 severe: %d of %d'%(len(set(sess[sm])),len(set(sess))))
    cs=collections.Counter(sess[sm]); print('   per-session severe counts:',sorted(cs.values(),reverse=True))
print()
# within-session evaluation, restricted to sessions with >=2 classes and >=10 clips
for nm,g,sev in (('g3',g3,[2]),('g5',g5,[3,4])):
    y=g['y']; K=g['V'].shape[2]; Vb=g['V'].mean(0); Eb=g['E'].mean(0)
    P={'video_only':Vb.argmax(1),'gate_eegLOAO':gated(gate_loao,Vb),
       'gate_eegEns':gated(Ebin[:,1]>=0.5,Vb),'gate_video':gated(Vbin[:,1]>=0.5,Vb),
       'eeg_only_ens':Eb.argmax(1),'gate_video_gradEEG':gated(Vbin[:,1]>=0.5,Eb),
       'fusion_w0.5':(0.5*Vb+0.5*Eb).argmax(1)}
    elig=[s for s in sorted(set(sess)) if (sess==s).sum()>=10 and len(set(y[sess==s].tolist()))>=2]
    print('== %s  within-session (n=%d eligible sessions of %d) =='%(nm,len(elig),len(set(sess))))
    for n,p in P.items():
        f=[prf(y[sess==s],p[sess==s],K)['mF'] for s in elig]
        # within-session severe recall: sessions containing >=1 severe clip
        sev_s=[s for s in sorted(set(sess)) if np.isin(y[sess==s],sev).sum()>0]
        rec=[]; tot=0; hit=0
        for s in sev_s:
            m=(sess==s)&np.isin(y,sev); tot+=int(m.sum()); hit+=int(np.isin(p[m],sev).sum())
            rec.append(np.isin(p[m],sev).mean())
        print('   %-20s withinSess mF1 %.4f (sd %.3f)   sev-sessions=%d  sev recall pooled %d/%d=%.4f  session-mean %.4f'%(
            n,np.mean(f),np.std(f),len(sev_s),hit,tot,hit/tot,np.mean(rec)))
    print()
