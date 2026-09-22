import os
import sys, numpy as np, json, collections
sys.path.insert(0,'./gate')
from common import *
db=load('bin'); subj=db['subj']; sess=db['sess']; us=np.array(sorted(set(subj)))
ybin=db['y']; Ebin_all=db['E']; Ebin=Ebin_all.mean(0); Vbin=db['V'].mean(0)
gate_loao=np.zeros(len(ybin),bool)
for s in us:
    te=subj==s; tr=~te
    sc=[prf(ybin[tr],Ebin_all[i][tr].argmax(1),2)['mF'] for i in range(Ebin_all.shape[0])]
    gate_loao[te]=Ebin_all[int(np.argmax(sc))][te,1]>=0.5
g5=load('g5'); y5=g5['y']; V5=g5['V'].mean(0)
print('Stage5 (n=14) animals:',dict(collections.Counter(subj[y5==4])))
print('Stage3 (n=74) animals:',dict(collections.Counter(subj[y5==3])))
print('Stage5 sessions:',len(set(sess[y5==4])),'Stage3 sessions:',len(set(sess[y5==3])))
print()
# per-animal: video-alone Stage1 recall vs gated
p_v=V5.argmax(1); p_g=gated(gate_loao,V5); p_ge=gated(Ebin[:,1]>=0.5,V5)
print('%-8s %5s %6s %6s | %s'%('animal','n_S1','vidR','gateR','n_total'))
for s in us:
    m=(subj==s)&(y5==1)
    if m.sum()==0: continue
    print('%-8s %5d %6.3f %6.3f | %d'%(s,m.sum(),(p_v[m]==1).mean(),(p_g[m]==1).mean(),(subj==s).sum()))
print()
# Gate binary performance summary, all three gates
for nm,gp in (('EEG ensemble @0.5',Ebin[:,1]>=0.5),('EEG LOAO best-member @0.5',gate_loao),
              ('EEG ensemble @0.42',Ebin[:,1]>=0.42),('VIDEO ensemble @0.5',Vbin[:,1]>=0.5)):
    r=prf(ybin,gp.astype(int),2)
    print('%-28s binF1 %.4f  sens %.4f (%d/%d)  spec %.4f (%d/%d)'%(nm,r['mF'],r['R'][1],
        int(gp[ybin==1].sum()),int((ybin==1).sum()),r['R'][0],int((~gp[ybin==0]).sum()),int((ybin==0).sum())))
