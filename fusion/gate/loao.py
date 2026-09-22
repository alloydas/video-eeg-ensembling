import os
import sys, numpy as np, json, collections
sys.path.insert(0,'./gate')
from common import *
RNG=np.random.default_rng(7)
db=load('bin'); subj=db['subj']; us=np.array(sorted(set(subj)))
Ebin_all=db['E']; ybin=db['y']
# --- LOAO-honest selection of the EEG binary gate member AND of tau ---
gate_loao=np.zeros(len(ybin),bool); pick=collections.Counter(); tau_pick=[]
for s in us:
    te=subj==s; tr=~te
    sc=[prf(ybin[tr],Ebin_all[i][tr].argmax(1),2)['mF'] for i in range(Ebin_all.shape[0])]
    j=int(np.argmax(sc)); pick[db['enames'][j]]+=1
    gate_loao[te]=Ebin_all[j][te,1]>=0.5
print('LOAO gate-member picks:',dict(pick))
r=prf(ybin,gate_loao.astype(int),2)
print('LOAO EEG gate binary macroF1 %.4f  spec %.4f  sens %.4f'%(r['mF'],r['R'][0],r['R'][1]))
Ebin=db['E'].mean(0); Vbin=db['V'].mean(0)
es=[prf(ybin,db['E'][i].argmax(1),2)['mF'] for i in range(db['E'].shape[0])]
Ebest=db['E'][int(np.argmax(es))]

def cb(y,pd,subj,K,sev,nb=4000,seed=99):
    rng=np.random.default_rng(seed); idx={s:np.where(subj==s)[0] for s in us}
    names=list(pd); mF={n:np.empty(nb) for n in names}; sR={n:np.full(nb,np.nan) for n in names}
    s1={n:np.full(nb,np.nan) for n in names}
    for b in range(nb):
        p=rng.choice(len(us),len(us),replace=True); ii=np.concatenate([idx[us[j]] for j in p]); yb=y[ii]
        sm=np.isin(yb,sev); m1=yb==1
        for n in names:
            pb=pd[n][ii]; mF[n][b]=prf(yb,pb,K)['mF']
            if sm.sum(): sR[n][b]=np.isin(pb[sm],sev).mean()
            if m1.sum(): s1[n][b]=(pb[m1]==1).mean()
    return mF,sR,s1

OUT={}
for task in ('g3','g5'):
    g=load(task); y=g['y']; K=g['V'].shape[2]; Vb=g['V'].mean(0); Eb=g['E'].mean(0)
    sev=[2] if task=='g3' else [3,4]
    P={'video_only':Vb.argmax(1),
       'gate_eegLOAO@0.5':gated(gate_loao,Vb),
       'gate_eegEns@0.5':gated(Ebin[:,1]>=0.5,Vb),
       'gate_eegEns@bestTau':gated(Ebin[:,1]>=0.42,Vb),
       'gate_video@0.5':gated(Vbin[:,1]>=0.5,Vb),
       'fusion_w0.5':(0.5*Vb+0.5*Eb).argmax(1)}
    mF,sR,s1=cb(y,P,subj,K,sev)
    print('\n== %s =='%task)
    print('%-22s %-22s %-22s %-22s %6s'%('pipeline','mF1 [CI]','sevR [CI]','Stage1/mild R [CI]','P>vid'))
    for n,p in P.items():
        r=prf(y,p,K); s=float(np.isin(p[np.isin(y,sev)],sev).mean())
        d=mF[n]-mF['video_only']
        print('%-22s %.4f [%.4f,%.4f] %.4f [%.4f,%.4f] %.4f [%.4f,%.4f] %.3f'%(
          n,r['mF'],np.percentile(mF[n],2.5),np.percentile(mF[n],97.5),
          s,np.nanpercentile(sR[n],2.5),np.nanpercentile(sR[n],97.5),
          r['R'][1],np.nanpercentile(s1[n],2.5),np.nanpercentile(s1[n],97.5),np.mean(d>0)))
    # paired delta on class-1 recall (the mechanism)
    for n in ('gate_eegEns@0.5','gate_eegLOAO@0.5'):
        d1=s1[n]-s1['video_only']
        print('   %s  d(class1 recall) = %+.4f [%+.4f,%+.4f]  P=%.3f'%(n,np.nanmean(d1),np.nanpercentile(d1,2.5),np.nanpercentile(d1,97.5),np.mean(d1[~np.isnan(d1)]>0)))
    # how many clips change
    ch=(P['gate_eegEns@0.5']!=P['video_only'])
    corr=int(((P['gate_eegEns@0.5']==y)&ch).sum()); wrong=int(((P['video_only']==y)&ch).sum())
    print('   clips changed by EEG gate: %d (fixed %d, broke %d, net %+d)'%(ch.sum(),corr,wrong,corr-wrong))
    byc=collections.Counter((int(y[i]),int(P['video_only'][i]),int(P['gate_eegEns@0.5'][i])) for i in np.where(ch)[0])
    print('   (true,video,gated) ->', dict(sorted(byc.items(),key=lambda t:-t[1])))
