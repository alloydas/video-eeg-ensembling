import numpy as np, json, collections, sys
sys.path.insert(0,'.')
from fz_core import *
OUT={}
for task,K,lo in (('g5',5,3),('g3',3,2)):
    d=load(task); y=d['y']; subj=d['subj']
    V=amean(d['V']); E=amean(d['E']); F=amean([V,E])
    pv,pe,pf=V.argmax(1),E.argmax(1),F.argmax(1)
    sev=y>=lo
    Vok=sev&(pv>=lo); Fok=sev&(pf>=lo)
    A=Vok&~Fok          # video right, fusion wrong  (the damage)
    B=Vok&Fok           # both right
    C=sev&~Vok&Fok      # fusion rescues one video missed
    Dd=sev&~Vok&~Fok    # both wrong
    ent=lambda P:-(P*np.log(np.clip(P,1e-12,None))).sum(1)/np.log(P.shape[1])
    def desc(m,nm):
        if m.sum()==0: return dict(name=nm,n=0)
        return dict(name=nm,n=int(m.sum()),
            animals=sorted(set(subj[m].tolist())),
            eeg_mass_true_class=float(E[m,y[m]].mean()),
            eeg_mass_severe_band=float(E[m,lo:].sum(1).mean()),
            eeg_maxprob=float(E[m].max(1).mean()),
            eeg_norm_entropy=float(ent(E[m]).mean()),
            eeg_argmax_hist={int(k):int((pe[m]==k).sum()) for k in range(K)},
            vid_mass_severe_band=float(V[m,lo:].sum(1).mean()),
            vid_maxprob=float(V[m].max(1).mean()),
            vid_margin_sevband=float((V[m,lo:].sum(1)-V[m,:lo].max(1)).mean()))
    cells=[desc(A,'A_video_right_fusion_wrong'),desc(B,'B_both_right'),
           desc(C,'C_fusion_rescues'),desc(Dd,'D_both_wrong')]
    # flat vs confidently-wrong: EEG uniform would be 1/K
    unif=1.0/K
    OUT[task]=dict(n_sev=int(sev.sum()),K=K,uniform_prob=unif,cells=cells,
        clips_A=[d['clips'][i] for i in np.where(A)[0]],
        detail_A=[dict(clip=d['clips'][i],y=int(y[i]),
                       vid=[round(float(x),3) for x in V[i]],
                       eeg=[round(float(x),3) for x in E[i]],
                       fus=[round(float(x),3) for x in F[i]],
                       vid_pred=int(pv[i]),eeg_pred=int(pe[i]),fus_pred=int(pf[i])) for i in np.where(A)[0]])
    print('\n### %s  severe n=%d  (K=%d, uniform=%.3f)'%(task,sev.sum(),K,unif))
    for c in cells:
        if c['n']==0: print('  %-28s n=0'%c['name']); continue
        print('  %-28s n=%-3d  EEGmass(true cls)=%.4f  EEGmass(sev band)=%.4f  EEGmaxp=%.3f  EEGentropy=%.3f  EEGargmax=%s'%(
            c['name'],c['n'],c['eeg_mass_true_class'],c['eeg_mass_severe_band'],c['eeg_maxprob'],c['eeg_norm_entropy'],c['eeg_argmax_hist']))
        print('     %28s  VIDmass(sev band)=%.4f  VIDmaxp=%.3f  VIDmargin=%+.4f  animals=%s'%('',c['vid_mass_severe_band'],c['vid_maxprob'],c['vid_margin_sevband'],','.join(c['animals'])))
    if task=='g5':
        print('  --- the %d flipped clips in detail ---'%A.sum())
        for r in OUT[task]['detail_A']:
            print('   y=%d vid%s->%d  eeg%s->%d  fus%s->%d  %s'%(r['y'],r['vid'],r['vid_pred'],r['eeg'],r['eeg_pred'],r['fus'],r['fus_pred'],r['clip'].split('/')[-1][:44]))
json.dump(OUT,open('fz_p3.json','w'),indent=1); print('\nwrote fz_p3.json')
