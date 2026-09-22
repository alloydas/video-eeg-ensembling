import numpy as np, json, collections, sys
sys.path.insert(0,'.')
from fz_core import *
db=load('bin'); Ebin=amean(db['E'])
OUT={}
for task,K,lo in (('g5',5,3),('g3',3,2)):
    d=load(task); y=d['y']; subj=d['subj']; u=np.array(sorted(set(subj)))
    V=amean(d['V']); E=amean(d['E'])
    f1s=[macro_f1(y,m.argmax(1),K)[0] for m in d['E']]; Eb=d['E'][int(np.argmax(f1s))]
    pvsz=V[:,1:]/np.clip(V[:,1:].sum(1,keepdims=True),1e-300,None)
    GATE=norm(np.concatenate([Ebin[:,[0]],Ebin[:,[1]]*pvsz],1))
    def veto(w):
        F=amean([V,E],[1-w,w]); p=F.argmax(1); mv=V.argmax(1); m=mv>=lo; p[m]=mv[m]; return p
    preds=collections.OrderedDict([
        ('eeg_ens',E.argmax(1)),('eeg_best_single',Eb.argmax(1)),
        ('fuse_arith_50_50',amean([V,E]).argmax(1)),
        ('fuse_with_EEGbest_50_50',amean([V,Eb]).argmax(1)),
        ('fuse_arith_90_10',amean([V,E],[.9,.1]).argmax(1)),
        ('gate_eegbin_x_vidsev',GATE.argmax(1)),
        ('R2_video_veto_on_severe(w=.5)',veto(0.5))])
    pv=V.argmax(1)
    def score(yy,pp):
        f,_=macro_f1(yy,pp,K); s,_=sev_recall(yy,pp,lo); return f,(0.0 if np.isnan(s) else s)
    r=np.random.default_rng(21); idx={a:np.where(subj==a)[0] for a in u}
    st={n:dict(bF=0,bN=0,tot=0) for n in preds}
    for _ in range(4000):
        ii=np.concatenate([idx[u[j]] for j in r.integers(0,len(u),len(u))]); yy=y[ii]
        if (yy>=lo).sum()==0: continue
        bf,bs=score(yy,pv[ii])
        for n,pp in preds.items():
            f,s=score(yy,pp[ii]); st[n]['tot']+=1
            if f>bf and s>=bs: st[n]['bF']+=1
            if s>=bs: st[n]['bN']+=1
    sev=y>=lo; ani=sorted(set(subj[sev]))
    pera={}
    for n,pp in list(preds.items())+[('video_ens',pv)]:
        pera[n]={a:dict(n=int((sev&(subj==a)).sum()),
                        rec=float((pp[sev&(subj==a)]>=lo).mean())) for a in ani}
    rows={}
    for n,pp in preds.items():
        f,s=score(y,pp)
        rows[n]=dict(macroF1=f,sev_recall=s,d_f1=f-score(y,pv)[0],d_sev=s-score(y,pv)[1],
                     P_F1better_and_sev_not_worse=st[n]['bF']/st[n]['tot'],
                     P_sev_not_worse=st[n]['bN']/st[n]['tot'])
    OUT[task]=dict(rows=rows,per_animal=pera,video=dict(macroF1=score(y,pv)[0],sev_recall=score(y,pv)[1]))
    print('\n### %s  (video macroF1=%.4f sevRec=%.4f)'%(task,*score(y,pv)))
    for n,v in rows.items():
        print('  %-32s F1=%.4f(%+.4f) sev=%.4f(%+.4f)  P(F1 up & sev not down)=%.3f  P(sev not down)=%.3f'%(
            n,v['macroF1'],v['d_f1'],v['sev_recall'],v['d_sev'],v['P_F1better_and_sev_not_worse'],v['P_sev_not_worse']))
    print('  per-animal severe recall (video -> fuse50 -> veto):')
    for a in ani:
        print('    %-7s n=%2d  vid=%.3f  fuse50=%.3f (%+.3f)  veto=%.3f  gate=%.3f  eeg=%.3f'%(
            a,pera['video_ens'][a]['n'],pera['video_ens'][a]['rec'],pera['fuse_arith_50_50'][a]['rec'],
            pera['fuse_arith_50_50'][a]['rec']-pera['video_ens'][a]['rec'],
            pera['R2_video_veto_on_severe(w=.5)'][a]['rec'],pera['gate_eegbin_x_vidsev'][a]['rec'],pera['eeg_ens'][a]['rec']))
json.dump(OUT,open('fz_p5.json','w'),indent=1); print('\nwrote fz_p5.json')
