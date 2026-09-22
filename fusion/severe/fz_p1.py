import os
import numpy as np, json, collections, sys
sys.path.insert(0,'.')
from fz_core import *
rng=np.random.default_rng(0)
OUT={}
db=load('bin')
Ebin=amean(db['E']); Vbin=amean(db['V'])

def variants(d,K,sev_lo):
    V=amean(d['V']); E=amean(d['E'])
    f1s=[macro_f1(d['y'],m.argmax(1),K)[0] for m in d['E']]
    bi=int(np.argmax(f1s)); Ebest=d['E'][bi]
    out=collections.OrderedDict()
    out['video_ens']=V
    out['eeg_ens']=E
    out['eeg_best_single(%s)'%d['enames'][bi]]=Ebest
    out['fuse_arith_50_50']=amean([V,E])
    out['fuse_arith_70_30']=amean([V,E],[0.7,0.3])
    out['fuse_arith_90_10']=amean([V,E],[0.9,0.1])
    out['fuse_geom_50_50']=gmean([V,E])
    # R3: EEG detection gate x video severity  (the standing project position)
    pv_sz=V[:,1:]/np.clip(V[:,1:].sum(1,keepdims=True),1e-300,None)
    R3=np.concatenate([Ebin[:,[0]], Ebin[:,[1]]*pv_sz],1)
    out['gate_eegbin_x_vidsev']=norm(R3)
    return out,bi,f1s

def clusters(subj):
    u=np.array(sorted(set(subj))); idx={a:np.where(subj==a)[0] for a in u}
    return u,idx

def boot_ci(y,preds,subj,K,sev_lo,B=4000,seed=0):
    """cluster bootstrap over animals; returns dict name-> (f1 lo,hi), (sev lo,hi) and paired deltas vs video_ens"""
    r=np.random.default_rng(seed); u,idx=clusters(subj)
    names=list(preds)
    f1b={n:[] for n in names}; svb={n:[] for n in names}
    dF={n:[] for n in names}; dS={n:[] for n in names}
    for b in range(B):
        pick=r.choice(len(u),len(u),replace=True)
        ii=np.concatenate([idx[u[j]] for j in pick])
        yy=y[ii]
        if (yy>=sev_lo).sum()==0: continue
        base_f=base_s=None
        for n in names:
            pp=preds[n][ii]
            f,_=macro_f1(yy,pp,K); s,_=sev_recall(yy,pp,sev_lo)
            f1b[n].append(f); svb[n].append(s)
            if n=='video_ens': base_f,base_s=f,s
        for n in names:
            dF[n].append(f1b[n][-1]-base_f); dS[n].append(svb[n][-1]-base_s)
    q=lambda a:(float(np.percentile(a,2.5)),float(np.percentile(a,97.5)))
    res={}
    for n in names:
        res[n]=dict(f1_ci=q(f1b[n]),sev_ci=q(svb[n]),
                    d_f1_ci=q(dF[n]),d_sev_ci=q(dS[n]),
                    p_f1_worse=float(np.mean(np.array(dF[n])<0)),
                    p_sev_worse=float(np.mean(np.array(dS[n])<0)),
                    nboot=len(f1b[n]))
    return res

def signtest(k,n):
    """two-sided exact sign test p for k successes of n"""
    from math import comb
    if n==0: return 1.0
    p=sum(comb(n,i) for i in range(0,min(k,n-k)+1))/2**n*2
    return float(min(1.0,p))

for task,K,sev_lo in (('g5',5,3),('g3',3,2)):
    d=load(task); y=d['y']; subj=d['subj']
    P,bi,f1s=variants(d,K,sev_lo)
    preds={n:p.argmax(1) for n,p in P.items()}
    rows={}
    for n,pp in preds.items():
        f,per=macro_f1(y,pp,K); s,ns=sev_recall(y,pp,sev_lo)
        rows[n]=dict(macroF1=f,perclass_f1=per,sev_recall=s,n_sev=ns)
        if task=='g5':
            rows[n]['s4_recall']=float((pp[y==3]>=3).mean()); rows[n]['s5_recall']=float((pp[y==4]>=3).mean())
            rows[n]['s4_exact']=float((pp[y==3]==3).mean()); rows[n]['s5_exact']=float((pp[y==4]==4).mean())
    ci=boot_ci(y,preds,subj,K,sev_lo,B=4000,seed=1)
    for n in rows: rows[n].update(ci[n])
    # per-animal sign test vs video
    sev=y>=sev_lo; ani=sorted(set(subj[sev]))
    per_animal={}
    for n,pp in preds.items():
        rec={}
        for a in ani:
            m=sev&(subj==a); rec[a]=dict(n=int(m.sum()),recall=float((pp[m]>=sev_lo).mean()))
        per_animal[n]=rec
    st={}
    base=per_animal['video_ens']
    for n in preds:
        w=l=t=0
        for a in ani:
            dv=per_animal[n][a]['recall']-base[a]['recall']
            if dv>0: w+=1
            elif dv<0: l+=1
            else: t+=1
        st[n]=dict(wins=w,losses=l,ties=t,n_animals_with_severe=len(ani),
                   p_sign=signtest(min(w,l),w+l))
    OUT[task]=dict(rows=rows,per_animal=per_animal,signtest=st,
                   n_animals=len(set(subj)),n_animals_severe=len(ani),
                   eeg_best_idx=bi,eeg_member_f1_range=[float(min(f1s)),float(max(f1s))])
    print('\n### task=%s  K=%d  severe=y>=%d  (n_sev=%d, animals with severe=%d/%d)'%(task,K,sev_lo,int(sev.sum()),len(ani),len(set(subj))))
    print('%-34s %7s %-15s %7s %-15s %-17s %-17s'%('variant','macroF1','[95% CI]','sevRec','[95% CI]','d_macroF1 vs vid','d_sevRec vs vid'))
    for n in rows:
        r=rows[n]
        print('%-34s %7.4f [%.4f,%.4f] %7.4f [%.4f,%.4f] %+.4f[%+.4f,%+.4f] %+.4f[%+.4f,%+.4f] Pworse=%.3f'%(
            n,r['macroF1'],r['f1_ci'][0],r['f1_ci'][1],r['sev_recall'],r['sev_ci'][0],r['sev_ci'][1],
            r['macroF1']-rows['video_ens']['macroF1'],r['d_f1_ci'][0],r['d_f1_ci'][1],
            r['sev_recall']-rows['video_ens']['sev_recall'],r['d_sev_ci'][0],r['d_sev_ci'][1],r['p_sev_worse']))
    print(' sign test vs video (per animal severe recall):')
    for n in st: print('    %-34s W%d L%d T%d  p=%.4f'%(n,st[n]['wins'],st[n]['losses'],st[n]['ties'],st[n]['p_sign']))
json.dump(OUT,open('fz_p1.json','w'),indent=1)
print('\nwrote fz_p1.json')
