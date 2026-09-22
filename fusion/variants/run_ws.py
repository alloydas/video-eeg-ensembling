import os
import json, sys, numpy as np
sys.path.insert(0,'./fuse')
from fuse_lib import *
D={t:load(t) for t in ('bin','g3','g5')}
rng=np.random.default_rng(11)
OUT={}

def ws_auc(score, sess, pos, neg):
    num=den=0.0
    for s in np.unique(sess):
        m=sess==s; p=score[m&pos]; n=score[m&neg]
        if p.size==0 or n.size==0: continue
        d=p[:,None]-n[None,:]; num+=(d>0).sum()+0.5*(d==0).sum(); den+=d.size
    return num/den if den else np.nan

print('### Paired animal-clustered bootstrap on WITHIN-SESSION severe-vs-mild AUROC (delta vs VIDEO)')
for task in ('g3','g5'):
    d=D[task]; y=d['y']; K=d['K']; sess=d['sess']; subj=d['subj']
    Vens=d['V'].mean(0); Eens=d['E'].mean(0); Xcat=np.hstack([Vens,Eens])
    sevi=SEVERE[task]
    pos=np.isin(y,sevi); neg=(y>0)&~pos
    def ss(S): return S[:,sevi].sum(1)
    cand=dict(VIDEO=ss(Vens),EEG=ss(Eens),AVG=ss(0.5*Vens+0.5*Eens),
              GEO=ss(geometric(Vens,Eens)),RANK=ss(rank_avg(Vens,Eens)),
              MAXCONF=ss(maxconf(Vens,Eens)),STACK=ss(stack_loao(Xcat,y,subj,K)),
              STACK_V=ss(stack_loao(Vens,y,subj,K)))
    an=np.unique(subj); names=list(cand)
    bs={n:[] for n in names}
    for b in range(3000):
        pick=rng.choice(an,an.size,replace=True)
        ii=np.concatenate([np.where(subj==a)[0] for a in pick])
        tag=np.concatenate([np.full((subj==a).sum(),'#%d'%j) for j,a in enumerate(pick)])
        sb=np.char.add(sess[ii],tag); pb=pos[ii]; nb=neg[ii]
        for n in names:
            bs[n].append(ws_auc(cand[n][ii],sb,pb,nb))
    obs={n:ws_auc(cand[n],sess,pos,neg) for n in names}
    rows=[]
    print('\n-- task=%s  severe=%d mild=%d  animals=%d'%(task,pos.sum(),neg.sum(),an.size))
    print('   %-9s %8s %+9s %-20s %6s'%('scorer','within','delta','95% CI (paired)','P>0'))
    for n in names:
        dd=np.array(bs[n])-np.array(bs['VIDEO']); dd=dd[~np.isnan(dd)]
        lo,hi=np.percentile(dd,[2.5,97.5])
        rows.append(dict(scorer=n,within=float(obs[n]),delta=float(obs[n]-obs['VIDEO']),
                         lo=float(lo),hi=float(hi),pgt=float((dd>0).mean())))
        print('   %-9s %8.4f %+9.4f [%+.4f,%+.4f] %6.3f'%(n,obs[n],obs[n]-obs['VIDEO'],lo,hi,(dd>0).mean()))
    OUT[task]=rows

# full weight sweeps
SW={}
print('\n\n### WEIGHT SWEEP   S = w*VIDEO + (1-w)*EEG   (w=1 is video-only, w=0 is EEG-only)')
for task in ('bin','g3','g5'):
    d=D[task]; y=d['y']; K=d['K']
    Vens=d['V'].mean(0); Eens=d['E'].mean(0)
    rows=[]
    print('\n-- task=%s  (%s)'%(task,', '.join('%s n=%d'%(n,c) for n,c in zip(NAMES[task],np.bincount(y,minlength=K)))))
    print('   %5s %7s %7s %7s %7s %7s   %s'%('w','P','R','F1','AUC','MCC','per-class recall'))
    for w in np.round(np.arange(0,1.0001,0.05),2):
        S=w*Vens+(1-w)*Eens; m=metrics(y,S,K)
        sv=float(sum(m['cm'][c][c] for c in SEVERE[task])/sum(m['sup'][c] for c in SEVERE[task]))
        rows.append(dict(w=float(w),P=m['P'],R=m['R'],F1=m['F1'],AUC=m['AUC'],MCC=m['MCC'],sev=sv,rec=m['rec']))
        print('   %5.2f %7.4f %7.4f %7.4f %7.4f %7.4f   %s'%(w,m['P'],m['R'],m['F1'],m['AUC'],m['MCC'],
              ' '.join('%.3f'%r for r in m['rec'])))
    SW[task]=rows
json.dump(dict(within=OUT,sweep=SW),open('./fuse/ws.json','w'),indent=1)
