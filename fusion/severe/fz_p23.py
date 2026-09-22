import numpy as np, json, collections, sys
sys.path.insert(0,'.')
from fz_core import *
OUT={}
db=load('bin'); Ebin=amean(db['E'])

def build(d,lo):
    V=amean(d['V']); E=amean(d['E'])
    pv=V[:,1:]/np.clip(V[:,1:].sum(1,keepdims=True),1e-300,None)
    g=norm(np.concatenate([Ebin[:,[0]],Ebin[:,[1]]*pv],1))
    return collections.OrderedDict([
        ('video_ens',V),('eeg_ens',E),
        ('fuse_arith_50_50',amean([V,E])),('fuse_arith_70_30',amean([V,E],[.7,.3])),
        ('fuse_arith_90_10',amean([V,E],[.9,.1])),('fuse_geom_50_50',gmean([V,E])),
        ('gate_eegbin_x_vidsev',g)])

for task,K,lo in (('g5',5,3),('g3',3,2)):
    d=load(task); y=d['y']; sess=d['sess']; subj=d['subj']
    P=build(d,lo)
    sev=y>=lo; mild=(y>=1)&(y<lo)
    mixed=[s for s in sorted(set(sess)) if ((sess==s)&sev).sum() and ((sess==s)&mild).sum()]
    npair=sum(int(((sess==s)&sev).sum())*int(((sess==s)&mild).sum()) for s in mixed)
    res={}
    for n,pp in P.items():
        sc=pp[:,lo:].sum(1)
        conc=0.0; per=[]; w=[]
        for s in mixed:
            m=sess==s; a=sc[m&sev]; b=sc[m&mild]
            au=auroc(a,b); per.append(au); w.append(len(a)*len(b)); conc+=au*len(a)*len(b)
        pooled=auroc(sc[sev],sc[mild])
        # animal-clustered bootstrap on the within-session pooled concordance
        r=np.random.default_rng(7); u=sorted(set(subj[sev|mild])); bs=[]
        sidx={s:np.where(sess==s)[0] for s in mixed}
        asess={a:[s for s in mixed if subj[sidx[s][0]]==a] for a in u}
        for _ in range(4000):
            pick=[u[i] for i in r.integers(0,len(u),len(u))]
            num=den=0.0
            for a in pick:
                for s in asess[a]:
                    m=sidx[s]; aa=sc[m][sev[m]]; bb=sc[m][mild[m]]
                    if len(aa)and len(bb): num+=auroc(aa,bb)*len(aa)*len(bb); den+=len(aa)*len(bb)
            if den>0: bs.append(num/den)
        res[n]=dict(within_pooledpairs=float(conc/npair),within_mean_sess=float(np.nanmean(per)),
                    pooled_ignoring_session=float(pooled),
                    within_ci=[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))],
                    per_session={s:float(v) for s,v in zip(mixed,per)})
    # trap control: session identity only
    rs=np.random.default_rng(3); sid=np.array([hash(s)%10007 for s in sess],float)
    res['CONTROL_session_id_only']=dict(pooled_ignoring_session=float(auroc(sid[sev],sid[mild])),
        within_pooledpairs=float(sum(auroc(sid[(sess==s)&sev],sid[(sess==s)&mild])*((sess==s)&sev).sum()*((sess==s)&mild).sum() for s in mixed)/npair),
        within_mean_sess=0.5,pooled_note='session id is constant within session -> within AUROC is 0.5 by construction')
    OUT[task]=dict(n_mixed_sessions=len(mixed),n_pairs=npair,
                   n_sev=int(sev.sum()),n_mild=int(mild.sum()),
                   n_animals_mixed=len(set(subj[np.isin(sess,mixed)&(sev|mild)])),res=res)
    print('\n### %s  mixed sessions=%d  within-session severe/mild pairs=%d  (sev=%d mild=%d, animals=%d)'%(
        task,len(mixed),npair,sev.sum(),mild.sum(),OUT[task]['n_animals_mixed']))
    print('%-24s %8s %-16s %8s %8s'%('variant','WITHIN','[95%CI animal]','meanSess','POOLED'))
    for n,v in res.items():
        if 'within_ci' in v:
            print('%-24s %8.4f [%.4f,%.4f] %8.4f %8.4f'%(n,v['within_pooledpairs'],v['within_ci'][0],v['within_ci'][1],v['within_mean_sess'],v['pooled_ignoring_session']))
        else:
            print('%-24s %8.4f %-16s %8.4f %8.4f'%(n,v['within_pooledpairs'],'-',v['within_mean_sess'],v['pooled_ignoring_session']))
json.dump(OUT,open('fz_p2.json','w'),indent=1); print('\nwrote fz_p2.json')
