import numpy as np, json, collections, sys
sys.path.insert(0,'.')
from fz_core import *
db=load('bin'); Ebin=amean(db['E'])
OUT={}
for task,K,lo in (('g5',5,3),('g3',3,2)):
    d=load(task); y=d['y']; subj=d['subj']; u=np.array(sorted(set(subj)))
    V=amean(d['V']); E=amean(d['E'])
    pvsz=V[:,1:]/np.clip(V[:,1:].sum(1,keepdims=True),1e-300,None)
    GATE=norm(np.concatenate([Ebin[:,[0]],Ebin[:,[1]]*pvsz],1))
    # ---- parameterised rule families ----
    def rule_gate_unc(tau,w):           # fuse only when video is uncertain
        F=amean([V,E],[1-w,w]); p=V.argmax(1).copy()
        m=V.max(1)<tau; p[m]=F[m].argmax(1); return p
    def rule_sevfloor(w):               # fuse, but video has a veto on the severe band
        F=amean([V,E],[1-w,w]); p=F.argmax(1); mv=V.argmax(1)
        m=mv>=lo; p[m]=mv[m]; return p
    def rule_wfuse(w):
        return amean([V,E],[1-w,w]).argmax(1)
    def rule_gate_only():
        return GATE.argmax(1)
    TAUS=np.round(np.arange(0.30,1.001,0.025),3); WS=np.round(np.arange(0.0,0.55,0.05),3)
    fams={
     'R1_uncertainty_gate':(lambda g:rule_gate_unc(g[0],g[1]),[(t,w) for t in TAUS for w in WS if w>0]),
     'R2_video_veto_on_severe':(lambda g:rule_sevfloor(g[0]),[(w,) for w in WS if w>0]),
     'R0_plain_weighted_fusion':(lambda g:rule_wfuse(g[0]),[(w,) for w in WS if w>0]),
    }
    def score(yy,pp):
        f,_=macro_f1(yy,pp,K); s,_=sev_recall(yy,pp,lo); return f,(0.0 if np.isnan(s) else s)
    res={}
    pv=V.argmax(1); fbase,sbase=score(y,pv)
    res['video_ens (reference)']=dict(macroF1=fbase,sev_recall=sbase,params='-',mode='fixed')
    res['gate_eegbin_x_vidsev']=dict(zip(('macroF1','sev_recall'),score(y,rule_gate_only())),params='none',mode='fixed')
    for fam,(fn,grid) in fams.items():
        # (a) oracle: best grid point on the WHOLE set -- optimistic, reported for contrast
        best=max(grid,key=lambda g:sum(score(y,fn(g))))
        of,os_=score(y,fn(best))
        # (b) honest: leave-one-animal-out selection of the grid point
        ph=np.empty_like(pv); chosen={}
        for a in u:
            te=subj==a; tr=~te
            g=max(grid,key=lambda g:sum(score(y[tr],fn(g)[tr])))
            ph[te]=fn(g)[te]; chosen[a]=list(map(float,g))
        hf,hs=score(y,ph)
        res[fam]=dict(oracle_params=list(map(float,best)),oracle_macroF1=of,oracle_sev_recall=os_,
                      loao_macroF1=hf,loao_sev_recall=hs,loao_chosen=chosen,mode='tuned')
        res[fam]['_pred_loao']=ph
    # ---- joint paired bootstrap: P(beats video on BOTH) ----
    preds={'eeg_ens':E.argmax(1),'fuse_arith_50_50':amean([V,E]).argmax(1),
           'gate_eegbin_x_vidsev':rule_gate_only()}
    for fam in fams: preds['%s (LOAO)'%fam]=res[fam].pop('_pred_loao')
    r=np.random.default_rng(11); idx={a:np.where(subj==a)[0] for a in u}
    joint={n:dict(both=0,f1=0,sev=0,tot=0,dF=[],dS=[]) for n in preds}
    for _ in range(4000):
        pick=r.integers(0,len(u),len(u)); ii=np.concatenate([idx[u[j]] for j in pick]); yy=y[ii]
        if (yy>=lo).sum()==0: continue
        bf,bs=score(yy,pv[ii])
        for n,pp in preds.items():
            f,s=score(yy,pp[ii]); j=joint[n]
            j['tot']+=1; j['dF'].append(f-bf); j['dS'].append(s-bs)
            if f>bf: j['f1']+=1
            if s>bs: j['sev']+=1
            if f>bf and s>bs: j['both']+=1
    jj={}
    for n,j in joint.items():
        jj[n]=dict(P_beats_macroF1=j['f1']/j['tot'],P_beats_sev_recall=j['sev']/j['tot'],
                   P_beats_BOTH=j['both']/j['tot'],nboot=j['tot'],
                   d_f1_ci=[float(np.percentile(j['dF'],2.5)),float(np.percentile(j['dF'],97.5))],
                   d_sev_ci=[float(np.percentile(j['dS'],2.5)),float(np.percentile(j['dS'],97.5))])
    OUT[task]=dict(res={k:v for k,v in res.items()},joint=jj,video_ref=dict(macroF1=fbase,sev_recall=sbase))
    print('\n### %s   video reference: macroF1=%.4f  sevRecall=%.4f (%d/%d)'%(task,fbase,sbase,int(round(sbase*(y>=lo).sum())),int((y>=lo).sum())))
    for n,v in res.items():
        if v.get('mode')=='fixed':
            print('  %-28s FIXED    macroF1=%.4f (%+.4f)  sevRec=%.4f (%+.4f)'%(n,v['macroF1'],v['macroF1']-fbase,v['sev_recall'],v['sev_recall']-sbase))
        else:
            print('  %-28s ORACLE   macroF1=%.4f (%+.4f)  sevRec=%.4f (%+.4f)   params=%s'%(n,v['oracle_macroF1'],v['oracle_macroF1']-fbase,v['oracle_sev_recall'],v['oracle_sev_recall']-sbase,v['oracle_params']))
            print('  %-28s LOAO     macroF1=%.4f (%+.4f)  sevRec=%.4f (%+.4f)'%('',v['loao_macroF1'],v['loao_macroF1']-fbase,v['loao_sev_recall'],v['loao_sev_recall']-sbase))
    print('  --- animal-clustered bootstrap, P(strictly beats video) ---')
    for n,v in jj.items():
        print('    %-32s P(F1)=%.3f P(sev)=%.3f P(BOTH)=%.3f  dF1[%+.4f,%+.4f] dSev[%+.4f,%+.4f]'%(
            n,v['P_beats_macroF1'],v['P_beats_sev_recall'],v['P_beats_BOTH'],*v['d_f1_ci'],*v['d_sev_ci']))
json.dump(OUT,open('fz_p4.json','w'),indent=1,default=str); print('\nwrote fz_p4.json')
