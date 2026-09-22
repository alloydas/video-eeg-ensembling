import os
import sys, numpy as np, json, collections
sys.path.insert(0,'./gate')
from common import *

RNG=np.random.default_rng(20260922)
NB=4000

def build(task):
    d=load(task); db=load('bin')
    assert np.array_equal(d['clips'],db['clips'])
    K=d['V'].shape[2]
    out=dict(d); out['K']=K
    out['Vb']=d['V'].mean(0); out['Eb']=d['E'].mean(0)
    out['Vbin']=db['V'].mean(0); out['Ebin']=db['E'].mean(0)
    out['ybin']=db['y']
    # best single EEG binary member, selected by macro-F1 on THIS set (oracle selection; flagged)
    es=[prf(db['y'],db['E'][i].argmax(1),2)['mF'] for i in range(db['E'].shape[0])]
    bi=int(np.argmax(es)); out['Ebin_best']=db['E'][bi]; out['Ebin_best_name']=db['enames'][bi]
    eg=[prf(d['y'],d['E'][i].argmax(1),K)['mF'] for i in range(d['E'].shape[0])]
    gi=int(np.argmax(eg)); out['Eg_best']=d['E'][gi]; out['Eg_best_name']=d['enames'][gi]
    return out

def preds(D):
    y=D['y']; K=D['K']; Vb=D['Vb']; Eb=D['Eb']; Vbin=D['Vbin']; Ebin=D['Ebin']
    ybin=D['ybin']
    P={}
    P['video_only']            = Vb.argmax(1)
    P['eeg_only_ens']          = Eb.argmax(1)
    P['eeg_only_best']         = D['Eg_best'].argmax(1)
    P['gate_eegEns_gradVideo'] = gated(Ebin[:,1]>=0.5, Vb)
    P['gate_eegBest_gradVideo']= gated(D['Ebin_best'][:,1]>=0.5, Vb)
    P['gate_video_gradEEG']    = gated(Vbin[:,1]>=0.5, Eb)
    P['gate_video_gradVideo']  = gated(Vbin[:,1]>=0.5, Vb)
    P['gate_oracle_gradVideo'] = gated(ybin.astype(bool), Vb)
    P['gate_oracle_gradEEG']   = gated(ybin.astype(bool), Eb)
    P['fusion_w0.5']           = (0.5*Vb+0.5*Eb).argmax(1)
    # best-w fusion (swept on this same set -> oracle; flagged)
    ws=np.arange(0.0,1.001,0.05); best=(-1,None,None)
    for w in ws:
        p=(w*Vb+(1-w)*Eb).argmax(1); f=prf(y,p,K)['mF']
        if f>best[0]: best=(f,w,p)
    P['fusion_bestw']=best[2]; D['bestw']=float(best[1])
    # gate + fusion grading
    P['gate_eegEns_gradFusion']= gated(Ebin[:,1]>=0.5, 0.5*Vb+0.5*Eb)
    return P

def within_session(y,p,sess,K,min_n=10):
    rows=[]
    for s in sorted(set(sess)):
        m=sess==s
        if m.sum()<min_n: continue
        if len(set(y[m].tolist()))<2: continue
        rows.append((s,int(m.sum()),prf(y[m],p[m],K)['mF']))
    return rows

def cluster_boot(y,preds_dict,subj,K,sev_idx,nb=NB):
    """cluster bootstrap over animals; returns per-pipeline arrays of macroF1 and severe recall"""
    us=np.array(sorted(set(subj))); idx={s:np.where(subj==s)[0] for s in us}
    names=list(preds_dict)
    mF={n:np.empty(nb) for n in names}; sR={n:np.full(nb,np.nan) for n in names}
    for b in range(nb):
        pick=RNG.choice(len(us),len(us),replace=True)
        ii=np.concatenate([idx[us[j]] for j in pick])
        yb=y[ii]
        for n in names:
            pb=preds_dict[n][ii]; r=prf(yb,pb,K)
            mF[n][b]=r['mF']
            ns=sum(r['sup'][c] for c in sev_idx)
            if ns>0:
                tp=sum(int(((pb==c)&(yb==c)).sum()) for c in sev_idx)
                # severe recall = fraction of severe-truth clips predicted into ANY severe class
                tp=int(np.isin(pb[np.isin(yb,sev_idx)],sev_idx).sum())
                sR[n][b]=tp/ns
    return mF,sR

def main():
    OUT={}
    for task in ('g3','g5'):
        D=build(task); K=D['K']; y=D['y']
        sev_idx=[2] if task=='g3' else [3,4]
        P=preds(D)
        res={}
        for n,p in P.items():
            r=prf(y,p,K)
            sm=np.isin(y,sev_idx)
            r['sev_recall']=float(np.isin(p[sm],sev_idx).mean()); r['sev_n']=int(sm.sum())
            ws=within_session(y,p,D['sess'],K)
            r['within_sess_mF']=float(np.mean([x[2] for x in ws])); r['within_sess_n']=len(ws)
            res[n]=r
        mF,sR=cluster_boot(y,P,D['subj'],K,sev_idx)
        ref='video_only'
        for n in P:
            res[n]['mF_ci']=[float(np.percentile(mF[n],2.5)),float(np.percentile(mF[n],97.5))]
            res[n]['sevR_ci']=[float(np.nanpercentile(sR[n],2.5)),float(np.nanpercentile(sR[n],97.5))]
            dF=mF[n]-mF[ref]; dS=sR[n]-sR[ref]
            res[n]['d_mF_vs_video']=float(np.mean(dF))
            res[n]['d_mF_ci']=[float(np.percentile(dF,2.5)),float(np.percentile(dF,97.5))]
            res[n]['p_mF_gt_video']=float(np.mean(dF>0))
            ok=~np.isnan(dS)
            res[n]['d_sevR_vs_video']=float(np.nanmean(dS))
            res[n]['d_sevR_ci']=[float(np.nanpercentile(dS,2.5)),float(np.nanpercentile(dS,97.5))]
            res[n]['p_sevR_gt_video']=float(np.mean(dS[ok]>0))
            res[n]['p_beats_both']=float(np.mean((dF[ok]>0)&(dS[ok]>0)))
        OUT[task]=dict(res=res, bestw=D['bestw'],
                       eeg_bin_best=str(D['Ebin_best_name']), eeg_grad_best=str(D['Eg_best_name']),
                       counts=np.bincount(y,minlength=K).tolist())
    json.dump(OUT,open('./gate/pipelines.json','w'),indent=1)
    for task in OUT:
        print('\n===== %s  counts=%s  bestw=%.2f  eegbin_best=%s'%(task,OUT[task]['counts'],OUT[task]['bestw'],OUT[task]['eeg_bin_best']))
        print('%-26s %7s %7s %7s %7s %8s %8s  %s'%('pipeline','mP','mR','mF1','acc','sevR','winSess','per-class recall'))
        for n,r in OUT[task]['res'].items():
            print('%-26s %7.4f %7.4f %7.4f %7.4f %8.4f %8.4f  %s'%(n,r['mP'],r['mR'],r['mF'],r['acc'],r['sev_recall'],r['within_sess_mF'],[round(x,3) for x in r['R']]))
main()
