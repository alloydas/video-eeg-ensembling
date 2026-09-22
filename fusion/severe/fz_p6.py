import numpy as np,sys,collections,json
sys.path.insert(0,'.')
from fz_core import *
db=load('bin'); Ebin=amean(db['E'])
for task,K,lo in (('g3',3,2),('g5',5,3)):
    d=load(task); y=d['y']; V=amean(d['V']); E=amean(d['E']); F=amean([V,E])
    pv,pf=V.argmax(1),F.argmax(1); p2=pf.copy(); m=pv>=lo; p2[m]=pv[m]
    print('\n%s per-class F1  (n per class %s)'%(task,np.bincount(y,minlength=K).tolist()))
    for nm,pp in (('video',pv),('fuse50',pf),('R2_veto',p2)):
        f,per=macro_f1(y,pp,K); print('  %-8s macroF1=%.4f  per-class=%s'%(nm,f,[round(x,4) for x in per]))
    print('  preds differing video vs R2_veto: %d / %d (%.2f%%)'%((pv!=p2).sum(),len(y),100*(pv!=p2).mean()))
    print('  preds differing video vs fuse50 : %d / %d (%.2f%%)'%((pv!=pf).sum(),len(y),100*(pv!=pf).mean()))
    ch=collections.Counter(zip(y[pv!=p2].tolist(),pv[pv!=p2].tolist(),p2[pv!=p2].tolist()))
    print('  (true,video,R2) changes:',dict(sorted(ch.items(),key=lambda x:-x[1])))
