import os
import sys, numpy as np; sys.path.insert(0,'./gate')
from common import *
for task in ('bin','g3','g5'):
    d=load(task); y=d['y']; K=d['V'].shape[2]
    Vb=d['V'].mean(0); Eb=d['E'].mean(0)
    rv=prf(y,Vb.argmax(1),K); re_=prf(y,Eb.argmax(1),K)
    # best single each modality (ORACLE selection on this same set)
    vs=[prf(y,d['V'][i].argmax(1),K)['mF'] for i in range(d['V'].shape[0])]
    es=[prf(y,d['E'][i].argmax(1),K)['mF'] for i in range(d['E'].shape[0])]
    bi=int(np.argmax(es)); bvi=int(np.argmax(vs))
    print('== %s  K=%d  N=%d'%(task,K,len(y)))
    print('   video ens  macroF1 %.4f  perclassR %s'%(rv['mF'],[round(x,3) for x in rv['R']]))
    print('   eeg   ens  macroF1 %.4f  perclassR %s'%(re_['mF'],[round(x,3) for x in re_['R']]))
    print('   video single: mean %.4f sd %.4f  best %.4f (%s)'%(np.mean(vs),np.std(vs,ddof=1),vs[bvi],d['vnames'][bvi]))
    print('   eeg   single: mean %.4f sd %.4f  best %.4f (%s)'%(np.mean(es),np.std(es,ddof=1),es[bi],d['enames'][bi]))
    if task=='bin':
        print('   sessions:',len(set(d['sess'])),' subjects:',len(set(d['subj'])))
