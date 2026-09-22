import os
import numpy as np, re, collections, json
D='./'
for t in ('bin','g3','g5'):
    z=np.load(D+'aligned_%s.npz'%t,allow_pickle=True)
    print('==',t,{k:z[k].shape for k in z.files})
z=np.load(D+'aligned_g5.npz',allow_pickle=True)
clips=z['clips']; y=z['y']
print('sample clips:'); [print('  ',c) for c in clips[:3]]
subj=np.array([re.search(r'Data_(RN\d+)',c).group(1) for c in clips])
sess=np.array(['/'.join(c.split('/')[:3]) for c in clips])
print('n subj',len(set(subj)),'n sess',len(set(sess)))
print('vnames',list(np.load(D+'aligned_g5.npz',allow_pickle=True)['vnames'])[:5],'...')
print('enames',list(z['enames'])[:5],'...')
# label consistency across tasks
zb=np.load(D+'aligned_bin.npz',allow_pickle=True); z3=np.load(D+'aligned_g3.npz',allow_pickle=True)
print('clips identical bin/g3/g5:', np.array_equal(zb['clips'],clips), np.array_equal(z3['clips'],clips))
print('g5 counts',np.bincount(y).tolist(),' g3 counts',np.bincount(z3['y']).tolist(),' bin',np.bincount(zb['y']).tolist())
sev=(y>=3)
print('severe n',sev.sum())
print('crosstab g3top vs g5>=3:', int(((z3['y']==2)==sev).all()))
# per-subject severe
cnt=collections.Counter(subj[sev]); tot=collections.Counter(subj)
print('animals total',len(tot),'animals with any severe',len(cnt))
for a in sorted(tot): print('   %-8s n=%4d severe=%3d (s4=%d s5=%d)'%(a,tot[a],cnt.get(a,0),int(((subj==a)&(y==3)).sum()),int(((subj==a)&(y==4)).sum())))
# sessions
ssev=collections.Counter(sess[sev]); stot=collections.Counter(sess)
print('sessions',len(stot),'sessions with severe',len(ssev))
# mixed sessions: contain severe AND mild(seizure but not severe: y in 1,2)
mild=(y>=1)&(y<=2)
mixed=[s for s in stot if ((sess==s)&sev).sum()>0 and ((sess==s)&mild).sum()>0]
print('mixed sessions (severe & mild seizure both present):',len(mixed))
for s in sorted(mixed): print('   %-45s sev=%d mild=%d non=%d'%(s,int(((sess==s)&sev).sum()),int(((sess==s)&mild).sum()),int(((sess==s)&(y==0)).sum())))
