import json, os, re, collections
import numpy as np
from scipy import stats
D=os.path.dirname(os.path.abspath(__file__))
S=json.load(open(f'{D}/s1.json'))
STAGE={"Stage_2":1,"Stage_3":2,"Stage_4":3,"Stage_5":4}
def lab(p):
    b=os.path.basename(p)
    if b.startswith('seizure_'):
        m=re.search(r'Stage_[0-9]+',b); return STAGE[m.group()]
    return 0
def subj(p): return re.search(r'Data_(RN\d+)_',p).group(1)

V=set(S['video_val']); E=set(S['eeg_val']); I=set(S['inter'])
pops={'intersection':sorted(I),'video_only':sorted(V-I),'eeg_only':sorted(E-I),
      'video_full':sorted(V),'eeg_full':sorted(E),'aligned':sorted(S['aligned'])}
out={}
for n,ps in pops.items():
    cc=collections.Counter(lab(p) for p in ps)
    out[n]=dict(n=len(ps), classes=[cc[i] for i in range(5)],
                pct_S45=100*(cc[3]+cc[4])/len(ps), pct_S2=100*cc[1]/len(ps),
                n_animals=len(set(map(subj,ps))), n_sessions=len(set('/'.join(p.split('/')[1:3]) for p in ps)))
    print(n, out[n]['n'], out[n]['classes'], 'S4+5 %.2f%%'%out[n]['pct_S45'], 'S2 %.2f%%'%out[n]['pct_S2'],
          'animals',out[n]['n_animals'],'sess',out[n]['n_sessions'])

# class chi2 intersection vs video_only / eeg_only
for other in ('video_only','eeg_only'):
    t=np.array([out['intersection']['classes'], out[other]['classes']],float)
    c2,p,dof,_=stats.chi2_contingency(t)
    print(f'class chi2 inter vs {other}: chi2={c2:.1f} dof={dof} p={p:.3g}')
t=np.array([out[k]['classes'] for k in ('intersection','video_only','eeg_only')],float)
c2,p,dof,_=stats.chi2_contingency(t); print(f'class chi2 3x5: chi2={c2:.1f} dof={dof} p={p:.3g}')

# animal composition
an=lambda ps: collections.Counter(map(subj,ps))
A_i, A_v, A_e, A_vf = an(pops['intersection']), an(pops['video_only']), an(pops['eeg_only']), an(pops['video_full'])
allan=sorted(set(A_i)|set(A_v)|set(A_e))
for other,Ao in (('video_only',A_v),('eeg_only',A_e)):
    t=np.array([[A_i[a] for a in allan],[Ao[a] for a in allan]],float)
    keep=t.sum(0)>0; t=t[:,keep]
    c2,p,dof,_=stats.chi2_contingency(t)
    print(f'animal chi2 inter vs {other}: chi2={c2:.1f} dof={dof} p={p:.3g} (cols={t.shape[1]})')

rows=[]
ni=len(I); nv=len(V)
for a in sorted(allan,key=lambda a:-A_i[a]):
    pi=100*A_i[a]/ni; pv=100*A_vf[a]/nv
    rows.append([a, round(pi,2), round(pv,2), round(pi/pv,2) if pv>0 else None, A_i[a]])
    print(f'  {a}: inter {A_i[a]:5d} ({pi:5.2f}%)  videoval {A_vf[a]:5d} ({pv:5.2f}%)  ratio {pi/pv if pv else float("nan"):.2f}')
inv=lambda c,n: 1.0/sum((v/n)**2 for v in c.values())
print('inverse-Simpson animals: intersection %.2f  video_val %.2f  eeg_val %.2f  aligned %.2f'%(
  inv(A_i,ni), inv(A_vf,nv), inv(an(pops['eeg_full']),len(E)), inv(an(pops['aligned']),len(pops['aligned']))))
print('animals in BOTH parent val sets but zero in intersection:',
      sorted((set(an(pops['video_full']))&set(an(pops['eeg_full'])))-set(A_i)))
print('n animals: video_val',len(A_vf),'eeg_val',len(an(pops['eeg_full'])),'inter',len(A_i),'aligned',len(an(pops['aligned'])))
json.dump(dict(pops=out, per_animal=rows), open(f'{D}/s2.json','w'), indent=1)
