"""Signed residual (recomputed - CSV)/CSV per column, to separate random
requantisation noise (zero-mean) from a systematic difference between the clip
copy and the source the export read.
"""
import json, os, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness2 import iter_sample, unit_scale
from stress2 import canonical

def main():
    animals=sys.argv[1].split(','); days=int(sys.argv[2])
    acc=collections.defaultdict(list); meta=[]
    for d in iter_sample(animals, days_per_animal=days, clips_per_day=3,
                         epochs_per_clip=4, seed=77):
        if d['shift']!=0:      # keep only cleanly aligned clips
            continue
        x,fs,r=d['x'],d['fs'],d['row']
        got,f,P=canonical(x,fs)
        h=d['header']; ch=d['ch']; sc,_=unit_scale(h,ch)
        dq=(h.phys_max[ch]-h.phys_min[ch])/(h.dig_max[ch]-h.dig_min[ch])*sc
        for k,gv in got.items():
            try: rv=float(r[k])
            except (TypeError,ValueError,KeyError): continue
            if abs(rv)<1e-12: continue
            acc[k].append((gv-rv)/abs(rv))
        meta.append(dict(animal=d['animal'], vintage=d['vintage'], dq=dq,
                         std=got['std_uV'], q=dq/got['std_uV'],
                         ll_bias=(got['line_length_uV_per_s']-float(r['line_length_uV_per_s']))/float(r['line_length_uV_per_s']),
                         gamma_bias=(got['gamma_power_uV2']-float(r['gamma_power_uV2']))/float(r['gamma_power_uV2']),
                         delta_bias=(got['delta_power_uV2']-float(r['delta_power_uV2']))/float(r['delta_power_uV2'])))
    print('n epochs (shift==0 only):',len(meta))
    print('%-36s %11s %11s %11s'%('column','median signed','mean signed','median |.|'))
    for k in sorted(acc, key=lambda k: -abs(np.median(acc[k]))):
        a=np.array(acc[k])
        print('%-36s %11.3e %11.3e %11.3e'%(k,np.median(a),a.mean(),np.median(np.abs(a))))
    q=np.array([m['q'] for m in meta]); lb=np.array([m['ll_bias'] for m in meta])
    gb=np.array([m['gamma_bias'] for m in meta])
    print('\nquant_step/std   n   median line_length bias   median gamma bias')
    for lo,hi in [(0,1e-4),(1e-4,3e-4),(3e-4,1e-3),(1e-3,3e-3),(3e-3,1e-2),(1e-2,1)]:
        s=(q>=lo)&(q<hi)
        if s.sum()<5: continue
        print('  [%8.1e,%8.1e) %5d  %12.3e  %12.3e'%(lo,hi,s.sum(),np.median(lb[s]),np.median(gb[s])))
    print('\nby vintage:')
    for v in ('A','B'):
        s=np.array([m['vintage']==v for m in meta])
        if s.sum()<5: continue
        print('  %s n=%d median dq=%.4g uV  median q=%.3e  median ll_bias=%.3e  median gamma_bias=%.3e'%(
            v,s.sum(),np.median([m['dq'] for m in meta if m['vintage']==v]),
            np.median(q[s]),np.median(lb[s]),np.median(gb[s])))
    json.dump(meta,open('out/bias_meta.json','w'))

if __name__=='__main__': main()
