# Independent verification: parse every raw EDF header (reads header bytes only) and check size consistency.
import os, json
from concurrent.futures import ThreadPoolExecutor
RAW = "/work/mech-ai/alloydas/EEG/Data"
OUTD = "/work/mech-ai-scratch/alloy/EEG/output/raw_audit/verify"
rows=[json.loads(l) for l in open(OUTD+"/walk.jsonl")]
edfs=[r for r in rows if r['rel'].lower().endswith('.edf')]
def hdr(r):
    p=os.path.join(RAW,r['rel']); o={"rel":r['rel'],"size":r['size']}
    if r['size']==0: o['zero']=True; return o
    with open(p,'rb') as f:
        h=f.read(256)
        ns=int(h[252:256].decode().strip()); o['ns']=ns
        o['startdate']=h[168:176].decode(); o['starttime']=h[176:184].decode()
        o['hdrbytes']=int(h[184:192].decode().strip()); o['reserved']=h[192:236].decode().strip()
        o['nrec']=int(h[236:244].decode().strip()); o['recdur']=float(h[244:252].decode().strip())
        s=f.read(ns*256)
        lab=[s[i*16:(i+1)*16].decode('latin1').strip() for i in range(ns)]
        off=ns*16+ns*80+ns*8
        pmin=[s[off+i*8:off+(i+1)*8].decode().strip() for i in range(ns)]; off+=ns*8
        pmax=[s[off+i*8:off+(i+1)*8].decode().strip() for i in range(ns)]; off+=ns*8
        off+=ns*8*2+ns*80
        nsamp=[int(s[off+i*8:off+(i+1)*8].decode().strip()) for i in range(ns)]
        o['labels']=lab; o['nsamp']=nsamp; o['pmin']=pmin; o['pmax']=pmax
        recbytes=2*sum(nsamp); o['recbytes']=recbytes
        o['expected']=o['hdrbytes']+o['nrec']*recbytes if o['nrec']>=0 else None
        o['nrec_from_size']=(r['size']-o['hdrbytes'])//recbytes; o['partial']=(r['size']-o['hdrbytes'])%recbytes
    return o
with ThreadPoolExecutor(32) as ex: res=list(ex.map(hdr, edfs))
with open(OUTD+"/edfhdr.jsonl","w") as fo:
    for o in res: fo.write(json.dumps(o)+"\n")
print(len(res))
