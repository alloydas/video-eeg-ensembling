# Independent verification: hash 3 random 1-MiB chunks (not first/last MiB) of both copies of sampled duplicate mp4 names.
import os, json, random, hashlib, collections
from concurrent.futures import ThreadPoolExecutor
RAW = "/work/mech-ai/alloydas/EEG/Data"
OUTD = "/work/mech-ai-scratch/alloy/EEG/output/raw_audit/verify"
rows=[json.loads(l) for l in open(OUTD+"/walk.jsonl")]
byname=collections.defaultdict(list)
for r in rows:
    if r['rel'].endswith('.mp4') and '/seizure_clips/' not in r['rel']:
        byname[os.path.basename(r['rel'])].append(r)
dups=sorted(k for k,v in byname.items() if len(v)==2)
rng=random.Random(12345)
# stratify: 6 per partner pair
bypair=collections.defaultdict(list)
for k in dups: bypair['|'.join(sorted(x['rel'].split('/')[0] for x in byname[k]))].append(k)
sample=[]
for p,ks in sorted(bypair.items()): sample+=rng.sample(ks,min(6,len(ks)))
def chk(k):
    a,b=byname[k]; size=a['size']; M=1<<20
    offs=sorted(rng.randrange(M, max(M+1,size-2*M)) for _ in range(3))
    hs=[]
    for r in (a,b):
        h=hashlib.sha1()
        with open(os.path.join(RAW,r['rel']),'rb') as f:
            for o in offs: f.seek(o); h.update(f.read(M))
        hs.append(h.hexdigest())
    return k, a['rel'].split('/')[0], b['rel'].split('/')[0], hs[0]==hs[1]
with ThreadPoolExecutor(8) as ex: res=list(ex.map(chk, sample))
print(len(res), sum(r[3] for r in res), [r for r in res if not r[3]])
# the moov-less file: is everything after mdat zero? is mdat non-zero?
p=os.path.join(RAW,"RN199/12-12-2023/RN199-RN201.20231213214000.mp4")
with open(p,'rb') as f:
    f.seek(466418981); t=f.read()
    print("tail len",len(t),"nonzero bytes",sum(1 for x in t if x), "first nonzero at", next((i for i,x in enumerate(t) if x),None))
    for o in (1000, 100_000_000, 300_000_000, 466_000_000):
        f.seek(o); c=f.read(4096); print(o, "nonzero frac", sum(1 for x in c if x)/len(c))
