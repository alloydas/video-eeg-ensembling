# Parse the video track's stts (sample durations) and stss (keyframes) from the moov of given raw mp4s (read-only).
import sys, struct, json, os
RAW="/work/mech-ai/alloydas/EEG/Data"
def boxes(buf):
    i=0
    while i+8<=len(buf):
        s,t=struct.unpack('>I4s',buf[i:i+8]); hdr=8
        if s==1: s=struct.unpack('>Q',buf[i+8:i+16])[0]; hdr=16
        yield t.decode('latin1'), buf[i+hdr:i+s]; i+=s
def find(buf,path):
    for t,b in boxes(buf):
        if t==path[0]:
            return b if len(path)==1 else find(b,path[1:])
for rel in sys.argv[1:]:
    p=os.path.join(RAW,rel); size=os.path.getsize(p)
    with open(p,'rb') as f:
        off=0
        while off<size:
            f.seek(off); h=f.read(16); s,t=struct.unpack('>I4s',h[:8])
            if s==1: s=struct.unpack('>Q',h[8:16])[0]
            if t==b'moov': f.seek(off); moov=f.read(s); break
            off+=s
    stbl=find(moov[8:],['trak','mdia','minf','stbl']); mdhd=find(moov[8:],['trak','mdia','mdhd'])
    ts=struct.unpack('>I',mdhd[12:16])[0]
    stts=find(stbl,['stts']); n=struct.unpack('>I',stts[4:8])[0]
    ent=[struct.unpack('>II',stts[8+8*k:16+8*k]) for k in range(n)]
    stss=find(stbl,['stss']); nk=struct.unpack('>I',stss[4:8])[0] if stss else 0
    ks=[struct.unpack('>I',stss[8+4*k:12+4*k])[0] for k in range(nk)]
    import collections
    kd=collections.Counter(b-a for a,b in zip(ks,ks[1:]))
    # cumulative time and large deltas
    t=0; big=[]; nsamp=0
    for c,d in ent:
        if d/ts>0.2: big.append((round(t/ts,2), c, round(d/ts,3)))
        t+=c*d; nsamp+=c
    print(rel, "timescale",ts,"samples",nsamp,"stts entries",n,"total",round(t/ts,2),"deltas>0.2s:",big[:10], "keyframe spacing",kd.most_common(3), "nkey",nk)
