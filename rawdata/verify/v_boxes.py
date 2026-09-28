# Independent verification: walk top-level MP4 boxes of every raw mp4 (reads box headers only).
import os, sys, json, struct
from concurrent.futures import ThreadPoolExecutor
RAW = "/work/mech-ai/alloydas/EEG/Data"
OUTD = "/work/mech-ai-scratch/alloy/EEG/output/raw_audit/verify"
assert OUTD.startswith("/work/mech-ai-scratch/alloy/EEG/output/raw_audit/")
rows=[json.loads(l) for l in open(OUTD+"/walk.jsonl")]
mp4=[r for r in rows if r['rel'].endswith('.mp4') and '/seizure_clips/' not in r['rel']]
def boxes(r):
    p=os.path.join(RAW,r['rel']); size=r['size']; out={"rel":r['rel'],"size":size,"boxes":[]}
    try:
        with open(p,'rb') as f:
            off=0
            while off < size:
                f.seek(off); h=f.read(16)
                if len(h)<8: out['boxes'].append(("short",off,len(h))); break
                bsz,typ=struct.unpack('>I4s',h[:8]); typ=typ.decode('latin1')
                hdr=8
                if bsz==1: bsz=struct.unpack('>Q',h[8:16])[0]; hdr=16
                elif bsz==0: bsz=size-off
                if not typ.isprintable() or bsz<8:
                    out['boxes'].append(("junk",off,typ,bsz)); 
                    f.seek(off); out['junk_head']=f.read(64).hex(); break
                out['boxes'].append((typ,off,bsz))
                if typ=='moov':
                    f.seek(off+hdr); m=f.read(min(bsz-hdr, 4096))
                    i=0
                    while i+8<=len(m):
                        s2,t2=struct.unpack('>I4s',m[i:i+8])
                        if t2==b'mvhd':
                            ver=m[i+8]
                            if ver==0: ts,dur=struct.unpack('>II',m[i+20:i+28])
                            else: ts,dur=struct.unpack('>IQ',m[i+28:i+40])
                            out['mvhd_ts']=ts; out['mvhd_dur']=dur; break
                        if s2<8: break
                        i+=s2
                if off+bsz>size: out['overrun']=off+bsz-size; break
                off+=bsz
            # trailer: if last box is not standard, capture first bytes after moov
            last=out['boxes'][-1]
            if last[0] not in ('moov','mdat','ftyp','free'):
                f.seek(last[1]); out['tail_head']=f.read(96).hex()
    except Exception as e:
        out['err']=repr(e)
    return out
with ThreadPoolExecutor(32) as ex:
    res=list(ex.map(boxes, mp4))
with open(OUTD+"/boxes.jsonl","w") as fo:
    for o in res: fo.write(json.dumps(o)+"\n")
print(len(res))
