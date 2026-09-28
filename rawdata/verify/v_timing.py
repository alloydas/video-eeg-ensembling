# Independent verification: per raw mp4, XML fields + mdhd video-track duration + Noldus trailer (read-only, small reads).
import os, json, struct, re
from concurrent.futures import ThreadPoolExecutor
RAW = "/work/mech-ai/alloydas/EEG/Data"
OUTD = "/work/mech-ai-scratch/alloy/EEG/output/raw_audit/verify"
bx={json.loads(l)['rel']:json.loads(l) for l in open(OUTD+"/boxes.jsonl")}
def find(buf, path):
    # recursive search for box types along path in buffer
    i=0
    while i+8<=len(buf):
        s,t=struct.unpack('>I4s',buf[i:i+8]); t=t.decode('latin1')
        if s<8: return None
        if t==path[0]:
            if len(path)==1: return buf[i:i+s]
            r=find(buf[i+8:i+s], path[1:])
            if r is not None: return r
        i+=s
    return None
def one(rel):
    o=bx[rel]; out={"rel":rel}
    p=os.path.join(RAW,rel)
    xp=p[:-4]+'.XML'
    if not os.path.exists(xp): xp=p[:-4]+'.xml'
    x=open(xp,encoding='latin1').read()
    for tag in ('DSI_utc_start_time','DSI_camera_name','DSI_local_start_time','DSI_sync_offset','Duration','Dsi_VideoFullPath'):
        m=re.search(f'<{tag}>(.*?)</{tag}>',x,re.S); out[tag]=m.group(1).strip() if m else None
    out['ids']=re.findall(r'<id>(.*?)</id>',x)
    moov=[b for b in o['boxes'] if b[0]=='moov']
    if moov:
        _,off,sz=moov[0]
        with open(p,'rb') as f:
            f.seek(off); m=f.read(min(sz,16384))
            mdhd=find(m[8:],['trak','mdia','mdhd'])
            if mdhd:
                ver=mdhd[8]
                if ver==0: ts,dur=struct.unpack('>II',mdhd[20:28])
                else: ts,dur=struct.unpack('>IQ',mdhd[28:40])
                out['mdhd_ts']=ts; out['mdhd_dur']=dur
            elst=find(m[8:],['trak','edts','elst'])
            if elst: out['elst']=elst[8:].hex()
            f.seek(off+sz); t=f.read(4096); out['trailer']=t.hex()
    return out
rels=sorted(bx)
with ThreadPoolExecutor(32) as ex: res=list(ex.map(one, rels))
with open(OUTD+"/timing.jsonl","w") as fo:
    for o in res: fo.write(json.dumps(o)+"\n")
print(len(res))
