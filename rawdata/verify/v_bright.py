# Sample mean frame brightness at given pts of a raw mp4 (one seek + one decoded frame per sample; read-only).
import sys, subprocess, numpy as np, json
from concurrent.futures import ThreadPoolExecutor
FF="/work/mech-ai/alloy/miniconda3/bin"; RAW="/work/mech-ai/alloydas/EEG/Data"
rel=sys.argv[1]; t0=float(sys.argv[2]); t1=float(sys.argv[3]); step=float(sys.argv[4])
def one(t):
    p=subprocess.run([f"{FF}/ffmpeg","-nostdin","-v","error","-ss",f"{t:.2f}","-i",f"{RAW}/{rel}","-frames:v","1","-vf","scale=80:60,format=gray","-f","rawvideo","pipe:1"],capture_output=True)
    a=np.frombuffer(p.stdout,np.uint8)
    return t, float(a.mean()) if a.size else None
ts=list(np.arange(t0,t1,step))
with ThreadPoolExecutor(8) as ex: r=list(ex.map(one,ts))
print(json.dumps(r))
