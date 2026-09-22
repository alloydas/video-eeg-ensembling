import os
"""Independent reproduction of both pipelines' validation splits. No repo imports."""
import json, os, re, random, collections
import numpy as np

ROOT = os.environ.get('EEG_ROOT','/work/mech-ai-scratch/alloy/EEG')
STAGE = {"Stage_2":1,"Stage_3":2,"Stage_4":3,"Stage_5":4}

def video_items():
    meta = json.load(open(f'{ROOT}/cache_frames/f32s224/index.json'))
    items=[]
    for mp4 in meta['paths']:
        d=os.path.dirname(mp4); b=os.path.basename(d); sess=os.path.basename(os.path.dirname(d))
        subj=re.search(r'Data_(RN\d+)_',mp4).group(1)
        if b.startswith('seizure_'):
            m=re.search(r'Stage_[0-9]+',b)
            if not m or m.group() not in STAGE: continue
            y=STAGE[m.group()]
        else:
            y=0
        items.append((d,y,f'{subj}/{sess}',subj))
    return items

def my_split(sessions, labels_by_sess_items, seed, nc=5, val_frac=0.2):
    """sessions: list of session keys per item; labels: per item. Independent rewrite."""
    uniq = sorted(set(sessions))
    for s in range(seed, seed+500):
        rng = random.Random(s); order = list(uniq); rng.shuffle(order)
        k = max(1, round(len(order)*val_frac))
        val = set(order[:k])
        ctr = collections.Counter(l for l,se in zip(labels_by_sess_items,sessions) if se not in val)
        cva = collections.Counter(l for l,se in zip(labels_by_sess_items,sessions) if se in val)
        if all(ctr[c]>0 for c in range(nc)) and all(cva[c]>0 for c in range(nc)):
            return val, s
    raise SystemExit('none')

vi = video_items()
vsess=[i[2] for i in vi]; vlab=[i[1] for i in vi]
vval, vseed = my_split(vsess, vlab, 49)
vval_clips = sorted(i[0] for i in vi if i[2] in vval)
print(f'VIDEO: items={len(vi)} sessions={len(set(vsess))} used_seed={vseed} val_sessions={len(vval)} val_clips={len(vval_clips)}')

z = np.load(f'{ROOT}/stage_segments_pooled_v3.npz', allow_pickle=True)
csess = [str(x) for x in z['clip_sess']]; clab = z['clip_lab'].tolist()
cpath = [str(x) for x in z['clip_path']]
eval_, eseed = my_split(csess, clab, 49)
eval_clips = sorted(p for p,se in zip(cpath,csess) if se in eval_)
print(f'EEG  : clips={len(cpath)} sessions={len(set(csess))} used_seed={eseed} val_sessions={len(eval_)} val_clips={len(eval_clips)}')

inter = sorted(set(vval_clips) & set(eval_clips))
print(f'INTERSECTION clips={len(inter)}  sessions={len(set(vval)&set(eval_))}')
print('sessions video-not-EEG universe:', sorted(set(vsess)-set(csess)))
print('sessions EEG-not-video universe:', sorted(set(csess)-set(vsess)))
print('val sessions shared:', len(set(vval)&set(eval_)), 'video-only', len(vval-eval_), 'eeg-only', len(eval_-vval))

# aligned universe (EEG using VIDEO session list)
eeg_in_vval = [p for p,se in zip(cpath,csess) if se in vval]
print(f'ALIGNED EEG val clips (EEG clips whose session in video val): {len(eeg_in_vval)}')
print('  subset of video val clips?', set(eeg_in_vval) <= set(vval_clips))
print('  video val clips without EEG:', len(set(vval_clips)-set(eeg_in_vval)))
json.dump(dict(video_val=vval_clips, eeg_val=eval_clips, inter=inter,
               aligned=sorted(eeg_in_vval), vval_sess=sorted(vval), eval_sess=sorted(eval_)),
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'s1.json'),'w'))

# native (discover-order) video val clip list, for runs that stored only `idx`
native = [i[0] for i in vi if i[2] in vval]
json.dump(native, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'s1_native.json'),'w'))
print('native val order written:', len(native))
