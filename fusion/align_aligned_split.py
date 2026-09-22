"""Pair video and EEG on the ALIGNED split -- the one fusion should actually use.

Background. The two training pipelines both shuffle sessions with seed 49, but they
shuffle different lists: train_pooled.discover() finds 604 sessions with video, while the
EEG segment cache holds 601 (three RN216 sessions have video but no usable EEG window).
A three-element difference re-permutes everything, so the two 20% prefixes agree on only
56 sessions / 2,830 clips. train_pooled_eeg.video_val_sessions() (P1.1) fixes this by
taking the session universe from the VIDEO item list, making the EEG validation set a
strict SUBSET of video's. output/v3_eegalign/ holds runs trained that way.

Why it matters. The 2,830-clip accidental intersection drops 3 of 18 animals, cuts the
effective animal count from 10.8 to 6.2, holds only 44% of the severe clips, is 6x
enriched in Stage 2, and shifts each modality's own baseline by up to 0.056 -- amplifying
whichever modality already leads. Do not measure fusion on it.

Coverage limit. v3_eegalign currently covers the 3-class task only, with GRU and TCN
(5 seeds each). Detection and 5-class fusion on the aligned split need EEG retraining.

Writes aligned_eegalign_g3.npz with the same layout as align_modalities.py:
    clips (N,)  y (N,)  V (nv,N,K)  vnames (nv,)  E (ne,N,K)  enames (ne,)
"""
import glob, os, sys
import numpy as np

ROOT = os.environ.get('EEG_ROOT', '/work/mech-ai-scratch/alloy/EEG')
VID  = ROOT + '/output/v3_vidseeds'
EEG  = ROOT + '/output/v3_eegalign'
_TOL = 1e-3


def is_double_softmax(P):
    k = P.shape[1]; e = np.e
    return abs(P.min() - 1.0 / (k - 1 + e)) < _TOL and abs(P.max() - e / (k - 1 + e)) < _TOL


def unsquash(P):
    k = P.shape[1]
    lp = np.log(np.clip(P, 1e-12, None))
    q = np.clip(lp + (1.0 - lp.sum(1, keepdims=True)) / k, 0.0, None)
    return q / q.sum(1, keepdims=True)


def clipkey(p):
    p = str(p)
    return (p[:-len('/video.mp4')] if p.endswith('/video.mp4') else p).rstrip('/')


def load(pattern, fname):
    """Load runs, repair double-softmax, and key every member by clip path."""
    out = {}
    for d in sorted(glob.glob(pattern)):
        f = os.path.join(d, fname)
        if not os.path.exists(f):
            continue
        z = np.load(f, allow_pickle=True)
        P = z['probs'].astype(np.float64)
        if is_double_softmax(P):
            P = unsquash(P)
        y = z['y'].astype(int)
        if 'path' in z.files:
            out[os.path.basename(d)] = (np.array([clipkey(x) for x in z['path']]), y, P)
        else:
            # the HuggingFace trainers store no path. Admit them only if their raw label
            # sequence matches a path-bearing reference, which means they are already in
            # reference row order -- the same rule ens_both.py applies.
            out[os.path.basename(d)] = (None, y, P)
    ref = next((n for n in sorted(out) if out[n][0] is not None), None)
    if ref is None:
        return {}
    rp, ry, _ = out[ref]
    for n in list(out):
        if out[n][0] is None:
            if out[n][1].shape == ry.shape and np.array_equal(out[n][1], ry):
                out[n] = (rp, out[n][1], out[n][2])
            else:
                print('  dropped (no path, label order differs):', n)
                del out[n]
    return out


def main(task='g3'):
    V = load(os.path.join(VID, '*_%s_s*' % task), 'val_preds.npz')
    E = load(os.path.join(EEG, '*_%s_s*' % task), 'val_clip_preds.npz')
    if not V or not E:
        raise SystemExit('no runs found -- check EEG_ROOT=%s' % ROOT)
    eref = sorted(E)[0]
    keys = list(E[eref][0])
    kset = set(keys)
    vref = sorted(V)[0]
    if not kset <= set(V[vref][0]):
        raise SystemExit('EEG clips are NOT a subset of the video val set -- wrong run dir?')

    order = np.argsort(np.array(keys))
    clips = np.array(keys)[order]

    def stack(runs, label):
        mats, names = [], []
        for n in sorted(runs):
            p, y, P = runs[n]
            idx = {k: i for i, k in enumerate(p)}
            if not kset <= set(idx):
                print('  dropped (missing clips):', n); continue
            sel = np.array([idx[k] for k in clips])
            mats.append(P[sel]); names.append(n)
        print('  %-6s members kept: %d' % (label, len(names)))
        return np.stack(mats), np.array(names)

    yref = E[eref][1][np.array([list(E[eref][0]).index(k) for k in clips])]
    Vm, vn = stack(V, 'video')
    Em, en = stack(E, 'eeg')

    # the two pipelines must agree on the label of every shared clip
    p, y, _ = V[vref]
    vidx = {k: i for i, k in enumerate(p)}
    yv = y[np.array([vidx[k] for k in clips])]
    agree = int((yv == yref).sum())
    print('  clips %d | labels agree %d/%d | animals %d'
          % (clips.size, agree, clips.size,
             len(set(c.split('Data_')[1].split('_')[0] for c in clips))))
    if agree != clips.size:
        raise SystemExit('label mismatch on the aligned split -- stop and investigate')

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'aligned_eegalign_%s.npz' % task)
    np.savez(out, clips=clips, y=yref, V=Vm, vnames=vn, E=Em, enames=en)
    print('  wrote', out)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'g3')
