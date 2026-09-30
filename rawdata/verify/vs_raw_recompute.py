#!/usr/bin/env python3
"""Independent raw-file recomputation of step-A clips (read-only; writes only under ttg_scan/verify/).

For a seeded sample of step-A clips (non-seizure flagged / unflagged, every seizure stratum, the three partners):
  * own packet index of the raw mp4 (PyAV demux, not ffprobe), compared with stepA/packets;
  * own r0: the raw frame (any frame within +-64 of the seek) whose box crop best matches clip frame 0 (NCC);
  * n = the clip's header frame count (cv2), and the actual decoded frame count;
  * own arm E frames (r0 + linspace(0, n-1, 16)), arm K frames (nearest keyframe, ties earlier), arm C (the
    f16s224 row), compared bit for bit with stepA/frames_{E,K,C}.npy; per-frame alignment of E against C;
  * own EEG: own EDF reader on the raw EDF at the clip start, own resample / windows / z-score, compared with the
    clip EDF and with stepA/eeg_R.npz;
  * the networks run on CPU in fp32 on these inputs (C, own K, own E; own EEG), compared with stepA/preds.npz;
  * the gate decision (G_all from gates.json, re-fitted independently in vs_gates_stepa.py) for all 9 pairs.
"""
import csv
import datetime as dt
import json
import os
import re
import sys
import time

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import numpy as np  # noqa: E402

E = "/work/mech-ai-scratch/alloy/EEG"
OUT = f"{E}/output/ttg_scan/verify"
SCAN = f"{E}/output/ttg_scan"
RAW = "/work/mech-ai/alloydas/EEG/Data"
SEEDS = (1, 2, 3)
os.makedirs(OUT, exist_ok=True)


def animal(p):
    return re.search(r"Data_(RN\d+)_cropped", p).group(1)


# ---------------------------------------------------------------- own EDF reader
def edf_hdr(path):
    with open(path, "rb") as f:
        h = f.read(256)
        ns = int(h[252:256])
        s = f.read(ns * 256)
    def fld(off, w):
        return [s[off * ns + i * w: off * ns + (i + 1) * w].decode("latin1").strip() for i in range(ns)]
    labels = fld(0, 16)
    o = 16 + 80
    dim = fld(o, 8); o += 8
    pmin = [float(x) for x in fld(o, 8)]; o += 8
    pmax = [float(x) for x in fld(o, 8)]; o += 8
    dmin = [float(x) for x in fld(o, 8)]; o += 8
    dmax = [float(x) for x in fld(o, 8)]; o += 8
    o += 80
    nsamp = [int(x) for x in fld(o, 8)]
    hb = int(h[184:192])
    rdur = float(h[244:252])
    d, m, y = (int(x) for x in h[168:176].decode().split("."))
    H, M, S = (int(x) for x in h[176:184].decode().split("."))
    start = dt.datetime(2000 + y if y < 85 else 1900 + y, m, d, H, M, S)
    recb = 2 * sum(nsamp)
    nrec_file = (os.path.getsize(path) - hb) // recb
    return dict(labels=labels, dim=dim, pmin=pmin, pmax=pmax, dmin=dmin, dmax=dmax, nsamp=nsamp, hb=hb,
                rdur=rdur, start=start, recb=recb, nrec_file=nrec_file, path=path,
                off=[sum(nsamp[:i]) for i in range(ns)])


def edf_read(h, ch, s0, n):
    ns_ = h["nsamp"][ch]
    out = np.empty(n, np.float64)
    k = 0
    with open(h["path"], "rb") as f:
        while k < n:
            s = s0 + k
            r, o = divmod(s, ns_)
            if r >= h["nrec_file"]:
                return None
            take = min(ns_ - o, n - k)
            f.seek(h["hb"] + r * h["recb"] + 2 * (h["off"][ch] + o))
            out[k:k + take] = np.frombuffer(f.read(2 * take), "<i2")
            k += take
    g = (h["pmax"][ch] - h["pmin"][ch]) / (h["dmax"][ch] - h["dmin"][ch])
    return (out - h["dmin"][ch]) * g + h["pmin"][ch], out


def pick_channel(labels):
    for key in ("EEG", "ECG"):
        for i, l in enumerate(labels):
            if key in l.upper():
                return i
    return None


def eeg_windows(sig):
    from scipy.signal import resample_poly
    x = resample_poly(sig, 1, 8).astype(np.float32)
    starts = range(0, len(x) - 750 + 1, 375)
    w = np.stack([x[s:s + 750] for s in starts]).astype(np.float32)
    mu = w.mean(1, keepdims=True)
    sd = w.std(1, keepdims=True)
    sd[sd == 0] = 1
    return ((w - mu) / sd).astype(np.float32)


# ---------------------------------------------------------------- video helpers
def packets(path):
    import av
    c = av.open(path)
    s = c.streams.video[0]
    pts, key = [], []
    for pk in c.demux(s):
        if pk.pts is None:
            continue
        pts.append(int(pk.pts))
        key.append(bool(pk.is_keyframe))
    tb = float(s.time_base)
    c.close()
    o = np.argsort(pts, kind="stable")
    return np.array(pts)[o], np.array(key)[o], tb


def decode_range(path, pts, i0, i1):
    """{frame index: BGR} for file frames i0..i1 (sequential decode from the keyframe at or before i0)."""
    import av
    want = {int(pts[i]): i for i in range(i0, i1 + 1)}
    c = av.open(path)
    s = c.streams.video[0]
    c.seek(int(pts[i0]), stream=s, backward=True, any_frame=False)
    out = {}
    for fr in c.decode(s):
        if fr.pts in want:
            out[want[fr.pts]] = fr.to_ndarray(format="bgr24")
        if fr.pts is not None and fr.pts >= pts[i1]:
            break
    c.close()
    return out


def crop224(img, box):
    import cv2
    x, y, w, h = box
    c = cv2.cvtColor(img[y:y + h, x:x + w], cv2.COLOR_BGR2RGB)
    return cv2.resize(c, (224, 224), interpolation=cv2.INTER_AREA)


def ncc(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    a -= a.mean(); b -= b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def linspace16(n):
    return np.linspace(0, n - 1, 16).round().astype(int) if n >= 16 else np.r_[np.arange(n), np.full(16 - n, n - 1)]


def snap(pos, kf):
    out = []
    for p in pos:
        d = np.abs(kf - p)
        m = d.min()
        out.append(int(kf[d == m].min()))          # ties -> earlier keyframe
    return np.array(out)


def main():
    import cv2
    import torch
    torch.set_num_threads(int(os.environ.get("NT", "8")))
    sys.path.insert(0, "/work/mech-ai-scratch/alloy/video-eeg-ensembling/grader")
    import ttg_common  # noqa: F401
    import train_grader as TG
    import train_pooled as tp
    import train_pooled_eeg as tpe

    it = json.load(open(f"{SCAN}/stepA/items.json"))
    items = it["items"]
    eeg_rows = {r["key"]: r for r in it["eeg"]}
    P = np.load(f"{SCAN}/stepA/preds.npz")
    gj = json.load(open(f"{SCAN}/s0/gates.json"))
    fK = np.load(f"{SCAN}/stepA/frames_K.npy", mmap_mode="r")
    fE = np.load(f"{SCAN}/stepA/frames_E.npy", mmap_mode="r")
    fC = np.load(f"{SCAN}/stepA/frames_C.npy", mmap_mode="r")
    eR = np.load(f"{SCAN}/stepA/eeg_R.npz")
    clips = {f"data/Data_{r['animal']}_cropped/{r['session']}/{r['clip_dir']}/video.mp4": r
             for r in csv.DictReader(open(f"{E}/output/raw_audit/align/clips.csv"))}
    boxes = {(r["animal"], r["camera"]): tuple(int(r[k]) for k in "xywh")
             for r in csv.DictReader(open(f"{E}/output/raw_audit/align/crop_boxes.csv"))}
    idx16 = json.load(open(f"{E}/cache_frames/f16s224/index.json"))
    row16 = {p: i for i, p in enumerate(idx16["paths"])}
    mm16 = np.memmap(f"{E}/cache_frames/f16s224/frames.u8", np.uint8, "r",
                     shape=(idx16["n"], 16, 224, 224, 3))
    # stored video + EEG predictions (for the gate)
    stV, stE = {}, {}
    for f in range(5):
        for s in SEEDS:
            z = np.load(f"{E}/output/ttg_recut/vsubj_patched/x3dfix_dual_s{s}_fold{f}/val_ep12.npz", allow_pickle=True)
            pos = {str(p): i for i, p in enumerate(z["path"])}
            stV[(s, f)] = (pos, z["probs_g3"], z["probs_g5"])
            z = np.load(f"{E}/output/ttg_eeg/subject/tcn_bin_fold{f}_s{s}/val_clip_ep30.npz", allow_pickle=True)
            stE[(s, f)] = {str(p): float(v) for p, v in zip(z["path"], z["p_logmean"])}

    # ---- sample
    def gflag(Pv, pe, coef):
        lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        q = 1 / (1 + np.exp(-(coef[0] * lg(1 - Pv[:, 0]) + coef[1] * lg(pe) + coef[2])))
        sz = Pv[:, 1:] / np.clip(Pv[:, 1:].sum(1, keepdims=True), 1e-12, None)
        return np.c_[1 - q, q[:, None] * sz].argmax(1), q

    ky = [x["key"] for x in items]
    y5 = np.array([x["y5"] for x in items])
    anyflag = np.zeros(len(items), bool)
    for arm in "CKE":
        for vs in SEEDS:
            pa = P[f"{arm}|g5|s{vs}"].astype(np.float64)
            for es in SEEDS:
                pe = np.array([stE[(es, x["fold"])][x["key"][:-10]] for x in items])
                c, _ = gflag(pa, pe, gj["gates"][f"g5|x3dfix_dual_s{vs}_foldF|tcn_bin_foldF_s{es}"]["all"])
                anyflag |= c > 0
    rng = np.random.default_rng(20260930)
    sel = []
    ns_f = np.flatnonzero((y5 == 0) & anyflag)
    ns_u = np.flatnonzero((y5 == 0) & ~anyflag)
    sel += list(rng.choice(ns_f, min(6, len(ns_f)), replace=False))
    sel += list(rng.choice(ns_u, 6, replace=False))
    for y in (1, 2, 3, 4):
        sel += list(rng.choice(np.flatnonzero(y5 == y), 6, replace=False))
    for pa_ in ("RN199", "RN204", "RN242"):
        have = [i for i in sel if animal(ky[i]) == pa_]
        pool = [i for i in range(len(items)) if animal(ky[i]) == pa_ and i not in sel]
        while len(have) < 3 and pool:
            j = int(rng.choice(pool)); pool.remove(j); sel.append(j); have.append(j)
    extra = [i for i, k in enumerate(ky) if "RN227" in k and "seizure_110" in k]
    sel += [i for i in extra if i not in sel]
    sel = [int(i) for i in sel]
    print(f"{len(sel)} clips; animals {sorted(set(animal(ky[i]) for i in sel))}", flush=True)

    # ---- models (CPU fp32)
    need_f = sorted({items[i]["fold"] for i in sel})
    VM, EM = {}, {}
    for f in need_f:
        for s in SEEDS:
            d = f"{E}/output/ttg_vsubj/x3dfix_dual_s{s}_fold{f}"
            ck = torch.load(f"{d}/last.pt", map_location="cpu", weights_only=False)
            assert ck["state"]["next_epoch"] == 13 and ck["run_key"]["epochs"] == 12 and ck["run_key"]["fix_x3d"]
            m = TG.GraderNet("x3d", ["g3", "g5"], True, 16, 224, pretrained=False, logit_bound=0.0)
            m.load_state_dict(ck["model"], strict=True)
            VM[(s, f)] = m.eval()
            d = f"{E}/output/ttg_eeg/subject/tcn_bin_fold{f}_s{s}"
            fin = torch.load(f"{d}/final.pt", map_location="cpu", weights_only=False)
            assert int(fin["epoch"]) == 30
            em = tpe.build_model("tcn", 2, fin["run_key"]["hidden"])
            with torch.no_grad():
                em(torch.zeros(2, 750, 1))
            em.load_state_dict(fin["model"], strict=True)
            EM[(s, f)] = em.eval()
    dev = torch.device("cpu")

    def vpred(frames, s, f):
        x = torch.from_numpy(np.ascontiguousarray(frames[None].transpose(0, 4, 1, 2, 3)))
        x = tp.norm_batch(x, dev)
        with torch.no_grad():
            o = VM[(s, f)](x)
        return {t: torch.softmax(o[t].float(), 1).numpy()[0].astype(np.float64) for t in ("g3", "g5")}

    def epred(w, s, f):
        with torch.no_grad():
            p = torch.softmax(EM[(s, f)](torch.from_numpy(w).unsqueeze(-1)), 1).numpy()
        q = np.exp(np.log(np.clip(p, 1e-9, 1)).mean(0))
        return float((q / q.sum())[1])

    rows = []
    pk_cache = {}
    for n_, i in enumerate(sel):
        t0 = time.time()
        x = items[i]
        k = x["key"]
        a_ = animal(k)
        f = x["fold"]
        c = clips[k]
        rel = c["raw_video"]
        rp = os.path.join(RAW, rel)
        box = boxes[(a_, c["raw_camera"])]
        R = dict(i=i, key=k, animal=a_, y5=x["y5"], fold=f, box_matches_items=list(box) == list(x["box"]))
        # clip frame count and frame 0
        full = os.path.join(E, "data_full", k[5:])
        cap = cv2.VideoCapture(full)
        n_hdr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, fr0 = cap.read()
        n_dec = 1
        while cap.grab():
            n_dec += 1
        cap.release()
        g0 = cv2.cvtColor(fr0, cv2.COLOR_BGR2GRAY)
        R.update(n_hdr=n_hdr, n_decoded=n_dec, n_matches_items=n_hdr == x["n"])
        # packets
        if rel not in pk_cache:
            pk_cache[rel] = packets(rp)
        pts, key, tb = pk_cache[rel]
        zp = np.load(f"{SCAN}/stepA/packets/{os.path.basename(rel).replace('.mp4', '.npz')}")
        R["packets_match_ffprobe"] = bool(np.array_equal(zp["pts"], pts) and np.array_equal(zp["key"], key))
        kf = np.flatnonzero(key)
        seek = float(c["info_seek_s"]) if c["info_seek_s"] else float(c["seek_pred_s"])
        js = int(np.argmin(np.abs(pts * tb - seek)))
        lo, hi = max(0, js - 64), min(len(pts) - 1, js + 64)
        pos_end = None
        # decode from lo to (candidate r0 + n + 40)
        last = min(len(pts) - 1, hi + n_hdr + 40)
        frames = decode_range(rp, pts, lo, last)
        x0, y0, w0, h0 = box
        scores = {j: ncc(cv2.cvtColor(frames[j][y0:y0 + h0, x0:x0 + w0], cv2.COLOR_BGR2GRAY), g0)
                  for j in range(lo, hi + 1) if j in frames}
        r0 = max(scores, key=scores.get)
        srt = sorted(scores.values())
        R.update(r0_own=r0, r0_items=x["r0"], r0_match=r0 == x["r0"], ncc_best=scores[r0], ncc_second=srt[-2],
                 r0_is_keyframe=bool(key[r0]))
        pos = r0 + linspace16(n_hdr)
        pk = snap(pos, kf)
        R.update(pos_match=pos.tolist() == x["pos_exact"], posk_match=pk.tolist() == x["pos_key"],
                 max_snap=int(np.abs(pk - pos).max()))
        need_last = int(max(pos.max(), pk.max()))
        if need_last > last:
            frames.update(decode_range(rp, pts, last, need_last))
        myE = np.stack([crop224(frames[j], box) for j in pos])
        myK = np.stack([crop224(frames[j], box) for j in pk])
        myC = np.asarray(mm16[row16[k]])
        R.update(E_bit_identical=bool(np.array_equal(myE, fE[i])), K_bit_identical=bool(np.array_equal(myK, fK[i])),
                 C_bit_identical=bool(np.array_equal(myC, fC[i])),
                 E_vs_C_mae=float(np.abs(myE.astype(np.int16) - myC.astype(np.int16)).mean()),
                 K_vs_C_mae=float(np.abs(myK.astype(np.int16) - myC.astype(np.int16)).mean()))
        # frame alignment of E against C: best offset per frame in -3..+3
        offs = []
        for t_, j in enumerate(pos):
            best = None
            for o in range(-3, 4):
                if (j + o) in frames:
                    v = np.abs(crop224(frames[j + o], box).astype(np.int16) - myC[t_].astype(np.int16)).mean()
                    if best is None or v < best[1]:
                        best = (o, v)
            offs.append(best[0])
        R["E_best_offset_per_frame"] = offs
        # networks on CPU
        preds = {}
        for arm, FR in (("C", myC), ("K", myK), ("E", myE)):
            for s in SEEDS:
                pr = vpred(FR, s, f)
                for t in ("g3", "g5"):
                    preds[(arm, t, s)] = pr[t]
        dmax = {}
        agree = {}
        for (arm, t, s), v in preds.items():
            ref = P[f"{arm}|{t}|s{s}"][i].astype(np.float64)
            dmax.setdefault(arm, 0.0)
            dmax[arm] = max(dmax[arm], float(np.abs(v - ref).max()))
            agree.setdefault(arm, []).append(int(v.argmax()) == int(ref.argmax()))
        R["cpu_vs_gpu_max_dp"] = dmax
        R["cpu_vs_gpu_argmax_agree"] = {a: f"{sum(v)}/{len(v)}" for a, v in agree.items()}
        # EEG from the raw EDF
        hr = edf_hdr(os.path.join(RAW, c["raw_edf"]))
        ch = pick_channel(hr["labels"])
        hc = edf_hdr(os.path.join(E, "data_full", k[5:-10], "eeg.edf"))
        chc = pick_channel(hc["labels"])
        nclip = hc["nrec_file"] * hc["nsamp"][chc]
        cs = dt.datetime.fromisoformat(c["info_clip_start"])
        s0 = int(round((cs - hr["start"]).total_seconds() * 1000))
        sig, dig = edf_read(hr, ch, s0, nclip)
        csig, _ = edf_read(hc, chc, 0, nclip)
        runs = np.diff(np.flatnonzero(np.r_[True, np.diff(dig) != 0, True])).max()
        w = eeg_windows(sig)
        er = eeg_rows[k]
        seg = eR["segs"][eR["offsets"][i]:eR["offsets"][i + 1]]
        R.update(eeg_label=hr["labels"][ch], eeg_fs=hr["nsamp"][ch] / hr["rdur"], s0_own=s0, s0_items=er["s0"],
                 corr_raw_vs_clip=float(np.corrcoef(sig, csig)[0, 1]), longest_run=int(runs),
                 windows=len(w), windows_match_stepA=bool(w.shape == seg.shape and np.allclose(w, seg, atol=1e-5)),
                 windows_max_abs_diff=float(np.abs(w - seg).max()) if w.shape == seg.shape else None)
        pe_own = {s: epred(w, s, f) for s in SEEDS}
        R["pe_own"] = pe_own
        R["pe_vs_stepA_R_max"] = max(abs(pe_own[s] - float(P[f"R|s{s}"][i])) for s in SEEDS)
        R["pe_vs_stored_max"] = max(abs(pe_own[s] - stE[(s, f)][k[:-10]]) for s in SEEDS)
        # gate decisions, 9 pairs, both tasks, stored EEG; own CPU video preds vs preds.npz; and with own EEG
        G = {}
        for t in ("g3", "g5"):
            for arm in "CKE":
                same, flags_cpu, flags_gpu, flags_ownE, flags_stored = 0, 0, 0, 0, 0
                for vs in SEEDS:
                    pst = stV[(vs, f)][1 if t == "g3" else 2][stV[(vs, f)][0][k]].astype(np.float64)[None]
                    pc = preds[(arm, t, vs)][None]
                    pg = P[f"{arm}|{t}|s{vs}"][i].astype(np.float64)[None]
                    for es in SEEDS:
                        coef = gj["gates"][f"{t}|x3dfix_dual_s{vs}_foldF|tcn_bin_foldF_s{es}"]["all"]
                        pe = np.array([stE[(es, f)][k[:-10]]])
                        cc, _ = gflag(pc, pe, coef)
                        cg, _ = gflag(pg, pe, coef)
                        co, _ = gflag(pc, np.array([pe_own[es]]), coef)
                        cs_, _ = gflag(pst, pe, coef)
                        same += int(cc[0] == cg[0])
                        flags_cpu += int(cc[0] > 0); flags_gpu += int(cg[0] > 0); flags_ownE += int(co[0] > 0)
                        flags_stored += int(cs_[0] > 0)
                G[f"{arm}|{t}"] = dict(class_agree_cpu_vs_gpu=f"{same}/9", flags_cpu=flags_cpu, flags_gpu=flags_gpu,
                                       flags_cpu_own_eeg=flags_ownE, flags_stored=flags_stored)
        R["gate"] = G
        R["secs"] = round(time.time() - t0, 1)
        rows.append(R)
        print(f"[{n_ + 1}/{len(sel)}] {k.split('/')[1]}/{k.split('/')[3][:28]} y5={x['y5']} r0 {R['r0_match']} "
              f"E {R['E_bit_identical']} K {R['K_bit_identical']} C {R['C_bit_identical']} offs {set(offs)} "
              f"dp {json.dumps({a: round(v, 4) for a, v in dmax.items()})} EEG corr {R['corr_raw_vs_clip']:.4f} "
              f"win {R['windows_match_stepA']} pe {R['pe_vs_stepA_R_max']:.2e} gate g5 "
              f"{json.dumps({a: G[a + '|g5'] for a in 'CKE'})} {R['secs']} s", flush=True)
        json.dump(rows, open(f"{OUT}/raw_recompute.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
