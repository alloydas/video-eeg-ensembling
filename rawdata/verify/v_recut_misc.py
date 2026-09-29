# Small independent checks for the ttg_recut step, written to verify/indep/misc.json:
#  repro      the 5 reproduction controls decode to frames bit-identical (per-frame md5) to the parent data_full clips
#  cache      ttg_recut/cache_f16s224 rows are the re-cut clips (ffmpeg decode at the linspace indices) not the old
#  duplicate  RN229 s42 (old) == RN229 s41 (new) footage, RN213 s41 (old) == RN213 s42 (new) (frame-diff NCC)
#  calls      re-cut seizure clips called seizure by a majority of the 3 video seeds (g5 argmax), before / after
#  disputed   re-cut seizure clips whose 3-seed mean EEG P(sz) (p_logmean, epoch 30) is < 0.5
#  negative   the mislabelled negative's window holds an annotated seizure (raw_audit/events.csv)
import os, csv, json, glob, random, hashlib, subprocess
import numpy as np
FF = "/work/mech-ai/alloy/miniconda3/bin"
EEG = "/work/mech-ai-scratch/alloy/EEG"
R = f"{EEG}/output/ttg_recut"
csv.field_size_limit(1 << 30)


def dec(p, w, h, vf_pre=""):
    r = subprocess.run([f"{FF}/ffmpeg", "-nostdin", "-v", "error", "-i", p, "-vf",
                        f"{vf_pre}scale={w}:{h}:flags=area,format=gray", "-f", "rawvideo", "pipe:1"], capture_output=True)
    a = np.frombuffer(r.stdout, np.uint8); n = a.size // (w * h)
    return a[:n * w * h].reshape(n, h, w).astype(np.float32)


def ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))


def framemd5(p):
    r = subprocess.run([f"{FF}/ffmpeg", "-nostdin", "-v", "error", "-i", p, "-f", "framemd5", "-"], capture_output=True, text=True)
    return [ln.split(",")[-1].strip() for ln in r.stdout.splitlines() if ln and not ln.startswith("#")]


def main():
    res = {}
    man = {r["clip_key"]: r for r in csv.DictReader(open(f"{R}/manifest.csv"))}
    # repro
    rr = []
    for p in sorted(glob.glob(f"{R}/repro/data_full/**/video.mp4", recursive=True)):
        rel = os.path.relpath(p, f"{R}/repro/data_full")
        a, b = framemd5(p), framemd5(f"{EEG}/data_full/{rel}")
        rr.append(dict(clip=rel, n=len(a), n_orig=len(b), identical=a == b))
    res["repro_framemd5"] = rr
    # cache rows
    def mm(d):
        m = json.load(open(d + "/index.json"))
        return {p: i for i, p in enumerate(m["paths"])}, np.memmap(d + "/frames.u8", dtype=np.uint8, mode="r",
                                                                    shape=(m["n"], m["frames"], m["size"], m["size"], 3))
    rn, An = mm(f"{R}/cache_f16s224"); ro, Ao = mm(f"{EEG}/cache_frames/f16s224")
    g = lambda A: (A[..., 0] * 0.299 + A[..., 1] * 0.587 + A[..., 2] * 0.114).astype(np.float32)  # noqa: E731
    cc = []
    for k in sorted(man):
        p = os.path.join(EEG, man[k]["out"])
        F = dec(p, 224, 224)
        idx = np.linspace(0, len(F) - 1, 16).round().astype(int)
        cn = g(np.asarray(An[rn[k]], np.float32)); co = g(np.asarray(Ao[ro[k]], np.float32))
        cc.append((ncc(F[idx], cn), ncc(F[idx], co), float(np.abs(F[idx] - cn).mean()), float(np.abs(F[idx] - co).mean())))
    c = np.array(cc)
    res["cache_rows"] = dict(n=len(cc), in_cache=len(set(man) & set(rn)), new_row_ncc_min=float(c[:, 0].min()),
                             old_row_ncc_max=float(c[:, 1].max()), new_row_closer=int((c[:, 2] < c[:, 3]).sum()),
                             new_row_mae_max=float(c[:, 2].max()))
    # duplicates
    dd = []
    base = f"{EEG}/data_full"; nb = f"{R}/data_full"
    for name, a, b, w, h in (
            ("RN229 s42 old vs s41 new", f"{base}/Data_RN229_cropped/11-03-2023 to 11-05-2023/seizure_42_Stage_4_20231105_061409/video.mp4",
             f"{nb}/Data_RN229_cropped/11-03-2023 to 11-05-2023/seizure_41_Stage_2_20231105_051400/video.mp4", 180, 104),
            ("RN213 s41 old vs s42 new", f"{base}/Data_RN213_cropped/11-2-2023/seizure_41_Stage_3_20231103_050307/video.mp4",
             f"{nb}/Data_RN213_cropped/11-2-2023/seizure_42_Stage_2_20231103_050406/video.mp4", 160, 240),
            ("control RN229 s42 old vs s42 new", f"{base}/Data_RN229_cropped/11-03-2023 to 11-05-2023/seizure_42_Stage_4_20231105_061409/video.mp4",
             f"{nb}/Data_RN229_cropped/11-03-2023 to 11-05-2023/seizure_42_Stage_4_20231105_061409/video.mp4", 180, 104)):
        DA = np.diff(dec(a, w, h), axis=0); DB = np.diff(dec(b, w, h), axis=0)
        m = min(len(DA), len(DB)) // 2; best = (-2, None); sc = []
        for sh in range(-len(DB) + m, len(DA) - m):
            i0 = max(0, -sh); i1 = min(len(DB), len(DA) - sh)
            if i1 - i0 < m:
                continue
            v = ncc(DA[i0 + sh:i1 + sh], DB[i0:i1]); sc.append(v)
            if v > best[0]:
                best = (v, sh)
        dd.append(dict(pair=name, best_ncc=best[0], shift_s=best[1] / 15.0, p99_other=float(np.percentile(sc, 99))))
    res["duplicates"] = dd
    # calls + disputed
    def load(t):
        d = {}
        for f in range(5):
            z = np.load(t.format(f=f), allow_pickle=True)
            for p, b, y in zip(z["path"], z["probs_g5"], z["y5"]):
                d[str(p)] = (b, int(y))
        return d
    calls = {}
    for rc in ("x3dfix", "x3dbug"):
        B = [load(f"{EEG}/output/ttg_vsubj/{rc}_dual_s{s}_fold{{f}}/val_ep12.npz") for s in (1, 2, 3)]
        A = [load(f"{R}/vsubj_patched/{rc}_dual_s{s}_fold{{f}}/val_ep12.npz") for s in (1, 2, 3)]
        for grp, cond in (("seizure", lambda y: y > 0), ("non-seizure", lambda y: y == 0)):
            ks = [k for k in man if cond(B[0][k][1])]
            calls[f"{rc} {grp}"] = dict(n=len(ks), before=sum(sum(D[k][0].argmax() != 0 for D in B) >= 2 for k in ks),
                                        after=sum(sum(D[k][0].argmax() != 0 for D in A) >= 2 for k in ks))
    res["video_calls_majority_g5"] = {k: {a: int(b) for a, b in v.items()} for k, v in calls.items()}
    E = {}
    for s in (1, 2, 3):
        for f in range(5):
            z = np.load(f"{EEG}/output/ttg_eeg/subject/tcn_bin_fold{f}_s{s}/val_clip_ep30.npz", allow_pickle=True)
            for p, v, y in zip(z["path"], z["p_logmean"], z["y5"]):
                E.setdefault(str(p).replace("/video.mp4", ""), []).append((float(v), int(y)))
    mk = {k[:-len("/video.mp4")]: k for k in man}
    res["eeg_disputed_recut"] = sorted((k, round(float(np.mean([v for v, _ in E[k]])), 3)) for k in mk
                                       if E[k][0][1] > 0 and np.mean([v for v, _ in E[k]]) < 0.5)
    allm = {k: np.mean([v for v, _ in x]) for k, x in E.items()}; ally = {k: x[0][1] for k, x in E.items()}
    res["eeg_base_rate"] = dict(seizure_below_05=int(sum(1 for k in allm if ally[k] > 0 and allm[k] < 0.5)),
                                n_seizure=int(sum(1 for k in allm if ally[k] > 0)))
    # mislabelled negative
    inf = {}
    for ln in open(f"{EEG}/data_full/Data_RN242_cropped/10-12-2023/clip_03_vs_seizure_04_Stage_4_20231012_234446/info.txt"):
        if ":" in ln:
            a, b = ln.split(":", 1); inf[a.strip()] = b.strip()
    cs, ce = inf["Clip start"], inf["Clip end"]
    ev = [(r["label"], r["start"], r["end"]) for r in csv.DictReader(open(f"{EEG}/output/raw_audit/events.csv"))
          if r["animal"] == "RN242" and r["start"] < ce and r["end"] > cs]
    res["mislabelled_negative"] = dict(clip=[cs, ce], overlapping_events=ev)
    json.dump(res, open(f"{R}/verify/indep/misc.json", "w"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
