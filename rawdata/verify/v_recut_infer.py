# Independent forward-pass check for the ttg_recut re-score (eeg conda env: torch + the grader model code).
# One run's last.pt (the post-epoch-12 state), GraderNet in eval mode, fp32 on CPU; frames read straight from
# the memmaps (cache_frames/f16s224 for unaffected clips and for the re-cut clips' OLD frames,
# ttg_recut/cache_f16s224 for their NEW frames), normalised with train_pooled.norm_batch, softmax via
# train_grader.head_probs. Clip choice uses a different RNG seed from rescore_recut.py.
# Compares: unaffected clips vs the stored dump rows; re-cut clips on old frames vs the stored rows; re-cut clips on
# new frames vs the patched rows. Writes infer_<run>.json.
import os, sys, json, csv, random
import numpy as np
EEG = "/work/mech-ai-scratch/alloy/EEG"
R = f"{EEG}/output/ttg_recut"
OUT = f"{R}/verify/indep"
GR = "/work/mech-ai-scratch/alloy/video-eeg-ensembling/grader"
T, S = 16, 224


def memmap(d):
    meta = json.load(open(os.path.join(d, "index.json")))
    mm = np.memmap(os.path.join(d, "frames.u8"), dtype=np.uint8, mode="r",
                   shape=(meta["n"], meta["frames"], meta["size"], meta["size"], 3))
    return {p: i for i, p in enumerate(meta["paths"])}, mm


def main(run="x3dfix_dual_s1_fold2", n_ctl=6, n_aff=4):
    sys.path.insert(0, GR); sys.path.insert(0, EEG)
    os.chdir(EEG)
    import torch
    import train_grader as TG
    import train_pooled as tp
    torch.set_num_threads(8)
    d = f"{EEG}/output/ttg_vsubj/{run}"
    cfg = json.load(open(f"{d}/config.json")); rk = cfg["run_key"]
    ck = torch.load(f"{d}/last.pt", map_location="cpu", weights_only=False)
    info = dict(next_epoch=ck["state"]["next_epoch"], train_done=ck["state"]["train_done"],
                hist=[h["epoch"] for h in ck["state"]["history"]])
    m = TG.GraderNet("x3d", ["g3", "g5"], rk["fix_x3d"], T, S, pretrained=False, logit_bound=0.0)
    m.load_state_dict(ck["model"], strict=True); m.eval()
    zo = np.load(f"{d}/val_ep12.npz", allow_pickle=True)
    zp = np.load(f"{R}/vsubj_patched/{run}/val_ep12.npz", allow_pickle=True)
    man = {r["clip_key"] for r in csv.DictReader(open(f"{R}/manifest.csv"))}
    cand = {r["key"] for r in csv.DictReader(open(f"{R}/plan/candidates.csv"))}
    exc = {r["key"] for r in csv.DictReader(open(f"{R}/exclude.csv"))}
    po = [str(p) for p in zo["path"]]; pp = [str(p) for p in zp["path"]]
    rng = random.Random(777)
    ctl = rng.sample(sorted(p for p in po if p not in cand and p not in exc), n_ctl)
    aff = rng.sample(sorted(p for p in po if p in man), n_aff)
    row_old, mm_old = memmap(f"{EEG}/cache_frames/f16s224")
    row_new, mm_new = memmap(f"{R}/cache_f16s224")

    def predict(mm, rows, keys):
        x = torch.from_numpy(np.stack([np.ascontiguousarray(mm[rows[k]].transpose(3, 0, 1, 2)) for k in keys]))
        with torch.no_grad():
            o = m(tp.norm_batch(x, torch.device("cpu")))
            return {t: TG.head_probs(m, o[t]).numpy().astype(np.float32) for t in ("g3", "g5")}

    res = dict(run=run, ckpt=info, head_softmax_bug=bool(m.head_softmax_bug), ctl=ctl, aff=aff)
    ipo = {p: i for i, p in enumerate(po)}; ipp = {p: i for i, p in enumerate(pp)}
    for tag, mm, rows, keys, z, idx in (("ctl_vs_stored", mm_old, row_old, ctl, zo, ipo),
                                        ("aff_oldframes_vs_stored", mm_old, row_old, aff, zo, ipo),
                                        ("aff_newframes_vs_patched", mm_new, row_new, aff, zp, ipp)):
        pr = predict(mm, rows, keys)
        for t in ("g3", "g5"):
            ref = z[f"probs_{t}"][[idx[k] for k in keys]]
            res[f"{tag}_{t}"] = dict(max_abs_dp=float(np.abs(pr[t] - ref).max()),
                                     argmax_agree=int((pr[t].argmax(1) == ref.argmax(1)).sum()), n=len(keys),
                                     ours=pr[t].round(5).tolist(), ref=ref.round(5).tolist())
        print(tag, {t: (res[f"{tag}_{t}"]["max_abs_dp"], res[f"{tag}_{t}"]["argmax_agree"]) for t in ("g3", "g5")}, flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/infer_{run}.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:2])
