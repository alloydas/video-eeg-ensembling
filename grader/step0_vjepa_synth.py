#!/usr/bin/env python3
"""
TEST-ONLY feature files for grader/step0_vjepa_analysis.py (Step 0; grader/step0_vjepa_prereg.md).
Nothing written here is ever a result: every file carries `synthetic` = True and the analysis labels
any run on it MEANINGLESS.

PURPOSE
  Writes <out_dir>/features.npz with exactly the keys the V-JEPA 2 extraction writes (section 2 of
  the pre-registration): path, y5, starts, n_header, single_snippet, failed, not_extracted, perm,
  sparse, dense_mean, shuffled_mean [N,D] f32, dense_tt, shuffled_tt [N,8,8,D] f16, dense_tv,
  shuffled_tv [N,D] f32, sparse_ml, dense_ml, shuffled_ml [N,3,D] f32, plus provenance scalars.
  Rows are the 24,497 sorted items of Stage 2's features.npz (same fingerprint); starts / n_header /
  single_snippet are copied from it and perm is regenerated with the registered seed rule, so the
  file passes every consistency check of the merge (the *_tt means equal *_mean, the f16 token speed
  equals *_tv, perm equals the seed rule) within the registered tolerance.

  --kind standin     the code-freeze stand-in (section 8): the REAL Stage 2 X3D sparse / dense_mean /
                     shuffled_mean copied into the V-JEPA 2 slots (D = 2048), seeded synthetic *_tt,
                     *_tv, *_ml around them. VJ-x arms then equal X3D-x arms, so d_B must be 0 exactly.
  --kind go | weak | go_encoder | kill
                     a planted severity signal (severe = S4+S5) along one direction, with animal and
                     session offsets as nuisance, at registered-style amplitudes per arm:
                       go          VJ-dense strong, VJ-shuffled none, X3D weak   -> M and B -> GO
                       weak        VJ-dense medium, VJ-shuffled none, X3D strong -> M only  -> WEAK
                       go_encoder  VJ-dense = VJ-shuffled (shuffled slightly stronger), X3D weak -> GO-ENCODER
                       kill        no signal anywhere                             -> KILL
                     Also writes <out_dir>/x3d_features.npz (Stage 2 layout, same small D) with its
                     own planted amplitudes; pass it as --x3d.
  --kind leak_animal_dir
                     severity along a DIFFERENT orthonormal direction for every animal: a probe can
                     only use it on an animal it was trained on, so subject-disjoint OOF WS must be
                     ~0.5, and a leaky (in-sample) fit ~1 (--check prints both).
  --kind leak_prevalence
                     VJ-dense = VJ-shuffled + session- and animal-level offsets proportional to that
                     session's / animal's severe prevalence (label information that is constant within
                     a session). Pooled AUROC must rise; WS must not rise (every within-session pair
                     shares the offset). WS can FALL: the probe spends weight on the confounded
                     direction, which carries within-session noise.
  --fail_seizure N / --not_extracted_seizure N / --not_extracted_nonseizure N plant NaN rows listed
  in `failed` / `not_extracted` to test the drop accounting and the stop rules.

  --check <step0_results.json> compares a finished analysis with the expectation stored in
  <out_dir>/synth.json and prints PASS / FAIL lines (exit 1 on any FAIL).

USAGE (eeg env python; outputs only under $EEG_ROOT/output/ttg_tmp/vjepa_<name>/)
  python grader/step0_vjepa_synth.py --kind go --out_dir $EEG_ROOT/output/ttg_tmp/vjepa_go
  python grader/step0_vjepa_analysis.py --vjepa output/ttg_tmp/vjepa_go/features.npz \
      --x3d output/ttg_tmp/vjepa_go/x3d_features.npz --out $EEG_ROOT/output/ttg_tmp/vjepa_go --reps 500
  python grader/step0_vjepa_synth.py --out_dir $EEG_ROOT/output/ttg_tmp/vjepa_go \
      --check $EEG_ROOT/output/ttg_tmp/vjepa_go/step0_results.json
"""
import argparse
import json
import os
import sys
import tempfile
import time
import zipfile

sys.dont_write_bytecode = True
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ttg_common as C                                               # noqa: E402

REAL_X3D = os.path.join(C.EEG_ROOT, "output", "ttg_probe", "features.npz")
TMP_DIR = os.path.join(C.EEG_ROOT, "output", "ttg_tmp")
K, T = 8, 16

# per-kind amplitudes along the content direction: VJ dense / shuffled / sparse, X3D dense / shuffled / sparse
AMPS = {"go": (1.5, 0.0, 0.3, 0.5, 0.5, 0.5),
        "weak": (1.0, 0.0, 0.0, 1.8, 1.8, 1.8),
        "go_encoder": (1.5, 1.7, 1.5, 0.5, 0.5, 0.5),
        "kill": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "leak_animal_dir": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "leak_prevalence": (0.8, 0.8, 0.0, 0.0, 0.0, 0.0)}
EXPECT = {"go": dict(cell="GO", M=True, B=True), "weak": dict(cell="WEAK", M=True, B=False),
          "go_encoder": dict(cell="GO-ENCODER", M=False, B=True), "kill": dict(cell="KILL", M=False, B=False),
          "standin": dict(d_B_exactly_zero=True),
          "leak_animal_dir": dict(M=False, max_oof_ws=0.6, min_leaky_ws=0.75),
          # one-sided: a session-constant offset cancels in every within-session pair, so it cannot RAISE
          # WS; it can lower it (the probe spends weight on the confounded direction, which adds
          # within-session noise). Pooled AUROC has no such protection.
          "leak_prevalence": dict(M=False, max_d_ws=0.005, min_d_pooled=0.05)}
LEAK_DIR_AMP = 5.0          # per-animal direction amplitude (2.0 was too weak: one linear probe can put at most
                            # ~1/sqrt(20) of its weight on each of 20 orthonormal directions, so even a leaky
                            # in-sample fit reached only WS 0.67)


class NpzWriter:
    """np.savez-compatible (uncompressed, zip64) writer that can stream an array chunk by chunk."""

    def __init__(s, path):
        s.path = path
        fd, s.tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".npz", dir=os.path.dirname(path))
        os.close(fd)
        s.zf = zipfile.ZipFile(s.tmp, "w", zipfile.ZIP_STORED, allowZip64=True)

    def add(s, key, arr):
        with s.zf.open(key + ".npy", "w", force_zip64=True) as fh:
            np.lib.format.write_array(fh, np.asanyarray(arr), allow_pickle=False)

    def add_stream(s, key, shape, dtype, chunks):
        dtype = np.dtype(dtype)
        with s.zf.open(key + ".npy", "w", force_zip64=True) as fh:
            np.lib.format.write_array_header_1_0(fh, dict(descr=np.lib.format.dtype_to_descr(dtype),
                                                          fortran_order=False, shape=tuple(shape)))
            n = 0
            for ch in chunks:
                ch = np.ascontiguousarray(ch, dtype=dtype)
                assert ch.shape[1:] == tuple(shape[1:]), (key, ch.shape, shape)
                fh.write(ch.tobytes())
                n += len(ch)
            assert n == shape[0], (key, n, shape)

    def close(s):
        s.zf.close()
        os.chmod(s.tmp, 0o664)
        os.replace(s.tmp, s.path)


def check_out_dir(p):
    rp = os.path.realpath(p)
    if not (os.path.dirname(rp) == TMP_DIR and os.path.basename(rp).startswith("vjepa_")
            and len(os.path.basename(rp)) > 6):
        raise SystemExit(f"--out_dir must be {TMP_DIR}/vjepa_<name>, got {rp}")
    os.makedirs(rp, exist_ok=True)
    return rp


def unit(rng, D, n=1):
    q, _ = np.linalg.qr(rng.standard_normal((D, n)))
    return q.T                                                          # (n, D) orthonormal rows


def tt_views(mean_rows, P, q_rows, sigma):
    """fp32 temporal-token means u (n,8,8,D) = mean + sigma * P * q, their f16 store and fp32 speed v."""
    u = (mean_rows[:, None, None, :].astype(np.float32)
         + np.float32(sigma) * P[None].astype(np.float32) * q_rows[:, None, None, :].astype(np.float32))
    v = np.abs(np.diff(u, axis=2)).mean(axis=(1, 2), dtype=np.float32)
    return u.astype(np.float16), v.astype(np.float32)


def perms_for(paths):
    out = np.empty((len(paths), K, T), np.int8)
    for i, p in enumerate(paths):
        rng = np.random.default_rng(C.clip_seed(p))
        out[i] = np.stack([rng.permutation(T) for _ in range(K)])
    return out


def build(a):
    t0 = time.time()
    out = check_out_dir(a.out_dir)
    X = np.load(REAL_X3D, allow_pickle=False)
    paths = X["path"].astype(str)
    y5 = X["y5"].astype(int)
    N = len(paths)
    an = np.array([C.animal_of(p) for p in paths])
    se = np.array([C.session_of(p) for p in paths])
    rng = np.random.default_rng(a.seed)
    sev = (y5 >= 3).astype(np.float32)                                 # S4 + S5
    meta = dict(kind=a.kind, seed=a.seed, created=time.strftime("%F %T"))
    views = {}
    if a.kind == "standin":
        for k in ("sparse", "dense_mean", "shuffled_mean"):
            views[k] = X[k].astype(np.float32)
        D = views["dense_mean"].shape[1]
    else:
        D = a.dim
        ua, ai = np.unique(an, return_inverse=True)
        us, si = np.unique(se, return_inverse=True)
        u = unit(rng, D, 4)                                              # content, prevalence (session, animal), X3D

        def base():
            return (rng.standard_normal((N, D)) + rng.standard_normal((len(ua), D))[ai]
                    + 0.5 * rng.standard_normal((len(us), D))[si]).astype(np.float32)
        vd, vs, vp, xd, xs, xp = AMPS[a.kind]
        b = base()
        views["dense_mean"] = b + vd * sev[:, None] * u[0] + 0.3 * rng.standard_normal((N, D)).astype(np.float32)
        views["shuffled_mean"] = b + vs * sev[:, None] * u[0] + 0.3 * rng.standard_normal((N, D)).astype(np.float32)
        views["sparse"] = b + vp * sev[:, None] * u[0] + 0.3 * rng.standard_normal((N, D)).astype(np.float32)
        if a.kind == "leak_animal_dir":
            if D < len(ua):
                raise SystemExit(f"--dim must be >= {len(ua)} animals for leak_animal_dir")
            dirs = unit(rng, D, len(ua))
            views["dense_mean"] = views["dense_mean"] + LEAK_DIR_AMP * sev[:, None] * dirs[ai]
            meta["animal_directions"] = f"orthonormal, one per animal, amplitude {LEAK_DIR_AMP} on severe clips"
        if a.kind == "leak_prevalence":
            sz = y5 >= 1
            prev_s = np.bincount(si, weights=sev * sz, minlength=len(us)) / np.maximum(
                np.bincount(si, weights=sz.astype(float), minlength=len(us)), 1)
            prev_a = np.bincount(ai, weights=sev * sz, minlength=len(ua)) / np.maximum(
                np.bincount(ai, weights=sz.astype(float), minlength=len(ua)), 1)
            views["shuffled_mean"] = views["dense_mean"].copy()          # identical content and noise
            views["dense_mean"] = (views["dense_mean"] + 6.0 * prev_s[si][:, None] * u[1]
                                   + 6.0 * prev_a[ai][:, None] * u[2]).astype(np.float32)
            meta["offsets"] = "dense = shuffled + 6*session severe prevalence*u1 + 6*animal severe prevalence*u2"
        xb = base()
        xw = unit(rng, D, 1)[0]
        xviews = {"dense_mean": xb + xd * sev[:, None] * xw + 0.3 * rng.standard_normal((N, D)).astype(np.float32),
                  "shuffled_mean": xb + xs * sev[:, None] * xw + 0.3 * rng.standard_normal((N, D)).astype(np.float32),
                  "sparse": xb + xp * sev[:, None] * xw + 0.3 * rng.standard_normal((N, D)).astype(np.float32)}
        meta["amplitudes"] = dict(zip(["VJ-dense", "VJ-shuffled", "VJ-sparse", "X3D-dense", "X3D-shuffled",
                                       "X3D-sparse"], AMPS[a.kind]))
        w = NpzWriter(os.path.join(out, "x3d_features.npz"))
        w.add("path", paths)
        w.add("y5", y5)
        for k, v in xviews.items():
            w.add(k, v.astype(np.float32))
        for k in ("starts", "n_header", "single_snippet"):
            w.add(k, X[k])
        w.add("failed", np.array([], dtype=paths.dtype))
        w.add("fingerprint", X["fingerprint"])
        w.add("feature_dim", np.int64(D))
        w.add("synthetic", np.array(True))
        w.close()
    meta["dim"] = int(D)

    # planted failures (NaN in every V-JEPA array, listed)
    szrows = np.flatnonzero(y5 >= 1)
    nsrows = np.flatnonzero(y5 == 0)
    prng = np.random.default_rng(a.seed + 1)
    fail = np.sort(prng.choice(szrows, a.fail_seizure, replace=False)) if a.fail_seizure else np.array([], int)
    rest = np.setdiff1d(szrows, fail)
    notx = np.sort(np.r_[prng.choice(rest, a.not_extracted_seizure, replace=False) if a.not_extracted_seizure
                         else np.array([], int),
                         prng.choice(nsrows, a.not_extracted_nonseizure, replace=False) if a.not_extracted_nonseizure
                         else np.array([], int)]).astype(int)
    nan_rows = np.zeros(N, bool)
    nan_rows[fail] = True
    nan_rows[notx] = True

    scale = float(np.std(views["dense_mean"]))
    sigma = 0.25 * scale
    P = {}
    for vname in ("dense", "shuffled"):
        g = np.random.default_rng(a.seed + (11 if vname == "dense" else 12)).standard_normal((K, K, D))
        P[vname] = g - g.mean(axis=(0, 1), keepdims=True)                 # zero mean over (snippet, t)
    tv = {}
    bad_row = int(np.setdiff1d(szrows, np.flatnonzero(nan_rows))[100])   # a seizure row that stays finite
    meta["corrupt"] = dict(kind=a.corrupt, row=bad_row if a.corrupt != "none" else None)
    w = NpzWriter(os.path.join(out, "features.npz"))
    w.add("path", paths)
    w.add("y5", y5)
    for k in ("starts", "n_header", "single_snippet"):
        v = X[k].copy()
        if a.corrupt == "starts" and k == "starts":
            v[bad_row, 0] += 1
        w.add(k, v)
    w.add("failed", paths[fail] if len(fail) else np.array([], dtype=paths.dtype))
    w.add("not_extracted", paths[notx] if len(notx) else np.array([], dtype=paths.dtype))
    pm = perms_for(paths)
    if a.corrupt == "perm":
        pm[bad_row, 0, [0, 1]] = pm[bad_row, 0, [1, 0]]
    w.add("perm", pm)
    for k in ("sparse", "dense_mean", "shuffled_mean"):
        v = views[k].copy()
        v[nan_rows] = np.nan
        w.add(k, v)
    for vname in ("dense", "shuffled"):
        mean = views[f"{vname}_mean"]
        tv[vname] = np.empty((N, D), np.float32)

        def chunks(vname=vname, mean=mean):
            for s in range(0, N, 256):
                r = np.arange(s, min(N, s + 256))
                q = np.random.default_rng([a.seed, 7 if vname == "dense" else 8, s]).standard_normal((len(r), D))
                tt16, v = tt_views(mean[r], P[vname], q, sigma)
                if a.corrupt == "tt" and vname == "dense" and s <= bad_row < s + len(r):
                    tt16[bad_row - s, 3, 5] += np.float16(0.5)              # one token mean off by 0.5
                if a.corrupt == "tv" and vname == "dense" and s <= bad_row < s + len(r):
                    v[bad_row - s] *= np.float32(1.2)
                tt16[nan_rows[r]] = np.nan
                v[nan_rows[r]] = np.nan
                tv[vname][r] = v
                yield tt16
        w.add_stream(f"{vname}_tt", (N, K, K, D), np.float16, chunks())
        w.add(f"{vname}_tv", tv[vname])
    for vname, src in (("sparse", "sparse"), ("dense", "dense_mean"), ("shuffled", "shuffled_mean")):
        r2 = np.random.default_rng([a.seed, 21, len(vname)])
        ml = np.empty((N, 3, D), np.float32)
        for s in range(0, N, 2048):
            e = min(N, s + 2048)
            ml[s:e] = views[src][s:e, None, :] + 0.5 * scale * r2.standard_normal((e - s, 3, D)).astype(np.float32)
        ml[nan_rows] = np.nan
        w.add(f"{vname}_ml", ml)
    w.add("fingerprint", X["fingerprint"])
    w.add("feature_dim", np.int64(D))
    w.add("synthetic", np.array(True))
    w.add("synthetic_kind", np.array(a.kind))
    w.add("extraction_note", np.array("TEST-ONLY synthetic file from grader/step0_vjepa_synth.py"))
    w.close()
    meta.update(n=N, failed_seizure=int(len(fail)), not_extracted=int(len(notx)),
                not_extracted_seizure=int(a.not_extracted_seizure), tt_sigma=sigma,
                expect=EXPECT.get(a.kind, {}), seconds=round(time.time() - t0, 1))
    C.atomic_json(os.path.join(out, "synth.json"), meta)
    print(f"wrote {out}/features.npz" + ("" if a.kind == "standin" else " and x3d_features.npz")
          + f" (kind {a.kind}, D {D}, {len(fail)} failed, {len(notx)} not extracted; {time.time() - t0:.0f}s)")


def leaky_ws(features, cname="severe_vs_mild"):
    """Counterfactual for leak_animal_dir: a probe trained ON the scored animals (in-sample)."""
    import probe_analysis as PA
    Z = np.load(features, allow_pickle=False)
    paths = Z["path"].astype(str)
    y5 = Z["y5"].astype(int)
    m = (y5 >= 1) & np.isfinite(Z["dense_mean"]).all(1)
    X = Z["dense_mean"][m].astype(np.float64)
    y = (y5[m] >= 3).astype(float)
    A, _ = PA.standardise(X, X)
    w = PA.fit_logreg(A, y, 1e-2)
    s = A @ w[:-1] + w[-1]
    an = np.array([C.animal_of(p) for p in paths[m]])
    se = np.array([C.session_of(p) for p in paths[m]])
    return C.within_auroc(s, y > 0, se, an)


def check(a):
    out = check_out_dir(a.out_dir)
    meta = json.load(open(os.path.join(out, "synth.json")))
    R = json.load(open(a.check))
    ex = meta["expect"]
    ok = True

    def line(name, cond, detail):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"{'PASS' if cond else 'FAIL'}  {name}: {detail}")

    d = R.get("decision") or {}
    P = R["contrasts"]["severe_vs_mild"]
    if "cell" in ex:
        line("decision cell", d.get("cell") == ex["cell"],
             f"{d.get('cell')} (expected {ex['cell']}); d_M {d.get('d_M')} CI_M {d.get('CI_M')}; "
             f"d_B {d.get('d_B')} CI_B {d.get('CI_B')}")
    if "M" in ex:
        line("M", d.get("M") == ex["M"], f"{d.get('M')} (expected {ex['M']})")
    if "B" in ex:
        line("B", d.get("B") == ex["B"], f"{d.get('B')} (expected {ex['B']})")
    line("decision labelled meaningless", d.get("valid") is False and "MEANINGLESS" in d.get("label", ""),
         d.get("label"))
    if ex.get("d_B_exactly_zero"):
        for cname, blk in R["contrasts"].items():
            A = blk["arms"]
            for v in ("dense", "shuffled", "sparse"):
                va, xa = A.get(f"VJ-{v}"), A.get(f"X3D-{v}")
                if va is None or xa is None:
                    line(f"{cname} VJ-{v} vs X3D-{v}", False, "arm missing")
                    continue
                same = all(va[k] == xa[k] for k in ("ws", "ws_ci", "pooled", "pooled_ci", "per_fold", "C_per_fold",
                                                    "crit_per_fold"))
                line(f"{cname} VJ-{v} == X3D-{v} bit for bit (WS, pooled, CIs, per-fold, C, inner criteria)", same,
                     f"WS {va['ws']!r} vs {xa['ws']!r}; C {va['C_per_fold']} vs {xa['C_per_fold']}")
            for x in blk["differences"]:
                if (x["a"], x["b"]) in (("VJ-dense", "X3D-dense"), ("VJ-sparse", "X3D-sparse")):
                    line(f"{cname} {x['a']} - {x['b']} exactly 0", x["d_ws"] == 0.0 and x["ci_ws"] == [0.0, 0.0]
                         and x["d_pooled"] == 0.0, f"d_ws {x['d_ws']!r} CI {x['ci_ws']} d_pooled {x['d_pooled']!r}")
            if blk.get("M_B"):
                line(f"{cname} d_B exactly 0 and B false", blk["M_B"]["d_B"] == 0.0 and blk["M_B"]["B"] is False,
                     f"d_B {blk['M_B']['d_B']!r} CI_B {blk['M_B']['CI_B']} B {blk['M_B']['B']}")
    if "max_oof_ws" in ex:
        ws = P["arms"]["VJ-dense"]["ws"]
        line("OOF WS of VJ-dense (animal-specific directions) stays near chance", ws <= ex["max_oof_ws"],
             f"{ws:.4f} <= {ex['max_oof_ws']}")
        lw = leaky_ws(os.path.join(out, "features.npz"))
        line("a leaky in-sample probe on the same features scores high (the test can see a leak)",
             lw >= ex["min_leaky_ws"], f"{lw:.4f} >= {ex['min_leaky_ws']}")
    if "max_d_ws" in ex:
        x = [x for x in P["differences"] if x["a"] == "VJ-dense" and x["b"] == "VJ-shuffled"][0]
        line("session/animal prevalence offsets do not RAISE WS (one-sided)", x["d_ws"] <= ex["max_d_ws"],
             f"d_ws {x['d_ws']:+.4f} CI {[round(v, 4) for v in x['ci_ws']]} (a drop is the probe spending weight "
             f"on the confounded direction)")
        line("...but they do inflate pooled AUROC", x["d_pooled"] >= ex["min_d_pooled"],
             f"d_pooled {x['d_pooled']:+.4f} CI {[round(v, 4) for v in x['ci_pooled']]}")
    dr = R["drops"]
    line("drop accounting", dr["by_reason"]["vjepa_failed (listed in failed)"] == meta["failed_seizure"]
         and dr["n_dropped"] == meta["failed_seizure"] + meta["not_extracted_seizure"],
         f"dropped {dr['n_dropped']} by reason {dr['by_reason']} (planted failed {meta['failed_seizure']})")
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--kind", choices=["standin"] + list(AMPS))
    ap.add_argument("--out_dir", required=True, help=f"{TMP_DIR}/vjepa_<name>")
    ap.add_argument("--dim", type=int, default=32, help="feature dim of the synthetic kinds (standin: 2048)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fail_seizure", type=int, default=0)
    ap.add_argument("--not_extracted_seizure", type=int, default=0)
    ap.add_argument("--not_extracted_nonseizure", type=int, default=0)
    ap.add_argument("--corrupt", choices=["none", "tv", "tt", "perm", "starts"], default="none",
                    help="negative tests: break one merge invariant on one seizure row (the analysis must abort)")
    ap.add_argument("--check", default=None, help="step0_results.json to compare with <out_dir>/synth.json")
    a = ap.parse_args()
    if a.check:
        return check(a)
    if not a.kind:
        ap.error("--kind is required unless --check is given")
    build(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
