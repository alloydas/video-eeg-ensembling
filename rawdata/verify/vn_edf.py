#!/usr/bin/env python3
"""Priority 5 (eeg.edf format) and the EEG half of priority 2 (clip EEG equals the raw EDF), with an own EDF reader.

For every new clip, and for up to N_CTRL existing clips per session (the same checks as a control):
  header: version, ids, start date/time, header bytes, reserved, record duration, ns, labels (vs the raw EDF),
          transducer, dimension, digital range, prefilter, samples per record, file size = header + nrec * record
  data:   every 1 kHz raw channel (ECG/EEG and the FIR-HP copies) against the raw EDF at the implied offset
          (Clip start - EDF header start): lag by NCC over +-3 s, then max |clip - raw| in units of the clip's own
          quantisation step over the whole clip; nrec against the clip window; aux channels (Activity, SignalStr,
          Temp): all-zero or not, and whether the raw has nonzero values there
  new vs existing: every header field that is not data-dependent is compared with the existing clips of the session
Writes verify/vn_edf.csv, vn_edf.json.
"""
import glob
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy.signal import fftconvolve

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vn_common as C  # noqa: E402

N_CTRL = 6
_RAW = {}


def raw_edf(path):
    if path not in _RAW:
        _RAW[path] = C.EDF(path)
    return _RAW[path]


def header_sig(E):
    """Fields that must not depend on the data."""
    return dict(version=E.version, patient=E.patient.strip(), recording=E.recording.strip(), startdate=E.startdate,
                starttime=E.starttime, header_bytes=E.header_bytes, reserved=E.reserved.strip(), rec_dur=E.rec_dur,
                ns=E.ns, transducer=tuple(x.strip() for x in E.transducer), dim=tuple(x.strip() for x in E.phys_dim),
                dmin=tuple(x.strip() for x in E.dmin_raw), dmax=tuple(x.strip() for x in E.dmax_raw),
                prefilter=tuple(x.strip() for x in E.prefilter), nsamp=tuple(int(x) for x in E.nsamp),
                sig_reserved=tuple(x.strip() for x in E.sig_reserved))


def check_clip(clip_dir, info, raw_path):
    E = C.EDF(os.path.join(clip_dir, "eeg.edf"))
    R = raw_edf(raw_path)
    out = dict(ns=E.ns, nrec=E.nrec, labels="|".join(E.labels), raw_labels="|".join(R.labels))
    out["size_ok"] = E.size == E.header_bytes + E.nrec * E.rec_bytes and E.nrec == E.nrec_by_size
    out["header_bytes_ok"] = E.header_bytes == 256 * (E.ns + 1)
    cs, ce = C.parse_dt(info["Clip start"]), C.parse_dt(info["Clip end"])
    off = (cs - R.start()).total_seconds()
    win = (ce - cs).total_seconds()
    # MNE crop keeps round(win * 1000) + 1 samples; the writer trims to whole 1-s records
    exp_nrec = int((round(win * 1000) + 1) // 1000)
    raw_end = R.nrec * R.rec_dur
    if off + win > raw_end:
        exp_nrec = int((round((raw_end - off) * 1000)) // 1000)
    out["exp_nrec"] = exp_nrec
    out["nrec_ok"] = E.nrec == exp_nrec
    out["offset_s"] = off
    out["pick"] = E.labels[C.eeg_channel(E.labels)] if C.eeg_channel(E.labels) is not None else None
    worst = dict(lag=0, qmax=0.0, r=1.0)
    per = []
    for k in range(E.ns):
        if R.fs[k] != 1000:
            continue
        x, dig = E.read_all_phys(k)
        step = E.gain[k]
        # raw window with margin for the lag search
        m = 3.0
        y, _ = R.read_phys(k, off - m, off + len(x) / 1000.0 + m)
        a = int(round(m * 1000))
        # lag on first differences (the unfiltered channel drifts; differencing removes the baseline)
        seg = np.diff(x[:10001]); seg = seg - seg.mean()
        if len(y) < 2 * a + len(seg) + 1:
            per.append(dict(ch=E.labels[k], lag=None, q=float("nan"), r=float("nan"), n=0, uniq=0))
            continue
        yc = np.diff(y[: 2 * a + len(seg) + 1])
        cc = fftconvolve(yc - yc.mean(), seg[::-1], mode="valid")          # cc[j] = sum seg * y[j:j+len]
        # local energy normalisation (NCC)
        cs2 = np.concatenate([[0.0], np.cumsum((yc - yc.mean()) ** 2)])
        en = np.sqrt((cs2[len(seg):] - cs2[:-len(seg)]) * float((seg * seg).sum())) + 1e-30
        ncc = cc / en
        j = int(np.argmax(ncc))
        best = (float(ncc[j]), j - a)
        lag = best[1]
        y0 = y[a: a + len(x)]                                              # at the implied offset (lag 0)
        n0 = min(len(y0), len(x))
        q0 = float(np.abs(x[:n0] - y0[:n0]).max() / step) if n0 else float("nan")
        yy = y[a + lag: a + lag + len(x)]
        n = min(len(yy), len(x))
        diff = np.abs(x[:n] - yy[:n])
        q = float(diff.max() / step) if n else float("nan")
        r = float(np.corrcoef(x[:n], yy[:n])[0, 1]) if n > 10 and yy[:n].std() > 0 else float("nan")
        per.append(dict(ch=E.labels[k], lag=lag, ncc=best[0], q=q, q_lag0=q0, r=r, n=n, uniq=int(len(np.unique(dig)))))
        if lag is not None and abs(lag) > abs(worst["lag"]):
            worst["lag"] = lag
        worst["qmax"] = max(worst["qmax"], q)
        worst["r"] = min(worst["r"], r)
    out["kHz_channels"] = json.dumps(per)
    out["lag_max_abs"] = worst["lag"]
    out["q_max"] = worst["qmax"]
    out["r_min"] = worst["r"]
    pk = C.eeg_channel(E.labels)
    pr = [p for p in per if p["ch"] == E.labels[pk]]
    out["pick_lag"] = pr[0]["lag"] if pr else None
    out["pick_q"] = pr[0]["q"] if pr else None
    out["pick_uniq"] = pr[0]["uniq"] if pr else None
    # aux channels
    aux = {}
    for k in range(E.ns):
        if R.fs[k] == 1000:
            continue
        x, _ = E.read_all_phys(k)
        rv, _ = R.read_phys(k, off, off + E.nrec)
        aux[E.labels[k]] = dict(clip_all_zero=bool(np.all(np.abs(x) < 1e-9)), raw_nonzero=bool(np.any(rv != 0)))
    out["aux"] = json.dumps(aux)
    out["aux_all_zero"] = all(v["clip_all_zero"] for v in aux.values())
    return out, header_sig(E)


def main():
    M = [r for r in C.manifest() if r["status"] != "excluded"]
    rows = []
    hdr_new = defaultdict(list)
    for i, r in enumerate(M):
        d = os.path.join(C.NEWFULL, f"Data_{r['animal']}_cropped", r["session"], r["clip_dir"])
        info = C.parse_info(os.path.join(d, "info.txt"))[0]
        o, h = check_clip(d, info, os.path.join(C.RAW, r["raw_edf"]))
        o.update(key=r["key"], set="new", animal=r["animal"], session=r["session"], kind=r["kind"])
        # labels equal the raw EDF's source labels
        o["labels_eq_raw"] = o["labels"] == o["raw_labels"]
        rows.append(o)
        hdr_new[(r["animal"], r["session"])].append(h)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(M)}", flush=True)
    # controls: existing clips of the same sessions
    hdr_ex = defaultdict(list)
    ctrl = []
    for a, s in sorted(hdr_new):
        full = os.path.join(C.FULL, f"Data_{a}_cropped", s)
        dirs = sorted(d for d in glob.glob(os.path.join(glob.escape(full), "*"))
                      if os.path.exists(os.path.join(d, "eeg.edf")) and os.path.exists(os.path.join(d, "info.txt")))
        pick = dirs[:: max(1, len(dirs) // N_CTRL)][:N_CTRL]
        for d in pick:
            info = C.parse_info(os.path.join(d, "info.txt"))[0]
            # the raw EDF named in info.txt
            ef = info.get("EDF file")
            cand = glob.glob(os.path.join(glob.escape(os.path.join(C.RAW, a, s)), "**", ef), recursive=True)
            if not cand:
                continue
            o, h = check_clip(d, info, cand[0])
            o.update(key=os.path.relpath(d, C.FULL), set="existing", animal=a, session=s,
                     kind="seizure" if os.path.basename(d).startswith("seizure_") else "nonseizure")
            o["labels_eq_raw"] = o["labels"] == o["raw_labels"]
            ctrl.append(o)
            hdr_ex[(a, s)].append(h)
    # header comparison new vs existing (per session)
    cmp_rows = []
    for key in sorted(hdr_new):
        ex = hdr_ex.get(key, [])
        for fld in hdr_new[key][0]:
            nv = Counter(str(h[fld]) for h in hdr_new[key])
            ev = Counter(str(h[fld]) for h in ex)
            cmp_rows.append(dict(session="/".join(key), field=fld, new=dict(nv), existing=dict(ev),
                                 same=set(nv) == set(ev) if ev else None))
    diffs = [c for c in cmp_rows if c["same"] is False]
    allrows = rows + ctrl

    def summ(rs):
        return dict(n=len(rs), size_ok=Counter(x["size_ok"] for x in rs), header_bytes_ok=Counter(x["header_bytes_ok"] for x in rs),
                    nrec_ok=Counter(x["nrec_ok"] for x in rs), lag_max_abs=max(abs(x["lag_max_abs"]) for x in rs),
                    q_max=max(x["q_max"] for x in rs), r_min=min(x["r_min"] for x in rs),
                    pick=Counter(x["pick"] for x in rs), labels_eq_raw=Counter(x["labels_eq_raw"] for x in rs),
                    aux_all_zero=Counter(x["aux_all_zero"] for x in rs), ns=Counter(x["ns"] for x in rs),
                    pick_uniq_min=min(x["pick_uniq"] for x in rs))
    out = dict(new=summ(rows), existing_controls=summ(ctrl), header_field_diffs_new_vs_existing=diffs,
               nrec_bad=[(x["key"], x["nrec"], x["exp_nrec"]) for x in allrows if not x["nrec_ok"]][:20],
               lag_bad=[(x["key"], x["lag_max_abs"]) for x in allrows if x["lag_max_abs"] != 0][:20],
               q_bad=[(x["key"], x["q_max"]) for x in allrows if x["q_max"] > 0.5 + 1e-6][:20])
    C.write_csv("vn_edf.csv", allrows)
    C.write_json("vn_edf.json", out)
    print(json.dumps(out, indent=1, default=str)[:8000])


if __name__ == "__main__":
    main()
