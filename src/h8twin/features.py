"""Per-image features for SEN12FLOOD S1 images (CP3).

Two families, both computed from VV/VH backscatter in dB:

* single-image ("instant"): distribution of VV, VH and VV-VH, dark-pixel
  fractions, plus a VV histogram used by the global-Otsu baseline;
* temporal ("anomaly vs. normal"): each image is compared with the per-pixel
  median of the *previous* images of the same sequence and orbit direction.
  Only the past is used, so the same features can run in real time on Galați.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .sen12flood import read_s1

HIST_EDGES = np.arange(-35.0, 10.0 + 0.25, 0.25)       # 180 bins for VV histograms
DARK_VV = (-15, -18, -20, -22)
DARK_VH = (-22, -25, -28)
Q = (5, 25, 50, 75, 95)


def _q(v, prefix):
    if v.size == 0:
        return {f"{prefix}_q{q}": np.nan for q in Q} | {f"{prefix}_mean": np.nan, f"{prefix}_std": np.nan}
    qs = np.percentile(v, Q)
    out = {f"{prefix}_q{q}": float(x) for q, x in zip(Q, qs)}
    out[f"{prefix}_mean"] = float(v.mean())
    out[f"{prefix}_std"] = float(v.std())
    return out


def instant_features(x: np.ndarray) -> tuple[dict, np.ndarray]:
    """x: (2, H, W) dB. Returns (features, VV histogram counts)."""
    vv, vh = x[0], x[1]
    ok = np.isfinite(vv) & np.isfinite(vh)
    f = {"valid_frac": float(ok.mean())}
    v, h = vv[ok], vh[ok]
    f |= _q(v, "vv") | _q(h, "vh") | _q(v - h, "ratio")
    for t in DARK_VV:
        f[f"vv_lt{-t}"] = float((v < t).mean()) if v.size else np.nan
    for t in DARK_VH:
        f[f"vh_lt{-t}"] = float((h < t).mean()) if h.size else np.nan
    hist, _ = np.histogram(np.clip(v, HIST_EDGES[0], HIST_EDGES[-1] - 1e-6), bins=HIST_EDGES)
    return f, hist.astype("int64")


def temporal_features(x: np.ndarray, ref: np.ndarray | None, n_prev: int) -> dict:
    """Compare x with the per-pixel median `ref` of earlier images (same orbit)."""
    f = {"n_prev": n_prev}
    keys = ["d_vv_mean", "d_vv_q5", "d_vv_q50", "d_vh_mean", "d_vh_q5", "d_vh_q50",
            "d_vv_lt3", "d_vv_lt6", "d_vh_lt3", "d_vh_lt6", "new_dark_vv18", "new_dark_vh25"]
    if ref is None:
        return f | {k: np.nan for k in keys}
    d = x - ref
    ok = np.isfinite(d[0]) & np.isfinite(d[1])
    dv, dh = d[0][ok], d[1][ok]
    if dv.size == 0:
        return f | {k: np.nan for k in keys}
    f |= {"d_vv_mean": float(dv.mean()), "d_vv_q5": float(np.percentile(dv, 5)),
          "d_vv_q50": float(np.median(dv)), "d_vh_mean": float(dh.mean()),
          "d_vh_q5": float(np.percentile(dh, 5)), "d_vh_q50": float(np.median(dh)),
          "d_vv_lt3": float((dv < -3).mean()), "d_vv_lt6": float((dv < -6).mean()),
          "d_vh_lt3": float((dh < -3).mean()), "d_vh_lt6": float((dh < -6).mean())}
    vv, rvv, vh, rvh = x[0][ok], ref[0][ok], x[1][ok], ref[1][ok]
    f["new_dark_vv18"] = float(((vv < -18) & (rvv >= -18)).mean())
    f["new_dark_vh25"] = float(((vh < -25) & (rvh >= -25)).mean())
    return f


def sequence_features(rows: pd.DataFrame, max_ref: int = 6) -> tuple[list[dict], list[np.ndarray]]:
    """Features for all available S1 images of one sequence, in date order."""
    rows = rows.sort_values("date")
    imgs = [read_s1(json.loads(r)) for r in rows["files"]]
    h = min(a.shape[1] for a in imgs)
    w = min(a.shape[2] for a in imgs)
    imgs = [a[:, :h, :w] for a in imgs]
    feats, hists = [], []
    for i, (_, r) in enumerate(rows.iterrows()):
        f, hist = instant_features(imgs[i])
        prev = [j for j in range(i) if rows.iloc[j]["orbit"] == r["orbit"]][-max_ref:]
        ref = np.nanmedian(np.stack([imgs[j] for j in prev]), axis=0) if prev else None
        f |= temporal_features(imgs[i], ref, len(prev))
        f |= {"seq": r["seq"], "img_key": r["img_key"], "date": r["date"], "orbit": r["orbit"],
              "flooding": int(bool(r["flooding"])), "official_split": r["official_split"],
              "geography": r["geography"], "t_index": i}
        feats.append(f)
        hists.append(hist)
    return feats, hists


def extract_features(index: pd.DataFrame, n_jobs: int = -1, verbose: int = 0):
    """Run `sequence_features` over all sequences. Returns (features DataFrame, VV histograms)."""
    from joblib import Parallel, delayed

    s1 = index[(index.sensor == "S1") & index.available]
    groups = [g for _, g in s1.groupby("seq")]
    out = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(sequence_features)(g) for g in groups)
    feats = [f for fs, _ in out for f in fs]
    hists = [h for _, hs in out for h in hs]
    df = pd.DataFrame(feats)
    return df.reset_index(drop=True), np.stack(hists)
