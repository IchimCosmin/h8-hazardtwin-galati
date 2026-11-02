"""Numbers for the data card, computed from the index (and a pixel sample)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .sen12flood import read_s1


def _gaps_days(dates: pd.Series) -> np.ndarray:
    d = pd.to_datetime(dates.dropna()).sort_values().drop_duplicates()
    return d.diff().dt.days.dropna().to_numpy()


def index_stats(index: pd.DataFrame) -> dict:
    out = {}
    for s, df in index.groupby("sensor"):
        per_seq = df.groupby("seq").size()
        gaps = np.concatenate([_gaps_days(g.date) for _, g in df.groupby("seq")] or [np.array([])])
        flood_seq = df.groupby("seq").flooding.apply(lambda v: bool(v.fillna(False).any()))
        out[s] = {
            "sequences": int(df.seq.nunique()),
            "images_listed": int(len(df)),
            "images_with_all_files": int(df.available.sum()),
            "images_missing_files": int((~df.available).sum()),
            "images_per_seq_min_median_max": [int(per_seq.min()), float(per_seq.median()), int(per_seq.max())],
            "date_min": str(df.date.min().date()) if df.date.notna().any() else None,
            "date_max": str(df.date.max().date()) if df.date.notna().any() else None,
            "pct_images_flooded": round(100 * df.flooding.fillna(False).mean(), 1),
            "pct_sequences_with_flood": round(100 * flood_seq.mean(), 1),
            "pct_full_coverage": (None if df.full_coverage.isna().all()
                                  else round(100 * df.full_coverage.fillna(False).mean(), 1)),
            "revisit_days_median": float(np.median(gaps)) if gaps.size else None,
            "revisit_days_p90": float(np.percentile(gaps, 90)) if gaps.size else None,
            "revisit_days_max": float(gaps.max()) if gaps.size else None,
            "missing_dates": int(df.date.isna().sum()),
            "missing_labels": int(df.flooding.isna().sum()),
        }
        if s == "S1":
            out[s]["orbit_counts"] = {str(k): int(v) for k, v in df.orbit.value_counts(dropna=False).items()}
    return out


def raster_profile(path: str) -> dict:
    import rasterio

    with rasterio.open(path) as src:
        res = src.res
        return {"crs": str(src.crs), "width": src.width, "height": src.height,
                "res": [float(res[0]), float(res[1])], "dtype": src.dtypes[0],
                "count": src.count, "nodata": src.nodata}


def pixel_stats(index: pd.DataFrame, n: int = 60, seed: int = 0) -> dict:
    """Backscatter (dB) statistics on a random sample of available S1 images, split by label."""
    s1 = index[(index.sensor == "S1") & index.available]
    if s1.empty:
        return {}
    sample = s1.sample(min(n, len(s1)), random_state=seed)
    vals = {True: {"VV": [], "VH": []}, False: {"VV": [], "VH": []}}
    nan_frac = []
    for _, r in sample.iterrows():
        x = read_s1(json.loads(r["files"]))
        nan_frac.append(float(np.isnan(x).mean()))
        lab = bool(r["flooding"])
        for i, pol in enumerate(["VV", "VH"]):
            v = x[i][np.isfinite(x[i])]
            if v.size:
                vals[lab][pol].append(np.median(v))
    res = {"images_sampled": int(len(sample)), "nodata_pct_mean": round(100 * float(np.mean(nan_frac)), 2)}
    for lab, name in [(True, "flooded"), (False, "not_flooded")]:
        for pol in ["VV", "VH"]:
            a = np.array(vals[lab][pol])
            res[f"median_{pol}_dB_{name}"] = round(float(np.median(a)), 2) if a.size else None
    first = json.loads(sample.iloc[0]["files"])
    res["profile_VV"] = raster_profile(first["VV"])
    return res
