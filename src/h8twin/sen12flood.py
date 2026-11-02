"""SEN12FLOOD ingestion: metadata index, image loading, sequence tensors, splits.

The dataset (Rambour et al., 2020; doi:10.21227/w6xz-s898) ships as
    <root>/S1list.json, <root>/S2list.json, <root>/0001 ... <root>/0336
Each JSON maps a sequence id to a dict holding sequence-level fields
(image count, folder, geography) and one dict per image with the acquisition
date, FLOODING, FULL-DATA-COVERAGE, a file prefix and, for S1, the orbit.

The parser is deliberately tolerant about key spelling so that small
differences between dataset mirrors do not break ingestion.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
S1_POLS = ["VV", "VH"]


# --------------------------------------------------------------------------- helpers
def _norm(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def _get(d: dict, *names, default=None):
    """Case/punctuation-insensitive dict lookup."""
    table = {_norm(k): v for k, v in d.items()}
    for n in names:
        v = table.get(_norm(n))
        if v is not None:
            return v
    return default


def _to_bool(v) -> bool | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "t"}:
        return True
    if s in {"false", "0", "no", "f"}:
        return False
    return None


def _is_image_entry(v) -> bool:
    return isinstance(v, dict) and any("flood" in _norm(k) for k in v)


# --------------------------------------------------------------------------- index
def parse_list(json_path: Path, sensor: str) -> pd.DataFrame:
    """Flatten S1list.json / S2list.json into one row per image."""
    with open(json_path, encoding="utf-8") as f:
        raw = json.load(f)
    seqs = raw.items() if isinstance(raw, dict) else enumerate(raw)
    rows = []
    for seq_key, seq in seqs:
        if not isinstance(seq, dict):
            continue
        folder = str(_get(seq, "folder", "foldername", "id", default=seq_key))
        geography = _get(seq, "geography", "geo", "geometry", "location")
        n_declared = _get(seq, "count", "nimages", "number", "total", "length")
        for img_key, img in seq.items():
            if not _is_image_entry(img):
                continue
            rows.append({
                "sensor": sensor,
                "seq": folder,
                "img_key": str(img_key),
                "date": pd.to_datetime(_get(img, "date", "acquisitiondate", "datetime"), errors="coerce"),
                "flooding": _to_bool(_get(img, "FLOODING", "flood", "flooded")),
                "full_coverage": _to_bool(_get(img, "FULL-DATA-COVERAGE", "fulldatacoverage", "coverage")),
                "prefix": _get(img, "filename", "prefix", "file", "name"),
                "orbit": _get(img, "orbit", "orbitdirection") if sensor == "S1" else None,
                "geography": json.dumps(geography) if isinstance(geography, (dict, list)) else geography,
                "n_declared": int(n_declared) if str(n_declared).isdigit() else None,
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["seq", "date"]).reset_index(drop=True)
        # the IEEE DataPort release keeps the original split in the folder names:
        # 4-digit folders (0001..0336) = train, unpadded folders (0..67) = test
        df["official_split"] = np.where(df["seq"].str.fullmatch(r"\d{4}"), "train", "test")
    return df


def find_files(root: Path, row: pd.Series) -> dict[str, Path]:
    """Map band/polarisation -> GeoTIFF path for one image row."""
    folder = root / row["seq"]
    prefix = str(row["prefix"] or "")
    cands = sorted(folder.glob(f"{prefix}*.tif")) if prefix else []
    keys = S1_POLS if row["sensor"] == "S1" else S2_BANDS
    out = {}
    for k in keys:
        hit = [p for p in cands if re.search(rf"(^|[_\-.]){k}([_\-.]|$)", p.stem, re.I)]
        if hit:
            out[k] = hit[0]
    return out


def build_index(root: str | Path, sensors: Iterable[str] = ("S1", "S2")) -> pd.DataFrame:
    """One row per image, with file paths and an `available` flag."""
    root = Path(root)
    frames = []
    for s in sensors:
        jp = root / f"{s}list.json"
        if not jp.exists():
            raise FileNotFoundError(f"missing {jp}")
        df = parse_list(jp, s)
        files = df.apply(lambda r: find_files(root, r), axis=1)
        keys = S1_POLS if s == "S1" else S2_BANDS
        df["n_files"] = files.map(len)
        df["available"] = df["n_files"] == len(keys)
        df["files"] = files.map(lambda d: json.dumps({k: str(v) for k, v in d.items()}))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- pixels
def read_s1(paths: dict[str, Path | str], to_db: bool = True) -> np.ndarray:
    """(2, H, W) float32 VV/VH. Linear sigma0 is converted to dB; nodata -> NaN."""
    import rasterio

    arrs = []
    for pol in S1_POLS:
        with rasterio.open(paths[pol]) as src:
            a = src.read(1).astype("float32")
            if src.nodata is not None:
                a[a == src.nodata] = np.nan
        arrs.append(a)
    h = min(a.shape[0] for a in arrs)
    w = min(a.shape[1] for a in arrs)
    x = np.stack([a[:h, :w] for a in arrs])
    x[~np.isfinite(x)] = np.nan
    if to_db and is_linear(x):
        x[x <= 0] = np.nan
        x = 10.0 * np.log10(x)
    return x


def is_linear(x: np.ndarray) -> bool:
    """Heuristic: sigma0 in linear power is mostly in (0, 1]; in dB it is mostly negative."""
    v = x[np.isfinite(x)]
    if v.size == 0:
        return False
    return float(np.nanmedian(v)) > 0 and float(np.nanpercentile(v, 95)) < 5


@dataclass
class Sequence:
    seq: str
    dates: list
    x: np.ndarray        # (T, 2, H, W) dB
    y: np.ndarray        # (T,) 0/1
    orbit: list


def load_sequence(index: pd.DataFrame, seq: str, size: int | None = 256,
                  only_full: bool = False) -> Sequence:
    """Stack the S1 time series of one sequence into a tensor (centre-cropped to `size`)."""
    rows = index[(index.sensor == "S1") & (index.seq == seq) & index.available]
    if only_full:
        rows = rows[rows.full_coverage.fillna(False)]
    rows = rows.sort_values("date")
    xs, ys, ds, ob = [], [], [], []
    for _, r in rows.iterrows():
        x = read_s1(json.loads(r["files"]))
        xs.append(x)
        ys.append(int(bool(r["flooding"])))
        ds.append(r["date"])
        ob.append(r["orbit"])
    if not xs:
        raise ValueError(f"no S1 images available for sequence {seq}")
    h = min(a.shape[1] for a in xs)
    w = min(a.shape[2] for a in xs)
    if size:
        h, w = min(h, size), min(w, size)
    def crop(a):
        t, l = (a.shape[1] - h) // 2, (a.shape[2] - w) // 2
        return a[:, t:t + h, l:l + w]
    return Sequence(seq, ds, np.stack([crop(a) for a in xs]).astype("float32"),
                    np.array(ys, dtype="int8"), ob)


# --------------------------------------------------------------------------- splits
def split_sequences(index: pd.DataFrame, test_frac: float = 0.2, seed: int = 42,
                    group_by: str = "region") -> pd.DataFrame:
    """Sequence-level split; images of one sequence never cross train/test.

    group_by="region" keeps whole geographic clusters (rounded centroid, ~2 deg)
    on one side, the spatial split the course asks for; "seq" splits by sequence only.
    """
    seqs = index.groupby("seq").agg(geography=("geography", "first")).reset_index()
    if group_by == "region":
        seqs["group"] = seqs["geography"].map(_region_key)
        seqs.loc[seqs.group.isna(), "group"] = seqs["seq"]
    else:
        seqs["group"] = seqs["seq"]
    groups = seqs["group"].drop_duplicates().sample(frac=1.0, random_state=seed).tolist()
    n_test_target = int(round(test_frac * len(seqs)))
    test, n = set(), 0
    sizes = seqs.group.value_counts()
    for g in groups:  # greedy fill, skipping groups that would overshoot the target by >15 %
        if n >= n_test_target:
            break
        if n + sizes[g] > 1.15 * n_test_target and n > 0:
            continue
        test.add(g)
        n += int(sizes[g])
    seqs["split"] = np.where(seqs.group.isin(test), "test", "train")
    return seqs[["seq", "group", "split"]]


def centroid(geo) -> tuple[float, float] | None:
    """(lon, lat) mean of all coordinates of a GeoJSON-like geometry."""
    try:
        g = json.loads(geo) if isinstance(geo, str) else geo
    except (TypeError, ValueError):
        return None
    coords = []

    def walk(o):
        if isinstance(o, (list, tuple)) and len(o) == 2 and all(isinstance(c, (int, float)) for c in o):
            coords.append(o)
        elif isinstance(o, (list, tuple)):
            for c in o:
                walk(c)
        elif isinstance(o, dict):
            for c in o.values():
                walk(c)
    walk(g)
    if not coords:
        return None
    lon, lat = np.mean(np.array(coords, dtype=float), axis=0)
    return float(lon), float(lat)


def official_split_leakage(index: pd.DataFrame) -> dict:
    """How close the official test tiles are to train tiles, and how many S1 scenes they share."""
    s1 = index[index.sensor == "S1"]
    seqs = s1.groupby("seq").agg(geo=("geography", "first"), split=("official_split", "first"))
    cen = seqs.geo.map(centroid)
    seqs = seqs[cen.notna()].assign(lon=cen.dropna().map(lambda c: c[0]), lat=cen.dropna().map(lambda c: c[1]))
    tr, te = seqs[seqs.split == "train"], seqs[seqs.split == "test"]
    if tr.empty or te.empty:
        return {}
    d = []
    for _, r in te.iterrows():
        dx = (tr.lon - r.lon) * 111.32 * np.cos(np.radians(r.lat))
        dy = (tr.lat - r.lat) * 110.57
        d.append(float(np.sqrt(dx ** 2 + dy ** 2).min()))
    d = np.array(d)
    ptr = set(s1[s1.official_split == "train"].prefix)
    pte = set(s1[s1.official_split == "test"].prefix)
    return {"official_train_seqs": int(len(tr)), "official_test_seqs": int(len(te)),
            "test_to_nearest_train_km_median": round(float(np.median(d)), 1),
            "test_within_10km_of_train": int((d < 10).sum()),
            "test_s1_scenes_also_in_train": int(len(pte & ptr)), "test_s1_scenes": int(len(pte))}


def _region_key(geo, deg: float = 2.0):
    """Round the centroid of a GeoJSON-like geometry (or lon/lat pair) to a coarse grid cell."""
    if geo is None or (isinstance(geo, float) and np.isnan(geo)):
        return None
    try:
        g = json.loads(geo) if isinstance(geo, str) else geo
    except (TypeError, ValueError):
        return None
    coords = []

    def walk(o):
        if isinstance(o, (list, tuple)) and len(o) == 2 and all(isinstance(c, (int, float)) for c in o):
            coords.append(o)
        elif isinstance(o, (list, tuple)):
            for c in o:
                walk(c)
        elif isinstance(o, dict):
            for c in o.values():
                walk(c)
    walk(g)
    if not coords:
        return None
    lon, lat = np.mean(np.array(coords, dtype=float), axis=0)
    return f"{np.floor(lat / deg) * deg:+.0f}_{np.floor(lon / deg) * deg:+.0f}"
