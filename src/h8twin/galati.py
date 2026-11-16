"""Galați: Sentinel-1 series from Microsoft Planetary Computer and the alert scenario (CP3).

No Google Earth Engine needed. The series comes from the Planetary Computer STAC
collection ``sentinel-1-rtc`` (terrain-corrected gamma0, 10 m, COG), read only over the
Galați tiles. Each tile has the size of a SEN12FLOOD tile (512 x 512 px at 10 m), so the
model trained on SEN12FLOOD sees inputs of the same shape.

Pipeline
  1. tile grid over Galați county (UTM 35N), 5.12 km tiles;
  2. for every S1 pass: read VV/VH over the whole grid, convert to dB, cut the tiles;
  3. per tile: the same single-image + "anomaly vs. past" features as in training
     (``features.instant_features`` / ``features.temporal_features``, past = median of
     the previous <= 6 images with the same orbit direction);
  4. M2 (trained on all of SEN12FLOOD) gives P(flood) per tile and pass;
  5. alert: a tile is flagged when P >= threshold; the county alert counts flagged tiles.

gamma0 vs. sigma0: SEN12FLOOD holds sigma0 (GEE S1_GRD), Planetary Computer holds RTC
gamma0. On flat ground gamma0 = sigma0 / cos(theta), so we shift gamma0 by
10*log10(cos 37 deg) = -0.98 dB (mid-swath incidence of IW) before computing features.
The temporal (anomaly) features are not affected by this shift.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import instant_features, temporal_features

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-1-rtc"
EPSG = 32635                      # UTM 35N covers Galați (24-30 deg E)
TILE_M = 5120                     # 512 px x 10 m, the SEN12FLOOD tile size
RES = 10.0
GAMMA_TO_SIGMA_DB = 10 * math.log10(math.cos(math.radians(37.0)))   # -0.98 dB

# Galați county, from the Danube at Galați city (south-east) to Pechea / Cudalbi (north-west)
AOI_LONLAT = (27.66, 45.37, 28.12, 45.80)
PLACES = {  # lon, lat (approximate, for labelling tiles on the map)
    "Galați (Dunăre)": (28.05, 45.43),
    "Lacul Brateș": (28.07, 45.49),
    "Confluența Siret": (27.98, 45.40),
    "Tulucești": (28.03, 45.57),
    "Slobozia Conachi": (27.78, 45.58),
    "Pechea": (27.80, 45.63),
    "Cudalbi": (27.70, 45.78),
}


@dataclass
class Grid:
    x0: float
    y0: float          # upper-left corner (UTM)
    ncol: int
    nrow: int

    @property
    def bounds(self):
        return (self.x0, self.y0 - self.nrow * TILE_M, self.x0 + self.ncol * TILE_M, self.y0)

    @property
    def shape(self):
        return (int(self.nrow * TILE_M / RES), int(self.ncol * TILE_M / RES))

    def tiles(self) -> pd.DataFrame:
        from rasterio.warp import transform

        rows = []
        for r in range(self.nrow):
            for c in range(self.ncol):
                cx = self.x0 + (c + 0.5) * TILE_M
                cy = self.y0 - (r + 0.5) * TILE_M
                lon, lat = transform(f"EPSG:{EPSG}", "EPSG:4326", [cx], [cy])
                rows.append({"tile": f"r{r}c{c}", "row": r, "col": c, "x": cx, "y": cy,
                             "lon": lon[0], "lat": lat[0]})
        t = pd.DataFrame(rows)
        t["place"] = ""
        for name, (lo, la) in PLACES.items():
            d = (t.lon - lo) ** 2 + ((t.lat - la) * 1.4) ** 2
            i = int(d.idxmin())
            t.loc[i, "place"] = (t.loc[i, "place"] + ", " if t.loc[i, "place"] else "") + name
        return t


def make_grid(aoi=AOI_LONLAT) -> Grid:
    from rasterio.warp import transform

    xs, ys = transform("EPSG:4326", f"EPSG:{EPSG}", [aoi[0], aoi[2], aoi[0], aoi[2]],
                       [aoi[1], aoi[1], aoi[3], aoi[3]])
    x0, x1 = math.floor(min(xs) / TILE_M) * TILE_M, math.ceil(max(xs) / TILE_M) * TILE_M
    y0, y1 = math.floor(min(ys) / TILE_M) * TILE_M, math.ceil(max(ys) / TILE_M) * TILE_M
    return Grid(x0, y1, int((x1 - x0) / TILE_M), int((y1 - y0) / TILE_M))


# --------------------------------------------------------------------------- STAC
def search_passes(start: str, end: str, aoi=AOI_LONLAT) -> pd.DataFrame:
    """One row per S1 item over the AOI; items of the same pass share a `pass_id`."""
    import planetary_computer
    import pystac_client

    cat = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    items = list(cat.search(collections=[COLLECTION], bbox=list(aoi), datetime=f"{start}/{end}").items())
    rows = []
    for it in items:
        p = it.properties
        rows.append({"id": it.id, "datetime": pd.Timestamp(p["datetime"]).tz_localize(None),
                     "orbit": str(p.get("sat:orbit_state", "")).upper(),
                     "rel_orbit": p.get("sat:relative_orbit"), "platform": p.get("platform"),
                     "vv": it.assets["vv"].href, "vh": it.assets["vh"].href})
    df = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)
    df["date"] = df.datetime.dt.normalize()
    df["pass_id"] = df.date.dt.strftime("%Y-%m-%d") + "_" + df.orbit.str[:3] + "_" + df.rel_orbit.astype(str)
    return df


def read_pass(group: pd.DataFrame, grid: Grid) -> np.ndarray:
    """(2, H, W) VV/VH in dB (sigma0-equivalent) over the grid; frames of one pass are mosaicked."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT

    H, W = grid.shape
    out = np.full((2, H, W), np.nan, dtype="float32")
    for _, it in group.iterrows():
        for b, key in enumerate(["vv", "vh"]):
            with rasterio.open(it[key]) as src:
                with WarpedVRT(src, crs=f"EPSG:{EPSG}", resampling=Resampling.bilinear,
                               transform=rasterio.transform.from_origin(grid.x0, grid.y0, RES, RES),
                               width=W, height=H) as vrt:
                    a = vrt.read(1).astype("float32")
                    nd = vrt.nodata
            if nd is not None:
                a[a == nd] = np.nan
            a[~np.isfinite(a) | (a <= 0)] = np.nan
            fill = np.isnan(out[b]) & np.isfinite(a)
            out[b][fill] = 10 * np.log10(a[fill]) + GAMMA_TO_SIGMA_DB
    return out


def _nanmedian(a: np.ndarray) -> np.ndarray:
    try:
        import bottleneck as bn          # ~20x faster than numpy on (k, 2, 512, 512) stacks
        return bn.nanmedian(a, axis=0)
    except ImportError:
        return np.nanmedian(a, axis=0)


def cut(img: np.ndarray, r: int, c: int) -> np.ndarray:
    n = int(TILE_M / RES)
    return img[:, r * n:(r + 1) * n, c * n:(c + 1) * n]


def galati_features(passes: pd.DataFrame, grid: Grid, max_ref: int = 6, min_valid: float = 0.8,
                    verbose: bool = True) -> pd.DataFrame:
    """Read every pass, cut tiles, compute the training features (past = same orbit direction)."""
    import time

    tiles = grid.tiles()
    history: dict[tuple[str, str], list[np.ndarray]] = {}
    rows = []
    groups = list(passes.groupby("pass_id", sort=False))
    t0 = time.time()
    for k, (pid, g) in enumerate(groups):
        try:
            img = read_pass(g, grid)
        except Exception as e:  # a broken asset should not stop the series
            print("skip", pid, type(e).__name__, str(e)[:120])
            continue
        orbit = g.orbit.iloc[0]
        for _, t in tiles.iterrows():
            x = cut(img, t.row, t.col)
            f, _ = instant_features(x)
            if f["valid_frac"] < min_valid:
                continue
            key = (t.tile, orbit)
            prev = history.get(key, [])[-max_ref:]
            ref = _nanmedian(np.stack(prev)) if prev else None
            f |= temporal_features(x, ref, len(prev))
            f |= {"tile": t.tile, "pass_id": pid, "date": g.date.iloc[0], "orbit": orbit,
                  "rel_orbit": g.rel_orbit.iloc[0]}
            rows.append(f)
            history.setdefault(key, []).append(x.astype("float16").astype("float32"))
            history[key] = history[key][-max_ref:]
        if verbose and (k % 10 == 0 or k == len(groups) - 1):
            print(f"{k + 1}/{len(groups)} passes, {len(rows)} tile-images, {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- model + alert
def train_m2(df_train: pd.DataFrame, seed: int = 0):
    """M2 exactly as in learn.run_split, trained on all SEN12FLOOD images."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.utils.class_weight import compute_sample_weight

    from .learn import feature_sets

    _, cols = feature_sets(df_train)
    y = df_train.flooding.values
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                       l2_regularization=1.0, random_state=seed)
    m.fit(df_train[cols], y, sample_weight=compute_sample_weight("balanced", y))
    return m, cols


def alert_table(gf: pd.DataFrame, model, cols, threshold: float = 0.5, min_prev: int = 3,
                event: tuple[str, str] = ("2024-09-13", "2024-09-30")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per tile-image probability, and per pass: tiles flagged, share flagged, in-event flag.

    Tiles with fewer than `min_prev` earlier images on the same orbit direction have no
    reliable "normal" yet (start of the series) and are not scored.
    """
    g = gf.copy()
    g["p_flood"] = model.predict_proba(g[cols])[:, 1]
    g["scored"] = g.n_prev >= min_prev
    g["flag"] = g.scored & (g.p_flood >= threshold)
    per = (g[g.scored].groupby(["pass_id", "date", "orbit"], as_index=False)
           .agg(tiles=("tile", "size"), flagged=("flag", "sum"), p_max=("p_flood", "max"),
                p_mean=("p_flood", "mean")))
    per["share_flagged"] = per.flagged / per.tiles
    e0, e1 = pd.Timestamp(event[0]), pd.Timestamp(event[1])
    per["in_event"] = per.date.between(e0, e1)
    return g, per.sort_values("date").reset_index(drop=True)
