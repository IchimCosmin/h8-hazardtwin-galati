"""CP3 experiment: method vs. baselines on SEN12FLOOD S1, spatial splits over seeds.

Task: per-image flood label (FLOODING) from Sentinel-1 VV/VH.

Methods
  B0  majority class                                   (no learning)
  B1  global Otsu threshold on VV + tuned area cut-off (single image, SOA baseline)
  B2  change threshold: share of pixels with VV drop > 3 dB vs. earlier images
  M1  gradient boosting on single-image features       (learning, no time)
  M2  gradient boosting on single-image + anomaly-vs-past features (proposed)

Evaluation: region-grouped spatial split (2 deg cells), repeated over seeds;
95 % CI across seeds (t distribution) and paired differences vs. the baselines.
The leaky official split is run once, for comparison with Rambour et al. (2020).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.utils.class_weight import compute_sample_weight

from .features import HIST_EDGES
from .sen12flood import split_sequences

META = {"seq", "img_key", "date", "orbit", "flooding", "official_split", "geography", "t_index"}
TEMPORAL = {"n_prev", "d_vv_mean", "d_vv_q5", "d_vv_q50", "d_vh_mean", "d_vh_q5", "d_vh_q50",
            "d_vv_lt3", "d_vv_lt6", "d_vh_lt3", "d_vh_lt6", "new_dark_vv18", "new_dark_vh25"}
METHODS = ["B0 majoritar", "B1 prag Otsu (o imagine)", "B2 prag pe schimbare (serie)",
           "M1 GBM o imagine", "M2 GBM + anomalie temporală"]


def otsu_from_hist(hist: np.ndarray, edges: np.ndarray = HIST_EDGES) -> float:
    """Otsu threshold from a histogram (counts per bin)."""
    p = hist.astype(float) / max(hist.sum(), 1)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    w1 = 1 - w0
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu_t * w0 - mu) ** 2 / (w0 * w1)
    between[~np.isfinite(between)] = -1
    return float(edges[1:][np.argmax(between)])


def frac_below(hists: np.ndarray, thr: float, edges: np.ndarray = HIST_EDGES) -> np.ndarray:
    k = np.searchsorted(edges[1:], thr, side="right")
    tot = hists.sum(1)
    return hists[:, :k].sum(1) / np.maximum(tot, 1)


def tune_cutoff(score: np.ndarray, y: np.ndarray) -> float:
    """Cut-off on a 1-D score that maximises balanced accuracy on training data."""
    s = np.nan_to_num(score, nan=0.0)
    cands = np.unique(np.quantile(s, np.linspace(0, 1, 201)))
    best, best_t = -1, cands[0]
    for t in cands:
        b = balanced_accuracy_score(y, (s > t).astype(int))
        if b > best:
            best, best_t = b, t
    return float(best_t)


def metrics(y, pred, score) -> dict:
    out = {"bal_acc": balanced_accuracy_score(y, pred), "acc": accuracy_score(y, pred),
           "f1": f1_score(y, pred, zero_division=0), "precision": precision_score(y, pred, zero_division=0),
           "recall": recall_score(y, pred, zero_division=0)}
    try:
        out["auc"] = roc_auc_score(y, np.nan_to_num(score, nan=0.0))
    except ValueError:
        out["auc"] = np.nan
    return out


def feature_sets(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    feats = [c for c in df.columns if c not in META]
    inst = [c for c in feats if c not in TEMPORAL]
    return inst, feats


def run_split(df: pd.DataFrame, hists: np.ndarray, is_test: np.ndarray, seed: int) -> list[dict]:
    tr, te = ~is_test, is_test
    y_tr, y_te = df.flooding.values[tr], df.flooding.values[te]
    rows = []

    # B0 majority
    maj = int(y_tr.mean() > 0.5)
    rows.append((METHODS[0], np.full(te.sum(), maj), np.full(te.sum(), 0.5)))

    # B1 global Otsu on pooled training histograms + tuned area cut-off
    thr = otsu_from_hist(hists[tr].sum(0))
    s_tr, s_te = frac_below(hists[tr], thr), frac_below(hists[te], thr)
    c = tune_cutoff(s_tr, y_tr)
    rows.append((METHODS[1], (s_te > c).astype(int), s_te))

    # B2 change threshold (DeVries-style), no learning beyond the cut-off
    s_tr, s_te = df.d_vv_lt3.values[tr], df.d_vv_lt3.values[te]
    c = tune_cutoff(s_tr, y_tr)
    rows.append((METHODS[2], (np.nan_to_num(s_te) > c).astype(int), s_te))

    # M1 / M2 gradient boosting, class-balanced
    inst, full = feature_sets(df)
    for name, cols in [(METHODS[3], inst), (METHODS[4], full)]:
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                           l2_regularization=1.0, random_state=seed)
        m.fit(df.loc[tr, cols], y_tr, sample_weight=compute_sample_weight("balanced", y_tr))
        p = m.predict_proba(df.loc[te, cols])[:, 1]
        rows.append((name, (p > 0.5).astype(int), p))

    out = []
    for name, pred, score in rows:
        r = {"method": name, "seed": seed, "n_test_img": int(te.sum()),
             "n_test_seq": int(df.seq[te].nunique()), "test_pos_rate": float(y_te.mean()),
             "otsu_thr_db": thr if name == METHODS[1] else np.nan}
        r |= metrics(y_te, pred, score)
        out.append(r)
    return out


def ci95(v: np.ndarray) -> tuple[float, float, float]:
    v = np.asarray(v, float)
    m = v.mean()
    if len(v) < 2:
        return m, np.nan, np.nan
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return m, m - h, m + h


def run_experiment(df: pd.DataFrame, hists: np.ndarray, seeds=(0, 1, 2, 3, 4),
                   test_frac: float = 0.2) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    res = []
    for s in seeds:
        sp = split_sequences(df, test_frac=test_frac, seed=s).set_index("seq").split
        is_test = df.seq.map(sp).eq("test").values
        for r in run_split(df, hists, is_test, s):
            res.append({"row_type": "run", "split": "spatial_2deg"} | r)
    # leaky official split, once, for comparison with the literature
    is_test = df.official_split.eq("test").values
    if 0 < is_test.sum() < len(df):
        for r in run_split(df, hists, is_test, 0):
            res.append({"row_type": "run", "split": "official_leaky"} | r)
    runs = pd.DataFrame(res)

    summ = []
    sp = runs[runs.split == "spatial_2deg"]
    for name, g in sp.groupby("method", sort=False):
        for met in ["bal_acc", "f1", "auc", "acc"]:
            m, lo, hi = ci95(g[met].values)
            summ.append({"row_type": "summary", "split": "spatial_2deg", "method": name, "metric": met,
                         "mean": m, "ci95_lo": lo, "ci95_hi": hi, "n_seeds": len(g)})
    piv = sp.pivot(index="seed", columns="method", values="bal_acc")
    for a, b in [(METHODS[4], METHODS[1]), (METHODS[4], METHODS[3]), (METHODS[3], METHODS[1]),
                 (METHODS[4], METHODS[2])]:
        m, lo, hi = ci95((piv[a] - piv[b]).values)
        summ.append({"row_type": "paired_diff", "split": "spatial_2deg", "method": f"{a} − {b}",
                     "metric": "bal_acc", "mean": m, "ci95_lo": lo, "ci95_hi": hi, "n_seeds": len(piv)})
    return pd.concat([runs, pd.DataFrame(summ)], ignore_index=True)
