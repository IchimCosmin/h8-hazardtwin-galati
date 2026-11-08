"""Command-line entry point.

    python -m h8twin.cli ingest  --root D:/data/SEN12FLOOD --out outputs
    python -m h8twin.cli figure  --out outputs [--seq 1]

`ingest` writes outputs/index.csv, outputs/split.csv, outputs/stats.json.
`figure` writes outputs/fig_cp1_sequence.png (dark-pixel fraction vs. label).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .sen12flood import build_index, official_split_leakage, split_sequences
from .stats import index_stats, pixel_stats


def cmd_ingest(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    idx = build_index(a.root)
    idx.to_csv(out / "index.csv", index=False)
    split = split_sequences(idx[idx.sensor == "S1"], test_frac=a.test_frac, seed=a.seed)
    split.to_csv(out / "split.csv", index=False)
    stats = {"root": str(a.root), "index": index_stats(idx),
             "split": split.split.value_counts().to_dict(),
             "split_groups": int(split.group.nunique()),
             "official_split_leakage": official_split_leakage(idx)}
    if not a.no_pixels:
        stats["pixels_S1"] = pixel_stats(idx, n=a.n_pixels)
    (out / "stats.json").write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    print(json.dumps(stats, indent=2, default=str))


def cmd_figure(a):
    from .figures import cp1_figure

    out = Path(a.out)
    idx = pd.read_csv(out / "index.csv", parse_dates=["date"], dtype={"seq": str})
    path = cp1_figure(idx, out, seq_main=a.seq, thr_db=a.water_db, lang=a.lang)
    print("wrote", path)


def cmd_learn(a):
    import time

    import numpy as np

    from .features import extract_features
    from .learn import run_experiment

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fp, hp = out / "features.csv", out / "vv_hist.npy"
    if fp.exists() and hp.exists() and not a.recompute:
        df, hists = pd.read_csv(fp, dtype={"seq": str}), np.load(hp)
    else:
        t0 = time.time()
        idx = build_index(a.root, sensors=("S1",))
        print(f"index: {int(idx.available.sum())} S1 images with files / {len(idx)} listed", flush=True)
        df, hists = extract_features(idx, n_jobs=a.jobs, verbose=5)
        df.to_csv(fp, index=False)
        np.save(hp, hists)
        print(f"features: {df.shape} in {time.time() - t0:.0f}s", flush=True)
    seeds = list(range(a.seeds))
    res = run_experiment(df, hists, seeds=seeds, test_frac=a.test_frac)
    res.round(4).to_csv(out / a.results, index=False)  # 4 zecimale: CSV-ul raportat
    s = res[res.row_type != "run"][["row_type", "method", "metric", "mean", "ci95_lo", "ci95_hi"]]
    print(s.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    off = res[(res.row_type == "run") & (res.split == "official_leaky")][["method", "bal_acc", "acc", "f1", "auc"]]
    print("\nofficial (leaky) split:\n" + off.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("wrote", out / a.results)


def main(argv=None):
    p = argparse.ArgumentParser(prog="h8twin")
    sub = p.add_subparsers(required=True)
    i = sub.add_parser("ingest")
    i.add_argument("--root", required=True, help="folder with S1list.json, S2list.json, 0001..0336")
    i.add_argument("--out", default="outputs")
    i.add_argument("--test-frac", type=float, default=0.2)
    i.add_argument("--seed", type=int, default=42)
    i.add_argument("--n-pixels", type=int, default=60)
    i.add_argument("--no-pixels", action="store_true")
    i.set_defaults(fn=cmd_ingest)
    f = sub.add_parser("figure")
    f.add_argument("--out", default="outputs")
    f.add_argument("--seq", default=None)
    f.add_argument("--water-db", type=float, default=-18.0)
    f.add_argument("--lang", choices=["ro", "en"], default="ro")
    f.set_defaults(fn=cmd_figure)
    l = sub.add_parser("learn")
    l.add_argument("--root", required=True)
    l.add_argument("--out", default="outputs")
    l.add_argument("--seeds", type=int, default=5)
    l.add_argument("--test-frac", type=float, default=0.2)
    l.add_argument("--jobs", type=int, default=-1)
    l.add_argument("--results", default="results_v1_H8.csv")
    l.add_argument("--recompute", action="store_true")
    l.set_defaults(fn=cmd_learn)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
