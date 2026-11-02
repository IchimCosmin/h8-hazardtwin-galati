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
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
