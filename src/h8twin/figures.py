"""CP1 figure: does the per-image FLOODING label show up in the S1 pixels?"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .sen12flood import load_sequence

BLUE, ORANGE, INK, MUTED, GRID = "#3F6EA8", "#C8702A", "#1B2A41", "#6A7179", "#E3E0D8"


def dark_fraction(x: np.ndarray, thr_db: float) -> np.ndarray:
    """% of valid VV pixels below `thr_db`, per time step. x: (T, 2, H, W) in dB."""
    vv = x[:, 0].reshape(len(x), -1)
    valid = np.isfinite(vv)
    return 100 * np.where(valid, vv < thr_db, False).sum(1) / np.maximum(valid.sum(1), 1)


def cp1_figure(index: pd.DataFrame, out: Path, seq_main: str | None = None,
               thr_db: float = -18.0, lang: str = "ro") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    names = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({"font.family": "IBM Plex Sans" if "IBM Plex Sans" in names else "DejaVu Sans",
                         "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK})
    T = {"ro": dict(flood="etichetat inundat", dry="etichetat neinundat",
                    y=f"% pixeli VV < {thr_db:g} dB", a="a) Secvența {s}: apa „întunecată” crește după etichetă",
                    b="b) Mediana pe secvență: {k} din {n} cresc clar", seq="secvența"),
         "en": dict(flood="labelled flooded", dry="labelled not flooded",
                    y=f"% pixels VV < {thr_db:g} dB", a="a) Sequence {s}: dark water rises after the label",
                    b="b) Per-sequence median: {k} of {n} rise clearly", seq="sequence")}[lang]

    s1 = index[(index.sensor == "S1") & index.available]
    mixed = s1.groupby("seq").flooding.agg(lambda v: 0 < v.mean() < 1)
    seqs = [s for s in mixed[mixed].index]
    if not seqs:
        raise ValueError("no available sequence with both labels")
    seq_main = seq_main or max(seqs, key=lambda s: (s1.seq == s).sum())

    per_seq = {}
    for s in seqs:
        sq = load_sequence(index, s, size=None)
        per_seq[s] = (sq, dark_fraction(sq.x, thr_db))

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.2, 3.3), dpi=220,
                                 gridspec_kw={"width_ratios": [1.75, 1]})
    sq, dk = per_seq[seq_main]
    pos = np.arange(len(dk))
    cols = [ORANGE if y else BLUE for y in sq.y]
    ax.bar(pos, dk, width=0.62, color=cols, edgecolor="white", linewidth=1.5)
    ax.set_xticks(pos, [pd.Timestamp(d).strftime("%d.%m") for d in sq.dates], rotation=0, fontsize=7.5)
    ax.set_ylabel(T["y"])
    ax.set_title(T["a"].format(s=seq_main), loc="left", fontsize=9.5, fontweight="bold")
    ax.bar([0], [0], color=BLUE, label=T["dry"])
    ax.bar([0], [0], color=ORANGE, label=T["flood"])
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)

    rows = sorted(per_seq, key=lambda s: str(s))
    k = 0
    for i, s in enumerate(rows):
        q, d = per_seq[s]
        pre, post = np.median(d[q.y == 0]), np.median(d[q.y == 1])
        k += int(post > 1.5 * pre + 0.5)
        bx.plot([pre, post], [i, i], color=GRID, lw=2.5, zorder=1)
        bx.scatter([post], [i], s=60, color=ORANGE, zorder=2, edgecolor="white", linewidth=1)
        bx.scatter([pre], [i], s=30, color=BLUE, zorder=3, edgecolor="white", linewidth=1)
    bx.set_yticks(range(len(rows)), [f"{T['seq']} {s}" for s in rows])
    bx.set_xlabel(T["y"])
    bx.set_title(T["b"].format(k=k, n=len(rows)), loc="left", fontsize=9.5, fontweight="bold")
    bx.xaxis.grid(True, color=GRID, lw=0.8)
    bx.set_axisbelow(True)
    bx.set_ylim(-0.6, len(rows) - 0.4)
    for a in (ax, bx):
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(w_pad=2.5)
    path = Path(out) / ("fig_cp1_sequence.png" if lang == "ro" else f"fig_cp1_sequence_{lang}.png")
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path
