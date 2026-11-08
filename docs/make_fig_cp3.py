"""Figura CP3 din docs/results_v1_H8.csv: python docs/make_fig_cp3.py docs/results_v1_H8.csv docs/fig_cp3_results.png"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

res = pd.read_csv(sys.argv[1]); out = sys.argv[2]
BLUE, ORANGE, INK, MUTED, GRID = "#3F6EA8", "#C8702A", "#1B2A41", "#6A7179", "#E3E0D8"
plt.rcParams.update({"font.family": "IBM Plex Sans", "font.size": 10.5, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK})
runs = res[res.row_type == "run"]
sp, off = runs[runs.split == "spatial_2deg"], runs[runs.split == "official_leaky"]
summ = res[(res.row_type == "summary") & (res.metric == "bal_acc")].set_index("method")
methods = list(summ.index)
fig, ax = plt.subplots(figsize=(8.4, 3.15), dpi=220)
for i, m in enumerate(methods):
    y = i
    v = sp[sp.method == m].bal_acc.values
    ax.scatter(v, np.full(len(v), y + 0.17), s=16, color=BLUE, alpha=0.45, lw=0, zorder=2)
    mu, lo, hi = summ.loc[m, ["mean", "ci95_lo", "ci95_hi"]]
    ax.plot([lo, hi], [y, y], color=BLUE, lw=2, solid_capstyle="round", zorder=3)
    ax.scatter([mu], [y], s=46, color=BLUE, edgecolor="white", lw=1.2, zorder=4)
    o = off[off.method == m].bal_acc.values[0]
    ax.scatter([o], [y - 0.17], s=40, marker="D", color=ORANGE, edgecolor="white", lw=1, zorder=4)
    ax.text(0.362, y, f"{mu:.2f}".replace(".", ","), va="center", ha="left", fontsize=10, color=INK,
            fontweight="bold" if m.startswith("M2") else "normal")
ax.axvline(0.5, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
ax.set_yticks(range(len(methods)), methods)
ax.set_ylim(len(methods) - 0.45, -0.7)
ax.set_xlim(0.355, 1.0)
ax.set_xlabel("balanced accuracy pe imagine (0,5 = șansă)")
ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.spines[["top", "right", "left"]].set_visible(False); ax.tick_params(axis="y", length=0)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",")))
h = [plt.Line2D([], [], marker="o", ls="", color=BLUE, alpha=0.45, ms=4.5, label="split spațial, fiecare seed"),
     plt.Line2D([], [], marker="o", ls="-", color=BLUE, ms=6, label="medie și IC 95 % (5 seed-uri)"),
     plt.Line2D([], [], marker="D", ls="", color=ORANGE, ms=5.5, label="split oficial (cu scurgere)")]
ax.legend(handles=h, frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, borderaxespad=0.2)
fig.tight_layout()
fig.savefig(out, facecolor="white")
print("wrote", out)
