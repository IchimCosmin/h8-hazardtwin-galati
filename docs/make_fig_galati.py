"""Figura CP3 Galați: scenariul de alertă pe seria Sentinel-1 (ieșirile comenzii `galati`).

    python docs/make_fig_galati.py docs/galati docs/fig_cp3_galati.png

Citește galati_passes.csv (pragul primului --thresholds, implicit 0,5) și galati_summary.json.

a) procentul de tile-uri de 5 km cu P(inundație) >= prag, pe fiecare trecere S1; trecerile din fereastra
   evenimentului EMSR758 (13–30 sept. 2024) sunt portocalii.
b) alarmele false pe luni: procentul de tile-uri semnalate în trecerile fără eveniment.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

src, out = Path(sys.argv[1]), sys.argv[2]
BLUE, ORANGE, INK, MUTED, GRID = "#3F6EA8", "#C8702A", "#1B2A41", "#6A7179", "#E3E0D8"
plt.rcParams.update({"font.family": "IBM Plex Sans", "font.size": 10, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
summ = json.loads((src / "galati_summary.json").read_text())
thr = float(next(iter(summ["thresholds"])))
e0, e1 = (pd.Timestamp(d) for d in summ["event"])
per = pd.read_csv(src / "galati_passes.csv", parse_dates=["date"])
per["share"] = 100 * per.flagged / per.tiles
per["event"] = per.date.between(e0, e1)
quiet = per[~per.event]
fmt = lambda v: f"{v:.0f}"

fig = plt.figure(figsize=(7.4, 2.78), dpi=285)
ax = fig.add_axes([0.075, 0.17, 0.585, 0.68])
bx = fig.add_axes([0.75, 0.17, 0.24, 0.68])

ax.axvspan(e0, e1, color=ORANGE, alpha=0.15, lw=0)
for ev, col, lw in [(False, BLUE, 1.6), (True, ORANGE, 2.6)]:
    d = per[per.event == ev]
    ax.vlines(d.date, 0, d.share, color=col, lw=lw)
    ax.scatter(d.date, d.share, s=10 if not ev else 22, color=col, zorder=3)
ax.set_ylim(0, 128); ax.set_yticks(range(0, 101, 20))
ax.set_ylabel(f"% tile-uri cu P ≥ {thr:g}".replace(".", ","))
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m.%Y"))
ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.set_title(f"a) % tile-uri în alertă pe fiecare trecere S1 ({len(per)} treceri)",
             loc="left", fontsize=10, fontweight="bold")
top = per[per.event].sort_values("share").iloc[-1]
higher = 100 * (quiet.share >= top.share).mean()
ax.annotate(f"{top.date:%d.%m.%Y} (viitura): {fmt(top.share)} %\n"
            f"{fmt(higher)} % din trecerile fără eveniment au la fel de multe",
            xy=(top.date, top.share), xytext=(top.date - pd.Timedelta(days=12), 127), color=ORANGE, fontsize=8.5,
            va="top", ha="right", bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="none"),
            arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.8))

m = quiet.assign(m=quiet.date.dt.month).groupby("m").share.mean().reindex(range(1, 13))
lab = ["ian", "feb", "mar", "apr", "mai", "iun", "iul", "aug", "sep", "oct", "nov", "dec"]
bx.bar(range(12), m.values, color=BLUE, width=0.72)
bx.axhline(quiet.share.mean(), color=INK, lw=0.9, ls="--")
bx.text(-0.4, 101, f"- - medie {fmt(quiet.share.mean())} %", ha="left", va="bottom", fontsize=8, color=INK)
bx.set_xticks(range(12), lab, fontsize=7, rotation=90)
bx.set_ylim(0, 128); bx.set_yticks(range(0, 101, 20))
bx.yaxis.grid(True, color=GRID, lw=0.8); bx.set_axisbelow(True)
bx.spines[["top", "right"]].set_visible(False)
bx.set_title("b) Alarme false pe luni", loc="left", fontsize=10, fontweight="bold")
bx.set_ylabel("% tile-uri, treceri fără eveniment", fontsize=8.5)
fig.savefig(out, facecolor="white")
print("wrote", out, "| passes", len(per), "| quiet passes with alert:", int((quiet.flagged > 0).sum()), "/",
      len(quiet), f"| quiet share {quiet.share.mean():.1f} % | event", per[per.event].share.round(1).tolist(),
      f"| quiet >= max event: {higher:.1f} %")
