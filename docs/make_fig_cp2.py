"""Figura CP2: reprezentarea „anomalie față de imaginile anterioare” pe o secvență SEN12FLOOD.

    python docs/make_fig_cp2.py <root SEN12FLOOD> docs/fig_cp2_representation.jpg [seq] [data]
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, "src")
from h8twin.sen12flood import build_index, read_s1  # noqa: E402

root, out = sys.argv[1], sys.argv[2]
seq = sys.argv[3] if len(sys.argv) > 3 else "1"
when = sys.argv[4] if len(sys.argv) > 4 else "2019-03-18"
INK, MUTED, BLUE, ORANGE = "#1B2A41", "#6A7179", "#3F6EA8", "#C8702A"
plt.rcParams.update({"font.family": "IBM Plex Sans", "font.size": 10, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})

idx = build_index(root, sensors=("S1",))
r = idx[(idx.seq == seq) & idx.available].sort_values("date").reset_index(drop=True)
imgs = [read_s1(json.loads(f)) for f in r.files]
h, w = min(a.shape[1] for a in imgs), min(a.shape[2] for a in imgs)
imgs = [a[:, :h, :w] for a in imgs]
i = int(np.where(r.date.dt.strftime("%Y-%m-%d") == when)[0][0])
prev = [j for j in range(i) if r.orbit[j] == r.orbit[i]][-6:]          # same rule as features.py
ref = np.nanmedian(np.stack([imgs[j] for j in prev]), axis=0)
vv, rv = imgs[i][0], ref[0]
d = vv - rv
ok = np.isfinite(d)
share = 100 * (d[ok] < -3).mean()
dark = 100 * (vv[ok] < -18).mean()

cmap = LinearSegmentedColormap.from_list("div", [BLUE, "#E9E7E2", ORANGE])
fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.3), dpi=220)
def ro(v):
    return f"{v:.1f}".replace(".", ",")


panels = [(rv, "gray", (-25, 0), f"a) Referință: mediana a {len(prev)} treceri\nanterioare, aceeași orbită", None),
          (vv, "gray", (-25, 0), f"b) {r.date[i].strftime('%d.%m.%Y')}, etichetat inundat\n{ro(dark)} % pixeli VV < −18 dB", "VV (dB)"),
          (d, cmap, (-8, 8), f"c) Anomalie Δ = b − a\n{ro(share)} % pixeli cu Δ < −3 dB", "Δ VV (dB)")]
for ax, (img, cm, (lo, hi), title, lab) in zip(axs, panels):
    im = ax.imshow(img, cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
    ax.set_title(title, fontsize=9.5, loc="left", color=INK)
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    if lab is None:
        cb.ax.set_visible(False)
        continue
    cb.set_label(lab, color=MUTED, fontsize=8.5); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=8)
fig.tight_layout(w_pad=1.2)
fig.savefig(out, facecolor="white", bbox_inches="tight", pad_inches=0.05)
print(f"seq {seq} {when}: ref={len(prev)} imagini, dark={dark:.1f}%, delta<-3dB={share:.1f}%  -> {out}")
