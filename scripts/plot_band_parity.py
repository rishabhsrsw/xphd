#!/usr/bin/env python3
"""
plot_band_parity.py
===================
Electronic bands along a high-symmetry path, coloured by mirror parity.

Run it by editing the SETTINGS block below and running the file -- in VS
Code, Run Python File. No command line is needed. Where the text below
mentions a --flag, the setting is the same name in capitals: --first-band is
FIRST_BAND.

Reads the file written by `xphd band-parity`. Each point is a mesh k-point
lying on the path: red is sigma_h-even, blue odd, grey where the parity is
undetermined (a degenerate pair of opposite parity returned mixed). Grey
lines join consecutive points of the same energy-ordered band.

A colour flip along one band is a crossing with a band of opposite parity:
the index keeps its energy order and changes character. Those crossings are
what make the exciton parity change with Q, since

    chi_S(Q) = sum_kvc |A^S_kvc|^2 p_c(k+Q) p_v(k) .

Energies are Kohn-Sham, relative to the valence maximum over the whole zone
unless --absolute is given. The parity does not depend on the quasiparticle
correction; a scissor would move the conduction bands rigidly and change no
colour.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

# ==========================================================================
# 0. SETTINGS -- edit these, then run this file (VS Code: Run Python File)
# ==========================================================================
NPZ = "band_parity.npz"

# plot the Kohn-Sham energies as stored, unshifted
ABSOLUTE = False
EMIN = None
EMAX = None

# marker area
SIZE = 36
FIGSIZE = [4.2, 4.6]
DPI = 300
OUT = "band_parity.pdf"

# also open the figure in a window after saving it; False only saves it
SHOW = True


_have = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams.update({
    "font.size": 13,
    "font.family": "Montserrat" if "Montserrat" in _have else "DejaVu Sans",
    "mathtext.fontset": "stix", "axes.linewidth": 1.6,
})

EVEN, ODD, UNDEF = "#c0392b", "#2c5aa0", "0.6"
GREEK = {"G": r"$\Gamma$", "M": "M", "K": "K", "K'": "K$'$",
         "M'": "M$'$", "M''": "M$''$", "Q": "Q", "Q'": "Q$'$"}


d = np.load(NPZ)
E, par, ok = d["E"], d["p"], d["defined"]
idx, x = d["path_idx"], d["path_x"]
nv, bands = int(d["nv"]), d["bands"]
tx, tl = d["tick_x"], d["tick_labels"]

ref = 0.0 if ABSOLUTE else float(E[:, :nv].max())
Ep, pp, okp = E[idx] - ref, par[idx], ok[idx]

fig, ax = plt.subplots(figsize=tuple(FIGSIZE))
for n in range(E.shape[1]):
    ax.plot(x, Ep[:, n], color="0.75", lw=1.0, zorder=1)
    col = np.where(~okp[:, n], UNDEF, np.where(pp[:, n] > 0, EVEN, ODD))
    ax.scatter(x, Ep[:, n], c=col, s=SIZE, edgecolors="black",
               linewidths=0.6, zorder=3)

for t in tx:
    ax.axvline(t, color="black", lw=0.8, ls="--", zorder=0)
if not ABSOLUTE:
    ax.axhline(0.0, color="black", lw=0.6, zorder=0)
ax.set_xticks(tx)
ax.set_xticklabels([GREEK.get(str(s), str(s)) for s in tl])
ax.set_xlim(tx[0], tx[-1])
lo = EMIN if EMIN is not None else Ep.min() - 0.3
hi = EMAX if EMAX is not None else Ep.max() + 0.3
ax.set_ylim(lo, hi)
ax.set_ylabel("Energy (eV)" if ABSOLUTE else r"$E - E_{\rm VBM}$ (eV)")
ax.tick_params(direction="in", top=True, right=True)
# the legend goes in the band gap: "best" weighs lines but not the markers,
# and lands on the valence maximum, where the parities matter most
if Ep.shape[1] > nv:
    mid = 0.5 * (Ep[:, :nv].max() + Ep[:, nv:].min())
    f = min(max((mid - lo) / (hi - lo), 0.05), 0.95)
    legend_at = dict(loc="center right", bbox_to_anchor=(1.0, f))
else:
    legend_at = dict(loc="best")
ax.legend(handles=[
    Line2D([], [], marker="o", ls="", mfc=EVEN, mec="black", label="even"),
    Line2D([], [], marker="o", ls="", mfc=ODD, mec="black", label="odd"),
    Line2D([], [], marker="o", ls="", mfc=UNDEF, mec="black",
           label="undetermined")],
    fontsize=10, frameon=False, **legend_at)
ax.set_title(f"bands {bands[0]}-{bands[-1]}, " + r"$\sigma_h$ parity",
             fontsize=12)

fig.tight_layout()
fig.savefig(OUT, dpi=DPI, bbox_inches="tight")
n_undef = int((~okp).sum())
print(f"wrote {OUT}: {len(idx)} path points x {E.shape[1]} bands, "
      f"{n_undef} undetermined; energies "
      f"{'absolute' if ABSOLUTE else f'relative to the VBM at {ref:.3f} eV'}")

# ==========================================================================
# show the figure -- set SHOW = False in SETTINGS to only save it
# ==========================================================================
if SHOW:
    plt.show()
