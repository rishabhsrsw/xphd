#!/usr/bin/env python3
"""
plot_arpes.py
=============
Plot the exciton photoemission written by `xphd arpes`.

Run it by editing the SETTINGS block below and running the file -- in VS
Code, Run Python File. No command line is needed. Where the text below
mentions a --flag, the setting is the same name in capitals: --first-band is
FIRST_BAND.

Two panels:
  (a) I(k, E) along the high-symmetry path -- the band-like view an
      experiment reports;
  (b) a constant-energy cut of I across the whole Brillouin zone, at the
      energy of the brightest feature unless --ecut is given.

The exciton appears BELOW the conduction band by its binding energy, at the
momentum of its electron. A population sitting at K in the transport
therefore shows up as intensity at K in (a), and as six spots at the zone
corners in (b).

Intensities are relative: the photoemission matrix element is constant, so
the map carries no information on absolute cross-sections or on the
polarisation dependence of the measurement.

Usage:"""
from __future__ import annotations


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager
from scipy.interpolate import griddata

# ==========================================================================
# 0. SETTINGS -- edit these, then run this file (VS Code: Run Python File)
# ==========================================================================
NPZ = "arpes.npz"

# eV for the constant-energy cut; default is the peak
ECUT = None

# eV half-width of the constant-energy cut
WINDOW = 0.05
CMAP = "afmhot"

# exponent on the intensity, to lift weak features
GAMMA = 0.5
RES = 300
LIMIT = 0.8
DPI = 400
OUT = "arpes.pdf"

# also open the figure in a window after saving it; False only saves it
SHOW = True


_have = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams.update({
    "font.size": 13,
    "font.family": "Montserrat" if "Montserrat" in _have else "DejaVu Sans",
    "mathtext.fontset": "stix",
    "xtick.direction": "in",
    "ytick.direction": "in",
})

ROT = np.radians(-30.0)
CR, SR = np.cos(ROT), np.sin(ROT)


def to_rot(qf):
    x = qf[:, 0] + 0.5 * qf[:, 1]
    y = (np.sqrt(3.0) / 2.0) * qf[:, 1]
    return x * CR - y * SR, x * SR + y * CR



d = np.load(NPZ)
I, om, kf = d["I"], d["omega"], d["kf"]
Ip, ticks, path = d["I_path"], d["ticks"], [str(x) for x in d["path"]]
T, t = float(d["T"]), float(d["time"])
tag = f"T = {T:.0f} K" if np.isfinite(T) else f"t = {t:.0f} fs"
print(f"{NPZ}: I {I.shape}, path {Ip.shape}, {tag}")

ip, ie = np.unravel_index(np.argmax(I), I.shape)
ecut = ECUT if ECUT is not None else om[ie]
print(f"   brightest feature at E = {om[ie]:.3f} eV; cut at {ecut:.3f} eV")

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2),
                         gridspec_kw={"width_ratios": [1.25, 1.0]})

# ---- (a) along the path
ax = axes[0]
x = np.arange(len(Ip))
ax.pcolormesh(x, om, (Ip.T / max(Ip.max(), 1e-300)) ** GAMMA,
              cmap=CMAP, shading="auto", rasterized=True)
for tk in ticks[1:-1]:
    ax.axvline(tk, color="white", lw=0.8, ls="--")
ax.set_xticks(ticks)
ax.set_xticklabels([l.replace("G", r"$\Gamma$") for l in path])
ax.axhline(ecut, color="cyan", lw=0.9, ls=":")
ax.set_ylabel("Energy (eV)")
ax.set_title(f"(a)  {tag}", fontsize=12, loc="left")

# ---- (b) constant-energy cut
ax = axes[1]
sel = np.abs(om - ecut) <= WINDOW
cut = I[:, sel].sum(axis=1)
w = kf[:, :2] - np.rint(kf[:, :2])
qx, qy, dd = [], [], []
for dx in (-2, -1, 0, 1, 2):
    for dy in (-2, -1, 0, 1, 2):
        qx.append(w[:, 0] + dx)
        qy.append(w[:, 1] + dy)
        dd.append(cut)
kx, ky = to_rot(np.column_stack([np.concatenate(qx), np.concatenate(qy)]))
xi = np.linspace(-LIMIT, LIMIT, RES)
X, Y = np.meshgrid(xi, xi)
Z = griddata((kx, ky), np.concatenate(dd), (X, Y), method="cubic",
             fill_value=0.0)
Z = np.clip(Z, 0, None)
ax.pcolormesh(X, Y, (Z / max(Z.max(), 1e-300)) ** GAMMA, cmap=CMAP,
              shading="auto", rasterized=True)
v = np.array([[1/3, 1/3], [-1/3, 2/3], [-2/3, 1/3],
              [-1/3, -1/3], [1/3, -2/3], [2/3, -1/3]])
vx, vy = to_rot(v)
ax.add_patch(patches.Polygon(np.column_stack([vx, vy]), closed=True,
                             fill=False, edgecolor="white", lw=1.4))
ax.set_xlim(-LIMIT, LIMIT)
ax.set_ylim(-LIMIT, LIMIT)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title(f"(b)  E = {ecut:.2f} eV", fontsize=12, loc="left")

fig.tight_layout()
fig.savefig(OUT, dpi=DPI, bbox_inches="tight")
print(f"saved {OUT}")

# ==========================================================================
# show the figure -- set SHOW = False in SETTINGS to only save it
# ==========================================================================
if SHOW:
    plt.show()
