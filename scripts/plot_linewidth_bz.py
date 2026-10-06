"""
plot_linewidth_bz.py
====================
Exciton linewidth over the Brillouin zone, as a smooth map.

Run it by editing the SETTINGS block below and running the file -- in VS
Code, Run Python File. No command line is needed.

It reads the linewidth field written by

    xphd sweep "GI_ExcPh_Q*.npz" ... --unfold <SAVE> -o lw_77K.npz

which already holds every Q of the full zone, the linewidths as FWHM in eV,
and the temperatures they were computed at. So nothing here is typed in by
hand: the zone is not unfolded from a list of 61 points, and the colourbar
label takes its units and temperature from the file. A field that holds only
the irreducible points is refused -- run `xphd unfold` on it first.

Filling the map: the Q-points are placed in Cartesian coordinates, repeated
over the neighbouring zones so the interpolation sees the periodic
boundaries, and interpolated onto a fine pixel grid (cubic, with nearest
filling any gaps). The white hexagon is the first Brillouin zone.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata

import xphd

# ==========================================================================
# 0. SETTINGS -- edit these, then run this file (VS Code: Run Python File)
# ==========================================================================
# the unfolded field from xphd sweep ... --unfold
FIELD = "lw_77K.npz"

# which exciton, counted from 1 as in the linewidth tables
STATE = 1

# which of the field's temperatures, in K; None takes the first
TEMPERATURE = None

# the map
GRID_RES = 800
METHOD = "cubic"
LEVELS = 150
ROTATION_DEG = 30.0          # rotates the zone so its flat edges lie along x
ZOOM = 0.65
CMAP = "l_turqsat1"          # a `colormaps` name, or any matplotlib colormap
HEX_COLOR = "white"
HEX_WIDTH = 1.5

FIGSIZE = [4.5, 4.5]
DPI = 300
OUT = None                   # None: Exciton_<STATE>_2D_Map_<T>K.png

# also open the figure in a window after saving it; False only saves it
SHOW = True

_have = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = ("Montserrat" if "Montserrat" in _have
                               else "DejaVu Sans")
plt.rcParams["mathtext.fontset"] = "stix"

# ==========================================================================
# 1. LOAD THE LINEWIDTH FIELD
# ==========================================================================
print(f"Loading {FIELD} ...")
fld = xphd.LinewidthField.load(FIELD)
Q = np.asarray(fld.Q_red, float)[:, :2]
n = int(round(np.sqrt(len(Q))))
if n * n != len(Q):
    raise SystemExit(f"{FIELD} holds {len(Q)} Q-points, not a full n x n "
                     f"zone. Unfold it first: xphd unfold {FIELD} "
                     f"--savepath <SAVE> -o <unfolded.npz>")
Ts = np.atleast_1d(fld.T)
it = 0 if TEMPERATURE is None else int(np.argmin(np.abs(Ts - TEMPERATURE)))
if TEMPERATURE is not None and abs(Ts[it] - TEMPERATURE) > 0.5:
    raise SystemExit(f"{TEMPERATURE} K is not in {FIELD}; it holds "
                     f"{np.round(Ts, 1).tolist()} K")
T = float(Ts[it])
nexc = fld.LW.shape[2]
if not 1 <= STATE <= nexc:
    raise SystemExit(f"STATE must be 1..{nexc}")
gamma = fld.LW[it, :, STATE - 1] * 1e3                       # FWHM, meV
iG = int(np.argmin(np.linalg.norm((Q + 0.5) % 1 - 0.5, axis=1)))
dK = np.linalg.norm(((Q - 1 / 3) + 0.5) % 1 - 0.5, axis=1)
iK = int(np.argmin(dK))
print(f"   {n}x{n} zone, T = {T:.1f} K, state {STATE}: FWHM {gamma.min():.3f}"
      f" .. {gamma.max():.3f} meV; at Gamma {gamma[iG]:.3f}, at K "
      f"{gamma[iK]:.3f} meV")

# ==========================================================================
# 2. CARTESIAN COORDINATES, REPEATED OVER THE NEIGHBOURING ZONES
# ==========================================================================
print("Tiling the Brillouin zone ...")
Qw = (Q + 0.5) % 1.0 - 0.5                                   # nearest image
shifts = np.array([(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)], float)
tiled = np.concatenate([Qw + s for s in shifts])
vals = np.tile(gamma, len(shifts))
kx = tiled[:, 0] + 0.5 * tiled[:, 1]                         # b1 = (1, 0)
ky = np.sqrt(3.0) / 2.0 * tiled[:, 1]                        # b2 = (1/2, sqrt3/2)
th = np.radians(ROTATION_DEG)
kx, ky = kx * np.cos(th) - ky * np.sin(th), kx * np.sin(th) + ky * np.cos(th)

# ==========================================================================
# 3. INTERPOLATE ONTO A FINE GRID AND PLOT
# ==========================================================================
print("Plotting the 2D map ...")
fig, ax = plt.subplots(figsize=tuple(FIGSIZE), dpi=DPI)
xi = np.linspace(-1.1, 1.1, GRID_RES)
X, Y = np.meshgrid(xi, xi)
Z = griddata((kx, ky), vals, (X, Y), method=METHOD)
Zn = griddata((kx, ky), vals, (X, Y), method="nearest")
Z = np.where(np.isnan(Z), Zn, Z)

try:
    import colormaps as _cm
    cmap = getattr(_cm, CMAP)
except (ImportError, AttributeError):
    cmap = plt.get_cmap(CMAP if CMAP in plt.colormaps() else "viridis")
contour = ax.contourf(X, Y, Z, levels=LEVELS, cmap=cmap)

r_hex = 1.0 / np.sqrt(3.0)                                  # Gamma to K
ang = np.linspace(0, 2 * np.pi, 7) + np.pi / 6 + th
ax.plot(r_hex * np.cos(ang), r_hex * np.sin(ang), color=HEX_COLOR,
        linewidth=HEX_WIDTH, zorder=10)
ax.set_xlim(-ZOOM, ZOOM)
ax.set_ylim(-ZOOM, ZOOM)
ax.set_aspect("equal")
ax.axis("off")

cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(contour, cax=cax)
cbar.set_label(f"Exciton linewidth, FWHM (meV) at {T:.0f} K", fontsize=9,
               rotation=270, labelpad=20)

out = OUT or f"Exciton_{STATE}_2D_Map_{T:.0f}K.png"
plt.tight_layout()
plt.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
print(f"saved {out}")

# ==========================================================================
# show the figure -- set SHOW = False in SETTINGS to only save it
# ==========================================================================
if SHOW:
    plt.show()
