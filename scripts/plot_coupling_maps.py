"""
plot_coupling_maps.py
=====================
Squared exciton-phonon coupling |G|^2 from one initial exciton at Q = 0 into
a chosen final exciton band at q, one panel per phonon branch (a grid, so
six branches give 2 x 3 and nine give 3 x 3), with the sigma_h parity of the
excitons overlaid.

Run it by editing the SETTINGS block below and running the file -- in VS
Code, Run Python File. No command line is needed.

The overlays
------------
"contour"    cyan lines where the lowest sigma_h-even and the lowest
             sigma_h-odd exciton bands cross: the lowest band is even on one
             side and odd on the other. Found as the zero of
             delta(q) = E_lowest_odd - E_lowest_even, interpolated smoothly,
             so the line is not stair-stepped by the mesh.
"forbidden"  cyan hatching where the coupling of THAT panel's branch is
             forbidden by symmetry: an even phonon connects excitons of equal
             parity, an odd phonon excitons of opposite parity, so the
             coupling is forbidden where  p_phonon(q) * p_final(q) != p_initial.
             Needs the phonon parities (LABELS). The zeros of each map should
             sit exactly under the hatching.
"both", "none".

The parity of a phonon branch is exact at every q, but the branch INDEX is
energy-ordered and changes character where two branches cross, so the
hatched region of a panel need not be a single domain.

What it prints
--------------
The initial state's parity, and for every panel the share of that panel's
coupling that lies in the forbidden region (zero to numerical precision when
the selection rule holds).
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

import xphd
from xphd.selection import grid, zone

# ==========================================================================
# 0. SETTINGS -- edit these, then run this file (VS Code: Run Python File)
# ==========================================================================
ARCHIVE = "GI_ExcPh_Q0001.npz"
PARITY = "parity.npz"                  # xphd parity ... --nstates >= the archive's states
LABELS = "mode_labels_elph.npy"        # phonon parities from labels_from_elph.py; "" for none

STATE = 1                              # initial exciton at Q = 0 (1-based)
FINAL = [1]                            # final state(s) summed over (1-based); [1] = lowest band

OVERLAY = "contour"                    # "contour", "forbidden", "both" or "none"
NCOLS = 3                              # panels per row; rows follow from the branch count

SCALE = 1e6                            # eV^2 -> meV^2
CMAP = "afmhot"
CLIP = 99.0                            # per-panel colour maximum: this percentile of the map
LIMIT = 0.8                            # half-width of the plotted region, in |b|

CONTOUR_COLOR = "cyan"
CONTOUR_LW = 1.6
FIGSIZE = None                         # None: 3.6 x 3.2 inches per panel
LETTERS = True                         # (a), (b), ... above each panel
TAG_BRANCH = True                      # small "nu = k" tag inside each panel
DPI = 300
OUT = "coupling_maps.pdf"
SHOW = True

# ==========================================================================
# 1. LOAD
# ==========================================================================
arc = xphd.ExcPhArchive(ARCHIVE)
g2 = np.asarray(arc.grid("g2"))[arc._i, arc._j]                 # (nq, nmod, init, final)
nq, nmod, ne, _ = g2.shape
S = STATE - 1
F = [f - 1 for f in FINAL]
if not 0 <= S < ne or any(not 0 <= f < ne for f in F):
    raise SystemExit(f"STATE/FINAL must lie in 1..{ne}")
qgrid = np.mod(np.asarray(arc.q_red, float), 1.0)
print(f"{ARCHIVE}: {nq} q-points, {nmod} branches, {ne} states; initial state "
      f"{STATE}, final {FINAL}")

d = np.load(PARITY)
Qp = np.asarray(d["Q_red"], float)
chi = np.asarray(d["chi"], float)
Ep = np.asarray(d["E"], float)
if chi.shape[1] < max(F) + 1:
    raise SystemExit(f"{PARITY} holds {chi.shape[1]} states per Q, FINAL needs "
                     f"{max(F) + 1}: rerun  xphd parity ... --nstates {ne}")

# ---- the initial state's parity, at Gamma ------------------------------------
iG = int(np.argmin(np.linalg.norm(((Qp[:, :2] + 0.5) % 1) - 0.5, axis=1)))
p_lam = float(np.sign(chi[iG, S]))
print(f"initial state {STATE} at Gamma: chi = {chi[iG, S]:+.3f}  ->  "
      f"{'even' if p_lam > 0 else 'odd'}")

# ---- match the archive's q to the parity file, by periodic distance ------------
dd = qgrid[:, None, :2] - np.mod(Qp[:, :2], 1.0)[None, :, :]
dd -= np.rint(dd)
dist = np.linalg.norm(dd, axis=-1)
idx = np.argmin(dist, axis=1)
worst = float(dist[np.arange(nq), idx].max())
if worst > 1e-5:
    raise SystemExit(f"the archive's q-mesh and {PARITY} differ: worst match "
                     f"{worst:.2e}")
p_fin = np.sign(chi[idx][:, F[0]])                               # parity of the final band
ok_fin = np.abs(chi[idx][:, F[0]]) > 0.5

# ---- phonon parity, for the forbidden overlay ------------------------------------
forbid = None
if OVERLAY in ("forbidden", "both"):
    if not LABELS:
        raise SystemExit(f"OVERLAY = {OVERLAY!r} needs the phonon parities: set LABELS")
    lab = np.load(LABELS)
    if lab.shape != (nq, nmod):
        raise SystemExit(f"{LABELS} has shape {lab.shape}, the archive needs "
                         f"{(nq, nmod)}: build it with labels_from_elph.py")
    p_nu = np.where(np.isin(lab, [0, 3]), -1.0, 1.0)
    good = (lab >= 0) & ok_fin[:, None]
    forbid = good & (np.abs(p_nu * p_fin[:, None] - p_lam) > 0.5)   # (nq, nmod)

# ---- the smooth field whose zero is the parity contour ------------------------------
even = np.where(chi > 0.5, Ep, np.inf).min(axis=1)
odd = np.where(chi < -0.5, Ep, np.inf).min(axis=1)
delta = odd - even
delta = np.where(np.isfinite(delta), delta, np.where(np.isfinite(even), 0.5, -0.5))
print(f"lowest exciton at Gamma: {'even' if delta[iG] > 0 else 'odd'}   "
      f"(E_odd - E_even = {delta[iG] * 1e3:+.1f} meV)")

# ==========================================================================
# 2. PLOT
# ==========================================================================
nrows = int(np.ceil(nmod / NCOLS))
figsize = tuple(FIGSIZE) if FIGSIZE else (3.6 * NCOLS, 3.2 * nrows)
fig, axs = plt.subplots(nrows, NCOLS, figsize=figsize, squeeze=False)
Xc, Yc, Zc = grid(Qp, delta, limit=LIMIT, method="cubic")

print("\n branch   share of |G|^2 in the forbidden region")
for nu in range(nrows * NCOLS):
    ax = axs[nu // NCOLS, nu % NCOLS]
    if nu >= nmod:
        ax.axis("off")
        continue
    data = SCALE * g2[:, nu, S, :][:, F].sum(axis=1)
    X, Y, Z = grid(qgrid, data, limit=LIMIT)
    vmax = float(np.percentile(data, CLIP)) or 1.0
    im = ax.pcolormesh(X, Y, np.clip(Z, 0, vmax), cmap=CMAP, vmin=0, vmax=vmax,
                       shading="auto", zorder=1, rasterized=True)
    zone(ax)
    if OVERLAY in ("forbidden", "both"):
        _, _, M = grid(qgrid, forbid[:, nu].astype(float), limit=LIMIT, method="nearest")
        with plt.rc_context({"hatch.color": CONTOUR_COLOR, "hatch.linewidth": 0.6}):
            ax.contourf(X, Y, (M > 0.5).astype(float), levels=[0.5, 1.5],
                        colors="none", hatches=["////"], zorder=4)
        share = data[forbid[:, nu]].sum() / max(data.sum(), 1e-300)
        print(f"   {nu + 1:4d}    {share:.2e}")
    if OVERLAY in ("contour", "both"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax.contour(Xc, Yc, Zc, levels=[0.0], colors=CONTOUR_COLOR,
                       linewidths=CONTOUR_LW, zorder=6)
    ax.set_xlim(-LIMIT, LIMIT)
    ax.set_ylim(-LIMIT, LIMIT)
    ax.set_aspect("equal")
    ax.axis("off")
    if LETTERS:
        ax.set_title(f"({chr(ord('a') + nu)})", fontsize=13, pad=4)
    if TAG_BRANCH:
        ax.text(0.03, 0.05, rf"$\nu={nu + 1}$", transform=ax.transAxes, color="white",
                fontsize=10, va="bottom", ha="left", zorder=12)
    cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.05)
    cb = plt.colorbar(im, cax=cax, ticks=[0, vmax / 2, vmax])
    cb.ax.set_yticklabels([f"{t:.1f}" for t in (0, vmax / 2, vmax)], fontsize=9)
if OVERLAY in ("contour", "both"):
    print(f"\ncyan line: where the lowest even and odd exciton bands cross "
          f"(lowest band even where E_odd - E_even > 0)")
if OVERLAY in ("forbidden", "both"):
    print("hatching: symmetry-forbidden for that branch; the zeros should lie under it")

fig.tight_layout()
fig.savefig(OUT, dpi=DPI, bbox_inches="tight")
print(f"saved {OUT}")

# ==========================================================================
# show the figure -- set SHOW = False in SETTINGS to only save it
# ==========================================================================
if SHOW:
    plt.show()
