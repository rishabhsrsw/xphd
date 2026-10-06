#!/usr/bin/env python3
"""
plot_parity_bz.py
=================
sigma_h parity across the Brillouin zone, for excitons and for phonons.

    python plot_parity_bz.py                     # both panels
    python plot_parity_bz.py --only exciton      # one panel, column width

Rendering
---------
The parity of a state is +1 or -1; there is no continuum between them. Cubic
interpolation across the mesh produces a smooth gradient at the boundary that
does not exist -- the physical transition is abrupt, and what the gradient
actually shows is the mesh spacing. Nearest-neighbour rendering is used
instead, so each cell carries the value of the q-point it belongs to and the
boundary appears as the step it is. The cost is a visible pixel grid, which
is honest about the resolution rather than concealing it.

Values away from +-1 are kept and shown: they occur where two branches are
nearly degenerate and mix within a mesh spacing, and mark the crossings
rather than indicating an error.

Phonon panel
------------
Parity is taken from the eigenvectors, as the out-of-plane weight
|e_z|^2 / |e|^2, and not from a branch classification: indices are
energy-ordered and exchange character at crossings, so a label-based map
inherits every misassignment. The panel plots the NUMBER of odd branches at
each q, which is invariant under relabelling. The mesh is read from
matdyn.modes, so the refined grid can be used directly and need not match
the exciton mesh.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager
from matplotlib.colors import BoundaryNorm, ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata

_have = {f.name for f in font_manager.fontManager.ttflist}
FONT = "Montserrat" if "Montserrat" in _have else "DejaVu Sans"
plt.rcParams.update({
    "font.size": 13, "font.family": FONT, "mathtext.fontset": "stix",
    "axes.linewidth": 1.8,
})

ROT = np.radians(-30.0)
C, S = np.cos(ROT), np.sin(ROT)


def to_rotated_cart(qf):
    """Fractional -> Cartesian -> rotated so a zone edge faces the viewer."""
    x = qf[:, 0] + 0.5 * qf[:, 1]
    y = (np.sqrt(3.0) / 2.0) * qf[:, 1]
    return x * C - y * S, x * S + y * C


def tile_and_grid(Q_red, data, res=400, limit=0.8, method="nearest"):
    """Replicate over neighbouring cells and sample onto a square grid.

    `method` defaults to nearest: a parity is +1 or -1 and the transition
    between them is a step, so interpolating across it draws a gradient that
    is not in the data.
    """
    w = Q_red[:, :2] - np.rint(Q_red[:, :2])
    qx, qy, dd = [], [], []
    for dx in (-2, -1, 0, 1, 2):
        for dy in (-2, -1, 0, 1, 2):
            qx.append(w[:, 0] + dx)
            qy.append(w[:, 1] + dy)
            dd.append(data)
    qf = np.column_stack([np.concatenate(qx), np.concatenate(qy)])
    kx, ky = to_rotated_cart(qf)
    dd = np.concatenate(dd)
    xi = np.linspace(-limit, limit, res)
    X, Y = np.meshgrid(xi, xi)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Z = griddata((kx, ky), dd, (X, Y), method=method)
        if np.isnan(Z).any():
            Z = np.where(np.isnan(Z),
                         griddata((kx, ky), dd, (X, Y), method="nearest"), Z)
    return X, Y, Z


def draw_zone(ax, lw=2.0, color="black"):
    v = np.array([[1/3, 1/3], [-1/3, 2/3], [-2/3, 1/3],
                  [-1/3, -1/3], [1/3, -2/3], [2/3, -1/3]])
    vx, vy = to_rotated_cart(v)
    ax.add_patch(patches.Polygon(np.column_stack([vx, vy]), closed=True,
                                 fill=False, edgecolor=color, lw=lw, zorder=10))


def panel_exciton(ax, path, band, res, limit, method, cbar=True):
    d = np.load(path)
    Q, chi = d["Q_red"], d["chi"][:, band]
    n_odd = (chi < -0.5).sum()
    print(f"   exciton band {band+1}: {len(chi)} q-points, "
          f"{n_odd} odd ({100*n_odd/len(chi):.0f}%), "
          f"{np.sum(np.abs(np.abs(chi)-1) > 0.2)} away from +-1")
    X, Y, Z = tile_and_grid(Q, chi, res, limit, method)
    im = ax.pcolormesh(X, Y, np.clip(Z, -1, 1), cmap="coolwarm",
                       vmin=-1, vmax=1, shading="auto", zorder=1,
                       rasterized=True)
    draw_zone(ax)
    _square(ax, limit)
    if cbar:
        cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.08)
        cb = plt.colorbar(im, cax=cax, ticks=[-1, 0, 1])
        cb.set_ticklabels([r"$-1$", "0", r"$+1$"])
        cb.ax.tick_params(labelsize=11)
    return im


def panel_phonon(ax, modes_file, res, limit, method, cbar=True):
    """Number of sigma_h-odd branches at each q, from the eigenvectors.

    The parity of a mode is decided by its out-of-plane weight,
    |e_z|^2 / |e|^2, and needs no branch classification at all. That matters
    here: branch indices are energy-ordered and exchange character wherever
    two branches cross, so a map built from labels inherits every
    misassignment. The weight does not depend on labelling or ordering.

    The COUNT is plotted rather than any single branch's character, for the
    same reason: it is invariant under relabelling. In a planar layer
    sigma_h is exact at every in-plane q, so every mode is purely in-plane
    or purely out-of-plane and the count should be the number of flexural
    branches everywhere.

    Reads its own q-mesh from the file, so it does not need to be aligned
    with the exciton mesh and can use the refined 360x360 grid directly.
    """
    from xphd.modes import read_modes
    q, freq, ev = read_modes(modes_file, verbose=False)
    q = np.asarray(q, float)
    ev = np.asarray(ev)
    nq, nmod = ev.shape[0], ev.shape[1]
    ev = ev.reshape(nq, nmod, -1)                       # (nq, nmod, 3*nat)

    zw_num = np.abs(ev[:, :, 2::3]) ** 2                # z components
    zw = zw_num.sum(-1) / np.maximum((np.abs(ev) ** 2).sum(-1), 1e-30)
    n = (zw > 0.5).sum(axis=1).astype(float)

    clean = 100.0 * np.mean((zw < 0.05) | (zw > 0.95))
    vals, cnts = np.unique(n.astype(int), return_counts=True)
    print(f"   phonons: {nq} q-points, {nmod} branches")
    print(f"   odd-branch count: "
          + ", ".join(f"{v}x{100*c/nq:.1f}%" for v, c in zip(vals, cnts)))
    print(f"   out-of-plane weights within 0.05 of 0 or 1: {clean:.2f}%"
          + ("" if clean > 99.0 else
             "\n   [warn] sigma_h is exact in a planar layer, so this should"
             "\n   be 100%. A lower value means the structure is not planar,"
             "\n   or the eigenvector convention is not displacement."))

    X, Y, Z = tile_and_grid(q[:, :2], n, res, limit, method)
    lo, hi = int(np.nanmin(Z)), int(np.nanmax(Z))
    cols = plt.cm.viridis(np.linspace(0.15, 0.9, max(hi - lo + 1, 2)))
    im = ax.pcolormesh(X, Y, Z, cmap=ListedColormap(cols),
                       norm=BoundaryNorm(np.arange(lo - 0.5, hi + 1),
                                         len(cols)),
                       shading="auto", zorder=1, rasterized=True)
    draw_zone(ax, color="white")
    _square(ax, limit)
    if cbar:
        cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.08)
        cb = plt.colorbar(im, cax=cax, ticks=np.arange(lo, hi + 1))
        cb.ax.tick_params(labelsize=11)
    return im


def _square(ax, limit):
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parity", default="parity.npz")
    p.add_argument("--modes", default="../matdyn.modes",
                   help="matdyn.modes; the parity is taken "
                        "from the eigenvectors, so the "
                        "refined mesh can be used directly")
    p.add_argument("--band", type=int, default=0, help="0 = lowest exciton")
    p.add_argument("--only", choices=("exciton", "phonon"), default=None)
    p.add_argument("--res", type=int, default=400)
    p.add_argument("--limit", type=float, default=0.8)
    p.add_argument("--method", default="nearest",
                   choices=("nearest", "linear", "cubic"),
                   help="nearest keeps the parity step sharp; the smooth "
                        "options draw a boundary gradient that is not in "
                        "the data")
    p.add_argument("--dpi", type=int, default=400)
    p.add_argument("--out", default="parity_bz.pdf")
    a = p.parse_args()

    print(f"rendering with method={a.method}")
    if a.only == "exciton":
        fig, ax = plt.subplots(figsize=(3.4, 3.0))
        panel_exciton(ax, a.parity, a.band, a.res, a.limit, a.method)
        ax.set_title(r"exciton $\chi(\sigma_h)$", fontsize=13, pad=6)
    elif a.only == "phonon":
        fig, ax = plt.subplots(figsize=(3.4, 3.0))
        panel_phonon(ax, a.modes, a.res, a.limit, a.method)
        ax.set_title(r"$\sigma_h$-odd branches", fontsize=13, pad=6)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0))
        panel_exciton(axes[0], a.parity, a.band, a.res, a.limit, a.method)
        panel_phonon(axes[1], a.modes, a.res, a.limit, a.method)
        axes[0].set_title(r"exciton $\chi(\sigma_h)$", fontsize=13, pad=6)
        axes[1].set_title(r"$\sigma_h$-odd phonon branches",
                          fontsize=13, pad=6)
        for ax, t in zip(axes, "ab"):
            ax.text(0.02, 0.98, f"({t})", transform=ax.transAxes,
                    ha="left", va="top", fontsize=14, fontweight="bold")
        fig.subplots_adjust(wspace=0.25)

    fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight")
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()