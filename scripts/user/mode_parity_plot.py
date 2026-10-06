#!/usr/bin/env python3
"""
plot_coupling_parity.py
=======================
Exciton-phonon coupling at Q = 0, split by the mirror parity of the phonon.

    python plot_coupling_parity.py
    python plot_coupling_parity.py --parity parity.npz    # overlay the
                                                          # exciton contour

The two panels should be close to complementary. A sigma_h-even exciton can
be connected to another even state only by an even phonon, and to an odd
state only by an odd one, so the even coupling should occupy the region where
the final exciton is even and the odd coupling its complement. Overlaying the
zero contour of the exciton parity tests that directly: if the boundary
between the two maps coincides with it, the selection rule is visible without
any group theory in the reading.

The colour scale is shared between panels, since the point is a comparison
and per-panel normalisation would destroy it. Both are given as a fraction of
the largest value in either, so the numbers are relative and the physical
units of |G|^2 do not need stating.
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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata

_have = {f.name for f in font_manager.fontManager.ttflist}
FONT = "Montserrat" if "Montserrat" in _have else "DejaVu Sans"
plt.rcParams.update({
    "font.size": 13, "font.family": FONT, "mathtext.fontset": "stix",
    "axes.linewidth": 1.8,
})

ROT = np.radians(-30.0)
CR, SR = np.cos(ROT), np.sin(ROT)


def to_rot(qf):
    x = qf[:, 0] + 0.5 * qf[:, 1]
    y = (np.sqrt(3.0) / 2.0) * qf[:, 1]
    return x * CR - y * SR, x * SR + y * CR


def grid(Q, data, res, limit, method):
    w = Q[:, :2] - np.rint(Q[:, :2])
    qx, qy, dd = [], [], []
    for dx in (-2, -1, 0, 1, 2):
        for dy in (-2, -1, 0, 1, 2):
            qx.append(w[:, 0] + dx)
            qy.append(w[:, 1] + dy)
            dd.append(data)
    kx, ky = to_rot(np.column_stack([np.concatenate(qx),
                                     np.concatenate(qy)]))
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


def zone(ax, color="white", lw=1.6):
    v = np.array([[1/3, 1/3], [-1/3, 2/3], [-2/3, 1/3],
                  [-1/3, -1/3], [1/3, -2/3], [2/3, -1/3]])
    vx, vy = to_rot(v)
    ax.add_patch(patches.Polygon(np.column_stack([vx, vy]), closed=True,
                                 fill=False, edgecolor=color, lw=lw,
                                 zorder=10))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default="coupling_by_parity.npz")
    p.add_argument("--parity", default=None,
                   help="parity.npz; its zero contour is overlaid on both "
                        "panels as the boundary the coupling should respect")
    p.add_argument("--res", type=int, default=400)
    p.add_argument("--limit", type=float, default=0.8)
    p.add_argument("--method", default="cubic",
                   choices=("nearest", "linear", "cubic"),
                   help="the coupling is a smooth function of q, unlike a "
                        "parity, so interpolation is appropriate here")
    p.add_argument("--cmap", default="afmhot")
    p.add_argument("--clip", type=float, default=100.0,
                   help="percentile at which the shared scale is capped")
    p.add_argument("--dpi", type=int, default=400)
    p.add_argument("--out", default="coupling_parity.pdf")
    a = p.parse_args()

    d = np.load(a.npz)
    Q, Ge, Go = d["Q_red"], d["G_even"], d["G_odd"]
    Ge, Go = np.asarray(Ge).ravel(), np.asarray(Go).ravel()
    if Ge.size != len(Q):                      # stored as a 2D grid
        Ge = d["G_even"][d["i"], d["j"]] if "i" in d.files else Ge
        Go = d["G_odd"][d["i"], d["j"]] if "i" in d.files else Go
    tot = Ge.sum() + Go.sum()
    print(f"{a.npz}: {len(Q)} q-points")
    print(f"   even-mode coupling: {100*Ge.sum()/tot:5.1f}% of the total")
    print(f"   odd-mode  coupling: {100*Go.sum()/tot:5.1f}%")

    vmax = np.percentile(np.concatenate([Ge, Go]), a.clip)
    print(f"   shared scale 0 .. {vmax:.4g}")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    for ax, dat, ttl in ((axes[0], Ge, r"even ($\sigma_h$-symmetric) modes"),
                         (axes[1], Go, r"odd ($\sigma_h$-antisymmetric) modes")):
        X, Y, Z = grid(Q, dat, a.res, a.limit, a.method)
        im = ax.pcolormesh(X, Y, np.clip(Z, 0, vmax), cmap=a.cmap,
                           vmin=0, vmax=vmax, shading="auto", zorder=1,
                           rasterized=True)
        zone(ax)
        ax.set_xlim(-a.limit, a.limit)
        ax.set_ylim(-a.limit, a.limit)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(ttl, fontsize=12, pad=6)

    # ---- the exciton parity boundary ---------------------------------
    if a.parity:
        dp = np.load(a.parity)
        chi = dp["chi"][:, 0]
        Xp, Yp, Zp = grid(dp["Q_red"], chi, a.res, a.limit, "cubic")
        for ax in axes:
            ax.contour(Xp, Yp, Zp, levels=[0.0], colors="deepskyblue",
                       linewidths=1.8, zorder=11)
        print(f"   overlaid the chi = 0 contour of the exciton parity;")
        print(f"   the two couplings should divide along it")

    cax = make_axes_locatable(axes[1]).append_axes("right", size="5%",
                                                  pad=0.08)
    cb = plt.colorbar(im, cax=cax, ticks=[0, vmax / 2, vmax])
    cb.set_ticklabels(["0", "", "max"])
    cb.ax.tick_params(labelsize=11)
    for ax, t in zip(axes, "ab"):
        ax.text(0.02, 0.98, f"({t})", transform=ax.transAxes, ha="left",
                va="top", fontsize=14, fontweight="bold", color="white")
    fig.subplots_adjust(wspace=0.08)
    fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight")
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()