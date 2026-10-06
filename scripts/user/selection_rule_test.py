#!/usr/bin/env python3
"""
selection_rule_test.py
======================
The exciton-phonon selection rule, tested directly.

For a sigma_h-even initial exciton, an EVEN phonon connects it to an even
final state and an ODD phonon to an odd one. The complementary combinations
are forbidden. Mapping the coupling split only by phonon parity cannot show
this, because an even phonon is allowed for even->even AND for odd->odd, so
it appears everywhere; the rule lives in the correlation between the phonon
parity and the FINAL STATE parity, not in either alone.

This script forms

    G_allowed   = sum over (mode, m) with  p_m(q) * p_nu == p_lambda
    G_forbidden = sum over (mode, m) with  p_m(q) * p_nu != p_lambda

and reports their ratio. A working selection rule makes the second small.

It also resolves the coupling by final state, which answers a different
question: given an exciton at Q = 0, which state does it scatter into, and
where in the zone does that channel live.

    python selection_rule_test.py --state 1
    python selection_rule_test.py --state 1 --by-final
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

import xphd

_have = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams.update({
    "font.size": 12,
    "font.family": "Montserrat" if "Montserrat" in _have else "DejaVu Sans",
    "mathtext.fontset": "stix", "axes.linewidth": 1.6,
})

ROT = np.radians(-30.0)
CR, SR = np.cos(ROT), np.sin(ROT)


def to_rot(qf):
    x = qf[:, 0] + 0.5 * qf[:, 1]
    y = (np.sqrt(3.0) / 2.0) * qf[:, 1]
    return x * CR - y * SR, x * SR + y * CR


def grid(Q, data, res=350, limit=0.8, method="cubic"):
    w = Q[:, :2] - np.rint(Q[:, :2])
    qx, qy, dd = [], [], []
    for dx in (-2, -1, 0, 1, 2):
        for dy in (-2, -1, 0, 1, 2):
            qx.append(w[:, 0] + dx); qy.append(w[:, 1] + dy); dd.append(data)
    kx, ky = to_rot(np.column_stack([np.concatenate(qx), np.concatenate(qy)]))
    dd = np.concatenate(dd)
    xi = np.linspace(-limit, limit, res)
    X, Y = np.meshgrid(xi, xi)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Z = griddata((kx, ky), dd, (X, Y), method=method)
        if np.isnan(Z).any():
            Z = np.where(np.isnan(Z),
                         griddata((kx, ky), dd, (X, Y), "nearest"), Z)
    return X, Y, Z


def zone(ax, color="white", lw=1.5):
    v = np.array([[1/3, 1/3], [-1/3, 2/3], [-2/3, 1/3],
                  [-1/3, -1/3], [1/3, -2/3], [2/3, -1/3]])
    vx, vy = to_rot(v)
    ax.add_patch(patches.Polygon(np.column_stack([vx, vy]), closed=True,
                                 fill=False, edgecolor=color, lw=lw,
                                 zorder=10))


def panel(ax, Q, dat, vmax, title, cmap="afmhot", limit=0.8):
    X, Y, Z = grid(Q, dat, limit=limit)
    im = ax.pcolormesh(X, Y, np.clip(Z, 0, vmax), cmap=cmap, vmin=0,
                       vmax=vmax, shading="auto", zorder=1, rasterized=True)
    zone(ax)
    ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=11, pad=5)
    return im


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", default="GI_ExcPh_Q0001.npz")
    p.add_argument("--parity", default="parity.npz")
    p.add_argument("--labels", default="mode_labels.npy")
    p.add_argument("--state", type=int, default=1, help="initial exciton")
    p.add_argument("--tol", type=float, default=0.5,
                   help="|chi| below this is treated as undefined and the "
                        "point is excluded rather than forced to a parity")
    p.add_argument("--by-final", action="store_true",
                   help="also map the coupling resolved by final state")
    p.add_argument("--nfinal", type=int, default=4)
    p.add_argument("--modes-for", type=int, default=None,
                   help="final state index; adds one panel per "
                        "phonon branch for that channel")
    p.add_argument("--clip", type=float, default=98.0)
    p.add_argument("--dpi", type=int, default=400)
    p.add_argument("--out", default="selection_rule.pdf")
    a = p.parse_args()

    arc = xphd.ExcPhArchive(a.archive)
    g2 = arc.grid("g2")                       # (n1, n2, nmod, nexc, nexc)
    print(f"{a.archive}: g2 {g2.shape}")
    n1, n2, nmod, nexc, _ = g2.shape
    S = a.state - 1

    # ---- parity of the final exciton at each q -----------------------
    d = np.load(a.parity)
    Qp, chi = d["Q_red"], d["chi"]
    # An exact lookup fails when a coordinate sits within rounding of 1.0,
    # which maps to 0.0 on one side of the cell and 1.0 on the other. Match
    # by periodic distance instead, which has no such edge.
    qgrid = np.mod(arc.q_red, 1.0)
    dd = qgrid[:, None, :2] - np.mod(Qp[:, :2], 1.0)[None, :, :]
    dd -= np.rint(dd)
    dist = np.linalg.norm(dd, axis=-1)
    idx = np.argmin(dist, axis=1)
    worst = float(dist[np.arange(len(qgrid)), idx].max())
    if worst > 1e-5:
        raise SystemExit(f"meshes differ: worst match {worst:.2e}")
    print(f"   matched all {len(qgrid)} q-points to {a.parity} "
          f"(worst {worst:.1e})")

    p_beta = np.sign(chi[idx][:, :nexc])              # (nq, nexc)
    ok_beta = np.abs(chi[idx][:, :nexc]) > a.tol
    print(f"   final-state parity defined at "
          f"{100*ok_beta.mean():.1f}% of (q, state) pairs")

    # ---- parity of the phonon ----------------------------------------
    lab = np.load(a.labels)
    p_nu = np.where(np.isin(lab, [0, 3]), -1.0, 1.0)   # (nq, nmod)
    p_nu[lab < 0] = 0.0
    ok_nu = lab >= 0
    print(f"   phonon parity defined at {100*ok_nu.mean():.1f}% of "
          f"(q, mode) pairs")

    # ---- the initial state's parity ----------------------------------
    iG = int(np.argmin(np.linalg.norm(qgrid[:, :2], axis=1)))
    p_lam = float(np.sign(chi[idx[iG], S]))
    print(f"   initial state {a.state} at Gamma: chi = "
          f"{chi[idx[iG], S]:+.3f}  -> p = {p_lam:+.0f}")

    # ---- allowed and forbidden ---------------------------------------
    # flat (nq, nmod, nexc) arrays over the archive's q ordering
    g2f = np.zeros((len(qgrid), nmod, nexc))
    g2f[:] = g2[arc._i, arc._j][:, :, S, :]
    prod = p_nu[:, :, None] * p_beta[:, None, :]       # (nq, nmod, nexc)
    good = ok_nu[:, :, None] & ok_beta[:, None, :]
    allowed = good & (np.abs(prod - p_lam) < 0.5)
    forbid = good & (np.abs(prod + p_lam) < 0.5)

    G_all = np.where(allowed, g2f, 0.0).sum(axis=(1, 2))
    G_for = np.where(forbid, g2f, 0.0).sum(axis=(1, 2))
    G_und = np.where(~good, g2f, 0.0).sum(axis=(1, 2))
    tot = G_all.sum() + G_for.sum() + G_und.sum()
    print(f"\n   SELECTION RULE, initial state {a.state}")
    print(f"     allowed    {G_all.sum():12.4e}  {100*G_all.sum()/tot:6.2f}%")
    print(f"     forbidden  {G_for.sum():12.4e}  {100*G_for.sum()/tot:6.2f}%")
    print(f"     undefined  {G_und.sum():12.4e}  {100*G_und.sum()/tot:6.2f}%"
          f"   (parity not assignable)")
    r = G_for.sum() / max(G_all.sum(), 1e-30)
    print(f"     forbidden / allowed = {r:.4f}")
    print("     a working rule makes this small; a value near 1 means the"
          "\n     rule is not operating, or a parity assignment is wrong")

    vmax = np.percentile(np.concatenate([G_all, G_for]), a.clip)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.1))
    panel(axes[0], arc.q_red, G_all, vmax, "symmetry-allowed")
    im = panel(axes[1], arc.q_red, G_for, vmax, "symmetry-forbidden")
    cax = make_axes_locatable(axes[1]).append_axes("right", size="5%",
                                                  pad=0.08)
    cb = plt.colorbar(im, cax=cax, ticks=[0, vmax])
    cb.set_ticklabels(["0", "max"])
    for ax, t in zip(axes, "ab"):
        ax.text(0.02, 0.98, f"({t})", transform=ax.transAxes, ha="left",
                va="top", fontsize=13, fontweight="bold", color="white")
    fig.subplots_adjust(wspace=0.06)
    fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight")
    print(f"\nsaved {a.out}")

    # ---- coupling resolved by final state ----------------------------
    if not a.by_final:
        return
    print(f"\n   COUPLING BY FINAL STATE (where does state {a.state} go?)")
    LAB = ["ZA", "TA", "LA", "ZO", "TO", "LO"]
    nf = min(a.nfinal, nexc)
    per = [g2f[:, :, m].sum(axis=1) for m in range(nf)]
    for m in range(nf):
        w = per[m]
        j = int(np.argmax(w))                      # dominant q for this channel
        bym = g2f[:, :, m].sum(axis=0)             # per mode, summed over q
        order = np.argsort(bym)[::-1]
        kx, ky = to_rot(qgrid[j:j + 1, :2])
        par = ("odd" if p_beta[j, m] < 0 else "even") if ok_beta[j, m] \
            else "undefined"
        print(f"     -> state {m+1}: {100*w.sum()/g2f.sum():5.1f}% of the "
              f"coupling")
        print(f"        peaks at q = {np.round(qgrid[j, :2], 4)}, "
              f"|q| = {float(np.hypot(kx, ky)):.3f}; the final state is "
              f"{par} there")
        # the branch CHARACTER is local: no branch holds one character across
        # the zone, so it is quoted at the dominant q rather than globally
        parts = []
        for nu in order[:3]:
            ch = LAB[lab[j, nu]] if lab[j, nu] >= 0 else "--"
            pp = "odd" if p_nu[j, nu] < 0 else "even"
            parts.append(f"nu{nu+1} [{ch}, {pp} at this q] "
                         f"{100*bym[nu]/max(bym.sum(), 1e-30):.0f}%")
        print(f"        modes: " + ";  ".join(parts))

    vmax2 = np.percentile(np.concatenate(per), a.clip)
    fig2, ax2 = plt.subplots(1, nf, figsize=(3.2 * nf, 3.1))
    ax2 = np.atleast_1d(ax2)
    for m in range(nf):
        panel(ax2[m], arc.q_red, per[m], vmax2,
              rf"$\lambda={a.state} \to \beta={m+1}$")
    fig2.subplots_adjust(wspace=0.06)
    out2 = a.out.replace(".pdf", "_by_final.pdf")
    fig2.savefig(out2, dpi=a.dpi, bbox_inches="tight")
    print(f"saved {out2}")
    print("   these are intervalley maps: the value at q is the coupling for")
    print("   the exciton at Gamma to scatter INTO that state at momentum q.")
    print("   They show where coupling EXISTS, not where scattering happens:")
    print("   energy conservation still has to be satisfied, and it is what")
    print("   selects the active channels.")

    # ---- one panel per phonon branch, for a chosen final state -------
    if a.modes_for:
        m = a.modes_for - 1
        per_mode = [g2f[:, nu, m] for nu in range(nmod)]
        share = np.array([x.sum() for x in per_mode])
        print(f"\n   BY PHONON BRANCH, for {a.state} -> {a.modes_for}")
        for nu in range(nmod):
            frac_odd = (np.where(p_nu[:, nu] < 0, per_mode[nu], 0).sum()
                        / max(share[nu], 1e-30))
            print(f"     nu{nu+1}: {100*share[nu]/max(share.sum(),1e-30):5.1f}%"
                  f" of this channel, {100*frac_odd:5.1f}% of it where the "
                  f"branch is sigma_h-odd")
        vm = np.percentile(np.concatenate(per_mode), a.clip)
        fig3, ax3 = plt.subplots(1, nmod, figsize=(2.6 * nmod, 2.9))
        ax3 = np.atleast_1d(ax3)
        for nu in range(nmod):
            panel(ax3[nu], arc.q_red, per_mode[nu], vm, rf"$\nu={nu+1}$")
        fig3.subplots_adjust(wspace=0.05)
        out3 = a.out.replace(".pdf", f"_modes_b{a.modes_for}.pdf")
        fig3.savefig(out3, dpi=a.dpi, bbox_inches="tight")
        print(f"saved {out3}")


if __name__ == "__main__":
    main()