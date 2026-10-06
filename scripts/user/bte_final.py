#!/usr/bin/env python3
"""
bte_final.py
============
Layout:
  - Left: a column of three 3D Brillouin-zone maps of N_Q(t), with a small
    centred horizontal colorbar beneath them.
  - Right: two rows at specific times (--panel-times).
        1. exciton dispersion, markers coloured by the phonon mode filling
           each state (circles for emission, squares for absorption)
        2. F(E), rotated and sharing the energy axis with the dispersion

N_Q(t) is the exciton occupation summed over bands at momentum Q -- a
dimensionless occupation number, not a density.

Usage:
    python bte_final.py snaps.npz --times 0 500 10000 --panel-times 10 100
"""
from __future__ import annotations
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

plt.rcParams.update({
    'font.size': 13,
    'font.family': 'Montserrat',
    'mathtext.fontset': 'stix',
    'xtick.direction': 'in',
    'ytick.direction': 'in',
})

MODE_COLORS = ['#8f8fd0', '#5bc8e8', '#7fe0a0', '#f0e17a', '#f5a45a', '#c0655a']


def wrap(d):
    d = np.asarray(d, float)
    return d - np.rint(d)


def to_cart(qr, alat, fold=True):
    b = 4 * np.pi / (np.sqrt(3) * alat)
    q = wrap(qr[:, :2]) if fold else qr[:, :2]
    x = (q[:, 0] + 0.5 * q[:, 1]) * b
    y = (np.sqrt(3) / 2) * q[:, 1] * b
    return np.stack([x, y], 1)


def _shortest_G(alat):
    b = 4 * np.pi / (np.sqrt(3) * alat)
    b1 = np.array([b, 0.0])
    b2 = np.array([0.5 * b, np.sqrt(3) / 2 * b])
    return (b1, -b1, b2, -b2, b1 - b2, b2 - b1)


def in_first_bz(X, Y, alat):
    """Wigner-Seitz test: k.G <= |G|^2/2 for the six shortest G."""
    inside = np.ones(X.shape, bool)
    for G in _shortest_G(alat):
        inside &= (X * G[0] + Y * G[1]) <= 0.5 * (G @ G) + 1e-12
    return inside


def bz_edge_mask(X, Y, alat, width=0.015):
    """Cells lying on the zone boundary, where max_G [k.G/(|G|^2/2)] = 1."""
    f = np.full(X.shape, -np.inf)
    for G in _shortest_G(alat):
        f = np.maximum(f, (X * G[0] + Y * G[1]) / (0.5 * (G @ G)))
    return np.abs(f - 1.0) < width


def path_from_disp(disp, nb, alat):
    """Path coordinate and Cartesian points, taken FROM THE FILE.

    Band lines and population markers must lie on the same path. Building an
    analytic path separately and rescaling the file's abscissa only works if
    the two visit the same corners in the same order.
    """
    expect = 1 + nb + 3
    if disp.shape[1] != expect:
        raise SystemExit(
            f"the dispersion file has {disp.shape[1]} columns; with "
            f"nbands={nb} it should have {expect} (q_dist + {nb} bands + "
            f"qx qy qz). Try --nbands {disp.shape[1] - 4}.")
    q_dist = disp[:, 0]
    q_red = disp[:, 1 + nb:1 + nb + 3]
    return q_dist, to_cart(q_red, alat, fold=False), q_red


def ticks_from_path(q_dist, q_red, labels, tol=0.02):
    pts = {"G": (0.0, 0.0), "M": (0.5, 0.0), "K": (1 / 3, 1 / 3),
           "M'": (-0.5, 0.0), "K'": (-1 / 3, -1 / 3),
           "Q": (1 / 6, 1 / 6), "Q'": (-1 / 6, -1 / 6)}
    w = wrap(q_red[:, :2])
    out, keep = [], []
    for l in labels:
        if l not in pts:
            raise SystemExit(f"unknown path label {l!r}")
        d = np.linalg.norm(wrap(w - np.array(pts[l])[None, :]), axis=1)
        k = int(np.argmin(d))
        if d[k] > tol:
            print(f"  [warn] the path never passes {l} "
                  f"(closest approach {d[k]:.3f}); dropping it")
            continue
        out.append(q_dist[k])
        keep.append(l)
    o = np.argsort(out)
    return np.array(out)[o], [keep[k] for k in o]


def project_to_path(pts_cart, path_cart, q_dist):
    d = np.linalg.norm(pts_cart[:, None, :] - path_cart[None, :, :], axis=2)
    near = np.argmin(d, axis=1)
    return q_dist[near], d[np.arange(len(pts_cart)), near]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("snaps")
    p.add_argument("--times", type=float, nargs=3, default=None,
                   help="fs; three times for the 3D column")
    p.add_argument("--panel-times", type=float, nargs=2, default=[10.0, 300.0],
                   help="fs for the two right-hand dispersion panels")
    p.add_argument("--disp", default="o-output.excitons_interpolated")
    p.add_argument("--nbands", type=int, default=5)
    p.add_argument("--alat", type=float, default=3.2)
    p.add_argument("--path", nargs="*", default=["M", "G", "K"],
                   help="labels to tick; must be points the file visits")
    p.add_argument("--sigma", type=float, default=8.0, help="meV, F(E) width")
    p.add_argument("--clip", type=float, default=100,
                   help="percentile of the in-zone surface at which the "
                        "height is clipped. The Gamma peak is orders of "
                        "magnitude above everything else, so normalising to "
                        "it flattens the rest of the zone into the baseline. "
                        "100 disables clipping.")
    p.add_argument("--gamma", type=float, default=0.35,
                   help="height exponent for the 3D surfaces")
    p.add_argument("--tol", type=float, default=0.06,
                   help="1/Ang; max distance from the path to draw a state")
    p.add_argument("--floor", type=float, default=1e-10)
    p.add_argument("--emin", type=float, default=None,
                   help="eV; default is just below the lowest exciton")
    p.add_argument("--emax", type=float, default=5.0)
    p.add_argument("--limit", type=float, default=0.78,
                   help="half-width of the interpolation square, in |b|")
    p.add_argument("--pad", type=float, default=1.12,
                   help="axis limits as a multiple of the data half-width")
    p.add_argument("--zoom", type=float, default=1.35,
                   help="enlarge the 3D surfaces inside their panels")
    p.add_argument("--elev", type=float, default=28.0)
    p.add_argument("--azim", type=float, default=-90.0,
                   help="-90 puts a square edge toward the viewer, so the "
                        "hexagon reads symmetric; -65 shows a corner")
    p.add_argument("--hexcolor", default="white")
    p.add_argument("--hexwidth", type=float, default=0.015,
                   help="boundary width, as a fraction of the Gamma-to-edge "
                        "distance")
    p.add_argument("--hexz", type=float, default=0.02,
                   help="paint the boundary only where the surface is below "
                        "this fraction of the peak height")
    p.add_argument("--edgewidth", type=float, default=0.6,
                   help="black outline on the dispersion markers")
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--out", default="bte_combined.png")
    a = p.parse_args()

    # ---- load ------------------------------------------------------------
    d = np.load(a.snaps)
    t, F, E, Qr = d["t"], d["F"], d["E"], d["Q_red"]
    has_outflux = "dom_mode" in d
    has_influx = "dom_mode_t" in d
    has_chan = has_outflux or has_influx

    want = a.times if a.times else [t[0], t[len(t) // 3], t[-1]]
    idx = [int(np.argmin(np.abs(t - w))) for w in want]
    kp_list = [int(np.argmin(np.abs(t - pt))) for pt in a.panel_times]

    print(f"{a.snaps}: {len(t)} snapshots, {len(E)} states.")
    if has_chan:
        print(f"  channel markers: "
              f"{'outflux (static)' if has_outflux else 'influx (time-dependent)'}")

    # ---- 3D map data -----------------------------------------------------
    xu, inv = np.unique(np.round(wrap(Qr[:, :2]), 5), axis=0,
                        return_inverse=True)
    inv = np.asarray(inv).ravel()
    F_sum = [np.bincount(inv, weights=F[k]) for k in idx]
    fmax = max(f.max() for f in F_sum) or 1e-12
    print(f"  N_Q spans 0 .. {fmax:.3e} (occupation, dimensionless)")

    q_tiled = np.vstack([xu + np.array([dx, dy])
                         for dx in (-1, 0, 1) for dy in (-1, 0, 1)])
    cart_tiled = to_cart(q_tiled, a.alat, fold=False)

    b = 4 * np.pi / (np.sqrt(3) * a.alat)
    limit = b * a.limit
    xi = np.linspace(-limit, limit, 250)
    X, Y = np.meshgrid(xi, xi)
    bz = in_first_bz(X, Y, a.alat)
    edge = bz_edge_mask(X, Y, a.alat, a.hexwidth)

    vc = plt.cm.viridis(np.linspace(0, 1, 256))
    vc[:4, :] = mcolors.to_rgba('gray')
    cmap3d = mcolors.ListedColormap(vc)
    norm3d = mcolors.Normalize(0, fmax ** a.gamma)

    # ---- dispersion + projection ----------------------------------------
    disp = np.loadtxt(a.disp, comments='#')
    nb = a.nbands
    bx, path_cart, path_red = path_from_disp(disp, nb, a.alat)
    E_band = disp[:, 1:1 + nb]
    ticks, tick_labels = ticks_from_path(bx, path_red, a.path)

    st_cart = to_cart(Qr, a.alat, fold=True)
    s_coord, dmin = project_to_path(st_cart, path_cart, bx)
    keep = dmin < a.tol
    print(f"  path {bx[0]:.3f} .. {bx[-1]:.3f}, ticks {np.round(ticks, 3)} "
          f"= {tick_labels}")
    print(f"  {keep.sum()} of {len(E)} states within {a.tol} 1/Ang of it")

    # ---- figure ----------------------------------------------------------
    fig = plt.figure(figsize=(12, 8))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.4], wspace=0.05)
    gl = outer[0].subgridspec(4, 3, height_ratios=[1, 1, 1, 0.04],
                              width_ratios=[0.2, 0.6, 0.2], hspace=0.08)
    gr = outer[1].subgridspec(2, 3, height_ratios=[1, 1],
                              width_ratios=[1, 0.25, 0.04],
                              hspace=0.30, wspace=0.05)

    zmax = fmax ** a.gamma
    for i, (k, fk) in enumerate(zip(idx, F_sum)):
        ax3d = fig.add_subplot(gl[i, :], projection='3d')

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Z = griddata(cart_tiled, np.tile(fk, 9), (X, Y),
                         method='cubic', fill_value=0.0)
            if not np.isfinite(Z).all():
                Z = griddata(cart_tiled, np.tile(fk, 9), (X, Y),
                             method='linear', fill_value=0.0)

        Z = np.clip(Z, 0, None)
        Z = np.clip(Z - (fmax * 0.002), 0, None)
        if a.clip < 100:
            zc = np.percentile(Z[bz], a.clip)
            Z = np.clip(Z, 0, zc)
        R = np.sqrt(X ** 2 + Y ** 2)
        Z = Z * np.clip((limit - R) / (limit - limit * 0.85), 0.0, 1.0)
        # mode='constant' pads with zero; the default 'reflect' folds a
        # replica peak back across the domain edge and lifts the sheet there
        Z = gaussian_filter(Z ** a.gamma, sigma=1.2,
                            mode='constant', cval=0.0)

        # Paint the boundary INTO the surface, and only where the surface is
        # flat. A separate flat polygon at z=0 gets sliced by the K peaks,
        # which sit exactly on the boundary; colouring the full boundary makes
        # the line climb those peaks instead. Restricting to low cells gives
        # what the geometry actually implies -- a line lying on the floor,
        # simply hidden where a peak rises through it.
        cols = cmap3d(norm3d(Z))
        cols[edge & (Z < a.hexz * zmax)] = mcolors.to_rgba(a.hexcolor)

        surf = ax3d.plot_surface(X, Y, Z, facecolors=cols, edgecolor='none',
                                 rstride=1, cstride=1, antialiased=True,
                                 shade=False)
        surf.set_clip_on(False)

        ax3d.set_xlim(-limit * a.pad, limit * a.pad)
        ax3d.set_ylim(-limit * a.pad, limit * a.pad)
        ax3d.set_zlim(0, zmax * 1.1)
        ax3d.view_init(elev=a.elev, azim=a.azim)
        try:
            ax3d.set_box_aspect((1, 1, 0.45), zoom=a.zoom)
        except TypeError:                       # matplotlib < 3.6
            ax3d.set_box_aspect((1, 1, 0.45))
        ax3d.axis('off')
        ax3d.text2D(0.04, 0.84, f"$t = {t[k]:.0f}$ fs",
                    transform=ax3d.transAxes, fontsize=16, fontweight='bold')

    # facecolors leaves the surface without a norm, so the colorbar needs its
    # own mappable
    cax_3d = fig.add_subplot(gl[3, 1])
    sm = plt.cm.ScalarMappable(cmap=cmap3d, norm=norm3d)
    sm.set_array([])
    cb_3d = fig.colorbar(sm, cax=cax_3d, orientation='horizontal')
    cb_3d.set_label(rf'$[N_\mathbf{{Q}}(t)]^{{{a.gamma}}}$' if a.gamma != 1
                    else r'$N_\mathbf{Q}(t)$', fontsize=15, labelpad=8)
    cb_3d.ax.tick_params(labelsize=11, direction='in')

    # ---- right panels ----------------------------------------------------
    Es = np.linspace(E.min() - 0.02, E.max() + 0.02, 600)
    sig = a.sigma * 1e-3
    emin = a.emin if a.emin is not None else E.min() - 0.03

    # one size reference across both panels, so a decayed population reads as
    # smaller markers rather than being renormalised back to full size
    gmax = max(F[k][keep].max() for k in kp_list)

    for row, kp in enumerate(kp_list):
        axb = fig.add_subplot(gr[row, 0])
        axe = fig.add_subplot(gr[row, 1], sharey=axb)

        for n in range(nb):
            axb.plot(bx, E_band[:, n], linestyle='-', lw=2, c='gray',
                     zorder=1, alpha=0.8)
        for x in ticks:
            axb.axvline(x, color="k", lw=0.7, zorder=1)

        if has_outflux:
            DM_kp, IE_kp = d["dom_mode"], d["is_emission"]
        elif has_influx:
            DM_kp = d["dom_mode_t"][kp]
            IE_kp = d["is_emission_t"][kp]

        f = F[kp]
        m = keep & (f > a.floor)
        if m.any():
            lf = np.log10(f[m] / a.floor)
            lref = np.log10(max(gmax, a.floor * 10) / a.floor)
            sizes = 20 + 260 * np.clip(lf / max(lref, 1e-12), 0, 1)
            colors = (np.array([MODE_COLORS[int(v) % 6] for v in DM_kp[m]])
                      if has_chan else np.full(m.sum(), '#2b7bba'))
            emis = IE_kp[m] if has_chan else np.ones(m.sum(), bool)
            o = np.argsort(sizes)[::-1]
            for want_e, mk in ((True, 'o'), (False, 's')):
                sel = emis[o] == want_e
                if sel.any():
                    axb.scatter(s_coord[m][o][sel], E[m][o][sel],
                                s=sizes[o][sel], c=colors[o][sel], marker=mk,
                                alpha=0.9, edgecolors='black',
                                linewidths=a.edgewidth, zorder=3)

        axb.set_xlim(bx[0], bx[-1])
        axb.set_xticks(ticks)
        if row == len(kp_list) - 1:
            axb.set_xticklabels([l.replace("G", r"$\Gamma$")
                                 for l in tick_labels])
        else:
            axb.set_xticklabels([])
        axb.set_ylim(3.2, a.emax)
        axb.set_ylabel("Exciton energy (eV)", fontsize=14)
        axb.set_title(f"Scattering dynamics at $t = {t[kp]:.0f}$ fs",
                      fontsize=16, fontweight='bold', pad=8)
        if row == 0:
            axb.legend(handles=[
                Line2D([], [], marker='o', ls='', c='0.4', mec='k',
                       label='emission'),
                Line2D([], [], marker='s', ls='', c='0.4', mec='k',
                       label='absorption')],
                frameon=False, fontsize=11, loc='lower right')

        nz = f > 0
        fe = np.zeros_like(Es)
        if nz.any():
            fe = (f[nz][None, :] * np.exp(
                -0.5 * ((Es[:, None] - E[nz][None, :]) / sig) ** 2)).sum(1)
        axe.fill_betweenx(Es, 0, fe, color='magenta', alpha=0.15)
        axe.plot(fe, Es, color='magenta', lw=1.8)
        axe.set_xlim(0, max(fe.max() * 1.1, 1e-12))
        axe.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        # fold the 1e-3 into the label; as an offset text it collides with it
        axe.xaxis.offsetText.set_visible(False)
        if row == len(kp_list) - 1:
            ex = int(np.floor(np.log10(max(fe.max(), 1e-30))))
            axe.set_xlabel(rf"$F(E)$ ($\times 10^{{{ex}}}$)", fontsize=13)
        plt.setp(axe.get_yticklabels(), visible=False)

    if has_chan:
        cax2 = fig.add_subplot(gr[:, 2])
        cb2 = matplotlib.colorbar.ColorbarBase(
            cax2, cmap=mcolors.ListedColormap(MODE_COLORS),
            norm=mcolors.BoundaryNorm(np.arange(7) + 0.5, 6),
            ticks=np.arange(1, 7))
        cb2.set_label("dominant phonon mode", fontsize=13, labelpad=10)
        cb2.ax.tick_params(direction='in')

    fig.savefig(a.out, dpi=a.dpi, bbox_inches='tight')
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()