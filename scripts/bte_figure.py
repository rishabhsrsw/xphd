#!/usr/bin/env python3
"""
bte_final.py
============
Layout:
  - Left: a column of three 3D Brillouin-zone maps of N_Q(t), with a
    horizontal colorbar beneath, matched in width to the plotted square.
  - Right: two rows at specific times (--panel-times).
        1. exciton dispersion, markers coloured by the phonon mode filling
           each state (circles for emission, squares for absorption)
        2. flux carried by each phonon mode, rotated so the modes run
           vertically; its bar colours are the marker colours, so it doubles
           as the legend and no separate colorbar is needed
        3. F(E), sharing the energy axis with the dispersion

Run it by editing the SETTINGS block below and running the file -- in VS
Code, Run Python File. No command line is needed. Where the text below
mentions a --flag, the setting is the same name in capitals: --first-band is
FIRST_BAND.

N_Q(t) is the exciton occupation summed over bands at momentum Q -- a
dimensionless occupation number, not a density.

Usage:"""
from __future__ import annotations
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import colormaps as cmaps

# ==========================================================================
# 0. SETTINGS -- edit these, then run this file (VS Code: Run Python File)
# ==========================================================================
SNAPS = 'snaps.npz'

# fs; three times for the 3D column
TIMES = None

# fs for the two right-hand dispersion panels
PANEL_TIMES = [10.0, 300.0]
DISP = "o-output.excitons_interpolated"
NBANDS = 5

# REQUIRED. lattice constant in Angstrom -- no default: GaN 3.23, hBN 2.50;
# a wrong value silently distorts every k-distance
ALAT = None

# labels to tick; must be points the file visits
PATH = ["M", "G", "K"]

# meV, F(E) width
SIGMA = 8.0

# percentile of the in-zone surface at which the height is clipped. The
# Gamma peak is orders of magnitude above everything else, so normalising to
# it flattens the rest of the zone. 100 disables clipping.
CLIP = 100

# height exponent for the 3D surfaces
GAMMA = 0.35

# 1/Ang; max distance from the path to draw a state
TOL = 0.06
FLOOR = 1e-10

# eV; default is just below the lowest exciton
EMIN = None
EMAX = 5.0

# half-width of the interpolation square, in |b|
LIMIT = 0.82

# axis limits as a multiple of the data half-width
PAD = 1.12

# enlarge the 3D surfaces inside their panels
ZOOM = 1.85
ELEV = 35.0

# -90 puts a square edge toward the viewer, so the hexagon reads symmetric;
# -65 shows a corner
AZIM = -90.0

# width of the 3D colorbar as a fraction of the 3D panel, so it can be
# matched to the plotted square. The square's rendered width depends on
# --zoom and --elev, so this is set by eye rather than derived.
CBWIDTH = 1.2
HEXCOLOR = "white"

# boundary width, as a fraction of the Gamma-to-edge distance
HEXWIDTH = 0.015

# paint the boundary only where the surface is below this fraction of the
# peak height
HEXZ = 0.02

# black outline on the dispersion markers
EDGEWIDTH = 0.6
DPI = 600
OUT = "bte_combined.pdf"

# also open the figure in a window after saving it; False only saves it
SHOW = True


TICKSIZE = 10         # one tick-label size for every axis in the figure

plt.rcParams.update({
    'font.size': 13,
    'font.family': 'Montserrat',
    'mathtext.fontset': 'stix',
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.labelsize': TICKSIZE,
    'ytick.labelsize': TICKSIZE,
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
        dd = np.linalg.norm(wrap(w - np.array(pts[l])[None, :]), axis=1)
        k = int(np.argmin(dd))
        if dd[k] > tol:
            print(f"  [warn] the path never passes {l} "
                  f"(closest approach {dd[k]:.3f}); dropping it")
            continue
        out.append(q_dist[k])
        keep.append(l)
    o = np.argsort(out)
    return np.array(out)[o], [keep[k] for k in o]


def project_to_path(pts_cart, path_cart, q_dist):
    dd = np.linalg.norm(pts_cart[:, None, :] - path_cart[None, :, :], axis=2)
    near = np.argmin(dd, axis=1)
    return q_dist[near], dd[np.arange(len(pts_cart)), near]


def main():
    for _name in ('ALAT',):
        if globals()[_name] is None:
            raise SystemExit(f"set {_name} in the SETTINGS block at the top of this file")

    # ---- load ------------------------------------------------------------
    d = np.load(SNAPS)
    t, F, E, Qr = d["t"], d["F"], d["E"], d["Q_red"]
    has_outflux = "dom_mode" in d
    has_influx = "dom_mode_t" in d
    has_chan = has_outflux or has_influx
    has_flux = "flux_t" in d
    FLUX = np.asarray(d["flux_t"], float) if has_flux else None

    want = TIMES if TIMES else [t[0], t[len(t) // 3], t[-1]]
    idx = [int(np.argmin(np.abs(t - w))) for w in want]
    kp_list = [int(np.argmin(np.abs(t - pt))) for pt in PANEL_TIMES]

    print(f"{SNAPS}: {len(t)} snapshots, {len(E)} states.")
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
    cart_tiled = to_cart(q_tiled, ALAT, fold=False)

    b = 4 * np.pi / (np.sqrt(3) * ALAT)
    limit = b * LIMIT
    xi = np.linspace(-limit, limit, 250)
    X, Y = np.meshgrid(xi, xi)
    bz = in_first_bz(X, Y, ALAT)
    edge = bz_edge_mask(X, Y, ALAT, HEXWIDTH)

    vc = cmaps.cet_l_bmy1(np.linspace(0, 1, 256))
    vc[:4, :] = mcolors.to_rgba('slategray')
    cmap3d = mcolors.ListedColormap(vc)
    norm3d = mcolors.Normalize(0, fmax ** GAMMA)

    # ---- dispersion + projection ----------------------------------------
    disp = np.loadtxt(DISP, comments='#')
    nb = NBANDS
    bx, path_cart, path_red = path_from_disp(disp, nb, ALAT)
    E_band = disp[:, 1:1 + nb]
    ticks, tick_labels = ticks_from_path(bx, path_red, PATH)

    st_cart = to_cart(Qr, ALAT, fold=True)
    s_coord, dmin = project_to_path(st_cart, path_cart, bx)
    keep = dmin < TOL
    print(f"  path {bx[0]:.3f} .. {bx[-1]:.3f}, ticks {np.round(ticks, 3)} "
          f"= {tick_labels}")
    print(f"  {keep.sum()} of {len(E)} states within {TOL} 1/Ang of it")

    # ---- figure ----------------------------------------------------------
    fig = plt.figure(figsize=(15, 7))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.4], wspace=-0.095)
    gl = outer[0].subgridspec(4, 1, height_ratios=[1, 1, 1, 0.045],
                              hspace=0.05)
    # per row: dispersion, the rotated flux histogram, then F(E). The
    # histogram's bars carry the marker colours and its axis names the modes,
    # so it serves as the legend and a separate colorbar would be redundant.
    gr = outer[1].subgridspec(2, 3, height_ratios=[1, 1],
                              width_ratios=[1, 0.40, 0.26],
                              hspace=0.30, wspace=0.25)

    zmax = fmax ** GAMMA
    ax3d_last = None
    for i, (k, fk) in enumerate(zip(idx, F_sum)):
        ax3d = fig.add_subplot(gl[i, 0], projection='3d')
        ax3d.set_anchor('N')
        ax3d_last = ax3d

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Z = griddata(cart_tiled, np.tile(fk, 9), (X, Y),
                         method='cubic', fill_value=0.0)
            if not np.isfinite(Z).all():
                Z = griddata(cart_tiled, np.tile(fk, 9), (X, Y),
                             method='linear', fill_value=0.0)

        Z = np.clip(Z, 0, None)
        Z = np.clip(Z - (fmax * 0.002), 0, None)
        if CLIP < 100:
            Z = np.clip(Z, 0, np.percentile(Z[bz], CLIP))
        R = np.sqrt(X ** 2 + Y ** 2)
        Z = Z * np.clip((limit - R) / (limit - limit * 0.85), 0.0, 1.0)
        # mode='constant' pads with zero; the default 'reflect' folds a
        # replica peak back across the domain edge and lifts the sheet there
        Z = gaussian_filter(Z ** GAMMA, sigma=1.2,
                            mode='constant', cval=0.0)

        # Paint the boundary INTO the surface, and only where the surface is
        # flat. A flat polygon at z=0 gets sliced by the K peaks, which sit
        # exactly on the boundary; colouring the full boundary makes the line
        # climb them instead. Restricting to low cells gives what the geometry
        # implies: a line on the floor, hidden where a peak rises through it.
        cols = cmap3d(norm3d(Z))
        cols[edge & (Z < HEXZ * zmax)] = mcolors.to_rgba(HEXCOLOR)

        surf = ax3d.plot_surface(X, Y, Z, facecolors=cols, edgecolor='none',
                                 rstride=1, cstride=1, antialiased=False,
                                 rasterized=True, shade=False)
        surf.set_clip_on(False)

        ax3d.set_xlim(-limit * PAD, limit * PAD)
        ax3d.set_ylim(-limit * PAD, limit * PAD)
        ax3d.set_zlim(-zmax * 0.6, zmax * 1.1)
        ax3d.view_init(elev=ELEV, azim=AZIM)
        try:
            ax3d.set_box_aspect((1, 1, 0.45), zoom=ZOOM)
        except TypeError:                       # matplotlib < 3.6
            ax3d.set_box_aspect((1, 1, 0.45))
        ax3d.axis('off')
        ax3d.text2D(0.04, 0.96, f"$t = {t[k]:.0f}$ fs",
                    transform=ax3d.transAxes, fontsize=15, fontweight='bold')

    # facecolors leaves the surface without a norm, so the colorbar needs its
    # own mappable. Its width is then set from the 3D panel's position rather
    # than from a gridspec column, so it can be matched to the plotted square.
    cax_3d = fig.add_subplot(gl[3, 0])
    sm = plt.cm.ScalarMappable(cmap=cmap3d, norm=norm3d)
    sm.set_array([])
    cb_3d = fig.colorbar(sm, cax=cax_3d, orientation='horizontal')
    cb_3d.set_label(rf'$[N_\mathbf{{Q}}(t)]^{{{GAMMA}}}$' if GAMMA != 1
                    else r'$N_\mathbf{Q}(t)$', fontsize=14, labelpad=6)
    cb_3d.ax.tick_params(labelsize=TICKSIZE, direction='in')

    p3, pc = ax3d_last.get_position(), cax_3d.get_position()
    w = CBWIDTH * p3.width
    cax_3d.set_position([p3.x0 + 0.5 * (p3.width - w), pc.y0, w, pc.height])

    # ---- right panels ----------------------------------------------------
    Es = np.linspace(E.min() - 0.02, E.max() + 0.02, 600)
    sig = SIGMA * 1e-3
    emin = EMIN if EMIN is not None else E.min() - 0.03

    # one size reference across both panels, so a decayed population reads as
    # smaller markers rather than being renormalised back to full size
    gmax = max(F[k][keep].max() for k in kp_list)

    # ONE flux axis for both rows. The flux is sum_ij F_i P^(c)_ij, an
    # ABSOLUTE rate in 1/fs -- it is not normalised by population. The
    # collision operator conserves particle number, so the drop between the
    # two panels is not decay: it is the population settling into low-lying
    # states where fewer channels are energetically open. A shared axis makes
    # that drop readable, which per-panel limits would hide entirely.
    FL0 = FL1 = None
    if has_flux:
        act = np.concatenate([FLUX[k][FLUX[k] > 0].ravel() for k in kp_list])
        FL0 = max(act.min() * 0.4, 1e-15) if act.size else 1e-15
        FL1 = max(max(FLUX[k].max() for k in kp_list) * 3, FL0 * 10)
        for k in kp_list:
            print(f"  flux at t = {t[k]:7.0f} fs: total {FLUX[k].sum():.3e}"
                  f"  max {FLUX[k].max():.3e} /fs   "
                  f"(N = {F[k].sum():.4e}, conserved)")
        print(f"  shared flux axis {FL0:.2e} .. {FL1:.2e} /fs")

    for row, kp in enumerate(kp_list):
        axb = fig.add_subplot(gr[row, 0])
        axe = fig.add_subplot(gr[row, 2], sharey=axb)

        for n in range(nb):
            axb.plot(bx, E_band[:, n], linestyle='-', lw=2, c='gray',
                     zorder=1, alpha=1)
        for x in ticks:
            axb.axvline(x, color="k", linestyle='--', lw=0.7, zorder=1)

        if has_outflux:
            DM_kp, IE_kp = d["dom_mode"], d["is_emission"]
        elif has_influx:
            DM_kp = d["dom_mode_t"][kp]
            IE_kp = d["is_emission_t"][kp]

        f = F[kp]
        m = keep & (f > FLOOR)
        if m.any():
            lf = np.log10(f[m] / FLOOR)
            lref = np.log10(max(gmax, FLOOR * 10) / FLOOR)
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
                                linewidths=EDGEWIDTH, zorder=3)

        axb.set_xlim(bx[0], bx[-1])
        axb.set_xticks(ticks)
        if row == len(kp_list) - 1:
            axb.set_xticklabels([l.replace("G", r"$\Gamma$")
                                 for l in tick_labels])
        else:
            axb.set_xticklabels([])
        axb.set_ylim(3.2, EMAX)
        axb.set_ylabel("Exciton energy (eV)", fontsize=13)
        axb.tick_params(labelsize=TICKSIZE)
        if row == 0:
            axb.legend(handles=[
                Line2D([], [], marker='o', ls='', c='0.4', mec='k',
                       label='emission'),
                Line2D([], [], marker='s', ls='', c='0.4', mec='k',
                       label='absorption')],
                frameon=False, fontsize=10, loc='lower left')

        # ---- flux, rotated so the modes run vertically ------------------
        if has_flux:
            axf = fig.add_subplot(gr[row, 1])
            nmp = FLUX.shape[2]
            ypos = np.arange(1, nmp + 1)
            axf.barh(ypos - 0.2, np.maximum(FLUX[kp, 1], FL0), height=0.38,
                     left=FL0, color=MODE_COLORS[:nmp], edgecolor='k',
                     lw=0.6, label='emission')
            axf.barh(ypos + 0.2, np.maximum(FLUX[kp, 0], FL0), height=0.38,
                     left=FL0, color=MODE_COLORS[:nmp], edgecolor='k',
                     lw=0.6, hatch='///', label='absorption')
            axf.set_xscale('log')
            axf.set_xlim(FL0, FL1)
            # a log axis spanning many decades draws a dense minor-tick comb
            # that reads as noise at this panel size
            axf.xaxis.set_minor_locator(NullLocator())
            axf.set_ylim(0.4, nmp + 0.6)
            axf.set_yticks(ypos)
            axf.set_yticklabels([str(v) for v in ypos])
            axf.tick_params(labelsize=TICKSIZE)
            axf.set_ylabel("phonon mode", fontsize=13)
            if row == len(kp_list) - 1:
                axf.set_xlabel("flux (1/fs)", fontsize=13)
            else:
                axf.set_xticklabels([])
            if row == 1:
                leg = axf.legend(frameon=False, fontsize=8, loc='lower right')
                for txt in leg.get_texts():
                    txt.set_va('center')

        # ---- F(E) --------------------------------------------------------
        nz = f > 0
        fe = np.zeros_like(Es)
        if nz.any():
            fe = (f[nz][None, :] * np.exp(
                -0.5 * ((Es[:, None] - E[nz][None, :]) / sig) ** 2)).sum(1)
        axe.fill_betweenx(Es, 0, fe, color='magenta', alpha=0.15)
        axe.plot(fe, Es, color='magenta', lw=1.8)
        axe.set_xlim(0, max(fe.max() * 1.1, 1e-12))
        axe.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        # fold the exponent into the label; as offset text it collides with it
        axe.xaxis.offsetText.set_visible(False)
        axe.tick_params(labelsize=TICKSIZE)
        if row == len(kp_list) - 1:
            ex = int(np.floor(np.log10(max(fe.max(), 1e-30))))
            axe.set_xlabel(rf"F(E) ($\times 10^{{{ex}}}$)", fontsize=13)
        plt.setp(axe.get_yticklabels(), visible=False)

    fig.savefig(OUT, dpi=DPI, bbox_inches='tight')
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
    if SHOW:
        plt.show()
