#!/usr/bin/env python3
"""
bte_animate.py
==============
Animate exciton populations, from the snapshots written by xphd bte.

Layout matches bte_combined:
  - Left  : animated 3D Brillouin-zone map of N_Q(t), colorbar beneath
  - Right : exciton dispersion (markers coloured by phonon mode) with F(E)
            beside it, and the per-mode flux histogram below

The 3D surface cannot be updated in place the way a scatter can -- matplotlib
rebuilds a Poly3DCollection from scratch -- so each frame removes the old
surface and draws a new one. The expensive part is the interpolation, not the
drawing, so the Delaunay triangulation of the tiled q-points is built ONCE and
reused for every frame. That turns a per-frame triangulation into a per-frame
solve and is worth roughly an order of magnitude.

Usage:
    python bte_animate.py snaps.npz --disp o-output.excitons_interpolated_01 \
        --t-end 2000 --stride 5 --out cascade.mp4
"""
from __future__ import annotations
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.collections import PolyCollection
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import gaussian_filter

# ==========================================================================
# 0. SETTINGS -- edit these, then run this file (VS Code: Run Python File)
# ==========================================================================
SNAPS = 'snaps.npz'
DISP = "o-output.excitons_interpolated"
NBANDS = 5

# REQUIRED. lattice constant in Angstrom -- no default: GaN 3.23, hBN 2.50;
# a wrong value silently distorts every k-distance
ALAT = None
PATH = ["M", "G", "K"]
TOL = 0.06
FLOOR = 1e-10
SMAX = 260.0

# meV, F(E) width
SIGMA = 8.0

# One of ("global", "frame").
SCALE = "global"
EMIN = None
EMAX = 5.0

# skip the 3D column; much faster while iterating
NO3D = False

# 3D interpolation grid. 250 for the final render; 160 is four times cheaper
# and looks the same in motion.
RES = 160

# rstride/cstride on the 3D surface. Cost scales as 1/stride^2 and the
# surface is already smoothed, so 2 looks identical to 1 and is four times
# cheaper.
STRIDE3D = 2
GAMMA = 0.35
LIMIT = 0.82
PAD = 1.12
ZOOM = 1.6
ELEV = 35.0
AZIM = -90.0
HEXCOLOR = "white"
HEXWIDTH = 0.015
HEXZ = 0.02

# name from the colormaps package; falls back to viridis if unavailable
CMAP3D = None
T_START = None

# fs. Applied BEFORE --stride, so the stride thins the window you asked for,
# not the whole run.
T_END = None
STRIDE = 1
FPS = 20
DPI = 140
BITRATE = 4000

# explicit path to ffmpeg.exe; matplotlib caches the location at import and
# does not rescan PATH
FFMPEG = None
OUT = "bte_animation.mp4"


TICKSIZE = 10

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
    inside = np.ones(X.shape, bool)
    for G in _shortest_G(alat):
        inside &= (X * G[0] + Y * G[1]) <= 0.5 * (G @ G) + 1e-12
    return inside


def bz_edge_mask(X, Y, alat, width=0.015):
    f = np.full(X.shape, -np.inf)
    for G in _shortest_G(alat):
        f = np.maximum(f, (X * G[0] + Y * G[1]) / (0.5 * (G @ G)))
    return np.abs(f - 1.0) < width


def path_from_disp(disp, nb, alat):
    expect = 1 + nb + 3
    if disp.shape[1] != expect:
        raise SystemExit(
            f"the dispersion file has {disp.shape[1]} columns; with "
            f"nbands={nb} it should have {expect}. "
            f"Try --nbands {disp.shape[1] - 4}.")
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
    global OUT
    # ---- 3D panel ----
    # ---- timeline ----
    for _name in ('ALAT',):
        if globals()[_name] is None:
            raise SystemExit(f"set {_name} in the SETTINGS block at the top of this file")

    if FFMPEG:
        plt.rcParams['animation.ffmpeg_path'] = FFMPEG

    # ---- load and window -------------------------------------------------
    d = np.load(SNAPS)
    t_all, F_all, E, Qr = d["t"], d["F"], d["E"], d["Q_red"]

    sel = np.ones(len(t_all), bool)
    if T_START is not None:
        sel &= t_all >= T_START
    if T_END is not None:
        sel &= t_all <= T_END
    if not sel.any():
        raise SystemExit(f"no snapshots in [{T_START}, {T_END}] fs; the "
                         f"file spans {t_all[0]:.0f} .. {t_all[-1]:.0f}")
    idx = np.where(sel)[0][::STRIDE]
    t, F = t_all[idx], F_all[idx]
    print(f"{SNAPS}: window {t[0]:.0f} .. {t[-1]:.0f} fs, "
          f"{len(idx)} of {len(t_all)} frames")

    has_influx = "dom_mode_t" in d
    has_outflux = "dom_mode" in d
    has_chan = has_influx or has_outflux
    has_flux = "flux_t" in d

    if has_influx:
        DM = np.asarray(d["dom_mode_t"], int)[idx]
        IE = np.asarray(d["is_emission_t"], bool)[idx]
        nm_ph = int(DM.max()) + 1
    elif has_outflux:
        DM = np.tile(np.asarray(d["dom_mode"], int), (len(t), 1))
        IE = np.tile(np.asarray(d["is_emission"], bool), (len(t), 1))
        nm_ph = int(DM.max()) + 1
    else:
        DM = np.zeros((len(t), len(E)), int)
        IE = np.ones((len(t), len(E)), bool)
        nm_ph = 1
    FLUX = np.asarray(d["flux_t"], float)[idx] if has_flux else None

    # ---- dispersion path -------------------------------------------------
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

    # ---- 3D surfaces, precomputed ---------------------------------------
    Zs = None
    if not NO3D:
        xu, inv = np.unique(np.round(wrap(Qr[:, :2]), 5), axis=0,
                            return_inverse=True)
        inv = np.asarray(inv).ravel()
        q_tiled = np.vstack([xu + np.array([dx, dy])
                             for dx in (-1, 0, 1) for dy in (-1, 0, 1)])
        cart_tiled = to_cart(q_tiled, ALAT, fold=False)

        b = 4 * np.pi / (np.sqrt(3) * ALAT)
        limit = b * LIMIT
        xi = np.linspace(-limit, limit, RES)
        X, Y = np.meshgrid(xi, xi)
        bz = in_first_bz(X, Y, ALAT)
        edge = bz_edge_mask(X, Y, ALAT, HEXWIDTH)
        R = np.sqrt(X ** 2 + Y ** 2)
        taper = np.clip((limit - R) / (limit - limit * 0.85), 0.0, 1.0)

        # ONE triangulation for every frame. LinearNDInterpolator accepts a
        # prebuilt Delaunay, so only the linear solve is repeated.
        print(f"  triangulating {len(cart_tiled)} tiled q-points once...")
        tri = Delaunay(cart_tiled)

        F_sum = np.stack([np.bincount(inv, weights=F[k]) for k in range(len(t))])
        fmax = float(F_sum.max()) or 1e-12
        print(f"  N_Q spans 0 .. {fmax:.3e}; interpolating {len(t)} frames "
              f"on {RES}x{RES}")

        Zs = np.empty((len(t), RES, RES), np.float32)
        for k in range(len(t)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                Z = LinearNDInterpolator(tri, np.tile(F_sum[k], 9),
                                         fill_value=0.0)(X, Y)
            Z = np.clip(Z, 0, None)
            Z = np.clip(Z - fmax * 0.002, 0, None) * taper
            Zs[k] = gaussian_filter(Z ** GAMMA, sigma=1.2,
                                    mode='constant', cval=0.0)
            if (k + 1) % max(len(t) // 10, 1) == 0:
                print(f"    {k + 1}/{len(t)}")
        zmax = fmax ** GAMMA

        try:
            import colormaps as cmaps
            base = (getattr(cmaps, CMAP3D) if CMAP3D
                    else cmaps.cet_l_bmy1)(np.linspace(0, 1, 256))
        except Exception:
            base = plt.cm.viridis(np.linspace(0, 1, 256))
        base[:4, :] = mcolors.to_rgba('slategray')
        cmap3d = mcolors.ListedColormap(base)
        norm3d = mcolors.Normalize(0, zmax)

    # ---- figure ----------------------------------------------------------
    if NO3D:
        fig = plt.figure(figsize=(9, 8), dpi=DPI)
        outer = fig.add_gridspec(1, 1)
        gr = outer[0].subgridspec(2, 3, height_ratios=[1.6, 1.0],
                                  width_ratios=[1, 0.40, 0.26],
                                  hspace=0.30, wspace=0.25)
    else:
        fig = plt.figure(figsize=(14, 7), dpi=DPI)
        # both columns span the same vertical extent, with a clear gap
        # between them; the colorbar sits immediately under the surface
        outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.45],
                                 wspace=0.16)
        gl = outer[0].subgridspec(2, 1, height_ratios=[1, 0.035],
                                  hspace=0.0)
        gr = outer[1].subgridspec(2, 2, height_ratios=[1.6, 1.0],
                                  width_ratios=[1.0, 0.24],
                                  hspace=0.30, wspace=0.04)
        ax3d = fig.add_subplot(gl[0, 0], projection='3d')
        ax3d.set_xlim(-limit * PAD, limit * PAD)
        ax3d.set_ylim(-limit * PAD, limit * PAD)
        ax3d.set_zlim(-zmax * 0.6, zmax * 1.1)
        ax3d.view_init(elev=ELEV, azim=AZIM)
        try:
            ax3d.set_box_aspect((1, 1, 0.45), zoom=ZOOM)
        except TypeError:
            ax3d.set_box_aspect((1, 1, 0.45))
        ax3d.axis('off')

        ax3d.set_anchor('S')          # sit right on the colorbar
        cax3 = fig.add_subplot(gl[1, 0])
        sm = plt.cm.ScalarMappable(cmap=cmap3d, norm=norm3d)
        sm.set_array([])
        cb3 = fig.colorbar(sm, cax=cax3, orientation='horizontal')
        cb3.set_label(rf'$[N_\mathbf{{Q}}(t)]^{{{GAMMA}}}$' if GAMMA != 1
                      else r'$N_\mathbf{Q}(t)$', fontsize=13, labelpad=5)
        cb3.ax.tick_params(labelsize=TICKSIZE, direction='in')

    # ---- dispersion ------------------------------------------------------
    axb = fig.add_subplot(gr[0, 0])
    for n in range(nb):
        axb.plot(bx, E_band[:, n], '-', lw=2, c='gray', zorder=1)
    for x in ticks:
        axb.axvline(x, color="k", ls='--', lw=0.7, zorder=1)
    axb.set_xlim(bx[0], bx[-1])
    axb.set_xticks(ticks)
    axb.set_xticklabels([l.replace("G", r"$\Gamma$") for l in tick_labels])
    axb.set_ylim(EMIN if EMIN is not None else E[keep].min() - 0.03,
                 EMAX)
    axb.set_ylabel("Exciton energy (eV)", fontsize=13)
    axb.tick_params(labelsize=TICKSIZE)
    title = axb.set_title("", fontsize=15, fontweight='bold', pad=10)
    axb.legend(handles=[
        Line2D([], [], marker='o', ls='', c='0.4', mec='k', label='emission'),
        Line2D([], [], marker='s', ls='', c='0.4', mec='k',
               label='absorption')],
        frameon=False, fontsize=10, loc='lower left')

    cmap = mcolors.ListedColormap(MODE_COLORS)
    norm = mcolors.BoundaryNorm(np.arange(7) + 0.5, 6)
    scat_em = axb.scatter([], [], s=[], c=[], marker="o", cmap=cmap, norm=norm,
                          alpha=0.9, edgecolors='black', linewidths=0.6,
                          zorder=3)
    scat_ab = axb.scatter([], [], s=[], c=[], marker="s", cmap=cmap, norm=norm,
                          alpha=0.9, edgecolors='black', linewidths=0.6,
                          zorder=3)

    # ---- flux ------------------------------------------------------------
    if has_flux:
        # spans both right-hand columns. The panel is wide and short,
        # so the bars run vertically with the modes on x -- the
        # opposite of the rotated form used in a narrow column.
        axf = fig.add_subplot(gr[1, :])
        xpos = np.arange(1, nm_ph + 1)
        act = FLUX[FLUX > 0]
        FL0 = max(act.min() * 0.4, 1e-15) if act.size else 1e-15
        FL1 = max(FLUX.max() * 3, FL0 * 10)
        bars_em = axf.bar(xpos - 0.2, np.full(nm_ph, FL0), width=0.38,
                          bottom=FL0, color=MODE_COLORS[:nm_ph],
                          edgecolor='k', lw=0.6, label='emission')
        bars_ab = axf.bar(xpos + 0.2, np.full(nm_ph, FL0), width=0.38,
                          bottom=FL0, color=MODE_COLORS[:nm_ph],
                          edgecolor='k', lw=0.6, hatch='///',
                          label='absorption')
        axf.set_yscale('log')
        axf.set_ylim(FL0, FL1)
        axf.yaxis.set_minor_locator(NullLocator())
        axf.set_xlim(0.4, nm_ph + 0.6)
        axf.set_xticks(xpos)
        axf.set_xlabel('phonon mode', fontsize=13)
        axf.set_ylabel('flux (1/fs)', fontsize=13)
        axf.tick_params(labelsize=TICKSIZE)
        axf.legend(frameon=False, fontsize=9, loc='upper left', ncol=2)
        print(f"  shared flux axis {FL0:.2e} .. {FL1:.2e} /fs "
              f"(absolute rate; particle number is conserved)")

    # ---- F(E) ------------------------------------------------------------
    axe = fig.add_subplot(gr[0, 1], sharey=axb)
    Es = np.linspace(E.min() - 0.02, E.max() + 0.02, 600)
    sig = SIGMA * 1e-3
    fline, = axe.plot([], [], color='magenta', lw=1.8)
    fe_max = 0.0
    for k in range(len(t)):
        nz = F[k] > 0
        if nz.any():
            fe_max = max(fe_max, (F[k][nz][None, :] * np.exp(
                -0.5 * ((Es[:, None] - E[nz][None, :]) / sig) ** 2)
            ).sum(axis=1).max())
    axe.set_xlim(0, max(fe_max * 1.1, 1e-12))
    axe.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    axe.xaxis.offsetText.set_visible(False)
    ex = int(np.floor(np.log10(max(fe_max, 1e-30))))
    axe.set_xlabel(rf"F(E) ($\times 10^{{{ex}}}$)", fontsize=12)
    axe.tick_params(labelsize=TICKSIZE)
    axe.locator_params(axis='x', nbins=2)
    plt.setp(axe.get_yticklabels(), visible=False)

    # A PolyCollection whose vertices are updated, rather than a fresh
    # fill_between each frame: blitting redraws only artists it already
    # knows about, and a newly created one is invisible to the cache.
    fillpoly = PolyCollection([np.zeros((3, 2))], facecolor='magenta',
                              alpha=0.15, edgecolor='none')
    axe.add_collection(fillpoly)

    gmax = F[:, keep].max()
    hold = {'surf': None}

    def frame(k):
        f = F[k]
        m = keep & (f > FLOOR)
        for art, want_em in ((scat_em, True), (scat_ab, False)):
            mm = m & (IE[k] == want_em)
            if mm.any():
                ref = gmax if SCALE == "global" else max(f[m].max(), FLOOR)
                lf = np.log10(np.maximum(f[mm], FLOOR) / FLOOR)
                lref = np.log10(max(ref, FLOOR * 10) / FLOOR)
                s_ = 20 + SMAX * np.clip(lf / max(lref, 1e-12), 0, 1)
                o = np.argsort(s_)[::-1]
                art.set_offsets(np.column_stack([s_coord[mm], E[mm]])[o])
                art.set_sizes(s_[o])
                art.set_array(DM[k][mm].astype(float)[o] + 1)
            else:
                art.set_offsets(np.empty((0, 2)))
                art.set_sizes([])
                art.set_array(np.array([]))

        nz = f > 0
        fe = np.zeros_like(Es)
        if nz.any():
            fe = (f[nz][None, :] * np.exp(
                -0.5 * ((Es[:, None] - E[nz][None, :]) / sig) ** 2)).sum(axis=1)
        fline.set_data(fe, Es)
        fillpoly.set_verts([np.column_stack(
            [np.r_[0.0, fe, 0.0], np.r_[Es[0], Es, Es[-1]]])])

        if has_flux:
            for i in range(nm_ph):
                bars_em[i].set_width(max(FLUX[k, 1, i], FL0) - FL0)
                bars_ab[i].set_width(max(FLUX[k, 0, i], FL0) - FL0)

        if not NO3D:
            # a Poly3DCollection cannot be updated in place; remove and redraw
            if hold['surf'] is not None:
                hold['surf'].remove()
            Z = Zs[k].astype(float)
            cols = cmap3d(norm3d(Z))
            cols[edge & (Z < HEXZ * zmax)] = mcolors.to_rgba(HEXCOLOR)
            hold['surf'] = ax3d.plot_surface(
                X, Y, Z, facecolors=cols, edgecolor='none',
                rstride=STRIDE3D, cstride=STRIDE3D,
                antialiased=False, shade=False)
            hold['surf'].set_clip_on(False)

        title.set_text(f"$t = {t[k]:.0f}$ fs")
        arts = [scat_em, scat_ab, fline, fillpoly, title]
        if has_flux:
            arts += list(bars_em) + list(bars_ab)
        if not NO3D and hold['surf'] is not None:
            arts.append(hold['surf'])
        return arts

    # Blitting redraws only the artists frame() returns and leaves the axes,
    # bands and labels cached. It cannot be used while the 3D surface is
    # rebuilt every frame, because a newly created artist is not in the
    # cached background -- so --no3d turns it on.
    use_blit = NO3D
    print(f"  blitting {'ON' if use_blit else 'OFF (3D rebuilt each frame)'}"
          f"; 3D stride {STRIDE3D}")
    anim = FuncAnimation(fig, frame, frames=len(t), blit=use_blit,
                         interval=1000 / FPS)

    if OUT.lower().endswith(".mp4"):
        try:
            anim.save(OUT, writer=FFMpegWriter(fps=FPS,
                                                 bitrate=BITRATE),
                      dpi=DPI)
        except Exception as e:
            print(f"  ffmpeg failed ({e}); falling back to GIF")
            OUT = OUT[:-4] + ".gif"
            anim.save(OUT, writer=PillowWriter(fps=FPS), dpi=DPI)
    else:
        anim.save(OUT, writer=PillowWriter(fps=FPS), dpi=DPI)
    print(f"saved {OUT} ({len(t)} frames at {FPS} fps)")


if __name__ == "__main__":
    main()
