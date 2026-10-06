#!/usr/bin/env python3
"""
bte_animate.py
==============
Animate exciton populations on the band structure, from the snapshots written
by rt-BTE.py.

Layout explicitly matched to the right-hand panel of bte_final.py:
  1. Exciton dispersion with markers at computed Q-points perfectly 
     projected onto the analytic path (colored by phonon mode).
  2. F(E) energy distribution (magenta).
  3. Flux carried by each phonon mode (split emission/absorption).

Usage:
    python bte_animate.py snaps.npz --disp o-output.excitons_interpolated_01 --out cascade.mp4
"""
from __future__ import annotations
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

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

def path_from_disp(disp, nb, alat):
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
            print(f"  [warn] the path never passes {l} (closest approach {d[k]:.3f}); dropping it")
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
    p.add_argument("--disp", default="o-output.excitons_interpolated")
    p.add_argument("--nbands", type=int, default=5)
    p.add_argument("--alat", type=float, default=3.2)
    p.add_argument("--path", nargs="*", default=["G", "M", "K", "G"])
    p.add_argument("--tol", type=float, default=0.06)
    p.add_argument("--floor", type=float, default=1e-10)
    p.add_argument("--smax", type=float, default=260.0)
    p.add_argument("--sigma", type=float, default=8.0, help="meV, F(E) width")
    p.add_argument("--scale", choices=("global", "frame"), default="global")
    p.add_argument("--fps", type=int, default=40)
    p.add_argument("--emax", type=float, default=5.0)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--out", default="bte_animation.mp4")
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--bitrate", type=int, default=4000)
    a = p.parse_args()

    # ---- Load Data ----
    d = np.load(a.snaps)
    t, F, E, Qr = d["t"][::a.stride], d["F"][::a.stride], d["E"], d["Q_red"]
    
    has_influx = "dom_mode_t" in d
    has_outflux = "dom_mode" in d
    has_chan = has_influx or has_outflux
    has_flux = "flux_t" in d

    if has_influx:
        DM = np.asarray(d["dom_mode_t"], int)[::a.stride]
        IE = np.asarray(d["is_emission_t"], bool)[::a.stride]
        nm_ph = int(DM.max()) + 1
    elif has_outflux:
        _dm = np.asarray(d["dom_mode"], int)
        _ie = np.asarray(d["is_emission"], bool)
        DM = np.tile(_dm, (len(t), 1))
        IE = np.tile(_ie, (len(t), 1))
        nm_ph = int(DM.max()) + 1
    else:
        DM = np.zeros((len(t), len(E)), int)
        IE = np.ones((len(t), len(E)), bool)
        nm_ph = 1

    if has_flux:
        FLUX = np.asarray(d["flux_t"], float)[::a.stride]

    print(f"{a.snaps}: {len(t)} frames, {len(E)} states, t = {t[0]:.0f} .. {t[-1]:.0f} fs")

    # ---- Geometry & Projection ----
    disp = np.loadtxt(a.disp, comments='#')
    nb = a.nbands
    bx, path_cart, path_red = path_from_disp(disp, nb, a.alat)
    E_band = disp[:, 1:1 + nb]
    ticks, tick_labels = ticks_from_path(bx, path_red, a.path)

    st_cart = to_cart(Qr, a.alat, fold=True)
    s_coord, dmin = project_to_path(st_cart, path_cart, bx)
    keep = dmin < a.tol

    print(f"  path {bx[0]:.3f} .. {bx[-1]:.3f}, ticks {np.round(ticks, 3)} = {tick_labels}")
    print(f"  {keep.sum()} of {len(E)} states within {a.tol} 1/Ang of it")

    # ---- Figure Layout ----
    fig = plt.figure(figsize=(9, 12), dpi=a.dpi)
    gs = fig.add_gridspec(3, 2, height_ratios=[2.4, 1.0, 1.2], width_ratios=[1, 0.035], hspace=0.35, wspace=0.05)

    # 1. Band Structure Panel
    axb = fig.add_subplot(gs[0, 0])
    for n in range(nb):
        axb.plot(bx, E_band[:, n], linestyle='-', lw=2, c='gray', zorder=1, alpha=0.8)
    for x in ticks:
        axb.axvline(x, color="k", lw=0.7, zorder=1)

    axb.set_xlim(bx[0], bx[-1])
    axb.set_xticks(ticks)
    axb.set_xticklabels([l.replace("G", r"$\Gamma$") for l in tick_labels])
    axb.set_ylim(E[keep].min() - 0.03, a.emax)
    axb.set_ylabel("Exciton energy (eV)", fontsize=14)
    title = axb.set_title("", fontsize=16, fontweight='bold', pad=12)

    axb.legend(handles=[
        Line2D([], [], marker='o', ls='', c='0.4', label='emission'),
        Line2D([], [], marker='s', ls='', c='0.4', label='absorption')],
        frameon=False, fontsize=12, loc='lower right')

    cmap = mcolors.ListedColormap(MODE_COLORS)
    norm = mcolors.BoundaryNorm(np.arange(7) + 0.5, 6)

    # Scatter plots updated with black edgecolors as requested
    scat_em = axb.scatter([], [], s=[], c=[], marker="o", cmap=cmap, norm=norm,
                          alpha=0.85, edgecolors='black', linewidths=0.7, zorder=3)
    scat_ab = axb.scatter([], [], s=[], c=[], marker="s", cmap=cmap, norm=norm,
                          alpha=0.85, edgecolors='black', linewidths=0.7, zorder=3)

    if has_chan:
        cax2 = fig.add_subplot(gs[0, 1])
        cb2 = matplotlib.colorbar.ColorbarBase(
            cax2, cmap=cmap, norm=norm, ticks=np.arange(1, 7))
        cb2.set_label("dominant phonon mode", fontsize=13, labelpad=10)
        cb2.ax.tick_params(direction='in')

    # 2. F(E) Panel
    axe = fig.add_subplot(gs[1, :])
    Es = np.linspace(E.min() - 0.02, a.emax + 0.02, 600)
    sig = a.sigma * 1e-3
    fline, = axe.plot([], [], color='magenta', lw=1.8)
    axe.set_xlabel("Energy (eV)", fontsize=14)
    axe.set_ylabel(r"$F(E)$", fontsize=14)
    axe.set_xlim(Es.min(), Es.max())
    
    # Pre-calculate max F(E) to fix ylim securely
    fe_max = 0.0
    for k in range(len(t)):
        nz = F[k] > 0
        if nz.any():
            fe_max = max(fe_max, (F[k][nz][None, :] * np.exp(-0.5 * ((Es[:, None] - E[nz][None, :]) / sig) ** 2)).sum(axis=1).max())
    axe.set_ylim(0, max(fe_max * 1.1, 1e-12))
    axe.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    fill_tracker = {'poly': None}

    # 3. Flux Panel
    if has_flux:
        axf = fig.add_subplot(gs[2, :])
        xpos = np.arange(1, nm_ph + 1)
        
        # Calculate dynamic global floor for log scale
        active_flux = FLUX[FLUX > 0]
        flux_floor = max(active_flux.min() * 0.4, 1e-15) if active_flux.size else 1e-15
        
        bars_em = axf.bar(xpos - 0.2, np.full(nm_ph, flux_floor), width=0.38,
                          bottom=flux_floor, color=MODE_COLORS[:nm_ph],
                          edgecolor="k", lw=0.6, label="emission")
        bars_ab = axf.bar(xpos + 0.2, np.full(nm_ph, flux_floor), width=0.38,
                          bottom=flux_floor, color=MODE_COLORS[:nm_ph],
                          edgecolor="k", lw=0.6, hatch="///", label="absorption")
                          
        axf.set_yscale('log')
        axf.set_ylim(flux_floor, max(FLUX.max() * 2, flux_floor * 10))
        axf.set_xticks(xpos)
        axf.set_xlabel("Phonon mode", fontsize=14)
        axf.set_ylabel("Flux carried (1/fs)", fontsize=14)
        axf.legend(frameon=False, fontsize=11)

    gmax = F.max()

    def frame(k):
        f = F[k]
        m = keep & (f > a.floor)
        dm_k = DM[k]
        ie_k = IE[k]
        
        for art, want_em in ((scat_em, True), (scat_ab, False)):
            mm = m & (ie_k == want_em)
            if mm.any():
                ref = gmax if a.scale == "global" else max(f[m].max(), a.floor)
                lf = np.log10(np.maximum(f[mm], a.floor) / a.floor)
                lref = np.log10(max(ref, a.floor * 10) / a.floor)
                size_arr = 20 + a.smax * np.clip(lf / max(lref, 1e-12), 0, 1)
                
                # Z-Order Sorting
                sort_idx = np.argsort(size_arr)[::-1]
                
                sorted_offsets = np.column_stack([s_coord[mm], E[mm]])[sort_idx]
                sorted_sizes = size_arr[sort_idx]
                # +1 to match the 1-6 BoundaryNorm mapping of the colormap
                sorted_colors = dm_k[mm].astype(float)[sort_idx] + 1 

                art.set_offsets(sorted_offsets)
                art.set_sizes(sorted_sizes)
                art.set_array(sorted_colors)
            else:
                art.set_offsets(np.empty((0, 2)))
                art.set_sizes([])
                art.set_array(np.array([]))
                
        # F(E) Poly Update
        nz = f > 0
        fe = np.zeros_like(Es)
        if nz.any():
            fe = (f[nz][None, :] * np.exp(-0.5 * ((Es[:, None] - E[nz][None, :]) / sig) ** 2)).sum(axis=1)
        
        fline.set_data(Es, fe)
        
        if fill_tracker['poly'] is not None:
            fill_tracker['poly'].remove()
        fill_tracker['poly'] = axe.fill_between(Es, 0, fe, color='magenta', alpha=0.15)

        # Flux Bars Update
        if has_flux:
            for i in range(nm_ph):
                bars_em[i].set_height(max(FLUX[k, 1, i], flux_floor) - flux_floor)
                bars_ab[i].set_height(max(FLUX[k, 0, i], flux_floor) - flux_floor)

        title.set_text(f"Scattering dynamics at $t = {t[k]:.0f}$ fs")
        return scat_em, scat_ab, fline, title

    plt.tight_layout()
    anim = FuncAnimation(fig, frame, frames=len(t), blit=False, interval=1000 / a.fps)

    if a.out.lower().endswith(".mp4"):
        try:
            anim.save(a.out, writer=FFMpegWriter(fps=a.fps, bitrate=a.bitrate), dpi=a.dpi)
        except Exception as e:
            print(f"  ffmpeg failed ({e}); falling back to GIF")
            a.out = a.out[:-4] + ".gif"
            anim.save(a.out, writer=PillowWriter(fps=a.fps), dpi=a.dpi)
    else:
        anim.save(a.out, writer=PillowWriter(fps=a.fps), dpi=a.dpi)
        
    print(f"saved {a.out} ({len(t)} frames at {a.fps} fps)")

if __name__ == "__main__":
    main()