#!/usr/bin/env python3
"""
bte_snapshots_3d_stacked.py
===========================
Layout:
  - Top Row: 2D Energy distribution F(E) vs Energy (magenta).
  - Bottom Row: 3D Brillouin Zone maps for N_Q(t) (custom viridis with gray base). 
  - Right Column: A single colorbar spanning the height.

Usage:
    python bte_snapshots_3d_stacked.py snaps.npz --times 30 200 400 --gamma 0.35
"""
from __future__ import annotations
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import griddata
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter
import warnings

# --- Style Settings ---
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'DejaVu Sans',
    'mathtext.fontset': 'stix',
})

def wrap(d):
    d = np.asarray(d, float)
    return d - np.rint(d)

def to_cart_unwrapped(qr, alat):
    """Reduced -> Cartesian (1/Ang) for a hexagonal cell (unwrapped)."""
    b = 4 * np.pi / (np.sqrt(3) * alat)
    x = (qr[:, 0] + 0.5 * qr[:, 1]) * b
    y = (np.sqrt(3) / 2) * qr[:, 1] * b
    return np.stack([x, y], 1)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("snaps")
    p.add_argument("--times", type=float, nargs="*", default=None,
                   help="fs; the nearest saved snapshots are used")
    p.add_argument("--alat", type=float, default=3.2, help="Angstrom")
    p.add_argument("--sigma", type=float, default=8.0,
                   help="meV, broadening for the F(E) panel")
    p.add_argument("--gamma", type=float, default=1.0,
                   help="height exponent; e.g., 0.35 severely compresses huge peaks to reveal small ones")
    p.add_argument("--out", default="bte_3d_stacked.png")
    a = p.parse_args()

    # 1. Load Data
    d = np.load(a.snaps)
    t, F, E, Qr = d["t"], d["F"], d["E"], d["Q_red"]
    
    print(f"Loaded {a.snaps}: {len(t)} snapshots, {len(E)} states total.")
    
    want = a.times if a.times else [t[len(t)//4], t[len(t)//2], t[-1]]
    idx = [int(np.argmin(np.abs(t - w))) for w in want]

    # 2. Prepare F(E) Energy Distribution Data (Top Row)
    Es = np.linspace(E.min() - 0.02, E.max() + 0.02, 600)
    sig = a.sigma * 1e-3
    FEs = []
    
    for k in idx:
        fk = F[k]
        nz = fk > 0
        if nz.any():
            FE = (fk[nz][None, :] * np.exp(-0.5*((Es[:, None]-E[nz][None, :])/sig)**2)).sum(axis=1)
        else:
            FE = np.zeros_like(Es)
        FEs.append(FE)
        
    max_FE = max([fe.max() for fe in FEs])
    if max_FE <= 0: max_FE = 1e-10

    # 3. Collapse bands for 3D BZ Map (Bottom Row)
    w = wrap(Qr[:, :2])
    xu, inv = np.unique(np.round(w, 5), axis=0, return_inverse=True)
    
    F_summed = []
    max_f_global = 0.0
    for k in idx:
        fk = np.bincount(inv, weights=F[k])
        F_summed.append(fk)
        max_f_global = max(max_f_global, fk.max())
        
    if max_f_global <= 0: max_f_global = 1e-12

    # 4. Tile the BZ (3x3 grid) for smooth edge interpolation
    q_tiled = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            q_tiled.append(xu + np.array([dx, dy]))
    q_tiled = np.vstack(q_tiled)
    cart_tiled = to_cart_unwrapped(q_tiled, a.alat)

    b = 4 * np.pi / (np.sqrt(3) * a.alat)
    limit = b * 0.72  
    
    # Restored high resolution to fix jaggedness
    grid_res = 250
    xi = np.linspace(-limit, limit, grid_res)
    yi = np.linspace(-limit, limit, grid_res)
    X, Y = np.meshgrid(xi, yi)

    # Hexagon boundaries
    vertices_red = np.array([
        [ 1/3,  1/3], [-1/3,  2/3], [-2/3,  1/3],
        [-1/3, -1/3], [ 1/3, -2/3], [ 2/3, -1/3], [ 1/3,  1/3]
    ])
    hex_x, hex_y = to_cart_unwrapped(vertices_red, a.alat).T

    # --- CUSTOM COLORMAP ---
    # Replaces the absolute bottom of Viridis with a flat light gray to clearly distinguish the plane
    viridis_colors = plt.cm.viridis(np.linspace(0, 1, 256))
    viridis_colors[:4, :] = mcolors.to_rgba('black') 
    custom_cmap = mcolors.ListedColormap(viridis_colors)

    # ==================================================================
    # 5. BUILD THE FIGURE
    # ==================================================================
    fig = plt.figure(figsize=(5.5 * len(idx) + 1.5, 7.5))
    
    # Adjusted hspace and height ratios to prevent the 3D plot from overlapping the 2D plot
    gs = fig.add_gridspec(2, len(idx) + 1, 
                          width_ratios=[1]*len(idx) + [0.05],
                          height_ratios=[0.7, 3.5],
                          wspace=0.15, hspace=0.35)

    zmax = max_f_global ** a.gamma
    surf = None

    for i, (k, fe, fk) in enumerate(zip(idx, FEs, F_summed)):
        
        # --- TOP ROW: 2D F(E) PLOT ---
        ax2d = fig.add_subplot(gs[0, i])
        ax2d.fill_between(Es, fe, color="magenta", alpha=0.15)
        ax2d.plot(Es, fe, color="magenta", lw=2, zorder=3)
        ax2d.set_xlim(Es.min(), Es.max())
        ax2d.set_ylim(0, max_FE * 1.1)
        ax2d.set_xlabel("Energy (eV)", fontsize=13)
        ax2d.set_title(f'$t = {t[k]:.0f}$ fs', fontsize=18, fontweight='bold', pad=15)
        
        if i == 0:
            ax2d.set_ylabel(r"$F(E)$", fontsize=15)
        else:
            ax2d.set_yticks([])

# --- BOTTOM ROW: 3D BZ MAP ---
        ax3d = fig.add_subplot(gs[1, i], projection='3d')
        fk_tiled = np.tile(fk, 9)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Z = griddata(cart_tiled, fk_tiled, (X, Y), method='cubic', fill_value=0.0)
            if not np.isfinite(Z).all():
                Z = griddata(cart_tiled, fk_tiled, (X, Y), method='linear', fill_value=0.0)
        
        # --- THE FIX ---
        # 1. Clean up any negative ringing from the cubic interpolation
        Z = np.clip(Z, 0, None)
        
        # 2. Smoothly subtract the thermal baseline (the "water level").
        # Unlike np.where (which creates a cliff), subtraction ensures the data 
        # slopes perfectly into 0.0 continuously. 
        # (Subtracting 0.5% of the global max is usually enough to sink the sheet).
        cutoff = max_f_global * 0.005 
        Z = np.clip(Z - cutoff, 0, None)
        
        # 3. Apply the gamma compression. Since the background is now exactly 0.0,
        # 0.0 ** gamma remains 0.0, completely preventing the raised sheet effect.
        Z = Z ** a.gamma
        
        # 4. Finally, apply a gentle blur to the compressed, clean peaks.
        Z = gaussian_filter(Z, sigma=1.2)

        surf = ax3d.plot_surface(X, Y, Z, cmap=custom_cmap, edgecolor='none', 
                                 rstride=1, cstride=1, antialiased=True,
                                 vmin=0, vmax=zmax)
        
        # Hexagon outline floating just above zero
        ax3d.plot(hex_x, hex_y, np.full_like(hex_x, zmax * 0.015), 
                  'w--', lw=2.5, alpha=0.8, zorder=10)
        
        ax3d.set_xlim(-limit, limit)
        ax3d.set_ylim(-limit, limit)
        ax3d.set_zlim(0, zmax * 1.1)
        ax3d.view_init(elev=35, azim=-65)
        
        # Completely remove all axes, grids, background panes, and ticks
        ax3d.axis('off')
        
        if i == 0:
            ax3d.text(hex_x[0]*1.1, hex_y[0]*1.1, zmax * 0.05, "K", 
                      color='black', fontsize=16, fontweight='bold', zorder=20)

    # --- COLORBAR ---
    cax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(surf, cax=cax)
    
    if a.gamma != 1.0:
        cbar.set_label(rf'$[N_{{\mathbf{{Q}}}}(t)]^{{{a.gamma}}}$', fontsize=16, labelpad=15)
    else:
        cbar.set_label(r'$N_{\mathbf{Q}}(t)$', fontsize=18, labelpad=15)
        
    cbar.ax.tick_params(labelsize=14)

    plt.savefig(a.out, dpi=300, bbox_inches='tight')
    print(f"Saved {a.out}")

if __name__ == "__main__":
    main()