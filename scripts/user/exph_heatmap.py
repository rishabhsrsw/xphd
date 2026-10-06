import numpy as np
import colormaps as cmaps
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import Rbf
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Yambopy imports
from yambopy import YamboLatticeDB
from yambopy.units import ha2ev

try:
    from yambopy.plot.plotting_results import BZ_Wigner_Seitz
except ImportError:
    from yambopy import BZ_Wigner_Seitz

mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.family"] = "Montserrat"

# ==============================================================================
# 1. Data Processing Functions
# ==============================================================================
def GET_G2_to_plot(G_squared, exc_in, exc_out, ph_in):
    """Slices and sums the matrix elements for specific exciton and phonon states."""
    if exc_in  == 'all': exc_in  = range(G_squared.shape[2])
    if exc_out == 'all': exc_out = range(G_squared.shape[3])
    if ph_in   == 'all': ph_in   = range(G_squared.shape[1])

    G_squared = G_squared[:, ph_in, :, :].sum(axis=1)
    G_squared = G_squared[:, exc_in, :].sum(axis=1)
    G_squared = G_squared[:, exc_out].sum(axis=1)

    F_q = G_squared * (ha2ev**2)
    return F_q

# ==============================================================================
# 2. Tiling Function (Extended Zone Scheme)
# ==============================================================================
def get_tiled_qpoints(lattice, q_car, weights):
    """Replicates q-points across a 3x3 grid to prevent boundary cutoffs."""
    qx_tiled, qy_tiled, w_tiled = [], [], []
    for i_shift in [-1, 0, 1]:
        for j_shift in [-1, 0, 1]:
            shift = i_shift * lattice.rlat[0] + j_shift * lattice.rlat[1]
            shifted_qpts = q_car + shift

            qx_tiled.extend(shifted_qpts[:, 0])
            qy_tiled.extend(shifted_qpts[:, 1])
            w_tiled.extend(weights)

    return np.array(qx_tiled), np.array(qy_tiled), np.array(w_tiled)

# ==============================================================================
# 3. Main Script
# ==============================================================================
if __name__ == "__main__":

    # --- USER INPUTS ---
    path = 'GaN' 
    ns_db1 = f'..//LELPH/SAVE/ns.db1'
    ns_ypy = 'GaN-ph.npy'

    exc_in  = [0,1]  
    exc_out = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    # --- LOAD DATA (ONLY ONCE TO SAVE TIME) ---
    print("Loading lattice and matrix elements...")
    ylat = YamboLatticeDB.from_db_file(filename=ns_db1)
    indx2D = (ylat.car_kpoints[:, 2] == 0.).nonzero()[0]
    qpts = ylat.car_kpoints[indx2D]

    X_py = np.load(ns_ypy)
    G_squared = np.abs(X_py)**2.

    # --- LOOP OVER PHONON MODES (0 to 5) ---
    for p_idx in range(6):
        ph_in = [p_idx]
        print(f"\n======================================")
        print(f"Processing phonon mode: ph_in = {p_idx}")
        
        gkkp = GET_G2_to_plot(G_squared, exc_in, exc_out, ph_in)[indx2D] * 1e4

        # --- TILE DATA ---
        print("Tiling Brillouin Zone...")
        qx_tiled, qy_tiled, gkkp_tiled = get_tiled_qpoints(ylat, qpts, gkkp)

        # --- SETUP FIGURE ---
        print("Plotting...")
        fig = plt.figure(figsize=(5.0, 4.0))

        ax = fig.add_axes([0.05, 0.05, 0.72, 0.90])
        cax = fig.add_axes([0.79, 0.05, 0.04, 0.90])

        cmap = cmaps.yel15_r
        ax.set_facecolor(cmap(0.0))
        lim = 0.15

        # --- RBF INTERPOLATION ---
        npts = 150
        rbfi = Rbf(qx_tiled, qy_tiled, gkkp_tiled, function='linear')
        xi = yi = np.linspace(-lim, lim, npts)
        grid_weights = np.zeros([npts, npts])

        for col in range(npts):
            grid_weights[:, col] = rbfi(xi, np.ones_like(xi) * yi[col])

        im = ax.imshow(grid_weights.T, interpolation='bicubic', origin='lower',
                       extent=[-lim, lim, -lim, lim], cmap=cmap, vmin=0.0, vmax=np.max(gkkp))

        # --- DRAW BRILLOUIN ZONE ---
        bz = BZ_Wigner_Seitz(ylat)
        bz.set_facecolor('none')
        bz.set_edgecolor('white')
        bz.set_linewidth(1.5)
        bz.set_zorder(10)
        ax.add_patch(bz)

        # --- FORMAT COLORBAR ---
        max_value = np.max(gkkp)
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cbar.ax.tick_params(labelsize=25)
        cbar.set_ticks([0.0, max_value / 3, (2 * max_value) / 3, max_value])
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        # --- AESTHETICS ---
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')

        # --- EXPORT ---
        out_name = f'({p_idx + 1}).png'
        plt.savefig(out_name, dpi=600, bbox_inches='tight', transparent=False)
        plt.close(fig) # IMPORTANT: Prevents memory leaks by closing the figure after saving
        print(f"Saved successfully: {out_name}")

    print("\nAll phonon modes processed!")
