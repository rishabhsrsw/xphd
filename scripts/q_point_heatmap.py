import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import Rbf
from matplotlib.ticker import FormatStrFormatter
import colormaps as cmaps

# Yambopy imports
from yambopy import YamboLatticeDB, LetzElphElectronPhononDB
from yambopy.units import ha2ev

try:
    from yambopy.plot.plotting_results import BZ_Wigner_Seitz
except ImportError:
    from yambopy import BZ_Wigner_Seitz

mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.family"] = "Montserrat"


# ==============================================================================
# 1. Tiling Function (Dimension-Safe)
# ==============================================================================
def get_tiled_qpoints(lattice, q_car, weights):
    """Replicates q-points across a 3x3 grid to prevent boundary cutoffs."""
    qx_tiled, qy_tiled, w_tiled = [], [], []
    v1 = lattice.rlat[0, :2]
    v2 = lattice.rlat[1, :2]
    for i_shift in [-1, 0, 1]:
        for j_shift in [-1, 0, 1]:
            shift = i_shift * v1 + j_shift * v2
            shifted_qpts = q_car[:, :2] + shift
            qx_tiled.extend(shifted_qpts[:, 0])
            qy_tiled.extend(shifted_qpts[:, 1])
            w_tiled.extend(weights)
    return np.array(qx_tiled), np.array(qy_tiled), np.array(w_tiled)


# ==============================================================================
# 2. Main Script
# ==============================================================================
if __name__ == "__main__":

    # --- USER INPUTS ---
    save_path = '../LELPH/SAVE'
    ndb_elph = '../LELPH/ndb.elph'
    exph_file = 'GaN-ph_path.npy'

    iQ = 12
    dark_idx = 0
    bright_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    out_name = 'Q_heatmap.png'

    # --- LOAD DATA ---
    print("Loading lattice and matrix elements...")
    ylat = YamboLatticeDB.from_db_file(filename=f'{save_path}/ns.db1')
    elph = LetzElphElectronPhononDB(ndb_elph, read_all=False)
    exph = np.load(exph_file)

    # --- SLICE AND SUM OVER MODES AND FINAL STATES ---
    # exph[iQ] is [q_points, modes, exc_in, exc_out]. Summing over BOTH the
    # phonon modes and the final exciton states leaves a scalar per q, which
    # is the quantity that is invariant under the little group of Q: individual
    # modes and individual final states mix among themselves under the point
    # group, and only the complete sums are well defined.
    data_slice = exph[iQ, :, :, dark_idx, :]           # [q, modes, exc_out]
    coupling_raw = np.abs(data_slice[:, :, bright_idx]) ** 2
    coupling_map = np.sum(coupling_raw, axis=(1, 2))   # [q]

    # --- MASK THE 2D PLANE ---
    indx2D = np.where(np.abs(ylat.car_kpoints[:, 2]) < 1e-5)[0]
    qpts_2D = ylat.car_kpoints[indx2D]
    gkkp = coupling_map[indx2D] * (ha2ev ** 2) * 1e4

    print(f"Summed over {coupling_raw.shape[1]} modes and "
          f"{len(bright_idx)} final states; {len(gkkp)} q-points in plane")
    print(f"  max |G|^2 (scaled) = {np.max(gkkp):.3f}")

    # --- TILE DATA ---
    print("Tiling Brillouin Zone...")
    qx_tiled, qy_tiled, gkkp_tiled = get_tiled_qpoints(ylat, qpts_2D, gkkp)

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
                   extent=[-lim, lim, -lim, lim], cmap=cmap,
                   vmin=0.0, vmax=np.max(gkkp))

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
    plt.savefig(out_name, dpi=600, bbox_inches='tight', transparent=False)
    plt.close(fig)
    print(f"Saved successfully: {out_name}")

    # The little group of Q sets the symmetry of this map. At Gamma it is the
    # full point group; at K under D3h it is C3h, whose only operations acting
    # on in-plane q are the three rotations -- sigma_h and S3 leave (qx, qy)
    # unchanged. The map is therefore THREE-fold symmetric about K, not
    # six-fold. Six-fold would mean something is symmetrising it that should
    # not be.
