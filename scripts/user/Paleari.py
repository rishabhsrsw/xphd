import numpy as np
import colormaps as cmaps
import os
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from yambopy.dbs.excitondb import YamboExcitonDB
from yambopy.dbs.latticedb import YamboLatticeDB
from yambopy.dbs.wfdb import YamboWFDB

try:
    from yambopy.plot.plotting_results import BZ_Wigner_Seitz
except ImportError:
    from yambopy import BZ_Wigner_Seitz

# ==============================================================================
# 1. Fulvio's Corrected Weight Extractor (Strict 3D Indexing)
# ==============================================================================
def get_rotated_weights(excdb, Akcv_state):
    """Calculates weights using strict 3D [k, c, v] indices to prevent scrambling."""
    weights = np.zeros([excdb.nkpoints, excdb.mband])
    
    for k in range(excdb.nkpoints):
        for c in excdb.unique_cbands:
            c_tmp = c - excdb.start_band - len(excdb.unique_cbands) 
            for v in excdb.unique_vbands:
                v_tmp = v - excdb.start_band 
                
                # NO FLATTENING: Exact 3D Coordinate mapping
                this_weight = np.abs(Akcv_state[k, c_tmp, v_tmp])**2
                weights[k, c] += this_weight
                weights[k, v] += this_weight
                
    return weights

# ==============================================================================
# 2. Custom 2D Formatter & Tiler
# ==============================================================================
def get_rotated_2D(excdb, lattice, Akcv_state):
    """Sums bands and replicates K-mesh for the extended zone scheme."""
    weights = get_rotated_weights(excdb, Akcv_state)

    # Sum all the bands
    weights_bz_sum = np.sum(weights, axis=1)
    weights_bz_sum = weights_bz_sum / np.max(weights_bz_sum)

    # Optional NOISE FLOOR (uncomment to clean up interpolation ghosts)
    #noise_tolerance = 0.15 
    #weights_bz_sum[weights_bz_sum < noise_tolerance] = 0.0

    # Replicate the mesh (3x3 grid) to prevent cutoff at the BZ edges
    kx_tiled, ky_tiled, w_tiled = [], [], []

    for i_shift in [-1, 0, 1]:
        for j_shift in [-1, 0, 1]:
            shift = i_shift * lattice.rlat[0] + j_shift * lattice.rlat[1]
            shifted_kpts = lattice.car_kpoints + shift

            kx_tiled.extend(shifted_kpts[:, 0])
            ky_tiled.extend(shifted_kpts[:, 1])
            w_tiled.extend(weights_bz_sum)

    return np.array(kx_tiled), np.array(ky_tiled), np.array(w_tiled)

# ==============================================================================
# 3. Custom RBF Plotter
# ==============================================================================
def plot_rotated_2D_ax(ax, lattice, x, y, weights_bz_sum, limfactor=0.65, cmap='Greens', npts=300):
    """Uses Scipy's Rbf and imshow to create smooth, published-quality blobs."""
    ax.set_facecolor(plt.get_cmap(cmap)(0.0))

    lim = np.max(np.linalg.norm(lattice.rlat[:2], axis=1)) * limfactor

    # Draw Brillouin Zone
    bz = BZ_Wigner_Seitz(lattice)
    bz.set_facecolor('none')
    bz.set_edgecolor('black')
    bz.set_linewidth(2.0)
    bz.set_zorder(10)
    ax.add_patch(bz)

    # Smooth RBF Interpolation
    rbfi = Rbf(x, y, weights_bz_sum, function='cubic')
    xi = yi = np.linspace(-lim, lim, npts)
    grid_weights = np.zeros([npts, npts])

    for col in range(npts):
        grid_weights[:, col] = rbfi(xi, np.ones_like(xi) * yi[col])

    # Plot using imshow
    ax.imshow(grid_weights.T, interpolation='bicubic', origin='lower',
              extent=[-lim, lim, -lim, lim], cmap=cmap, vmin=0.0)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_yticks([])
    ax.set_xticks([])

    # Format axes
   # ax.set_xlabel(r'$k_x$ ($\mathrm{\AA}^{-1}$)', fontsize=12)
   # ax.set_ylabel(r'$k_y$ ($\mathrm{\AA}^{-1}$)', fontsize=12)
   # ax.tick_params(direction='in', top=True, right=True, labelsize = 10)

    return ax

# ==============================================================================
# MAIN CALCULATION
# ==============================================================================
## Inputs
iqpt = 1
path = '/scratch/sitangshu.iiita/GaN/24x24x1/work_dir/BSE_GW_diago/'
gw_bse_dir = 'output'
iexe = 38
degen_tol = 1e-3

# Load databases
lattice = YamboLatticeDB.from_db_file(os.path.join(path, 'SAVE', 'ns.db1'))
filename = 'ndb.BS_diago_Q%d' % (iqpt)
excdb = YamboExcitonDB.from_db_file(lattice, filename=filename,
                                    folder=os.path.join(path, gw_bse_dir), neigs = iexe + 101)
wfdb = YamboWFDB(path=path, latdb=lattice, bands_range=[np.min(excdb.table[:, 1]) - 1, np.max(excdb.table[:, 2])])

# Compute Total Crystal Angular Momentum
symm_mat_cart = lattice.sym_car[1]
frac_trans_cart = np.zeros(3)
sbasis = excdb.total_crys_angular_momentum(wfdb, iexe, symm_mat_cart, frac_trans_cart, degen_tol=degen_tol)

jvals = sbasis[0]
Akcv_j = sbasis[1]
idegen = np.array(excdb.get_degenerate(iexe + 1, eps=degen_tol), dtype=int) - 1

print('Total crystal angular momentum : ', jvals)

# Overwrite the db state array with the rotated ones
old_eig_states = excdb.Akcv[idegen].copy()
excdb.Akcv[idegen] = Akcv_j

print("="*80)
for i, iexc in enumerate(idegen):
    state_idx = iexc + 1
    j_val = jvals[i]

    print('*********** Plotting K-space wf for state %d with j = %.2f ***************' % (state_idx, j_val))

    # Safely extract the coefficients exactly as Fulvio sliced them
    # iexc = state index, 0 = first Q-point, 0 = first spin polarization
    Akcv_state = excdb.Akcv[iexc, 0, 0]

    # Extract and format the data
    x_tiled, y_tiled, w_tiled = get_rotated_2D(excdb, lattice, Akcv_state)

    # Plotting
    fig = plt.figure(figsize=(3,3))
    ax  = fig.add_axes([0.15, 0.15, 0.80, 0.80])

    cmap_choice = cmaps.heart_light if j_val > 0 else cmaps.teals_light
    color_code = cmaps.pr_mist if j_val > 0 else 'red'

    plot_rotated_2D_ax(ax, lattice, x_tiled, y_tiled, w_tiled, cmap=cmap_choice, limfactor=0.65)

    #ax.set_title(rf"State {state_idx} ($j_z = {j_val:.0f}$)", fontsize=16, pad=10, color=color_code)

    plt.savefig(f"Kspace_state_{state_idx}_jz_{j_val:.2f}.png", dpi=300, bbox_inches='tight')
    plt.show()

print("="*80)

# Restore original states
excdb.Akcv[idegen] = old_eig_states
