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
    """Replicates q-points across a 3x3 grid."""
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
    ndb_elph  = '../LELPH/ndb.elph'
    exph_file = 'GaN-ph_path.npy'

    iQ = 12                       
    dark_idx = 0                  
    bright_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]           

    # --- LOAD DATA ---
    print("Loading lattice and matrix elements...")
    ylat = YamboLatticeDB.from_db_file(filename=f'{save_path}/ns.db1')
    elph = LetzElphElectronPhononDB(ndb_elph, read_all=False)
    exph = np.load(exph_file)     

    # --- SLICE AND SUM (MODE RESOLVED) ---
    # Step 1: Extract 3D array [q_points, modes, all_excitons]
    data_slice = exph[iQ, :, :, dark_idx, :]
    
    # Step 2: Select bright excitons [576, N_modes, 2]
    coupling_raw = np.abs(data_slice[:, :, bright_idx])**2
    
    # Step 3: Sum ONLY over bright states (axis 2). 
    # Do NOT sum over modes (axis 1). Result: [576, N_modes]
    coupling_maps = np.sum(coupling_raw, axis=2)

    # --- MASKING FOR THE 2D PLANE ---
    indx2D = np.where(np.abs(ylat.car_kpoints[:, 2]) < 1e-5)[0]
    qpts_2D = ylat.car_kpoints[indx2D]
    
    # Apply mask and scale units. Shape remains [N_2D_points, N_modes]
    gkkp_2D_modes = coupling_maps[indx2D, :] * (ha2ev**2) * 1e4

    n_qpts, n_modes = gkkp_2D_modes.shape
    print(f"Data shape matched: Q-pts={n_qpts}, Modes={n_modes}")

    # --- SETUP DYNAMIC FIGURE ---
    cols = min(3, n_modes) 
    rows = int(np.ceil(n_modes / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4.5))
    if n_modes > 1: axes = axes.flatten()
    else: axes = [axes]

    cmap = cmaps.yel15_r
    lim = 0.15
    vmax = np.max(gkkp_2D_modes) # Global max for consistent color scaling

    print("Plotting individual modes with local scaling...")
    for i_mode in range(n_modes):
        ax = axes[i_mode]
        ax.set_facecolor(cmap(0.0))
        
        # 1. Extract 1D array for this specific mode
        gkkp_single_mode = gkkp_2D_modes[:, i_mode]
        
        # --- THE FIX: Calculate max for THIS MODE ONLY ---
        local_vmax = np.max(gkkp_single_mode)
        # Prevent division by zero if a mode is truly mathematically 0
        if local_vmax < 1e-10: local_vmax = 1e-10 
        
        # 2. Tile data
        qx_tiled, qy_tiled, gkkp_tiled = get_tiled_qpoints(ylat, qpts_2D, gkkp_single_mode)
        
        # 3. Interpolate
        rbfi = Rbf(qx_tiled, qy_tiled, gkkp_tiled, function='linear')
        xi = yi = np.linspace(-lim, lim, 150)
        grid_weights = np.zeros([150, 150])
        for col in range(150):
            grid_weights[:, col] = rbfi(xi, np.ones_like(xi) * yi[col])
            
        # 4. Render using the LOCAL vmax
        im = ax.imshow(grid_weights.T, interpolation='bicubic', origin='lower',
                       extent=[-lim, lim, -lim, lim], cmap=cmap, 
                       vmin=0.0, vmax=local_vmax)
                       
        # Draw BZ
        bz = BZ_Wigner_Seitz(ylat)
        bz.set_facecolor('none')
        bz.set_edgecolor('white')
        bz.set_linewidth(1.5)
        ax.add_patch(bz)
        
        # Update title to show the true numerical maximum!
        ax.set_title(f'Phonon Mode {i_mode+1}\nMax: {local_vmax:.1f}', fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
    # Cleanup unused subplots
    for j in range(i_mode + 1, len(axes)):
        axes[j].axis('off')

    # Add a single colorbar for the entire figure
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=14)
    if vmax > 0:
        cbar.set_ticks([0.0, vmax / 3, (2 * vmax) / 3, vmax])
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    plt.subplots_adjust(wspace=0.1, hspace=0.2, right=0.9)
    plt.savefig('M.png', dpi=600, bbox_inches='tight')
    print("Saved successfully as GaN_Mode_Resolved_Coupling.png!")
