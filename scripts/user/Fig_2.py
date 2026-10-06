import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.image as mpimg
import re
from scipy.interpolate import interp1d, griddata
from matplotlib.collections import LineCollection
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Set styling
plt.rcParams['font.family'] = 'Montserrat'
plt.rcParams["mathtext.fontset"] = 'stix'
plt.rcParams.update({'font.size': 18})

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def autocrop_white(img, thresh=0.985):
    if img.dtype != np.float32 and img.dtype != np.float64:
        im = img.astype(np.float32) / 255.0
    else:
        im = img
    rgb = im[..., :3] 
    mask = np.any(rgb < thresh, axis=-1)
    if not mask.any(): return img
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    return img[y0:y1, x0:x1]

def add_image(ax, path, coord, zoom, crop=False):
    try:
        img = mpimg.imread(path)
        if crop: img = autocrop_white(img)
        imagebox = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(imagebox, coord, frameon=False, zorder=5)
        ax.add_artist(ab)
    except FileNotFoundError:
        print(f"Warning: {path} not found.")

def colored_line(ax, x, y, cvals, cmap, norm, lw=2.5, alpha=1.0):
    pts = np.column_stack([x, y])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=lw, alpha=alpha)
    lc.set_array(cvals[:-1])
    ax.add_collection(lc)
    return lc

# =========================================================
# 1. LOAD DATA
# =========================================================
# --- GW Data ---
file_name5_a = "o-output.bands_interpolated"
x_data5_a, y_data5_a = [], [[] for _ in range(8)]
qp_data5_a = []
FLOAT_RE = re.compile(r'[-+]?(?:\d*\.\d+|\d+\.\d*|\d+)(?:[Ee][-+]?\d+)?')

try:
    with open(file_name5_a, 'r') as f:
        for line in f:
            if line.lstrip().startswith('#'): continue
            values = [float(v) for v in FLOAT_RE.findall(line)]
            if len(values) < 8: continue
            x_data5_a.append(values[0])
            for i in range(8): y_data5_a[i].append(values[i + 1])
            if len(values) >= 11:  # Capture the reduced qx, qy coords for parity interpolation
                qp_data5_a.append([values[9], values[10]])
except FileNotFoundError:
    print(f"Warning: {file_name5_a} not found.")

x5_a = np.array(x_data5_a)
y5_a = [np.array(c) for c in y_data5_a]
qp5_a = np.array(qp_data5_a)

# --- GW Parity Processing ---
# band_parity.npz (xphd band-parity) holds the sigma_h parity of the bands in
# the BSE window only; its "bands" array gives their 1-based indices (7-12 for
# the GaN run). The columns of the ypp file are a different window, starting
# at GW_FIRST_BAND. The two are matched by BAND NUMBER, never by column
# position -- pairing by position coloured every band with another band's
# parity. GW bands outside the npz window are drawn grey.
GW_FIRST_BAND = 6      # 1-based index of the first band column of the ypp file:
                       # the first number of the BANDS range in your ypp input
PARITY_NPZ = "band_parity.npz"

chi_path = None
if len(qp5_a) == len(x5_a) and len(qp5_a) > 0:
    try:
        from scipy.spatial import cKDTree

        def to_cart(q):
            q = np.asarray(q, float)
            return np.stack([q[:, 0] + 0.5 * q[:, 1], np.sqrt(3.0) / 2 * q[:, 1]], 1)

        def dist_to(q, target):
            # distance in |b| to the nearest lattice-equivalent copy of target
            d = np.asarray(q, float) - np.asarray(target, float)
            d -= np.rint(d)
            return np.linalg.norm(to_cart(d), axis=1)

        d = np.load(PARITY_NPZ)
        Q = np.asarray(d["kf"])[:, :2]
        chi = np.asarray(d["p"])
        bands_npz = np.asarray(d["bands"]).astype(int)
        ok = np.asarray(d["defined"]) if "defined" in d.files else np.ones(chi.shape, bool)

        # The ypp k-columns must be REDUCED coordinates: check that the K tick
        # really sits on K (or K') and the Gamma tick on Gamma
        i_G = int(np.argmin(np.abs(x5_a - 0.597328244)))
        i_K = int(np.argmin(np.abs(x5_a - 1.2870635)))
        dG = dist_to(qp5_a[[i_G]], [0.0, 0.0])[0]
        dK = min(dist_to(qp5_a[[i_K]], [1 / 3, 1 / 3])[0],
                 dist_to(qp5_a[[i_K]], [2 / 3, 2 / 3])[0])
        print(f"ypp k-point at the Gamma tick {qp5_a[i_G]}, at the K tick {qp5_a[i_K]}")
        if dG > 1e-3 or dK > 1e-3:
            raise ValueError("the ypp k-columns are not reduced coordinates on "
                             "M-Gamma-K (Gamma/K ticks miss by "
                             f"{dG:.3f}/{dK:.3f} |b|); check the file header")

        # parity is +-1 and jumps at band crossings: take the NEAREST mesh
        # point (with periodic images), never an interpolated value
        img = np.array([(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)], float)
        Qt = np.concatenate([Q + s for s in img])
        near = cKDTree(to_cart(Qt)).query(to_cart(qp5_a))[1] % len(Q)

        chi_path = np.full((len(x5_a), 8), np.nan)
        print(f"{PARITY_NPZ}: bands {bands_npz.min()}-{bands_npz.max()}; "
              f"ypp bands {GW_FIRST_BAND}-{GW_FIRST_BAND + 7}")
        for i in range(8):
            n = GW_FIRST_BAND + i
            hit = np.where(bands_npz == n)[0]
            if not hit.size:
                print(f"   ypp column {i}: band {n:3d}  -> not in {PARITY_NPZ}, drawn grey")
                continue
            j = int(hit[0])
            chi_path[:, i] = np.where(ok[near, j], chi[near, j], np.nan)
            pG = chi_path[i_G, i]
            print(f"   ypp column {i}: band {n:3d}  -> parity at Gamma "
                  f"{'even' if pG > 0 else 'odd' if pG < 0 else 'undefined'}")
    except Exception as e:
        chi_path = None
        print(f"Parity overlay skipped: {e}")

# --- Phonon Data ---
def read_ph_dispersion(filename, nbranches=6):
    x_data, y_data = [], [[] for _ in range(nbranches)]
    try:
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('#'): continue
                vals = line.split()
                if len(vals) < (1 + nbranches): continue
                x_data.append(float(vals[0]))
                for i in range(nbranches): y_data[i].append(float(vals[1 + i]))
    except Exception: pass
    return np.array(x_data, dtype=float), [np.array(col, dtype=float) for col in y_data]

def read_ph_dos(filename):
    freqs, dos_a, dos_b = [], [], []
    try:
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('#'): continue
                vals = line.split()
                if len(vals) < 4: continue
                freqs.append(float(vals[0]))
                dos_a.append(float(vals[2]))
                dos_b.append(float(vals[3]))
    except Exception: pass
    
    if not freqs: return np.array([]), np.array([])
    freqs, dos_a, dos_b = map(np.array, (freqs, dos_a, dos_b))
    order = np.argsort(freqs)
    freqs, dos_a, dos_b = freqs[order], dos_a[order], dos_b[order]
    total = np.where(np.abs(dos_a + dos_b) < 1e-12, 1e-12, dos_a + dos_b)
    return freqs, np.clip(dos_a / total, 0.0, 1.0)

x_ph, branches_ph = read_ph_dispersion("gan.freq.gp", nbranches=6)
frequencies_ph, normalized_dos_a = read_ph_dos("gan.dos")

# --- Absorptance Data ---
try:
    data_24 = np.loadtxt('o-output.alpha_q1_diago_bse')
    energy_24 = data_24[:, 0]
    alpha_24 = data_24[:, 1]
    absorptance_24 = (1 - np.exp(-4 * np.pi * alpha_24 * energy_24 / (27.2114 * 137.036))) * 100
except Exception:
    energy_24, absorptance_24 = np.array([]), np.array([])

try:
    exciton_E, exciton_I = np.loadtxt('o-output.exc_qpt1_E_sorted', unpack=True, usecols=(0, 1))
except Exception:
    exciton_E, exciton_I = np.array([]), np.array([])

# =========================================================
# 2. FIGURE SETUP (GridSpec)
# =========================================================
fig = plt.figure(figsize=(10, 12), dpi=300)

# Main 2-row Grid. Top row = GW + Phonon. Bottom row = Absorptance
gs = fig.add_gridspec(nrows=2, ncols=2, width_ratios=[1, 0.8], height_ratios=[1.2, 1], hspace=0.15, wspace=0.15)

# Assign Subplots
ax_gw  = fig.add_subplot(gs[0, 0])
ax_abs = fig.add_subplot(gs[1, :])

# The Phonon column needs its own sub-grid for the colorbar
gs_ph = gs[0, 1].subgridspec(nrows=2, ncols=1, height_ratios=[1, 0.04], hspace=0.2)
ax_ph = fig.add_subplot(gs_ph[0])
cax   = fig.add_subplot(gs_ph[1])

# =========================================================
# 3. TOP LEFT: GW BANDS
# =========================================================
if len(x5_a) > 0:
    if chi_path is not None:
        cmap_gw = plt.get_cmap("coolwarm")
        norm_gw = mpl.colors.Normalize(vmin=-1, vmax=1)
        for i in range(8):
            if np.isfinite(chi_path[:, i]).any():
                colored_line(ax_gw, x5_a, y5_a[i], chi_path[:, i], cmap=cmap_gw, norm=norm_gw, lw=2.5)
            else:
                ax_gw.plot(x5_a, y5_a[i], color='0.7', linewidth=2)
                
    else:
        for band in y5_a:
            ax_gw.plot(x5_a, band, color='crimson', linewidth=2)

ax_gw.axvline(0.597328244, color='black', linestyle='--', linewidth=1.5)
ax_gw.axvline(1.2870635, color='black', linestyle='--', linewidth=1.5)
ax_gw.axhline(0, color='black', linewidth=1)

ax_gw.set_xlim(0, 1.2870635)
ax_gw.set_ylim(-5, 9)
ax_gw.set_ylabel(r'$\mathrm{E-E_F}$ (eV)', fontsize=20)

# Axis formatting
ax_gw.set_xticks([0, 0.597328244, 1.2870635])
ax_gw.set_xticklabels(['M', r'$\Gamma$', 'K'], fontsize=20)
ax_gw.tick_params(axis='both', which='major', direction='in', length=6, width=1.5, labelsize=18)
for side in ('top', 'right', 'bottom', 'left'): ax_gw.spines[side].set_linewidth(1.5)

# Annotations & Markers
fs = 25
sc_s = 250

ax_gw.scatter(0.597328244, 0.5, s=sc_s, facecolor='lime', edgecolor='black', alpha=1, linewidth=1.5, zorder=20)
ax_gw.scatter(0.56, 4.2, s=sc_s, facecolor='lime', edgecolor='black', alpha=1, linewidth=1.5, zorder=20)
ax_gw.text(0.597328244, 0.4, '+', ha='center', va='center', fontsize=fs, zorder=21)
ax_gw.text(0.56, 4.2, '-', ha='center', va='center', fontsize=fs, zorder=21)

ax_gw.scatter(1.26, 0.5, s=sc_s, facecolor='deepskyblue', edgecolor='black', alpha=1, linewidth=1.5, zorder=20)
ax_gw.scatter(0.635, 4.2, s=sc_s, facecolor='deepskyblue', edgecolor='black', alpha=1, linewidth=1.5, zorder=20)
ax_gw.text(1.26, 0.4, '+', ha='center', va='center', fontsize=fs, zorder=21)
ax_gw.text(0.635, 4.2, '-', ha='center', va='center', fontsize=fs, zorder=21)

# Orbital character of the band edges, in place of a parity colorbar. Each
# label sits next to a band (by BAND NUMBER, not column) at a position x on
# the path, shifted by dy (eV), and is coloured like that band's parity:
# sigma (in-plane s, p_x, p_y) even = red, pi (p_z) odd = blue.
# (text, band number, x on the path, dy in eV, expected parity)
ORBITAL_LABELS = [
    (r'$\pi$',       9, 1.12, -0.75, -1),   # valence-band maximum at K: N p_z
    (r'$\sigma$',    9, 0.42, +0.55, +1),   # top valence at Gamma: N p_x, p_y
    (r'$\sigma^*$', 10, 0.45, +0.60, +1),   # conduction-band minimum at Gamma: Ga s
    (r'$\pi^*$',    10, 1.12, +0.60, -1),   # lowest conduction band near K
]
if len(x5_a) > 0:
    cmap_lab = plt.get_cmap("coolwarm")
    for text, band, xl, dy, expect in ORBITAL_LABELS:
        col = band - GW_FIRST_BAND
        if not 0 <= col < len(y5_a):
            print(f"label {text}: band {band} is not in the ypp file, skipped")
            continue
        il = int(np.argmin(np.abs(x5_a - xl)))
        if chi_path is not None and np.isfinite(chi_path[il, col]) \
                and np.sign(chi_path[il, col]) != expect:
            print(f"WARNING label {text}: band {band} at x = {xl} has parity "
                  f"{chi_path[il, col]:+.2f}, not {expect:+d} -- move the label")
        ax_gw.text(x5_a[il], y5_a[col][il] + dy, text, ha='center', va='center',
                   fontsize=24, color=cmap_lab(0.95 if expect > 0 else 0.05), zorder=22)

ax_gw.text(-0.15, 1, '(a)', transform=ax_gw.transAxes, fontsize=20, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='none', alpha=0.85))

# =========================================================
# 4. TOP RIGHT: PHONON DISPERSION
# =========================================================
ax_ph.yaxis.tick_right()
ax_ph.yaxis.set_label_position('right')

if len(x_ph) > 0:
    cmap = plt.get_cmap('cool')
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)

    for i in range(6): 
        ax_ph.plot(x_ph, branches_ph[i], color='black', linewidth=1.5, alpha=0.35)

    x_dense = np.linspace(x_ph.min(), x_ph.max(), 2500)
    for i in range(6):
        f = interp1d(x_ph, branches_ph[i], kind='linear', fill_value="extrapolate")
        y_dense = f(x_dense)
        cvals = np.interp(y_dense, frequencies_ph, normalized_dos_a, left=normalized_dos_a[0], right=normalized_dos_a[-1])
        colored_line(ax_ph, x_dense, y_dense, cvals, cmap=cmap, norm=norm, lw=2.5, alpha=1.0)
    
    # Colorbar
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='horizontal', ticks=[0, 1])
    cb.set_ticklabels(['N', 'Ga'])
    cb.ax.tick_params(labelsize=16, width=1.5, length=6)
    for spine in cb.ax.spines.values(): spine.set_linewidth(1.5)

ax_ph.axvline(0.584667, color='black', linestyle='--', linewidth=1.5)
ax_ph.set_xticks([0, 0.584667, 1.259781])
ax_ph.set_xticklabels(['M', r'$\Gamma$', 'K'], fontsize=20)
ax_ph.set_xlim(0, 1.259781)
ax_ph.set_ylim(0, 850)
ax_ph.set_ylabel(r'$\omega$ (cm$^{-1}$)', fontsize=20, labelpad=15)

for spine in ax_ph.spines.values(): spine.set_linewidth(1.5)
ax_ph.spines['right'].set_color('green')
ax_ph.spines['right'].set_linestyle(':')
ax_ph.spines['right'].set_linewidth(3)
ax_ph.tick_params(axis='both', which='major', labelsize=18, width=1.5, length=6, direction='in')

# Phonon Mode Images
add_image(ax_ph, 'Mode1.png', (0.25, 500), zoom=0.25, crop=True)
add_image(ax_ph, 'Mode2.png', (0.95, 500), zoom=0.25, crop=True)

ax_ph.text(-0.15, 1, '(b)', transform=ax_ph.transAxes, fontsize=20, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='none', alpha=0.85))

# =========================================================
# 5. BOTTOM: MACROSCOPIC ABSORPTANCE
# =========================================================
if len(energy_24) > 0:
    ax_abs.fill_between(energy_24, absorptance_24, color="mediumpurple", alpha=0.9, zorder=1)
    ax_abs.plot(energy_24, absorptance_24, color='black', linewidth=1, zorder=2)

y_max = np.max(absorptance_24) if len(absorptance_24) > 0 else 1.0

if len(exciton_E) > 0:
    scaling_factor = 25 
    stem_colors = ["#EAD00D"] * len(exciton_E)
    stem_sizes = [15] * len(exciton_E)

    if len(stem_colors) > 0: 
        stem_colors[0] = 'green'
        stem_sizes[0] = 60  
    if len(stem_colors) > 1: 
        stem_colors[1] = 'orange'
        stem_sizes[1] = 60
    if len(stem_colors) > 38: 
        stem_colors[38] = 'lightseagreen'
        stem_sizes[38] = 60 
    if len(stem_colors) > 39: 
        stem_colors[39] = 'lightpink'
        stem_sizes[39] = 60

    ax_abs.vlines(x=exciton_E, ymin=0, ymax=exciton_I * scaling_factor, colors=stem_colors, linewidth=1.5, zorder=3)
    ax_abs.scatter(exciton_E, exciton_I * scaling_factor, c=stem_colors, s=stem_sizes, zorder=4)

    # Inset target positions
    target_1  = (3, y_max * 0.40)
    target_2  = (3.1, y_max * 1.10)
    target_39 = (4.7, y_max * 0.9)
    target_40 = (6.1, y_max * 0.9)

    # Connected Callout Arrows
    arrow_targets = [
        {"idx": 0,  "target": target_1,  "color": 'green',         "rad": -0.10},
        {"idx": 1,  "target": target_2,  "color": 'orange',        "rad": 0.10},
        {"idx": 38, "target": target_39, "color": 'lightseagreen', "rad": -0.08},
        {"idx": 39, "target": target_40, "color": 'lightpink',     "rad": 0.08},
    ]

    for item in arrow_targets:
        idx = item["idx"]
        if len(exciton_E) > idx:
            peak_x = exciton_E[idx]
            peak_y = exciton_I[idx] * scaling_factor
            ax_abs.annotate(
                '',
                xy=item["target"],
                xytext=(peak_x, peak_y),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=item["color"],
                    lw=2.0,
                    mutation_scale=16,
                    shrinkA=5,
                    shrinkB=38,
                    connectionstyle=f"arc3,rad={item['rad']}"
                ),
                zorder=5
            )

# Absorptance K-space Images
add_image(ax_abs, 'Kspace_state_1_jz_1.00.png', (2.8, y_max * 0.40), 0.14)
add_image(ax_abs, 'Kspace_state_2_jz_-1.00.png', (2.8, y_max * 1.10), 0.14)
add_image(ax_abs, 'Kspace_state_39_jz_-1.00.png', (4.7, y_max * 1.10), 0.14)
add_image(ax_abs, 'Kspace_state_40_jz_1.00.png', (6.1, y_max * 1.10), 0.14)

# Circular Polarization Labels
y_offset = 0.35 * y_max
ax_abs.text(4.7, (y_max * 1.02) - y_offset, r'$\sigma^-$', fontsize=20, ha='center', va='center')
ax_abs.text(6.1, (y_max * 1.02) - y_offset, r'$\sigma^+$', fontsize=20, ha='center', va='center')

# Axis formatting
ax_abs.set_xlim(2.0, 7.0)
ax_abs.set_ylim(0, y_max * 1.5)
ax_abs.set_xlabel('Photon energy (eV)', fontsize=20, labelpad=8)
ax_abs.set_ylabel('Absorptance (%)', fontsize=20, labelpad=8)

for side in ('top', 'right', 'bottom', 'left'): ax_abs.spines[side].set_linewidth(1.5)
ax_abs.tick_params(axis='y', direction='in', length=6, width=1.5, labelsize=18)
ax_abs.tick_params(axis='x', direction='out', length=6, width=1.5, labelsize=18)
ax_abs.text(-0.1, 1, '(c)', transform=ax_abs.transAxes, fontsize=20, fontweight='bold', va='top')

# Final tweaks & Save
plt.savefig('Fig_2.pdf', dpi=600, bbox_inches='tight')