#!/usr/bin/env python3
"""
combined_figure.py
==================
Plots Exciton Fatbands and overlays the PL spectra directly inside the 
band structure plot, shifted leftwards into the negative space.
Uses a Piecewise Functional Y-Axis to stretch the bottom phonon-assisted region.
Includes an inset zoom of the phonon energy at the bottom left.
Adds a 5x1 column of Brillouin Zone maps to the right of the colorbar.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from scipy.interpolate import griddata
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings

# ======================================================================
# 0.  FILES AND KNOBS
# ======================================================================
T_LW = 77          
T_PL = 10          

LW_NPZ    = 'lw_77K_sym.npz'
DISP_FILE = 'o-output.excitons_interpolated_01'
PL_TOT    = f'{T_PL}K.o-output.pl_bse_ph_ass_dbgd'
EPS_TOT   = f'{T_PL}K.o-output.eps_bse_ph_ass_dbgd'
PL_MODE   = '{i}.o-output.pl_bse_ph_ass_dbgd'
EPS_MODE  = '{i}.o-output.eps_bse_ph_ass_dbgd'
OUT       = 'Fig_3.png'

N_BANDS, N_MODES = 5, 6
PL_SCALE  = 1e7      
PL_SPLIT  = 3.5      
PL_EMAX   = 4.0      
PL_EMIN   = 3.20
I_MAX     = 150.0    

E_MIN, E_MAX = 3.25, 5.55        
MIN_SIZE, MAX_SIZE = 14, 300    
LW_PCT = (0, 100)               

# Colour-scale floor, meV. The linewidth minimum at K is a LOWER BOUND: the
# quasi-elastic acoustic channel sits at q* ~ 0.005 |b1|, inside the Gamma
# cell of the source mesh where the constant-coupling scheme assigns zero
# coupling. Rendering it against a zero floor would show the deepest feature
# in the figure as the least converged number in it. The DATA are untouched;
# only the colour mapping is floored, so anything below LW_FLOOR renders at
# the bottom of the scale.
LW_FLOOR = 0.5

# --- COLORMAPS: change these two ----------------------------------------
CMAP_FAT = 'plasma'      # fat bands (panel a)
CMAP_BZ  = None          # BZ maps (b-f); None = the colormaps-package default

SHOW_PHOTONS = True
SHOW_INSET   = True

MODE_COLORS = {1: 'blue', 2: 'orange', 3: 'green',
               4: 'red',  5: 'purple', 6: 'brown'}
SHADE_COLOR, SHADE_ALPHA = 'teal', 0.18

# --- PIECEWISE STRETCH SETTINGS ---
E_HINGE = 3.354886    
STRETCH_FACTOR = 10.0 

def stretch_forward(y):
    y = np.asarray(y)
    return np.where(y < E_HINGE, 
                    y * STRETCH_FACTOR, 
                    (E_HINGE * STRETCH_FACTOR) + (y - E_HINGE))

def stretch_inverse(yt):
    yt = np.asarray(yt)
    hinge_mapped = E_HINGE * STRETCH_FACTOR
    return np.where(yt < hinge_mapped, 
                    yt / STRETCH_FACTOR, 
                    E_HINGE + (yt - hinge_mapped))

# ======================================================================
# 1.  STYLE
# ======================================================================
_have = {f.name for f in font_manager.fontManager.ttflist}
FONT = 'Montserrat' if 'Montserrat' in _have else 'DejaVu Sans'
plt.rcParams.update({
    'font.size': 18, 'font.family': FONT, 'mathtext.fontset': 'stix',
    'axes.linewidth': 2.2,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.major.width': 2.2, 'ytick.major.width': 2.2,
    'xtick.major.size': 7, 'ytick.major.size': 7,
})

def _resolve_cmap(spec, fallback='YlOrBr_r'):
    """Accept a name, a Colormap, or None (the colormaps-package default)."""
    if spec is None:
        try:
            import colormaps as cmaps
            return cmaps.yel15_r
        except Exception:
            return plt.get_cmap(fallback)
    if isinstance(spec, str):
        try:
            import colormaps as cmaps
            if hasattr(cmaps, spec):
                return getattr(cmaps, spec)
        except Exception:
            pass
        return plt.get_cmap(spec)
    return spec

CMAP     = _resolve_cmap(CMAP_BZ)      # kept for anything still referring to it
CMAP_FAT = _resolve_cmap(CMAP_FAT)
CMAP_BZ  = _resolve_cmap(CMAP_BZ)

# ======================================================================
# 2.  LOAD DATA
# ======================================================================
fbz = np.load(LW_NPZ)
full_kx, full_ky = fbz['Qx'], fbz['Qy']
LW = fbz['LW']
if LW.ndim == 2 and LW.shape[0] == N_BANDS:
    LW = LW.T

disp = np.loadtxt(DISP_FILE, comments='#')
q_dist = disp[:, 0]
E_band = disp[:, 1:1 + N_BANDS]
qx_rlu, qy_rlu = disp[:, 1 + N_BANDS], disp[:, 2 + N_BANDS]
kx_path = qx_rlu + 0.5 * qy_rlu
ky_path = (np.sqrt(3.0) / 2.0) * qy_rlu

idx_M, idx_K = 0, len(q_dist) - 1
idx_G = int(np.argmin(qx_rlu ** 2 + qy_rlu ** 2))
x_M, x_G, x_K = q_dist[idx_M], q_dist[idx_G], q_dist[idx_K]
L = x_K - x_M

def load_xy(path, scale=1.0):
    a = np.loadtxt(path)
    return a[:, 0], a[:, 1] * scale

x_pl,  y_pl  = load_xy(PL_TOT,  PL_SCALE)
x_eps, y_eps = load_xy(EPS_TOT, 1.0)

# ======================================================================
# 3.  KEY ENERGIES
# ======================================================================
E_BRIGHT = E_band[idx_G, 0]
E_DARK   = E_band[idx_K, 0]

m = x_pl <= PL_SPLIT
E_IND = x_pl[m][np.argmax(y_pl[m])]
m = x_eps >= PL_SPLIT
E_DIR = x_eps[m][np.argmax(y_eps[m])]
HW = E_DARK - E_IND

# ======================================================================
# 4.  FIGURE SKELETON (DECOUPLED GRIDSPECS)
# ======================================================================
fig = plt.figure(figsize=(17, 12))

gs_main = fig.add_gridspec(1, 2, width_ratios=[1, 0.03], wspace=0.03,
                           left=0.08, right=0.75, bottom=0.08, top=0.95)
ax_main = fig.add_subplot(gs_main[0, 0])
cax = fig.add_subplot(gs_main[0, 1])

gs_bz = fig.add_gridspec(5, 1, hspace=0.15,
                         left=0.795, right=0.98, bottom=0.08, top=0.95)
ax_bz1 = fig.add_subplot(gs_bz[0, 0])
ax_bz2 = fig.add_subplot(gs_bz[1, 0])
ax_bz3 = fig.add_subplot(gs_bz[2, 0])
ax_bz4 = fig.add_subplot(gs_bz[3, 0])
ax_bz5 = fig.add_subplot(gs_bz[4, 0])

for s in ('top', 'right', 'bottom', 'left'):
    ax_main.spines[s].set_linewidth(2.2)

ax_main.set_yscale('function', functions=(stretch_forward, stretch_inverse))
ax_main.set_yticks([3.32, 3.34095, 3.354886,  4.0, 4.5, 5.0, 5.5])

# Panel Label (a) INSIDE the top-left of the main plot
ax_main.text(0.02, 0.98, "(a)", transform=ax_main.transAxes, ha='left', va='top', fontsize=26, fontweight='bold', zorder=100)

# ======================================================================
# 5.  FAT BANDS
# ======================================================================
qxw = full_kx - np.rint(full_kx)
qyw = full_ky - np.rint(full_ky)
qxt, qyt, tile_idx = [], [], []
for dx in (-1, 0, 1):
    for dy in (-1, 0, 1):
        qxt.append(qxw + dx)
        qyt.append(qyw + dy)
        tile_idx.append(np.arange(len(qxw)))
qxt, qyt = np.concatenate(qxt), np.concatenate(qyt)
tile_idx = np.concatenate(tile_idx)
points = np.column_stack((qxt + 0.5 * qyt, (np.sqrt(3.0) / 2.0) * qyt))

vmin = max(np.percentile(LW, LW_PCT[0]), LW_FLOOR)
vmax = np.percentile(LW, LW_PCT[1])
print(f'fat bands: colour scale {vmin:.2f} - {vmax:.1f} meV'
      f'   (data minimum {LW.min():.3f} meV, floored for display)')

for n in range(N_BANDS):
    raw = (LW[:, n] if LW.ndim > 1 else LW)[tile_idx]
    lin = griddata(points, raw, (kx_path, ky_path), method='linear')
    nea = griddata(points, raw, (kx_path, ky_path), method='nearest')
    lw_path = np.clip(np.where(np.isnan(lin), nea, lin), 0, None)

    frac = np.clip((lw_path - vmin) / (vmax - vmin + 1e-8), 0, 1)
    ax_main.plot(q_dist, E_band[:, n], color='black', lw=0.9, alpha=0.5, zorder=3)
    blobs = ax_main.scatter(q_dist, E_band[:, n],
                       s=MIN_SIZE + frac * (MAX_SIZE - MIN_SIZE),
                       c=lw_path, cmap=CMAP_FAT, vmin=vmin, vmax=vmax,
                       edgecolors='none', zorder=4, alpha=0.85)

ax_main.set_xticks([x_M, x_G, x_K])
ax_main.set_xticklabels(['M', r'$\Gamma$', 'K'], fontsize=26, fontweight='bold')
ax_main.set_ylabel('Exciton Energy (eV)', fontsize=24, fontweight='bold', labelpad=10)
ax_main.set_xlim(x_M, x_K)
ax_main.set_ylim(E_MIN, E_MAX)
ax_main.axvline(x_G, ls='--', color='black', lw=1.4, zorder=1)
ax_main.tick_params(labelsize=20)

# ======================================================================
# 6.  PL SPECTRA OVERLAY (Shifted Inside)
# ======================================================================
PL_BASELINE = x_M + 0.32 * L  

def map_intensity_to_x(intensity):
    max_plot_width = 0.30 * L 
    return PL_BASELINE - (intensity / I_MAX) * max_plot_width

def un_stretch_E(E_array):
    return E_IND + (E_array - E_IND) / STRETCH_FACTOR

emax = PL_EMAX if PL_EMAX is not None else np.inf
emin = PL_EMIN if PL_EMIN is not None else -np.inf

mi = (x_pl >= emin) & (x_pl <= PL_SPLIT)                      
md = (x_eps >= PL_SPLIT) & (x_eps <= emax)

x_pl_unstretched = un_stretch_E(x_pl[mi])

ax_main.fill_betweenx(x_pl_unstretched, PL_BASELINE, map_intensity_to_x(y_pl[mi]), color='gray', alpha=0.1, zorder=8)
ax_main.fill_betweenx(x_eps[md], PL_BASELINE, map_intensity_to_x(y_eps[md]), color='gray', alpha=0.1, zorder=8)

ax_main.plot(map_intensity_to_x(y_pl[mi]), x_pl_unstretched, color='k', lw=2.6, zorder=10)
ax_main.plot(map_intensity_to_x(y_eps[md]), x_eps[md], color='k', lw=2.6, zorder=10)

for i in range(1, N_MODES + 1):
    try:
        xi, yi = load_xy(PL_MODE.format(i=i), PL_SCALE)
        k = (xi >= emin) & (xi <= PL_SPLIT)
        ax_main.plot(map_intensity_to_x(yi[k]), un_stretch_E(xi[k]), color=MODE_COLORS[i], lw=2.4, zorder=9)
        
        xe, ye = load_xy(EPS_MODE.format(i=i), 1.0)
        k = (xe >= PL_SPLIT) & (xe <= emax)
        ax_main.plot(map_intensity_to_x(ye[k]), xe[k], color=MODE_COLORS[i], lw=2.4, zorder=9)
    except FileNotFoundError:
        pass

handles = [Line2D([], [], color='k', lw=2.6, label='Total PL')] + \
          [Line2D([], [], color=MODE_COLORS[i], lw=2.4, label=f'Mode {i}') for i in [1, 2]]
ax_main.legend(handles=handles, frameon=False, fontsize=14, loc='upper right', bbox_to_anchor=(0.98, 0.98))

ax_main.text(PL_BASELINE - 0.18, E_DIR + 0.05, 'Direct PL', ha='left', va='bottom', fontsize=17, color='red', fontweight='bold')
ax_main.text(PL_BASELINE - 0.35, E_IND + 0.005, 'Phonon-assisted PL', ha='left', va='bottom', fontsize=17, color='blue', fontweight='bold')

# ======================================================================
# 7.  SCHEMATIC OVERLAY (Arrows and Labels)
# ======================================================================
def fwhm_band_y(ax_in, y0, fwhm, color, alpha=0.18, zorder=0):
    ax_in.axhspan(y0 - 0.5*fwhm, y0 + 0.5*fwhm, color=color, alpha=alpha, zorder=zorder)

ax_main.axhline(E_BRIGHT, ls='--', color='black',     lw=1.6, zorder=2)
ax_main.axhline(E_DARK,   ls='--', color='goldenrod', lw=1.6, zorder=2)

fwhm_band_y(ax_main, y0=(E_IND + E_DARK)/2.0, fwhm=abs(E_DARK - E_IND), color=SHADE_COLOR, alpha=SHADE_ALPHA)

x_Kmark = x_K - 0.008 * L
ax_main.scatter([x_G],     [E_BRIGHT], s=430, color='red',       alpha=0.65, zorder=25)
ax_main.scatter([x_Kmark], [E_DARK],   s=430, color='blue',      alpha=0.65, zorder=25)
ax_main.scatter([x_G],     [E_IND],    s=360, color='goldenrod', alpha=0.80, zorder=25)

ax_main.add_patch(FancyArrowPatch(
    posA=(x_Kmark, E_DARK), posB=(x_G + 0.01 * L, E_IND),
    connectionstyle='arc3,rad=0.08', arrowstyle='-|>',
    mutation_scale=26, lw=2.6, color='blue', zorder=24))

y_Q = 3.26  
ax_main.annotate('', xy=(x_K, y_Q), xytext=(x_G, y_Q),
            arrowprops=dict(arrowstyle='<->', lw=2.2, color='black',
                            mutation_scale=26, shrinkA=0, shrinkB=0), zorder=20)
ax_main.text(0.5 * (x_G + x_K), y_Q + 0.005, r'$Q_{\mathrm{phonon}}$',
        ha='center', va='bottom', fontsize=20, zorder=21)

ax_main.annotate('Bright exciton\n' + r'$\Gamma$-$\Gamma$', xy=(x_G, E_BRIGHT),
            xycoords='data', xytext=(0.4065, 0.38), textcoords='axes fraction',
            color='crimson', ha='left', va='bottom' )
ax_main.annotate('Dark exciton\n' + r'$\Gamma$-K', xy=(x_Kmark, E_DARK),
            xycoords='data', xytext=(0.9, 0.2), textcoords='axes fraction',
            color='crimson', ha='center', va='bottom')
ax_main.annotate('Virtual state', xy=(x_G, E_IND),
            xycoords='data', xytext=(0.48, 0.25), textcoords='axes fraction',
            color='goldenrod', ha='left', va='bottom')

# ---------------- DAMPED PHOTON FUNCTION ----------------
def draw_damped_photon(ax_in, start, end, amp=0.015, wavelength=0.025, decay=10.0, color='blue', tail_frac=0.2):
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)
    
    head_length = 0.04
    line_length = length - head_length
    
    num_points = 500
    t = np.linspace(0, line_length, num_points)
    
    envelope = np.exp(-decay * t)
    
    tail_start_idx = int((1.0 - tail_frac) * num_points)
    mask = np.ones(num_points)
    fade_len = num_points - tail_start_idx
    if fade_len > 0:
        mask[tail_start_idx:] = 1.0 - np.linspace(0, 1, fade_len)**2
    mask = np.clip(mask, 0, 1)
    
    y_local = amp * envelope * mask * np.sin(2 * np.pi * t / wavelength)
    
    x_plot = t * np.cos(angle) - y_local * np.sin(angle) + x1
    y_plot = t * np.sin(angle) + y_local * np.cos(angle) + y1
    
    ax_in.plot(x_plot, y_plot, color=color, lw=2.5, zorder=50, clip_on=False, solid_capstyle='butt')
    
    ax_in.arrow(x_plot[-1], y_plot[-1], 
             head_length * np.cos(angle), head_length * np.sin(angle), 
             head_width=amp*1.8, head_length=head_length, 
             fc=color, ec=color, length_includes_head=True, zorder=50, clip_on=False)

if SHOW_PHOTONS:
    WAVE = 0.0150
    DECAY = 5.0
    
    AMP_RED = 0.0250 
    AMP_BLUE = AMP_RED / STRETCH_FACTOR 
    
    Y_SHIFT_RED = 0.15
    Y_SHIFT_BLUE = Y_SHIFT_RED / STRETCH_FACTOR

    START_X = PL_BASELINE - 0.02 * L
    END_X = START_X - 0.18 

    draw_damped_photon(ax_main, start=(START_X, E_DIR - Y_SHIFT_RED), end=(END_X, E_DIR - Y_SHIFT_RED), 
                       color='red',  wavelength=WAVE, decay=DECAY, amp=AMP_RED, tail_frac=0.25)
                       
    draw_damped_photon(ax_main, start=(START_X, E_IND - Y_SHIFT_BLUE), end=(END_X, E_IND - Y_SHIFT_BLUE), 
                       color='blue', wavelength=WAVE, decay=DECAY, amp=AMP_BLUE, tail_frac=0.25)

# ======================================================================
# 7.5 INSET -- resolve the phonon energy
# ======================================================================
if SHOW_INSET:
    INSET_PAD = (2.0, 1.3)
    z_lo = E_IND - INSET_PAD[0] * abs(HW)
    z_hi = E_DARK + INSET_PAD[1] * abs(HW)

    axi = ax_main.inset_axes([0.039, 0.01, 0.28, 0.18])
    axi.set_facecolor('white')
    
    mi_inset = x_pl <= PL_SPLIT
    axi.plot(y_pl[mi_inset], x_pl[mi_inset], color='k', lw=2.6, zorder=10)
    
    for i in range(1, N_MODES + 1):
        try:
            xi, yi = load_xy(PL_MODE.format(i=i), PL_SCALE)
            k = xi <= PL_SPLIT
            axi.plot(yi[k], xi[k], color=MODE_COLORS[i], lw=2.4, zorder=9)
        except FileNotFoundError:
            pass

    axi.axhline(E_DARK, ls='--', color='goldenrod', lw=1.6)
    axi.axhline(E_IND,  ls='--', color='black',     lw=1.4)
    axi.axhspan(E_IND, E_DARK, color=SHADE_COLOR, alpha=SHADE_ALPHA)
    
    axi.set_xlim(I_MAX, 0)
    axi.set_ylim(z_lo, z_hi)
    axi.set_xticks([])
    
    axi.yaxis.tick_right()
    axi.set_yticks([E_IND, E_DARK])
    
    axi.set_yticklabels([f'{E_IND:.3f}', f'{E_DARK:.3f}'], fontsize=10)
    axi.tick_params(axis='y', length=4, width=1.4)
    
    for s in axi.spines.values():
        s.set_linewidth(1.6)
        
    axi.annotate('', xy=(0.28 * I_MAX, E_DARK), xytext=(0.28 * I_MAX, E_IND),
                 arrowprops=dict(arrowstyle='<->', color='black', lw=1.8,
                                 mutation_scale=16, shrinkA=0, shrinkB=0))
                                 
    axi.text(0.04, 0.94, r'$\hbar\omega$ = ' + f'{1000 * HW:.1f} meV',
             transform=axi.transAxes, ha='left', va='top',
             fontsize=15, fontweight='bold')

    # # Draw a dotted box in the main plot
    # rect_x = PL_BASELINE - 0.30 * L
    # rect_w = 0.30 * L
    # bx = [rect_x, rect_x + rect_w, rect_x + rect_w, rect_x, rect_x]
    # by = [z_lo, z_lo, z_hi, z_hi, z_lo]
    
    # bx_stretched = bx
    # by_stretched = stretch_forward(by)
    
    # ax_main.plot(bx_stretched, by_stretched, color='dimgray', lw=1.5, ls=':', zorder=30)
    # ax_main.plot([rect_x, x_M + 0.15*L], [by_stretched[0], E_MIN + 0.20], color='dimgray', lw=1.0, ls=':', zorder=29)

# ======================================================================
# 10. COLORBAR AND SAVE
# ======================================================================
cbar = fig.colorbar(blobs, cax=cax)
mid = 0.5 * (vmin + vmax)
cbar.set_ticks([vmin, mid, vmax])
cbar.set_ticklabels([f'{vmin:.1f}', f'{mid:.1f}', f'{vmax:.1f}'])
# cbar.set_label(r'Exciton linewidth $\Gamma$ (meV) @ ' + f'{T_LW} K',
#                fontsize=20, rotation=270, labelpad=32, fontweight='bold')
cbar.ax.tick_params(labelsize=16)

# ======================================================================
# 11. BZ PLOTS (ADDED ON THE FAR RIGHT)
# ======================================================================
def plot_new_bz_map(ax_in, band_idx, panel_label, title_text,
                    cmap=None, fontsize=14, floor=None):
    cmap = CMAP_BZ if cmap is None else cmap
    floor = LW_FLOOR if floor is None else floor
    plot_data = LW[:, band_idx] if LW.ndim > 1 else LW
    
    q_red = np.column_stack((full_kx, full_ky))
    w = q_red - np.rint(q_red)
    
    qx_tiled, qy_tiled, data_tiled = [], [], []
    for dx in [-2, -1, 0, 1, 2]:
        for dy in [-2, -1, 0, 1, 2]:
            qx_tiled.extend(w[:, 0] + dx)
            qy_tiled.extend(w[:, 1] + dy)
            data_tiled.extend(plot_data)
            
    qx_tiled = np.array(qx_tiled)
    qy_tiled = np.array(qy_tiled)
    data_tiled = np.array(data_tiled)
    
    kx_cart = qx_tiled + 0.5 * qy_tiled
    ky_cart = (np.sqrt(3.0) / 2.0) * qy_tiled
    
    theta_rot = np.radians(-30.0)
    c_rot, s_rot = np.cos(theta_rot), np.sin(theta_rot)
    kx_rot = kx_cart * c_rot - ky_cart * s_rot
    ky_rot = kx_cart * s_rot + ky_cart * c_rot
    
    grid_res = 400  
    plot_limit = 0.8  
    
    xi = np.linspace(-plot_limit, plot_limit, grid_res)
    yi = np.linspace(-plot_limit, plot_limit, grid_res)
    X, Y = np.meshgrid(xi, yi)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Z = griddata((kx_rot, ky_rot), data_tiled, (X, Y), method='cubic')
        Z_nearest = griddata((kx_rot, ky_rot), data_tiled, (X, Y), method='nearest')
        Z = np.where(np.isnan(Z), Z_nearest, Z)
    
    color_max_bz = np.percentile(plot_data, 100)
    # same floor as the fat bands: a no-op for every band whose minimum is
    # already above it, and it only affects the band containing the K minimum
    color_min_bz = max(np.percentile(plot_data, 0), floor)
    print(f'  {panel_label} band {band_idx+1}: '
          f'{color_min_bz:.2f} - {color_max_bz:.1f} meV'
          f'   (data minimum {plot_data.min():.3f})')
    Z = np.clip(Z, color_min_bz, color_max_bz)
    
    vertices_frac = np.array([
        [ 1/3,  1/3], [-1/3,  2/3], [-2/3,  1/3],
        [-1/3, -1/3], [ 1/3, -2/3], [ 2/3, -1/3]
    ])
    
    vx_cart = vertices_frac[:, 0] + 0.5 * vertices_frac[:, 1]
    vy_cart = (np.sqrt(3.0) / 2.0) * vertices_frac[:, 1]
    vx_rot = vx_cart * c_rot - vy_cart * s_rot
    vy_rot = vx_cart * s_rot + vy_cart * c_rot
    vertices_final = np.column_stack((vx_rot, vy_rot))
    
    levels = np.linspace(color_min_bz, color_max_bz, 300)
    contour = ax_in.contourf(X, Y, Z, levels=levels, cmap=cmap, zorder=1)
    
    hex_patch = patches.Polygon(vertices_final, closed=True, fill=False, edgecolor='black', lw=1.5, zorder=10)
    ax_in.add_patch(hex_patch)
    
    ax_in.set_xlim(-plot_limit, plot_limit)
    ax_in.set_ylim(-plot_limit, plot_limit)
    ax_in.set_aspect('equal')
    ax_in.axis('off')
    
    # Placed panel labels clearly outside the BZ plot to the top-left using clip_on=False
    ax_in.text(-0.1, 0.85, panel_label, transform=ax_in.transAxes, ha='right', va='bottom', fontsize=20, fontweight='bold', clip_on=False)
    
    divider_bz = make_axes_locatable(ax_in)
    mid_val_bz = (color_min_bz + color_max_bz) / 2.0
    cax_bz = divider_bz.append_axes("right", size="5%", pad=0.05)
    cbar_bz = plt.colorbar(contour, cax=cax_bz, ticks=[color_min_bz, mid_val_bz, color_max_bz])
    cbar_bz.set_ticklabels([f"{color_min_bz:.1f}", f"{mid_val_bz:.1f}", f"{color_max_bz:.1f}"])
    cbar_bz.ax.tick_params(labelsize=fontsize)

plot_new_bz_map(ax_bz1, band_idx=0, panel_label="(b)", title_text="n=1")
plot_new_bz_map(ax_bz2, band_idx=1, panel_label="(c)", title_text="n=2")
plot_new_bz_map(ax_bz3, band_idx=2, panel_label="(d)", title_text="n=3")
plot_new_bz_map(ax_bz4, band_idx=3, panel_label="(e)", title_text="n=4")
plot_new_bz_map(ax_bz5, band_idx=4, panel_label="(f)", title_text="n=5")

fig.savefig(OUT, dpi=600, bbox_inches='tight')
print(f'Wrote {OUT}')