"""
plot_phonon_parity.py
=====================
Phonon dispersion along a high-symmetry path, every branch coloured by its
sigma_h parity -- the phonon counterpart of plot_parity_dispersion.py.

Run it by editing the SETTINGS block below and running the file -- in VS
Code, Run Python File. No command line is needed.

How the parity is found
-----------------------
sigma_h maps atom kappa onto atom PERM[kappa] and flips z. The parity of a mode
is  chi = sum_kappa e*_{PERM[kappa]} . diag(1, 1, -1) . e_kappa  = +-1
(xphd.mirror.phonon_parity). In a planar layer (PERM = None: every atom on the
mirror plane) this is chi = 1 - 2w with w = sum |e_z|^2 / sum |e|^2, the
out-of-plane weight. In 1H-WSe2 or MoS2 the mirror swaps the two chalcogens:
set PERM = [0, 2, 1] -- the out-of-plane weight would call the A1' breathing
mode odd and the E'' mode even, both wrong. Colour: +1 even (red), -1 odd (blue).

Where the path comes from -- set MODES, and the script works out which
-------------------------------------------------------------------------
1. MODES is a FULL n x n mesh (the matdyn.modes behind hw_fine.npy): the
   points of that mesh lying exactly on MESH_PATH are taken, in order. On a
   360 x 360 mesh that is 361 points for G-M-K-G, 1/360 of a reciprocal
   vector apart -- smooth, with exact parities, and with the very
   frequencies the linewidths were computed from. Recommended.
2. MODES is the eigenvector file (flvec) of a matdyn run ALONG the path,
   q_in_band_form = .true.: used as it is. That run must share the mesh
   run's force constants and settings -- above all loto_2d = .true. In a
   2D layer the two in-plane optical modes of a two-atom cell meet at Gamma;
   if they do not, the 3D non-analytic form was used and the script says so.
3. MODES is a full mesh and PATH_FREQ is a path .freq without eigenvectors:
   each path point takes the parity of its nearest mesh point. Near a
   crossing between an even and an odd branch that can give a point the
   wrong parity; prefer route 1.

Joining the points
------------------
With CONNECT = "parity", points are joined within each parity separately:
at every q the odd modes are sorted by energy and joined to the odd modes of
the next q, and likewise the even ones. Each line then keeps one colour, and
an even and an odd branch visibly cross -- which they must, because modes of
opposite parity cannot mix. This needs every mode to be clearly even or odd
and the number of each to stay fixed along the path, which holds in a planar
layer (a two-atom cell has two odd branches, ZA and ZO, and four even ones).
If it does not hold, the script says so and falls back to energy order.

With CONNECT = "energy", lines follow energy order, as matdyn writes them,
and a colour change along a line marks a crossing between branches of
opposite parity.

The frequencies and parities at each high-symmetry point are printed as a
table, in the order they would go into a paper's table of modes.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from xphd.band_parity import path_points
from xphd.io.matdyn import read_freq
from xphd.modes import read_modes

# ==========================================================================
# 0. SETTINGS -- edit these, then run this file (VS Code: Run Python File)
# ==========================================================================
# matdyn eigenvectors: a full n x n mesh, or a run along the path
MODES = "matdyn.modes"

# how sigma_h permutes the atoms: None for a planar layer (GaN, hBN: every atom
# lies in the mirror plane and maps onto itself); [0, 2, 1] for WSe2 or MoS2,
# where the mirror swaps the two chalcogens (order as in the cell)
PERM = None

# when MODES is a mesh: the high-symmetry points to walk through, by name
# (G, M, K), and optionally a path .freq to take the frequencies from instead
MESH_PATH = ["G", "M", "K", "G"]
PATH_FREQ = None

# one label per high-symmetry point, in order
TICK_LABELS = [r"$\Gamma$", "M", "K", r"$\Gamma$"]

# "meV", "cm-1" or "THz"
UNITS = "meV"

# "parity": join points within each parity; "energy": follow energy order
CONNECT = "parity"

# a mode counts as clearly even or odd when w is within this of 0 or 1
PARITY_TOL = 0.05

EMIN = None
EMAX = None
LW = 2.6
CMAP = "coolwarm"
FIGSIZE = [5.0, 4.6]
TITLE = r"Phonon parity ($\sigma_h$)"
DPI = 400
OUT = "phonon_parity_dispersion.pdf"

# also open the figure in a window after saving it; False only saves it
SHOW = True

CM1_TO = {"meV": 0.12398419843320026, "cm-1": 1.0, "THz": 0.0299792458}
_have = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = ("Montserrat" if "Montserrat" in _have
                               else "DejaVu Sans")
plt.rcParams["mathtext.fontset"] = "stix"

# ==========================================================================
# 1. LOAD THE EIGENVECTORS AND DECIDE WHAT THEY ARE
# ==========================================================================
if UNITS not in CM1_TO:
    raise SystemExit(f"UNITS must be one of {list(CM1_TO)}")
if not MODES:
    raise SystemExit("set MODES to a matdyn eigenvector file")


def z_weight(ev):
    """Odd weight per mode, (1 - p)/2, with p the sigma_h parity of the mode
    taken from how the mirror permutes the atoms (xphd.mirror); ev is
    (nq, nmod, nat, 3). 0 = even, 1 = odd. For a planar layer (PERM None)
    this is exactly the out-of-plane weight |e_z|^2 / |e|^2."""
    from xphd.mirror import phonon_parity
    ev = np.asarray(ev)
    perm = np.arange(ev.shape[-2]) if PERM is None else np.asarray(PERM, int)
    if len(perm) != ev.shape[-2]:
        raise SystemExit(f"PERM has {len(perm)} entries, the modes {ev.shape[-2]} atoms")
    return (1.0 - phonon_parity(ev, perm)) / 2.0


def as_mesh(qc, tol=5e-4):
    """(n, B, red) if the q-points are an n x n mesh, else None.

    matdyn prints q to four decimals, so fractions recomputed from single
    printed points would be off by up to n x 1e-4 -- far too much on a
    360 x 360 mesh. Instead each point's EXACT fractions i/n, j/n are taken
    from its position in the list, the two reciprocal vectors are fitted to
    every point by least squares, and the file counts as a mesh only if every
    point then lies within printing precision of its place. Row- and
    column-major orders are tried, with fractions in [0, 1) or [-1/2, 1/2).
    A path can have a square number of points by chance -- 361 does -- but it
    cannot pass this test.
    """
    qc = np.asarray(qc, float)[:, :2]
    n = int(round(np.sqrt(len(qc))))
    if n < 2 or n * n != len(qc):
        return None
    i, j = np.divmod(np.arange(n * n), n)
    best = None
    for u, v in ((i, j), (j, i)):
        for wrap in (False, True):
            r = np.stack([u / n, v / n], 1)
            if wrap:
                r = (r + 0.5) % 1.0 - 0.5
            B, *_ = np.linalg.lstsq(r, qc, rcond=None)
            res = float(np.abs(r @ B - qc).max())
            if best is None or res < best[0]:
                best = (res, B, r)
    res, B, r = best
    if res > tol or abs(np.linalg.det(B)) < 1e-10:
        return None
    return n, B, r


def hs_points(B):
    """Gamma, M and K in the mesh's own reduced basis, checked for a
    hexagonal lattice: |M| = |b|/2 and |K| = |b|/sqrt3, adjacent."""
    b = np.linalg.norm(B[0])
    ang = np.degrees(np.arccos(B[0] @ B[1] / b / np.linalg.norm(B[1])))
    if abs(ang - 60) < 1:
        pts = {"G": (0.0, 0.0), "M": (0.5, 0.0), "K": (1 / 3, 1 / 3)}
    elif abs(ang - 120) < 1:
        pts = {"G": (0.0, 0.0), "M": (0.5, 0.0), "K": (2 / 3, 1 / 3)}
    else:
        raise SystemExit(f"reciprocal vectors at {ang:.1f} deg: not a "
                         f"hexagonal lattice, so G, M and K are not defined")
    # a geometry check, not a precision one: B is fitted to q printed to
    # four decimals, so agreement to 1e-3 of |b| is what to expect
    M = np.array(pts["M"]) @ B
    K = np.array(pts["K"]) @ B
    for got, want, what in ((np.linalg.norm(M), b / 2, "|M|"),
                            (np.linalg.norm(K), b / np.sqrt(3), "|K|"),
                            (np.linalg.norm(K - M), b / (2 * np.sqrt(3)), "|K-M|")):
        if abs(got - want) > 1e-3 * b:
            raise SystemExit(f"{what} = {got:.5f}, expected {want:.5f} for a "
                             f"hexagonal lattice -- G, M, K are not where "
                             f"MESH_PATH assumes")
    return pts


print(f"Loading {MODES} ...")
q, f_cm, ev = read_modes(MODES, verbose=False)
q = np.asarray(q, float)
mesh = as_mesh(q)
path_x = path_ticks = None

if mesh is not None and PATH_FREQ is None:
    # ---- route 1: the mesh's own points on the path
    n, B, red = mesh
    pts = hs_points(B)
    bad = [p for p in MESH_PATH if p not in pts]
    if bad:
        raise SystemExit(f"MESH_PATH names {bad}; use G, M and K")
    rlat = np.array([[*B[0], 0.0], [*B[1], 0.0], [0.0, 0.0, 1.0]])
    kf = np.column_stack([red, np.zeros(len(red))])
    idx, path_x, path_ticks = path_points(kf, rlat,
                                          [(p, pts[p]) for p in MESH_PATH])
    q, f_cm, ev = q[idx], np.asarray(f_cm)[idx], np.asarray(ev)[idx]
    w = z_weight(ev)
    source = (f"the {n}x{n} mesh -- its {len(idx)} points lying exactly on "
              f"{'-'.join(MESH_PATH)}")
elif mesh is not None:
    # ---- route 3: frequencies from a path .freq, parity from the mesh
    n, B, red_m = mesh
    print(f"Loading {PATH_FREQ} ...")
    qp, f_ev = read_freq(PATH_FREQ, verbose=False)
    qp = np.asarray(qp, float)
    fm_cm = np.asarray(f_cm)
    f_cm = np.asarray(f_ev) / CM1_TO["meV"] * 1e3
    red = qp[:, :2] @ np.linalg.inv(B)
    ij = np.rint(red * n).astype(int) % n
    near = ij[:, 0] * n + ij[:, 1]
    w = z_weight(np.asarray(ev))[near]
    q = qp
    # the two runs must agree where a path point lies exactly on the mesh
    on = np.all(np.abs(red * n - np.rint(red * n)) < 1e-3, axis=1)
    if on.any():
        mism = float(np.abs(np.sort(f_cm[on], 1)
                            - np.sort(fm_cm[near[on]], 1)).max()) * CM1_TO["meV"]
        source = (f"nearest points of the {n}x{n} mesh; at the {int(on.sum())} "
                  f"path points on the mesh the frequencies agree to "
                  f"{mism:.3f} meV")
        if mism > 0.1:
            print("   WARNING: where the path meets the mesh the frequencies "
                  "differ by more than 0.1 meV -- are the two runs from the "
                  "same force constants and settings?")
    else:
        source = f"nearest points of the {n}x{n} mesh; no path point lies on it"
    print("   [note] near a crossing between an even and an odd branch the "
          "nearest mesh point can\n          lie on its other side; route 1 "
          "(PATH_FREQ = None) has no such error.")
else:
    # ---- route 2: a matdyn run along the path
    if PATH_FREQ is not None:
        raise SystemExit("PATH_FREQ needs MODES to be a full mesh; MODES here "
                         "is already a path, so set PATH_FREQ = None")
    w = z_weight(np.asarray(ev))
    source = "eigenvectors of a run along the path (exact at every point)"

f = np.asarray(f_cm, float) * CM1_TO[UNITS]
nq, nmod = f.shape
print(f"   {nq} path points, {nmod} branches; parity from {source}")

# ==========================================================================
# 2. PARITY OF EVERY MODE
# ==========================================================================
chi = 1.0 - 2.0 * w
dev = float(np.minimum(w, 1.0 - w).max())
n_odd = (chi < 0).sum(axis=1)
planar = dev <= PARITY_TOL and n_odd.min() == n_odd.max()
print(f"   largest distance of any mode from pure even/odd: {dev:.1e}")
print(f"   odd branches per q: {n_odd.min()}..{n_odd.max()}"
      + ("  -- planar: every mode is even or odd" if planar else ""))

# In a two-atom planar cell the two in-plane optical modes form E' at Gamma
# and must be degenerate there in a 2D treatment. The 3D non-analytic form
# splits them -- the usual sign of a matdyn run without loto_2d = .true.
if nmod == 6:
    for k in np.where(np.linalg.norm(q, axis=1) < 1e-8)[0]:
        opt = np.sort(f[k][(chi[k] > 0) & (f[k] * (1 / CM1_TO[UNITS])
                                          * CM1_TO["meV"] > 1.0)])
        if len(opt) == 2:
            gap = abs(opt[1] - opt[0]) / CM1_TO[UNITS] * CM1_TO["meV"]
            if gap > 0.5:
                print(f"   WARNING: at Gamma the in-plane optical modes differ "
                      f"by {gap:.1f} meV. In a 2D layer they\n            meet "
                      f"there; a split this size is the 3D non-analytic form. "
                      f"Rerun matdyn\n            with loto_2d = .true., as "
                      f"for the mesh.")
                break

# ==========================================================================
# 3. PATH DISTANCE AND HIGH-SYMMETRY POINTS
# ==========================================================================
if path_x is not None:
    # from the mesh: the distance comes with the points, image-aware
    x = np.asarray(path_x, float)
    corners = [int(np.argmin(np.abs(x - t[0]))) for t in path_ticks]
else:
    # matdyn prints q to four decimals, so on a dense path the direction of a
    # single short step wobbles. The turn at each point is measured over
    # several steps instead, and the sharpest turns are taken as the corners
    # -- as many as TICK_LABELS implies. Real corners on G-M-K-G turn by 90
    # degrees or more; a turn under 10 degrees is not accepted as one.
    step = np.linalg.norm(np.diff(q, axis=0), axis=1)
    x = np.concatenate([[0.0], np.cumsum(step)])
    win = max(1, min(4, nq // 20))
    turn = np.zeros(nq)
    for k in range(win, nq - win):
        a, b = q[k] - q[k - win], q[k + win] - q[k]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 1e-10 and nb > 1e-10:
            turn[k] = np.degrees(np.arccos(np.clip(a @ b / na / nb, -1, 1)))
    want = max(len(TICK_LABELS) - 2, 0)
    inner = []
    for k in np.argsort(turn)[::-1]:
        if len(inner) == want or turn[k] < 10.0:
            break
        if all(abs(k - c) > win for c in inner):
            inner.append(int(k))
    corners = [0] + sorted(inner) + [nq - 1]
ticks = [x[k] for k in corners]
labels = list(TICK_LABELS)
if len(labels) != len(ticks):
    print(f"   [note] found {len(ticks)} high-symmetry points but TICK_LABELS "
          f"has {len(labels)}; labelling them by number")
    labels = [str(i + 1) for i in range(len(ticks))]

# ==========================================================================
# 4. FREQUENCIES AND PARITIES AT THE HIGH-SYMMETRY POINTS
# ==========================================================================
print(f"\n   frequencies ({UNITS}) and parity at each high-symmetry point")
for k, lab in zip(corners, labels):
    order = np.argsort(f[k])
    row = "  ".join(f"{f[k, m]:7.2f} {'e' if chi[k, m] > 0 else 'o'}"
                    for m in order)
    print(f"   {lab.replace('$', '').replace(chr(92), ''):>6}:  {row}")
print("   e = even, o = odd under sigma_h")

# ==========================================================================
# 5. PLOT
# ==========================================================================
print("\nPlotting ...")
fig, ax = plt.subplots(figsize=tuple(FIGSIZE))
cmap = plt.get_cmap(CMAP)
norm = Normalize(-1, 1)

mode = CONNECT
if mode == "parity" and not planar:
    print("   CONNECT = 'parity' needs every mode clearly even or odd with a "
          "fixed count;\n   falling back to energy order")
    mode = "energy"

if mode == "parity":
    for sign in (-1, +1):
        sel = np.where(chi * sign > 0, f, np.nan)          # this parity only
        lines = np.sort(sel, axis=1)[:, :int((sel == sel).sum(1)[0])]
        for b in range(lines.shape[1]):
            ax.plot(x, lines[:, b], color=cmap(norm(sign)), lw=LW,
                    solid_capstyle="round", zorder=2)
else:
    for m in range(nmod):
        pts2 = np.stack([x, f[:, m]], 1)[:, None, :]
        seg = np.concatenate([pts2[:-1], pts2[1:]], axis=1)
        lc = LineCollection(seg, cmap=cmap, norm=norm, linewidths=LW,
                            capstyle="round", zorder=2)
        lc.set_array(0.5 * (chi[:-1, m] + chi[1:, m]))
        ax.add_collection(lc)

for v in ticks[1:-1]:
    ax.axvline(v, color="0.6", ls="--", lw=0.9, zorder=0)
ax.set_xticks(ticks)
ax.set_xticklabels(labels, fontsize=15)
ax.set_xlim(x[0], x[-1])
lo = EMIN if EMIN is not None else min(0.0, float(np.nanmin(f)))
hi = EMAX if EMAX is not None else float(np.nanmax(f)) * 1.04
ax.set_ylim(lo, hi)
ax.set_ylabel(f"Frequency ({'cm$^{-1}$' if UNITS == 'cm-1' else UNITS})",
              fontsize=14)
ax.tick_params(direction="in", labelsize=12, right=True)

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
cb = fig.colorbar(sm, ax=ax, pad=0.04, fraction=0.05)
cb.set_ticks([-1, 0, 1])
cb.set_ticklabels(["-1 (Odd)", "0 (Mixed)", "+1 (Even)"])
cb.set_label(r"$\sigma_h$ parity character", fontsize=12)
if TITLE:
    ax.set_title(TITLE, fontsize=14)
fig.tight_layout()
fig.savefig(OUT, dpi=DPI)
print(f"saved {OUT}  (lines joined by {mode})")

# ==========================================================================
# show the figure -- set SHOW = False in SETTINGS to only save it
# ==========================================================================
if SHOW:
    plt.show()
