"""
crossing_radii.py
=================
The two radii that bound the fourth branch's ring in Fig. 2(d):

  - the exciton parity contour: where the lowest sigma_h-even and the lowest
    sigma_h-odd exciton bands cross (the lowest band is even inside, odd
    outside); from parity.npz on the BSE mesh;
  - q_x: where the fourth phonon branch turns from the out-of-plane ZO mode
    (odd) into an in-plane mode (even); from matdyn.modes on the fine mesh.

Run it by editing the SETTINGS block below and running the file -- in VS
Code, Run Python File. No command line is needed.

How each radius is found
------------------------
Exciton contour. The parity of the lowest state at each mesh point brackets
the switch to one mesh step (|b|/24 along Gamma-M, sqrt3|b|/24 along
Gamma-K). Since the contour is the crossing of the lowest even and lowest odd
band, the script then interpolates their energy difference
dE = E_odd - E_even linearly to zero between the two bracketing points. That
needs a state of each parity among the computed ones at both points; near
Gamma the odd states can all lie above the last computed state, which is
harmless unless it happens AT the bracket -- then the midpoint of the bracket
is reported instead, and the script says so.

q_x. Along each line on the fine mesh, the out-of-plane weight
w_z = sum |e_z|^2 / sum |e|^2 of the fourth (energy-ordered) branch is 1 for
ZO and 0 for an in-plane mode; q_x is where it first drops below 1/2,
interpolated between the two fine points around it (|b|/360 apart).

Both are radii from Gamma in units of |b|. The fine-mesh points are picked
by the q-list's row-major order (q = (i/N, j/N)); the script checks that
order against the q-vectors matdyn printed, and stops on a mismatch.
"""
import numpy as np

from xphd.modes import read_modes

# ==========================================================================
# 0. SETTINGS -- edit these, then run this file (VS Code: Run Python File)
# ==========================================================================
# exciton parities on the BSE mesh (compute_parity.py / xphd parity)
PARITY = "parity.npz"

# phonon eigenvectors on the fine mesh (the chunked matdyn run)
MODES = "matdyn.modes"
N_FINE = 360

# which phonon branch (1-based, energy-ordered)
BRANCH = 4

# ==========================================================================
# 1. HELPERS
# ==========================================================================
def to_cart(k):
    k = np.atleast_2d(np.asarray(k, float))
    return np.stack([k[:, 0] + 0.5 * k[:, 1], np.sqrt(3) / 2 * k[:, 1]], 1)


def radius(k):
    return np.linalg.norm(to_cart(k), axis=1)


def zero_crossing(r, f):
    """First radius where f changes sign (from its value at the smallest r),
    by linear interpolation; also returns the bracketing radii."""
    s0 = np.sign(f[0])
    for n in range(1, len(f)):
        if np.sign(f[n]) != s0 and f[n] != 0:
            r0, r1, f0, f1 = r[n - 1], r[n], f[n - 1], f[n]
            return r0 + (r1 - r0) * f0 / (f0 - f1), r0, r1
    return np.nan, np.nan, np.nan


LINES = {"Gamma-K": (np.array([1, 1]), 1 / 3),      # direction, end (reduced)
         "Gamma-M": (np.array([1, 0]), 1 / 2)}

# ==========================================================================
# 2. EXCITON CONTOUR
# ==========================================================================
d = np.load(PARITY)
Q = np.mod(np.asarray(d["Q_red"], float)[:, :2] + 1e-9, 1.0) - 1e-9
chi, E = np.asarray(d["chi"], float), np.asarray(d["E"], float)
n_mesh = int(round(np.sqrt(len(Q))))
print(f"{PARITY}: {len(Q)} Q-points ({n_mesh}x{n_mesh}), {chi.shape[1]} states")

contour = {}
for name, (u, end) in LINES.items():
    steps = int(round(end * n_mesh))
    r, low, dE = [], [], []
    for s in range(steps + 1):
        target = u * s / n_mesh
        m = int(np.argmin(np.linalg.norm(Q - target, axis=1)))
        if np.linalg.norm(Q[m] - target) > 1e-6:
            raise SystemExit(f"{name}: mesh point {target} missing from {PARITY}")
        # the parity of the LOWEST state is always defined, and brackets the
        # contour; the even-odd energy difference needs a state of each
        # parity among those computed, which near Gamma may not hold
        low.append(np.sign(chi[m][int(np.argmin(E[m]))]))
        even = E[m][chi[m] > 0.5]
        odd = E[m][chi[m] < -0.5]
        dE.append(odd.min() - even.min() if len(even) and len(odd) else np.nan)
        r.append(radius(target)[0])
    r, low, dE = np.array(r), np.array(low), np.array(dE)
    flip = np.where(low != low[0])[0]
    first = "even" if low[0] > 0 else "odd"
    if not flip.size:
        contour[name] = np.nan
        print(f"   {name}: lowest band {first} all the way to the zone boundary -- "
              f"no contour on this line")
        continue
    n = int(flip[0])
    r0, r1 = r[n - 1], r[n]
    if np.isfinite(dE[n - 1]) and np.isfinite(dE[n]):
        rc = r0 + (r1 - r0) * dE[n - 1] / (dE[n - 1] - dE[n])
        how = "interpolated from the even-odd energy difference"
    else:
        rc = 0.5 * (r0 + r1)
        how = ("MIDPOINT of the bracket: one end has no state of the other parity "
               "among those computed, so the crossing cannot be interpolated")
    contour[name] = rc
    print(f"   {name}: lowest band {first} at Gamma; switches between "
          f"{r0:.3f} and {r1:.3f} |b|  ->  contour at {rc:.3f} |b| ({how})")

# ==========================================================================
# 3. q_x FROM THE FINE MESH
# ==========================================================================
q_cart, freq, evec = read_modes(MODES)
if len(q_cart) != N_FINE * N_FINE:
    raise SystemExit(f"{MODES} has {len(q_cart)} q-points, not {N_FINE}^2")
nu = BRANCH - 1
qx = {}
for name, (u, end) in LINES.items():
    steps = int(round(end * N_FINE))
    idx = [int(s * u[0]) * N_FINE + int(s * u[1]) for s in range(steps + 1)]
    r = radius(np.array([u * s / N_FINE for s in range(steps + 1)]))
    # the order check: matdyn's printed |q| must be proportional to r. matdyn
    # prints q to four decimals, so near Gamma the relative rounding is large;
    # compare in absolute terms instead -- a misordered list is off by a whole
    # fine-mesh step, some 40 times the rounding
    qp = np.linalg.norm(np.asarray(q_cart)[idx, :2], axis=1)
    scale = qp[-1] / r[-1]                       # from the farthest point
    miss = np.abs(qp - scale * r)
    step = scale * r[1]                          # one fine-mesh step, printed units
    if miss.max() > 0.25 * step:
        bad = int(np.argmax(miss))
        raise SystemExit(f"{name}: matdyn's q-vectors do not follow the row-major "
                         f"q-list along this line (point {bad}: |q| printed "
                         f"{qp[bad]:.4f}, expected {scale * r[bad]:.4f}; one step is "
                         f"{step:.4f}) -- check the q-list order")
    e = np.asarray(evec)[idx, nu]                      # (npts, nat, 3)
    wz = (np.abs(e[..., 2]) ** 2).sum(1) / (np.abs(e) ** 2).sum((1, 2))
    rq, r0, r1 = zero_crossing(r[1:], wz[1:] - 0.5)    # skip Gamma itself
    qx[name] = rq
    print(f"   {name}: branch {BRANCH} w_z = {wz[1]:.2f} near Gamma; "
          f"drops through 1/2 between {r0:.4f} and {r1:.4f} |b|  ->  "
          f"q_x = {rq:.3f} |b| ({freq[idx[int(np.argmin(np.abs(r - rq)))], nu]:.0f} cm-1)")

# ==========================================================================
# 4. THE ANNULUS
# ==========================================================================
print("\nthe fourth branch couples only between the two radii:")
for name in LINES:
    a, b = sorted([contour[name], qx[name]])
    print(f"   {name}: {a:.3f} - {b:.3f} |b|  (width {b - a:.3f} |b|)")
