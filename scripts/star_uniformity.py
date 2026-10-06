"""
star_uniformity.py
==================
The star-uniformity check of the SI (Sec. C): T(q) = sum_{nu, lambda, beta}
|G_{beta lambda nu}(Q, q)|^2, summed over all branches and over complete
manifolds of initial and final states, must be identical for every member of a
star; the script reports T_max / T_min within each star and the worst star.

Run it by editing the SETTINGS block below and running the file.

Stars
-----
    SYM = "full"   all images of q under C3, time reversal (q -> -q) and the
                   vertical mirror (q1, q2) -> (q2, q1): twelve for a generic q.
    SYM = "c3tr"   C3 and time reversal only (six images), the two operations
                   `xphd check-archive` tests manifold by manifold.
The q-points are matched as integer mesh indices (xphd.core.mesh), as in
`check-archive`: archives store q in single precision, so matching rounded
floats fails on large meshes.

Complete manifolds
------------------
T(q) must sum over COMPLETE degenerate manifolds. If the last final state kept
is one member of a degenerate pair at some Q + q, rotations mix it with the
missing partner and T is not invariant there. N_FINAL stops the final-state sum
below the top of the archive -- choose it at a gap in the spectrum -- and
INITIAL picks the initial states (e.g. the bright doublet, [1, 2]).
Only time-reversal-invariant Q (Gamma, M) keep the full star symmetry.
"""
import numpy as np

from xphd.core.mesh import mesh_indices
from xphd.io.excph import ExcPhArchive

# ==========================================================================
# 0. SETTINGS -- edit these, then run this file (VS Code: Run Python File)
# ==========================================================================
ARCHIVE = "GI_ExcPh_Q0001.npz"         # an archive at a time-reversal-invariant Q (Gamma)
SYM = "full"                           # "full" (C3, TR, mirror) or "c3tr"
INITIAL = None                         # 1-based initial states summed, e.g. [1, 2]; None = all
N_FINAL = None                         # final states 1..N_FINAL summed; None = all in the archive
N_WORST = 5                            # how many of the worst stars to list


# ==========================================================================
# 1. T(q)
# ==========================================================================
arc = ExcPhArchive(ARCHIVE)
d = np.load(ARCHIVE, allow_pickle=True)
g2 = None
for k in ("g2_grid", "g2"):
    if k in d.files:
        g2 = np.asarray(d[k], float)
        break
if g2 is None:
    for k in ("G_grid", "G"):
        if k in d.files:
            g2 = np.abs(np.asarray(d[k])) ** 2
            break
if g2 is None:
    raise SystemExit(f"{ARCHIVE}: neither g2_grid nor G_grid")
nq, nmod, ni, nf = g2.shape
ini = list(range(ni)) if INITIAL is None else [s - 1 for s in INITIAL]
fin = nf if N_FINAL is None else int(N_FINAL)
T = g2[:, :, ini, :fin].reshape(nq, -1).sum(axis=1)
n1, n2 = arc.n1, arc.n2
q = np.asarray(arc.q_red, float)[:, :2]
I, J = mesh_indices(q, n1, n2)
print(f"{ARCHIVE}: {nq} q-points on a {n1}x{n2} mesh, {nmod} branches, {ni} x {nf} states")
print(f"  T(q) summed over initial states {[s + 1 for s in ini]} and final states 1-{fin}")
Q = np.asarray(d["Q_red"], float).ravel()[:2] if "Q_red" in d.files else np.zeros(2)
if not np.allclose((2 * Q + 0.5) % 1 - 0.5, 0, atol=1e-5):
    print(f"  NOTE: Q = {Q.round(4)} is not time-reversal invariant; its little group is "
          f"smaller and the full stars need not be uniform.")
if n1 != n2:
    raise SystemExit(f"a {n1}x{n2} mesh is not mapped onto itself by C3")

# ==========================================================================
# 2. STARS, ON INTEGER MESH INDICES
# ==========================================================================
row = {(int(i), int(j)): r for r, (i, j) in enumerate(zip(I, J))}
C3_60 = np.array([[-1, -1], [1, 0]])
C3_120 = np.array([[0, -1], [1, -1]])
MIRROR = np.array([[0, 1], [1, 0]])


def ops(c3):
    g = [np.eye(2, dtype=int), c3, c3 @ c3]
    g += [-m for m in g]                                   # time reversal
    if SYM == "full":
        g += [MIRROR @ m for m in g]
    return g


best = None
for name, c3 in (("60-degree basis", C3_60), ("120-degree basis", C3_120)):
    G = ops(c3)
    images = []
    for r in range(nq):
        v = np.array([I[r], J[r]])
        im = [row.get(tuple(int(x) for x in np.mod(m @ v, n1))) for m in G]
        images.append(sorted(set(im)))
    if any(None in im for im in images):
        continue
    seen, stars = set(), []
    for r in range(nq):
        if r in seen:
            continue
        seen.update(images[r])
        stars.append(images[r])
    ratio = np.array([T[s].max() / T[s].min() if T[s].min() > 0 else np.inf for s in stars])
    score = np.nanmax(ratio[np.isfinite(ratio)]) if np.any(np.isfinite(ratio)) else np.inf
    if best is None or score < best[0]:
        best = (score, name, stars, ratio)
if best is None:
    raise SystemExit("the mesh points are not closed under C3 in either basis convention")
score, name, stars, ratio = best

# ==========================================================================
# 3. REPORT
# ==========================================================================
nontriv = [k for k, s in enumerate(stars) if len(s) > 1]
r = ratio[nontriv]
print(f"\n  C3 in the {name}; symmetry: {'C3, time reversal, mirror' if SYM == 'full' else 'C3, time reversal'}")
print(f"  {len(stars)} stars ({len(nontriv)} with more than one member)")
print(f"  T_max / T_min over the stars:  worst {np.max(r):.6f}   median {np.median(r):.6f}")
order = np.argsort(r)[::-1][:N_WORST]
print(f"\n  worst {len(order)} stars:")
for k in order:
    s = stars[nontriv[k]]
    rep = q[s[0]]
    print(f"    q = ({rep[0]:+.4f}, {rep[1]:+.4f})  members {len(s):2d}   "
          f"T_max / T_min = {r[k]:.6f}   T = {T[s].mean():.3e}")
print("\n  1 is exact symmetry. A few bad stars at large T with all states summed usually mean")
print("  a degenerate pair cut at the top of the archive: lower N_FINAL to a gap and rerun.")
