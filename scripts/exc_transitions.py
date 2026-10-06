"""
exc_transitions.py
==================
Which band transitions, and which valleys, build each of the lowest excitons.

For WSe2 at Q = 0 the four lowest states should be the bright A doublet
(62 -> 64, the spin-allowed transition, sigma_h-even) and the spin-forbidden
dark/grey pair (62 -> 63, sigma_h-odd), with the dark pair BELOW the bright
one. This shows the weight of each (valence, conduction) pair,
sum_k |A_{s,k,v,c}|^2, and how much of it sits near K, near K' and near
Gamma, so the energy ordering can be checked against the transitions.

Run it by editing the SETTINGS block below and running the file.
"""
import numpy as np

from xphd.yambo import full_zone_kpoints, lattice, load_excitons

# ==========================================================================
# 0. SETTINGS -- edit these, then run this file (VS Code: Run Python File)
# ==========================================================================
SAVE = "LELPH/SAVE"
BSE_DB = "SLEPC/output/ndb.BS_diago_Q1"
NSTATES = 6                    # lowest states to decompose
TOP = 3                        # (v, c) pairs listed per state
VALLEY_RADIUS = 0.14           # |b|: a k-point belongs to a valley within this
# valleys of the ELECTRON (yambo's table k; the hole sits at k + Q), each a
# list of centres in reduced coordinates. Lambda: the six midpoints of
# Gamma-K, where WSe2's second conduction valley lies. A K-Lambda exciton at
# Q = M shows its electron weight in Lambda.
CENTERS = {"Gamma": [[0, 0]], "K": [[1 / 3, 1 / 3]], "K'": [[2 / 3, 2 / 3]],
           "Lambda": [[1 / 6, 1 / 6], [-1 / 6, 1 / 3], [-1 / 3, 1 / 6],
                      [-1 / 6, -1 / 6], [1 / 6, -1 / 3], [1 / 3, -1 / 6]]}

# ==========================================================================
# 1. LOAD
# ==========================================================================
ylat = lattice(SAVE)
exc, A, E = load_excitons(BSE_DB, ylat, NSTATES)
tab = exc.table.astype(int)
v0, c0 = int(tab[:, 1].min()), int(tab[:, 2].min())
kf = full_zone_kpoints(ylat)
print(f"{BSE_DB}: A {A.shape} (states, k, v, c); valence bands "
      f"{v0}-{v0 + A.shape[2] - 1}, conduction {c0}-{c0 + A.shape[3] - 1}")


def cart(k):
    return np.stack([k[:, 0] + 0.5 * k[:, 1], np.sqrt(3) / 2 * k[:, 1]], 1)


def near(target):
    d = kf[:, :2] - np.array(target)
    d -= np.rint(d)
    return np.linalg.norm(cart(d), axis=1) < VALLEY_RADIUS


valleys = {name: np.any([near(c) for c in pts], axis=0) for name, pts in CENTERS.items()}

# ==========================================================================
# 2. DECOMPOSE
# ==========================================================================
for s in range(min(NSTATES, len(E))):
    w = np.abs(A[s]) ** 2                               # (k, v, c)
    tot = w.sum()
    pair = w.sum(axis=0) / tot                          # (v, c)
    order = np.dstack(np.unravel_index(np.argsort(pair, axis=None)[::-1],
                                       pair.shape))[0][:TOP]
    val = "  ".join(f"{name} {100 * w[mask].sum() / tot:5.1f}%"
                    for name, mask in valleys.items())
    tr = "  ".join(f"{v0 + v}->{c0 + c} {100 * pair[v, c]:5.1f}%" for v, c in order)
    print(f"   state {s + 1:2d}  E = {E[s]:.4f} eV   {tr}   |   {val}")
