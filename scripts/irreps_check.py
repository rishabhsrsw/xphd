"""
irreps_check.py
===============
Why does `xphd irreps` call a manifold's representation non-unitary? For one
point and one group of states, this prints, per symmetry operation:

    trace        the character of the operation on the group
    |MM^+ - 1|   the unitarity defect `xphd irreps` reports (its maximum)
    in group     the smallest weight a rotated state keeps INSIDE the group
    in all       the smallest weight it keeps inside ALL loaded states

and first, the overlap matrix of the group's own BSE eigenvectors.

How to read it
--------------
  overlap != identity   the eigenvectors are not orthonormal (e.g. a BSE
                        solved beyond Tamm-Dancoff, whose resonant parts are
                        not orthonormal): the defect measures that, not a
                        broken symmetry; the characters can still be right.
  in group < 1 but      the rotated states leave the group but stay among
  in all ~ 1            the computed states: the group is not closed (a
                        partner outside --deg-tol, or mixing with a nearby
                        manifold of the same symmetry).
  in all < 1            the rotated states leave the computed space: the
                        D matrices of that operation, or the band window.
  only some operations  the defect is tied to those operations (e.g. the
  fail                  vertical mirrors), not to the states.

Run it by editing the SETTINGS block below and running the file.
"""
import os

import numpy as np

from xphd.excsym import (HIGH_SYMMETRY, degeneracy_groups, group_name,
                         rep_matrix_Q, shift_map, stabilizer)
from xphd.yambo import (full_zone_kpoints, ibz_parents, lattice, load_dmats,
                        load_excitons, spatial_ops)

# ==========================================================================
# 0. SETTINGS -- edit these, then run this file (VS Code: Run Python File)
# ==========================================================================
SAVE = "SAVE"
BSE_DIR = "../../BSE_EXCPH/output"
DMATS = "Dmats.npy"
NV = 3
POINT = "G"                     # G, K, M, ...
NSTATES = 10                    # states loaded (the "all" in the table)
STATES = None                   # 1-based list, e.g. [1, 2]; None = lowest manifold
DEG_TOL = 0.012                 # eV, only used when STATES is None


# ==========================================================================
# 1. THE POINT, ITS DATABASE AND LITTLE GROUP
# ==========================================================================
ylat = lattice(SAVE)
D = load_dmats(DMATS)
kf = full_zone_kpoints(ylat)
kidx, first = ibz_parents(ylat)
ops = spatial_ops(SAVE, kf, D, NV)

Qreq = np.array([*HIGH_SYMMETRY[POINT], 0.0])
d = kf[:, :2] - np.mod(Qreq[:2], 1.0)
d -= np.rint(d)
i = int(np.argmin(np.linalg.norm(d, axis=1)))
if np.linalg.norm(d[i]) > 1e-5:
    raise SystemExit(f"{POINT} is not on this mesh")
j = int(kidx[i])
Q = kf[first[j]]
path = os.path.join(BSE_DIR, f"ndb.BS_diago_Q{j + 1}")
if not os.path.exists(path):
    raise SystemExit(f"missing {path}")
stab = stabilizer(ops, Q)
kQ, w = shift_map(kf, Q)
if kQ is None:
    raise SystemExit(f"k+Q leaves the grid ({w:.1e})")
_, A, E = load_excitons(path, ylat, NSTATES)
A = np.asarray(A)
n = A.shape[0]
if STATES is None:
    grp = list(degeneracy_groups(E, tol=DEG_TOL)[0])
else:
    grp = [s - 1 for s in STATES]
print(f"{POINT}: database {os.path.basename(path)} at {np.round(Q[:2], 4)}; little group "
      f"{group_name(stab)} (order {len(stab)})")
print(f"  group: states {[g + 1 for g in grp]} at "
      f"{', '.join(f'{E[g]:.4f}' for g in grp)} eV; {n} states loaded")

# ==========================================================================
# 2. ARE THE EIGENVECTORS ORTHONORMAL?
# ==========================================================================
S = np.einsum("skvc,tkvc->st", np.conj(A[grp]), A[grp])
S_all = np.einsum("skvc,tkvc->st", np.conj(A), A)
off = np.abs(S_all - np.diag(np.diag(S_all)))
norm_dev = np.abs(np.real(np.diag(S_all)) - 1).max()
nonortho = off.max() > 1e-3 or norm_dev > 1e-3
print(f"\n  norms of the group's eigenvectors: {np.round(np.real(np.diag(S)), 6)}")
print(f"  largest |norm - 1| of the loaded states: {norm_dev:.2e}; largest overlap between "
      f"distinct states: {off.max():.2e}" + ("   <-- NOT orthonormal" if nonortho else ""))

# ==========================================================================
# 3. PER OPERATION
# ==========================================================================
allst = list(range(n))
print(f"\n  {'operation':<10} {'trace':>18} {'|MM^+ - 1|':>11} {'in group':>9} {'in all':>8}")
worst = []
for o in stab:
    M = rep_matrix_Q(A, o[-3], o[-2], o[-1], kQ, grp)
    Mall = rep_matrix_Q(A, o[-3], o[-2], o[-1], kQ, allst)
    dev = np.abs(M @ M.conj().T - np.eye(len(grp))).max()
    w_grp = np.min(np.sum(np.abs(M) ** 2, axis=0))                 # each rotated state, in the group
    w_all = np.min(np.sum(np.abs(Mall[:, grp]) ** 2, axis=0))      # ... among all loaded states
    tr = np.trace(M)
    worst.append((dev, o[0]))
    print(f"  {o[0]:<10} {tr.real:+8.4f}{tr.imag:+8.4f}i {dev:11.2e} {w_grp:9.4f} {w_all:8.4f}")

bad = [lab for dev, lab in worst if dev > 1e-2]
print("\n  verdict:")
if nonortho:
    print("   the eigenvectors are not orthonormal: the defect reflects that. Check BSEmod in the"
          " BSE input (resonant vs coupling); the characters (traces) can still be right.")
elif not bad:
    print("   unitary for every operation: the labels are safe.")
else:
    print(f"   defect on: {', '.join(bad)}. Compare 'in group' with 'in all': if 'in all' stays near 1,"
          " the group is not closed (raise DEG_TOL or STATES); if it drops, the rotation of those"
          " operations leaves the computed space (D matrices / band window).")
