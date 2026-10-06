"""
labels_from_elph.py -- phonon parities in g2's own mode order.

mode_labels.npy was built from matdyn and ordered by matdyn's frequencies;
g2's modes are ordered by the frequencies LetzElPhC used. Where the two
calculations differ by a few meV their energy orderings disagree, and a
coupling gets the other branch's parity. This takes each mode's parity from
the eigenvectors stored in the el-ph database itself -- exactly the modes g2
uses -- and writes a labels file selection_rule_test.py reads unchanged
(odd -> 3, even -> 1, undefined -> -1).

The parity comes from how sigma_h permutes the atoms (xphd.mirror):
p = sum_kappa <e_pi(kappa) | M e_kappa>, M = diag(1, 1, -1). For a planar
layer (GaN, hBN) the permutation is the identity and this is exactly the old
1 - 2 w_z, with the same thresholds, so their labels do not change. For a
layer like WSe2, where sigma_h swaps the two Se atoms, the old out-of-plane
weight mislabels modes such as A1' (out of plane but even) and E'' (in plane
but odd); the permutation formula does not.

Run it by editing the SETTINGS block below and running the file.
"""
import os

import numpy as np
from netCDF4 import Dataset

import xphd
from xphd.mirror import phonon_parity, sigma_h_permutation

# ==========================================================================
# 0. SETTINGS -- edit these, then run this file (VS Code: Run Python File)
# ==========================================================================
ELPH = "../../LELPH/ndb.elph"          # the LetzElPhC database behind the archives
ARCHIVE = "GI_ExcPh_Q0001.npz"         # any archive: only its q-list is used
SAVE = "SAVE"                          # yambo SAVE: the atomic positions
PERM = None                            # or give the permutation by hand, e.g. [0, 2, 1]
OLD = "mode_labels.npy"                # optional: compare with an older labels file
OUT = "mode_labels_elph.npy"

# ==========================================================================
# 1. THE MIRROR'S ACTION ON THE ATOMS
# ==========================================================================
if PERM is None:
    from xphd.yambo import lattice
    ylat = lattice(SAVE)
    perm = sigma_h_permutation(np.asarray(ylat.car_atomic_positions, float),
                               np.asarray(ylat.atomic_numbers), ylat.lat,
                               tol=1e-2)
else:
    perm = np.asarray(PERM, int)
planar = bool(np.all(perm == np.arange(len(perm))))
print(f"sigma_h maps atoms {list(range(len(perm)))} -> {[int(x) for x in perm]}"
      f"  ({'planar layer' if planar else 'non-planar layer'})")

# ==========================================================================
# 2. READ THE EL-PH DATABASE, MATCH THE ARCHIVE'S q-POINTS
# ==========================================================================
RY2EV = 13.605693122994
db = Dataset(ELPH)
q_db = np.asarray(db.variables["qpoints"][:], float)
freq = np.asarray(db.variables["FREQ"][...].data, float) * RY2EV          # (nq, nm) eV
ev = db.variables["POLARIZATION_VECTORS"][:]
ev = np.asarray(ev[..., 0]) + 1j * np.asarray(ev[..., 1])                  # (nq, nm, nat, 3)
db.close()
if ev.shape[-2] != len(perm):
    raise SystemExit(f"{ELPH} has {ev.shape[-2]} atoms, the permutation "
                     f"{len(perm)}")

arc = xphd.ExcPhArchive(ARCHIVE)
qa = np.mod(arc.q_red, 1.0)
hw = arc.grid("hw")[arc._i, arc._j]                                        # g2's order
d = qa[:, None, :2] - np.mod(q_db[:, :2], 1.0)[None, :, :]
d -= np.rint(d)
dist = np.linalg.norm(d, axis=-1)
m = np.argmin(dist, axis=1)
worst = float(dist[np.arange(len(qa)), m].max())
if worst > 1e-4:
    raise SystemExit(f"q-points do not match the archive (worst {worst:.1e}): "
                     f"are the el-ph q-points in crystal coordinates?")
freq, ev = freq[m], ev[m]
print(f"matched {len(qa)} q-points (worst {worst:.1e}), {ev.shape[1]} modes")
dfreq = np.abs(np.abs(freq) - hw).max() * 1e3
print(f"el-ph frequencies vs the archive's hw_grid: max difference {dfreq:.4f} meV"
      + ("   (same modes, same order)" if dfreq < 0.01 else "   <-- NOT the same order"))

# ==========================================================================
# 3. PARITY FROM THE PERMUTATION
# ==========================================================================
p = phonon_parity(ev, perm)                                                 # (nq, nm)
lab = np.where(p < -0.9, 3, np.where(p > 0.9, 1, -1))
away = np.linalg.norm(qa[:, :2] - np.rint(qa[:, :2]), axis=1) > 1e-6
print(f"parity defined for {100 * np.mean(lab >= 0):.1f}% of (q, mode); "
      f"largest distance from +-1 outside Gamma: "
      f"{np.abs(np.abs(p[away]) - 1).max():.1e}")
if not planar:
    wz = (np.abs(ev[..., 2]) ** 2).sum(-1) / (np.abs(ev) ** 2).sum((-1, -2))
    differ = np.sign(p) != np.sign(1 - 2 * wz)
    print(f"the out-of-plane-weight formula would disagree at "
          f"{100 * differ.mean():.1f}% of (q, mode) for this layer")

# ==========================================================================
# 4. OPTIONAL COMPARISON WITH AN OLDER LABELS FILE, AND WRITE
# ==========================================================================
if OLD and os.path.exists(OLD):
    old = np.load(OLD)
    if old.shape == lab.shape:
        p_old = np.where(np.isin(old, [0, 3]), -1, 1)
        p_new = np.where(lab == 3, -1, 1)
        both = (old >= 0) & (lab >= 0)
        bad = both & (p_old != p_new)
        print(f"\nparity disagrees with {OLD} at {bad.sum()} of {both.sum()} "
              f"(q, mode) pairs")
        if bad.any():
            print("   q (reduced)          mode  hw (meV)  old  new")
            for iq, nu in list(zip(*np.where(bad)))[:12]:
                print(f"   {str(np.round(qa[iq, :2], 4)):20} {nu + 1:3d}  "
                      f"{hw[iq, nu] * 1e3:8.2f}   "
                      f"{'odd ' if p_old[iq, nu] < 0 else 'even'} "
                      f"{'odd' if p_new[iq, nu] < 0 else 'even'}")
    else:
        print(f"\n{OLD} has shape {old.shape}, not {lab.shape}: not compared")
np.save(OUT, lab)
print(f"\nwrote {OUT}. Next:\n   python selection_rule_test.py --labels {OUT}")
