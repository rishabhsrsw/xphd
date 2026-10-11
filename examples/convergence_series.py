"""
Usage: with an archive and hw_fine.npy in each <N>x<N>/ folder listed in
MESHES, run

    python convergence_series.py
"""
import numpy as np
import xphd

FINE = 360
MESHES = [6, 8, 12, 18, 24]
T = [10.0, 77.0, 150.0, 300.0]

rows = []
for n in MESHES:
    arc = xphd.ExcPhArchive(f"{n}x{n}/GI_ExcPh_Q0001.npz")
    arc.check()
    hw = np.load(f"{n}x{n}/hw_fine.npy")
    res = xphd.compute(arc, T, hw_fine=hw, refine=FINE // n,
                       acoustic_cut=5e-4)
    rows.append((n, arc.E_n[0], res.total[:, 0] * 1e3))

print(f"\n{'mesh':>6} {'E_1 (eV)':>10} " +
      " ".join(f"{t:>9.0f}K" for t in T))
for n, e1, g in rows:
    print(f"{n:>6} {e1:10.4f} " + " ".join(f"{v:10.3f}" for v in g))
