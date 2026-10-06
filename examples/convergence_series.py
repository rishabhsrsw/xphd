"""Linewidth convergence across source meshes at a fixed fine mesh.

Hold the integration grid at 360x360 and vary only the source, so the two
convergences are not mixed. Note the BSE k-mesh is the q-mesh, so exciton
energies move too -- track states by ENERGY, not index.
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
