"""
compare_band_interp.py
======================
Which way of refining the exciton bands -- local quadratics (xphd's default)
or the Fourier interpolant used before -- predicts YOUR bands better?

Run it by editing the SETTINGS block below and running the file -- in VS
Code, Run Python File. No command line is needed.

The test
--------
Keep every other point of the BSE mesh (24x24 -> 12x12), interpolate back to
24x24 with each method, and compare with the BSE energies that were left
out. Nothing here is modelled: the left-out energies are your own BSE
results, which neither interpolant has seen.

The test is biased AGAINST the local method. Doubling the spacing multiplies
its error by about 8 (it is third order), while Fourier's ringing from a
crossing does not shrink with the spacing. At the production refinement
(24 -> 360) both errors are smaller, the local one much more so.

What it reports
---------------
1. For each band, the rms and largest error over the left-out points.
2. At the band minimum, the curvature each method predicts from the 12x12
   points against the curvature of the left-out BSE points around it -- the
   quantity the width of an exciton at a minimum depends on.

Any archive will do: its E_m grid holds the whole exciton band structure,
shifted by that archive's Q.
"""
import numpy as np

from xphd.core.interp import fine_band
from xphd.core.mesh import hex_norm
from xphd.io.excph import ExcPhArchive

# ==========================================================================
# 0. SETTINGS -- edit these, then run this file (VS Code: Run Python File)
# ==========================================================================
ARCHIVE = "GI_ExcPh_Q0001.npz"

# test the lowest this many bands
BANDS = 6

# ==========================================================================
# 1. LEAVE HALF OUT, INTERPOLATE BACK
# ==========================================================================
arc = ExcPhArchive(ARCHIVE)
n1, n2 = arc.n1, arc.n2
if n1 % 2 or n2 % 2:
    raise SystemExit(f"the mesh is {n1}x{n2}; this test needs even sides")
E = np.asarray(arc.grid("E_m"), float)[..., :BANDS]        # (n1, n2, nb)
nb = E.shape[-1]
kept = E[::2, ::2]
held = np.ones((n1, n2), bool)
held[::2, ::2] = False
print(f"{ARCHIVE}: {n1}x{n2} mesh, Q = {np.round(arc.Q_red[:2], 4)}; "
      f"keeping {n1//2}x{n2//2}, testing on the {int(held.sum())} points "
      f"left out")
pred = {m: fine_band(kept, 2, m) for m in ("fourier", "local")}
for m, F in pred.items():
    assert np.allclose(F[::2, ::2], kept), f"{m} does not keep its nodes"

# ==========================================================================
# 2. ERRORS BAND BY BAND
# ==========================================================================
print("\n1. Error at the left-out points (meV)")
print(f"   band | {'fourier rms':>11} {'max':>7} | {'local rms':>9} {'max':>7}"
      f" | better")
wins = {"fourier": 0, "local": 0}
for b in range(nb):
    row, rms = [], {}
    for m in ("fourier", "local"):
        d = (pred[m][..., b] - E[..., b])[held] * 1e3
        rms[m] = np.sqrt((d ** 2).mean())
        row.append(f"{rms[m]:{11 if m == 'fourier' else 9}.2f} "
                   f"{np.abs(d).max():7.2f}")
    best = min(rms, key=rms.get)
    wins[best] += 1
    print(f"   {b+1:4d} | {row[0]} | {row[1]} | {best}")
print(f"   lower rms error: local on {wins['local']} of {nb} bands, fourier "
      f"on {wins['fourier']}")

# ==========================================================================
# 3. CURVATURE AT THE MINIMUM
# ==========================================================================
i0, j0 = np.unravel_index(np.argmin(E[..., 0]), (n1, n2))
print(f"\n2. Curvature at the minimum of band 1, at q = "
      f"({i0}/{n1}, {j0}/{n2})")
if i0 % 2 or j0 % 2:
    print("   the minimum is not on the kept 12x12 points; its curvature "
          "cannot be tested this way")
else:
    nbrs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
    q2 = hex_norm(np.array([[1 / n1, 0.0, 0.0]]))[0] ** 2
    e0 = E[i0, j0, 0]

    def curv(F):
        return np.array([(F[(i0 + a) % n1, (j0 + c) % n2, 0] - e0) / q2
                         for a, c in nbrs])

    true = curv(E)
    print(f"   left-out BSE points: b = {true.mean():7.3f} eV/|b|^2 "
          f"(range {true.min():.3f}..{true.max():.3f})")
    for m in ("fourier", "local"):
        c = curv(pred[m])
        print(f"   {m:>7} prediction:  b = {c.mean():7.3f} eV/|b|^2 "
              f"(range {c.min():.3f}..{c.max():.3f})   off by "
              f"{100*(c.mean()/true.mean()-1):+.0f}%")
    print("   a minimum's width scales as 1/b, so this error carries straight "
          "into it")
