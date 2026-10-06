"""
window_closure.py
=================
Is the BSE band window closed under sigma_h?

The exciton's sigma_h matrix is built from the single-particle matrix
D(sigma_h) restricted to the BSE window. Every eigen-parity can be exactly
+-1 only if that restricted matrix is unitary at every k: each band in the
window must have its whole sigma_h image inside the window. A band whose
mirror partner lies OUTSIDE the window (band 66 degenerate with 67, or 59
with 58, at some k) leaks: its row of D has norm below 1, and every exciton
that uses it maps partly outside the BSE space -- which no rotation among the
computed excitons can repair.

This checks the norm of every row and column of D(sigma_h) in the window,
band by band, and how much of the BSE's transition space (k, v, c) touches a
leaking band.

Run it by editing the SETTINGS block below and running the file -- in VS
Code, Run Python File. No command line is needed. soc_probe.py prints the
index of sigma_h among the operations ('sigma_h is operation 6' for WSe2).
"""
import numpy as np

# ==========================================================================
# 0. SETTINGS -- edit these, then run this file (VS Code: Run Python File)
# ==========================================================================
DMATS = "Dmats.npy"
SIGMA_H_OP = 6                 # index of sigma_h, as soc_probe.py prints it
FIRST_BAND = 59                # 1-based number of the window's first band
NV = 4                         # valence bands in the window
TOL = 0.99                     # a row norm below this counts as leaking

# ==========================================================================
# 1. LOAD
# ==========================================================================
D = np.load(DMATS)
D = D[:, :, 0] if D.ndim == 5 else D           # (nsym, nk, nb, nb)
Dk = D[SIGMA_H_OP]
nk, nb, _ = Dk.shape
bands = np.arange(FIRST_BAND, FIRST_BAND + nb)
sq = np.einsum("kij,kjl->kil", Dk, Dk)
sq_diag = float(np.real(np.einsum("kii->", sq)) / (nk * nb))
print(f"{DMATS}: operation {SIGMA_H_OP}, {nk} k-points, bands {bands[0]}-{bands[-1]} "
      f"({NV} valence)")
print(f"   mean diagonal of D^2 = {sq_diag:+.4f}  (-1 spinors, +1 spinless: is this sigma_h?)")

# ==========================================================================
# 2. UNITARITY BAND BY BAND
# ==========================================================================
rows = (np.abs(Dk) ** 2).sum(axis=2)          # (nk, nb): |sigma_h band i> inside the window
cols = (np.abs(Dk) ** 2).sum(axis=1)          # (nk, nb)
vc = (np.abs(Dk[:, :NV, NV:]) ** 2).sum() + (np.abs(Dk[:, NV:, :NV]) ** 2).sum()
print(f"   valence <-> conduction weight in D (should be 0): {vc:.1e}")
print("\n   band   min row norm   k below TOL   min column norm")
leak = (rows < TOL) | (cols < TOL)
for i, b in enumerate(bands):
    kind = "v" if i < NV else "c"
    print(f"   {b:4d}{kind}     {rows[:, i].min():.4f}       {int((rows[:, i] < TOL).sum()):4d}"
          f"          {cols[:, i].min():.4f}")

# ==========================================================================
# 3. HOW MUCH OF THE BSE SPACE IS AFFECTED
# ==========================================================================
lv, lc = leak[:, :NV], leak[:, NV:]
touched = lv[:, :, None] | lc[:, None, :]       # (k, v, c) transitions using a leaking band
print(f"\n   transitions (k, v, c) touching a leaking band: "
      f"{100 * touched.mean():.1f}% of {touched.size}")
worst = np.argsort(rows.min(axis=1))[:6]
print("   worst k-points (full-zone index): "
      + ", ".join(f"{k} (band {bands[np.argmin(rows[k])]}, {rows[k].min():.3f})" for k in worst))

print("\nVERDICT")
if not leak.any():
    print("   D(sigma_h) is unitary in the window at every k: the window is closed,")
    print("   and imperfect eigen-parities come from somewhere else")
else:
    edge = leak[:, [0, nb - 1]].any()
    print(f"   the window is NOT closed under sigma_h: {int(leak.any(axis=0).sum())} band(s) leak at "
          f"{int(leak.any(axis=1).sum())} of {nk} k-points"
          + (" -- including the window's edge bands" if edge else ""))
    print("   Excitons built on those transitions cannot be exact parity eigenstates")
    print("   inside the BSE space. A wider band window (so that its edges fall in")
    print("   gaps) removes it; otherwise report their coupling as undefined.")
