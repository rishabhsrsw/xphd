"""
add_osc.py
==========
Add the Q = 0 oscillator strengths that `xphd bte --inject optical` needs
(field osc_Q0) to the Gamma archive, from the exciton dipoles written by
`xphd dipoles`. `xphd generate` does not store them.

Run it by editing the SETTINGS block below and running the file -- in VS
Code, Run Python File. No command line is needed.

Only RELATIVE strengths matter: the injected population is normalised. For
light at normal incidence the strength is the in-plane |d_x|^2 + |d_y|^2;
sigma_h-odd (z-polarised) excitons then receive nothing, as they should.

The dipoles must come from the SAME BSE as the archive. The first n states of
the dipole file are matched to the archive's n states at Q = 0, and the
script refuses if the counts disagree. A backup of the archive is written
once, as <name>.bak.npz.
"""
import os

import numpy as np

# ==========================================================================
# 0. SETTINGS -- edit these, then run this file (VS Code: Run Python File)
# ==========================================================================
ARCHIVE = "GI_ExcPh_Q0001.npz"      # the Q = 0 archive
DIPOLES = "exc_dipoles.npy"         # from xphd dipoles, (3, n) complex
POLARISATION = "in-plane"           # "in-plane" (normal incidence) or "total"

# ==========================================================================
# 1. LOAD
# ==========================================================================
d = dict(np.load(ARCHIVE, allow_pickle=False))
Q = np.asarray(d["Q_red"], float)[:2]
if np.abs(Q - np.rint(Q)).max() > 1e-6:
    raise SystemExit(f"{ARCHIVE} is at Q = {Q}: optical injection fills Q = 0 only")
En = np.asarray(d["E_n_grid"], float)
ne = len(En)
dip = np.load(DIPOLES)
if dip.ndim != 2 or dip.shape[0] != 3:
    raise SystemExit(f"{DIPOLES} has shape {dip.shape}; expected (3, n) from xphd dipoles")
if dip.shape[1] < ne:
    raise SystemExit(f"{DIPOLES} holds {dip.shape[1]} states, the archive {ne}")

# ==========================================================================
# 2. STRENGTHS, AND WRITE
# ==========================================================================
if POLARISATION == "in-plane":
    osc = (np.abs(dip[:2, :ne]) ** 2).sum(axis=0)
elif POLARISATION == "total":
    osc = (np.abs(dip[:, :ne]) ** 2).sum(axis=0)
else:
    raise SystemExit('POLARISATION must be "in-plane" or "total"')
rel = osc / max(osc.max(), 1e-300)

print(f"{ARCHIVE}: {ne} states at Q = 0;  {POLARISATION} oscillator strengths (relative)")
for n in range(ne):
    print(f"   state {n + 1:3d}  E = {En[n]:8.4f} eV   {rel[n]:.3e}"
          + ("   <- bright" if rel[n] > 0.05 else ""))

bak = ARCHIVE[:-4] + ".bak.npz"
if not os.path.exists(bak):
    np.savez(bak, **d)
    print(f"backup written: {bak}")
d["osc_Q0"] = osc
np.savez(ARCHIVE, **d)
print(f"wrote osc_Q0 into {ARCHIVE}; xphd bte --inject optical can now use it")
