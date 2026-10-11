"""
Usage: set the inputs in the SETTINGS block below, then run

    python add_osc.py
"""
import os

import numpy as np

# ==========================================================================
# 0. SETTINGS -- edit these, then run this file (VS Code: Run Python File)
# ==========================================================================
ARCHIVE = "GI_ExcPh_Q0001.npz"      # the Q = 0 archive
SOURCE = "dipoles"                  # "dipoles" (xphd dipoles) or "yambo" (BSE residuals)
DIPOLES = "exc_dipoles.npy"         # SOURCE = "dipoles": from xphd dipoles, (3, n) complex
POLARISATION = "in-plane"           # SOURCE = "dipoles": "in-plane" (normal incidence) or "total"
BSE_DB = "../BSE/output_all/ndb.BS_diago_Q1"   # SOURCE = "yambo": the Q = 0 BSE database
SAVE = "../phonons/SAVE"            # SOURCE = "yambo": the SAVE of that BSE
DEG_TOL = 0.012                     # eV, SOURCE = "yambo": states closer than this share equally
N_INJECT = 5                        # states the transport injects into (bands in the rate matrix)

# ==========================================================================
# 1. LOAD
# ==========================================================================
d = dict(np.load(ARCHIVE, allow_pickle=False))
Q = np.asarray(d["Q_red"], float)[:2]
if np.abs(Q - np.rint(Q)).max() > 1e-6:
    raise SystemExit(f"{ARCHIVE} is at Q = {Q}: optical injection fills Q = 0 only")
En = np.asarray(d["E_n_grid"], float)
ne = len(En)

if SOURCE == "dipoles":
    dip = np.load(DIPOLES)
    if dip.ndim != 2 or dip.shape[0] != 3:
        raise SystemExit(f"{DIPOLES} has shape {dip.shape}; expected (3, n) from xphd dipoles")
    if dip.shape[1] < ne:
        raise SystemExit(f"{DIPOLES} holds {dip.shape[1]} states, the archive {ne}")
    if POLARISATION == "in-plane":
        osc = (np.abs(dip[:2, :ne]) ** 2).sum(axis=0)
    elif POLARISATION == "total":
        osc = (np.abs(dip[:, :ne]) ** 2).sum(axis=0)
    else:
        raise SystemExit('POLARISATION must be "in-plane" or "total"')
    what = f"{POLARISATION} |d|^2 from {DIPOLES}"
elif SOURCE == "yambo":
    from xphd.yambo import lattice, load_excitons
    exc, _, Edb = load_excitons(BSE_DB, lattice(SAVE))
    rr = np.abs(np.asarray(getattr(exc, "r_residual", []))) ** 2
    if rr.size == 0:
        raise SystemExit(f"{BSE_DB}: no residuals (r_residual) in this database")
    res = rr.reshape(-1, rr.shape[-1]).sum(axis=0) if rr.ndim > 1 else rr
    Edb = np.asarray(Edb, float)
    if len(res) < ne or len(Edb) < ne:
        raise SystemExit(f"{BSE_DB} holds {min(len(res), len(Edb))} states, the archive {ne}")
    dE = np.abs(Edb[:ne] - En).max()
    if dE > 2e-3:
        raise SystemExit(f"{BSE_DB} and {ARCHIVE} differ in energy by {dE * 1e3:.1f} meV: "
                         f"not the same BSE")
    osc = res[:ne].astype(float).copy()
    groups, cur = [], [0]
    for n in range(1, ne):
        if abs(En[n] - En[cur[-1]]) < DEG_TOL:
            cur.append(n)
        else:
            groups.append(cur)
            cur = [n]
    groups.append(cur)
    for g in groups:
        osc[g] = osc[g].sum() / len(g)
    what = f"yambo residuals from {BSE_DB}, shared equally within manifolds ({DEG_TOL * 1e3:.0f} meV)"
else:
    raise SystemExit('SOURCE must be "dipoles" or "yambo"')

# ==========================================================================
# 2. REPORT AND WRITE
# ==========================================================================
rel = osc / max(osc.max(), 1e-300)
print(f"{ARCHIVE}: {ne} states at Q = 0; strengths: {what}")
for n in range(ne):
    print(f"   state {n + 1:3d}  E = {En[n]:8.4f} eV   {rel[n]:.3e}"
          + ("   <- bright" if rel[n] > 0.05 else ""))
k = min(N_INJECT, ne)
share = osc[:k] / max(osc[:k].sum(), 1e-300)
print(f"\n   with --inject optical and no pump window, states 1-{k} receive:")
for n in range(k):
    if share[n] > 1e-4:
        print(f"     state {n + 1}  E = {En[n]:.4f} eV   {100 * share[n]:5.1f}%")

bak = ARCHIVE[:-4] + ".bak.npz"
if not os.path.exists(bak):
    np.savez(bak, **d)
    print(f"backup written: {bak}")
d["osc_Q0"] = osc
np.savez(ARCHIVE, **d)
print(f"wrote osc_Q0 into {ARCHIVE}; xphd bte --inject optical can now use it")
