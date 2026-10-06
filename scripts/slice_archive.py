"""
slice_archive.py
================
Copies of an exciton-phonon archive that keep only the first n exciton states,
initial and final -- for the convergence of a linewidth with the number of
states in the sum over final states (SI Fig. 16b).

A state's width depends on the final states it can reach: every state below it
(emission) and those up to about one phonon energy above it (absorption). The
copies drop the states above n; if the width stops changing once n passes the
states within reach, the sum is converged.

Run it by editing the SETTINGS block below and running the file. It writes
one archive per n and prints the linewidth command for each.

What is sliced
--------------
G_grid, g2_grid, Ge_grid, Gh_grid   (nq, nmod, nexc, nexc)  -> [..., :n, :n]
E_n_grid                            (nexc,)                 -> [:n]
E_m_grid                            (nq, nexc)              -> [:, :n]
osc_Q0                              (nexc,)                 -> [:n]
Any other array whose LAST axis has length nexc is sliced along it, with a
note; everything else is copied unchanged.
"""
import os

import numpy as np

# ==========================================================================
# 0. SETTINGS -- edit these, then run this file (VS Code: Run Python File)
# ==========================================================================
ARCHIVE = "GI_ExcPh_Q0001.npz"         # the archive to slice
N_STATES = [1, 2, 3, 4, 5]             # keep the first n states, for each n
OUT_DIR = "sliced"                     # where the copies go
STATE = 1                              # 1-based: the state whose width you will converge
T = [77, 300]                          # temperatures for the printed commands
HW_FINE = "hw_fine.npy"                # for the printed commands


# ==========================================================================
# 1. READ
# ==========================================================================
d = dict(np.load(ARCHIVE, allow_pickle=True))
key4 = [k for k in ("G_grid", "g2_grid", "Ge_grid", "Gh_grid", "G", "g2", "Ge", "Gh") if k in d]
if not key4:
    raise SystemExit(f"{ARCHIVE}: no G_grid / g2_grid -- is this an exciton-phonon archive?")
nexc = d[key4[0]].shape[-1]
print(f"{ARCHIVE}: {nexc} exciton states, keys {sorted(d)}")
if max(N_STATES) > nexc:
    raise SystemExit(f"N_STATES goes up to {max(N_STATES)} but the archive has {nexc} states")
if STATE > min(N_STATES):
    print(f"  note: state {STATE} is absent from the copies with n < {STATE}")

# ==========================================================================
# 2. SLICE AND WRITE
# ==========================================================================
os.makedirs(OUT_DIR, exist_ok=True)
base = os.path.splitext(os.path.basename(ARCHIVE))[0]
written = []
for n in N_STATES:
    out, notes = {}, []
    for k, v in d.items():
        a = np.asarray(v)
        if k in key4:
            out[k] = a[..., :n, :n]
        elif k in ("E_n_grid", "E_n", "osc_Q0") and a.ndim == 1 and a.shape[0] == nexc:
            out[k] = a[:n]
        elif k in ("E_m_grid", "E_m") and a.ndim == 2 and a.shape[1] == nexc:
            out[k] = a[:, :n]
        elif a.ndim >= 1 and a.shape[-1] == nexc and k not in ("mesh", "q_red", "Q_red"):
            out[k] = a[..., :n]
            notes.append(k)
        else:
            out[k] = a
    path = os.path.join(OUT_DIR, f"{base}_n{n}.npz")
    np.savez(path, **out)
    written.append((n, path))
    print(f"  n = {n}: {path}" + (f"   (also sliced: {', '.join(notes)})" if notes else ""))

# ==========================================================================
# 3. THE COMMANDS TO RUN NEXT
# ==========================================================================
tt = " ".join(str(t) for t in T)
print("\nthen, for each copy (PowerShell):")
print(f'  foreach ($n in {",".join(str(n) for n, _ in written if n >= STATE)}) {{')
print(f'    xphd linewidth {OUT_DIR}\\{base}_n$n.npz --hw-fine {HW_FINE} --T {tt} '
      f'--acoustic-cut 5e-4 --acoustic-model q2 --n-shell 1 -o {OUT_DIR}\\lw_n$n.txt')
print("  }")
print(f"and read state {STATE}'s width in each lw_n*.txt against n.")
