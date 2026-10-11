"""
Usage: set the inputs in the SETTINGS block below, then run

    python final_state_window.py
"""
import glob

import numpy as np

# ==========================================================================
# 0. SETTINGS -- edit these, then run this file (VS Code: Run Python File)
# ==========================================================================
ARCHIVES = "GI_ExcPh_Q*.npz"           # all archives (one per irreducible Q)
N_INITIAL = 5                          # the initial states whose widths are quoted
N_WORST = 5                            # how many of the tightest archives to list


# ==========================================================================
# 1. MARGINS
# ==========================================================================
files = sorted(glob.glob(ARCHIVES))
if not files:
    raise SystemExit(f"no archives match {ARCHIVES!r}")
rows = []
for f in files:
    d = np.load(f, allow_pickle=True)
    En = np.asarray(d["E_n_grid"], float)              # states at Q
    Em = np.asarray(d["E_m_grid"], float)              # states at Q + q, (nq, nexc)
    hw = np.asarray(d["hw_grid"], float)               # phonon energies, (nq, nmod), eV
    n = min(N_INITIAL, len(En))
    reach = En[:n].max() + np.nanmax(hw)
    top = Em[:, -1].min()
    rows.append((top - reach, f, len(En), En[:n].max(), np.nanmax(hw), top))

rows.sort()
worst = rows[0]
print(f"{len(files)} archives; {worst[2]} states each; initial states 1-{N_INITIAL}; "
      f"largest phonon energy {1e3 * max(r[4] for r in rows):.1f} meV")
print(f"\n  tightest {min(N_WORST, len(rows))} archives:")
print(f"  {'archive':<26} {'E_top - reach (meV)':>20} {'highest initial (eV)':>21} {'lowest top state (eV)':>22}")
for m, f, ne, emax, hmax, top in rows[:N_WORST]:
    print(f"  {f.split('/')[-1].split(chr(92))[-1]:<26} {1e3 * m:20.1f} {emax:21.4f} {top:22.4f}")

print("\n  VERDICT")
if worst[0] > 0:
    print(f"   complete: at every Q the highest stored state lies at least {1e3 * worst[0]:.0f} meV above every")
    print(f"   energy reachable from states 1-{N_INITIAL}; states beyond the archive cannot contribute")
    print(f"   (broadening-free integration), so the sum over final states is exact.")
else:
    bad = sum(1 for r in rows if r[0] <= 0)
    print(f"   INCOMPLETE at {bad} of {len(rows)} archives: absorption into states beyond the archive is")
    print(f"   cut there (by up to {-1e3 * worst[0]:.0f} meV). Regenerate those with more states, or quote")
    print(f"   fewer initial states (lower N_INITIAL until the margin is positive).")
