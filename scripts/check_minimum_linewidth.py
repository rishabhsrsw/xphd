"""
check_minimum_linewidth.py
==========================
An independent check on the linewidth of an exciton at the BOTTOM of its band
-- the K minimum of monolayer GaN, the Gamma minimum of hBN.

Run it by editing the SETTINGS block below and running the file -- in VS
Code, Run Python File. No command line is needed.

Why a minimum still has a width
-------------------------------
At the minimum the exciton cannot emit, but it can ABSORB an acoustic phonon
of wavevector q and move up its own valley, provided

    b q^2 = hbar c q        ->   q* = hbar c / b ,   hbar w* = (hbar c)^2 / b

for a band E = E_min + b q^2 and a branch hbar w = hbar c q. Every such final
state lies inside the first mesh cell around the minimum, so the coupling (a
q^2 model) and the band are both extrapolated there. For a coupling
|G|^2 2 hbar w = C q^2 the result has a closed form,

    Gamma_FWHM = (4 pi^2 / sqrt3) C N(hbar w*) / b^2 ,

with q in units of |b|; the 4 pi^2/sqrt3 is 2pi x 2pi x the 2/sqrt3 that
turns d^2q into a fraction of the zone. It is linear in T once k_B T exceeds
hbar w*, and freezes out below. This script builds every ingredient from
YOUR files and evaluates it -- no mesh, no integration.

What it reports
---------------
1. The exciton curvature at the minimum from the genuine BSE points, and on
   the fine band xphd integrates (BAND_INTERP). The default local
   interpolation reproduces the BSE curvature; the Fourier interpolant's
   curvature is shown too, since its ringing is what distorted the old
   results (GaN's K minimum: 34% too steep).
2. Each acoustic branch's velocity, its coupling coefficient C for the
   minimum-to-minimum channel, and the deformation potential C implies,
       C = Xi^2 hbar^2 |b|^2 / M_cell ,
   the number a strain calculation checks independently.
3. The closed-form width, and where the ring |q| = q* sits: when it crosses
   the edge of the Gamma cell, the code uses the nearest mesh point's
   coupling there, unchanged, and overshoots.
4. xphd's own number for comparison (RUN_CODE), and a best estimate.
"""
import contextlib
import io

import numpy as np

from xphd.core.interp import fine_band
from xphd.core.mesh import hex_norm, shells
from xphd.io.excph import ExcPhArchive

# ==========================================================================
# 0. SETTINGS -- edit these, then run this file (VS Code: Run Python File)
# ==========================================================================
# the archive whose Q is the minimum (GaN: the K valley, Q0575)
ARCHIVE = "GI_ExcPh_Q0575.npz"

# the matdyn frequencies on the fine mesh, in eV (from xphd hw-fine)
HW_FINE = "hw_fine.npy"

# the exciton at the minimum, 0-based, and the final band it scatters into
STATE = 0
FINAL = 0

# temperatures (K)
TEMPERATURES = [5.0, 77.0, 300.0]

# coarse shells used for the q^2 coupling -- match --n-shell of your run
N_SHELL = 1

# only for converting to physical units: lattice constant (Angstrom) and the
# atomic masses (amu) of the cell
ALAT = 3.2
MASSES = [69.723, 14.007]

# how xphd refines the exciton energies: "local" (default) or "fourier"
BAND_INTERP = "local"

# also run xphd's own linewidth for the same state, for comparison
RUN_CODE = True
REFINE = 15
ACOUSTIC_CUT = 5e-4

# ==========================================================================
# 1. LOAD
# ==========================================================================
KB = 8.617333262e-5                 # eV/K
HBAR = 6.582119569e-16              # eV s
H2_2ME = 3.80998                    # hbar^2 / 2 m_e, eV Angstrom^2
AMU_ME = 1822.888

arc = ExcPhArchive(ARCHIVE)
n1, n2 = arc.n1, arc.n2
E_n = float(arc.E_n[STATE])
E_m = arc.grid("E_m")                            # (n1, n2, ne)
hw_c = arc.grid("hw").reshape(n1 * n2, -1)       # (nq, nmod)
g2 = arc.grid("g2")
g2 = g2.reshape(n1 * n2, *g2.shape[2:])          # (nq, nmod, ne, ne)
hw_fine = np.load(HW_FINE)
N1, N2 = n1 * REFINE, n2 * REFINE
if hw_fine.ndim == 2:
    hw_fine = hw_fine.reshape(N1, N2, -1)
if hw_fine.shape[:2] != (N1, N2):
    raise SystemExit(f"{HW_FINE} is {hw_fine.shape[:2]}, expected "
                     f"({N1}, {N2}) = {n1} x REFINE {REFINE}")
hwf = hw_fine.reshape(N1 * N2, -1)
bmag = 4 * np.pi / (np.sqrt(3) * ALAT)           # |b| in 1/Angstrom
print(f"{ARCHIVE}: {n1}x{n2} mesh, state {STATE} at {E_n:.4f} eV; "
      f"fine mesh {N1}x{N2}")


def grid_q(n):
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return np.stack([i.ravel() / n, j.ravel() / n, 0 * i.ravel()], 1)


sh = shells(hex_norm(grid_q(n1)))                # coarse, |q| = 0 first
shf = shells(hex_norm(grid_q(N1)))               # fine

# ==========================================================================
# 2. EXCITON CURVATURE AT THE MINIMUM
# ==========================================================================
dE = E_m.reshape(n1 * n2, -1)[:, FINAL] - E_n
if dE.min() < -1e-4:
    print(f"   WARNING: some final states lie {-dE.min()*1e3:.1f} meV BELOW "
          f"state {STATE}; it is not the minimum")
print("\n1. Exciton band around the minimum")
print("   genuine BSE points:")
for s in range(1, 4):
    q, idx = sh[s]
    bb = dE[idx] / q ** 2
    print(f"   shell {s}: |q| = {q:.4f} |b|   b = {bb.mean():7.3f} eV/|b|^2 "
          f"(range {bb.min():.3f}..{bb.max():.3f})")
b_bse = float((dE[sh[1][1]] / sh[1][0] ** 2).mean())


def fine_curvature(method):
    E = fine_band(E_m[..., FINAL], REFINE, method).reshape(N1 * N2)
    out = []
    for s in (1, 2, 4, 8):
        q, idx = shf[s]
        bb = (E[idx] - E_n) / q ** 2
        out.append((q, bb.mean(), bb.min(), bb.max()))
    return out


for method in dict.fromkeys([BAND_INTERP, "fourier"]):
    tag = ("what xphd integrates" if method == BAND_INTERP
           else "for comparison: its ringing distorted earlier results")
    print(f"   fine band, {method} interpolation ({tag}):")
    fc = fine_curvature(method)
    for (q, bm, blo, bhi), s in zip(fc, (1, 2, 4, 8)):
        print(f"   fine shell {s}: |q| = {q:.4f} |b|   b = {bm:7.3f} eV/|b|^2 "
              f"(range {blo:.3f}..{bhi:.3f})")
    if method == BAND_INTERP:
        b_int = float(np.mean([c[1] for c in fc[:2]]))
if abs(b_int / b_bse - 1) > 0.1:
    print(f"   WARNING: the band xphd integrates is {100*(b_int/b_bse-1):+.0f}% "
          f"off the BSE curvature here -- set BAND_INTERP = 'local'")

# ==========================================================================
# 3. ACOUSTIC VELOCITIES AND COUPLINGS
# ==========================================================================
names = ["ZA", "TA", "LA"]
m_cell = sum(MASSES) * AMU_ME
print("\n2. Acoustic branches (energy order at small q: ZA, TA, LA)")
branches = []
for nu in range(3):
    slopes = [float((hwf[idx, nu] / q).mean()) for q, idx in shf[1:7]]
    hc = float(np.median(slopes))
    lin = max(slopes) / max(min(slopes), 1e-30)
    shell_C = []
    for s in range(1, 5):
        q, idx = sh[s]
        shell_C.append(float((g2[idx, nu, FINAL, STATE] * 2 * hw_c[idx, nu])
                             .mean() / q ** 2))
    C = float(np.mean(shell_C[:N_SHELL]))
    # the exciton form factor lowers C with q, so the innermost shell slightly
    # underestimates the long-wavelength coupling: extrapolate C = C0 + a q^2
    q_sh = np.array([sh[s][0] for s in range(1, 4)])
    C0 = C
    if min(shell_C[:3]) > 0:
        C0 = max(float(np.polyfit(q_sh ** 2, shell_C[:3], 1)[1]), C)
    xi = np.sqrt(max(C, 0) * m_cell / (2 * H2_2ME * bmag ** 2))
    c_kms = hc / (bmag * 1e10 * HBAR) / 1e3
    print(f"   {names[nu]}: hbar c = {hc:.4f} eV/|b|  ({c_kms:5.1f} km/s"
          + (", NOT linear -- quadratic branch" if lin > 1.5 else "") + ")")
    print(f"       C = {C:.3e} eV^2/|b|^2 from {N_SHELL} shell(s);"
          f" per shell 1-4: " + ", ".join(f"{v:.2e}" for v in shell_C))
    print(f"       implied deformation potential Xi = {xi:.2f} eV")
    if lin <= 1.5 and C > 0:
        branches.append((names[nu], hc, C, C0))

# ==========================================================================
# 4. THE CLOSED FORM
# ==========================================================================
h_cell = 1 / (2 * n1)
Bm = np.array([[1.0, 0.0], [0.5, np.sqrt(3) / 2]])       # 60-degree basis
ang = np.linspace(0, 2 * np.pi, 721)[:-1]


def closed_form(b, C, hc, T):
    return ((4 * np.pi ** 2 / np.sqrt(3)) * C
            / np.expm1(hc ** 2 / b / (KB * T)) / b ** 2)


print(f"\n3. The absorbed phonon, on the band xphd integrates (b = {b_int:.3f})")
for name, hc, C, C0 in branches:
    qs, ws = hc / b_int, hc ** 2 / b_int
    red = np.stack([qs * np.cos(ang), qs * np.sin(ang)], 1) @ np.linalg.inv(Bm)
    inside = float(np.mean(np.all(np.abs(red) <= h_cell, axis=1)))
    print(f"   {name}: q* = {qs:.4f} |b| = {qs*N1:.1f} fine spacings, "
          f"hbar w* = {ws*1e3:.2f} meV; {inside*100:.0f}% of the ring inside "
          f"the Gamma cell")
    if qs * N1 < 3:
        print("       the ring spans under 3 fine spacings: expect the code "
              "to overshoot by ~10%")
    if inside < 0.99:
        print("       part of the ring lies outside the Gamma cell, where "
              "the code uses the\n       nearest mesh point's coupling "
              "unchanged: expect it to overshoot")

print("\n   Gamma at the minimum (meV), acoustic absorption within the valley")
print(f"   {'':>11}  {'band xphd integrates':>21}  {'BSE curvature':>14}  "
      f"{'BSE curv., C at q->0':>21}")
ref, best = {}, {}
for T in TEMPERATURES:
    ref[T] = sum(closed_form(b_int, C, hc, T) for _, hc, C, _ in branches)
    lo = sum(closed_form(b_bse, C, hc, T) for _, hc, C, _ in branches)
    hi = sum(closed_form(b_bse, C0, hc, T) for _, hc, _, C0 in branches)
    best[T] = (lo, hi)
    print(f"   T = {T:5.0f} K  {ref[T]*1e3:21.3f}  {lo*1e3:14.3f}  "
          f"{hi*1e3:21.3f}")
if branches:
    hc_dom = max(branches, key=lambda u: u[2])[1]
    print(f"   crossover hbar w*/k_B = {hc_dom**2/b_bse/KB:.0f} K: below it "
          f"the absorbed phonons freeze out and the line narrows")
print(f"   BSE curvature b = {b_bse:.3f} eV/|b|^2 -> exciton mass "
      f"{H2_2ME * bmag ** 2 / b_bse:.2f} m_e")

# ==========================================================================
# 5. xphd's OWN NUMBER, FOR COMPARISON
# ==========================================================================
if RUN_CODE:
    from xphd.linewidth import compute
    print(f"\n4. xphd linewidth for state {STATE} (q2, n_shell={N_SHELL}, "
          f"refine {REFINE}, band_interp={BAND_INTERP}) ...")
    with contextlib.redirect_stdout(io.StringIO()):
        res = compute(arc, TEMPERATURES, hw_fine=hwf, refine=REFINE,
                      acoustic_cut=ACOUSTIC_CUT, acoustic_model="q2",
                      n_shell=N_SHELL, band_interp=BAND_INTERP,
                      states=[STATE])
    for it, T in enumerate(TEMPERATURES):
        code = float(np.asarray(res.total)[it, STATE])
        print(f"       T = {T:5.0f} K: code {code*1e3:8.3f} meV   closed form "
              f"on the same band {ref[T]*1e3:8.3f} meV   ratio "
              f"{code/max(ref[T], 1e-30):.2f}")
    print("   ratio near 1: the integration is sound. Above 1 at high T "
          "only: other channels\n   (absorption into states beyond the Gamma "
          "cell), which the closed form omits --\n   the default-model run "
          "measures them.")

print("\nBEST ESTIMATE (closed form, BSE curvature; range = C at shell 1 .. "
      "extrapolated to q->0)")
for T in TEMPERATURES:
    print(f"   T = {T:5.0f} K: {best[T][0]*1e3:.3f} - {best[T][1]*1e3:.3f} meV"
          f"   plus other channels (default-model run)")
