"""
helicity_pl.py
==============
Phonon-assisted luminescence resolved by circular polarisation AND by the
valley of the emitting exciton.

Run it by editing the SETTINGS block below and running the file -- in VS
Code, Run Python File. No command line is needed.

Why valley resolution matters
-----------------------------
Symmetry predicts that the sideband from the K valley has one helicity and
the sideband from K' the other. Summed over both valleys, time reversal makes
the two helicities equal, so an ordinary PL spectrum is unpolarised whatever
the physics. The prediction can only be tested valley by valley.

Why one BSE is enough
---------------------
Circular polarisation lives in the relative phase between the x and y dipoles
of the SAME state. Dipoles from two separate BSE runs have independent phases,
and combining them gives |T_x|^2 + |T_y|^2 for both helicities. Here the
complex x and y dipoles come from `xphd dipoles`, which contracts the
single-particle ndb.dipoles with one set of eigenvectors -- the same Q = 0
eigenvectors as the archive's initial states.

What is computed
----------------
For each emitter beta at q, branch nu and phonon emission (+) or absorption
(-), the amplitude to emit light of polarisation p is the coherent sum over
all Q = 0 states lambda,

    a_p = sum_lambda  T^p_lambda  g_{lambda beta nu}(q)
                      / (E_beta(q) -+ hbar w_nu(q) - E_lambda + i eta),

with g the complex conjugate of the archive's G[q, nu, lambda, beta] (the
reverse of the process stored there), T^{+-} = (T_x +- i T_y)/sqrt2, and the
intensity |a_p|^2 N_beta(T_exc) (N_nu + 1/2 +- 1/2) placed at
E_beta(q) -+ hbar w_nu(q). Emitters in different states and phonons of
different branches add incoherently.

Checks it prints
----------------
- I+ + I- equals I_x + I_y at every energy (an identity: a failure means a
  bug);
- the K and K' valleys give equal total intensity and opposite polarisation
  (time reversal);
- the degree of circular polarisation P = (I+ - I-)/(I+ + I-) of each
  valley at the satellite, and which branches carry each helicity.

Which helicity is called sigma+ depends on the phase conventions of the
dipoles and the orientation of C3; |P| and the opposite signs at K and K'
do not.
"""
import numpy as np
import matplotlib.pyplot as plt

import xphd
from xphd.core.mesh import hex_norm

# ==========================================================================
# 0. SETTINGS -- edit these, then run this file (VS Code: Run Python File)
# ==========================================================================
# the Gamma archive: its initial states are the Q = 0 excitons
ARCHIVE = "GI_ExcPh_Q0001.npz"

# complex exciton dipoles (3, n) from `xphd dipoles`, same BSE as above
DIPOLES = "exc_dipoles.npy"

# lattice and exciton temperatures (K)
T_LATTICE = 10.0
T_EXC = 10.0

# Gaussian broadening of each line (eV, standard deviation), and the small
# imaginary part in the energy denominator (eV)
SIGMA = 0.005
ETA = 0.010

# phonons below this energy are left out (eV)
ACOUSTIC_CUT = 5e-4

# an emitter belongs to a valley if it lies within this of K or K' (|b|)
VALLEY_RADIUS = 0.12

# photon-energy grid, relative to the lowest emitter (eV)
OMEGA_RANGE = [-0.15, 0.05]
N_OMEGA = 2001

# half-width of the window around the satellite peak used for P (eV)
PEAK_HALF_WIDTH = 0.004

FIGSIZE = [8.0, 3.2]
DPI = 300
OUT = "helicity_pl.pdf"
NPZ_OUT = "helicity_pl.npz"
SHOW = True

# ==========================================================================
# 1. LOAD
# ==========================================================================
KB = 8.617333262e-5
arc = xphd.ExcPhArchive(ARCHIVE)
ii, jj = arc._i, arc._j
G = arc.grid("G")[ii, jj]                           # (nq, nmod, n_init, n_fin)
hw = arc.grid("hw")[ii, jj]                         # (nq, nmod)
Em = arc.grid("E_m")[ii, jj]                        # (nq, ne): emitters at q
En = np.asarray(arc.E_n, float)                     # (ne,): states at Q = 0
q = np.mod(np.asarray(arc.q_red, float), 1.0)
nq, nmod, ne, _ = G.shape

D = np.load(DIPOLES)
if D.shape[0] != 3:
    D = D.T
if D.shape[1] < ne:
    raise SystemExit(f"{DIPOLES} has {D.shape[1]} states; the archive needs {ne}")
Tx, Ty = D[0, :ne], D[1, :ne]
T = {"+": (Tx + 1j * Ty) / np.sqrt(2), "-": (Tx - 1j * Ty) / np.sqrt(2),
     "x": Tx, "y": Ty}
print(f"{ARCHIVE}: {nq} q-points, {nmod} branches, {ne} states; "
      f"dipoles for {D.shape[1]} states")
bright = np.argsort(np.abs(Tx) ** 2 + np.abs(Ty) ** 2)[::-1][:4]
print("   brightest Q = 0 states (in-plane |T|^2):  "
      + ",  ".join(f"{n+1}: {abs(Tx[n])**2 + abs(Ty[n])**2:.2e}" for n in bright))

# ==========================================================================
# 2. AMPLITUDES AND SPECTRA
# ==========================================================================
Emin = float(Em.min())
omega = Emin + np.linspace(*OMEGA_RANGE, N_OMEGA)
K, Kp = np.array([1 / 3, 1 / 3, 0.0]), np.array([2 / 3, 2 / 3, 0.0])
inK = hex_norm(q - K) < VALLEY_RADIUS
inKp = hex_norm(q - Kp) < VALLEY_RADIUS
keep_q = hex_norm(q) > 1e-8                          # q = 0 is direct emission
print(f"   emitters: {inK.sum()} q-points in the K valley, {inKp.sum()} in K'")

g = np.conj(G)                                       # emitter at q -> state at Q = 0
Nph = 1.0 / np.expm1(np.maximum(hw, 1e-12) / (KB * T_LATTICE))
Nexc = np.exp(-(Em - Emin) / (KB * T_EXC))           # (nq, ne)
live = (hw > ACOUSTIC_CUT) & keep_q[:, None]         # (nq, nmod)

spec = {v: {p: np.zeros(N_OMEGA) for p in T} for v in ("K", "Kp", "all")}
branch = {v: {p: np.zeros(nmod) for p in ("+", "-")} for v in ("K", "Kp")}
norm = 1.0 / (SIGMA * np.sqrt(2 * np.pi))
for s, fac in ((+1, Nph + 1.0), (-1, Nph)):          # emission, absorption
    e_ph = Em[:, None, :] - s * hw[:, :, None]       # (nq, nmod, ne): photon energy
    den = e_ph[:, :, None, :] - En[None, None, :, None] + 1j * ETA   # (q, nu, lam, beta)
    w = Nexc[:, None, :] * fac[:, :, None] * live[:, :, None]      # (q, nu, beta)
    for p, Tp in T.items():
        a = np.einsum("l,qnlb->qnb", Tp, g / den)    # coherent sum over lambda
        I = np.abs(a) ** 2 * w                       # (q, nu, beta)
        for v, mask in (("K", inK), ("Kp", inKp), ("all", np.ones(nq, bool))):
            sel = mask[:, None, None] & (I > 0)
            if not sel.any():
                continue
            e, wt = e_ph[sel], I[sel]
            spec[v][p] += norm * (wt[:, None] * np.exp(
                -0.5 * ((omega[None, :] - e[:, None]) / SIGMA) ** 2)).sum(0)
            if s == +1 and v in branch and p in branch[v]:
                branch[v][p] += np.where(mask[:, None, None], I, 0).sum(axis=(0, 2))

# ==========================================================================
# 3. CHECKS AND RESULTS
# ==========================================================================
tot = spec["all"]["+"] + spec["all"]["-"]
lin = spec["all"]["x"] + spec["all"]["y"]
print(f"\n1. Sum rule I+ + I- = I_x + I_y: largest relative difference "
      f"{np.abs(tot - lin).max() / max(lin.max(), 1e-300):.1e}"
      "   (must be ~1e-15)")

IK = spec["K"]["+"] + spec["K"]["-"]
IKp = spec["Kp"]["+"] + spec["Kp"]["-"]
print(f"2. Time reversal: integrated K / K' intensity = "
      f"{IK.sum() / max(IKp.sum(), 1e-300):.4f}   (must be ~1)")

below = omega < Emin
ipk = int(np.argmax(np.where(below, IK + IKp, 0)))
win = np.abs(omega - omega[ipk]) < PEAK_HALF_WIDTH
print(f"\n3. Satellite at {omega[ipk]:.4f} eV, "
      f"{(Emin - omega[ipk]) * 1e3:.1f} meV below the lowest emitter")
for v, lab in (("K", "K "), ("Kp", "K'")):
    Ip, Im = spec[v]["+"][win].sum(), spec[v]["-"][win].sum()
    P = (Ip - Im) / max(Ip + Im, 1e-300)
    print(f"   valley {lab}:  P = (I+ - I-)/(I+ + I-) = {P:+.4f}")
print("   symmetry predicts |P| ~ 1 with opposite signs at K and K';"
      " which one is 'sigma+' is a convention")

print("\n4. Phonon emission from the K valley, all energies")
ref = max(branch["K"]["+"].sum() + branch["K"]["-"].sum(), 1e-300)
print("   share of the valley's emission:  "
      + "   ".join(f"sigma{p} {branch['K'][p].sum() / ref:.2e}" for p in ("+", "-")))
print("   by branch, within each helicity:")
for p in ("+", "-"):
    b = branch["K"][p]
    if b.sum() < 1e-10 * ref:
        print(f"   sigma{p}:  no intensity")
        continue
    print(f"   sigma{p}:  " + "  ".join(f"nu{n+1} {100*b[n]/b.sum():6.2f}%"
                                      for n in range(nmod)))

np.savez(NPZ_OUT, omega=omega, **{f"I_{v}_{'p' if p == '+' else 'm' if p == '-' else p}":
                                  spec[v][p] for v in spec for p in spec[v]})
print(f"\nwrote {NPZ_OUT}")

# ==========================================================================
# 4. PLOT
# ==========================================================================
x = (omega - Emin) * 1e3
fig, axs = plt.subplots(1, 3, figsize=tuple(FIGSIZE), sharey=True)
scale = max((spec["all"]["+"] + spec["all"]["-"]).max(), 1e-300)
for ax, v, title in ((axs[0], "K", "K valley"), (axs[1], "Kp", "K$'$ valley"),
                     (axs[2], "all", "both valleys")):
    ax.plot(x, spec[v]["+"] / scale, color="tab:red", lw=1.4, label=r"$\sigma^+$")
    ax.plot(x, spec[v]["-"] / scale, color="tab:blue", lw=1.4, ls="--",
            label=r"$\sigma^-$")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(r"$\hbar\omega - E_{\min}$ (meV)")
    ax.tick_params(direction="in")
axs[0].set_ylabel("PL intensity (arb. units)")
axs[0].legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(OUT, dpi=DPI)
print(f"saved {OUT}")

# ==========================================================================
# show the figure -- set SHOW = False in SETTINGS to only save it
# ==========================================================================
if SHOW:
    plt.show()
