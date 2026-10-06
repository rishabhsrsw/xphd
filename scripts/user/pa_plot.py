"""
pa_plot.py
==========
Figure for the phonon-assisted radiative rate: the temperature dependence,
the mode decomposition, and the destructive interference between the two
bright intermediate states.

Numbers come from phonon_assisted.py; edit the block at the top to match a
given run rather than re-deriving them here.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HA = 27.211386245988; BOHR = 0.529177210903
C_AU = 137.035999084; AUT = 0.02418884326; KB = 8.617333262145e-5 / HA
ALPHA = 1.0 / C_AU

# ---- from the GaN run -------------------------------------------------
mu = np.array([0.27883, 0.29549, 0.0, 0.00719, 0.00690]) / BOHR   # bohr
Eg = np.array([3.7329, 3.7358, 4.2549, 4.3538, 4.3538]) / HA      # Ha
E_S = 3.3569 / HA                                                 # K' minimum
hw = np.array([16.22, 17.99, 28.69, 42.38, 93.77, 100.47]) * 1e-3 / HA

# |G| from K' state 1 to each Gamma state, per mode (meV)
Gmag = np.array([[55.932, 54.048, 0.008, 2.639, 0.343],
                 [0.012, 0.011, 19.898, 0.076, 0.104],
                 [0.003, 0.003, 0.000, 0.000, 0.002],
                 [0.000, 0.000, 0.001, 0.000, 0.000],
                 [0.014, 0.014, 0.000, 0.005, 0.001],
                 [0.004, 0.004, 0.000, 0.000, 0.000]]) * 1e-3 / HA

# phases of the mode-1 contributions, from the breakdown
arg1 = np.array([1.4825, -1.9045, -0.2685, 2.4301, 2.9928])
eta = 1e-3 / HA

def rate(T, coherent=True):
    out = np.zeros((len(T), len(hw)))
    for nu in range(len(hw)):
        w = hw[nu]
        terms = Eg * mu * Gmag[nu] / (Eg - E_S + w - 1j * eta) / C_AU
        if nu == 0:
            terms = np.abs(terms) * np.exp(1j * arg1)
        lam = np.sum(terms) if coherent else np.sum(np.abs(terms))
        lam_a = np.sum(Eg * mu * Gmag[nu] / (Eg - E_S - w - 1j * eta)) / C_AU
        for k, t in enumerate(T):
            N = 1.0 / np.expm1(min(w / (KB * t), 200.0))
            out[k, nu] = ((4/3) * ALPHA * (E_S - w) * abs(lam)**2 * (N + 1)
                          + (4/3) * ALPHA * (E_S + w) * abs(lam_a)**2 * N)
    return out


# --- Evaluate continuous line for drawing ---
T_line = np.linspace(5, 310, 200)
g_line = rate(T_line, True)
g_inc_line = rate(T_line, False)
tau_line = AUT / g_line.sum(axis=1) / 1e6            # ns
tau_inc_line = AUT / g_inc_line.sum(axis=1) / 1e6

# --- Evaluate specific discrete temperatures from terminal output ---
T_pts = np.array([10, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300], dtype=float)
g_pts = rate(T_pts, True)
tau_pts = AUT / g_pts.sum(axis=1) / 1e6              # ns

fig, ax = plt.subplots(1, 3, figsize=(13, 4))

# (a) lifetime vs T
# 1. Plot the continuous theoretical lines
ax[0].plot(T_line, tau_line / 1e3, lw=2, color="C0", label="with interference")
ax[0].plot(T_line, tau_inc_line / 1e3, lw=2, ls="--", color="C3", label="incoherent sum")

# 2. Scatter the exact discrete temperatures calculated by your run
ax[0].plot(T_pts, tau_pts / 1e3, 'o', color="C0", ms=5, zorder=5)

# 3. Annotate just the extrema and middle to keep it clean (10K, 150K, 300K)
for k in [0, 6, -1]: 
    ax[0].annotate(f"{int(T_pts[k])}K: {tau_pts[k]/1e3:.1f} $\\mu$s", 
                   (T_pts[k], tau_pts[k] / 1e3),
                   textcoords="offset points", xytext=(8, 4), fontsize=9)

ax[0].set_xlabel("Temperature (K)"); ax[0].set_ylabel(r"$\tau$ ($\mu$s)")
ax[0].set_title("(a) phonon-assisted lifetime")
ax[0].legend(fontsize=8, frameon=False); ax[0].grid(alpha=0.3, which="both")

# (b) mode decomposition
k77 = np.argmin(abs(T_line - 77))
frac = g_line[k77] / g_line[k77].sum()
cols = plt.cm.turbo(np.linspace(0.1, 0.9, len(hw)))
ax[1].bar(np.arange(1, 7), np.maximum(frac, 1e-12), color=cols, edgecolor="k", lw=0.6)
ax[1].set_yscale("log"); ax[1].set_ylim(1e-10, 2)
ax[1].set_xlabel("phonon mode"); ax[1].set_ylabel("fraction of rate")
ax[1].set_title("(b) mode decomposition, 77 K")
for i, f in enumerate(frac):
    if f > 1e-3:
        ax[1].text(i + 1, f * 1.4, f"{f:.0%}", ha="center", fontsize=9)
ax[1].text(0.03, 0.06, f"mode 1: {hw[0]*HA*1e3:.1f} meV", transform=ax[1].transAxes, fontsize=9)

# (c) interference on the complex plane
c = np.abs(Eg * mu * Gmag[0] / (Eg - E_S + hw[0] - 1j*eta) / C_AU) * np.exp(1j * arg1)
tot = c.sum()
ax[2].axhline(0, c="0.8", lw=0.6); ax[2].axvline(0, c="0.8", lw=0.6)
run = 0 + 0j
for i in (0, 1):
    ax[2].annotate("", xy=((run + c[i]).real, (run + c[i]).imag),
                   xytext=(run.real, run.imag),
                   arrowprops=dict(arrowstyle="->", lw=2, color=f"C{i}"))
    ax[2].text((run + c[i] / 2).real, (run + c[i] / 2).imag,
               f"  S'={i+1}", color=f"C{i}", fontsize=10)
    run = run + c[i]
ax[2].annotate("", xy=(tot.real, tot.imag), xytext=(0, 0),
               arrowprops=dict(arrowstyle="->", lw=2.5, color="k"))
ax[2].text(tot.real, tot.imag, "  sum", fontsize=10, fontweight="bold")
lim = max(abs(c).max(), abs(tot)) * 1.3
ax[2].set_xlim(-lim, lim); ax[2].set_ylim(-lim, lim)
ax[2].set_aspect("equal")
ax[2].set_xlabel(r"Re $\Lambda$"); ax[2].set_ylabel(r"Im $\Lambda$")
dphi = (arg1[0] - arg1[1]) / np.pi
ax[2].set_title("(c) interference, mode 1")
ax[2].text(0.03, 0.03,
           f"$\\Delta\\phi$ = {dphi:.3f}$\\pi$\n"
           f"suppression {(abs(c[0])+abs(c[1]))/abs(tot):.1f}$\\times$ "
           f"in amplitude\n{((abs(c[0])+abs(c[1]))/abs(tot))**2:.0f}$\\times$ in rate",
           transform=ax[2].transAxes, fontsize=9, va="bottom")

plt.tight_layout()
plt.savefig("phonon_assisted.png", dpi=300)
print("saved phonon_assisted.png")
print(f"  tau: 10 K {tau_pts[0]/1e3:.1f} us, "
      f"150 K {tau_pts[6]/1e3:.1f} us, "
      f"300 K {tau_pts[-1]/1e3:.1f} us")