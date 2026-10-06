"""
check_bte.py
============
Has the transport settled, and is where it settled right? The end state is
compared with the Boltzmann distribution on the SAME states at the run's
lattice temperature: the mean energy, the share at the minimum's Q-point
alone, and the share in each valley. The mean energy at several times shows
whether the run is still drifting.

Run it by editing the SETTINGS block below and running the file -- in VS
Code, Run Python File. No command line is needed.

At a finite temperature the equilibrium is NOT all at the minimum: on a
discrete mesh the first shell of neighbours can hold as much as the minimum's
own point once kT is comparable to the shell's energy. 'Settled' means the
end state matches Boltzmann, not that it sits on one point.

For GaN at 77 K: mean energy 4.26 meV against Boltzmann 4.09 meV. (A 26.6 meV
figure once quoted for 77 K was the 300 K Boltzmann value.)
"""
import numpy as np

from xphd.core.mesh import hex_norm

# ==========================================================================
# 0. SETTINGS -- edit these, then run this file (VS Code: Run Python File)
# ==========================================================================
BTE = "snaps_77K.npz"          # snapshots from xphd bte --snapshots
T_BTE = 77.0                   # the run's lattice temperature, K
T_CHECK = 30.0                 # also report the shares at this time (fs)
CENTERS = {"K": [1 / 3, 1 / 3], "K'": [2 / 3, 2 / 3]}   # hBN: {"Gamma": [0, 0]}
RADIUS = 0.12                  # 'in the valley': |Q - centre| below this, in |b|

# ==========================================================================
# 1. LOAD, AND THE BOLTZMANN TARGET ON THE SAME STATES
# ==========================================================================
d = np.load(BTE)
t, F, E, Q = d["t"], d["F"], d["E"], d["Q_red"]
kB = 8.617333e-5
dE = E - E.min()
wB = np.exp(-dE / (kB * T_BTE))
wB /= wB.sum()
iq_min = Q[np.argmin(E)]


def mean_E(w):
    return float((w * dE).sum() / w.sum()) * 1e3


print(f"{BTE}: {F.shape[1]} states, {len(t)} snapshots, t = {t[0]:.4g} .. {t[-1]:.4g} fs; "
      f"minimum at Q = {np.round(iq_min[:2], 4)}")

# ==========================================================================
# 2. IS IT STILL DRIFTING?
# ==========================================================================
print(f"\nmean energy above the minimum (Boltzmann at {T_BTE:.0f} K: {mean_E(wB):.2f} meV)")
for frac in (0.25, 0.5, 0.75, 1.0):
    i = min(int(round(frac * (len(t) - 1))), len(t) - 1)
    print(f"   t = {t[i]:9.4g} fs:  {mean_E(F[i]):8.2f} meV")
print(f"   Boltzmann at 300 K (contrast): {mean_E(np.exp(-dE / (kB * 300.0))):.2f} meV")

# ==========================================================================
# 3. WHERE IT SITS, AGAINST WHERE IT SHOULD
# ==========================================================================
masks = {"minimum's Q-point": hex_norm(Q - iq_min) < 1e-6}
masks.update({name: hex_norm(Q - np.array([*c, 0.0])) < RADIUS for name, c in CENTERS.items()})
times = [(T_CHECK, int(np.argmin(np.abs(t - T_CHECK)))), (t[-1], len(t) - 1)]
print(f"\nshare of the population          " + "".join(f"t = {tt:<9.4g}" for tt, _ in times)
      + f"Boltzmann {T_BTE:.0f} K")
for name, m in masks.items():
    row = "".join(f"{100 * F[i, m].sum() / F[i].sum():6.1f}%    " for _, i in times)
    print(f"   {name:<30}{row}{100 * wB[m].sum():6.1f}%")

print("\nVERDICT")
end, target = mean_E(F[-1]), mean_E(wB)
drift = abs(mean_E(F[-1]) - mean_E(F[max(len(t) - 1 - max(len(t) // 10, 1), 0)]))
if abs(end - target) <= max(0.1 * target, 1.0):
    print(f"   settled: the end state matches Boltzmann at {T_BTE:.0f} K "
          f"({end:.2f} against {target:.2f} meV)")
elif drift > 0.02 * max(end, 1e-9):
    print(f"   still relaxing: {end:.2f} meV against {target:.2f}, changing by {drift:.2f} meV over "
          f"the last tenth of the run -- run longer")
else:
    print(f"   stuck away from equilibrium: {end:.2f} meV against {target:.2f}, no longer changing --")
    print("   a bottleneck in the rates (check bte-check and the channels)")
