"""
Usage: set the inputs in the SETTINGS block below, then run

    python valley_times.py
"""
import numpy as np

from xphd.core.mesh import hex_norm

# ==========================================================================
# 0. SETTINGS -- edit these, then run this file (VS Code: Run Python File)
# ==========================================================================
SNAPS = "snaps_77K.npz"          # snapshots from xphd bte --snapshots
CENTERS = {"K": [1 / 3, 1 / 3], "K'": [2 / 3, 2 / 3]}
RADIUS = 0.12                    # 'in the valley': |Q - centre| below this, in |b|
TARGETS = [0.5, 0.9, 0.99]

# ==========================================================================
# 1. VALLEY FRACTIONS AGAINST TIME
# ==========================================================================
d = np.load(SNAPS)
t, F, Q = d["t"], d["F"], d["Q_red"]
tot = F.sum(axis=1)
masks = {name: hex_norm(Q - np.array([*c, 0.0])) < RADIUS for name, c in CENTERS.items()}
masks["all of them"] = np.any(np.stack(list(masks.values())), axis=0)

for name, m in masks.items():
    frac = F[:, m].sum(axis=1) / tot
    print(f"\n{name}  (final share {100 * frac[-1]:.1f}%)")
    for target in TARGETS:
        i = int(np.argmax(frac >= target))
        if frac[i] >= target:
            print(f"   {target:4.0%} by t = {t[i]:8.1f} fs")
        else:
            print(f"   {target:4.0%} never reached before t = {t[-1]:.0f} fs")
