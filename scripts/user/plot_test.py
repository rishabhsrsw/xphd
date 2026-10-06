import numpy as np

d = np.load("coupling_by_parity.npz")
Ge, Go = d["G_even"].ravel(), d["G_odd"].ravel()
print(f"  even: sum {Ge.sum():.3e}  max {Ge.max():.3e}  "
      f"median {np.median(Ge):.3e}")
print(f"  odd : sum {Go.sum():.3e}  max {Go.max():.3e}  "
      f"median {np.median(Go):.3e}")
print(f"  max/median ratio, even: {Ge.max()/max(np.median(Ge),1e-30):.1e}")
print(f"  nonzero fraction, odd: {100*np.mean(Go > 0):.1f}%")