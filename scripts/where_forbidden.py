"""
where_forbidden.py
==================
What carries the 'forbidden' coupling?

With exact sigma_h a forbidden coupling is exactly zero, so forbidden weight
means a parity was attached to the wrong mode, or a state is not a pure
parity eigenstate. Three fingerprints:

  PHONON SWAP  the mode lies within GAP of an opposite-parity mode at that q
               (labels and couplings in different orders -- cannot happen
               with labels_from_elph.py, which uses g2's own order).
  STATE MIX    the final state has an opposite-parity neighbour within GAP,
               or |chi| < 0.95: an even and an odd state came back mixed.
  ADMIXTURE    the final state looks clean (|chi| close to 1) but carries a
               small opposite-parity part, eps = (1 - |chi|) / 2. A coupling
               into that part is classified forbidden, and is of order eps
               times a normal coupling. If this is the mechanism, the
               forbidden weight sits on states whose eps is far above the
               average eps of the states carrying the allowed weight.

Classifies every coupling exactly as `xphd selection-rule` does. Run it in
the same folder with the same files, by editing SETTINGS and running the file.
"""
import numpy as np

import xphd

# ==========================================================================
# 0. SETTINGS -- the same files `xphd selection-rule` used
# ==========================================================================
ARCHIVE = "GI_ExcPh_Q0001.npz"
PARITY = "parity.npz"
LABELS = "mode_labels_elph.npy"
STATE = 1                    # initial state (1-based)
TOL = 0.5                    # |chi| above this assigns a parity, as --tol
GAP = 1.0e-3                 # eV: 'nearly degenerate'
TOP = 12

# ==========================================================================
# 1. CLASSIFY, as xphd selection-rule does
# ==========================================================================
arc = xphd.ExcPhArchive(ARCHIVE)
g2 = arc.grid("g2")
n1, n2, nmod, nexc, _ = g2.shape
S = STATE - 1
qgrid = np.mod(arc.q_red, 1.0)
hw = arc.grid("hw")[arc._i, arc._j]                  # (nq, nmod), g2's order
Em = arc.grid("E_m")[arc._i, arc._j]                 # (nq, nexc)
g2f = g2[arc._i, arc._j][:, :, S, :]                 # (nq, nmod, nexc)

d = np.load(PARITY)
dd = qgrid[:, None, :2] - np.mod(d["Q_red"][:, :2], 1.0)[None, :, :]
dd -= np.rint(dd)
idx = np.argmin(np.linalg.norm(dd, axis=-1), axis=1)
if d["chi"].shape[1] < nexc:
    raise SystemExit(f"{PARITY} has {d['chi'].shape[1]} states, the archive {nexc}")
chi = d["chi"][idx][:, :nexc]
p_beta = np.sign(chi)
ok_beta = np.abs(chi) > TOL
eps = (1.0 - np.minimum(np.abs(chi), 1.0)) / 2.0      # opposite-parity admixture
lab = np.load(LABELS)
p_nu = np.where(np.isin(lab, [0, 3]), -1.0, 1.0)
ok_nu = lab >= 0
iG = int(np.argmin(np.linalg.norm(((qgrid[:, :2] + 0.5) % 1) - 0.5, axis=1)))
chi_lam = float(d["chi"][idx[iG], S])
p_lam = float(np.sign(chi_lam))
prod = p_nu[:, :, None] * p_beta[:, None, :]
good = ok_nu[:, :, None] & ok_beta[:, None, :]
forbid = good & (np.abs(prod + p_lam) < 0.5)
allowed = good & (np.abs(prod - p_lam) < 0.5)
Wf = np.where(forbid, g2f, 0.0)
Wa = np.where(allowed, g2f, 0.0)
tot_f, tot_a = Wf.sum(), Wa.sum()
print(f"{ARCHIVE}: {nmod} branches, {nexc} states; initial state {STATE}, "
      f"chi = {chi_lam:+.6f} (eps = {(1 - abs(chi_lam)) / 2:.1e})")
print(f"forbidden / allowed = {tot_f / max(tot_a, 1e-300):.2e}   (as xphd selection-rule, --tol {TOL})")

# ==========================================================================
# 2. FINGERPRINTS
# ==========================================================================
dw = np.abs(hw[:, :, None] - hw[:, None, :])
opp = (p_nu[:, :, None] * p_nu[:, None, :]) < 0
gap_ph = np.where(opp, dw, np.inf).min(axis=2)       # (nq, nmod)
de = np.abs(Em[:, :, None] - Em[:, None, :])
oppe = (p_beta[:, :, None] * p_beta[:, None, :]) < 0
gap_ex = np.where(oppe, de, np.inf).min(axis=2)      # (nq, nexc)

swap = np.broadcast_to(gap_ph[:, :, None] < GAP, Wf.shape)
mix = np.broadcast_to((gap_ex[:, None, :] < GAP) | (np.abs(chi)[:, None, :] < 0.95), Wf.shape)
f = lambda m: 100 * Wf[m].sum() / max(tot_f, 1e-300)
print(f"\nof the forbidden weight:")
print(f"   phonon within {GAP * 1e3:.0f} meV of an opposite-parity mode:   {f(swap):6.1f}%")
print(f"   final state near-degenerate or |chi| < 0.95:          {f(mix):6.1f}%")
print(f"   neither (clean-looking states):                       {f(~(swap | mix)):6.1f}%")

# ---- the admixture test --------------------------------------------------------
E3 = np.broadcast_to(eps[:, None, :], Wf.shape)
eps_f = (Wf * E3).sum() / max(tot_f, 1e-300)
eps_a = (Wa * E3).sum() / max(tot_a, 1e-300)
print(f"\nadmixture eps = (1 - |chi|)/2 of the final states, weighted by")
print(f"   the forbidden coupling: {eps_f:.2e}")
print(f"   the allowed coupling:   {eps_a:.2e}")
if max(eps_f, eps_a) < 1e-12:
    print("   the final states are pure parity eigenstates (eps = 0): the forbidden")
    print("   weight cannot be admixture leakage -- check the phonon labels and the")
    print("   initial state's parity")
else:
    print(f"   ratio {eps_f / max(eps_a, 1e-300):.1f}  -- far above 1 means the forbidden weight sits on")
    print(f"   states carrying an opposite-parity part (admixture leakage); near 1")
    print(f"   means it does not follow the admixture, and something else is wrong")

# ---- the sharper test: state by state, does forbidden/allowed follow eps? ----------
# For each final state at each q, r = F/A (forbidden over allowed coupling into
# it). Leakage through an admixture gives r of order eps: log r and log eps
# then correlate strongly and r/eps is of order 1. Wrong labels give r with no
# relation to eps.
F = Wf.sum(axis=1)                                     # (nq, nexc)
A = Wa.sum(axis=1)
use = (F > 0) & (A > 0) & (eps > 0)
on_pure = 100 * F[(eps <= 0)].sum() / max(tot_f, 1e-300)
print(f"\nstate by state (forbidden / allowed into each final state, r = F/A):")
print(f"   forbidden weight on states with eps = 0 exactly: {on_pure:5.1f}%")
if use.sum() >= 5:
    x, y = np.log10(eps[use]), np.log10(F[use] / A[use])
    corr = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
    med = float(np.median(F[use] / A[use] / eps[use]))
    share = 100 * F[use].sum() / max(tot_f, 1e-300)
    print(f"   {int(use.sum())} (q, state) pairs carrying {share:.1f}% of the forbidden weight:")
    print(f"   correlation of log r with log eps: {corr:+.2f};  median r/eps: {med:.2g}")
    print("   strong correlation with r/eps of order 1 -> admixture leakage; none -> not")
else:
    print("   too few states with both forbidden and allowed coupling to test")

# ==========================================================================
# 3. WHERE
# ==========================================================================
print(f"\nby phonon branch and final state (share of the forbidden weight):")
bm = Wf.sum(axis=0)
for nu, m in sorted(np.ndindex(bm.shape), key=lambda t: -bm[t])[:6]:
    if bm[nu, m] > 0:
        print(f"   nu {nu + 1}  m {m + 1}:  {100 * bm[nu, m] / tot_f:5.1f}%")

print(f"\ntop {TOP} entries")
print("   q (reduced)          nu  hw(meV)  gap_ph   m   E_m (eV)  gap_ex    chi_m      eps     share")
flat = np.argsort(Wf.ravel())[::-1][:TOP]
for i in flat:
    iq, nu, m = np.unravel_index(i, Wf.shape)
    if Wf[iq, nu, m] <= 0:
        break
    gp = gap_ph[iq, nu] * 1e3
    ge = gap_ex[iq, m] * 1e3
    print(f"   {str(np.round(qgrid[iq, :2], 4)):20} {nu + 1:2d}  {hw[iq, nu] * 1e3:7.2f}  "
          f"{'  none' if not np.isfinite(gp) else f'{gp:6.2f}'}  {m + 1:2d}   {Em[iq, m]:7.4f}  "
          f"{'  none' if not np.isfinite(ge) else f'{ge:6.2f}'}  {chi[iq, m]:+.5f}  {eps[iq, m]:.1e}  "
          f"{100 * Wf[iq, nu, m] / tot_f:5.1f}%")
# ==========================================================================
# 4. VERDICT
# ==========================================================================
# the --tol that would exclude the final states carrying 95% of the forbidden
# weight: sort the forbidden weight by the |chi| of its final state
absx = np.broadcast_to(np.abs(chi)[:, None, :], Wf.shape).ravel()
w = Wf.ravel()
order = np.argsort(absx)
cum = np.cumsum(w[order]) / max(tot_f, 1e-300)
t95 = float(absx[order][np.searchsorted(cum, 0.95)]) if tot_f > 0 else 0.0
mix_share, pure_share = f(mix), on_pure
print("\nVERDICT")
if tot_f <= 0:
    print("   no forbidden weight: the rule holds exactly")
elif pure_share > 50:
    print(f"   {pure_share:.0f}% of the forbidden weight goes into PURE states: not mixing --")
    print("   a parity label is wrong (phonon labels, or the initial state's)")
elif mix_share > 80:
    print(f"   STATE MIX: {mix_share:.1f}% of the forbidden weight sits on final states that are")
    print("   near-degenerate with an opposite-parity partner or have |chi| < 0.95 -- pairs the")
    print("   solver returned mixed. The selection rule itself is not the issue. To confirm,")
    print(f"   rerun xphd selection-rule with --tol {min(np.ceil((t95 + 0.002) * 100) / 100, 0.99):.2f}"
          f" (just above |chi| = {t95:.3f}, which")
    print("   excludes the states carrying 95% of it); forbidden/allowed should collapse.")
else:
    print("   no single mechanism dominates: see the state-by-state test and the table")
print("\nFor the worst entry, `xphd decompose-parity --iq <full-zone index> --state <m>` shows")
print("which k-points and bands carry the opposite-parity part of that state.")
