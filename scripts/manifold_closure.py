"""
manifold_closure.py
===================
Are the near-degenerate manifolds complete inside the states the archive
keeps?

`xphd selection-rule --rotate` splits the coupling by the parity eigenstates
of each manifold. That is exact only if a manifold holds every state it mixes
with: then its sigma_h eigenvalues are exactly +-1. A manifold whose partner
lies just above the archive's last state (state 10 near-degenerate with
state 11, say) is cut, its eigenvalues fall short of +-1, and part of the
coupling stays undefined -- which no rotation inside the archive can repair.

This compares, for every manifold that touches the archive's states, the
eigen-parities of the manifold CUT at the archive's state count with those of
the COMPLETE manifold. It needs a parity file written with more states than
the archive:

    xphd parity ... --nstates 16 -o parity16.npz

Run it by editing the SETTINGS block below and running the file -- in VS
Code, Run Python File. No command line is needed.

Reading it: if the complete manifolds are far cleaner than the cut ones, the
cut is the cause, and the archives should be generated with more states
(--nexc 16, say) so that every manifold touching the low states is whole.
If both are equally poor, the partners are farther apart than the manifold
tolerance: rerun xphd parity with a larger --manifold-tol.
"""
import numpy as np

# ==========================================================================
# 0. SETTINGS -- edit these, then run this file (VS Code: Run Python File)
# ==========================================================================
PARITY = "parity16.npz"        # from xphd parity --nstates N, N > NARCHIVE
NARCHIVE = 10                  # states per Q in the archive (--nexc)
CLEAN = 0.02                   # |mu| within this of 1 counts as a clean parity

# ==========================================================================
# 1. LOAD
# ==========================================================================
d = np.load(PARITY)
if "sigma_ibz" not in d.files:
    raise SystemExit(f"{PARITY} has no manifold matrices: rerun xphd parity "
                     f"with --manifold-tol (the default)")
S, G, chi = d["sigma_ibz"], d["group_ibz"], np.asarray(d["chi_ibz"])
nibz, ns = G.shape
if ns <= NARCHIVE:
    raise SystemExit(f"{PARITY} holds {ns} states per Q; it needs more than "
                     f"NARCHIVE = {NARCHIVE} to show what the cut removes")
print(f"{PARITY}: {nibz} irreducible Q, {ns} states each; archive keeps {NARCHIVE}")


def eig_parities(M, idx):
    B = M[np.ix_(idx, idx)]
    return np.linalg.eigvalsh(0.5 * (B + B.conj().T))


# ==========================================================================
# 2. CUT AGAINST COMPLETE MANIFOLDS
# ==========================================================================
cut, full, n_cut = [], [], 0
for M, g in zip(S, G):
    for lab in np.unique(g[:NARCHIVE]):
        inside = np.where(g[:NARCHIVE] == lab)[0]
        whole = np.where(g == lab)[0]
        cut.append(eig_parities(M, inside))
        full.append(eig_parities(M, whole))
        n_cut += int(len(whole) > len(inside))
cut, full = np.abs(np.concatenate(cut)), np.abs(np.concatenate(full))


def summary(mu):
    return (f"{100 * np.mean(np.abs(mu - 1) < CLEAN):5.1f}% within {CLEAN} of +-1, "
            f"{100 * np.mean(mu < 0.5):5.1f}% below 0.5")


print(f"\nmanifolds touching states 1..{NARCHIVE}: {n_cut} of them continue above state {NARCHIVE}")
print(f"   cut at {NARCHIVE} states : {summary(cut)}")
print(f"   complete ({ns} states): {summary(full)}")

# ==========================================================================
# 3. WHICH STATES ARE MIXED, BY INDEX
# ==========================================================================
c = np.abs(chi)
print(f"\nper state, share of irreducible Q with |chi| < 0.5 (mixed before any rotation):")
print("   " + "  ".join(f"{n + 1}:{100 * np.mean(c[:, n] < 0.5):.0f}%" for n in range(ns)))
top = NARCHIVE - 1
gap_up = None
if "E" in d.files:
    E = np.asarray(d["E"])
    iqi = np.asarray(d["iQ_ibz"])
    first = np.array([np.where(iqi == j)[0][0] for j in range(nibz)])
    Eibz = E[first]
    gap_up = (Eibz[:, top + 1] - Eibz[:, top]) * 1e3
    print(f"\ngap between state {NARCHIVE} and state {NARCHIVE + 1}, meV, over the irreducible Q:")
    print(f"   min {gap_up.min():.2f}   median {np.median(gap_up):.2f}   "
          f"below 5 meV at {int((gap_up < 5).sum())} of {nibz}")

print("\nVERDICT")
gain = np.mean(np.abs(full - 1) < CLEAN) - np.mean(np.abs(cut - 1) < CLEAN)
if gain > 0.05:
    print(f"   the cut at {NARCHIVE} states spoils {100 * gain:.0f}% of the eigen-parities: generate the")
    print(f"   archives with more states (as many as this file holds, {ns}) and rerun")
    print("   xphd parity with the same --nstates")
elif np.mean(np.abs(full - 1) < CLEAN) < 0.9:
    print("   complete manifolds are no cleaner: the partners lie farther apart than the")
    print("   manifold tolerance -- rerun xphd parity with a larger --manifold-tol")
else:
    print("   the manifolds are whole and clean: the rotation is complete")
