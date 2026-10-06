#!/usr/bin/env python3
"""
interference.py
===============
Separate excitons that are dark by SYMMETRY from those that are dark by
CANCELLATION, following the interference figure of merit of Acharya et al.,
arXiv:2607.08355.

    I_S = |sum_i A^S_i rho_i|^2 / sum_i |A^S_i rho_i|^2 ,
    rho^alpha_kvc = 2 p^alpha_kcv / (E_ck - E_vk)

I >> 1  the transition channels add in phase and the state is bright.
I << 1  the channels are individually large but cancel, and the state is
        dark despite being dipole allowed.
I ~ 1   neither.

The distinction matters because the two have different consequences. A
symmetry-forbidden state has an exactly vanishing amplitude and cannot be
brightened without breaking the symmetry. An interference-dark state has
large underlying channels held apart by phase, so anything that disturbs
the relative phase -- strain, a field, a phonon -- can light it up. In
CrSBr the second kind was seen in resonant inelastic x-ray scattering and
in phonon-driven transient reflectivity while remaining invisible in
steady-state spectra.

Two diagnostics here need no dipoles at all and are worth reading first:

  kPR    the k-space participation ratio, 1 / sum_k P_S(k)^2 with
         P_S(k) = sum_vc |A^S_kvc|^2. How many k-points carry the state.
  overlap  the Bhattacharyya overlap of P_S(k) between two states. If a
         bright state and a dark one share a k distribution above ~0.99,
         their brightness difference cannot be a population effect and
         must be internal phase.

Usage:
    python interference.py --bse ../BSE_EXCPH/output/ndb.BS_diago_Q1
    python interference.py --dipoles ../BSE_EXCPH/output/ndb.dipoles
"""
from __future__ import annotations

import argparse

import numpy as np
from ..yambo import lattice, load_excitons

HA2EV = 27.211386245988


def kpr(A):
    """k-space participation ratio per state: how many k-points carry it."""
    P = np.sum(np.abs(A) ** 2, axis=(2, 3))          # (nS, nk)
    P = P / np.maximum(P.sum(axis=1, keepdims=True), 1e-300)
    return 1.0 / np.maximum(np.sum(P ** 2, axis=1), 1e-300), P


def bhattacharyya(P, i, j):
    """Overlap of two k-space distributions. 1.0 means identical support."""
    return float(np.sum(np.sqrt(P[i] * P[j])))


def interference(A, rho):
    """I_S for every state, given rho with the same (nk, nv, nc) shape."""
    prod = A * rho[None]
    num = np.abs(prod.sum(axis=(1, 2, 3))) ** 2
    den = np.sum(np.abs(prod) ** 2, axis=(1, 2, 3))
    return num / np.maximum(den, 1e-300)


def main(argv=None):
    p = argparse.ArgumentParser(prog="xphd coherence")
    p.add_argument("--save", default="../LELPH/SAVE")
    p.add_argument("--bse", default="../BSE_EXCPH/output/ndb.BS_diago_Q1")
    p.add_argument("--dipoles", default=None,
                   help="ndb.dipoles. Without it only the k-space "
                        "diagnostics are computed, which already settle "
                        "whether a bright/dark pair shares a manifold.")
    p.add_argument("--pair", type=int, nargs=2, default=None,
                   help="two 1-based state indices to compare in detail")
    p.add_argument("--nstates", type=int, default=None)
    a = p.parse_args(argv)

    ylat = lattice(a.save)
    exc, A, E = load_excitons(a.bse, ylat)
    if a.nstates:
        A, E = A[:a.nstates], E[:a.nstates]
    nS, nk, nv, nc = A.shape
    print(f"{a.bse}\n  {nS} states, {nk} k-points, {nv}v x {nc}c")

    mu = np.abs(np.asarray(getattr(exc, "r_residual",
                                   np.zeros(nS)))) ** 2
    if mu.ndim > 1:
        mu = mu.sum(axis=0)
    mu = mu[:nS]

    # ---- k-space diagnostics, no dipoles needed ----------------------
    pr, P = kpr(A)
    print(f"\nK-SPACE STRUCTURE")
    print(f"   {'state':>5} {'E (eV)':>9} {'|mu|^2':>11} {'kPR':>7}"
          f" {'overlap with state 1':>21}")
    for s in range(nS):
        ov = bhattacharyya(P, 0, s)
        print(f"   {s+1:5d} {E[s]:9.4f} {mu[s]:11.3e} {pr[s]:7.1f}"
              f" {ov:21.4f}")
    print("   kPR counts the k-points carrying the state; the overlap is the")
    print("   Bhattacharyya coefficient against state 1. A value above ~0.99")
    print("   means the two states are built from the SAME transitions, so a")
    print("   brightness difference between them cannot be a population")
    print("   effect and must come from internal phase.")

    # ---- the interference figure of merit ----------------------------
    if not a.dipoles:
        print("\n   [no --dipoles given] The figure of merit needs the "
              "momentum\n   matrix elements. Pass ndb.dipoles to compute it.")
        return

    try:
        from yambopy import YamboDipolesDB
        dip = YamboDipolesDB(ylat, save=a.dipoles.rsplit("/", 1)[0],
                             filename=a.dipoles.rsplit("/", 1)[-1])
        pmat = np.asarray(dip.dipoles)      # convention varies by version
    except Exception as e:
        print(f"\n   could not read {a.dipoles}: {e}")
        print("   The array wanted is p^alpha_kcv with shape "
              "(3, nk, nc, nv) or\n   (nk, 3, nc, nv); check what your "
              "yambopy version returns and\n   reshape before calling "
              "interference().")
        return
    print(f"\n   dipoles {pmat.shape}")

    # quasiparticle energies for the denominator of rho
    try:
        Ec = np.asarray(exc.eigenvalues_c)
        Ev = np.asarray(exc.eigenvalues_v)
    except AttributeError:
        print("   [warn] per-band quasiparticle energies not exposed by this")
        print("   yambopy version; rho cannot be built without them.")
        return

    print(f"\nINTERFERENCE FIGURE OF MERIT")
    print(f"   {'state':>5} {'E (eV)':>9} {'|mu|^2':>11} "
          + " ".join(f"{'I_' + x:>10}" for x in "xyz"))
    Is = {}
    for ax, name in enumerate("xyz"):
        pa = pmat[ax] if pmat.shape[0] == 3 else pmat[:, ax]
        rho = 2.0 * pa / np.maximum(Ec[:, :, None] - Ev[:, None, :], 1e-6)
        Is[name] = interference(A, rho.transpose(0, 2, 1))
    for s in range(nS):
        print(f"   {s+1:5d} {E[s]:9.4f} {mu[s]:11.3e} "
              + " ".join(f"{Is[x][s]:10.3e}" for x in "xyz"))

    print("""
   Reading the table:
     I >> 1 with large |mu|^2   bright, channels in phase
     I << 1 with large channels interference-dark: dipole allowed but
                                cancelling, and brightenable by anything
                                that disturbs the relative phase
     |mu|^2 at machine zero     symmetry-forbidden; no phase argument
                                applies and no perturbation preserving the
                                symmetry can light it up

   The second and third are physically different and are worth separating
   in any table of oscillator strengths.""")

    if a.pair:
        i, j = [x - 1 for x in a.pair]
        print(f"\nPAIR {i+1} and {j+1}")
        print(f"   energies    {E[i]:.4f}, {E[j]:.4f} eV "
              f"(split {(E[j]-E[i])*1e3:+.1f} meV)")
        print(f"   |mu|^2      {mu[i]:.3e}, {mu[j]:.3e}   "
              f"ratio {mu[i]/max(mu[j],1e-300):.2e}")
        print(f"   kPR         {pr[i]:.1f}, {pr[j]:.1f}")
        print(f"   k-overlap   {bhattacharyya(P, i, j):.4f}")
        for x in "xyz":
            print(f"   I_{x}         {Is[x][i]:.3e}, {Is[x][j]:.3e}")
        if bhattacharyya(P, i, j) > 0.99:
            print("   -> the two draw from the same k-space manifold, so the")
            print("      brightness contrast is an internal-phase effect")


if __name__ == "__main__":
    main()
