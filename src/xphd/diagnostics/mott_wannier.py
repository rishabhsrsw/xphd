#!/usr/bin/env python3
"""
mott_wannier.py
===============
Can a 2D Mott-Wannier model predict the BRIGHT-DARK EXCITON SPLITTING from
quantities available after an nscf calculation?

The splitting decomposes as

    dE = (Eg_dir - Eg_ind) - (Eb_dir - Eb_ind)

The first bracket is a valley energy difference, free from the band
structure. The second needs a binding energy for each electron-hole valley
pair, and that is what this script estimates.

Binding energies come from the Rytova-Keldysh problem, which is the standard
2D screened hydrogen model:

    [ -hbar^2/(2 mu) grad^2  +  V(r) ] psi = -Eb psi

    V(r) = -(pi / (2 r0)) [ H_0(r/r0) - Y_0(r/r0) ]        (atomic units)

with r0 = 2 pi alpha_2D the screening length, H_0 the Struve function and
Y_0 the Bessel function of the second kind. As r0 -> 0 this reduces to 2D
hydrogen with Eb = 2 mu (atomic units), a limit the script checks.

Inputs per valley pair: the reduced mass mu, and one alpha_2D for the
material. Everything else is arithmetic.

Usage:
    python mott_wannier.py                 # runs the self-test and the table
"""
from __future__ import annotations

import numpy as np
from scipy.special import struve, y0

HARTREE = 27.211386245988          # eV
BOHR = 0.529177210903              # Angstrom


def keldysh_binding(mu, alpha_2d_ang, eps_env=1.0, rmax=600.0, n=15000,
                    n_state=0):
    """Ground-state binding energy in eV for the Rytova-Keldysh problem.

    mu           : reduced mass in units of the electron mass
    alpha_2d_ang : 2D polarizability in Angstrom
    eps_env      : dielectric constant of the surrounding medium (1 = vacuum)
    rmax, n      : radial grid in Bohr; rmax must exceed the exciton radius

    Solved on a radial grid for the l=0 channel. The substitution
    psi = u/sqrt(r) turns the 2D radial equation into

        -1/(2 mu) [ u'' + u/(4 r^2) ] + V u = E u

    which is a standard symmetric eigenvalue problem in u.
    """
    r0 = 2.0 * np.pi * (alpha_2d_ang / BOHR) / eps_env      # Bohr
    r = np.linspace(rmax / n, rmax, n)
    h = r[1] - r[0]

    x = np.maximum(r / r0, 1e-12)
    V = -(np.pi / (2.0 * r0 * eps_env)) * (struve(0, x) - y0(x))

    # -1/(2mu) d2/dr2 with the 1/(4r^2) centrifugal-like term, plus V
    main = 1.0 / (mu * h ** 2) - 1.0 / (8.0 * mu * r ** 2) + V
    off = -0.5 / (mu * h ** 2) * np.ones(n - 1)
    from scipy.linalg import eigh_tridiagonal
    w = eigh_tridiagonal(main, off, select='i',
                         select_range=(0, max(n_state, 0)))[0]
    return -w[n_state] * HARTREE


def reduced_mass(m_e, m_h):
    """1/mu = 1/m_e + 1/m_h, masses in electron-mass units."""
    return 1.0 / (1.0 / m_e + 1.0 / m_h)


def alpha_from_eps(eps_inp, L_ang):
    """2D polarizability from a slab dielectric constant.

    alpha_2D = (eps_slab - 1) L / (4 pi), with L the cell height used in the
    slab calculation. This is the standard slab-to-2D conversion; it assumes
    the vacuum is thick enough that the slab response is converged.
    """
    return (eps_inp - 1.0) * L_ang / (4.0 * np.pi)


def selftest():
    """Validate against published Keldysh results, not against 2D hydrogen.

    The r0 -> 0 limit is NOT 2D hydrogen: as r0 shrinks the potential near
    the origin goes as (1/r0) ln(r0/r), a well whose depth diverges, and the
    true ground state collapses into it rather than approaching the Coulomb
    result. Testing that limit measures the grid, not the solver.
    """
    print("SELF-TEST -- against published Keldysh numbers")
    print("  Berkelbach, Hybertsen & Reichman, PRB 88, 045318 (2013) treat")
    print("  freestanding MoS2 with r0 = 41.5 Ang and mu near 0.25, and")
    print("  report a ground-state binding energy of roughly 0.54 eV.\n")
    for mu in (0.22, 0.25, 0.30):
        print(f"    mu={mu:.2f}, alpha_2D=6.6 Ang: "
              f"Eb = {keldysh_binding(mu, 6.6, rmax=600, n=15000):.4f} eV")
    print("\n  environment screening reduces it as it should (mu = 0.25):")
    for eps in (1.0, 2.4, 4.5):
        print(f"    eps_env={eps:.1f}: "
              f"Eb = {keldysh_binding(0.25, 6.6, eps_env=eps, rmax=600, n=15000):.4f} eV")
    print("\n  grid convergence at realistic r0:")
    for rmax, n in ((400, 4000), (600, 15000), (800, 25000)):
        print(f"    rmax={rmax} n={n}: "
              f"{keldysh_binding(0.25, 6.6, rmax=rmax, n=n):.4f} eV")


def splitting(name, Eg_dir, Eg_ind, mu_dir, mu_ind, alpha, dE_bse=None):
    """Predicted bright-dark splitting, and the comparison if given."""
    Eb_dir = keldysh_binding(mu_dir, alpha)
    Eb_ind = keldysh_binding(mu_ind, alpha)
    pred = (Eg_dir - Eg_ind) - (Eb_dir - Eb_ind)
    print(f"\n{name}")
    print(f"  Eg  direct {Eg_dir:.4f}  indirect {Eg_ind:.4f}  "
          f"-> dEg  {(Eg_dir-Eg_ind)*1e3:+8.1f} meV")
    print(f"  Eb  direct {Eb_dir:.4f}  indirect {Eb_ind:.4f}  "
          f"-> dEb  {(Eb_dir-Eb_ind)*1e3:+8.1f} meV")
    print(f"  predicted splitting            {pred*1e3:+8.1f} meV")
    if dE_bse is not None:
        print(f"  BSE splitting                  {dE_bse*1e3:+8.1f} meV")
        print(f"  error                          {(pred-dE_bse)*1e3:+8.1f} meV"
              f"   <- want |error| < 50 meV for the descriptor to be viable")
    return pred


def main(argv=None):
    """Self-test with no numbers; otherwise predict one material's splitting.

    The five inputs come from your own calculations:
      Eg_dir, Eg_ind   quasiparticle gaps at the direct and indirect
                       transitions (eV)
      mu_dir, mu_ind   reduced masses of the direct and indirect pairs --
                       from a parabolic fit to each band edge
                       (xphd effective-mass), geometric mean over two
                       in-plane directions
      alpha            2D polarisability in Angstrom; --eps and --L convert
                       a slab dielectric constant instead

    The indirect exciton's electron and hole sit in DIFFERENT valleys, which
    is why the two binding energies differ.
    """
    import argparse
    p = argparse.ArgumentParser(prog="xphd mott-wannier",
                                description=main.__doc__.split("\n\n")[0])
    p.add_argument("--name", default="material")
    p.add_argument("--Eg-dir", type=float)
    p.add_argument("--Eg-ind", type=float)
    p.add_argument("--mu-dir", type=float)
    p.add_argument("--mu-ind", type=float)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--alpha", type=float, help="2D polarisability, Angstrom")
    g.add_argument("--eps", type=float,
                   help="slab dielectric constant; needs --L")
    p.add_argument("--L", type=float, help="cell height in Angstrom, for --eps")
    p.add_argument("--dE-bse", type=float, default=None,
                   help="the BSE bright-dark splitting, to compare against")
    a = p.parse_args(argv)

    need = (a.Eg_dir, a.Eg_ind, a.mu_dir, a.mu_ind)
    if all(x is None for x in need) and a.alpha is None and a.eps is None:
        selftest()
        print("\n  Pass --Eg-dir --Eg-ind --mu-dir --mu-ind and --alpha")
        print("  (or --eps with --L) to predict a splitting.")
        return
    if any(x is None for x in need):
        raise SystemExit("need all of --Eg-dir --Eg-ind --mu-dir --mu-ind")
    if a.alpha is None:
        if a.eps is None or a.L is None:
            raise SystemExit("give --alpha, or --eps together with --L")
        a.alpha = alpha_from_eps(a.eps, a.L)
        print(f"  alpha from eps = {a.eps}, L = {a.L} A: {a.alpha:.3f} A")
    splitting(a.name, a.Eg_dir, a.Eg_ind, a.mu_dir, a.mu_ind, a.alpha,
              a.dE_bse)
    print("""
  |error| < 50 meV across materials -> the descriptor works and the
  splitting can be had from an nscf. Errors of hundreds of meV -> the
  binding energies are not captured by a one-parameter model. A large but
  CONSISTENT error may still rank materials correctly, which is all a
  screen needs.""")


if __name__ == "__main__":
    main()
