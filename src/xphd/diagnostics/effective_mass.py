#!/usr/bin/env python3
"""
effective_mass.py
=================
Effective masses from a parabolic fit to E(k) near a band extremum, and the
reduced masses the Mott-Wannier model needs.

    E(k) = E0 + hbar^2 k^2 / (2 m*)      ->      m*/m_e = 3.80998 / a

with a the quadratic coefficient in eV.Ang^2. Masses in a 2D crystal are
anisotropic, so fit along TWO orthogonal in-plane directions and take the
geometric mean sqrt(m1 m2); that is the density-of-states mass, which is the
right one for a radial hydrogenic problem.

The fit window matters more than anything else here. Too wide and the band
is no longer parabolic; too narrow and the fit is dominated by numerical
noise in the eigenvalues. `mass_from_band` scans the window and reports how
much the answer moves, so the number comes with its own error bar.

Usage:
    python effective_mass.py bands.gnu --band 12 --center 0.0
"""
from __future__ import annotations

import argparse
import numpy as np

HBAR2_2M = 3.8099821161      # hbar^2 / (2 m_e), eV.Ang^2


def mass_from_band(k, E, k0=None, windows=(0.02, 0.03, 0.05, 0.08)):
    """Effective mass in electron-mass units, with a window scan.

    k : (n,) 1/Ang, monotonic along one direction through the extremum
    E : (n,) eV
    k0: the extremum position; taken as the min or max of E if omitted

    Returns (mass, spread) where spread is the range over the fit windows --
    a small spread means the parabolic region is well resolved, a large one
    means the band is not parabolic over the range sampled.
    """
    k = np.asarray(k, float)
    E = np.asarray(E, float)
    if k0 is None:
        k0 = k[np.argmin(E)] if (E[0] > E[len(E) // 2]) else k[np.argmax(E)]
    dk = k - k0
    # a conduction valley curves up, a valence valley curves down
    sign = 1.0 if E[np.argmin(np.abs(dk))] < 0.5 * (E[0] + E[-1]) else -1.0

    out = []
    for w in windows:
        m = np.abs(dk) <= w
        if m.sum() < 4:
            continue
        a = np.polyfit(dk[m], sign * E[m], 2)[0]
        if a > 0:
            out.append(HBAR2_2M / a)
    if not out:
        raise SystemExit("no usable fit window; is k0 at the extremum?")
    out = np.array(out)
    return float(np.median(out)), float(out.max() - out.min())


def dos_mass(m1, m2):
    """Geometric mean of the two principal masses."""
    return float(np.sqrt(m1 * m2))


def reduced(m_e, m_h):
    """1/mu = 1/m_e + 1/m_h. Both positive; use |m_h| for the valence band."""
    return 1.0 / (1.0 / abs(m_e) + 1.0 / abs(m_h))


def report(tag, m1, s1, m2, s2):
    m = dos_mass(m1, m2)
    print(f"  {tag:>22}: m1 = {m1:.4f} (+-{s1:.4f})   "
          f"m2 = {m2:.4f} (+-{s2:.4f})   DOS mass {m:.4f}")
    if max(s1 / max(m1, 1e-9), s2 / max(m2, 1e-9)) > 0.15:
        print(f"  {'':>22}  [warn] the window scan moves by more than 15%; "
              f"the band is not\n  {'':>22}  parabolic over this range, or "
              f"the k-sampling is too coarse")
    return m


def main(argv=None):
    p = argparse.ArgumentParser(prog="xphd effective-mass")
    p.add_argument("gnu", nargs="?",
                   help="bands.x .gnu output: two columns, k and E, blank "
                        "line between bands")
    p.add_argument("--band", type=int, default=None,
                   help="1-based band index to fit")
    p.add_argument("--center", type=float, default=None,
                   help="k of the extremum along this segment")
    a = p.parse_args(argv)

    if a.gnu:
        blocks, cur = [], []
        for line in open(a.gnu):
            if line.strip():
                cur.append([float(x) for x in line.split()[:2]])
            elif cur:
                blocks.append(np.array(cur))
                cur = []
        if cur:
            blocks.append(np.array(cur))
        print(f"{a.gnu}: {len(blocks)} bands, {len(blocks[0])} k-points each")
        if a.band is None:
            raise SystemExit("pass --band to choose one")
        b = blocks[a.band - 1]
        m, s = mass_from_band(b[:, 0], b[:, 1], a.center)
        print(f"  band {a.band}: m* = {m:.4f} +- {s:.4f}")
        return

    # ------------------------------------------------------------------
    print(__doc__)
    print("""
WHICH VALLEYS, FOR WHICH EXCITON

  The exciton centre-of-mass momentum is Q = k_e - k_h.

    Q = 0    the BRIGHT exciton: electron and hole in the SAME valley, so
             m_e and m_h are both taken there
    Q = K    the DARK exciton: electron and hole in valleys separated by K,
             so m_e comes from one and m_h from the other

  That difference is the entire reason the two binding energies differ, and
  it is what the descriptor has to capture. Using one reduced mass for both
  guarantees the model predicts zero splitting from the binding term.

WORKED SHAPE (fill in your own numbers)

    m_e at the conduction valley, two directions -> DOS mass
    m_h at the valence valley,    two directions -> DOS mass
    mu = reduced(m_e, m_h)

  then feed mu_dir and mu_ind to mott_wannier.splitting().
""")
    print("EXAMPLE with placeholder numbers:\n")
    me_dir = report("e, direct valley", 0.28, 0.01, 0.31, 0.01)
    mh_dir = report("h, direct valley", 0.95, 0.06, 1.40, 0.09)
    me_ind = report("e, indirect (cond)", 0.28, 0.01, 0.31, 0.01)
    mh_ind = report("h, indirect (val)", 0.55, 0.03, 0.61, 0.04)
    print(f"\n  mu_dir = {reduced(me_dir, mh_dir):.4f}")
    print(f"  mu_ind = {reduced(me_ind, mh_ind):.4f}")


if __name__ == "__main__":
    main()
