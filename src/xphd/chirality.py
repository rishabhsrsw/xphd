"""Chirality of a degenerate exciton pair: helicity and valley polarisation.

    xphd chirality --states 39 40

Both come from a single Bethe-Salpeter run, and neither calls
yexc.get_Akcv() or reads exc.Akcv, which not every yambopy version provides.

Helicity
--------
Taken from the exciton dipoles written by `xphd dipoles`. A degenerate pair
comes out of the solver in an arbitrary basis, so it is first rotated into
the eigenbasis of L_z = P sigma_y P^dagger, with P the 2x2 matrix of in-plane
dipoles. Then

    rho = 2 Im(D_x D_y*) / (|D_x|^2 + |D_y|^2)

is +1 for sigma^+ and -1 for sigma^-. The rotation makes the two helicities
equal and opposite but NOT unity: an arbitrary pair gives |rho| ~ 0.7, and
even purely linear dipoles give ~0.73. Only a value at or very near 1 is
evidence of circular polarisation. The ratio D_y / (i D_x), exactly -+1 for
perfect circularity, is printed as well.

Valley polarisation
-------------------
The same rotation is applied to the exciton envelopes, and for each rotated
state

    P_valley = (W_K - W_K') / (W_K + W_K'),   W = sum_vc |A_kvc|^2

summed over the k-points nearer K than K' and vice versa; points equidistant
from both lie on neither valley and are left out. Using ONE rotation for
both quantities is the point: the states that are circularly polarised are
the ones whose valley weight is measured, so the two numbers test the same
claim from two directions -- one through the dipoles, one through the
envelope.

Momentum convention: the table index k of A^{SQ}_kvc is the electron, with
the hole at k+Q (see excsym.shift_map). At Q = 0 the two coincide, so it does not matter here.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from .yambo import full_zone_kpoints, lattice, load_excitons

SIGMA_Y = np.array([[0, -1j], [1j, 0]])


def jz_rotation(P):
    """Unitary taking a degenerate pair into the eigenbasis of L_z.

    P is 2x2, rows = states, columns = (D_x, D_y). The eigenvector with the
    larger eigenvalue comes first, so row 0 of the result is sigma^+.
    """
    ev, evec = np.linalg.eigh(P @ SIGMA_Y @ P.conj().T)
    return evec[:, np.argsort(ev)[::-1]].conj().T


def helicity(Dx, Dy):
    """2 Im(Dx Dy*) / (|Dx|^2 + |Dy|^2); +1 is sigma^+, -1 sigma^-."""
    return float(2.0 * np.imag(Dx * np.conj(Dy))
                 / (abs(Dx) ** 2 + abs(Dy) ** 2 + 1e-300))


def valley_masks(kf, tol=1e-6):
    """Full-zone points nearer K than K', and vice versa."""
    def pdist(a, b):
        d = a - b
        d -= np.rint(d)
        return np.linalg.norm(d, axis=-1)
    dK = pdist(kf[:, :2], np.array([1 / 3, 1 / 3]))
    dKp = pdist(kf[:, :2], np.array([2 / 3, 2 / 3]))
    return dK < dKp - tol, dKp < dK - tol


def valley_polarisation(A_state, near_K, near_Kp):
    W = np.sum(np.abs(A_state) ** 2, axis=(1, 2))
    W = W / max(W.sum(), 1e-300)
    wK, wKp = float(W[near_K].sum()), float(W[near_Kp].sum())
    return wK, wKp, (wK - wKp) / max(wK + wKp, 1e-300)


def main(argv=None):
    p = argparse.ArgumentParser(prog="xphd chirality",
                                description=__doc__.split("\n\n")[0])
    p.add_argument("--save", default="../phonons/SAVE")
    p.add_argument("--bse-dir", default="../BSE/output_all")
    p.add_argument("--dipoles", default="exc_dipoles.npy",
                   help="written by `xphd dipoles`")
    p.add_argument("--states", type=int, nargs=2, required=True,
                   help="1-based indices of the degenerate pair")
    p.add_argument("--out", default="chirality.txt")
    a = p.parse_args(argv)

    if not os.path.exists(a.dipoles):
        raise SystemExit(f"{a.dipoles} not found; run `xphd dipoles` first")
    D = np.load(a.dipoles)
    if D.shape[0] != 3:
        raise SystemExit(f"{a.dipoles} has shape {D.shape}; expected (3, n)")
    i1, i2 = a.states[0] - 1, a.states[1] - 1
    if max(i1, i2) >= D.shape[1]:
        raise SystemExit(f"state {max(i1, i2) + 1} beyond the {D.shape[1]} "
                         f"in {a.dipoles}")

    ylat = lattice(a.save)
    exc, A, E = load_excitons(f"{a.bse_dir}/ndb.BS_diago_Q1", ylat,
                              neigs=max(i1, i2) + 1)
    kf = full_zone_kpoints(ylat)

    P = np.array([[D[0, i1], D[1, i1]], [D[0, i2], D[1, i2]]])
    U = jz_rotation(P)
    Pj = U @ P
    raw = [helicity(*P[k]) for k in range(2)]
    rot = [helicity(*Pj[k]) for k in range(2)]

    Apair = np.stack([A[i1], A[i2]])
    Arot = np.einsum("mn,nkvc->mkvc", U, Apair)
    near_K, near_Kp = valley_masks(kf)

    L = []
    L.append("CHIRALITY OF A DEGENERATE EXCITON PAIR")
    L.append("=" * 64)
    L.append(f"states {a.states[0]} and {a.states[1]}: "
             f"E = {E[i1]:.5f}, {E[i2]:.5f} eV "
             f"(split {(E[i2] - E[i1]) * 1e3:+.3f} meV)")
    L.append(f"k-mesh: {len(kf)} points, {near_K.sum()} nearer K, "
             f"{near_Kp.sum()} nearer K', "
             f"{len(kf) - near_K.sum() - near_Kp.sum()} equidistant, excluded")
    L.append("")
    L.append("solver basis (arbitrary within the degenerate pair):")
    for k, s in enumerate(a.states):
        L.append(f"   state {s}: helicity {raw[k]:+.4f}")
    L.append("")
    L.append("rotated into the L_z eigenbasis:")
    L.append(f"   {'':>8} {'helicity':>10} {'Dy/(iDx)':>20} "
             f"{'W_K':>8} {'W_Kp':>8} {'P_valley':>9}")
    for k, lab in enumerate(("sigma+", "sigma-")):
        r = Pj[k, 1] / (1j * Pj[k, 0]) if abs(Pj[k, 0]) > 0 else np.nan
        wK, wKp, pv = valley_polarisation(Arot[k], near_K, near_Kp)
        L.append(f"   {lab:>8} {rot[k]:+10.4f} "
                 f"{r.real:+9.4f}{r.imag:+9.4f}j {wK:8.4f} {wKp:8.4f} "
                 f"{pv:+9.4f}")
    L.append("")
    L.append(f"|helicity| = {abs(rot[0]):.4f}. The rotation forces the two to "
             f"be equal and opposite, not to be")
    L.append("unity: an arbitrary pair gives ~0.7, so only a value near 1 "
             "shows circular")
    L.append("polarisation. P_valley of opposite sign on the two states, from "
             "the same")
    L.append("rotation, confirms it through the envelope rather than the "
             "dipoles.")

    txt = "\n".join(L)
    print(txt)
    with open(a.out, "w") as f:
        f.write(txt + "\n")
    print(f"\nwritten to {a.out}")


if __name__ == "__main__":
    main()
