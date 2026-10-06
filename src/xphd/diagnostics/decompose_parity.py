"""Why an exciton has a fractional sigma_h character.

For sigma_h the k-map is the identity and the rotation matrix is diagonal
with entries p_n(k) = +-1, so

    chi_S = sum_kvc |A^S_kvc|^2 p_c(k) p_v(k+Q) ,

with the electron at the table index k and the hole at k+Q (see
excsym.shift_map).

A non-degenerate eigenstate of a Hamiltonian commuting with sigma_h must give
+-1 exactly: the BSE Hamiltonian cannot couple an even band pair to an odd
one. A fractional value is therefore never physics. It means either

  (a) the parities are attached to the wrong transitions -- a wrong
      electron-hole pairing, a mismatched k order or band window -- which
      shows here as one band pair carrying weight on both even and odd k;
      on GaN this was the old pairing (electron at k+Q), which produced a
      spurious ring of fractional parity; or

  (b) two bands are degenerate at the dominant k-points and returned in a
      mixed basis, so p_n itself is not +-1 -- diagnosed by how close each
      p_n is to +-1.

The database is found through the irreducible parent of the point, read
from lat.kpoints_indexes. An earlier version located it through the
exciton-phonon archive filenames, which are numbered over the FULL zone,
and paired them with ndb.BS_diago_Q{n}, which is numbered over the
irreducible wedge -- opening the wrong database for every point but Gamma.

    xphd decompose-parity --nv 3 --parity parity.npz
    xphd decompose-parity --nv 3 --iq 37 --state 1
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from ..excsym import shift_map
from ..yambo import (full_zone_kpoints, ibz_parents, lattice, load_dmats,
                    load_excitons, sigma_h_op, spatial_ops)


def main(argv=None):
    p = argparse.ArgumentParser(prog="xphd decompose-parity",
                                description=__doc__.split("\n\n")[0])
    p.add_argument("--save", default="../phonons/SAVE")
    p.add_argument("--bse-dir", default="../BSE/output_all")
    p.add_argument("--dmats", default="Dmats.npy")
    p.add_argument("--parity", default="parity.npz",
                   help="written by xphd parity")
    p.add_argument("--nv", type=int, required=True,
                   help="valence bands inside the BSE window")
    p.add_argument("--iq", type=int, default=None,
                   help="full-zone index into parity.npz; default is the "
                        "first point with |chi| more than 0.2 from 1")
    p.add_argument("--state", type=int, default=1, help="1-based")
    a = p.parse_args(argv)

    d = np.load(a.parity)
    chi_all = d["chi"]
    S = a.state - 1
    if a.iq is None:
        amb = np.where(np.abs(np.abs(chi_all[:, S]) - 1) > 0.2)[0]
        if not len(amb):
            raise SystemExit("no point with a fractional character; every "
                             "|chi| is within 0.2 of 1")
        a.iq = int(amb[0])

    ylat = lattice(a.save)
    kf = full_zone_kpoints(ylat)
    kidx, first = ibz_parents(ylat)
    D = load_dmats(a.dmats)

    # the point, its irreducible parent, and the database at that parent
    j = int(d["iQ_ibz"][a.iq]) if "iQ_ibz" in d.files else int(kidx[a.iq])
    Q = kf[first[j]]
    path = f"{a.bse_dir}/ndb.BS_diago_Q{j + 1}"
    print(f"full-zone point {a.iq} at {np.round(d['Q_red'][a.iq][:2], 4)}; "
          f"reported chi = {chi_all[a.iq, S]:+.4f}")
    print(f"   irreducible parent {j}: BS_diago_Q{j + 1} at "
          f"{np.round(Q[:2], 4)}  (parity is constant across the star)")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}")

    # ---- sigma_h band parities ---------------------------------------
    ops = spatial_ops(a.save, kf, D, a.nv)
    _, _, Dc, Dv, _ = sigma_h_op(ops)
    pc = np.diagonal(Dc, axis1=1, axis2=2).real
    pv = np.diagonal(Dv, axis1=1, axis2=2).real
    dev = max(np.abs(np.abs(pc) - 1).max(), np.abs(np.abs(pv) - 1).max())
    print(f"   largest deviation of any p_n from +-1: {dev:.2e}"
          f"   {'OK' if dev < 0.05 else '<-- bands are NOT sigma_h eigenstates somewhere; mechanism (b)'}")

    # ---- the exciton ---------------------------------------------------
    kQ, w = shift_map(kf, Q)
    if kQ is None:
        raise SystemExit(f"k+Q leaves the grid (worst {w:.1e})")
    _, A, E = load_excitons(path, ylat, max(a.state, 1))
    print(f"   state {a.state}: E = {E[S]:.4f} eV; k -> k+Q exact to {w:.1e}")

    # ---- the decomposition ---------------------------------------------
    prod = pc[:, None, :] * pv[kQ][:, :, None]          # (nk, nv, nc): electron k, hole k+Q
    W = np.abs(A[S]) ** 2                               # (nk, nv, nc)
    W = W / W.sum()
    w_even = float(W[prod > 0].sum())
    w_odd = float(W[prod < 0].sum())
    chi = float((W * prod).sum())
    print(f"\n   weight on EVEN band pairs (p_c p_v = +1): {w_even:.4f}")
    print(f"   weight on ODD  band pairs (p_c p_v = -1): {w_odd:.4f}")
    print(f"   chi = even - odd = {chi:+.4f}   "
          f"(reported {chi_all[a.iq, S]:+.4f})")
    # with w_even + w_odd = 1 and chi = w_even - w_odd, a chi of -2/3
    # requires the split 1/6 : 5/6, not 1/3 : 2/3
    ex = (1 + chi) / 2
    print(f"   implied split: {ex:.4f} even : {1 - ex:.4f} odd")
    for num, den, lab in ((1, 6, "1/6 : 5/6"), (1, 4, "1/4 : 3/4"),
                          (1, 3, "1/3 : 2/3"), (1, 2, "1/2 : 1/2")):
        if abs(w_even - num / den) < 0.02:
            print(f"   close to {lab} -- a clean weight split, not noise")

    # ---- where the weight sits -----------------------------------------
    Wk = W.sum(axis=(1, 2))
    print("\n   the six heaviest k-points:")
    print(f"   {'k (reduced)':>22} {'weight':>8}  {'p_v(k+Q)':>18}  "
          f"{'p_c(k)':>18}")
    for k in np.argsort(Wk)[::-1][:6]:
        print(f"   {str(np.round(kf[k, :2], 4)):>22} {Wk[k]:8.4f}  "
              f"{str(np.round(pv[kQ[k]], 2)):>18}  "
              f"{str(np.round(pc[k], 2)):>18}")

    # ---- which (v, c) pairs carry the two signs ------------------------
    print("\n   weight by band pair:")
    print(f"   {'v':>3} {'c':>3} {'weight':>9} {'sign':>6}")
    for v in range(A.shape[2]):
        for c in range(A.shape[3]):
            wv = float(W[:, v, c].sum())
            if wv < 1e-3:
                continue
            sg = np.sign(prod[:, v, c]).mean()
            print(f"   {v + 1:3d} {c + 1:3d} {wv:9.4f} {sg:+6.2f}"
                  + ("   <-- sign varies with k"
                     if abs(abs(sg) - 1) > 0.05 else ""))
    print("""
   If the weight splits between pairs of opposite sign, the character is a
   weighted mean and sigma_h is not a good quantum number for this state --
   mechanism (a). If a single pair carries the weight but its sign varies
   with k, the parity of one band changes across the exciton's support,
   which is the same conclusion by another route. Either way report the
   value as undefined rather than as an intermediate parity.""")


if __name__ == "__main__":
    main()
