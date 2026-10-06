"""Irreducible-representation labels at the high-symmetry points.

    xphd irreps --nv 3 --points G K M

Four things this handles that are easy to get wrong:

1. Which database. BS_diago_Q{n} is numbered by the irreducible index; the
   point is found on the full-zone mesh and its parent read from
   kpoints_indexes. Pairing a full-zone number with a BS_diago number opens
   the wrong database for every point but Gamma.

2. The little group is computed as the stabilizer of Q, not selected by
   class name. At M only one C2' and one sigma_v fix a given M.

3. At K the little group C3h is Abelian and every irrep is one
   dimensional; nothing forces E'_1 and E'_2 to pair. A near-degenerate pair
   grouped by --deg-tol is reported as E'_1+E'_2 -- an approximate
   degeneracy, not a symmetry-protected one. Which of A, E_1, E_2 a K state
   gets depends on the C3 axis used, through an atom or the hexagon centre:
   moving it multiplies every C3 eigenvalue by w or w^2. The labels are
   relative to the origin of the SAVE; the sigma_h parity is absolute.

4. One representative per class. In C3h, C3 and C3^2 are distinct classes
   with conjugate characters, so averaging them destroys the phase.

Unitarity of the representation is checked on the lowest manifold at each
point before anything is labelled.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from .excsym import (HIGH_SYMMETRY, assign_irreps, class_characters,
                     degeneracy_groups, group_name, rep_matrix_Q, shift_map,
                     stabilizer)
from .yambo import (full_zone_kpoints, ibz_parents, lattice, load_dmats,
                    load_excitons, spatial_ops)


def main(argv=None):
    p = argparse.ArgumentParser(prog="xphd irreps",
                                description=__doc__.split("\n\n")[0])
    p.add_argument("--save", default="../phonons/SAVE")
    p.add_argument("--bse-dir", default="../BSE/output_all")
    p.add_argument("--dmats", default="Dmats.npy")
    p.add_argument("--nv", type=int, required=True)
    p.add_argument("--points", nargs="*", default=["G", "K", "M"],
                   help="any of: " + ", ".join(sorted(HIGH_SYMMETRY)))
    p.add_argument("--nstates", type=int, default=15)
    p.add_argument("--deg-tol", type=float, default=5e-3,
                   help="eV; states closer than this form one manifold")
    a = p.parse_args(argv)

    ylat = lattice(a.save)
    D = load_dmats(a.dmats)
    kf = full_zone_kpoints(ylat)
    kidx, first = ibz_parents(ylat)
    ops = spatial_ops(a.save, kf, D, a.nv)
    print(f"{len(ops)} spatial operations; {len(kf)} k-points, "
          f"{len(first)} irreducible")

    for pt in a.points:
        if pt not in HIGH_SYMMETRY:
            print(f"\n[skip] unknown point {pt!r}")
            continue
        Qreq = np.array([*HIGH_SYMMETRY[pt], 0.0])
        d = kf[:, :2] - np.mod(Qreq[:2], 1.0)
        d -= np.rint(d)
        i = int(np.argmin(np.linalg.norm(d, axis=1)))
        if np.linalg.norm(d[i]) > 1e-5:
            print(f"\n[skip] {pt} is not on this mesh")
            continue
        j = int(kidx[i])
        Q = kf[first[j]]
        path = f"{a.bse_dir}/ndb.BS_diago_Q{j + 1}"
        print(f"\n{'=' * 64}\n{pt}: requested {np.round(Qreq[:2], 4)}, "
              f"database BS_diago_Q{j + 1} at {np.round(Q[:2], 4)}")
        if not os.path.exists(path):
            print(f"   missing {path}")
            continue

        stab = stabilizer(ops, Q)
        grp = group_name(stab)
        counts = {}
        for o in stab:
            counts[o[0]] = counts.get(o[0], 0) + 1
        print(f"   little group {grp} (order {len(stab)}): "
              + ", ".join(f"{k}x{v}" for k, v in sorted(counts.items())))

        kQ, w = shift_map(kf, Q)
        if kQ is None:
            print(f"   k+Q leaves the grid ({w:.1e}); skipped")
            continue
        _, A, E = load_excitons(path, ylat, a.nstates)
        groups = degeneracy_groups(E, tol=a.deg_tol)

        g0 = groups[0]
        worst = max(np.abs(M @ M.conj().T - np.eye(len(g0))).max()
                    for M in (rep_matrix_Q(A, o[2], o[3], o[4], kQ, g0)
                              for o in stab))
        print(f"   unitarity on the lowest manifold: {worst:.1e}"
              + ("" if worst < 1e-2 else "   <-- labels below are unsafe"))

        print(f"   {'states':<9} {'E (eV)':>9} {'chi(sh)':>9} "
              f"{'irrep':>12} {'resid':>9}")
        for g in groups:
            chi = class_characters(A, stab, kQ, g)
            name, res = assign_irreps(chi, grp, verbose=False)
            idx = f"{g[0]+1}" if len(g) == 1 else f"{g[0]+1}-{g[-1]+1}"
            sh = np.real(chi.get("sh", np.nan))
            print(f"   {idx:<9} {np.mean(E[g]):9.4f} {sh:+9.3f} "
                  f"{name:>12} {res:9.2e}"
                  + ("" if res < 0.2 else "   <-- no clean match"))

    print("\n   A residual near 0 is a clean assignment; 0.1-0.2 reflects the "
          "finite mesh.\n   Larger values mean an incomplete manifold -- try "
          "another --deg-tol -- or mixed irreps.")


if __name__ == "__main__":
    main()
