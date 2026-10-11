"""Run as: xphd dipoles [options]. xphd dipoles --help lists the options.

    xphd dipoles --save ../phonons/SAVE --bse-dir ../BSE/output_all
"""
from __future__ import annotations

import argparse

import numpy as np

from .yambo import lattice, load_excitons


def read_cv_dipoles(ylat, path, bs_bands):
    """<c k|r|v k> on the irreducible k-points, (nk_ibz, 3, nc, nv), for the
    BSE window bs_bands = [first valence band, last conduction band]
    (Fortran numbering). See the module docstring for why yambopy's reader
    is not called with a band range."""
    from yambopy.dbs.dipolesdb import YamboDipolesDB

    class _Fixed(YamboDipolesDB):
        def __init__(self, lattice, nq_ibz, nq_bz, nk_ibz, nk_bz, spin,
                     min_band, max_band, indexv, indexc, bands_range, nbands,
                     nbandsv, nbandsc, dip_bands_ordered, dipoles, **kw):
            if not dip_bands_ordered:
                nbandsc = nbands - nbandsv        # yambopy: max_band - nbandsv
            super().__init__(lattice, nq_ibz, nq_bz, nk_ibz, nk_bz, spin,
                             min_band, max_band, indexv, indexc, bands_range,
                             nbands, nbandsv, nbandsc, dip_bands_ordered,
                             dipoles, **kw)

    ydip = _Fixed.from_db_file(ylat, filename=path, project=False,
                               expand=False)
    raw = np.asarray(ydip.dipoles)                # (k, xyz, c_db, v_db)
    b0, nocc = int(ydip.min_band), int(ylat.nbandsv)
    v = slice(int(bs_bands[0]) - b0, nocc - b0 + 1)
    c = slice(0, int(bs_bands[1]) - nocc)
    if not (0 <= v.start < v.stop <= raw.shape[3]
            and 0 < c.stop <= raw.shape[2]):
        raise SystemExit(f"BSE window {list(bs_bands)} is not inside the "
                         f"dipole database (bands {b0}-{int(ydip.max_band)}, "
                         f"{nocc} occupied)")
    print(f"dipoles: database bands {b0}-{int(ydip.max_band)}, {nocc} occupied;"
          f" BSE window {list(bs_bands)} -> {v.stop - v.start} valence, "
          f"{c.stop} conduction")
    return raw[:, :, c, v]


def residual_verdict(mine, ref, k=3, max_spread=1.25):
    """Compare per-manifold |D|^2 with yambo's residuals.

    Agreement means proportionality: the SAME constant ratio for every bright
    manifold (the units differ, so only the ratio's constancy matters), and the
    same manifolds brightest. Returns (verdict, spread), spread = max/min ratio
    over the manifolds compared.
    """
    mine, ref = np.asarray(mine, float), np.asarray(ref, float)
    k = min(k, len(mine))
    tm = set(np.argsort(mine)[::-1][:k])
    tr = set(np.argsort(ref)[::-1][:k])
    use = sorted(i for i in (tm | tr) if ref[i] > 0 and mine[i] > 0)
    r = mine[use] / ref[use] if use else np.array([np.nan])
    spread = float(np.nanmax(r) / np.nanmin(r)) if use else np.inf
    if tm == tr and spread <= max_spread:
        return "AGREE", spread
    if tm == tr:
        return (f"DISAGREE -- same brightest manifolds, but the ratio varies x{spread:.1f}: "
                f"the dipoles are NOT proportional to yambo's"), spread
    return "DISAGREE -- different brightest manifolds; inspect before use", spread


def main(argv=None):
    p = argparse.ArgumentParser(prog="xphd dipoles",
                                description=__doc__.split("\n\n")[0])
    p.add_argument("--save", default="../phonons/SAVE")
    p.add_argument("--bse-dir", default="../BSE/output_all")
    p.add_argument("--dipoles-dir", default=None,
                   help="directory holding ndb.dipoles; defaults to --bse-dir")
    p.add_argument("-o", "--out", default="exc_dipoles.npy")
    p.add_argument("--window", type=float, default=1.5,
                   help="eV above the lowest exciton for the residual check")
    p.add_argument("--deg-tol", type=float, default=2e-3)
    p.add_argument("--compare", default=None,
                   help="a reference exc_dipoles.npy, e.g. GaN's from the "
                        "working yambopy path, to compare |D|^2 against")
    a = p.parse_args(argv)

    ylat = lattice(a.save)
    exc, A, E = load_excitons(f"{a.bse_dir}/ndb.BS_diago_Q1", ylat)
    A = A.transpose(0, 1, 3, 2)                  # (n, k, v, c) -> (n, k, c, v)
    nc, nv = A.shape[2], A.shape[3]
    print(f"A {A.shape} (n, k, c, v)")

    dip = read_cv_dipoles(ylat, f"{a.dipoles_dir or a.bse_dir}/ndb.dipoles",
                          exc.bs_bands)
    if dip.shape[2:] != (nc, nv):
        raise SystemExit(f"dipole window {dip.shape[2:]} != BSE (nc, nv) "
                         f"{(nc, nv)}")

    rot = ylat.sym_car[ylat.kmap[:, 1], ...]
    dexp = np.einsum("kij,kjcv->kicv", rot, dip[ylat.kmap[:, 0], ...],
                     optimize=True)
    tr = ylat.kmap[:, 1] >= ylat.sym_car.shape[0] / (int(ylat.time_rev) + 1)
    dexp[tr] = dexp[tr].conj()
    if A.shape[1:] != (dexp.shape[0], dexp.shape[2], dexp.shape[3]):
        raise SystemExit(f"shape mismatch A {A.shape[1:]} vs dipoles "
                         f"{(dexp.shape[0], dexp.shape[2], dexp.shape[3])}")

    dexc = (np.einsum("nkcv,kicv->in", A, dexp, optimize=True)
            .astype(dip.dtype) / np.sqrt(exc.nkpoints))
    np.save(a.out, dexc)
    print(f"\nwrote {a.out}: {dexc.shape} (xyz, n)")

    fin = (np.abs(dexc[:2]) ** 2).sum(axis=0)
    fz = np.abs(dexc[2]) ** 2
    print("\n   brightest states:")
    for n in np.argsort(fin + fz)[::-1][:6]:
        print(f"     state {n+1:5d}  E = {E[n]:8.4f}  in-plane {fin[n]:.3e}  "
              f"z {fz[n]:.3e}")

    rr = np.asarray(getattr(exc, "r_residual", []))
    order = np.argsort(E)
    low = order[E[order] < E[order[0]] + a.window]
    groups, cur = [], [low[0]]
    for n in low[1:]:
        if abs(E[n] - E[cur[-1]]) < a.deg_tol:
            cur.append(n)
        else:
            groups.append(cur); cur = [n]
    groups.append(cur)

    if rr.size:
        rr = np.abs(rr) ** 2
        yam = rr.reshape(-1, rr.shape[-1]).sum(0) if rr.ndim > 1 else rr
        mine = np.array([fin[g].sum() for g in groups])
        ref = np.array([yam[g].sum() for g in groups])
        k = min(3, len(groups))
        tm, tr_ = set(np.argsort(mine)[::-1][:k]), set(np.argsort(ref)[::-1][:k])
        print(f"\n   against yambo's residuals, by degenerate manifold:")
        for gi in sorted(tm | tr_, key=lambda g: E[groups[g][0]]):
            g = groups[gi]
            lab = f"{g[0]+1}" if len(g) == 1 else f"{g[0]+1}-{g[-1]+1}"
            ratio = mine[gi] / ref[gi] if ref[gi] > 0 else np.nan
            print(f"     {lab:<10} {E[g[0]]:.4f}  this {mine[gi]:.4e}  "
                  f"yambo {ref[gi]:.4e}  ratio {ratio:.4e}")
        verdict, spread = residual_verdict(mine, ref, k)
        print(f"   brightest {k} manifolds: {verdict}")
        print("   Agreement means a CONSTANT ratio across manifolds (the units differ).")
        if verdict.startswith("DISAGREE") and tm == tr_:
            print("   Common causes: a BSE solved beyond Tamm-Dancoff (BSEmod = coupling), whose\n"
                  "   residuals mix resonant and antiresonant parts; or yambo rescaling the dipoles\n"
                  "   by quasiparticle energies. Relative weights WITHIN one manifold are unaffected.")
    spread = max(((fin[g].max() - fin[g].min()) / fin[g].max()
                  for g in groups if len(g) > 1 and fin[g].max() > 1e-6),
                 default=0.0)
    print(f"   spread within degenerate manifolds: {spread:.1e} "
          f"(symmetry requires 0)")

    if a.compare:
        ref = np.load(a.compare)
        if ref.shape == dexc.shape:
            r1, r2 = (np.abs(ref) ** 2).sum(0), (np.abs(dexc) ** 2).sum(0)
            dev = np.abs(r1 - r2).max() / max(r1.max(), 1e-300)
            print(f"\n   --compare: max relative deviation in |D|^2 "
                  f"{dev:.2e}  {'AGREES' if dev < 1e-6 else 'DIFFERS'}")
        else:
            print(f"\n   --compare: shapes differ {ref.shape} vs {dexc.shape}")


if __name__ == "__main__":
    main()
