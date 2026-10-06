"""Do the transport out-rates and the linewidths agree?

The linewidth of a state is its out-scattering rate in the dilute limit, and
the real-time transport assembles the same rates into a matrix. Evaluated on
the same mesh, with the same phonons and the same coupling, the out-rate
sum_j P_ij must equal Gamma_i / hbar. This compares them at Gamma.

Only the Gamma archive is passed to the refined out-rate: the refinement
returns entries only for the archives it is given, so this is what makes the
check fast.

    xphd bte-check --archives "ne5/GI_ExcPh_Q*.npz" --field lw_sym.npz \\
                   --hw-fine ../hw_fine.npy --T 77

The last row -- refined, with the matdyn phonons the sweep used -- is the
like-for-like comparison. A ratio near 1 confirms the two are the same
integral; the rows above show what a coarser mesh and interpolated phonons
cost.
"""
from __future__ import annotations

import argparse

import numpy as np

HBAR = 6.582119569e-4          # eV.ps


def main(argv=None):
    import xphd
    from xphd import bte
    from xphd.io.excph import load_archives

    p = argparse.ArgumentParser(prog="xphd bte-check",
                                description=__doc__.split("\n\n")[0])
    p.add_argument("--archives", required=True,
                   help="glob for the archives, quoted, e.g. "
                        "\"ne5/GI_ExcPh_Q*.npz\"")
    p.add_argument("--field", default="lw_sym.npz",
                   help="unfolded linewidth field from xphd unfold")
    p.add_argument("--hw-fine", default=None,
                   help="the SAME refined phonon file the sweep used; "
                        "without it only the interpolated rows are shown")
    p.add_argument("--T", type=float, default=77.0,
                   help="temperature in K; must be one the field contains")
    p.add_argument("--refine", type=int, nargs="*", default=[5, 10, 15])
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args(argv)

    arcs = load_archives(a.archives, verbose=False)
    if not arcs:
        raise SystemExit(f"no archives match {a.archives}")
    fld = xphd.LinewidthField.load(a.field)

    # the temperature index is looked up, not assumed: a hardcoded 0 only
    # matches if --T happens to be the first temperature of the field
    Ts = np.atleast_1d(fld.T)
    it = int(np.argmin(np.abs(Ts - a.T)))
    if abs(Ts[it] - a.T) > 1e-6:
        raise SystemExit(f"--T {a.T} not in the field; available: "
                         f"{np.round(Ts, 2).tolist()}")

    gk = min(arcs)
    g0 = int(np.argmin(np.linalg.norm(fld.Q_red[:, :2], axis=1)))
    print(f"Gamma: archive {gk}, field row {g0}, "
          f"Q = {np.round(fld.Q_red[g0], 4)}, T = {Ts[it]:.1f} K")
    ne = min(len(arcs[gk].E_n), len(fld.E_n[g0]))
    print(f"  E_n agree to "
          f"{np.abs(arcs[gk].E_n[:ne] - fld.E_n[g0][:ne]).max():.2e} eV")

    target = fld.LW[it, g0, :] / HBAR                # 1/ps
    one = {gk: arcs[gk]}

    print(f"\n  {'source':>26} {'state 1':>10} {'ratio':>8}")
    print(f"  {'linewidth sweep':>26} {target[0]:10.3f} {1.0:8.3f}")

    rm = bte.build_rates(arcs, T=a.T, verbose=False)
    print(f"  {'BTE matrix, no refine':>26} {rm.out_rate[0]*1e3:10.3f} "
          f"{rm.out_rate[0]*1e3/target[0]:8.3f}")

    for f in a.refine:
        r = bte.refined_out_rates(one, a.T, f, workers=a.workers,
                                  verbose=False)
        print(f"  {f'refine {f}, interp. hw':>26} {r[0]*1e3:10.3f} "
              f"{r[0]*1e3/target[0]:8.3f}")

    if a.hw_fine:
        r = bte.refined_out_rates(one, a.T, hw_fine=a.hw_fine,
                                  workers=a.workers, verbose=False)
        print(f"  {'refined, matdyn hw':>26} {r[0]*1e3:10.3f} "
              f"{r[0]*1e3/target[0]:8.3f}   <- matched conditions")
    else:
        print("  (pass --hw-fine for the matched-conditions row)")

    print("\n  The matched-conditions row is the like-for-like comparison.")
    print("  A ratio near 1 confirms the out-rate and the linewidth are the")
    print("  same integral.")


if __name__ == "__main__":
    main()
