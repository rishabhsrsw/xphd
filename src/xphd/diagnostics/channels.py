"""Where does a linewidth come from?

    Gamma = 2*pi * sum_{nu,m} [ sum_q W_q ] * |G|^2

so an unphysical Gamma is either a wrong |G| or a wrong joint density of
states. This separates them and ranks the channels.

The bare DOS (|G|^2 set to 1, units 1/eV) should be O(1-100) for a bandwidth
of a few hundred meV. Orders of magnitude more means the delta surface is
carried by a handful of near-degenerate triangles, i.e. the mesh cannot
support a delta-function integration at all.
"""
from __future__ import annotations

import numpy as np

from ..core.mesh import blockwise
from ..core.stats import KB_EV, bose
from ..tetra import delta_weights_2d

__all__ = ["channel_report"]


def channel_report(archive, state=0, T=10.0, flat_width=1e-5,
                   acoustic_cut=5e-4, top=12, verbose=True):
    g2 = archive.grid("g2")
    E_m = archive.grid("E_m")
    hw = archive.grid("hw")
    E_n = archive.E_n
    kT = max(KB_EV * T, 1e-12)
    rows = []

    for nu in range(archive.nmod):
        wq = hw[:, :, nu]
        live = wq > acoustic_cut
        if not live.any():
            continue
        N = bose(wq, kT)
        for m in range(archive.nexc):
            gg = g2[:, :, nu, state, m] * live
            dE = E_n[state] - E_m[:, :, m]
            for tag, arg, occ in (("emis", dE - wq, N + 1.0),
                                  ("abs", dE + wq, N)):
                W = delta_weights_2d(arg, flat_width, "anti")
                c = 2.0 * np.pi * float(np.sum(W * gg * occ))
                if abs(c) > 1e-14:
                    rows.append(dict(contrib=c, kind=tag, nu=nu, m=m,
                                     dos=float(np.sum(W * live)),
                                     gmax=float(np.sqrt(gg.max())),
                                     hw=float(wq[live].mean())))

    total = sum(r["contrib"] for r in rows)
    rows.sort(key=lambda d: -abs(d["contrib"]))
    out = dict(total=total, rows=rows,
               max_dos=max((r["dos"] for r in rows), default=0.0),
               concentration=abs(rows[0]["contrib"] / total) if rows else 0.0)

    if verbose:
        print(f"state {state+1} at {T} K: Gamma = {total*1e3:.3f} meV "
              f"over {len(rows)} live channels\n")
        print(f"   {'ch':>5} {'nu':>3} {'m':>3} {'<hw>':>8} {'DOS 1/eV':>10} "
              f"{'max|G| meV':>11} {'meV':>10} {'% tot':>7}")
        for r in rows[:top]:
            print(f"   {r['kind']:>5} {r['nu']+1:3d} {r['m']+1:3d} "
                  f"{r['hw']*1e3:8.2f} {r['dos']:10.3f} {r['gmax']*1e3:11.3f} "
                  f"{r['contrib']*1e3:10.3f} {r['contrib']/total*100:6.1f}%")
        if out["max_dos"] > 500:
            print("\n   *** a few near-degenerate triangles carry the delta "
                  "surface;\n       this mesh cannot support the integration")
        print("\n   note: concentration falls with T simply because more "
              "channels open.\n   Compare ACROSS MESHES at fixed T, and the "
              "lowest T is the most\n   stringent.")
    return out
