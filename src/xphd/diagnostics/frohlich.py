"""Electron/hole Frohlich cancellation in the long-wavelength limit.

For a NEUTRAL exciton the electron and hole Frohlich vertices cancel as
q -> 0, so the diagonal G_nn stays finite even though each piece diverges.

Two things must be handled or the test manufactures a false failure:
  * max|Ge| and max|Gh| can sit at DIFFERENT exciton indices, so their ratio
    is not element-wise. Everything is computed per element.
  * degenerate branches are defined only up to a unitary rotation among
    themselves, so |Ge| for one partner is meaningless alone. Branches are
    grouped by their Gamma frequency and summed within each group.

    C(q) = sum_manifold |Ge+Gh|^2 / sum_manifold (|Ge|^2 + |Gh|^2)

is invariant under both the exciton phase and the phonon rotation, because
the cross term carries the arbitrary phases in conjugate pairs.

    C << 1  destructive: cancellation working
    C ~  1  effectively independent
    C >  1  constructive

Note the cancellation only switches on for q * a_exc << 1. On a coarse mesh
the smallest accessible q may sit AT the crossover, and partial cancellation
is then the correct answer rather than a bug.
"""
from __future__ import annotations

import numpy as np

from ..core.mesh import hex_norm, shells
from ..core.stats import degeneracy_groups

__all__ = ["frohlich_report"]


def frohlich_report(archive, nshell=8, mode_tol=2e-3, verbose=True):
    if not (archive.has("Ge") and archive.has("Gh")):
        raise ValueError(
            "archive has no Ge_grid/Gh_grid. Regenerate with the electron "
            "and hole contributions -- without them this cannot be answered.")

    Ge = np.asarray(archive._raw("Ge"))
    Gh = np.asarray(archive._raw("Gh"))
    hw = np.asarray(archive._raw("hw"), float)
    r = hex_norm(archive.q_red)
    ne = Ge.shape[2]
    eye = np.eye(ne, dtype=bool)
    tot = Ge + Gh

    groups = degeneracy_groups(hw[0], mode_tol)
    out = {"groups": groups, "shells": {}}

    for g in groups:
        rec = []
        for rq, idx in shells(r)[:nshell]:
            e2 = (np.abs(Ge[np.ix_(idx, g)]) ** 2).sum(axis=1)[:, eye].sum()
            h2 = (np.abs(Gh[np.ix_(idx, g)]) ** 2).sum(axis=1)[:, eye].sum()
            t2 = (np.abs(tot[np.ix_(idx, g)]) ** 2).sum(axis=1)[:, eye].sum()
            rec.append(dict(q=rq, Ge=np.sqrt(e2), Gh=np.sqrt(h2),
                            tot=np.sqrt(t2), C=t2 / max(e2 + h2, 1e-300)))
        out["shells"][tuple(g)] = rec

        if verbose:
            print(f"--- branches {[m+1 for m in g]}, "
                  f"<hw> = {hw[:, g].mean()*1e3:.2f} meV ---")
            print(f"   {'|q|':>8} {'|Ge|':>10} {'|Gh|':>10} {'|Ge+Gh|':>10} "
                  f"{'C':>8}  verdict")
            for d in rec:
                v = ("destructive" if d["C"] < 0.5 else
                     "independent" if d["C"] < 1.5 else "CONSTRUCTIVE")
                print(f"   {d['q']:8.4f} {d['Ge']*1e3:10.2f} "
                      f"{d['Gh']*1e3:10.2f} {d['tot']*1e3:10.2f} "
                      f"{d['C']:8.3f}  {v}")
            print()
    return out
