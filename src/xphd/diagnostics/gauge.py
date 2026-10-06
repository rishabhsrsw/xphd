"""Which gauge is unfixed: excitonic, phononic, or neither?

Run after `localization` shows signal == null. Distinguishes:

  H1  exciton eigenvector gauge only. One random phase per (beta, q), common
      to all branches. Then D_{nu nu'} = sum_{beta} G_nu G*_nu' is invariant
      and must localise.
  H2  exciton AND phonon gauge. An extra phase per (nu, q); only the diagonal
      D_{nu nu} survives.
  H3  neither -- even the fully invariant diagonal fails, which is not a gauge
      problem at all. Check q ordering and the IBZ unfold.
"""
from __future__ import annotations

import numpy as np

from ..core.mesh import ws_vectors
from ..core.stats import envelope
from .localization import leakage, transform

__all__ = ["gauge_report"]


def gauge_report(G, a1, a2, verbose: bool = True):
    G = np.asarray(G)
    if G.ndim < 4:
        raise ValueError(f"expected (n1, n2, nmod, ...), got {G.shape}")
    n1, n2, nmod = G.shape[:3]
    a_len = float(np.linalg.norm(np.asarray(a1, float)[:2]))
    _, r_norm = ws_vectors(n1, n2, a1, a2)

    def lk(x):
        return leakage(envelope(transform(x)), r_norm, a_len)

    flat = G.reshape(n1, n2, nmod, -1)
    D = np.einsum("xync,xymc->xynm", flat, flat.conj())
    out = {
        "raw": lk(G),
        "D_offdiag": lk(D),
        "D_diag": lk(np.einsum("xynn->xyn", D).real.astype(complex)),
    }

    if out["raw"] > 0.1 and out["D_offdiag"] < 0.1:
        out["verdict"] = "H1: exciton eigenvector gauge only"
    elif out["raw"] > 0.1 and out["D_offdiag"] > 0.1 > out["D_diag"]:
        out["verdict"] = "H2: exciton AND phonon gauges unfixed"
    elif out["D_diag"] > 0.1:
        out["verdict"] = "H3: not a gauge problem -- check ordering / unfold"
    else:
        out["verdict"] = "mixed -- read the numbers"

    if verbose:
        print(f"   {'raw G':<34} {out['raw']:10.3e}")
        print(f"   {'D_nu-nu (exciton gauge removed)':<34} "
              f"{out['D_offdiag']:10.3e}")
        print(f"   {'D_nu-nu (both removed)':<34} {out['D_diag']:10.3e}")
        print(f"\n   {out['verdict']}")
    return out
