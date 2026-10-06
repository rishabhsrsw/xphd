"""Readers for yambo / LetzElPhC databases, shared by every command that
needs the Bethe-Salpeter eigenvectors or the symmetry operations.

Each of these used to exist in several standalone scripts, and the copies
drifted: one lacked the guard below, and three returned different tuples.
They live here once so that the irreducible-index convention -- the thing
that silently broke the archive-based pairing -- is defined in one place.

yambopy is imported inside each function, so importing this module does not
require it; only calling a reader does.
"""
from __future__ import annotations

import numpy as np

__all__ = ["lattice", "load_excitons", "full_zone_kpoints", "ibz_parents",
           "spatial_ops", "load_dmats", "sigma_h_op"]


def lattice(save):
    """YamboLatticeDB for a SAVE directory."""
    from yambopy import YamboLatticeDB
    return YamboLatticeDB.from_db_file(filename=f"{save}/ns.db1")


def load_excitons(path, ylat, neigs=None):
    """Exciton envelopes rebuilt from the BSE transition table.

    Returns (exc, A, E) with A of shape (n, k, v, c). Built from the table
    rather than from yexc.get_Akcv(), whose own comment limits it to "TDA and
    no spin pol".
    """
    from yambopy import YamboExcitonDB
    kw = {} if neigs is None else {"neigs": neigs}
    exc = YamboExcitonDB.from_db_file(ylat, filename=path, **kw)
    tab = exc.table.astype(int)
    k = tab[:, 0] - tab[:, 0].min()
    v = tab[:, 1] - tab[:, 1].min()
    c = tab[:, 2] - tab[:, 2].min()
    ev = exc.eigenvectors
    if ev.shape[0] == len(k) and ev.shape[1] != len(k):
        ev = ev.T
    A = np.zeros((ev.shape[0], k.max() + 1, v.max() + 1, c.max() + 1),
                 complex)
    A[:, k, v, c] = ev
    return exc, A, np.asarray(exc.eigenvalues).real


def full_zone_kpoints(ylat):
    """Full-zone k in reduced coordinates, in the lattice's own ordering."""
    kf = np.asarray(ylat.car_kpoints)[:, :3] @ np.linalg.inv(
        np.asarray(ylat.rlat))
    kf = np.mod(kf, 1.0)
    kf[np.abs(kf - 1.0) < 1e-6] = 0.0
    return kf


def ibz_parents(ylat):
    """(kidx, first): the irreducible parent of each full-zone point, and the
    first full-zone image of each irreducible point.

    ndb.BS_diago_Q{j+1} sits at kf[first[j]]. Built from kpoints_indexes
    rather than lat.ibz_kpoints, whose units vary between yambo versions and
    which does not always land on the mesh.
    """
    kidx = np.asarray(ylat.kpoints_indexes, dtype=int)
    first = np.full(int(kidx.max()) + 1, -1, dtype=int)
    for i, j in enumerate(kidx):
        if first[j] < 0:
            first[j] = i
    if (first < 0).any():
        raise SystemExit("some irreducible index never appears in "
                         "kpoints_indexes")
    return kidx, first


def load_dmats(path):
    """Electronic rotation matrices, with the spin axis dropped."""
    D = np.load(path)
    return D[:, :, 0, :, :] if D.ndim == 5 else D


def spatial_ops(save, kf, D, nv):
    """(label, R_reduced, Dc, Dv, kmap) for each SPATIAL operation.

    load_ops returns REDUCED-coordinate matrices -- a hexagonal C3 comes back
    as integers -- so the rotation acts on fractional k. The second half of
    the list repeats the first with time reversal and carries different
    D-matrices, so it is skipped. kmap[i] is the index of R^-1 k_i.
    """
    from .symmetry import load_ops
    from .excsym import classify_d3h
    R_all = np.asarray(load_ops(save))
    ops = []
    for iop in range(len(R_all) // 2):
        R = R_all[iop]
        lab = classify_d3h(R)
        if lab is None:
            continue
        d = kf[:, None, :] - (kf @ R.T)[None, :, :]
        d -= np.rint(d)
        dist = np.linalg.norm(d, axis=-1)
        kmap = np.argmin(dist, axis=1)
        if (dist[np.arange(len(kf)), kmap].max() > 1e-5
                or len(np.unique(kmap)) != len(kmap)):
            continue
        ops.append((lab, R, D[iop, :, nv:, nv:], D[iop, :, :nv, :nv], kmap))
    return ops


def sigma_h_op(ops):
    """The horizontal mirror from a spatial_ops list, or raise."""
    for o in ops:
        if o[0] == "sh":
            return o
    raise SystemExit("no horizontal mirror among the operations; is the "
                     "layer planar?")
