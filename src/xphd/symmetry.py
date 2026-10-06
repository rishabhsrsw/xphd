"""Point-group symmetry: stars, unfolding, and violation metrics.

Why unfold the FINAL field rather than symmetrise the inputs
------------------------------------------------------------
E(q) and omega_nu(q) are scalar functions of ONE momentum, so E(Rq) = E(q)
and symmetrising them is exact. They are already symmetric on the source mesh
by construction, so there is nothing to gain.

|G(Q,q)|^2 depends on TWO momenta and transforms with the Gamma_q and D_Q
representation matrices; only the full sum over modes and exciton indices is
invariant, so a naive point-group average of the coupling would be wrong.

What IS invariant is the resulting Gamma(Q) summed over states. So the
symmetry is restored where it is well defined: on the final field, by copying
each irreducible representative to its star. That removes the residual
triangulation noise without assuming anything about the coupling.
"""
from __future__ import annotations

import numpy as np

__all__ = ["reduced_ops", "load_ops", "star_map", "unfold",
           "violation", "full_mesh", "irreducible", "expand"]


#: Tolerance on reduced coordinates. It exists to catch a point that is not a
#: symmetry image at all -- residuals near the mesh half-spacing, 1/(2n) --
#: and must sit well above the noise in the data. Single-precision q-points
#: carry about 1e-7; at 1e-6, star_map starts splitting stars once that noise
#: reaches 4e-7 (measured on hBN's 24x24 mesh with time reversal), leaving a
#: genuine star member as its own "irreducible" point with no error. 1e-4 is
#: 1000x above float32 noise and still 200x below 1/(2n) at n = 24.

def reduced_ops(lat, tol=1e-4):
    """Point-group operations as matrices in reduced coordinates.

    Raises if they are not integer: that would mean the lattice convention
    differs from the one assumed here, and everything downstream is
    meaningless.
    """
    ops, seen = [], set()
    for s in lat.sym_car:
        R = lat.lat @ s @ np.linalg.inv(lat.lat)
        Ri = np.rint(R)
        if np.abs(R - Ri).max() > tol:
            raise SystemExit(
                f"reduced-coordinate rotation is not integer:\n{R}\n"
                "the lattice convention differs from the one assumed here")
        key = tuple(Ri.ravel().astype(int))
        if key not in seen:
            seen.add(key)
            ops.append(Ri)
    return ops


def load_ops(savepath="SAVE"):
    """Read the point group from a yambo ns.db1. Needs yambopy."""
    try:
        from yambopy import YamboLatticeDB
    except ImportError:
        raise SystemExit("load_ops needs yambopy; pass operations directly "
                         "if it is not installed")
    return reduced_ops(YamboLatticeDB.from_db_file(f"{savepath}/ns.db1"))


def _wrap(x):
    return x - np.rint(x)


def star_map(Q_red, ops, tol=1e-4):
    """Cluster Q-points into symmetry stars.

    Returns (reps, owner): `reps` are the indices of the irreducible
    representatives, `owner[i]` is the representative that point i belongs to.
    """
    Q = np.asarray(Q_red, float)
    owner = np.full(len(Q), -1, int)
    reps = []
    for i, q in enumerate(Q):
        if owner[i] != -1:
            continue
        reps.append(i)
        owner[i] = i
        for R in ops:
            d = _wrap(Q - (R @ q)[None, :])
            j = int(np.argmin(np.linalg.norm(d, axis=1)))
            if np.linalg.norm(d[j]) < tol:
                owner[j] = i
    return np.array(reps), owner


def unfold(values, Q_red, ops, tol=1e-4, verbose=True):
    """Replace every point by its star representative's value.

    values : (nq,) or (nq, ...) -- the leading axis is the Q index.

    This is exact for a quantity that is invariant under the point group, and
    removes residual triangulation noise. Use it on the state-SUMMED linewidth
    (or on each state only if you know the manifolds are not mixing), never on
    the coupling.
    """
    v = np.asarray(values, float)
    reps, owner = star_map(Q_red, ops, tol)
    out = v[owner]
    if verbose:
        sizes = np.bincount(owner)[reps]
        rel = np.abs(v - out) / np.maximum(np.abs(out), 1e-30)
        print(f"  {len(reps)} stars over {len(v)} points "
              f"(sizes {sizes.min()}-{sizes.max()})")
        print(f"  max shift from unfolding: {rel.max()*100:.2f}%  "
              f"(this is the triangulation noise being removed)")
    return out


def violation(values, Q_red, ops, tol=1e-4):
    """max relative spread of `values` within a star. Zero by symmetry."""
    v = np.asarray(values, float)
    reps, owner = star_map(Q_red, ops, tol)
    worst = 0.0
    for r in reps:
        m = owner == r
        s = v[m]
        if s.size > 1 and np.abs(s).max() > 0:
            worst = max(worst, float((s.max() - s.min())
                                     / max(abs(s).max(), 1e-30)))
    return worst


def full_mesh(n1, n2):
    """All reduced q-points of an (n1, n2) Gamma-centred mesh, row-major."""
    i, j = np.meshgrid(np.arange(n1), np.arange(n2), indexing="ij")
    return np.stack([i.ravel() / n1, j.ravel() / n2, np.zeros(n1 * n2)], 1)


def irreducible(n1, n2, ops, tol=1e-4):
    """Indices of one representative per star on the full (n1, n2) mesh.

    Generating exciton-phonon archives at these Q only is exact: Gamma(RQ) =
    Gamma(Q) per state, since E_n(RQ) = E_n(Q) and energy-sorted labels match.
    Each archive still holds G(Q, q) for EVERY q, so the integration inside it
    is full-BZ either way -- only the number of Q-points is reduced.

    Returns (reps, owner, Q_full) with reps as 0-based indices into Q_full.
    """
    Q = full_mesh(n1, n2)
    reps, owner = star_map(Q, ops, tol)
    return reps, owner, Q


def expand(values, Q_src, Q_target, ops, tol=1e-4, verbose=True):
    """Carry values computed on a subset of Q onto the full mesh.

    Every target point is matched to a source point through the point group.
    Raises if any target is unreachable -- a silently incomplete field would
    propagate as NaN into the BTE and the fat-band plots.
    """
    v = np.asarray(values)
    Q_src = np.asarray(Q_src, float)
    Q_target = np.asarray(Q_target, float)
    src = np.full(len(Q_target), -1, int)

    for i, q in enumerate(Q_target):
        for R in ops:
            d = _wrap(Q_src - (R @ q)[None, :])
            j = int(np.argmin(np.linalg.norm(d, axis=1)))
            if np.linalg.norm(d[j]) < tol:
                src[i] = j
                break

    miss = int((src < 0).sum())
    if miss:
        raise SystemExit(
            f"{miss} of {len(Q_target)} target points have no symmetry "
            f"partner among the {len(Q_src)} supplied. Either the source set "
            f"is not a complete set of representatives, or the operations do "
            f"not match the lattice.")
    if verbose:
        print(f"  expanded {len(Q_src)} -> {len(Q_target)} Q-points "
              f"({len(Q_target)/len(Q_src):.1f}x)")
    return v[src]


def full_mesh(n1, n2=None):
    """Reduced coordinates of a full Gamma-centred (n1 x n2) mesh, row-major."""
    n2 = n2 or n1
    i, j = np.meshgrid(np.arange(n1), np.arange(n2), indexing="ij")
    return np.stack([i.ravel() / n1, j.ravel() / n2,
                     np.zeros(n1 * n2)], 1)


def irreducible(n1, ops, n2=None, verbose=True):
    """One representative per star of an (n1 x n2) mesh.

    Returns (indices, Q_full). Since Gamma(Q) summed over states is invariant
    under the point group, only these need computing -- the rest follow by
    symmetry. On a 24x24 hexagonal mesh that is 61 of 576.
    """
    Q = full_mesh(n1, n2)
    reps, owner = star_map(Q, ops)
    if verbose:
        sizes = np.bincount(owner)[reps]
        print(f"  {len(reps)} stars over {len(Q)} points "
              f"(sizes {sizes.min()}-{sizes.max()}), "
              f"{len(Q)/len(reps):.1f}x saving")
    return np.sort(reps), Q


def expand(values, Q_computed, Q_full, ops, tol=1e-4, verbose=True):
    """Expand a field computed on a SUBSET of Q onto the full list.

    values      : (nq_computed,) or (nq_computed, ...) -- leading axis is Q
    Q_computed  : the Q-points actually computed
    Q_full      : every Q-point wanted

    Each output point takes the value of whichever computed point lies in its
    star. Raises if any star is empty, since a silently-missing star would
    show up later as a hole in the plot rather than as an error.
    """
    v = np.asarray(values, float)
    Qc = np.asarray(Q_computed, float)
    Qf = np.asarray(Q_full, float)

    src = np.full(len(Qf), -1, int)
    for k, qc in enumerate(Qc):
        for R in ops:
            d = _wrap(Qf - (R @ qc)[None, :])
            hit = np.linalg.norm(d, axis=1) < tol
            src[hit & (src < 0)] = k

    missing = int((src < 0).sum())
    if missing:
        raise SystemExit(
            f"{missing} of {len(Qf)} points are in no star of the computed "
            f"set. The subset does not cover the irreducible wedge -- check "
            f"it against xphd.symmetry.irreducible().")
    if verbose:
        print(f"  expanded {len(Qc)} computed -> {len(Qf)} points "
              f"({len(Qf)/max(len(Qc),1):.1f}x)")
    return v[src]
