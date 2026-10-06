"""Uniform 2D q-meshes: indexing, real-space duals, and shells.

Everything here assumes a Gamma-centred (n1 x n2 x 1) mesh with q given in
REDUCED coordinates, q = (i/n1, j/n2, 0).
"""
from __future__ import annotations

import numpy as np

__all__ = ["mesh_indices", "to_grid", "blockwise", "ws_vectors",
           "hex_norm", "shells", "shift_grid"]


def mesh_indices(q_red, n1: int, n2: int, tol: float = 1e-4):
    """Reduced q-points -> integer mesh indices, asserting exact coverage.

    Use this instead of a nearest-neighbour search: on a hexagonal lattice
    Euclidean distance in reduced coordinates is not the right metric and
    silently mis-assigns points.

    The residual is measured in units of the mesh index, so an error in the
    reduced coordinates is multiplied by n. A point is assigned to the wrong
    index only as the residual approaches 0.5; the tolerance exists to catch
    that, and Cartesian coordinates passed by mistake, which give residuals of
    order 0.1.

    It must NOT be set near machine precision. An electron-phonon database
    written in single precision carries q-points good to about 1e-7
    relative, which becomes ~1e-6 in index units on a 24x24 mesh and grows
    with n. A default of 1e-6 rejected a valid hBN archive for exactly this
    reason. 1e-4 sits two orders above float32 noise at any practical mesh
    and three below a genuine misassignment; the coverage check below still
    catches a wrong mesh size or a shifted grid.
    """
    q = np.mod(np.asarray(q_red, float), 1.0)
    fi, fj = q[:, 0] * n1, q[:, 1] * n2
    i = np.rint(fi).astype(int) % n1
    j = np.rint(fj).astype(int) % n2
    err = max(np.abs(fi - np.rint(fi)).max(), np.abs(fj - np.rint(fj)).max())
    if err > tol:
        raise ValueError(
            f"q-points are not on a {n1}x{n2} mesh (max residual {err:.2e} "
            f"in index units, tolerance {tol:.0e}); are they in reduced "
            f"coordinates, and is the mesh size right?")
    flat = i * n2 + j
    if len(np.unique(flat)) != n1 * n2 or len(flat) != n1 * n2:
        raise ValueError(
            f"mesh not covered exactly once: {len(np.unique(flat))} unique of "
            f"{n1*n2} expected, {len(flat)} points given. Unfold the IBZ first.")
    return i, j


def to_grid(flat, i, j, n1: int, n2: int):
    """Scatter a (nq, ...) array onto an (n1, n2, ...) mesh."""
    flat = np.asarray(flat)
    out = np.zeros((n1, n2) + flat.shape[1:], dtype=flat.dtype)
    out[i, j] = flat
    return out


def blockwise(a, r: int):
    """Piecewise-constant upsampling by NEAREST coarse point, with wrap.

    np.repeat maps fine (I,J) -> coarse (I//r, J//r), i.e. floor, which puts
    each coarse point at the left EDGE of its cell rather than the centre.
    Near Gamma that is not cosmetic: fine points at q ~ 0.99 fold to |q| ~ 0
    but floor hands them the coupling from q = -1/n. Pairing a finite acoustic
    coupling with a vanishing omega makes the Bose factor diverge and produces
    spurious low-temperature absorption.
    """
    a = np.asarray(a)
    if r == 1:
        return a
    n1, n2 = a.shape[:2]
    i = np.rint(np.arange(n1 * r) / r).astype(int) % n1
    j = np.rint(np.arange(n2 * r) / r).astype(int) % n2
    return a[np.ix_(i, j)]


def ws_vectors(n1: int, n2: int, a1, a2):
    """Cartesian R vectors of the Born-von Karman lattice, folded into the
    Wigner-Seitz cell of the supercell (minimum image over all 9 images).

    Correct for oblique and hexagonal lattices, where wrapping each index
    independently into [-n/2, n/2) gives the wrong shortest vector.
    """
    a1 = np.asarray(a1, float)[:2]
    a2 = np.asarray(a2, float)[:2]
    m1 = np.arange(n1)[:, None] * np.ones((1, n2), int)
    m2 = np.ones((n1, 1), int) * np.arange(n2)[None, :]

    best_n = np.full((n1, n2), np.inf)
    best_v = np.zeros((n1, n2, 2))
    for p in (-1, 0, 1):
        for s in (-1, 0, 1):
            v = (m1 + p * n1)[..., None] * a1 + (m2 + s * n2)[..., None] * a2
            nrm = np.linalg.norm(v, axis=-1)
            better = nrm < best_n - 1e-10
            best_n = np.where(better, nrm, best_n)
            best_v = np.where(better[..., None], v, best_v)
    return best_v, best_n


def hex_norm(q_red):
    """|q| in the hexagonal reciprocal metric, folded to [-1/2, 1/2)."""
    f = ((np.asarray(q_red, float)[:, :2] + 0.5) % 1.0) - 0.5
    return np.sqrt(f[:, 0] ** 2 + f[:, 1] ** 2 + f[:, 0] * f[:, 1])


def shells(r, tol: float = 1e-6):
    """Group q-points into stars of equal |q|. Returns [(|q|, indices), ...]."""
    r = np.asarray(r, float)
    out, seen = [], np.zeros(len(r), bool)
    for k in np.argsort(r):
        if seen[k]:
            continue
        m = np.abs(r - r[k]) < tol
        seen |= m
        out.append((float(r[k]), np.where(m)[0]))
    return out


def shift_grid(field, iQ: int, jQ: int):
    """E(Q+q) from E(q) by rolling with the integer mesh index of Q."""
    return np.roll(np.roll(np.asarray(field), -iQ, axis=0), -jQ, axis=1)
