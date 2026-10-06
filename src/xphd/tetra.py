"""Analytic linear-triangle ("2D tetrahedron") Brillouin-zone integration.

There is no broadening parameter here: the delta functions are integrated
exactly for a piecewise-linear interpolant of the delta argument over each
triangle of the mesh.

For a triangle of fractional area A with sorted vertex values e1 <= e2 <= e3
the isoline density at the target is

    e in [e1,e2]:  Dt = 2A (e-e1) / [(e2-e1)(e3-e1)]
    e in [e2,e3]:  Dt = 2A (e3-e) / [(e3-e2)(e3-e1)]

and since a linear function averaged along a segment equals the mean of its
endpoint values, the vertex weights follow directly. Only AREA RATIOS enter,
so the result is invariant under the affine map from reduced to cartesian
coordinates: a hexagonal cell needs no special treatment for correctness,
only for conditioning (see `diag`).

Reference: Jepsen & Andersen (1971); Lehmann & Taut (1972); Rath & Freeman
(1975). The Bloechl curvature correction is for occupation-type integrals and
does not apply to the response-function form used here (Kawamura 2014).
"""
from __future__ import annotations

import numpy as np

__all__ = ["delta_weights_2d", "triangle_weights"]


def _tri_weights(e, area, flat_width):
    """
    Vertex weights for  int_T f(q) delta(D(q)) dA / Omega_BZ  ~  sum_i w_i f(q_i).

    Parameters
    ----------
    e          : (T,3) float, vertex values of D (the delta argument; target is 0)
    area       : float, fractional area of one triangle (sum over all triangles = 1)
    flat_width : float (eV), degeneracy tolerance.  Triangles flatter than this
                 are regularised with a normalised Lorentzian of this half-width.

    Returns
    -------
    w : (T,3) float, in the ORIGINAL vertex order.
    """
    e = np.asarray(e, dtype=float)
    order = np.argsort(e, axis=-1, kind="stable")
    es = np.take_along_axis(e, order, axis=-1)
    e1, e2, e3 = es[:, 0], es[:, 1], es[:, 2]

    d21 = e2 - e1
    d31 = e3 - e1
    d32 = e3 - e2

    w = np.zeros_like(es)

    # --- branch 1: e1 <= 0 < e2 ; isoline cuts edges (1,2) and (1,3)
    m = (e1 <= 0.0) & (0.0 < e2) & (d21 > flat_width) & (d31 > flat_width)
    if m.any():
        a = (-e1[m]) / d21[m]
        b = (-e1[m]) / d31[m]
        Dt = 2.0 * area * (-e1[m]) / (d21[m] * d31[m])
        w[m, 0] = Dt * (2.0 - a - b) * 0.5
        w[m, 1] = Dt * a * 0.5
        w[m, 2] = Dt * b * 0.5

    # --- branch 2: e2 <= 0 <= e3 ; isoline cuts edges (1,3) and (2,3)
    m = (e2 <= 0.0) & (0.0 <= e3) & (d32 > flat_width) & (d31 > flat_width)
    if m.any():
        c = e3[m] / d31[m]
        d = e3[m] / d32[m]
        Dt = 2.0 * area * e3[m] / (d32[m] * d31[m])
        w[m, 0] = Dt * c * 0.5
        w[m, 1] = Dt * d * 0.5
        w[m, 2] = Dt * (2.0 - c - d) * 0.5

    # --- degenerate/flat triangles: regularise with a narrow Lorentzian
    m = d31 <= flat_width
    if m.any():
        eav = (e1[m] + e2[m] + e3[m]) / 3.0
        lor = (1.0 / np.pi) * flat_width / (eav ** 2 + flat_width ** 2)
        w[m, :] = (area * lor / 3.0)[:, None]

    out = np.empty_like(w)
    np.put_along_axis(out, order, w, axis=-1)
    return out


def delta_weights_2d(D, flat_width=1e-4, diag="anti"):
    """
    Integration weights on a periodic 2D mesh for  (1/Omega) int d2q f(q) delta(D(q)).

    Parameters
    ----------
    D          : (n1,n2) array, the delta argument sampled on the mesh
                 q = (i/n1, j/n2, 0),  i=0..n1-1,  j=0..n2-1.
    flat_width : eV, degeneracy tolerance (see `_tri_weights`).
    diag       : "anti" splits each microcell along the (i+1,j)-(i,j+1) diagonal
                 (the SHORT diagonal for a hexagonal lattice, best-conditioned
                 triangles); "main" splits along (i,j)-(i+1,j+1).

    Returns
    -------
    W : (n1,n2) array.  Then  sum_q W[q]*f[q]  is the integral.
        In particular sum(W) over an energy sweep integrates to 1 per band.
    """
    D = np.asarray(D, dtype=float)
    n1, n2 = D.shape
    N = n1 * n2
    idx = np.arange(N).reshape(n1, n2)

    def corners(a):
        a00 = a
        a10 = np.roll(a, -1, axis=0)
        a01 = np.roll(a, -1, axis=1)
        a11 = np.roll(np.roll(a, -1, axis=0), -1, axis=1)
        return a00, a10, a01, a11

    d00, d10, d01, d11 = corners(D)
    i00, i10, i01, i11 = corners(idx)

    if diag == "anti":                       # diagonal (1,0)-(0,1)
        tA = (d00, d10, d01);  iA = (i00, i10, i01)
        tB = (d10, d11, d01);  iB = (i10, i11, i01)
    elif diag == "main":                     # diagonal (0,0)-(1,1)
        tA = (d00, d10, d11);  iA = (i00, i10, i11)
        tB = (d00, d11, d01);  iB = (i00, i11, i01)
    else:
        raise ValueError("diag must be 'anti' or 'main'")

    Dv = np.concatenate([np.stack(tA, -1).reshape(-1, 3),
                         np.stack(tB, -1).reshape(-1, 3)], axis=0)
    Iv = np.concatenate([np.stack(iA, -1).reshape(-1, 3),
                         np.stack(iB, -1).reshape(-1, 3)], axis=0)

    area = 1.0 / (2.0 * N)                   # fractional area of one triangle
    w = _tri_weights(Dv, area, flat_width)

    W = np.bincount(Iv.ravel(), weights=w.ravel(), minlength=N)
    return W.reshape(n1, n2)


def triangle_weights(vertex_values, area: float = 1.0, flat_width: float = 0.0):
    """Weights for explicit triangles, targeting D = 0.

    `delta_weights_2d` builds its own triangulation from a grid; this takes
    the vertex values directly, which is what the BTE assembly needs since it
    triangulates once and reuses the connectivity for every (mode, band) pair.

    vertex_values : (3, ...) values of the delta argument at the three vertices
    returns       : (3, ...) weights, same layout

    Both entry points share `_tri_weights`, so the flat-branch regularisation
    cannot drift between them. It integrates to area * L(eps_bar) for a flat
    triangle -- note an independent implementation that folds the fractional
    area into a prefactor and ALSO multiplies by the unit-triangle area 1/2
    comes out a factor of two low.
    """
    v = np.asarray(vertex_values, float)
    if v.shape[0] != 3:
        raise ValueError(f"first axis must be 3, got {v.shape}")
    tail = v.shape[1:]
    flat = np.moveaxis(v, 0, -1).reshape(-1, 3)
    w = _tri_weights(flat, area, flat_width)
    return np.moveaxis(w.reshape(tail + (3,)), -1, 0)
