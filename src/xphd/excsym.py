"""Symmetry labels for exciton states.

An exciton at momentum Q is characterised by how it transforms under the
little group of Q. Given the electronic rotation matrices D^R_{mn}(k) -- the
same Dmats that LetzElPhC writes -- the exciton envelope rotates as

    (R A^S)_{vc,k} = sum_{v'c'} D^{c*}_{cc'}(k) A^S_{v'c',R^-1 k} D^{v}_{v'v}(k)

and the representation carried by a degenerate manifold follows from

    D^{(S)}_{ST}(R) = <A^S | R A^T>,      chi_S(R) = Tr D^{(S)}(R).

Characters are then matched against a character table. This module does the
projection and the matching; it does NOT attempt to guess the little group,
which must be supplied or read from the SAVE.

Two cautions worth stating plainly. Characters are only meaningful for a
COMPLETE degenerate manifold -- a single state drawn from a doublet has no
well-defined character, because the two partners mix under rotation. And the
trace is basis-independent only within that manifold, so the degeneracy
grouping has to be right before any label is assigned.
"""
from __future__ import annotations

import numpy as np

__all__ = ["CHARACTER_TABLES", "CLASSES", "HIGH_SYMMETRY", "LITTLE_GROUP",
           "rep_matrix", "rep_matrix_Q", "shift_map", "characters",
           "assign_irreps", "degeneracy_groups", "sigma_h_trace",
           "little_group_ops", "classify_d3h", "stabilizer", "group_name",
           "class_characters"]

# High-symmetry points of a 2D hexagonal zone, in reduced coordinates, and
# the little group at each. `Q` and `Q'` are the midpoints of Gamma-K, kept
# because intervalley scattering often passes through them.
HIGH_SYMMETRY = {
    "G":  (0.0, 0.0),      "M":  (0.5, 0.0),      "K":  (1/3, 1/3),
    "M'": (0.0, 0.5),      "M''": (0.5, 0.5),     "K'": (2/3, 2/3),
    "Q":  (1/6, 1/6),      "Q'": (5/6, 5/6),
}

# Classes of the full group that survive at each point. Everything else is
# absent because it moves Q to an inequivalent wavevector.
#   sigma_h and E act trivially on in-plane q and survive everywhere.
LITTLE_GROUP = {
    "G":  ("D3h", ("E", "2C3", "3C2", "sh", "2S3", "3sv")),
    "M":  ("C2v", ("E", "3C2", "sh", "3sv")),
    "M'": ("C2v", ("E", "3C2", "sh", "3sv")),
    "M''":("C2v", ("E", "3C2", "sh", "3sv")),
    "K":  ("C3h", ("E", "2C3", "sh", "2S3")),
    "K'": ("C3h", ("E", "2C3", "sh", "2S3")),
    "Q":  ("Cs",  ("E", "sh")),
    "Q'": ("Cs",  ("E", "sh")),
}

# chi[irrep][class]; classes in the order given by CLASSES
CLASSES = {
    "D3h": ("E", "2C3", "3C2", "sh", "2S3", "3sv"),
    "C3v": ("E", "2C3", "3sv"),
    "C3h": ("E", "2C3", "sh", "2S3"),
    "C3":  ("E", "2C3"),
}

_W = np.exp(2j * np.pi / 3)
_W2 = _W ** 2

CLASSES["C2v"] = ("E", "3C2", "sh", "3sv")
CLASSES["Cs"] = ("E", "sh")
CLASSES["C1"] = ("E",)

CHARACTER_TABLES = {
    "D3h": {"A1'":  (1,  1,  1,  1,  1,  1),
            "A2'":  (1,  1, -1,  1,  1, -1),
            "E'":   (2, -1,  0,  2, -1,  0),
            "A1''": (1,  1,  1, -1, -1, -1),
            "A2''": (1,  1, -1, -1, -1,  1),
            "E''":  (2, -1,  0, -2,  1,  0)},
    "C3v": {"A1": (1,  1,  1),
            "A2": (1,  1, -1),
            "E":  (2, -1,  0)},
    # C3h is Abelian: every irrep is ONE dimensional, with w = exp(2i pi/3).
    # Nothing pairs E'_1 with E'_2 at K. sigma_v combined with time reversal
    # does fix K, but it maps each C3 eigenvalue to ITSELF (A C3 A^-1 = C3^-1,
    # and the antiunitary conjugation undoes the inversion) and squares to +1,
    # so it adds no degeneracy -- as in hBN and the TMDs, whose K-valley
    # states carry complex C3 eigenvalues without partners. A singlet is
    # matched by its own C3 character.
    #
    # At K the A / E_1 / E_2 label also depends on WHICH C3 axis is used --
    # through one atom, the other, or the hexagon centre. A Bloch sum at K
    # picks up a phase from sites off the axis, so moving the axis multiplies
    # every C3 eigenvalue by w or w^2. Labels here are relative to the origin
    # of the SAVE; the sigma_h parity (prime / double prime) is absolute.
    "C3h": {"A'":    (1,  1,  1,  1),
            "E'_1":  (1, _W,  1, _W),
            "E'_2":  (1, _W2, 1, _W2),
            "A''":   (1,  1, -1, -1),
            "E''_1": (1, _W, -1, -_W),
            "E''_2": (1, _W2, -1, -_W2),
            # a NEAR-degenerate pair, grouped by --deg-tol, whose summed
            # characters are those of E'_1 + E'_2 -- an approximate degeneracy
            # such as a 2p+/2p- envelope pair split only by trigonal warping,
            # not one symmetry requires
            "E'_1+E'_2":   (2, -1, 2, -1),
            "E''_1+E''_2": (2, -1, -2, 1)},
    "C3":  {"A": (1,  1),
            "E": (2, -1)},
    # at M the surviving operations are E, one C2', sigma_h and one sigma_v;
    # labels follow the C2v convention with sigma_h as the sigma(xz) plane
    "C2v": {"A1": (1,  1,  1,  1),
            "A2": (1,  1, -1, -1),
            "B1": (1, -1,  1, -1),
            "B2": (1, -1, -1,  1)},
    "Cs":  {"A'":  (1,  1),
            "A''": (1, -1)},
    "C1":  {"A": (1,)},
}


def degeneracy_groups(E, tol=1e-4):
    """Indices grouped into degenerate manifolds. E in eV, tol in eV.

    Characters are defined for a manifold, not a state: partners of a doublet
    mix under rotation, so a trace taken over part of one is meaningless.
    """
    E = np.asarray(E, float)
    groups, cur = [], [0]
    for k in range(1, len(E)):
        if abs(E[k] - E[cur[-1]]) < tol:
            cur.append(k)
        else:
            groups.append(cur)
            cur = [k]
    groups.append(cur)
    return groups


def rep_matrix(A, Dc, Dv, kmap, group):
    """<A^S | R A^T> for the states in `group`.

    A     : (nexc, nk, nv, nc) exciton envelopes
    Dc,Dv : (nk, nc, nc) and (nk, nv, nv) electronic rotation matrices for
            the operation R
    kmap  : (nk,) index such that R^-1 k_i = k[kmap[i]]
    """
    A = np.asarray(A)
    RA = np.einsum("kcd,ksdb,kba->ksca",
                   np.conj(Dc), A[:, kmap].transpose(1, 0, 3, 2), Dv)
    RA = RA.transpose(1, 0, 3, 2)                    # back to (S, k, v, c)
    g = np.asarray(group, int)
    return np.einsum("skvc,tkvc->st", np.conj(A[g]), RA[g])


def characters(A, ops, group, E=None, tol=1e-4, verbose=True):
    """Character of a degenerate manifold under each operation.

    ops : list of (label, Dc, Dv, kmap) for each operation of the little group
    """
    chi = {}
    for lab, Dc, Dv, kmap in ops:
        D = rep_matrix(A, Dc, Dv, kmap, group)
        chi[lab] = complex(np.trace(D))
        if verbose:
            uni = np.abs(D @ np.conj(D).T - np.eye(len(group))).max()
            if uni > 1e-3:
                print(f"    [warn] {lab}: representation is not unitary "
                      f"(dev {uni:.1e}); the manifold is probably incomplete")
    return chi


def assign_irreps(chi_by_class, table="D3h", tol=0.15, verbose=True):
    """Nearest irrep by character, with the residual reported.

    chi_by_class : {class label: character}, classes as in CLASSES[table]
    Returns (name, residual). A large residual means the state does not carry
    a single irrep -- most often because the degeneracy grouping is wrong or
    the little group supplied is larger than the true one.
    """
    if table not in CHARACTER_TABLES:
        raise SystemExit(f"no character table for {table!r}; "
                         f"have {sorted(CHARACTER_TABLES)}")
    cls = CLASSES[table]
    missing = [c for c in cls if c not in chi_by_class]
    if missing:
        raise SystemExit(f"characters missing for classes {missing}")
    v = np.array([chi_by_class[c] for c in cls], complex)
    best, resid = None, np.inf
    for name, row in CHARACTER_TABLES[table].items():
        # complex, not float: the C3h conjugate partners differ only in
        # the phase of their rotation characters, and casting to real
        # makes them indistinguishable
        r = float(np.abs(v - np.array(row, complex)).max())
        if r < resid:
            best, resid = name, r
    if verbose and resid > tol:
        print(f"    [warn] closest irrep {best} but residual {resid:.3f} "
              f"exceeds {tol}; check the degeneracy grouping and the little "
              f"group")
    return best, resid


def shift_map(kf, Qfrac, tol=1e-5):
    """Index of k_i + Q for every k_i, or (None, worst) if it leaves the grid.

    yambo stores a finite-Q exciton A^{SQ}_kvc with its ELECTRON at the table
    index k and its HOLE at k+Q, so the valence rotation is evaluated at the
    shifted point. This was established on GaN: of the four ways of attaching
    the shift (electron or hole, +Q or -Q), only this one gives an exact mirror
    parity, |chi| = 1, for every state at all 61 Q-points; the others reach
    at most 35%. (Earlier versions put the electron at k+Q, which is exact
    only at Q = 0.)
    """
    d = np.asarray(kf)[:, None, :] - (np.asarray(kf) + np.asarray(Qfrac))[None, :, :]
    d -= np.rint(d)
    dist = np.linalg.norm(d, axis=-1)
    kQ = np.argmin(dist, axis=1)
    worst = float(dist[np.arange(len(kf)), kQ].max())
    if worst > tol or len(np.unique(kQ)) != len(kf):
        return None, worst
    return kQ, worst


def rep_matrix_Q(A, Dc, Dv, kmap, kQ, group):
    """<A^S | R A^T> for an exciton at finite Q.

    `rep_matrix` evaluates both rotations at k, which is right only at Q = 0.
    Here the electron is at the table index k and the hole at k+Q (see
    shift_map), so the VALENCE matrix is taken at the shifted index kQ.
    Passing kQ = arange(nk) reproduces `rep_matrix` exactly.
    """
    A = np.asarray(A)
    RA = np.einsum("kcd,ksdb,kba->ksca",
                   np.conj(np.asarray(Dc)),
                   A[:, kmap].transpose(1, 0, 3, 2), np.asarray(Dv)[kQ])
    RA = RA.transpose(1, 0, 3, 2)
    g = np.asarray(group, int)
    return np.einsum("skvc,tkvc->st", np.conj(A[g]), RA[g])


def sigma_h_trace(A, sh_op, kQ, state):
    """chi(sigma_h) for a single state.

    One number, no character table and no degeneracy grouping in the way.
    In a planar layer sigma_h is a symmetry at EVERY in-plane q, so this
    parity is conserved along a band: a change of sign between two
    wavevectors marks a crossing, not a gradual evolution, and a crossing
    between opposite parities opens no gap and leaves the dispersion smooth.
    """
    # operations come as (label, R, Dc, Dv, kmap) from yambo.spatial_ops, or
    # as the older (label, Dc, Dv, kmap); take the last three either way so
    # the rotation matrix is never passed where a D-matrix belongs
    Dc, Dv, kmap = sh_op[-3], sh_op[-2], sh_op[-1]
    return float(np.trace(rep_matrix_Q(A, Dc, Dv, kmap, kQ, [state])).real)


def little_group_ops(ops, point):
    """Operations of `ops` that survive at a high-symmetry point.

    `point` is a key of LITTLE_GROUP: 'G', 'M', 'K', 'Q' and their primed
    partners. Returns (table_name, filtered_ops).
    """
    if point not in LITTLE_GROUP:
        raise SystemExit(f"unknown point {point!r}; have "
                         f"{sorted(LITTLE_GROUP)}")
    table, classes = LITTLE_GROUP[point]
    keep = [o for o in ops if o[0] in classes]
    return table, keep


def classify_d3h(R):
    """Class of a reduced-coordinate D3h operation.

    Trace and determinant are basis independent; R[2,2] is the out-of-plane
    sign because the third lattice vector lies along z in a slab.
    """
    tr, det, r33 = (int(round(np.trace(R))), int(round(np.linalg.det(R))),
                    int(round(R[2, 2])))
    if det == 1:
        if tr == 3:
            return "E"
        if tr == 0:
            return "2C3"
        if tr == -1 and r33 == -1:
            return "3C2"
    else:
        if tr == 1 and r33 == -1:
            return "sh"
        if tr == -2:
            return "2S3"
        if tr == 1 and r33 == 1:
            return "3sv"
    return None


def stabilizer(ops, Q, tol=1e-5):
    """Operations with R Q = Q modulo a reciprocal lattice vector.

    Computed rather than selected by class name: at M only one C2' and one
    sigma_v fix a given M, and choosing "3C2" and "3sv" by name admits all
    three of each. ops are tuples whose second entry is the reduced matrix.
    """
    Q = np.asarray(Q, float)
    out = []
    for o in ops:
        dq = Q @ np.asarray(o[1]).T - Q
        if np.abs(dq - np.rint(dq)).max() < tol:
            out.append(o)
    return out


def group_name(stab):
    """Point group of a D3h stabilizer, from its order and classes."""
    n = len(stab)
    labs = {o[0] for o in stab}
    if n == 12:
        return "D3h"
    if n == 6 and "2C3" in labs:
        return "C3h"
    if n == 4:
        return "C2v"
    if n == 2 and "sh" in labs:
        return "Cs"
    if n == 1:
        return "C1"
    raise SystemExit(f"unrecognised little group of order {n}: {sorted(labs)}")


def class_characters(A, stab, kQ, manifold):
    """Characters of a degenerate manifold -- a list of state indices, not a
    point group -- taking one representative per class.

    Characters are class functions, so any member of a class gives the same
    value. In C3h, C3 and C3^2 are distinct classes with conjugate characters
    w and w^2; averaging them as a D3h class label would suggest destroys the
    phase separating E'_1 from E'_2. The _1/_2 label then depends on the
    sense of rotation chosen, which is a convention.
    """
    chi = {}
    for lab, R, Dc, Dv, kmap in stab:
        if lab not in chi:
            chi[lab] = complex(np.trace(rep_matrix_Q(A, Dc, Dv, kmap, kQ,
                                                     manifold)))
    return chi
