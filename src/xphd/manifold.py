"""Mirror parity of near-degenerate manifolds.

An exact symmetry such as sigma_h gives every NON-degenerate exciton an
exact parity. Two states of opposite parity lying close in energy (with spin-
orbit coupling such pairs are common) are a different matter: a small
symmetry-breaking error in the computed BSE Hamiltonian, of order 10 micro-eV,
is enough for the solver to return them mixed, and their individual
characters chi = <S|sigma_h|S> then fall well short of +-1. Per-state parity
must either guess or exclude them.

The fix is to work with the manifold, not the state: within each group of
near-degenerate states the sigma_h matrix M_mn = <m|sigma_h|n> is computed and
diagonalised. Its eigenvectors are the parity eigenstates the solver could
equally well have returned; its eigenvalues are +-1 for a closed manifold.
Any quantity carried by the manifold -- here the coupling of one initial
state into it -- is then split exactly by parity, with nothing excluded.

Conventions, matching xphd generate:
  * the archive stores G[q, nu, initial, final] = <final, Q+q| dV | initial, Q>
    (the final envelope conjugated, as in yambopy's exciton_X_matelem);
  * the state dV|initial> projected on a manifold is sum_n G_n |n>, so its
    weight on the eigenvector v_i of M is |v_i^H G|^2;
  * a final state reached from its irreducible parent through an operation
    containing time reversal is the complex conjugate image of the parent,
    so its manifold matrix is conj(M_parent).
"""
from __future__ import annotations

import numpy as np


def energy_groups(E, tol):
    """Group labels for energies E (any order): neighbours in sorted order
    closer than tol share a group. Returns an int array, labels 0, 1, ..."""
    E = np.asarray(E, float)
    order = np.argsort(E)
    lab = np.empty(len(E), int)
    g = 0
    for i, j in enumerate(order):
        if i > 0 and E[j] - E[order[i - 1]] >= tol:
            g += 1
        lab[j] = g
    return lab


def sigma_blocks(A, sh_op, kQ, groups, rep_matrix_Q):
    """Block-diagonal sigma_h matrix of the states in A, one block per group.

    A: envelopes (n, k, v, c) as from load_excitons; sh_op the sigma_h
    operation (..., Dc, Dv, kmap); kQ the shifted index of the hole.
    rep_matrix_Q is passed in to keep this module free of imports.
    """
    Dc, Dv, kmap = sh_op[-3], sh_op[-2], sh_op[-1]
    n = len(groups)
    M = np.zeros((n, n), complex)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        M[np.ix_(idx, idx)] = rep_matrix_Q(A, Dc, Dv, kmap, kQ, list(idx))
    return M


def time_reversed(ylat):
    """For every full-zone point: is it reached from its irreducible parent by
    an operation containing time reversal? The same rule as xphd generate."""
    sidx = np.asarray(ylat.symmetry_indexes, dtype=int)
    nsym = len(ylat.sym_car)
    return sidx >= nsym / (1 + int(np.rint(ylat.time_rev)))


def split_by_parity(G, M, groups, tol=0.5):
    """Split |G|^2 by the parity eigenstates of each manifold.

    G: (nmod, n) complex couplings of one initial state into the n final
    states at one q; M: (n, n) block-diagonal sigma_h matrix; groups: (n,)
    labels. Returns (mu, W): mu (n,) eigen-parities, one per eigenvector, and
    W (n, nmod) the weight of each eigenvector for each branch. The columns
    of W sum to sum_n |G_n|^2 for every branch: nothing is lost.
    """
    G = np.atleast_2d(G)
    n = M.shape[0]
    mu = np.zeros(n)
    W = np.zeros((n, G.shape[0]))
    pos = 0
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        Mg = M[np.ix_(idx, idx)]
        Mh = 0.5 * (Mg + Mg.conj().T)
        w, V = np.linalg.eigh(Mh)
        amp = V.conj().T @ G[:, idx].T                   # (ng, nmod)
        mu[pos:pos + len(idx)] = w
        W[pos:pos + len(idx)] = np.abs(amp) ** 2
        pos += len(idx)
    return mu, W
