"""Full exciton-phonon self-energy matrix and the absorption lineshape.

The diagonal self-energy gives each exciton a width and a shift, and the
absorption spectrum is then a sum of Lorentzians -- the "diagonal
approximation". Chan et al., Nano Lett. 23, 3971 (2023) show that this
underestimates the measured linewidth in monolayer MoS2 by about a factor of
three, and that agreement requires the OFF-DIAGONAL elements

    Sigma_{S lambda}(w) = (1/Nq) sum_{q,n,nu}
        G*_{nS nu}(0,q) G_{n lambda nu}(0,q)
        (N_{nu q} + 1/2 +- 1/2) / (w - E_{nq} -+ w_{nu q} + i gamma)

Setting S = lambda recovers |G|^2 and the diagonal case. Off diagonal, the
real weight is replaced by the COMPLEX product of two coupling amplitudes
sharing an intermediate state n. Physically these are interference terms
between transitions through different exciton bands, and they are what makes
the lineshape asymmetric.

Gauge
-----
This is safe despite the excitonic gauge being arbitrary. The intermediate
state's phase exp(-i theta_{n,q}) appears in both G*_{nS} and G_{n lambda}
and cancels in the conjugate pairing -- the same cancellation that makes the
mode covariance D_{nu nu'} gauge invariant. The Q = 0 phases survive as
Sigma -> exp(-i phi_S) exp(i phi_lambda) Sigma, but they cancel against the
dipoles Omega_S, which carry the opposite phase. One BSE diagonalisation
fixes both, so the spectrum is invariant.

Cost
----
One triangle integration per (branch, intermediate state, emission/absorption)
per frequency -- the same count as a diagonal linewidth, but repeated over the
frequency grid, and it yields the WHOLE matrix rather than one element. Start
with a modest refinement.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core.interp import fine_band, fourier_refine
from .core.stats import KB_EV, bose
from .selfenergy import kramers_kronig
from .tetra import delta_weights_2d

__all__ = ["SigmaMatrix", "sigma_matrix", "epsilon2",
           "read_dipoles", "lineshape_metrics"]


@dataclass
class SigmaMatrix:
    """Sigma_{S lambda}(omega), shape (n_omega, nexc, nexc), in eV."""
    omega: np.ndarray
    sigma: np.ndarray
    E_bare: np.ndarray
    T: float
    meta: dict = field(default_factory=dict)

    def diagonal_only(self):
        """The same object with off-diagonal elements removed."""
        s = np.zeros_like(self.sigma)
        d = np.arange(self.sigma.shape[1])
        s[:, d, d] = self.sigma[:, d, d]
        return SigmaMatrix(self.omega, s, self.E_bare, self.T,
                           {**self.meta, "diagonal_only": True})

    def report(self):
        ne = self.sigma.shape[1]
        d = np.arange(ne)
        on = np.array([np.interp(self.E_bare[k], self.omega,
                                 -2 * self.sigma[:, k, k].imag) for k in d])
        offd = np.abs(self.sigma).copy()
        offd[:, d, d] = 0.0
        dia = np.abs(self.sigma[:, d, d])
        r = offd.max(axis=(1, 2)) / np.maximum(dia.max(axis=1), 1e-300)
        print(f"  {'state':>6} {'Gamma (meV)':>13} {'tau (ps)':>11}")
        for k in d:
            print(f"  {k+1:6d} {on[k]*1e3:13.3f} "
                  f"{6.582119569e-4/max(on[k],1e-30):11.4f}")
        print(f"\n  max |off-diagonal| / max |diagonal| = {r.max():.3f}")
        if r.max() > 0.2:
            print("  the off-diagonal elements are a substantial fraction of")
            print("  the diagonal, so the diagonal approximation will")
            print("  underestimate the linewidth and miss the asymmetry.")


def sigma_matrix(archive, omega, T, hw_fine=None, refine=1, flat_width=None,
                 acoustic_cut=1e-4, band_interp="local", verbose=True):
    """Full Sigma_{S lambda}(omega) at the archive's Q.

    omega : frequency grid, eV. Must be uniform (Kramers-Kronig), and must
            span the FULL support of Im Sigma -- from the lowest final-state
            energy minus the largest phonon to the highest plus it. A grid
            covering only the exciton range truncates the transform and the
            real part, and hence the peak positions, come out wrong. The
            coverage is checked and reported.
    """
    if not archive.has("G"):
        raise ValueError("needs the COMPLEX G_grid; |G|^2 alone cannot give "
                         "the off-diagonal elements")

    n1, n2 = archive.n1, archive.n2
    r = int(refine)
    N1, N2 = n1 * r, n2 * r
    ne, nmod = archive.nexc, archive.nmod
    fw = flat_width if flat_width is not None else 2.4e-3 / N1
    kT = max(KB_EV * T, 1e-12)

    G = archive.grid("G").reshape(n1 * n2, nmod, ne, ne)   # (q, nu, S, n)
    E_m_c = archive.grid("E_m")
    hw_c = archive.grid("hw")
    E_m = fine_band(E_m_c, r, band_interp)
    hw = (np.asarray(hw_fine, float).reshape(N1, N2, nmod)
          if hw_fine is not None else fourier_refine(hw_c, r))

    # map every fine point to its coarse cell (nearest, matching `blockwise`),
    # so the constant-per-cell coupling can be contracted after reducing the
    # triangle weights rather than by materialising G on the fine mesh
    ci = np.rint(np.arange(N1) / r).astype(int) % n1
    cj = np.rint(np.arange(N2) / r).astype(int) % n2
    cell = (ci[:, None] * n2 + cj[None, :]).ravel()
    ncell = n1 * n2

    omega = np.asarray(omega, float)
    sig = np.zeros((len(omega), ne, ne), complex)
    live = [hw[:, :, nu] > acoustic_cut for nu in range(nmod)]
    Nb = [bose(hw[:, :, nu], kT) for nu in range(nmod)]

    if verbose:
        print(f"Sigma matrix: {ne} states, {nmod} branches, "
              f"{len(omega)} frequencies, fine {N1}x{N2}")

    for k, w in enumerate(omega):
        D = np.zeros((ne, ne), complex)
        for nu in range(nmod):
            if not live[nu].any():
                continue
            wq, Nq_, ok = hw[:, :, nu], Nb[nu], live[nu]
            for n in range(ne):
                dE = w - E_m[:, :, n]
                for arg, occ in ((dE - wq, Nq_ + 1.0), (dE + wq, Nq_)):
                    W = delta_weights_2d(arg, fw) * occ * ok
                    if not W.any():
                        continue
                    ws = np.bincount(cell, weights=W.ravel(),
                                     minlength=ncell)
                    g = G[:, nu, :, n]                    # (q, S)
                    D += np.pi * np.einsum("c,cs,cl->sl", ws,
                                           g.conj(), g)
        sig[k] = -1j * D
        if verbose and len(omega) >= 10 and (k + 1) % (len(omega) // 10) == 0:
            print(f"  {k+1}/{len(omega)}", end="\r")

    # Re from Im, element by element. D is Hermitian, so the transform of
    # Im Sigma = -D preserves that.
    im = sig.imag.reshape(len(omega), -1).T
    re = kramers_kronig(omega, im).T.reshape(len(omega), ne, ne)
    sig = re + 1j * sig.imag

    d_ = np.arange(ne)
    edge = max(np.abs(sig[0].imag[d_, d_]).max(),
               np.abs(sig[-1].imag[d_, d_]).max()) \
        / max(np.abs(sig.imag[:, d_, d_]).max(), 1e-300)
    if verbose:
        need_lo = float(E_m_c.min() - hw_c.max())
        need_hi = float(E_m_c.max() + hw_c.max())
        print(f"  |Im Sigma| at the grid edges: {edge:.2%} of its maximum")
        if edge > 0.05:
            print(f"    truncated -- Kramers-Kronig, and therefore every peak")
            print(f"    position, is unreliable. The grid should span at least")
            print(f"    {need_lo:.3f} .. {need_hi:.3f} eV; it spans "
                  f"{omega[0]:.3f} .. {omega[-1]:.3f}.")

    sm = SigmaMatrix(omega, sig, archive.E_n, T,
                     dict(refine=r, flat_width=fw, acoustic_cut=acoustic_cut,
                          edge=edge))
    if verbose:
        print()
        sm.report()
    return sm


def epsilon2(sm, dipoles, eta=1e-3):
    """Absorption spectrum from the self-energy matrix.

        eps_2(w) ~ -Im sum_{S,lambda} Omega*_S [w - E - Sigma(w)]^-1_{S lambda}
                   Omega_lambda

    Inverting the full matrix keeps the interference between exciton bands.
    Compare against `sm.diagonal_only()`, which reduces to a sum of
    Lorentzians and is the approximation used in most first-principles work.
    """
    d = np.asarray(dipoles, complex)
    ne = sm.sigma.shape[1]
    if d.size != ne:
        raise ValueError(f"{d.size} dipoles for {ne} states")
    out = np.zeros(len(sm.omega))
    I = np.eye(ne)
    for k, w in enumerate(sm.omega):
        M = (w + 1j * eta) * I - np.diag(sm.E_bare) - sm.sigma[k]
        out[k] = -np.imag(d.conj() @ np.linalg.solve(M, d))
    return out


def read_dipoles(bsdb, neigs=None):
    """Exciton dipole amplitudes from a yambo BS_diago database.

    Only RELATIVE magnitudes and phases matter for a lineshape, so the
    residual normalisation convention does not enter. The phases must come
    from the SAME diagonalisation as G_grid, or they will not cancel against
    the self-energy's Q = 0 phases.
    """
    from netCDF4 import Dataset
    with Dataset(bsdb) as f:
        keys = list(f.variables)
        lk = "BS_left_Residuals" if "BS_left_Residuals" in keys else "BS_Residuals"
        L = np.asarray(f[lk][:], float)
        R = (np.asarray(f["BS_right_Residuals"][:], float)
             if "BS_right_Residuals" in keys else L)
    amp = np.sqrt(np.abs((L[:, 0] + 1j * L[:, 1])
                         * (R[:, 0] + 1j * R[:, 1]).conj()))
    ph = np.angle(L[:, 0] + 1j * L[:, 1])
    d = amp * np.exp(1j * ph)
    return d[:neigs] if neigs else d


def lineshape_metrics(omega, y, frac=0.5):
    """Peak, width and asymmetry of a spectrum, by interpolation.

    The peak is located by a parabola through the maximum and its neighbours,
    so it is not quantised to the grid -- a shift smaller than one step would
    otherwise read as zero.

    Asymmetry is (right half-width) - (left half-width) at `frac` of the
    maximum. It is zero for a Lorentzian and is the signature the off-diagonal
    self-energy produces; the diagonal approximation cannot generate it.

    NOTE the INTEGRAL of eps_2 is fixed by the oscillator-strength sum rule,
    so comparing integrals between the full and diagonal treatments always
    gives 1. The self-energy redistributes weight, it does not create it.
    Width and asymmetry are the quantities that move.
    """
    w = np.asarray(omega, float)
    y = np.asarray(y, float)
    k = int(np.argmax(y))
    if 0 < k < len(w) - 1:
        y0, y1, y2 = y[k-1], y[k], y[k+1]
        den = y0 - 2 * y1 + y2
        pk = w[k] + 0.5 * (w[1] - w[0]) * (y0 - y2) / den if den != 0 else w[k]
    else:
        pk = w[k]

    h = y.max() * frac
    def cross(lo, hi):
        if y[lo] == y[hi]:
            return w[lo]
        t = (h - y[lo]) / (y[hi] - y[lo])
        return w[lo] + t * (w[hi] - w[lo])
    l = r = np.nan
    for i in range(k, 0, -1):
        if y[i-1] < h <= y[i]:
            l = cross(i-1, i); break
    for i in range(k, len(w) - 1):
        if y[i+1] < h <= y[i]:
            r = cross(i+1, i); break
    fw = (r - l) if np.isfinite(l) and np.isfinite(r) else np.nan
    asym = ((r - pk) - (pk - l)) if np.isfinite(fw) else np.nan
    return dict(peak=pk, fwhm=fw, asym=asym, left=l, right=r)
