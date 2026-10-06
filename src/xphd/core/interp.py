"""Zero-padded Fourier interpolation on a periodic mesh."""
from __future__ import annotations

import numpy as np

__all__ = ["fourier_refine", "ringing_report", "fine_band", "local_quadratic"]


def fourier_refine(f, r: int):
    """Refine a real periodic field by a factor r per axis.

    Exact at the original nodes by construction. For even n the Nyquist
    coefficient is split evenly between +n/2 and -n/2; without that the
    interpolant picks up a spurious imaginary part and no longer passes
    through the source values.
    """
    f = np.asarray(f, float)
    if r == 1:
        return f.copy()
    n1, n2 = f.shape[:2]
    extra = f.shape[2:]
    N1, N2 = n1 * r, n2 * r

    F = np.fft.fftshift(np.fft.fft2(f, axes=(0, 1)), axes=(0, 1))
    if n1 % 2 == 0:
        F[0] = F[0] * 0.5
    if n2 % 2 == 0:
        F[:, 0] = F[:, 0] * 0.5

    G = np.zeros((N1, N2) + extra, complex)
    o1, o2 = (N1 - n1) // 2, (N2 - n2) // 2
    G[o1:o1 + n1, o2:o2 + n2] = F
    if n1 % 2 == 0:
        G[o1 + n1, o2:o2 + n2] = F[0]
    if n2 % 2 == 0:
        G[o1:o1 + n1, o2 + n2] = F[:, 0]
    if n1 % 2 == 0 and n2 % 2 == 0:
        G[o1 + n1, o2 + n2] = F[0, 0]

    out = np.fft.ifft2(np.fft.ifftshift(G, axes=(0, 1)), axes=(0, 1))
    return out.real * (N1 * N2) / (n1 * n2)


def ringing_report(f, r: int):
    """Quantify Gibbs overshoot of a Fourier refinement.

    Band-limited interpolation of a kinked function overshoots near the kink,
    and the overshoot does NOT vanish as the source mesh is refined -- it
    saturates near the Gibbs constant. Compare `overshoot` against the phonon
    energies, not the bandwidth: a few percent of a 2 eV bandwidth is tens of
    meV, which is the scale that matters for a delta function.
    """
    f = np.asarray(f, float)
    F = fourier_refine(f, r)
    rng = float(f.max() - f.min())
    over = float(max(F.max() - f.max(), f.min() - F.min()))
    node = float(np.abs(F[::r, ::r] - f).max()) if r > 1 else 0.0
    return dict(overshoot=over, overshoot_frac=over / rng if rng else 0.0,
                data_range=rng, node_error=node, refined=F)



def local_quadratic(f, r: int):
    """Refine a periodic field by a factor r per axis with a LOCAL quadratic
    around every coarse point, applied across that point's cell.

    At each coarse point the model is

        f(u, v) = f0 + gx u + gy v + A u^2 + B u v + D v^2

    in reduced offsets (u, v). In the hexagonal (60-degree) basis the six
    nearest points are +-(1,0), +-(0,1), +-(1,-1): their three pair SUMS fix
    A, B, D exactly, and their pair DIFFERENCES fix the gradient by least
    squares -- the residual is the odd, trigonal-warping part, third order.
    Nodes are reproduced exactly.

    Why not Fourier: exciton bands are energy-ordered, so every crossing is a
    kink, and a global interpolant rings from each kink across the whole
    zone. At GaN's K minimum that made the band 34% too steep with a
    spurious two-fold anisotropy; on a test valley, a crossing 150 meV above
    the minimum moved the minimum's linewidth by 35%. A local model confines
    a kink's error to the cells beside it, and left the same linewidth
    exact to 2%. On a perfectly smooth band it is slightly less accurate than
    Fourier (0.3 meV rms on a 4 eV-wide test band).
    """
    fc = np.asarray(f, float)
    if r == 1:
        return fc.copy()
    n1, n2 = fc.shape[:2]
    if n1 < 3 or n2 < 3:
        raise ValueError("local_quadratic needs at least a 3 x 3 coarse mesh")
    N1, N2 = n1 * r, n2 * r

    def R(di, dj):                        # the value at (i + di, j + dj)
        return np.roll(fc, (-di, -dj), axis=(0, 1))

    h1, h2 = 1.0 / n1, 1.0 / n2
    S1 = R(1, 0) + R(-1, 0) - 2 * fc
    S2 = R(0, 1) + R(0, -1) - 2 * fc
    S3 = R(1, -1) + R(-1, 1) - 2 * fc
    A = S1 / (2 * h1 * h1)
    D = S2 / (2 * h2 * h2)
    B = (S1 + S2 - S3) / (2 * h1 * h2)
    d1 = R(1, 0) - R(-1, 0)
    d2 = R(0, 1) - R(0, -1)
    d3 = R(1, -1) - R(-1, 1)
    gx = (2 * d1 + d2 + d3) / 6 / h1
    gy = (d1 + 2 * d2 - d3) / 6 / h2
    # each fine point belongs to its nearest coarse point -- the same cells
    # the acoustic q^2 model and the constant-coupling scheme use
    ci = np.rint(np.arange(N1) / r).astype(int)
    cj = np.rint(np.arange(N2) / r).astype(int)
    u = (np.arange(N1) - ci * r) / N1
    v = (np.arange(N2) - cj * r) / N2
    I, J = np.meshgrid(ci % n1, cj % n2, indexing="ij")
    U, V = np.meshgrid(u, v, indexing="ij")
    k = fc.ndim - 2
    U = U.reshape(U.shape + (1,) * k)
    V = V.reshape(V.shape + (1,) * k)
    return (fc[I, J] + gx[I, J] * U + gy[I, J] * V + A[I, J] * U * U
            + B[I, J] * U * V + D[I, J] * V * V)


def fine_band(f, r: int, method: str = "local"):
    """Exciton energies on the fine mesh. method = "local" (default): local
    quadratics, which do not ring; "fourier": the zero-padded Fourier
    interpolant used before, to reproduce earlier numbers."""
    if method == "local":
        return local_quadratic(f, r)
    if method == "fourier":
        return fourier_refine(f, r)
    raise ValueError(f"method must be 'local' or 'fourier', not {method!r}")
