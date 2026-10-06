"""Real-space localisation: is a field band-limited on the source mesh?

W(R) = (1/Nq) sum_q F(q) exp(-i q.R) on the Born-von Karman lattice. If W has
decayed within the Wigner-Seitz cell of the supercell, zero-padded resampling
is exact to that tolerance; if it has not, the refinement is truncation rather
than interpolation.

Three curves make this a TEST rather than a display:
  signal   the complex amplitude
  null     the same after a random phase at each q -- a random gauge cannot
           localise, so this is the control
  contrast the real scalar |G|^2, i.e. the quantity deliberately not
           interpolated
"""
from __future__ import annotations

import numpy as np

from ..core.mesh import ws_vectors
from ..core.stats import envelope, shell_maxima

__all__ = ["transform", "leakage", "decay_length", "localization_report"]


def transform(field, drop_dc: bool = True):
    """W(R) with the R=0 coefficient optionally zeroed.

    The DC term is the BZ mean. For a positive-definite field it is guaranteed
    to dominate, so normalising to it makes any positive quantity look far
    better localised than a complex one. Dropping it puts every curve on the
    same footing.
    """
    f = np.asarray(field)
    n1, n2 = f.shape[:2]
    w = np.fft.fft2(f.astype(complex), axes=(0, 1)) / (n1 * n2)
    if drop_dc:
        w[0, 0] = 0.0
    return w


def leakage(env, r_norm, a_len):
    """Fraction of the peak still present in the outermost WS shell."""
    outer = r_norm > r_norm.max() - a_len
    inner = r_norm > 0
    return float(env[outer].max() / env[inner].max())


def decay_length(r, v, lo, hi):
    """Fit |W| ~ exp(-r/lambda) over [lo, hi]."""
    s = (r >= lo) & (r <= hi) & (v > 0)
    if s.sum() < 3:
        return np.nan
    slope = np.polyfit(r[s], np.log(v[s]), 1)[0]
    return -1.0 / slope if slope < 0 else np.nan


def localization_report(G, a1, a2, seed: int = 0, verbose: bool = True):
    """Signal / null / contrast leakages for a complex (n1,n2,...) field."""
    G = np.asarray(G)
    n1, n2 = G.shape[:2]
    a_len = float(np.linalg.norm(np.asarray(a1, float)[:2]))
    _, r_norm = ws_vectors(n1, n2, a1, a2)

    env_sig = envelope(transform(G))

    rng = np.random.default_rng(seed)
    phase = np.exp(2j * np.pi * rng.random((n1, n2)))
    env_null = envelope(transform(
        G * phase.reshape((n1, n2) + (1,) * (G.ndim - 2))))

    s_q = np.sum(np.abs(G) ** 2, axis=tuple(range(2, G.ndim)))
    env_mod = np.abs(transform(s_q))

    out = {}
    for tag, env in (("signal", env_sig), ("null", env_null),
                     ("contrast", env_mod)):
        c, p = shell_maxima(r_norm, env / env[r_norm > 0].max())
        out[tag] = dict(
            leakage=leakage(env, r_norm, a_len),
            decay_a=decay_length(c, p, a_len, min(6 * a_len,
                                                  0.8 * r_norm.max())) / a_len,
            envelope=env)
    out["ws_radius_a"] = r_norm.max() / a_len
    out["r_norm"] = r_norm
    out["a_len"] = a_len

    if verbose:
        print(f"WS radius {out['ws_radius_a']:.1f} a")
        print(f"   {'':<10} {'leakage':>10} {'decay (a)':>11}")
        for tag in ("signal", "null", "contrast"):
            d = out[tag]
            print(f"   {tag:<10} {d['leakage']:10.3e} {d['decay_a']:11.2f}")
        if out["signal"]["leakage"] > 0.5 * out["null"]["leakage"]:
            print("\n   signal is indistinguishable from the randomised-gauge "
                  "control:\n   the complex amplitude is NOT in a smooth "
                  "gauge, and Fourier\n   interpolation of it is not "
                  "defensible.")
    return out
