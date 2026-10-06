"""Electron / hole / interference decomposition of the linewidth.

With G = Ge + Gh,

    |G|^2 = |Ge|^2 + |Gh|^2 + 2 Re(Ge Gh*)

and each term integrates through the same tetrahedron weights, so

    Gamma = Gamma_e + Gamma_h + Gamma_interf

exactly. The cross term is not positive definite, so Gamma_interf may be
negative -- that is destructive interference, and it is the quantity of
interest.

SIGN CONVENTION. This assumes Ge + Gh == G, which is what the generator
asserts to 1e-10 -- the hole term already carries its minus sign. If your
Gh is stored WITHOUT that sign, the identity is |Ge - Gh|^2 and the cross
term flips sign. The sign is the whole result, so `check()` verifies the
decomposition sums to the total rather than trusting the convention.

GAUGE. The cross term is gauge-invariant: the arbitrary exciton phase
exp(-i theta) appears in both factors and cancels in the conjugate pairing.
Unlike the complex amplitude itself, this decomposition survives the fact
that the eigenvectors are not in a smooth gauge.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.interp import fine_band, fourier_refine
from ..core.mesh import blockwise
from ..core.stats import KB_EV, bose, degeneracy_groups
from ..tetra import delta_weights_2d

__all__ = ["Decomposition", "decompose"]


@dataclass
class Decomposition:
    """All arrays (nexc,) or (nexc, nmod), in eV."""
    total: np.ndarray
    electron: np.ndarray
    hole: np.ndarray
    interference: np.ndarray
    per_mode: dict
    E_n: np.ndarray
    groups: list
    T: float

    def check(self, tol=1e-8):
        """e + h + interference must reproduce the total."""
        s = self.electron + self.hole + self.interference
        return float(np.abs(s - self.total).max()
                     / max(np.abs(self.total).max(), 1e-300)) < tol

    def report(self, optical_modes=None):
        ok = "OK" if self.check() else "*** does not sum to total ***"
        print(f"T = {self.T:.1f} K   [FWHM, meV]   sum check: {ok}\n")
        head = (f"   {'manifold':>9} {'total':>9} {'e only':>9} {'h only':>9} "
                f"{'interf':>9} {'% of e+h':>9} {'h/e':>8}")
        if optical_modes is not None:
            head += f" {'% optical':>10}"
        print(head)
        for g in self.groups:
            t = self.total[g].mean() * 1e3
            e = self.electron[g].mean() * 1e3
            h = self.hole[g].mean() * 1e3
            x = self.interference[g].mean() * 1e3
            frac = 100.0 * x / max(e + h, 1e-300)
            ratio = h / e if e > 0 else np.inf
            lab = "{" + ",".join(str(k + 1) for k in g) + "}"
            row = (f"   {lab:>9} {t:9.2f} {e:9.3f} {h:9.2f} {x:+9.2f} "
                   f"{frac:+8.1f}% {ratio:8.1f}")
            if optical_modes is not None:
                pm = self.per_mode["total"][g].mean(axis=0)
                opt = 100.0 * pm[list(optical_modes)].sum() / max(pm.sum(), 1e-300)
                row += f" {opt:9.1f}%"
            print(row)

    def mode_table(self):
        """Interference per phonon branch, in meV. Sign should track
        acoustic vs optical character if the mechanism is what it looks like."""
        x = self.per_mode["interference"] * 1e3
        print(f"\n   interference by branch [meV]")
        print("   " + "manifold".rjust(9)
              + "".join(f"{f'nu={k+1}':>10}" for k in range(x.shape[1])))
        for g in self.groups:
            lab = "{" + ",".join(str(k + 1) for k in g) + "}"
            v = x[g].mean(axis=0)
            print(f"   {lab:>9}" + "".join(f"{y:+10.3f}" for y in v))


def decompose(archive, T=10.0, hw_fine=None, refine=1, flat_width=None,
              acoustic_cut=1e-4, fwhm=True, manifold_tol=2e-3,
              band_interp="local", verbose=True):
    """Gamma split into electron, hole and interference parts."""
    if not (archive.has("Ge") and archive.has("Gh")):
        raise ValueError(
            "archive has no Ge_grid/Gh_grid. Regenerate with the updated "
            "generate_excph.py -- without the separate amplitudes the "
            "decomposition is not defined, and a run that silently falls "
            "back to the total will report e == h == total.")

    n1, n2 = archive.n1, archive.n2
    r = int(refine)
    N1 = n1 * r
    fw = flat_width if flat_width is not None else 2.4e-3 / max(N1, 1)
    kT = max(KB_EV * T, 1e-12)

    Ge = archive.grid("Ge")
    Gh = archive.grid("Gh")
    nexc, nmod = archive.nexc, archive.nmod

    # the three pieces; they sum to |G|^2 by construction
    g2_e = np.abs(Ge) ** 2
    g2_h = np.abs(Gh) ** 2
    g2_x = 2.0 * np.real(Ge * np.conj(Gh))
    if verbose:
        dev = np.abs(g2_e + g2_h + g2_x
                     - archive.grid("g2")).max() / max(archive.grid("g2").max(), 1e-300)
        print(f"  |Ge|^2 + |Gh|^2 + 2Re(Ge Gh*) vs g2: {dev:.2e}"
              f"{'  OK' if dev < 1e-10 else '  *** sign convention differs ***'}")

    E_m = fine_band(archive.grid("E_m"), r, band_interp)
    if hw_fine is not None:
        hw = np.asarray(hw_fine, float)
        if hw.ndim == 2:
            hw = hw.reshape(N1, n2 * r, nmod)
    else:
        hw = fourier_refine(archive.grid("hw"), r)

    E_n = archive.E_n
    pref = (2.0 if fwhm else 1.0) * np.pi
    keys = ("total", "electron", "hole", "interference")
    per_mode = {k: np.zeros((nexc, nmod)) for k in keys}

    for nu in range(nmod):
        wq = hw[:, :, nu]
        live = wq > acoustic_cut
        if not live.any():
            continue
        N = bose(wq, kT)
        for n in range(nexc):
            for m in range(nexc):
                parts = {"electron": g2_e[:, :, nu, n, m],
                         "hole": g2_h[:, :, nu, n, m],
                         "interference": g2_x[:, :, nu, n, m]}
                if not any(np.any(v) for v in parts.values()):
                    continue
                dE = E_n[n] - E_m[:, :, m]
                We = delta_weights_2d(dE - wq, fw, "anti") * (N + 1.0)
                Wa = delta_weights_2d(dE + wq, fw, "anti") * N
                W = (We + Wa) * live
                for k, v in parts.items():
                    gg = blockwise(v, r) if r > 1 else v
                    per_mode[k][n, nu] += pref * float(np.sum(W * gg))

    for k in keys[1:]:
        per_mode["total"] += per_mode[k]

    tot = {k: per_mode[k].sum(axis=1) for k in keys}
    groups = degeneracy_groups(E_n, manifold_tol)
    d = Decomposition(tot["total"], tot["electron"], tot["hole"],
                      tot["interference"], per_mode, E_n, groups, T)
    if verbose:
        d.report()
        d.mode_table()
    return d
