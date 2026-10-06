"""Real-time Boltzmann equation for exciton populations.

    dF_i/dt = -sum_j [ P_ij F_i (1+F_j) - Q_ij F_j (1+F_i) ]

with the delta functions integrated exactly over the triangular energy
shells f_pm(q) = E_i - E_j(q) -/+ hw(q) = 0. No Lorentzian broadening.

States are (Q, band) pairs, so the rate matrix has dimension (Nq * nexc)^2.

Detailed balance
----------------
For an on-shell transition the phonon energy is fixed by the pair,
w = |E_i - E_j|, so with A the symmetric geometric weight

    P_ij = A (1+N)   for E_i > E_j      P_ji = A N

giving P_ij/P_ji = exp(w/kT) exactly. Naive assembly takes A from the
triangle containing q = Q_j - Q_i in one direction and the triangle
containing -q in the other; those differ, the relation fails, and no
distribution is stationary. Symmetrising the geometric weight and applying
the Bose factors afterwards restores it.

Nothing is ever divided by N, so weakly-occupied channels are not amplified.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core.interp import fourier_refine
from .core.mesh import blockwise
from .core.stats import KB_EV
from .tetra import triangle_weights

HBAR_EV_FS = 0.6582119569509065

__all__ = ["RateMatrix", "equilibrium", "build_rates", "refined_out_rates",
           "rescale_rows", "rhs_factory", "check_balance", "propagate",
           "inject_optical", "inject_thermal", "save_snapshots",
           "state_labels"]


# ---------------------------------------------------------------- equilibrium
def equilibrium(E, T, N_tot, stat="boltzmann"):
    """Stationary distribution at temperature T holding N_tot excitons.

    In the dilute limit the stimulated (1+F) factors are negligible and a
    dilute boson gas thermalises to BOLTZMANN, not Bose: the Boltzmann factor
    enters through (1+N)/N inside P, not as a ratio between P and Q. Testing a
    mu-constrained Bose function against a correct operator gives a large
    residual for no reason, so "boltzmann" is the default.
    """
    E = np.asarray(E, float)
    kT = max(KB_EV * T, 1e-12)
    x = np.clip((E - E.min()) / kT, 1e-12, 300.0)

    if stat == "boltzmann":
        F = np.exp(-x)
        return F / F.sum() * N_tot, None
    if stat == "bose0":
        F = 1.0 / np.expm1(x)
        return F / F.sum() * N_tot, 0.0
    if stat != "bose":
        raise ValueError(f"unknown stat {stat!r}")

    from scipy.optimize import brentq

    def residual(mu):
        return np.sum(1.0 / np.expm1(
            np.clip((E - mu) / kT, 1e-12, 700.0))) - N_tot

    hi = E.min() - 1e-12
    lo = E.min() - 100.0 * kT
    while residual(lo) > 0:
        lo -= 100.0 * kT
    try:
        mu = brentq(residual, lo, hi)
    except ValueError:
        mu = lo
    return 1.0 / np.expm1(np.clip((E - mu) / kT, 1e-12, 700.0)), mu


# ------------------------------------------------------------------ injection
def inject_optical(E_Q0, osc, N_tot, center=None, sigma=None, verbose=True):
    """Optically-created population at Q = 0: F ~ |d|^2 x pump lineshape.

    Only RELATIVE oscillator strengths matter, since the result is normalised
    to N_tot -- so the residual normalisation convention never enters.

    Light carries no momentum, so a pump fills only the light cone, which on
    any practical mesh is the Q=0 point alone. Weighting matters because the
    lowest excitons may be dark: injecting uniformly starts the run with
    population a laser could not have created and erases the bright-to-dark
    transfer before t = 0.
    """
    E_Q0 = np.asarray(E_Q0, float)
    w = np.abs(np.asarray(osc, float)).copy()
    if w.size != E_Q0.size:
        if w.size > E_Q0.size:
            w = w[:E_Q0.size]
        else:
            raise SystemExit(
                f"osc_Q0 has {w.size} entries but the archive has "
                f"{E_Q0.size} states. Trimming the archives and writing "
                f"osc_Q0 must use the SAME window, or states beyond the "
                f"oscillator list would be silently treated as dark.")
    if center is not None and sigma:
        w *= np.exp(-0.5 * ((E_Q0 - center) / sigma) ** 2)
    if w.sum() <= 0:
        raise ValueError("no oscillator strength inside the pump window")
    F = N_tot * w / w.sum()
    if verbose:
        order = np.argsort(w)[::-1]
        share = w[order] / w.sum()
        print("  optical injection, brightest states:")
        for k in order[:4]:
            print(f"    state {k+1:2d}  E = {E_Q0[k]:.4f} eV  "
                  f"{w[k]/w.sum()*100:5.1f}%")
        if share[0] > 0.9:
            print("    one state carries >90%: this is effectively "
                  "single-state injection")
    return F


def inject_thermal(E, N_tot, hot, width):
    """Phenomenological hot distribution, Gaussian in energy across the BZ.

    NOT optical injection -- a pump cannot populate finite Q. Label it as a
    hot-exciton initial condition if used.
    """
    x = (np.asarray(E, float) - np.min(E)) * 1e3
    F = np.exp(-((x - hot) ** 2) / (2.0 * width ** 2))
    return N_tot * F / F.sum()


# --------------------------------------------------------------- rate matrix
@dataclass
class RateMatrix:
    P: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    E: np.ndarray
    dims: tuple
    channels: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    @property
    def out_rate(self):
        """sum_j P_ij, in 1/fs."""
        return self.P.sum(axis=1)


def _triangulate(n1, n2):
    i, j = np.meshgrid(np.arange(n1), np.arange(n2), indexing="ij")
    v0 = (i * n2 + j).ravel()
    v1 = (((i + 1) % n1) * n2 + j).ravel()
    v2 = (i * n2 + ((j + 1) % n2)).ravel()
    v3 = (((i + 1) % n1) * n2 + ((j + 1) % n2)).ravel()
    return np.vstack([np.column_stack([v0, v1, v2]),
                      np.column_stack([v1, v3, v2])])


def build_rates(archives, T, acoustic_cut=1e-4, flat_width=0.0,
                channels=False, verbose=True):
    """Assemble the collision matrix from one archive per Q-point.

    archives : {(i, j): ExcPhArchive} keyed by the integer mesh index of Q.
    Unlike a linewidth, the rate matrix is not a scalar field on Q that can be
    unfolded afterwards: it connects states at DIFFERENT Q. An incomplete set
    leaves the missing Q-points at zero energy with no out-rate, i.e. as
    perfect sinks, and the run completes without complaining. Every Q is
    therefore required, and this is checked below.
    """
    first = next(iter(archives.values()))
    n1, n2 = first.n1, first.n2
    ne, nm = first.nexc, first.nmod
    Nq = n1 * n2
    nst = Nq * ne
    kT = max(KB_EV * T, 1e-12)

    q = np.mod(first.q_red, 1.0)
    qi = np.rint(q[:, 0] * n1).astype(int) % n1
    qj = np.rint(q[:, 1] * n2).astype(int) % n2

    if len(archives) < Nq:
        raise SystemExit(
            f"only {len(archives)} of {Nq} Q-points supplied. The rate matrix "
            f"connects states at different Q, so the missing points would "
            f"become zero-energy sinks with no out-rate and the run would "
            f"complete silently. Generate every Q.")

    E = np.zeros(nst)
    for (a, b), arc in archives.items():
        E[(a * n2 + b) * ne:(a * n2 + b) * ne + ne] = arc.E_n

    tri = _triangulate(n1, n2)
    area = 1.0 / len(tri)
    pref = 2.0 * np.pi / HBAR_EV_FS

    A_em = np.zeros((nst, nst))
    A_ab = np.zeros((nst, nst))
    Aem_nu = Aab_nu = None
    if channels:
        need = 2 * nm * nst * nst * 4 / 1e9
        if verbose:
            print(f"  channel matrices: {need:.2f} GB")
        Aem_nu = np.zeros((nm, nst, nst), np.float32)
        Aab_nu = np.zeros((nm, nst, nst), np.float32)

    if verbose:
        print(f"assembling over {len(tri)} triangles, {nst} states")

    for (a, b), arc in archives.items():
        ta, tb = (a + qi) % n1, (b + qj) % n2
        target = ta * n2 + tb

        g2 = np.maximum(np.asarray(arc._raw("g2"), float), 0.0)
        hw = np.asarray(arc._raw("hw"), float)
        live = hw > acoustic_cut
        N = np.zeros_like(hw)
        N[live] = 1.0 / np.expm1(np.clip(hw[live] / kT, 0.0, 200.0))
        g2 = np.where(live[:, :, None, None], g2, 0.0)

        E_fin = E[target[:, None] * ne + np.arange(ne)[None, :]]
        Et = np.transpose(E_fin[tri], (1, 0, 2))
        Wt = np.transpose(hw[tri], (1, 0, 2))
        Nt = np.transpose(N[tri], (1, 0, 2))
        gt = np.transpose(g2[tri], (1, 0, 2, 3, 4))
        nodes = np.transpose(
            target[tri][:, :, None] * ne + np.arange(ne)[None, None, :],
            (1, 0, 2))

        dE = arc.E_n[None, None, :, None] - Et[:, :, None, :]
        Wb = Wt[:, :, :, None, None]
        Wa = triangle_weights(dE[:, :, None, :, :] + Wb, area, flat_width)
        We = triangle_weights(dE[:, :, None, :, :] - Wb, area, flat_width)

        base = (a * n2 + b) * ne + np.arange(ne)
        for v in range(3):
            wt = pref * gt[v]
            em = (wt * We[v]).sum(axis=1)
            ab = (wt * Wa[v]).sum(axis=1)
            nl = em.shape[0]
            II = np.broadcast_to(base[None, :, None], (nl, ne, ne)).ravel()
            JJ = np.broadcast_to(nodes[v][:, None, :], (nl, ne, ne)).ravel()
            np.add.at(A_em, (II, JJ), em.ravel())
            np.add.at(A_ab, (II, JJ), ab.ravel())
            if Aem_nu is not None:
                for c in range(nm):
                    np.add.at(Aem_nu[c], (II, JJ), (wt * We[v])[:, c].ravel())
                    np.add.at(Aab_nu[c], (II, JJ), (wt * Wa[v])[:, c].ravel())

    # ---- symmetrise the geometric weight, then apply the Bose factors ----
    dE_pair = E[:, None] - E[None, :]
    w_pair = np.abs(dE_pair)
    N_pair = np.zeros_like(w_pair)
    m = w_pair > 1e-9
    N_pair[m] = 1.0 / np.expm1(np.clip(w_pair[m] / kT, 1e-12, 300.0))

    # With interpolated deltas BOTH directions fire for the same pair, since
    # the crossing energy is not exactly E_i - E_j. Pooling into B + B^T and
    # letting the DISCRETE energies decide which direction is emission puts
    # the whole Boltzmann ratio in `fac`, so P_ij/P_ji = exp(w/kT) exactly.
    B = 0.5 * (A_em + A_ab.T)
    B_tot = B + B.T
    fac = np.where(dE_pair > 0, 1.0 + N_pair, N_pair)
    P = B_tot * fac
    R = B_tot * np.sign(dE_pair)

    chan = None
    if Aem_nu is not None:
        emis = dE_pair > 0
        chan = np.zeros((2, nm, nst, nst), np.float32)
        for c in range(nm):
            Bn = 0.5 * (Aem_nu[c] + Aab_nu[c].T)
            Bt = (Bn + Bn.T).astype(float)
            chan[1, c] = np.where(emis, Bt * (1.0 + N_pair), 0.0)
            chan[0, c] = np.where(emis, 0.0, Bt * N_pair)
        if verbose:
            err = np.abs(chan.sum(axis=(0, 1)) - P).max()
            print(f"  channels sum to P: rel err "
                  f"{err / max(abs(P).max(), 1e-300):.2e}")

    if verbose:
        raw = (A_em * np.where(dE_pair > 0, 1 + N_pair, 0)
               + A_ab * np.where(dE_pair > 0, 0, N_pair)).sum()
        ratio = P.sum() / max(raw, 1e-300)
        both = int(((A_em > 0) & (A_ab.T > 0)).sum())
        anyp = int(((A_em > 0) | (A_ab.T > 0)).sum())
        print(f"  total rate after symmetrisation: {ratio:.4f} x raw   "
              f"({100*both/max(anyp,1):.0f}% of pairs seen both ways)")
        if abs(ratio - 1) > 0.2:
            print("    Enforcing detailed balance has changed the TOTAL rate")
            print("    by more than 20%. Branching ratios are unaffected, but")
            print("    the absolute timescale is not reliable at this level.")
            print("    Use --refine to rescale each row against a properly")
            print("    integrated out-rate.")

    return RateMatrix(P, P.T.copy(), R, E, (n1, n2, ne), chan,
                      meta=dict(T=T, acoustic_cut=acoustic_cut,
                                flat_width=flat_width))


# ------------------------------------------------------------ row refinement
def _outrate_worker(args):
    """One archive's contribution to the refined out-rates.

    Takes a PATH, not a loaded archive: an ExcPhArchive holds an open npz and
    does not survive pickling to a worker process.
    """
    path, cfg = args
    from .io.excph import ExcPhArchive
    arc = ExcPhArchive(path, mesh=cfg.get("mesh"))
    n1, n2 = arc.n1, arc.n2
    ne, nmod = arc.nexc, arc.nmod
    f = cfg["refine"]
    m1, m2 = n1 * f, n2 * f
    kT = max(KB_EV * cfg["T"], 1e-12)
    fw = cfg["flat_width"]

    tri = _triangulate(m1, m2)
    area = 1.0 / len(tri)
    pref = 2.0 * np.pi / HBAR_EV_FS

    Ef = fourier_refine(arc.grid("E_m"), f).reshape(m1 * m2, ne)
    if cfg.get("hw_fine"):
        # matdyn frequencies, the same ones the linewidths use. Interpolating
        # omega instead is worse near Gamma, where the acoustic branches are
        # non-analytic -- and that is exactly where the small-q channels live.
        hf = np.asarray(np.load(cfg["hw_fine"]), float).reshape(m1 * m2, nmod)
    else:
        hf = np.maximum(fourier_refine(arc.grid("hw"), f),
                        0.0).reshape(m1 * m2, nmod)
    if cfg["g2_mode"] == "complexfft":
        if not arc.has("G"):
            raise ValueError("complexfft needs G_grid")
        G = arc.grid("G")
        gf = np.clip(fourier_refine(G.real, f) ** 2
                     + fourier_refine(G.imag, f) ** 2, 0.0, None)
        gf = gf.reshape(m1 * m2, nmod, ne, ne)
    else:
        gf = blockwise(np.maximum(arc.grid("g2"), 0.0),
                       f).reshape(m1 * m2, nmod, ne, ne)

    live = hf > cfg["acoustic_cut"]
    N = np.zeros_like(hf)
    N[live] = 1.0 / np.expm1(np.clip(hf[live] / kT, 0.0, 200.0))
    gf = np.where(live[:, :, None, None], gf, 0.0)

    Et = np.transpose(Ef[tri], (1, 0, 2))
    Wt = np.transpose(hf[tri], (1, 0, 2))
    Nt = np.transpose(N[tri], (1, 0, 2))
    gt = np.transpose(gf[tri], (1, 0, 2, 3, 4))
    dE = arc.E_n[None, None, :, None] - Et[:, :, None, :]
    Wb = Wt[:, :, :, None, None]
    Wa = triangle_weights(dE[:, :, None, :, :] + Wb, area, fw)
    We = triangle_weights(dE[:, :, None, :, :] - Wb, area, fw)

    acc = np.zeros(ne)
    for v in range(3):
        Nv = Nt[v][:, :, None, None]
        acc += (pref * gt[v] * (Wa[v] * Nv + We[v] * (1.0 + Nv))
                ).sum(axis=(0, 1, 3))
    return arc.q_index(), acc


def refined_out_rates(archives, T, refine=None, acoustic_cut=1e-4,
                      flat_width=None, g2_mode="const", hw_fine=None,
                      workers=1, verbose=True):
    """Out-rates sum_j P_ij integrated on a refined q-mesh.

    The matrix itself cannot be refined -- each q is a state, so its size
    grows as f^4 -- but the ROW SUM is an ordinary BZ integral of a delta
    function, the same object a linewidth is, so it can be. Rescaling rows
    to match puts the converged out-rate (which sets the thermalisation time)
    at fine resolution while leaving the branching ratios, a much smoother
    function of q, at coarse resolution.

    This is by far the most expensive step: the triangle count grows as
    refine^2 per archive, so f=15 is 9x the cost of f=5. Use `workers`.
    """
    first = next(iter(archives.values()))
    n1, n2 = first.n1, first.n2
    ne = first.nexc

    # hw_fine fixes the refinement: the two must describe the same mesh
    if hw_fine is not None:
        h = np.load(hw_fine)
        N = int(round(np.sqrt(h.shape[0]))) if h.ndim == 2 else h.shape[0]
        if N % n1:
            raise SystemExit(f"hw_fine is {N}x{N}, not a multiple of the "
                             f"source mesh {n1}")
        f = N // n1
        if refine is not None and int(refine) != f:
            raise SystemExit(f"hw_fine implies refine={f} but refine="
                             f"{refine} was given; they must agree")
    else:
        f = int(refine or 1)

    if flat_width is None:
        flat_width = 2.4e-3 / (n1 * f)

    cfg = dict(T=T, refine=f, acoustic_cut=acoustic_cut,
               flat_width=flat_width, g2_mode=g2_mode, mesh=(n1, n2),
               hw_fine=str(hw_fine) if hw_fine is not None else None)
    args = [(arc.path, cfg) for arc in archives.values()]

    if verbose:
        print(f"refining out-rates on {n1*f}x{n2*f} "
              f"({2*(n1*f)*(n2*f)} triangles) x {len(args)} Q-points"
              + (f", {workers} workers" if workers > 1 else "")
              + ("\n  phonons from matdyn (matching the linewidths)"
                 if hw_fine is not None else
                 "\n  phonons Fourier-interpolated -- pass hw_fine= to use "
                 "the same\n  matdyn frequencies as the linewidth "
                 "calculation"))

    if workers > 1:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            res = list(pool.imap_unordered(_outrate_worker, args))
    else:
        res = [_outrate_worker(x) for x in args]

    out = np.zeros(n1 * n2 * ne)
    for (a, b), acc in res:
        out[(a * n2 + b) * ne:(a * n2 + b) * ne + ne] = acc
    return out


def rescale_rows(rm, out_ref, floor_frac=1e-4, verbose=True):
    """Bring sum_j P_ij toward the refined out-rate WITHOUT breaking balance.

    Scaling row i by s_i alone gives

        P'_ij / P'_ji = (P_ij s_i) / (P_ji s_j) = exp(dE/kT) * s_i/s_j

    so detailed balance survives only where s is constant. With a
    state-dependent s the stationary distribution becomes
    F_i ~ exp(-E_i/kT) / s_i, and the gas thermalises to the wrong
    temperature -- silently, since particle number is still conserved.

    Scaling the PAIR by sqrt(s_i s_j) leaves the ratio untouched and still
    moves the row sums to target, exactly when s is smooth and approximately
    otherwise. The residual mismatch is reported.

    Rows whose coarse sum is a tiny fraction of the median are numerically
    empty; dividing through them turns rounding error into a huge factor, so
    they are left alone.
    """
    out = rm.P.sum(axis=1)
    med = np.median(out[out > 0]) if np.any(out > 0) else 0.0
    good = out > max(1e-30, floor_frac * med)
    s = np.ones_like(out)
    s[good] = out_ref[good] / out[good]

    S = np.sqrt(np.outer(s, s))
    P = rm.P * S
    R = rm.R * S                      # antisymmetric x symmetric stays anti

    if verbose:
        if (~good).sum():
            print(f"  {(~good).sum()} rows left unscaled (below "
                  f"{floor_frac:g} x median)")
        print(f"  scale: median {np.median(s[good]):.4f}  "
              f"10-90 {np.percentile(s[good],10):.4f}-"
              f"{np.percentile(s[good],90):.4f}")
        hit = P.sum(axis=1)[good] / np.maximum(out_ref[good], 1e-300)
        print(f"  out-rates reach {np.median(hit):.3f} of target "
              f"(10-90 {np.percentile(hit,10):.3f}-"
              f"{np.percentile(hit,90):.3f})")
        print("    symmetric scaling preserves P_ij/P_ji = exp(dE/kT), so the")
        print("    equilibrium is untouched; the row sums land on target only")
        print("    to the extent that s varies slowly between coupled states.")

    return RateMatrix(P, P.T.copy(), 0.5 * (R - R.T), rm.E, rm.dims,
                      rm.channels, {**rm.meta, "row_scale": s}), s


# ----------------------------------------------------------------- dynamics
def rhs_factory(rm):
    Prow = rm.P.sum(axis=1)
    P, Q, R = rm.P, rm.Q, rm.R

    def rhs(t, F):
        return -(Prow * F - Q @ F + F * (R @ F))
    return rhs


def check_balance(rm, N_tot, verbose=True):
    """Symmetry and stationarity residuals."""
    T = rm.meta.get("T", 300.0)
    s1 = np.abs(rm.P - rm.Q.T).max() / max(rm.P.max(), 1e-300)
    s2 = np.abs(rm.R + rm.R.T).max() / max(np.abs(rm.R).max(), 1e-300)
    rhs = rhs_factory(rm)
    Prow = rm.P.sum(axis=1)

    out = {"P_minus_QT": s1, "R_plus_RT": s2}
    if verbose:
        print(f"  |P - Q^T| {s1:.3e}   |R + R^T| {s2:.3e}")
        print(f"\n  {'distribution':<20} {'residual':>12} {'weighted':>12}")
    for tag in ("boltzmann", "bose0", "bose"):
        try:
            F, _ = equilibrium(rm.E, T, N_tot, tag)
        except Exception:
            continue
        F = np.where(np.isfinite(F), F, 0.0)
        if F.sum() <= 0:
            continue
        r = rhs(0.0, F)
        sc = max(np.abs(Prow * F).max(), 1e-300)
        wt = F / F.sum()
        rel = np.abs(r) / np.maximum(np.abs(Prow * F), 1e-300)
        out[tag] = float(np.max(np.abs(r)) / sc)
        if verbose:
            print(f"  {tag:<20} {out[tag]:12.3e} {np.sum(wt*rel):12.3e}")
    if verbose:
        print("  This operator keeps the stimulated (1+F) factors, for which")
        print("  the exact stationary state is BOSE-EINSTEIN at any chemical")
        print("  potential -- read that row. Boltzmann is stationary only in")
        print("  the linear limit F << 1, so its residual measures how far")
        print("  the injected density is from that limit.")
    return out


def propagate(rm, F0, t_end=10000.0, nt=12, method="BDF",
              rtol=1e-7, atol=1e-12):
    from scipy.integrate import solve_ivp
    sol = solve_ivp(rhs_factory(rm), (0.0, t_end), np.asarray(F0, float),
                    t_eval=np.linspace(0.0, t_end, nt), method=method,
                    rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"integration failed: {sol.message}")
    return sol


# ---------------------------------------------------------------- snapshots
def state_labels(rm):
    """(Q_red, band) for every state, from the flat index (a*n2+b)*ne + l."""
    n1, n2, ne = rm.dims
    i = np.arange(len(rm.E))
    q, band = i // ne, i % ne
    a, b = q // n2, q % n2
    return np.stack([a / n1, b / n2, np.zeros(len(i))], 1), band


def save_snapshots(rm, sol, path, channels=True, verbose=True):
    """Write populations over time in the layout the plotting scripts read.

    Always writes t, F, E, Q_red, band, mesh, T.

    With `channels` and a RateMatrix carrying channel-resolved rates, also
    writes the time-dependent INFLUX decomposition: which (mode, emission or
    absorption) is filling each state at each instant. That is
    sum_j P_chan[c,j,i] F_j -- the transpose contracted with the occupation.
    The outflux version would have F_i factoring out, so its dominant channel
    would not change with time and the animation would be static.

    flux_t is the aggregate rate through each channel, sum_ij F_i P_chan[c,i,j].
    It differs from the per-state decomposition: a channel can fill most
    states while carrying little total flux, and the rare channel that finally
    lets excitons leave a band minimum shows up here and nowhere else.
    """
    Q_red, band = state_labels(rm)
    out = dict(t=sol.t, F=sol.y.T, E=rm.E, Q_red=Q_red, band=band,
               mesh=np.array([rm.dims[0], rm.dims[1], 1]),
               T=rm.meta.get("T", np.nan))

    if channels and rm.channels is not None:
        nch, nmq = rm.channels.shape[:2]
        flat = rm.channels.reshape(nch * nmq, *rm.channels.shape[2:])
        nt, nst = len(sol.t), len(rm.E)
        dm = np.zeros((nt, nst), np.int16)
        ie = np.zeros((nt, nst), bool)
        df = np.zeros((nt, nst), np.float32)
        fl = np.zeros((nt, nch, nmq), np.float32)
        for k in range(nt):
            F = sol.y[:, k]
            infl = np.stack([c.T @ F for c in flat])        # (nch*nmq, nst)
            tot = infl.sum(axis=0)
            best = np.argmax(infl, axis=0)
            dm[k] = (best % nmq).astype(np.int16)
            ie[k] = (best // nmq) == 1                      # index 1 = emission
            with np.errstate(invalid="ignore", divide="ignore"):
                df[k] = np.where(tot > 0, infl.max(axis=0)
                                 / np.maximum(tot, 1e-300), 0.0)
            fl[k] = np.array([F @ c.sum(axis=1) for c in flat]).reshape(nch, nmq)
        out.update(dom_mode_t=dm, is_emission_t=ie, dom_frac_t=df, flux_t=fl)
        if verbose:
            print(f"  channel decomposition saved: {nch} x {nmq} channels, "
                  f"{nt} times")

    np.savez(path, **out)
    if verbose:
        x = (rm.E - rm.E.min()) * 1e3
        N = sol.y.sum(axis=0)
        print(f"  wrote {path}: {len(sol.t)} snapshots, {len(rm.E)} states")
        print(f"    <E>-Emin: {(sol.y[:,0]@x)/N[0]:.2f} -> "
              f"{(sol.y[:,-1]@x)/N[-1]:.2f} meV   "
              f"N/N0 drift {abs(N[-1]/N[0]-1):.2e}")
    return out
