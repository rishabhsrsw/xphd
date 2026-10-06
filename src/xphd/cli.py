"""Command-line interface: ``xphd <subcommand>``."""
from __future__ import annotations

import argparse

import numpy as np


def _version():
    """Package version, without importing the package eagerly.

    A bare __import__('xphd').__version__ here makes --version able to break
    every other subcommand: any problem in __init__ surfaces before argparse
    has even looked at argv.
    """
    try:
        from importlib.metadata import version
        return version("xphd")
    except Exception:
        try:
            from . import __version__
            return __version__
        except Exception:
            return "unknown"


def _archive(a):
    from .io.excph import ExcPhArchive
    return ExcPhArchive(a.npz, mesh=getattr(a, "mesh", None))


def _hex(a):
    A = a.hex
    return np.array([A, 0.0]), np.array([-A / 2, A * np.sqrt(3) / 2])


def _refine_from(a, arc):
    """Refinement factor: --hw-fine wins, then --fine, then --refine.

    For a convergence series the FINE mesh is held fixed and only the source
    varies, so quoting the target is less error-prone than quoting the ratio.
    """
    hw = np.load(a.hw_fine) if getattr(a, "hw_fine", None) else None
    r = getattr(a, "refine", None) or 1
    if hw is not None:
        N = int(round(np.sqrt(hw.shape[0]))) if hw.ndim == 2 else hw.shape[0]
        if N % arc.n1:
            raise SystemExit(f"fine mesh {N} is not a multiple of the source "
                             f"mesh {arc.n1}")
        r = N // arc.n1
    elif getattr(a, "fine", None):
        r = a.fine // arc.n1
    return hw, r


# ---------------------------------------------------------------- handlers
def cmd_check(a):
    _archive(a).check()


def cmd_matdyn(a):
    from .io.matdyn import compare_to_archive, read_freq
    _, ev = read_freq(a.freq, nbnd=a.nbnd, nks=a.fine ** 2)
    print(f"  range {ev.min()*1e3:.4f} .. {ev.max()*1e3:.3f} meV "
          f"({int((ev < -1e-9).sum())} negative)")
    if a.npz:
        compare_to_archive(ev, a.fine, _archive(a), q_order=a.q_order)
    if a.out:
        np.save(a.out, ev)
        print(f"  wrote {a.out}  {ev.shape}  eV")


def cmd_linewidth(a):
    from .linewidth import compute
    arc = _archive(a)
    hw, r = _refine_from(a, arc)
    res = compute(arc, a.T, hw_fine=hw, refine=r, scheme=a.scheme,
                  flat_width=a.flat_width, acoustic_cut=a.acoustic_cut,
                  acoustic_model=a.acoustic_model, n_shell=a.n_shell,
                  band_interp=a.band_interp)
    if a.acoustic_model == "none":
        # a manifold with (almost) no emission at every temperature sits at a
        # band minimum; its only channel is the one 'none' drops
        mins = []
        for g in res.groups:
            em = res.emission[:, g].mean(axis=1)
            tot = res.total[:, g].mean(axis=1)
            if np.all(em <= 1e-2 * np.maximum(tot, 1e-300)):
                mins.append("{" + ",".join(str(k + 1) for k in g) + "}")
        if mins:
            print(f"\n  WARNING {', '.join(mins)} emit(s) nothing: a band "
                  f"minimum, whose only channel\n  is small-q acoustic "
                  f"absorption -- dropped by --acoustic-model none.\n  Its "
                  f"width here is not physical; rerun with --acoustic-model q2.")
    if a.per_mode:
        for it in range(len(res.T)):
            res.mode_report(it)
    if a.out:
        res.save(a.out)
        print(f"\nwrote {a.out}")
        if a.per_mode:
            import os
            stem, ext = os.path.splitext(a.out)
            if ext.lower() in (".txt", ".dat", ".csv"):
                mp = f"{stem}_modes{ext}"
                res.save_modes(mp)
                print(f"wrote {mp}")
            else:
                print("   (per-branch data is in the .npz as mode_em, mode_ab)")


def cmd_interference(a):
    from .linewidth import interference
    arc = _archive(a)
    hw, r = _refine_from(a, arc)
    interference(arc, a.T, hw_fine=hw, refine=r,
                 acoustic_cut=a.acoustic_cut, n_acoustic=a.n_acoustic,
                 band_interp=a.band_interp)


def cmd_sweep(a):
    from .sweep import sweep
    from .symmetry import load_ops, unfold, violation
    fld = sweep(a.pattern, a.T, hw_fine=a.hw_fine, fine=a.fine,
                refine=a.refine, scheme=a.scheme, flat_width=a.flat_width,
                acoustic_cut=a.acoustic_cut, mesh=a.mesh, workers=a.workers,
                acoustic_model=a.acoustic_model, n_shell=a.n_shell,
                band_interp=a.band_interp)
    fld.report()
    if a.unfold:
        ops = load_ops(a.unfold)
        print(f"\ntrace violation before: "
              f"{violation(fld.trace(0), fld.Q_red, ops)*100:.2f}%")
        for it in range(len(fld.T)):
            fld.LW[it] = unfold(fld.LW[it], fld.Q_red, ops, verbose=(it == 0))
            if fld.mode_em is not None:
                fld.mode_em[it] = unfold(fld.mode_em[it], fld.Q_red, ops,
                                         verbose=False)
                fld.mode_ab[it] = unfold(fld.mode_ab[it], fld.Q_red, ops,
                                         verbose=False)
        print(f"after:  {violation(fld.trace(0), fld.Q_red, ops)*100:.2f}%")
    fld.save(a.out, legacy=a.legacy)
    print(f"\nwrote {a.out}")


def cmd_ibz(a):
    """List the Q-points that actually need generating.

    Gamma(Q) summed over states is invariant under the point group, so only
    one representative per star needs computing; the rest follow from
    `xphd unfold`.

    The list to pass to `xphd generate` is built exactly as the generator
    builds its own: the irreducible parent of every full-zone point is read
    from lat.kpoints_indexes, and each parent's first full-zone image is its
    representative. `generate` takes that image's 0-based index in the
    LATTICE's k-ordering, which need not be row-major -- so an index worked
    out on a row-major mesh can name the wrong wavevector.
    """
    try:
        from .yambo import ibz_parents, lattice
        ylat = lattice(a.savepath)
    except Exception as e:                       # yambopy absent, or no SAVE
        ylat = None
        why = e
    if ylat is not None:
        kidx, first = ibz_parents(ylat)
        print(f"  {len(kidx)} full-zone points, {len(first)} irreducible")
        print("\narguments for `xphd generate` (0-based, lattice ordering):")
        print(" ".join(str(int(k)) for k in first))
        print("\nthe archives these produce:")
        print(" ".join(f"Q{int(k) + 1:04d}" for k in first))
        print("\nThe linewidth sweep needs only these; the real-time transport "
              "needs\nall {0} (`generate 0 .. {1}`), since a missing Q becomes "
              "a zero-energy\nsink that silently absorbs population."
              .format(len(kidx), len(kidx) - 1))
        return

    # fallback without yambopy: a row-major construction, which matches the
    # generator ONLY if the lattice happens to order its k-points row-major
    from .symmetry import irreducible, load_ops
    if a.mesh is None:
        raise SystemExit(f"could not read the lattice ({why}), and the "
                         f"row-major fallback needs --mesh N N")
    ops = load_ops(a.savepath)
    n1, n2 = a.mesh
    idx, Q = irreducible(n1, ops, n2)
    print(f"  [warn] could not read the lattice ({why}); falling back to a "
          f"row-major\n  construction, which matches `xphd generate` only if "
          f"the lattice orders\n  its k-points row-major. Check against "
          f"--iq-map before generating.")
    print(f"\n  {len(ops)} point-group operations, {len(idx)} irreducible")
    print("\nrow-major indices (0-based):")
    print(" ".join(str(int(k)) for k in idx))
    if a.iq_map:
        # match periodically: an exact lookup on rounded coordinates fails for
        # any point within rounding of 1.0, which maps to 0.0 on the other side
        qr = np.mod(np.asarray(np.load(a.iq_map)["q_red"], float), 1.0)[:, :2]
        d = Q[idx][:, None, :2] - qr[None, :, :]
        d -= np.rint(d)
        dist = np.linalg.norm(d, axis=-1)
        m = np.argmin(dist, axis=1)
        if dist[np.arange(len(idx)), m].max() > 1e-5:
            print("\n  [warn] some points are not in the archive's q_red")
        else:
            print(f"\nthe same points in the q-ordering of {a.iq_map} "
                  f"(0-based):")
            print(" ".join(str(int(v)) for v in sorted(m)))


def cmd_unfold(a):
    from .sweep import LinewidthField
    from .symmetry import expand, full_mesh, load_ops, unfold, violation
    fld = LinewidthField.load(a.field)
    ops = load_ops(a.savepath)
    n1, n2 = (a.mesh if a.mesh
              else np.asarray(fld.meta["mesh"]).ravel()[:2].astype(int))
    Qf = full_mesh(int(n1), int(n2))

    if len(fld.Q_red) < len(Qf):
        print(f"partial set: {len(fld.Q_red)} computed, {len(Qf)} wanted")
        LW = np.stack([expand(fld.LW[it], fld.Q_red, Qf, ops,
                              verbose=(it == 0)) for it in range(len(fld.T))])
        E_n = expand(fld.E_n, fld.Q_red, Qf, ops, verbose=False)
        # Gamma_nu(Q) summed over final states is invariant, and the
        # energy-sorted mode index is preserved by a rotation, so the
        # mode-resolved arrays unfold elementwise like the total.
        me = ma = None
        if fld.mode_em is not None:
            me = np.stack([expand(fld.mode_em[it], fld.Q_red, Qf, ops, verbose=False)
                           for it in range(len(fld.T))])
            ma = np.stack([expand(fld.mode_ab[it], fld.Q_red, Qf, ops, verbose=False)
                           for it in range(len(fld.T))])
        fld = LinewidthField(Qf, fld.T, LW, E_n, fld.meta, me, ma)
    else:
        print(f"trace violation before: "
              f"{violation(fld.trace(0), fld.Q_red, ops)*100:.2f}%")
        for it in range(len(fld.T)):
            fld.LW[it] = unfold(fld.LW[it], fld.Q_red, ops, verbose=(it == 0))
            if fld.mode_em is not None:
                fld.mode_em[it] = unfold(fld.mode_em[it], fld.Q_red, ops,
                                         verbose=False)
                fld.mode_ab[it] = unfold(fld.mode_ab[it], fld.Q_red, ops,
                                         verbose=False)
    print(f"trace violation after: "
          f"{violation(fld.trace(0), fld.Q_red, ops)*100:.2f}%")
    fld.save(a.out, legacy=a.legacy)
    print(f"wrote {a.out}  ({len(fld.Q_red)} points)"
          + ("  [+ Qx/Qy/LW for old plotting scripts]" if a.legacy else ""))


def cmd_selfenergy(a):
    from .selfenergy import self_energy
    arc = _archive(a)
    hw, r = _refine_from(a, arc)
    se = self_energy(arc, a.state - 1, a.T, hw_fine=hw, refine=r,
                     window=(a.window or None), n_omega=a.n_omega,
                     acoustic_cut=a.acoustic_cut, band_interp=a.band_interp)
    if a.out:
        import numpy as _np
        _np.savez(a.out, omega=se.omega, im=se.im, re=se.re,
                  spectral=se.spectral(), E_bare=se.E_bare, T=se.T)
        print(f"\nwrote {a.out}")


def cmd_lineshape(a):
    import numpy as _np
    from .lineshape import epsilon2, read_dipoles, sigma_matrix
    arc = _archive(a)
    hw, r = _refine_from(a, arc)
    # the grid must span the support of Im Sigma, not just the exciton range
    lo = float(arc.grid("E_m").min() - arc.grid("hw").max()) - a.pad
    hi = float(arc.grid("E_m").max() + arc.grid("hw").max()) + a.pad
    w = _np.linspace(lo, hi, a.n_omega)
    sm = sigma_matrix(arc, w, a.T, hw_fine=hw, refine=r,
                      acoustic_cut=a.acoustic_cut, band_interp=a.band_interp)
    out = dict(omega=w, sigma=sm.sigma, E_bare=sm.E_bare, T=a.T)
    if a.bsdb:
        d = read_dipoles(a.bsdb, neigs=arc.nexc)
        full = epsilon2(sm, d, eta=a.eta)
        diag = epsilon2(sm.diagonal_only(), d, eta=a.eta)
        out.update(eps2_full=full, eps2_diag=diag, dipoles=d)
        from .lineshape import lineshape_metrics
        mf, md = lineshape_metrics(w, full), lineshape_metrics(w, diag)
        print(f"\n  {'':>10} {'peak (eV)':>11} {'FWHM (meV)':>12} "
              f"{'asymmetry (meV)':>17}")
        for tag, m in (("full", mf), ("diagonal", md)):
            print(f"  {tag:>10} {m['peak']:11.4f} {m['fwhm']*1e3:12.2f} "
                  f"{m['asym']*1e3:+17.2f}")
        print(f"\n  FWHM ratio full/diagonal : "
              f"{mf['fwhm']/max(md['fwhm'],1e-300):.3f}")
        print(f"  peak shift               : "
              f"{(mf['peak']-md['peak'])*1e3:+.2f} meV   "
              f"(grid step {(w[1]-w[0])*1e3:.2f} meV)")
        print("  the INTEGRAL is fixed by the oscillator-strength sum rule and")
        print("  is 1.000 either way; width and asymmetry are what move.")
        if (w[1]-w[0]) > 0.2 * md['fwhm']:
            print(f"\n  [warn] the grid step is a large fraction of the width."
                  f"\n  Raise --n-omega before reading these numbers.")
    if a.out:
        _np.savez(a.out, **out)
        print(f"\nwrote {a.out}")


def cmd_bte(a):
    """Assemble the collision matrix, inject, propagate, write snapshots."""
    import numpy as _np
    from . import bte
    from .io.excph import load_archives

    arcs = load_archives(a.pattern, mesh=a.mesh)
    rm = bte.build_rates(arcs, T=a.T, acoustic_cut=a.acoustic_cut,
                         flat_width=a.flat_width, channels=a.channels)

    if a.refine > 1:
        ref = bte.refined_out_rates(arcs, a.T, a.refine,
                                    acoustic_cut=a.acoustic_cut,
                                    g2_mode=a.scheme, hw_fine=a.hw_fine,
                                    workers=a.workers)
        rm, _ = bte.rescale_rows(rm, ref)

    print()
    # test at the density actually injected, not a different one
    bte.check_balance(rm, N_tot=a.density * arcs[(0, 0)].nexc)

    # Injection. Light carries no momentum, so a pump fills only Q = 0.
    n1, n2, ne = rm.dims
    F0 = _np.zeros(len(rm.E))
    if a.inject == "optical":
        arc0 = arcs[(0, 0)]
        if not arc0.has("osc_Q0"):
            raise SystemExit(
                "optical injection needs osc_Q0 in the Q=0 archive. Add the "
                "exciton oscillator strengths in generate_excph.py; only "
                "RELATIVE values matter.")
        F0[:ne] = bte.inject_optical(arc0.E_n,
                                     _np.asarray(arc0._raw("osc_Q0")),
                                     a.density * ne, center=a.pump_center,
                                     sigma=a.pump_sigma)
    elif a.inject == "thermal":
        F0 = bte.inject_thermal(rm.E, a.density * len(rm.E), a.hot, a.width)
    else:
        sel = range(ne) if not a.state else [k - 1 for k in a.state]
        for k in sel:
            F0[k] = a.density
        print(f"  uniform injection into Q=0 states {[k+1 for k in sel]}")

    x = (rm.E - rm.E.min()) * 1e3
    print(f"  N_tot = {F0.sum():.4e}, <E>-Emin = {(F0 @ x)/F0.sum():.2f} meV")

    sol = bte.propagate(rm, F0, t_end=a.t_end, nt=a.nt)
    N = sol.y.sum(axis=0)
    print(f"\n  {'t (fs)':>10} {'<E>-Emin meV':>14} {'N/N0':>10} {'min F':>11}")
    for k, t in enumerate(sol.t):
        print(f"  {t:10.1f} {(sol.y[:,k] @ x)/N[k]:14.3f} "
              f"{N[k]/N[0]:10.6f} {sol.y[:,k].min():11.3e}")
    # Distinguish a real instability from integrator round-off. At
    # equilibrium the high-energy states sit many kT above the minimum and
    # carry occupations far below the solver's absolute tolerance; the solver
    # treats them as zero and round-off can take them slightly negative. That
    # is harmless. A genuine instability grows and breaks conservation.
    worst = float(sol.y.min())
    if worst < 0:
        drift = float(abs(N[-1] / N[0] - 1))
        rel = abs(worst) / max(a.density, 1e-300)
        print(f"\n  most negative occupation {worst:.2e} "
              f"({rel:.1e} of the injected density), N drift {drift:.1e}")
        if abs(worst) < 10 * 1e-12 and drift < 1e-6:
            print("    below the integrator's absolute tolerance and "
                  "conservation holds:\n    this is round-off in states whose "
                  "equilibrium occupation is far\n    smaller than atol. It "
                  "does not affect the dynamics. Lower atol\n    only if you "
                  "need those states quantitatively.")
        else:
            print("    LARGER than the tolerance, or conservation has "
                  "drifted: the\n    quadratic term may have destabilised. "
                  "Lower --density and check\n    whether the cooling curve "
                  "is unchanged.")

    if a.snapshots:
        print()
        bte.save_snapshots(rm, sol, a.snapshots, channels=a.channels)


def cmd_localization(a):
    from .diagnostics.localization import localization_report
    localization_report(_archive(a).grid("G"), *_hex(a))


def cmd_gauge(a):
    from .diagnostics.gauge import gauge_report
    gauge_report(_archive(a).grid("G"), *_hex(a))


def cmd_channels(a):
    from .diagnostics.channels import channel_report
    channel_report(_archive(a), state=a.state - 1, T=a.T,
                   acoustic_cut=a.acoustic_cut, top=a.top)


def cmd_frohlich(a):
    from .diagnostics.frohlich import frohlich_report
    frohlich_report(_archive(a), nshell=a.nshell)


# -------------------------------------------------------------------- main

# ---------------------------------------------------------------------------
# Commands implemented in their own modules. Each module owns its argparse
# block, so every option is declared once, where it is used; main() forwards
# the remaining arguments. Modules needing yambopy import it only here, when
# the command actually runs, so `xphd --help` never requires it.
# ---------------------------------------------------------------------------
DELEGATED = {
    # archives
    "dmats":           ("xphd.generate.dmats",   "main",
                        "rotation matrices Dmats.npy; band window from the BSE"),
    "generate":        ("xphd.generate.archive", "main",
                        "one exciton-phonon archive for a given Q"),
    "check-archive":   ("xphd.generate.check",   "main",
                        "validate ONE archive before generating the rest"),
    "verify-archives": ("xphd.generate.verify",  "main",
                        "confirm the full set of archives is complete"),
    # symmetry
    "parity":          ("xphd.parity",           "main",
                        "sigma_h parity of every exciton across the zone"),
    "irreps":          ("xphd.irreps",           "main",
                        "irrep labels at the high-symmetry points"),
    "selection-rule":  ("xphd.selection",        "main",
                        "coupling split into symmetry-allowed and -forbidden"),
    # optics
    "dipoles":         ("xphd.dipoles",          "main",
                        "exciton dipoles for emission, validated against yambo"),
    "coherence":       ("xphd.diagnostics.coherence", "main",
                        "k-space overlap and interference figure of merit"),
    "chirality":       ("xphd.chirality",        "main",
                        "helicity and valley polarisation of a degenerate pair"),
    "arpes":           ("xphd.arpes",            "main",
                        "exciton photoemission I(k,E), thermal or time-resolved"),
    # phonons
    "band-parity":     ("xphd.band_parity",      "main",
                        "sigma_h parity of the electronic bands, path and zone"),
    "mode-labels":     ("xphd.mode_labels",      "main",
                        "phonon branch characters on the archive mesh"),
    "matdyn-split":    ("xphd.matdyn_split",     "main_split",
                        "split a long q-list into chunks for matdyn.x"),
    "matdyn-merge":    ("xphd.matdyn_split",     "main_merge",
                        "merge chunked matdyn.modes and .freq outputs"),
    "modes-check":     ("xphd.matdyn_split",     "main_check",
                        "verify a merged matdyn.modes before using it"),
    # diagnostics
    "decompose-parity": ("xphd.diagnostics.decompose_parity", "main",
                        "why an exciton has a fractional sigma_h character"),
    "validate-planar":  ("xphd.diagnostics.planar",           "main",
                        "checks after recomputing with a planar structure"),
    "check-parity-asr": ("xphd.diagnostics.asr",              "main",
                        "band mirror parity and the acoustic-sum-rule residue"),
    "bte-check":        ("xphd.diagnostics.bte_check",        "main",
                        "do transport out-rates equal the linewidths?"),
    "effective-mass":   ("xphd.diagnostics.effective_mass",   "main",
                        "effective masses from a parabolic fit to a band edge"),
    "mott-wannier":     ("xphd.diagnostics.mott_wannier",     "main",
                        "2D Mott-Wannier estimate of the bright-dark splitting"),
}


def _delegate(argv):
    """Run a DELEGATED command if argv names one; return True if it did."""
    if not argv or argv[0] not in DELEGATED:
        return False
    import importlib
    mod, fn, _ = DELEGATED[argv[0]]
    getattr(importlib.import_module(mod), fn)(argv[1:])
    return True


def main(argv=None):
    p = argparse.ArgumentParser(prog="xphd", description=__doc__)
    p.add_argument("--version", action="version", version=f"xphd {_version()}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def archive_arg(sp):
        sp.add_argument("npz")
        sp.add_argument("--mesh", type=int, nargs=2, default=None)
        return sp

    archive_arg(sub.add_parser("check", help="archive integrity checks")) \
        .set_defaults(func=cmd_check)

    s = sub.add_parser("matdyn", help="read/validate matdyn frequencies")
    s.add_argument("freq")
    s.add_argument("--fine", type=int, default=360)
    s.add_argument("--nbnd", type=int, default=None)
    s.add_argument("--npz", default=None)
    s.add_argument("--mesh", type=int, nargs=2, default=None)
    s.add_argument("--q-order", choices=("rowmajor", "npz"), default="rowmajor")
    s.add_argument("-o", "--out", default=None)
    s.set_defaults(func=cmd_matdyn)

    s = archive_arg(sub.add_parser("linewidth", help="exciton linewidths"))
    s.add_argument("--hw-fine", default=None)
    s.add_argument("--fine", type=int, default=None)
    s.add_argument("--refine", type=int, default=1)
    s.add_argument("--scheme", choices=("const", "fft"), default="const")
    s.add_argument("--T", type=float, nargs="+", default=[300.0])
    s.add_argument("--flat-width", type=float, default=None)
    s.add_argument("--acoustic-cut", type=float, default=1e-4)
    s.add_argument("--acoustic-model", choices=("none", "q2", "auto"),
                   default="none",
                   help="acoustic coupling inside the cell around q = 0, where "
                        "the constant-coupling value is zero by the sum rule. "
                        "none: leave it zero, which drops small-q acoustic "
                        "scattering -- the ONLY channel at a band minimum. "
                        "q2: |G|^2 * 2w = C q^2, the form for a neutral "
                        "exciton; use this. auto: also keep a constant A where "
                        "it is above 2 sigma -- right for free carriers, but "
                        "for an exciton a kept A is a fitting artefact and is "
                        "flagged")
    s.add_argument("--n-shell", type=int, default=3,
                   help="innermost shells of the source mesh the fit uses; "
                        "at least 1 for q2, 3 for auto. The per-shell C it "
                        "prints shows whether the result depends on the choice")
    s.add_argument("--band-interp", choices=("local", "fourier"),
                   default="local",
                   help="how the exciton energies are refined onto the fine "
                        "mesh. local (default): a quadratic around every mesh "
                        "point from its six neighbours, which cannot ring. "
                        "fourier: the global interpolant, which rings from "
                        "every band crossing -- at GaN's K minimum it made the "
                        "band 34%% too steep. Reproduces earlier results")
    s.add_argument("-o", "--out", default=None)
    s.add_argument("--per-mode", action="store_true",
                   help="also print linewidths by phonon branch; with a text "
                        "-o, write <stem>_modes.txt as well")
    s.set_defaults(func=cmd_linewidth)

    s = archive_arg(sub.add_parser("interference",
                                   help="split Gamma into e / h / cross term"))
    s.add_argument("--hw-fine", default=None)
    s.add_argument("--fine", type=int, default=None)
    s.add_argument("--refine", type=int, default=1)
    s.add_argument("--T", type=float, default=77.0)
    s.add_argument("--acoustic-cut", type=float, default=1e-4)
    s.add_argument("--n-acoustic", type=int, default=3)
    s.add_argument("--band-interp", choices=("local", "fourier"),
                   default="local",
                   help="how the exciton energies are refined onto the fine "
                        "mesh. local (default): a quadratic around every mesh "
                        "point from its six neighbours, which cannot ring. "
                        "fourier: the global interpolant, which rings from "
                        "every band crossing -- at GaN's K minimum it made the "
                        "band 34%% too steep. Reproduces earlier results")
    s.set_defaults(func=cmd_interference)

    s = sub.add_parser("sweep", help="linewidths at every Q in the zone")
    s.add_argument("pattern", help="glob, e.g. 'GI_ExcPh_Q*.npz'")
    s.add_argument("--mesh", type=int, nargs=2, default=None)
    s.add_argument("--hw-fine", default=None)
    s.add_argument("--fine", type=int, default=None)
    s.add_argument("--refine", type=int, default=None)
    s.add_argument("--scheme", choices=("const", "fft"), default="const")
    s.add_argument("--T", type=float, nargs="+", default=[77.0])
    s.add_argument("--flat-width", type=float, default=None)
    s.add_argument("--acoustic-cut", type=float, default=1e-4)
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--unfold", metavar="SAVE", default=None)
    s.add_argument("--legacy", action="store_true",
                   help="also write Qx, Qy and a 2D LW (nexc, nq) in meV, as "
                        "aggregate.py did, for existing plotting scripts")
    s.add_argument("-o", "--out", default="lw_field.npz")
    s.add_argument("--acoustic-model", choices=("none", "q2", "auto"),
                   default="none",
                   help="small-q acoustic coupling, as in xphd linewidth; "
                        "use q2, and the same --n-shell as the single-Q runs")
    s.add_argument("--n-shell", type=int, default=3)
    s.add_argument("--band-interp", choices=("local", "fourier"),
                   default="local",
                   help="how the exciton energies are refined onto the fine "
                        "mesh. local (default): a quadratic around every mesh "
                        "point from its six neighbours, which cannot ring. "
                        "fourier: the global interpolant, which rings from "
                        "every band crossing -- at GaN's K minimum it made the "
                        "band 34%% too steep. Reproduces earlier results")
    s.set_defaults(func=cmd_sweep)

    s = archive_arg(sub.add_parser(
        "selfenergy", help="Sigma(omega), quasiparticle energy, spectral function"))
    s.add_argument("--state", type=int, default=1)
    s.add_argument("--T", type=float, default=77.0)
    s.add_argument("--hw-fine", default=None)
    s.add_argument("--refine", type=int, default=1,
                   help="the Kramers-Kronig transform integrates over omega "
                        "and does not need the on-shell resolution; a coarser "
                        "refinement here is usually enough")
    s.add_argument("--window", type=float, default=0.0,
                   help="eV, half-width about the bare energy. 0 (default) "
                        "spans the full support of Im Sigma instead, which is "
                        "what Kramers-Kronig needs: Im Sigma is one-sided for "
                        "states near the bottom of the manifold, so a "
                        "symmetric window cuts through its maximum.")
    s.add_argument("--n-omega", type=int, default=241)
    s.add_argument("--acoustic-cut", type=float, default=1e-4)
    s.add_argument("-o", "--out", default=None)
    s.add_argument("--band-interp", choices=("local", "fourier"),
                   default="local",
                   help="exciton-energy refinement, as in xphd linewidth")
    s.set_defaults(func=cmd_selfenergy)

    s = archive_arg(sub.add_parser(
        "lineshape", help="full Sigma matrix and the absorption lineshape"))
    s.add_argument("--T", type=float, default=77.0)
    s.add_argument("--hw-fine", default=None)
    s.add_argument("--refine", type=int, default=1)
    s.add_argument("--n-omega", type=int, default=201)
    s.add_argument("--pad", type=float, default=0.15,
                   help="eV beyond the support of Im Sigma. The grid already "
                        "spans the final-state range plus the largest phonon; "
                        "this is extra margin.")
    s.add_argument("--acoustic-cut", type=float, default=1e-4)
    s.add_argument("--bsdb", default=None,
                   help="ndb.BS_diago_Q1, for the dipoles and hence eps_2")
    s.add_argument("--eta", type=float, default=1e-3)
    s.add_argument("-o", "--out", default=None)
    s.add_argument("--band-interp", choices=("local", "fourier"),
                   default="local",
                   help="exciton-energy refinement, as in xphd linewidth")
    s.set_defaults(func=cmd_lineshape)

    s = sub.add_parser("bte", help="real-time Boltzmann transport")
    s.add_argument("pattern", help="glob, e.g. 'GI_ExcPh_Q*.npz'")
    s.add_argument("--mesh", type=int, nargs=2, default=None)
    s.add_argument("--T", type=float, default=300.0, help="lattice temperature")
    s.add_argument("--acoustic-cut", type=float, default=1e-4)
    s.add_argument("--flat-width", type=float, default=0.0)
    s.add_argument("--refine", type=int, default=1,
                   help="refine the OUT-RATE integral and rescale rows. The "
                        "matrix itself cannot be refined (each q is a state, "
                        "so its size grows as f^4) but the row sum can.")
    s.add_argument("--hw-fine", default=None,
                   help="matdyn frequencies on the refined mesh, as used by "
                        "the linewidths. Sets the refinement and makes the "
                        "two calculations use identical phonons.")
    s.add_argument("--scheme", choices=("const", "complexfft"), default="const")
    s.add_argument("--inject", choices=("optical", "thermal", "uniform"),
                   default="optical")
    s.add_argument("--state", type=int, nargs="*", default=None,
                   help="1-based states for --inject uniform")
    s.add_argument("--pump-center", type=float, default=None, help="eV")
    s.add_argument("--pump-sigma", type=float, default=None, help="eV")
    s.add_argument("--hot", type=float, default=150.0,
                   help="meV above the minimum, for --inject thermal")
    s.add_argument("--width", type=float, default=25.0, help="meV")
    s.add_argument("--density", type=float, default=1e-3,
                   help="per-state occupation scale")
    s.add_argument("--t-end", type=float, default=10000.0, help="fs")
    s.add_argument("--nt", type=int, default=40)
    s.add_argument("--workers", type=int, default=1,
                   help="parallel processes for the out-rate refinement, "
                        "which dominates the cost at large --refine")
    s.add_argument("--channels", action="store_true",
                   help="keep channel-resolved rates so the animation can "
                        "colour each state by the phonon mode filling it. "
                        "Costs 2*nmodes matrices of nstates^2 in float32.")
    s.add_argument("--snapshots", default=None)
    s.set_defaults(func=cmd_bte)

    s = sub.add_parser("ibz", help="list the Q-points that need generating")
    s.add_argument("--savepath", default="SAVE",
                   help="directory holding ns.db1")
    s.add_argument("--mesh", type=int, nargs=2, default=None,
                   help="only for the fallback when yambopy is unavailable")
    s.add_argument("--iq-map", default=None,
                   help="fallback only: an archive whose q_red ordering to "
                        "report the points in")
    s.set_defaults(func=cmd_ibz)

    s = sub.add_parser("unfold",
                       help="symmetrise, or expand an IBZ set onto the full BZ")
    s.add_argument("field")
    s.add_argument("--savepath", default="SAVE")
    s.add_argument("--mesh", type=int, nargs=2, default=None)
    s.add_argument("--legacy", action="store_true",
                   help="also write Qx, Qy and a 2D LW (nexc, nq) in meV")
    s.add_argument("-o", "--out", default="lw_field_sym.npz")
    s.set_defaults(func=cmd_unfold)

    for name, fn in (("localization", cmd_localization), ("gauge", cmd_gauge)):
        s = archive_arg(sub.add_parser(name, help=f"{name} diagnostic"))
        s.add_argument("--hex", type=float, default=3.2, metavar="A")
        s.set_defaults(func=fn)

    s = archive_arg(sub.add_parser("channels", help="channel decomposition"))
    s.add_argument("--state", type=int, default=1)
    s.add_argument("--T", type=float, default=10.0)
    s.add_argument("--acoustic-cut", type=float, default=5e-4)
    s.add_argument("--top", type=int, default=12)
    s.set_defaults(func=cmd_channels)

    s = archive_arg(sub.add_parser("frohlich",
                                   help="e/h Frohlich cancellation"))
    s.add_argument("--nshell", type=int, default=8)
    s.set_defaults(func=cmd_frohlich)

    for name, (_, _, helptext) in DELEGATED.items():
        sub.add_parser(name, help=helptext, add_help=False)

    import sys
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and not ({"-h", "--help", "--version"} & set(argv)):
        from .banner import banner
        banner(_version(), argv)
    if _delegate(argv):
        return
    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
