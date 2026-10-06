"""Tests for the pieces integrated from standalone scripts.

Each guards a specific failure that happened in practice:
  * matdyn-merge must keep a header on EVERY q-block -- stripping one per
    chunk once produced 129 547 headers for 129 600 blocks;
  * the .gp distance must stay monotonic across chunks;
  * every delegated command must import and parse its arguments.
"""
import math
import os
import pathlib

import numpy as np
import pytest

from xphd.matdyn_split import concat_gp, main_merge


def _write_chunks(tmp_path, n=9, nch=3):
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    q = np.stack([i.ravel() / n, j.ravel() / n, np.zeros(n * n)], 1)
    off = 0
    idx = ["# chunk first npts\n"]
    for c, part in enumerate(np.array_split(q, nch)):
        m, fr, gp, d = [], [f" &plot nbnd=   6, nks= {len(part):6d} /\n"], [], 0.0
        for k, qq in enumerate(part):
            # matdyn writes this header before EVERY block
            m += ["     diagonalizing the dynamical matrix ...\n", "\n",
                  f"     q = {qq[0]:12.6f} {qq[1]:12.6f} {qq[2]:12.6f}\n",
                  " " + "*" * 50 + "\n"]
            for mo in range(6):
                m.append(f"     freq ({mo+1:5d}) = 1.0 [THz] = "
                         f"{100.0*(mo+1):15.6f} [cm-1]\n")
                for _ in range(2):
                    m.append("  ( 0.1 0.0 0.1 0.0 0.1 0.0 )\n")
            m.append(" " + "*" * 50 + "\n")
            fr += [f"{qq[0]:12.6f}{qq[1]:12.6f}{qq[2]:12.6f}\n",
                   "  " + "".join(f"{100.0*(mo+1):10.4f}" for mo in range(6))
                   + "\n"]
            if k:
                d += math.dist(qq, part[k - 1])
            gp.append(f"{d:12.6f}" + "".join(f"{100.0*(mo+1):12.4f}"
                                             for mo in range(6)) + "\n")
        (tmp_path / f"modes_{c:03d}.modes").write_text("".join(m))
        (tmp_path / f"freq_{c:03d}.freq").write_text("".join(fr))
        (tmp_path / f"freq_{c:03d}.freq.gp").write_text("".join(gp))
        idx.append(f"{c:5d} {off:9d} {len(part):8d}\n")
        off += len(part)
    (tmp_path / "chunk_index.txt").write_text("".join(idx))
    return n * n


def test_merge_keeps_one_header_per_block(tmp_path):
    nq = _write_chunks(tmp_path)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        main_merge(["--chunks", "3", "-o", "m.modes", "--freq", "x.freq"])
        txt = (tmp_path / "m.modes").read_text()
        assert txt.count("q =") == nq
        # one block header per block -- not one per file, not one per chunk
        assert txt.count("diagonalizing") == nq
        assert (tmp_path / "x.freq").read_text().count("&plot") == 1
    finally:
        os.chdir(cwd)


def test_gp_distance_is_monotonic(tmp_path):
    nq = _write_chunks(tmp_path)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        rows = concat_gp(3, "freq", "out.gp", verbose=False)
        assert rows == nq
        d = np.loadtxt("out.gp")[:, 0]
        assert np.all(np.diff(d) >= -1e-9)
    finally:
        os.chdir(cwd)


def test_gp_absent_returns_none(tmp_path):
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        assert concat_gp(2, "nothing", "out.gp", verbose=False) is None
    finally:
        os.chdir(cwd)


def test_mott_wannier_numbers():
    from xphd.diagnostics.mott_wannier import (alpha_from_eps, reduced_mass,
                                               splitting)
    assert reduced_mass(1.0, 1.0) == pytest.approx(0.5)
    assert alpha_from_eps(1.0, 20.0) == pytest.approx(0.0)
    assert alpha_from_eps(3.0, 20.0) == pytest.approx(2.0 * 20 / (4 * np.pi))
    # identical pairs give no binding-energy difference: splitting = dEg
    assert splitting("t", 5.2, 4.8, 0.3, 0.3, 4.0) == pytest.approx(0.4)


@pytest.mark.parametrize("cmd", [
    "decompose-parity", "validate-planar", "check-parity-asr", "bte-check",
    "effective-mass", "mott-wannier", "matdyn-merge", "parity", "irreps",
    "dipoles", "generate", "check-archive", "verify-archives", "dmats",
])
def test_delegated_command_parses_help(cmd, capsys):
    from xphd.cli import _delegate
    with pytest.raises(SystemExit) as e:
        _delegate([cmd, "--help"])
    assert e.value.code == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_mode_labels_follow_archive_q_order(tmp_path, monkeypatch):
    """Labels must land on the archive's q-ordering, whatever it is."""
    import xphd.modes
    from xphd.mode_labels import main
    fine, n = 12, 4
    b1 = np.array([1.0, 0.0, 0.0])
    b2 = np.array([0.5, np.sqrt(3) / 2, 0.0])
    i, j = np.meshgrid(np.arange(fine), np.arange(fine), indexing="ij")
    frac = np.stack([i.ravel() / fine, j.ravel() / fine], 1)
    qc = frac[:, :1] * b1 + frac[:, 1:] * b2
    L = []
    for q in qc:
        L += ["     diagonalizing the dynamical matrix ...\n", "\n",
              f"     q = {q[0]:12.6f} {q[1]:12.6f} {q[2]:12.6f}\n",
              " " + "*" * 40 + "\n"]
        for m in range(6):
            L.append(f"     freq ({m+1:5d}) = 1.0 [THz] = "
                     f"{100.0*(m+1):12.6f} [cm-1]\n")
            L += ["  ( 0.1 0.0 0.1 0.0 0.1 0.0 )\n"] * 2
        L.append(" " + "*" * 40 + "\n")
    (tmp_path / "matdyn.modes").write_text("".join(L))
    a, b = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    qr = np.stack([a.ravel() / n, b.ravel() / n, np.zeros(n * n)], 1)
    qr = qr[np.random.default_rng(3).permutation(n * n)]
    G = np.zeros((n * n, 6, 2, 2), complex)
    np.savez(tmp_path / "arc.npz", G_grid=G, g2_grid=np.abs(G) ** 2,
             Ge_grid=G, Gh_grid=G, E_n_grid=np.zeros(2),
             E_m_grid=np.zeros((n * n, 2)), hw_grid=np.full((n * n, 6), 0.1),
             q_red=qr, Q_red=np.zeros(3), mesh=np.array([n, n, 1]))
    # stub the classifier: every branch of source point c is labelled c
    monkeypatch.setattr(xphd.modes, "classify",
                        lambda ev, **k: np.repeat(
                            np.arange(len(ev))[:, None], 6, axis=1))
    monkeypatch.chdir(tmp_path)
    main(["--masses", "10.81", "14.007", "--fine", str(fine),
          "--mesh", str(n), "--archive", "arc.npz"])
    out = np.load(tmp_path / "mode_labels.npy")
    want = [int(round(q[0] * n)) * n + int(round(q[1] * n)) for q in qr]
    assert np.array_equal(out[:, 3], want)
    iG = int(np.where((qr[:, 0] == 0) & (qr[:, 1] == 0))[0][0])
    assert np.all(out[iG, :3] == -1)


def test_mode_labels_requires_masses():
    from xphd.mode_labels import main
    with pytest.raises(SystemExit):
        main([])            # no --masses: must refuse, not assume GaN


def test_linewidth_npz_keeps_per_branch_split(archive, tmp_path):
    """`xphd linewidth -o x.npz` once dropped the per-branch arrays it had
    computed. They must be written, and -- since a linewidth is additive over
    phonon branches -- sum to the emission and absorption totals."""
    import xphd
    arc = xphd.ExcPhArchive(archive)
    res = xphd.compute(arc, [10.0, 300.0], refine=2, acoustic_cut=5e-4,
                       verbose=False)
    out = tmp_path / "lw.npz"
    res.save(str(out))
    d = np.load(out)
    assert "mode_em" in d.files and "mode_ab" in d.files
    nT, ne = len(d["T"]), len(d["E_n"])
    assert d["mode_em"].shape[:2] == (nT, ne)
    assert np.allclose(d["mode_em"].sum(-1), d["emission"], rtol=1e-8, atol=1e-14)
    assert np.allclose(d["mode_ab"].sum(-1), d["absorption"], rtol=1e-8, atol=1e-14)


# ------------------------------------------------------------ band parity
def _hex_mesh(n):
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    kf = np.stack([i.ravel() / n, j.ravel() / n, np.zeros(n * n)], 1)
    rlat = np.array([[1.0, 0.0, 0.0], [0.5, np.sqrt(3) / 2, 0.0],
                     [0.0, 0.0, 1.0]])            # b1, b2 at 60 degrees
    return kf, rlat


def test_band_parity_path_on_24_mesh():
    """Gamma-M-K-Gamma on 24x24: 13 + 5 + 9 points, 25 without repeats, and
    the M-K points at known indices."""
    from xphd.band_parity import path_points
    from xphd.excsym import HIGH_SYMMETRY as H
    kf, rlat = _hex_mesh(24)
    corners = [(lab, H[lab]) for lab in ("G", "M", "K", "G")]
    idx, x, ticks = path_points(kf, rlat, corners)
    assert len(idx) == 25
    assert np.all(np.diff(x) > 0)                 # strictly along the path
    assert [t[1] for t in ticks] == ["G", "M", "K", "G"]
    q = np.rint(kf[idx, :2] * 24).astype(int)
    mk = [tuple(v) for v in q[12:17]]             # Gamma-M ends at index 12
    assert mk == [(12, 0), (11, 2), (10, 4), (9, 6), (8, 8)]
    # the path length Gamma-M-K-Gamma is |b|(1/2 + 1/(2 sqrt3) + 1/sqrt3)
    assert ticks[-1][0] == pytest.approx(0.5 + 1 / (2 * np.sqrt(3))
                                         + 1 / np.sqrt(3), rel=1e-9)


def test_band_parity_finds_points_stored_as_other_images():
    """A point stored as 23/24 lies on a segment through -1/24."""
    from xphd.band_parity import path_points
    kf, rlat = _hex_mesh(24)
    idx, _, _ = path_points(kf, rlat, [("A", (-0.25, 0.0)), ("B", (0.25, 0.0))])
    assert len(idx) == 13


def test_band_parity_mixed_degenerate_pair_is_undetermined():
    """Even a and odd b returned as (a+b)/sqrt2, (a-b)/sqrt2: sigma_h swaps
    them, so D = [[0,1],[1,0]] -- diagonal 0, off-diagonal 1."""
    from xphd.band_parity import parity_from_D
    D = np.zeros((2, 3, 3))
    D[:, 0, 0] = 1.0                              # a clean even band
    D[0, 1:, 1:] = np.diag([1.0, -1.0])           # k0: clean even, odd
    D[1, 1:, 1:] = [[0.0, 1.0], [1.0, 0.0]]       # k1: the mixed pair
    p, off, ok = parity_from_D(D)
    assert ok[0].all()
    assert np.allclose(p[0], [1, 1, -1])
    assert ok[1, 0] and not ok[1, 1] and not ok[1, 2]
    assert np.allclose(off[1, 1:], 1.0)


def test_band_parity_flips_skip_undetermined_points():
    from xphd.band_parity import flips_along
    pp = np.array([1.0, 1.0, 0.0, -1.0, -1.0, 1.0])
    ok = np.array([True, True, False, True, True, True])
    assert flips_along(pp, ok) == 2


def test_band_parity_locates_sigma_h():
    from xphd.band_parity import sigma_h_index
    E, C3 = np.eye(3), np.array([[-1, -1, 0], [1, 0, 0], [0, 0, 1]])
    sh = np.diag([1, 1, -1])
    ops = [E, C3, sh, C3 @ sh]                    # spatial half
    assert sigma_h_index(ops + ops) == 2          # T-copies appended


# ------------------------------------------------ SETTINGS-block scripts

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"


def run_script(name, cwd, **settings):
    """Run a SETTINGS-block script with some settings replaced, headless."""
    import os, re, subprocess, sys
    src = (SCRIPTS / name).read_text()
    if re.search(r"^SHOW = ", src, re.M):      # scripts that draw nothing
        settings.setdefault("SHOW", False)     # have no SHOW setting
    for k, v in settings.items():
        src, n = re.subn(rf"^{k} = .*$", f"{k} = {v!r}", src, count=1,
                         flags=re.M)
        assert n == 1, f"{name} has no setting {k}"
    tmp = pathlib.Path(cwd) / f"_run_{name}"
    tmp.write_text(src)
    return subprocess.run([sys.executable, str(tmp)], cwd=cwd,
                          capture_output=True, text=True,
                          env=dict(os.environ, MPLBACKEND="Agg"))


def test_plot_band_parity_script_runs(tmp_path):
    import subprocess, sys, pathlib
    from xphd.band_parity import path_points
    from xphd.excsym import HIGH_SYMMETRY as H
    kf, rlat = _hex_mesh(12)
    idx, x, ticks = path_points(kf, rlat, [(l, H[l]) for l in ("G", "M", "K", "G")])
    nk = len(kf)
    E = np.stack([-1.0 - 0.1 * np.arange(nk) / nk, 3.0 + 0.1 * np.arange(nk) / nk], 1)
    p = np.stack([np.ones(nk), -np.ones(nk)], 1)
    f = tmp_path / "bp.npz"
    np.savez(f, kf=kf, E=E, p=p, off=np.zeros_like(p), defined=np.ones_like(p, bool),
             bands=np.array([4, 5]), nv=1, path_idx=idx, path_x=x,
             tick_x=np.array([t[0] for t in ticks]),
             tick_labels=np.array([t[1] for t in ticks]))
    r = run_script("plot_band_parity.py", tmp_path, NPZ=str(f),
                   OUT=str(tmp_path / "bp.pdf"))
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "bp.pdf").stat().st_size > 1000


# ------------------------------------------------ per-branch linewidths

def test_per_branch_linewidths_sum_to_total(archive):
    import xphd
    r = xphd.compute(xphd.ExcPhArchive(archive), [10.0, 300.0], refine=5,
                     verbose=False)
    assert r.mode_em is not None
    per = (r.mode_em + r.mode_ab).sum(axis=2)            # (nT, nexc)
    assert np.allclose(per, r.total, rtol=1e-10, atol=1e-15)
    assert np.allclose(r.mode_em.sum(axis=2), r.emission, rtol=1e-10,
                       atol=1e-15)


def test_linewidth_per_mode_keeps_the_main_table(archive, tmp_path, capsys):
    """--per-mode adds a table and a _modes file; the main .txt must be the
    same as without it, since plotting scripts read its columns by position."""
    from xphd.cli import main
    a, b = tmp_path / "plain.txt", tmp_path / "pm.txt"
    main(["linewidth", str(archive), "--refine", "5", "--T", "10", "300",
          "-o", str(a)])
    main(["linewidth", str(archive), "--refine", "5", "--T", "10", "300",
          "--per-mode", "-o", str(b)])
    assert a.read_text() == b.read_text()
    assert "per-branch linewidth" in capsys.readouterr().out
    rows = [l.split() for l in (tmp_path / "pm_modes.txt").read_text()
            .splitlines() if not l.startswith("#")]
    assert len(rows) == 2 * 4 * 6                        # nT x nexc x nmod
    tot = {}
    for T, n, m, e, ab, t in rows:
        assert float(t) == pytest.approx(float(e) + float(ab), abs=2e-6)
        tot[(T, n)] = tot.get((T, n), 0.0) + float(t)
    # the per-state block ends at the first blank line; a second block of
    # manifold averages follows it with fewer columns
    block = a.read_text().split("\n\n")[0].splitlines()
    main_rows = [l.split() for l in block if l.strip() and not l.startswith("#")]
    for r in main_rows:                                 # G_n columns
        T = f"{float(r[0]):.1f}"
        for n in range(4):
            assert tot[(T, str(n + 1))] == pytest.approx(float(r[1 + n]),
                                                         abs=1e-4)


# ------------------------------------------------ single-precision q-points

def _d3h_with_time_reversal():
    C3 = np.array([[-1, -1, 0], [1, 0, 0], [0, 0, 1]])
    M = np.array([[1, 1, 0], [0, -1, 0], [0, 0, 1]])
    sh = np.diag([1, 1, -1])
    sp = [Y.astype(float) for R in (np.eye(3, dtype=int), C3, C3 @ C3)
          for X in (R, R @ M) for Y in (X, X @ sh)]
    return sp + [-R for R in sp]


@pytest.mark.parametrize("noise", [0.0, 1e-7, 5e-7, 1e-6])
def test_star_map_survives_single_precision_q(noise):
    """hBN's q-points carry ~1e-7. At the old 1e-6 tolerance star_map split
    stars once the noise reached 4e-7, with no error."""
    from xphd.symmetry import star_map
    n = 24
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    Q = np.stack([i.ravel() / n, j.ravel() / n, np.zeros(n * n)], 1)
    Q += np.random.default_rng(0).uniform(-noise, noise, Q.shape) * [1, 1, 0]
    assert len(star_map(Q, _d3h_with_time_reversal())[0]) == 61


def test_check_accepts_a_single_precision_split(archive, tmp_path, capsys):
    """Ge + Gh reproduces G only to ~1e-7 in float32; that is correct."""
    import xphd
    d = dict(np.load(archive))
    G = d["G_grid"]
    d["Ge_grid"] = (0.4 * G).astype(np.complex64)
    d["Gh_grid"] = (0.6 * G).astype(np.complex64)
    p = tmp_path / "f32.npz"
    np.savez(p, **d)
    out = xphd.ExcPhArchive(str(p)).check(verbose=True)
    assert 1e-9 < out["Ge_plus_Gh"] < 1e-5
    assert "Ge + Gh vs G" in capsys.readouterr().out


def test_band_parity_flags_a_mixed_degenerate_pair():
    """sigma_h on two degenerate bands of opposite parity, returned rotated by
    45 degrees: the diagonal is 0, and the point must be undetermined, not a
    parity of zero."""
    from xphd.band_parity import parity_from_D
    c = np.cos(np.pi / 4)
    U = np.array([[c, -c], [c, c]])
    clean = np.diag([1.0, -1.0])
    mixed = U @ clean @ U.T
    p, off, ok = parity_from_D(np.stack([clean, mixed]), tol=0.1)
    assert np.allclose(p[0], [1, -1]) and ok[0].all()
    assert np.allclose(p[1], [0, 0], atol=1e-12) and not ok[1].any()
    assert np.allclose(off[1], [1, 1])


def _interp_fixture(tmp_path):
    """parity.npz and a yambo-format interpolated file from one dispersion."""
    n = 24
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    Q = np.stack([i.ravel() / n, j.ravel() / n, np.zeros(n * n)], 1)
    r2 = lambda q: ((((q[:, :2] + .5) % 1) - .5) ** 2).sum(1) + \
        (((q[:, 0] + .5) % 1) - .5) * (((q[:, 1] + .5) % 1) - .5)
    E = lambda q: np.array([3.8, 4.3])[None, :] + \
        np.array([-1.2, 1.5])[None, :] * r2(q)[:, None]
    np.savez(tmp_path / "parity.npz", Q_red=Q, E=E(Q),
             chi=np.where(r2(Q)[:, None] < 0.1, 1.0, -1.0) * np.ones((1, 2)))
    M, K = np.array([.5, 0, 0]), np.array([1 / 3, 1 / 3, 0])
    qp = np.array([M * (1 - t) for t in np.linspace(0, 1, 41)]
                  + [K * t for t in np.linspace(0, 1, 41)[1:]])
    rows = [f"{k:.6f} " + " ".join(f"{v:.6f}" for v in E(qp[k:k + 1])[0])
            + " " + " ".join(f"{v:.6f}" for v in qp[k]) for k in range(len(qp))]
    (tmp_path / "o.excitons_interpolated").write_text(
        "# |q| e1 e2 qx qy qz\n" + "\n".join(rows) + "\n")


def test_parity_dispersion_script_runs_and_checks_energies(tmp_path):
    import subprocess, sys, pathlib
    _interp_fixture(tmp_path)
    r = run_script("plot_parity_dispersion.py", tmp_path,
                   INTERP="o.excitons_interpolated", PARITY="parity.npz",
                   OUT="o.png")
    assert r.returncode == 0, r.stderr
    assert "2 bands" in r.stdout
    assert "largest mismatch 0.0 meV" in r.stdout
    assert "WARNING" not in r.stdout
    assert (tmp_path / "o.png").exists()


def test_parity_dispersion_one_tick_per_corner(tmp_path):
    """A dense path with repeated corner points gave six M ticks, five K."""
    import subprocess, sys, pathlib, re
    n = 24
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    Q = np.stack([i.ravel() / n, j.ravel() / n, np.zeros(n * n)], 1)
    np.savez(tmp_path / "parity.npz", Q_red=Q, E=np.full((n * n, 1), 4.0),
             chi=np.ones((n * n, 1)))
    G, M, K = np.zeros(3), np.array([.5, 0, 0]), np.array([1 / 3, 1 / 3, 0])
    legs = [(G, M), (M, K), (K, G)]
    qp = []
    for a_, b_ in legs:
        for t in np.linspace(0, 1, 1001):
            qp.append(a_ + t * (b_ - a_))          # corners appear twice
    qp = np.array(qp)
    c = np.stack([qp[:, 0] + .5 * qp[:, 1], np.sqrt(3) / 2 * qp[:, 1]], 1)
    x = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))])
    rows = [f"{x[k]:.6f} 4.000000 " + " ".join(f"{v:.6f}" for v in qp[k])
            for k in range(len(qp))]
    (tmp_path / "o.interp").write_text("# |q| e1 qx qy qz\n" + "\n".join(rows) + "\n")
    r = run_script("plot_parity_dispersion.py", tmp_path, INTERP="o.interp",
                   PARITY="parity.npz", OUT="o.png")
    assert r.returncode == 0, r.stderr
    line = next(l for l in r.stdout.splitlines() if l.startswith("ticks:"))
    labels = re.findall(r"(\$\\Gamma\$|M|K) at", line)
    assert labels == ["$\\Gamma$", "M", "K", "$\\Gamma$"], line


# ------------------------------------------------ small-q acoustic fit

def _acoustic_data(target):
    """g2, hw, |q| on a 12x12 hexagonal mesh with |G|^2 * 2w = target(|q|)."""
    from xphd.core.mesh import hex_norm
    n = 12
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    q = np.stack([i.ravel() / n, j.ravel() / n, np.zeros(n * n)], 1)
    r = hex_norm(q).reshape(n, n)
    hw = np.zeros((n, n, 3)); g2 = np.zeros((n, n, 3, 1, 1))
    for nu in range(3):
        hw[:, :, nu] = 0.05 * (nu + 1) * r + 1e-6
        g2[:, :, nu, 0, 0] = np.where(r > 0, target(r) / (2 * hw[:, :, nu]), 0)
    return g2, hw, r


def test_q2_recovers_a_pure_q2_coupling():
    from xphd.linewidth import fit_acoustic
    g2, hw, r = _acoustic_data(lambda q: 0.2 * q ** 2)
    for model in ("q2", "auto"):
        A, C, info = fit_acoustic(g2, hw, r, n_shell=3, verbose=False,
                                  model=model)
        assert np.allclose(C[:3], 0.2, rtol=1e-6)
        assert np.allclose(A, 0.0) and not any(d["piezo"] for d in info.values())


def test_q2_uses_the_ratio_mean_not_the_joint_slope():
    """The bug: with a real offset, auto keeps A; q2 used to zero A but keep
    the slope fitted ALONGSIDE it. q2 must be a fit with A = 0."""
    from xphd.linewidth import fit_acoustic
    g2, hw, r = _acoustic_data(lambda q: 0.01 + 0.2 * q ** 2)
    _, Ca, ia = fit_acoustic(g2, hw, r, n_shell=4, verbose=False, model="auto")
    _, Cq, iq = fit_acoustic(g2, hw, r, n_shell=4, verbose=False, model="q2")
    assert ia[0]["piezo"] and np.isclose(Ca[0, 0, 0], 0.2, rtol=1e-6)
    ratio_mean = float(iq[0]["C_shell"].mean())
    assert np.isclose(Cq[0, 0, 0], ratio_mean, rtol=1e-12)
    assert not np.isclose(Cq[0, 0, 0], 0.2, rtol=1e-3)     # not the joint slope


def test_n_shell_is_honoured_or_refused():
    from xphd.linewidth import fit_acoustic
    g2, hw, r = _acoustic_data(lambda q: 0.2 * q ** 2)
    with pytest.raises(SystemExit):
        fit_acoustic(g2, hw, r, n_shell=2, verbose=False, model="auto")
    for k in (1, 2):
        _, _, info = fit_acoustic(g2, hw, r, n_shell=k, verbose=False,
                                  model="q2")
        assert len(info[0]["C_shell"]) == k


def test_kept_constant_warns_for_excitons(capsys):
    from xphd.linewidth import fit_acoustic
    g2, hw, r = _acoustic_data(lambda q: 0.01 + 0.2 * q ** 2)
    fit_acoustic(g2, hw, r, n_shell=4, verbose=True, model="auto")
    out = capsys.readouterr().out
    assert "WARNING a constant term A was kept" in out
    assert "--acoustic-model q2" in out


def test_falling_ratio_is_reported_as_a_lower_bound(capsys):
    from xphd.linewidth import fit_acoustic
    # a form factor: q^2 bending over as q grows
    g2, hw, r = _acoustic_data(lambda q: 0.2 * q ** 2 / (1 + (q / 0.12) ** 2) ** 1.5)
    fit_acoustic(g2, hw, r, n_shell=4, verbose=True, model="q2")
    assert "LOWER bound" in capsys.readouterr().out


def test_sweep_applies_the_same_acoustic_model(archive, tmp_path, monkeypatch):
    """The zone maps must use the same small-q treatment as the single-Q
    runs, or a minimum reads differently in the two figures."""
    import sys, xphd
    from xphd.sweep import sweep
    mod = sys.modules["xphd.sweep"]     # xphd.sweep is shadowed by the function
    seen, real = [], mod.compute
    def spy(*args, **kw):
        seen.append((kw.get("acoustic_model"), kw.get("n_shell")))
        return real(*args, **kw)
    monkeypatch.setattr(mod, "compute", spy)
    kw = dict(refine=5, acoustic_cut=1e-4)
    fld = sweep([str(archive)], [300.0], verbose=False,
                acoustic_model="q2", n_shell=2, **kw)
    assert seen == [("q2", 2)]                       # it reaches compute
    one = real(xphd.ExcPhArchive(archive), [300.0], verbose=False,
               acoustic_model="q2", n_shell=2, **kw)
    assert np.allclose(fld.LW[0, 0], one.total[0], rtol=1e-12)
    p = tmp_path / "f.npz"
    fld.save(p)
    d = np.load(p)
    assert str(d["acoustic_model"]) == "q2" and int(d["n_shell"]) == 2


def test_linewidth_warns_when_a_minimum_is_left_at_none(archive, capsys,
                                                        monkeypatch):
    """A manifold with no emission at any temperature sits at a minimum."""
    import types, xphd.linewidth
    from xphd.cli import main
    fake = types.SimpleNamespace(
        groups=[[0], [1]], T=np.array([10.0, 300.0]),
        emission=np.array([[0.0, 1e-3], [0.0, 2e-3]]),     # {1} emits nothing
        total=np.array([[1e-4, 2e-3], [2e-3, 4e-3]]))
    monkeypatch.setattr(xphd.linewidth, "compute", lambda *a, **k: fake)
    main(["linewidth", str(archive), "--refine", "5", "--T", "10", "300"])
    out = capsys.readouterr().out
    assert "WARNING {1} emit(s) nothing" in out and "{2}" not in out.split("WARNING")[1]
    main(["linewidth", str(archive), "--refine", "5", "--T", "10", "300",
          "--acoustic-model", "q2"])
    assert "emit(s) nothing" not in capsys.readouterr().out


def test_parity_dispersion_accepts_a_gw_file_and_catches_a_wrong_band(tmp_path):
    """A GW file shifts each band by its own amount against the Kohn-Sham
    energies in band_parity.npz. That must pass; a wrong --first-band must
    still fail."""
    import subprocess, sys, pathlib
    n = 24
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    Q = np.stack([i.ravel() / n, j.ravel() / n, np.zeros(n * n)], 1)
    def E_ks(q):
        f = ((q[:, :2] + .5) % 1) - .5
        r2 = f[:, 0] ** 2 + f[:, 1] ** 2 + f[:, 0] * f[:, 1]
        c = (np.cos(2 * np.pi * q[:, 0]) + np.cos(2 * np.pi * q[:, 1])
             + np.cos(2 * np.pi * (q[:, 0] - q[:, 1])))
        return np.stack([-5.2 + 3 * r2, -3.0 - 2 * r2 + 0.2 * c,
                         -0.5 + 6 * r2, 2.2 - 0.4 * c], 1)
    np.savez(tmp_path / "bp.npz", kf=Q, p=np.ones((n * n, 4)), E=E_ks(Q),
             bands=np.arange(3, 7), nv=2, defined=np.ones((n * n, 4), bool))
    G, M, K = np.zeros(3), np.array([.5, 0, 0]), np.array([1 / 3, 1 / 3, 0])
    qp = np.array([a_ + t * (b_ - a_) for a_, b_ in ((G, M), (M, K), (K, G))
                   for t in np.linspace(0, 1, 201)])
    c = np.stack([qp[:, 0] + .5 * qp[:, 1], np.sqrt(3) / 2 * qp[:, 1]], 1)
    x = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))])
    Ek = E_ks(qp)
    Eqp = Ek + np.array([2.3, 2.7, 5.2, 4.8]) + 0.04 * (Ek - Ek.mean(0))
    (tmp_path / "o.qp").write_text("# h\n" + "\n".join(
        f"{x[k]:.6f} " + " ".join(f"{v:.6f}" for v in Eqp[k]) + " "
        + " ".join(f"{v:.6f}" for v in qp[k]) for k in range(len(qp))) + "\n")
    def run(first):
        return run_script("plot_parity_dispersion.py", tmp_path, INTERP="o.qp",
                          PARITY="bp.npz", FIRST_BAND=first, OUT="o.png").stdout
    ok = run(3)
    assert "band identity confirmed" in ok and "WARNING" not in ok
    assert "a quasiparticle-corrected file" in ok
    bad = run(4)
    assert "--first-band is probably wrong" in bad


def test_every_plot_script_is_run_from_its_settings():
    """No command line: a SETTINGS block, no argparse, and it compiles."""
    import ast
    plots = sorted(p for p in SCRIPTS.glob("*.py"))
    assert plots
    for p in plots:
        src = p.read_text()
        compile(src, str(p), "exec")
        assert "argparse" not in src, p.name
        if p.name not in ("fatbands_pl.py", "linewidth_temperature.py",
                          "q_point_heatmap.py"):
            assert "# 0. SETTINGS" in src, p.name


def _phonon_modes(path, red, split3d=False):
    """matdyn-format eigenvectors for a two-atom planar lattice, q printed to
    four decimals as matdyn does. Two odd branches, four even; the in-plane
    optical pair is degenerate at Gamma unless split3d."""
    B = np.array([[1.0, 1 / np.sqrt(3)], [0.0, 2 / np.sqrt(3)]])
    F = (np.cos(2 * np.pi * red[:, 0]) + np.cos(2 * np.pi * red[:, 1])
         + np.cos(2 * np.pi * (red[:, 0] + red[:, 1])))
    s2 = np.clip((3 - F) / 4.5, 0, None)
    s_ = np.sqrt(s2)
    LO = 700 + 80 * s_ + (40 if split3d else 0) * (s_ < 1e-9)
    Fr = np.stack([400 * s2, 450 * s_, 750 * s_, 780 - 160 * s2,
                   700 + 40 * s2, LO], 1)
    odd = [1, 0, 0, 1, 0, 0]
    L = []
    for i, q in enumerate(red @ B):
        L += ["     diagonalizing the dynamical matrix ...\n", "\n",
              f"     q = {q[0]:12.4f} {q[1]:12.4f} {0.0:12.4f}\n",
              " " + "*" * 60 + "\n"]
        for m in np.argsort(Fr[i]):
            L.append(f"     freq ({m + 1:5d}) = 1.0 [THz] = {Fr[i, m]:12.6f} [cm-1]\n")
            e = (0, 0, 1) if odd[m] else (1, 0, 0)
            L += ["  ( " + "  ".join(f"{c / np.sqrt(2):9.6f} 0.000000"
                                     for c in e) + " )\n"] * 2
        L.append(" " + "*" * 60 + "\n")
    pathlib.Path(path).write_text("".join(L))


def _gmkg(npts):
    G, M, K = np.zeros(2), np.array([0.5, 0.0]), np.array([1 / 3, 1 / 3])
    return np.array([a_ + t * (b_ - a_) for a_, b_ in ((G, M), (M, K), (K, G))
                     for t in np.linspace(0, 1, npts)[:-1]] + [G])


def _ticks(out):
    return [l.split(":")[0].strip() for l in out.splitlines()
            if l.strip().startswith(("Gamma:", "M:", "K:"))]


def test_phonon_parity_takes_the_path_from_a_mesh(tmp_path):
    """The mesh's own points on G-M-K-G, found despite four-decimal q."""
    n = 36
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    _phonon_modes(tmp_path / "mesh.modes", np.stack([i.ravel() / n,
                                                     j.ravel() / n], 1))
    r = run_script("plot_phonon_parity.py", tmp_path, MODES="mesh.modes",
                   OUT="p.png")
    assert r.returncode == 0, r.stderr
    # 18 + 6 + 12 steps -> 37 points on a 36 x 36 mesh
    assert "the 36x36 mesh -- its 37 points lying exactly on G-M-K-G" in r.stdout
    assert _ticks(r.stdout) == ["Gamma", "M", "K", "Gamma"]
    assert "WARNING" not in r.stdout and "[note] found" not in r.stdout


def test_phonon_parity_from_a_path_run(tmp_path):
    _phonon_modes(tmp_path / "path.modes", _gmkg(60))
    r = run_script("plot_phonon_parity.py", tmp_path, MODES="path.modes",
                   OUT="p.png")
    assert r.returncode == 0, r.stderr
    assert "odd branches per q: 2..2" in r.stdout
    assert "lines joined by parity" in r.stdout
    assert _ticks(r.stdout) == ["Gamma", "M", "K", "Gamma"]
    assert "WARNING" not in r.stdout
    gam = [l for l in r.stdout.splitlines() if l.strip().startswith("Gamma:")]
    assert gam[0].count(" o") == 2 and gam[0].count(" e") == 4


def test_phonon_parity_flags_a_3d_split_at_gamma(tmp_path):
    _phonon_modes(tmp_path / "p3d.modes", _gmkg(40), split3d=True)
    r = run_script("plot_phonon_parity.py", tmp_path, MODES="p3d.modes",
                   OUT="p.png")
    assert "in-plane optical modes differ" in r.stdout
    assert "loto_2d = .true." in r.stdout


def test_phonon_parity_square_path_is_not_a_mesh(tmp_path):
    _phonon_modes(tmp_path / "p64.modes", _gmkg(22)[:64])
    r = run_script("plot_phonon_parity.py", tmp_path, MODES="p64.modes",
                   OUT="p.png")
    assert r.returncode == 0, r.stderr
    assert "run along the path" in r.stdout


def test_phonon_parity_refuses_path_plus_freq(tmp_path):
    _phonon_modes(tmp_path / "path.modes", _gmkg(20))
    (tmp_path / "x.freq").write_text(" &plot nbnd= 6, nks= 1 /\n 0 0 0\n 1 2 3 4 5 6\n")
    r = run_script("plot_phonon_parity.py", tmp_path, MODES="path.modes",
                   PATH_FREQ="x.freq", OUT="p.png")
    assert r.returncode != 0 and "set PATH_FREQ = None" in (r.stdout + r.stderr)


def test_linewidth_bz_script(tmp_path):
    import xphd
    n = 12
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    Q = np.stack([i.ravel() / n, j.ravel() / n, np.zeros(n * n)], 1)
    LW = np.full((1, n * n, 2), 5e-3)
    LW[0, 0, 0] = 1e-3
    xphd.LinewidthField(Q, np.array([77.0]), LW, np.zeros((n * n, 2)),
                        dict(mesh=np.array([n, n]))).save(tmp_path / "f.npz")
    xphd.LinewidthField(Q[:19], np.array([77.0]), LW[:, :19],
                        np.zeros((19, 2)),
                        dict(mesh=np.array([n, n]))).save(tmp_path / "ibz.npz")
    r = run_script("plot_linewidth_bz.py", tmp_path, FIELD="f.npz", OUT="m.png",
                   GRID_RES=120)
    assert r.returncode == 0, r.stderr
    assert "at Gamma 1.000" in r.stdout and (tmp_path / "m.png").exists()
    r = run_script("plot_linewidth_bz.py", tmp_path, FIELD="ibz.npz",
                   GRID_RES=120)
    assert r.returncode != 0 and "Unfold it first" in (r.stderr + r.stdout)


def _valley(tmp_path, tag, b, hc, C, n=24, r=15):
    """An exactly solvable minimum: band E0 + b q^2, LA branch hbar c q and
    coupling |G|^2 2 hbar w = C q^2, built from first harmonics only so that
    Fourier refinement reproduces them exactly."""
    from xphd.core.mesh import hex_norm

    def s2(q1, q2):
        c = (np.cos(2 * np.pi * q1) + np.cos(2 * np.pi * q2)
             + np.cos(2 * np.pi * (q1 + q2))) / 3
        return 3 * (1 - c) / (4 * np.pi ** 2)

    def build(m):
        i, j = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")
        q1, q2 = i.ravel() / m, j.ravel() / m
        S = s2(q1, q2)
        return q1, q2, S, np.stack([0.8 * S, 0.75 * hc * np.sqrt(S),
                                    hc * np.sqrt(S)], 1)

    q1, q2, S, hw = build(n)
    qn = hex_norm(np.stack([q1, q2, 0 * q1], 1))
    g2 = np.zeros((n * n, 3, 1, 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        g2[:, 2, 0, 0] = np.where(hw[:, 2] > 0, C * qn ** 2 / (2 * hw[:, 2]), 0)
    G = np.sqrt(g2).astype(complex)
    np.savez(tmp_path / f"{tag}.npz", G_grid=G, g2_grid=g2, Ge_grid=G * .5,
             Gh_grid=G * .5, E_m_grid=(3.4 + b * S)[:, None], hw_grid=hw,
             E_n_grid=np.array([3.4]), q_red=np.stack([q1, q2, 0 * q1], 1),
             Q_red=np.zeros(3), mesh=np.array([n, n, 1]))
    np.save(tmp_path / f"{tag}_hw.npy", build(n * r)[3])


def test_minimum_linewidth_matches_closed_form(tmp_path):
    """With the ring well inside the Gamma cell, xphd's integration and the
    closed form agree -- which also pins the 4 pi^2 / sqrt3 prefactor."""
    _valley(tmp_path, "v", b=13.0, hc=0.18, C=0.01)
    r = run_script("check_minimum_linewidth.py", tmp_path, ARCHIVE="v.npz",
                   HW_FINE="v_hw.npy", TEMPERATURES=[77.0])
    assert r.returncode == 0, r.stderr
    assert "C = 1.000e-02" in r.stdout and "hbar c = 0.1800" in r.stdout
    assert "100% of the ring inside" in r.stdout
    ratio = float(r.stdout.split("ratio ")[1].split()[0])
    assert 0.97 < ratio < 1.05, ratio


def test_minimum_linewidth_flags_a_ring_leaving_the_cell(tmp_path):
    _valley(tmp_path, "e", b=7.0, hc=0.18, C=0.01)
    r = run_script("check_minimum_linewidth.py", tmp_path, ARCHIVE="e.npz",
                   HW_FINE="e_hw.npy", TEMPERATURES=[77.0], RUN_CODE=False)
    assert r.returncode == 0, r.stderr
    assert "part of the ring lies outside the Gamma cell" in r.stdout


def _crossing_valley(tmp_path, kink, b=13.0, hc=0.18, C=0.01, n=24, r=15):
    """A minimum whose energy-ordered band has a crossing `kink` eV up, far
    from q = 0 -- the kind of kink a global interpolant rings from."""
    from xphd.core.mesh import hex_norm

    def s2(q1, q2):
        c = (np.cos(2 * np.pi * q1) + np.cos(2 * np.pi * q2)
             + np.cos(2 * np.pi * (q1 + q2))) / 3
        return 3 * (1 - c) / (4 * np.pi ** 2)

    def build(m):
        i, j = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")
        q1, q2 = i.ravel() / m, j.ravel() / m
        S = s2(q1, q2)
        band = np.minimum(3.4 + b * S, 3.4 + kink)
        return q1, q2, band, np.stack([0.8 * S, 0.75 * hc * np.sqrt(S),
                                       hc * np.sqrt(S)], 1)

    q1, q2, band, hw = build(n)
    qn = hex_norm(np.stack([q1, q2, 0 * q1], 1))
    g2 = np.zeros((n * n, 3, 1, 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        g2[:, 2, 0, 0] = np.where(hw[:, 2] > 0, C * qn ** 2 / (2 * hw[:, 2]), 0)
    G = np.sqrt(g2).astype(complex)
    p = tmp_path / "x.npz"
    np.savez(p, G_grid=G, g2_grid=g2, Ge_grid=G * .5, Gh_grid=G * .5,
             E_m_grid=band[:, None], hw_grid=hw, E_n_grid=np.array([3.4]),
             q_red=np.stack([q1, q2, 0 * q1], 1), Q_red=np.zeros(3),
             mesh=np.array([n, n, 1]))
    kB = 8.617333e-5
    exact = ((4 * np.pi ** 2 / np.sqrt(3)) * C
             / np.expm1(hc ** 2 / b / (kB * 77.0)) / b ** 2)
    return p, build(n * r)[3], exact


def test_local_band_keeps_nodes_and_the_minimum_curvature(tmp_path):
    from xphd.core.interp import fine_band
    n, r = 24, 15
    p, _, _ = _crossing_valley(tmp_path, kink=0.15)
    E = np.load(p)["E_m_grid"][:, 0].reshape(n, n)
    for method in ("local", "fourier"):
        assert np.allclose(fine_band(E, r, method)[::r, ::r], E, atol=1e-12)
    # curvature at the first fine points around the minimum
    loc, fou = fine_band(E, r, "local"), fine_band(E, r, "fourier")
    q2 = (1 / (n * r)) ** 2
    pts = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    bl = [(loc[i % (n * r), j % (n * r)] - 3.4) / q2 for i, j in pts]
    bf = [(fou[i % (n * r), j % (n * r)] - 3.4) / q2 for i, j in pts]
    assert np.ptp(bl) < 1e-6 * np.mean(bl)          # isotropic, as the band is
    assert abs(np.mean(bl) / 13.0 - 1) < 0.01       # the BSE curvature
    assert abs(np.mean(bf) / 13.0 - 1) > 0.3        # Fourier rings from the kink


def test_local_band_fixes_the_minimum_linewidth(tmp_path):
    """A crossing 150 meV above the minimum moved its width by 35% under the
    Fourier interpolant; the local one keeps it at the closed form."""
    import xphd
    from xphd.linewidth import compute
    p, hwf, exact = _crossing_valley(tmp_path, kink=0.15)
    kw = dict(hw_fine=hwf, refine=15, acoustic_cut=5e-4, acoustic_model="q2",
              n_shell=1, verbose=False)
    loc = compute(xphd.ExcPhArchive(p), [77.0], band_interp="local", **kw)
    fou = compute(xphd.ExcPhArchive(p), [77.0], band_interp="fourier", **kw)
    assert abs(loc.total[0, 0] / exact - 1) < 0.05
    assert abs(fou.total[0, 0] / exact - 1) > 0.2
    assert loc.meta["band_interp"] == "local"


def test_sweep_applies_the_same_band_interpolation(archive, monkeypatch):
    import sys, xphd
    from xphd.sweep import sweep
    mod = sys.modules["xphd.sweep"]
    seen, real = [], mod.compute

    def spy(*args, **kw):
        seen.append(kw.get("band_interp"))
        return real(*args, **kw)

    monkeypatch.setattr(mod, "compute", spy)
    fld = sweep([str(archive)], [300.0], verbose=False, refine=5,
                band_interp="fourier")
    assert seen == ["fourier"]
    one = real(xphd.ExcPhArchive(archive), [300.0], verbose=False, refine=5,
               band_interp="fourier")
    assert np.allclose(fld.LW[0, 0], one.total[0], rtol=1e-12)
    sweep([str(archive)], [300.0], verbose=False, refine=5)
    assert seen[-1] == "local"                       # the default


def test_text_headers_record_the_settings(archive, tmp_path):
    """A linewidth file must say which small-q model and band interpolation
    produced it -- the numbers alone cannot."""
    import xphd
    from xphd.linewidth import compute
    res = compute(xphd.ExcPhArchive(archive), [77.0], refine=5, verbose=False,
                  acoustic_model="q2", n_shell=2, band_interp="local")
    res.save(str(tmp_path / "a.txt"))
    res.save_modes(str(tmp_path / "a_modes.txt"))
    for f in ("a.txt", "a_modes.txt"):
        head = (tmp_path / f).read_text()
        assert "# acoustic_model q2 (n_shell 2)   band_interp local" in head, f
    res = compute(xphd.ExcPhArchive(archive), [77.0], refine=5, verbose=False,
                  band_interp="fourier")
    res.save(str(tmp_path / "b.txt"))
    assert "# acoustic_model none   band_interp fourier" in (tmp_path / "b.txt").read_text()


def _helicity_case(tmp_path, tag, gK, gKp):
    """A bright doublet with linear x and y dipoles; the lowest emitter at K
    (K') couples to the doublet through amplitudes gK (gKp) on branch 1."""
    n, nm, ne = 6, 6, 3
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    q = np.stack([i.ravel() / n, j.ravel() / n, 0 * i.ravel()], 1)
    nq = n * n
    iK = int(np.where(np.all(np.isclose(q[:, :2], [1 / 3, 1 / 3]), 1))[0][0])
    iKp = int(np.where(np.all(np.isclose(q[:, :2], [2 / 3, 2 / 3]), 1))[0][0])
    Em = np.tile([4.3, 4.5, 4.7], (nq, 1))
    Em[0] = [3.81, 3.81, 4.2]
    Em[iK] = Em[iKp] = [3.43, 3.90, 4.10]
    hw = np.tile([0.0162, 0.018, 0.029, 0.042, 0.094, 0.100], (nq, 1))
    hw[0] = 0
    G = np.zeros((nq, nm, ne, ne), complex)
    G[iK, 0, :2, 0] = np.conj(gK)
    G[iKp, 0, :2, 0] = np.conj(gKp)
    np.savez(tmp_path / f"{tag}.npz", G_grid=G, g2_grid=np.abs(G) ** 2,
             Ge_grid=G * .5, Gh_grid=G * .5, E_m_grid=Em, hw_grid=hw,
             E_n_grid=np.array([3.81, 3.81, 4.2]), q_red=q, Q_red=np.zeros(3),
             mesh=np.array([n, n, 1]))
    D = np.zeros((3, ne), complex)
    D[0, 0] = D[1, 1] = 1.0
    np.save(tmp_path / "dip.npy", D)
    r = run_script("helicity_pl.py", tmp_path, ARCHIVE=f"{tag}.npz",
                   DIPOLES="dip.npy", OUT=f"{tag}.pdf", NPZ_OUT=f"{tag}_o.npz")
    assert r.returncode == 0, r.stderr
    P = [float(l.split("=")[-1]) for l in r.stdout.splitlines() if "valley K" in l]
    rule = float(r.stdout.split("largest relative difference ")[1].split()[0])
    return P, rule, r.stdout


def test_helicity_pl_valley_locked(tmp_path):
    s2 = np.sqrt(2)
    P, rule, out = _helicity_case(tmp_path, "chiral", np.array([1, 1j]) / s2,
                                  np.array([1, -1j]) / s2)
    assert rule < 1e-12                          # I+ + I- = I_x + I_y
    assert abs(abs(P[0]) - 1) < 1e-6 and abs(P[0] + P[1]) < 1e-6
    assert "16.2 meV below the lowest emitter" in out


def test_helicity_pl_linear_coupling_is_unpolarised(tmp_path):
    s2 = np.sqrt(2)
    P, rule, _ = _helicity_case(tmp_path, "linear", np.array([1, 1]) / s2,
                                np.array([1, 1]) / s2)
    assert rule < 1e-12
    assert abs(P[0]) < 1e-6 and abs(P[1]) < 1e-6


def test_crossing_radii_recovers_planted_radii(tmp_path):
    """Exciton contour planted at 0.300|b|, ZO crossing of branch 4 at 0.250|b|."""
    def cart(k):
        return np.stack([k[:, 0] + 0.5 * k[:, 1], np.sqrt(3) / 2 * k[:, 1]], 1)

    def rad(k):
        d = k - np.rint(k)
        return np.linalg.norm(cart(d), axis=1)

    n = 24
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    Q = np.stack([i.ravel() / n, j.ravel() / n, 0 * i.ravel()], 1).astype(float)
    r = rad(Q[:, :2])
    Ee, Eo = 3.8 + 10 * r ** 2, np.full_like(r, 4.7)
    E = np.stack([np.minimum(Ee, Eo), np.maximum(Ee, Eo), Ee + 1, Eo + 1], 1)
    chi = np.stack([np.where(Ee < Eo, 1., -1.), np.where(Ee < Eo, -1., 1.),
                    np.ones_like(r), -np.ones_like(r)], 1)
    chi[r < 0.20] = 1.0      # as in GaN: no odd state among those computed near Gamma
    np.savez(tmp_path / "parity.npz", Q_red=Q, chi=chi, E=E)
    N = 36
    with open(tmp_path / "matdyn.modes", "w") as f:
        f.write("     diagonalizing the dynamical matrix\n\n")
        for a in range(N):
            for b in range(N):
                k = np.array([[a / N, b / N]])
                rr = rad(k)[0]
                qc = cart(k - np.rint(k))[0] * (2 / np.sqrt(3))
                f.write(f"     q = {qc[0]:12.6f} {qc[1]:12.6f} {0.0:12.6f}\n **\n")
                for m in range(6):
                    f.write(f"     freq ({m + 1:5d}) = {1.0:15.6f} [THz] = "
                            f"{100.0 * (m + 1):15.6f} [cm-1]\n")
                    if m == 3:
                        wz = 1 / (1 + np.exp((rr - 0.25) / 0.01))
                        z, x = np.sqrt(wz / 2), np.sqrt((1 - wz) / 2)
                    else:
                        z, x = 0.0, np.sqrt(0.5)
                    for _ in range(2):
                        f.write(f" ( {x:10.6f} 0.0 0.0 0.0 {z:10.6f} 0.0 )\n")
                f.write(" **\n")
    res = run_script("crossing_radii.py", tmp_path, N_FINE=36)
    assert res.returncode == 0, res.stderr
    got = {l.split(":")[0].strip(): l for l in res.stdout.splitlines()
           if "contour at" in l or "q_x =" in l}
    cont = [float(l.split("contour at")[1].split()[0]) for l in res.stdout.splitlines()
            if "contour at" in l]
    qx = [float(l.split("q_x =")[1].split()[0]) for l in res.stdout.splitlines()
          if "q_x =" in l]
    assert all(abs(c - 0.300) < 0.005 for c in cont), cont
    assert all(abs(q - 0.250) < 0.01 for q in qx), qx


def _qp_file(path, dE63, dE64, skip=None):
    """A yambo-like o-*.qp: 19 k-points, bands 57-68; band 62 peaks at k = 7 (K),
    where the DFT conduction bands are 63 at 1.200 and 64 at 1.236 eV."""
    with open(path, "w") as f:
        f.write("#  K-point  Band  Eo [eV]  E-Eo [eV]  Sc|Eo [eV]\n")
        for k in range(1, 20):
            for b in range(57, 69):
                if skip and (b in skip):
                    continue
                eo = -0.5 + 0.01 * k if b <= 62 else 1.5
                if k == 7:
                    eo = {61: -0.45, 62: 0.0, 63: 1.200, 64: 1.236}.get(b, eo)
                de = {63: dE63, 64: dE64}.get(b, -0.3 if b <= 62 else 0.9)
                f.write(f"  {k:3d}  {b:3d}  {eo:10.4f}  {de:10.4f}  0.0\n")


def _band_parity(path):
    n = 12
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    kf = np.stack([i.ravel() / n, j.ravel() / n, 0 * i.ravel()], 1)
    lab = np.tile([-1, 1, 1, -1, 1, -1, -1, 1], (len(kf), 1)).astype(float)   # WSe2 at K
    np.savez(path, kf=kf, p=lab, bands=np.arange(59, 67))


def test_cb_splitting_flags_reversed_conduction_bands(tmp_path):
    _qp_file(tmp_path / "o.qp", dE63=0.90, dE64=0.84)          # GW flips 63/64
    _band_parity(tmp_path / "bp.npz")
    r = run_script("cb_splitting.py", tmp_path, QP_FILE="o.qp", BAND_PARITY="bp.npz")
    assert r.returncode == 0, r.stderr
    assert "at K (k 7):" in r.stdout
    assert "DFT +36.0 meV,  after GW -24.0 meV" in r.stdout
    assert "REVERSES" in r.stdout and "SAME spin as the valence maximum" in r.stdout
    assert "Mo-like" in r.stdout


def test_cb_splitting_correct_order_and_missing_band(tmp_path):
    _qp_file(tmp_path / "o.qp", dE63=0.90, dE64=0.91)
    _band_parity(tmp_path / "bp.npz")
    r = run_script("cb_splitting.py", tmp_path, QP_FILE="o.qp", BAND_PARITY="bp.npz")
    assert "REVERSES" not in r.stdout and "W-like" in r.stdout
    _qp_file(tmp_path / "o2.qp", 0.9, 0.91, skip={65})
    r = run_script("cb_splitting.py", tmp_path, QP_FILE="o2.qp", BAND_PARITY=None)
    assert "have no correction (bands [65])" in r.stdout


def test_labels_from_elph_nonplanar_permutation(tmp_path):
    from netCDF4 import Dataset
    n = 6
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    q = np.stack([i.ravel() / n, j.ravel() / n, 0 * i.ravel()], 1).astype(float)
    modes = [[[0, 0, 1], [0, 0, 1], [0, 0, 1]], [[0, 1, 0], [0, 1, 0], [0, 1, 0]],
             [[1, 0, 0], [1, 0, 0], [1, 0, 0]], [[0, 0, 0], [1, 0, 0], [-1, 0, 0]],
             [[0, 0, 0], [0, 1, 0], [0, -1, 0]], [[1, 0, 0], [-.5, 0, 0], [-.5, 0, 0]],
             [[0, 1, 0], [0, -.5, 0], [0, -.5, 0]], [[0, 0, 0], [0, 0, 1], [0, 0, -1]],
             [[0, 0, 1], [0, 0, -.5], [0, 0, -.5]]]
    ev = np.array(modes, float)[None].repeat(len(q), 0) * np.exp(0.4j)
    f_ry = np.tile(np.linspace(1, 9, 9) * 2e-4, (len(q), 1))
    with Dataset(tmp_path / "ndb.elph", "w") as db:
        for nm, s in (("nq", len(q)), ("nm", 9), ("nat", 3), ("x", 3), ("c", 2)):
            db.createDimension(nm, s)
        db.createVariable("qpoints", "f8", ("nq", "x"))[:] = q
        db.createVariable("FREQ", "f8", ("nq", "nm"))[:] = f_ry
        v = db.createVariable("POLARIZATION_VECTORS", "f8", ("nq", "nm", "nat", "x", "c"))
        v[:] = np.stack([ev.real, ev.imag], -1)
    hw = f_ry * 13.605693122994
    G = np.zeros((len(q), 9, 2, 2), complex)
    np.savez(tmp_path / "arc.npz", G_grid=G, g2_grid=np.abs(G) ** 2, Ge_grid=G,
             Gh_grid=G, E_m_grid=np.ones((len(q), 2)), hw_grid=hw,
             E_n_grid=np.ones(2), q_red=q, Q_red=np.zeros(3), mesh=np.array([n, n, 1]))
    r = run_script("labels_from_elph.py", tmp_path, ELPH="ndb.elph",
                   ARCHIVE="arc.npz", PERM=[0, 2, 1], OLD=None)
    assert r.returncode == 0, r.stderr
    assert "non-planar layer" in r.stdout and "(same modes, same order)" in r.stdout
    lab = np.load(tmp_path / "mode_labels_elph.npy")
    want = [3, 1, 1, 3, 3, 1, 1, 1, 3]          # ZA TA LA E'' E'' E' E' A1' A2''
    assert (lab == np.array(want)).all()
    assert "would disagree at 33.3%" in r.stdout


def _wse2_like_case(path, n=12, ne=6, nmod=9, seed=5, mislabel_branch=None):
    """Nine branches, a 12x12 mesh, couplings that obey the sigma_h rule exactly:
    the even and odd exciton families cross at |Q| ~ 0.25|b|, branches 1, 4 and 8
    are odd and branch 4 turns even beyond 0.30|b|."""
    rng = np.random.default_rng(seed)
    qf = np.array([[i / n, j / n] for i in range(n) for j in range(n)])
    nq = len(qf)
    d = qf - np.rint(qf)
    r = np.linalg.norm(np.stack([d[:, 0] + 0.5 * d[:, 1], np.sqrt(3) / 2 * d[:, 1]], 1), axis=1)
    Ee, Eo = 1.40 + 2.0 * r ** 2, 1.55 - 0.5 * r ** 2
    E, chi = np.zeros((nq, ne)), np.zeros((nq, ne))
    for q in range(nq):
        fam = sorted([(Ee[q], 1), (Eo[q], -1), (Ee[q] + .3, 1), (Eo[q] + .3, -1),
                      (Ee[q] + .5, 1), (Eo[q] + .5, -1)], key=lambda t: t[0])
        E[q] = [t[0] for t in fam][:ne]
        chi[q] = [t[1] for t in fam][:ne]
    p_lam = chi[0].copy()
    lab = np.full((nq, nmod), 1)
    for nu in (0, 3, 7):
        lab[:, nu] = 3
    lab[:, 3] = np.where(r < 0.30, 3, 1)
    p_nu = np.where(lab == 3, -1, 1)
    g2 = np.zeros((nq, nmod, ne, ne))
    for q in range(nq):
        for nu in range(nmod):
            for lam in range(ne):
                for b in range(ne):
                    if p_nu[q, nu] * chi[q, b] == p_lam[lam]:
                        g2[q, nu, lam, b] = rng.uniform(0.2, 1.0) * 1e-6
    lab_used = lab.copy()
    if mislabel_branch is not None:
        lab_used[:, mislabel_branch] = np.where(lab_used[:, mislabel_branch] == 3, 1, 3)
    G = np.sqrt(g2) * np.exp(1j * rng.uniform(0, 6.28, g2.shape))
    Q3 = np.column_stack([qf, np.zeros(nq)])
    np.savez(path / "a.npz", G_grid=G, g2_grid=np.abs(G) ** 2, Ge_grid=G / 2, Gh_grid=G / 2,
             E_n_grid=E[0], E_m_grid=E, hw_grid=np.tile(np.linspace(.003, .04, nmod), (nq, 1)),
             q_red=Q3, Q_red=np.zeros(3), mesh=np.array([n, n, 1]))
    np.savez(path / "parity.npz", Q_red=Q3, chi=chi, E=E)
    np.save(path / "labels.npy", lab_used)


def _shares(stdout):
    import re
    return [float(l.split()[1]) for l in stdout.splitlines()
            if re.match(r"\s+\d+\s+\d\.\d+e[+-]\d+$", l)]


def test_coupling_maps_nine_branches_zeros_under_the_hatching(tmp_path):
    _wse2_like_case(tmp_path)
    r = run_script("plot_coupling_maps.py", tmp_path, ARCHIVE="a.npz", PARITY="parity.npz",
                   LABELS="labels.npy", OVERLAY="both", OUT="m.pdf", DPI=60)
    assert r.returncode == 0, r.stderr
    assert "9 branches" in r.stdout
    assert "initial state 1 at Gamma: chi = +1.000  ->  even" in r.stdout
    assert "lowest exciton at Gamma: even" in r.stdout
    sh = _shares(r.stdout)
    assert len(sh) == 9 and max(sh) < 1e-12
    assert (tmp_path / "m.pdf").exists()


def test_coupling_maps_flag_a_wrong_phonon_parity(tmp_path):
    _wse2_like_case(tmp_path, mislabel_branch=1)
    r = run_script("plot_coupling_maps.py", tmp_path, ARCHIVE="a.npz", PARITY="parity.npz",
                   LABELS="labels.npy", OVERLAY="both", OUT="m.pdf", DPI=60)
    sh = _shares(r.stdout)
    assert sh[1] > 0.1 and max(sh[:1] + sh[2:]) < 1e-12


def test_selection_rule_with_nine_branches_and_parity_only_labels(tmp_path, capsys, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    from xphd.selection import main as sel
    _wse2_like_case(tmp_path)
    monkeypatch.chdir(tmp_path)
    sel(["--archive", "a.npz", "--parity", "parity.npz", "--labels", "labels.npy",
         "--state", "3", "--by-final", "--nfinal", "2", "--out", "s.pdf"])
    out = capsys.readouterr().out
    assert "forbidden    0.0000e+00    0.00%" in out
    assert "[odd, odd at this q]" in out or "[even, even at this q]" in out   # not ZO / TA
    with pytest.raises(SystemExit) as e:          # too few states in parity.npz
        d = dict(np.load("parity.npz"))
        d["chi"], d["E"] = d["chi"][:, :2], d["E"][:, :2]
        np.savez("p2.npz", **d)
        sel(["--archive", "a.npz", "--parity", "p2.npz", "--labels", "labels.npy",
             "--out", "s2.pdf"])
    assert "--nstates 6" in str(e.value)


def test_mode_labels_refuses_a_three_atom_layer(tmp_path, monkeypatch):
    from xphd.mode_labels import main as ml
    _wse2_like_case(tmp_path)
    n = 3
    with open(tmp_path / "matdyn.modes", "w") as f:
        f.write("     diagonalizing the dynamical matrix\n\n")
        for a in range(n):
            for b in range(n):
                f.write(f"     q = {a / n:12.4f}{b / n:12.4f}{0.0:12.4f}\n **\n")
                for m in range(9):
                    f.write(f"     freq ({m + 1:5d}) = 1.0 [THz] = {30.0 * (m + 1):15.6f} [cm-1]\n")
                    for _ in range(3):
                        f.write(" ( 0.5 0.0 0.0 0.0 0.5 0.0 )\n")
                f.write(" **\n")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as e:
        ml(["--modes", "matdyn.modes", "--fine", "3", "--mesh", "3",
            "--masses", "183.84", "78.96", "78.96", "--archive", "a.npz"])
    assert "labels_from_elph.py" in str(e.value)



def _leaky_case(path):
    """The rule-obeying case, plus one final state (index 2) carrying an
    opposite-parity admixture eps = 0.002 that couples at eps times normal."""
    _wse2_like_case(path)
    d = dict(np.load(path / "parity.npz"))
    chi = d["chi"].copy()
    rng0 = np.random.default_rng(2)
    e_q = 10 ** rng0.uniform(-4, -2, chi.shape[0])        # eps varies with q
    chi[:, 2] *= (1 - 2 * e_q)
    d["chi"] = chi
    np.savez(path / "parity.npz", **d)
    a = dict(np.load(path / "a.npz"))
    g2 = a["g2_grid"].copy()
    lab = np.load(path / "labels.npy")
    p_nu = np.where(lab == 3, -1, 1)
    p_lam = np.sign(chi[0])
    rng = np.random.default_rng(1)
    for q in range(g2.shape[0]):
        for nu in range(g2.shape[1]):
            for lam in range(g2.shape[2]):
                if p_nu[q, nu] * np.sign(chi[q, 2]) != p_lam[lam]:
                    g2[q, nu, lam, 2] = e_q[q] * rng.uniform(0.2, 1.0) * 1e-6
    a["g2_grid"] = g2
    a["G_grid"] = np.sqrt(g2)
    a["Ge_grid"] = a["Gh_grid"] = np.sqrt(g2) / 2
    np.savez(path / "a.npz", **a)


def test_where_forbidden_attributes_leakage_to_admixture(tmp_path):
    _leaky_case(tmp_path)
    r = run_script("where_forbidden.py", tmp_path, ARCHIVE="a.npz", PARITY="parity.npz",
                   LABELS="labels.npy")
    assert r.returncode == 0, r.stderr
    line = [l for l in r.stdout.splitlines() if "correlation of log r" in l][0]
    corr = float(line.split(":")[1].split(";")[0])
    med = float(line.split("median r/eps:")[1])
    # the synthetic couplings scatter (allowed/forbidden combination counts vary
    # with q), so +0.72 over two decades of eps is the expected value here
    assert corr > 0.6 and 0.1 < med < 10        # r follows eps, at order-1 ratio
    assert "forbidden weight on states with eps = 0 exactly:   0.0%" in r.stdout
    assert "m 3" in r.stdout                     # 1-based final state 3 carries it


def test_where_forbidden_says_labels_when_states_are_pure(tmp_path):
    _wse2_like_case(tmp_path, mislabel_branch=1)
    r = run_script("where_forbidden.py", tmp_path, ARCHIVE="a.npz", PARITY="parity.npz",
                   LABELS="labels.npy")
    assert r.returncode == 0, r.stderr
    assert "cannot be admixture leakage" in r.stdout
    assert "forbidden weight on states with eps = 0 exactly: 100.0%" in r.stdout



def test_where_forbidden_verdict_names_a_mixed_pair_and_the_tolerance(tmp_path):
    """Final state 2 comes back mixed (|chi| = 0.92) at a few q, where it
    couples through the 'wrong' phonons: the verdict must say STATE MIX and
    suggest a --tol just above 0.92."""
    _wse2_like_case(tmp_path)
    d = dict(np.load(tmp_path / "parity.npz"))
    chi = d["chi"].copy()
    qs = np.arange(5, chi.shape[0], 11)
    chi[qs, 1] *= 0.92
    d["chi"] = chi
    np.savez(tmp_path / "parity.npz", **d)
    a = dict(np.load(tmp_path / "a.npz"))
    g2 = a["g2_grid"].copy()
    lab = np.load(tmp_path / "labels.npy")
    p_nu = np.where(lab == 3, -1, 1)
    p_lam = np.sign(chi[0])
    for q in qs:
        for nu in range(g2.shape[1]):
            for lam in range(g2.shape[2]):
                if p_nu[q, nu] * np.sign(chi[q, 1]) != p_lam[lam]:
                    g2[q, nu, lam, 1] = 0.04e-6
    a["g2_grid"] = g2
    a["G_grid"] = np.sqrt(g2)
    a["Ge_grid"] = a["Gh_grid"] = np.sqrt(g2) / 2
    np.savez(tmp_path / "a.npz", **a)
    r = run_script("where_forbidden.py", tmp_path, ARCHIVE="a.npz", PARITY="parity.npz",
                   LABELS="labels.npy")
    assert r.returncode == 0, r.stderr
    assert "STATE MIX" in r.stdout and "--tol 0.93" in r.stdout


def test_add_osc_writes_in_plane_strengths_for_optical_injection(tmp_path):
    import xphd
    _wse2_like_case(tmp_path)
    dip = np.zeros((3, 8), complex)
    dip[0, 2], dip[1, 3], dip[2, 0] = 1.0, 1j, 0.5      # 3, 4 in-plane bright; 1 only z
    np.save(tmp_path / "exc_dipoles.npy", dip)
    r = run_script("add_osc.py", tmp_path, ARCHIVE="a.npz", DIPOLES="exc_dipoles.npy")
    assert r.returncode == 0, r.stderr
    a = xphd.ExcPhArchive(str(tmp_path / "a.npz"))
    osc = np.asarray(a._raw("osc_Q0"))
    assert a.has("osc_Q0") and np.allclose(osc, [0, 0, 1, 1, 0, 0])
    assert (tmp_path / "a.bak.npz").exists()
    np.save(tmp_path / "short.npy", dip[:, :3])           # too few states: refuse
    r = run_script("add_osc.py", tmp_path, ARCHIVE="a.npz", DIPOLES="short.npy")
    assert r.returncode != 0 and "holds 3 states" in (r.stdout + r.stderr)


def _snaps(path, tau):
    n = 12
    Q = np.array([[i / n, j / n, 0] for i in range(n) for j in range(n)])
    r = np.linalg.norm(((Q[:, :2] + 0.5) % 1 - 0.5) @ np.array([[1, 0], [0.5, np.sqrt(3) / 2]]), axis=1)
    E = 5.475 + 3.0 * r ** 2
    wB = np.exp(-(E - E.min()) / (8.617333e-5 * 300))
    wB /= wB.sum()
    t = np.linspace(0, 10000, 51)
    F0 = np.exp(-((E - 6.0) / 0.05) ** 2)
    F0 /= F0.sum()
    F = np.array([wB + (F0 - wB) * np.exp(-tt / tau) for tt in t])
    np.savez(path / "snaps.npz", t=t, F=F, E=E, Q_red=Q)


def test_check_bte_settled_at_300K_is_not_all_at_gamma(tmp_path):
    _snaps(tmp_path, tau=300.0)
    r = run_script("check_bte.py", tmp_path, BTE="snaps.npz", T_BTE=300.0, CENTERS={"Gamma": [0, 0]})
    assert r.returncode == 0, r.stderr
    assert "settled: the end state matches Boltzmann at 300 K" in r.stdout
    line = [l for l in r.stdout.splitlines() if "minimum's Q-point" in l][0]
    bte, boltz = float(line.split()[-2].rstrip("%")), float(line.split()[-1].rstrip("%"))
    assert abs(bte - boltz) < 0.5 and boltz < 50          # far from 100% at Gamma, and right


def test_check_bte_says_still_relaxing(tmp_path):
    _snaps(tmp_path, tau=20000.0)
    r = run_script("check_bte.py", tmp_path, BTE="snaps.npz", T_BTE=300.0, CENTERS={"Gamma": [0, 0]})
    assert "still relaxing" in r.stdout


def test_qp_decompose_attributes_a_correlation_anomaly(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    rng = np.random.default_rng(1)
    rows, tab, Zs = [], [], []
    for k in range(1, 20):
        for b in range(59, 67):
            z = 0.78 + 0.01 * rng.normal()
            sx = -0.6 if b <= 62 else 0.4
            sc = 0.1 + (0.5 if (k == 13 and b == 63) else 0.0)
            rows.append((k, b, (b - 62.5) * 0.4 + 0.01 * k, z * (sx + sc), sc))
            tab.append((b, b, k))
            Zs.append((z, 0.0))
    with open(tmp_path / "o-output.qp", "w") as f:
        f.write("#  K-point  Band  Eo  E-Eo  Sc|Eo\n")
        for r in rows:
            f.write(f"  {r[0]:4d} {r[1]:4d} {r[2]:10.4f} {r[3]:10.4f} {r[4]:10.4f}\n")
    db = netCDF4.Dataset(tmp_path / "ndb.QP", "w")
    db.createDimension("a", 3); db.createDimension("n", len(tab)); db.createDimension("c", 2)
    db.createVariable("QP_table", "f8", ("a", "n"))[:] = np.array(tab, float).T
    db.createVariable("QP_Z", "f8", ("n", "c"))[:] = np.array(Zs)
    db.close()
    r = run_script("qp_decompose.py", tmp_path, HF_FILE="")
    assert r.returncode == 0, r.stderr
    line = [l for l in r.stdout.splitlines() if l.strip().startswith("Q") and " 63 " in l][-1]
    dE, dx, dc = (float(v) for v in line.split()[3:6])
    assert abs(dc - 0.386) < 0.01 and abs(dx) < 0.01          # all in Z*Sc
    # with an HF file whose exchange matches, the column and the HF agree
    with open(tmp_path / "o-hf.hf", "w") as f:
        f.write("#    K-point            Band               Eo [eV]            Ehf [eV]           Vxc [eV]           Vnlxc [eV]\n")
        for r_ in rows:
            sx_vxc = -0.6 if r_[1] <= 62 else 0.4
            f.write(f"  {r_[0]:4d} {r_[1]:4d} {r_[2]:10.4f} {r_[2] + sx_vxc:10.4f} {-15.0:10.4f} {-15.0 + sx_vxc:10.4f}\n")
    r = run_script("qp_decompose.py", tmp_path)
    assert "exchange from the HF run" in r.stdout
    assert "agree within" in r.stdout
    line = [l for l in r.stdout.splitlines() if l.strip().startswith("Q") and " 63 " in l][-1]
    dE, dx, dc = (float(v) for v in line.split()[3:6])
    assert abs(dc - 0.386) < 0.01 and abs(dx) < 0.01
    # an HF exchange that does NOT match the GW's: the mismatch is reported
    with open(tmp_path / "o-hf.hf", "w") as f:
        f.write("#    K-point            Band               Eo [eV]            Ehf [eV]           Vxc [eV]           Vnlxc [eV]\n")
        for r_ in rows:
            sx_vxc = -0.6 if r_[1] <= 62 else 2.4                # 2 eV off for the conduction band
            f.write(f"  {r_[0]:4d} {r_[1]:4d} {r_[2]:10.4f} {r_[2] + sx_vxc:10.4f} {-15.0:10.4f} {-15.0 + sx_vxc:10.4f}\n")
    r = run_script("qp_decompose.py", tmp_path)
    assert "WARNING: the Sc the HF exchange implies disagrees" in r.stdout


def _qp_rows():
    # (k, band, Eo, E): K is k 2, Q is k 1; GW reverses 63/64 at K, Q stays above K
    return [(1, 62, -0.88, -2.18), (1, 63, 1.323, 1.753), (1, 64, 1.527, 1.985),
            (2, 62, 0.00, -1.12), (2, 63, 1.259, 1.320), (2, 64, 1.296, 1.302)]


def test_cb_splitting_reads_extendout_by_name(tmp_path):
    hdr = ("#    K-point            Band               Eo [eV]            E [eV]             "
           "E-Eo [eV]          Vxc [eV]           Vnlxc [eV]         Sc|Eo [eV]\n")
    with open(tmp_path / "o-GW.qp", "w") as f:
        f.write(hdr)
        for k, b, eo, e in _qp_rows():
            f.write(f"  {k:4d} {b:4d} {eo:10.4f} {e:10.4f} {e - eo:10.4f} -15.0 -12.0 -4.0\n")
    r = run_script("cb_splitting.py", tmp_path, WINDOW=[62, 64], Q_POINT=1)
    assert r.returncode == 0, r.stderr
    assert "GW     k  2, 62,  -1.1200      k  2, 64,   1.3020" in r.stdout
    assert "direct" in r.stdout and "REVERSES" in r.stdout
    assert "Q - K =  +451.0 meV" in r.stdout


def test_cb_splitting_reads_the_plain_layout(tmp_path):
    with open(tmp_path / "o-GW.qp", "w") as f:
        f.write("#    K-point            Band               Eo [eV]            E-Eo [eV]          Sc|Eo [eV]\n")
        for k, b, eo, e in _qp_rows():
            f.write(f"  {k:4d} {b:4d} {eo:10.4f} {e - eo:10.4f} -4.0\n")
    r = run_script("cb_splitting.py", tmp_path, WINDOW=[62, 64], Q_POINT=1)
    assert r.returncode == 0, r.stderr
    assert "k  2, 64,   1.3020" in r.stdout and "+451.0 meV" in r.stdout


def _honeycomb_modes(tmp_path):
    """matdyn.modes at K for a planar binary honeycomb (springs to 1st and 2nd neighbours)."""
    import shutil, subprocess, sys
    src = pathlib.Path(__file__).parent / "_honeycomb_model.py"
    shutil.copy(src, tmp_path / "make_model.py")
    subprocess.run([sys.executable, "make_model.py", "lattice"], cwd=tmp_path, check=True,
                   capture_output=True)


def test_phonon_irreps_K_labels_follow_the_axis(tmp_path):
    _honeycomb_modes(tmp_path)
    pos = [("Ga", [0.0, 0.0]), ("N", [1 / 3, 2 / 3])]
    r = run_script("phonon_irreps_K.py", tmp_path, POSITIONS=pos, AXIS="origin")
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "atom on the axis: Ga" in out and "do not depend on the phase convention" in out
    rows = [l.split() for l in out.splitlines() if l.strip()[:1].isdigit() and "%" in l]
    lab = {int(x[0]): (x[-3] if x[-1].endswith("%") and x[-3] in ("Ga", "N") else None) for x in rows}
    flex = [l for l in out.splitlines() if " -1.00 " in l]
    assert any("A''" in l and "Ga 100%" in l for l in flex)          # on-axis flexural: A''
    assert any("E''_" in l and "N 100%" in l for l in flex)          # the other: E''_k
    # axis through N: the roles swap
    r = run_script("phonon_irreps_K.py", tmp_path, POSITIONS=pos, AXIS="N")
    flex = [l for l in r.stdout.splitlines() if " -1.00 " in l]
    assert any("A''" in l and "N 100%" in l for l in flex)
    assert any("E''_" in l and "Ga 100%" in l for l in flex)


def test_phonon_irreps_K_needs_positions_and_k(tmp_path):
    _honeycomb_modes(tmp_path)
    r = run_script("phonon_irreps_K.py", tmp_path)
    assert r.returncode != 0 and "Set POSITIONS" in (r.stdout + r.stderr)
    r = run_script("phonon_irreps_K.py", tmp_path, POSITIONS=[("Ga", [0, 0]), ("N", [1 / 3, 2 / 3])],
                   K_POINT=[0.25, 0.0])
    assert r.returncode != 0


def test_slice_archive_keeps_the_first_states(tmp_path):
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from test_archive_symmetry import _archive
    from xphd.io.excph import ExcPhArchive
    _archive(tmp_path / "a.npz")
    r = run_script("slice_archive.py", tmp_path, ARCHIVE="a.npz", N_STATES=[1, 2, 4])
    assert r.returncode == 0, r.stderr
    full = np.load(tmp_path / "a.npz")
    for n in (1, 2, 4):
        s = np.load(tmp_path / "sliced" / f"a_n{n}.npz")
        assert s["G_grid"].shape[-2:] == (n, n) and s["E_m_grid"].shape[1] == n
        assert np.array_equal(s["G_grid"], full["G_grid"][..., :n, :n])
        assert np.array_equal(s["E_n_grid"], full["E_n_grid"][:n])
        assert ExcPhArchive(str(tmp_path / "sliced" / f"a_n{n}.npz")).nexc == n
    assert "xphd linewidth sliced" in r.stdout


def test_star_uniformity_symmetric_and_broken(tmp_path):
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from test_archive_symmetry import _archive
    _archive(tmp_path / "a.npz")
    r = run_script("star_uniformity.py", tmp_path, ARCHIVE="a.npz", SYM="c3tr")
    assert r.returncode == 0, r.stderr
    worst = float(r.stdout.split("worst ")[1].split()[0])
    assert abs(worst - 1) < 1e-9
    _archive(tmp_path / "b.npz", break_tr=True)
    r = run_script("star_uniformity.py", tmp_path, ARCHIVE="b.npz", SYM="c3tr")
    assert r.returncode == 0, r.stderr
    assert float(r.stdout.split("worst ")[1].split()[0]) > 1.01


def test_star_uniformity_float32_q_on_24x24(tmp_path):
    """Archives store q in single precision: matching must survive it."""
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from test_archive_symmetry import _archive
    _archive(tmp_path / "a.npz", n=24, nmod=3, ne=6)
    d = dict(np.load(tmp_path / "a.npz"))
    d["q_red"] = d["q_red"].astype(np.float32).astype(float)
    np.savez(tmp_path / "a32.npz", **d)
    r = run_script("star_uniformity.py", tmp_path, ARCHIVE="a32.npz", SYM="c3tr")
    assert r.returncode == 0, r.stdout + r.stderr
    assert abs(float(r.stdout.split("worst ")[1].split()[0]) - 1) < 1e-9
    r = run_script("star_uniformity.py", tmp_path, ARCHIVE="a32.npz", SYM="c3tr", INITIAL=[1, 2], N_FINAL=3)
    assert r.returncode == 0 and "initial states [1, 2] and final states 1-3" in r.stdout


def test_phonon_irreps_K_alat_positions_N_at_origin(tmp_path):
    """GaN written with ATOMIC_POSITIONS {alat}: N at the origin, Ga at crystal (2/3, 1/3)."""
    import shutil, subprocess, sys
    shutil.copy(pathlib.Path(__file__).parent / "_honeycomb_model.py", tmp_path / "make_model.py")
    subprocess.run([sys.executable, "make_model.py", "lattice", "NGa"], cwd=tmp_path, check=True,
                   capture_output=True)
    user = dict(POSITIONS=[("N", [0.0, 0.0]), ("Ga", [0.4937423323, 0.2850622742])],
                CELL=[[0.987484665, 0.0], [-0.493742332, 0.855186805]], AXIS="origin")
    r = run_script("phonon_irreps_K.py", tmp_path, **user)                      # read as crystal
    assert r.returncode != 0 and "set POSITIONS_UNITS = 'alat'" in (r.stdout + r.stderr)
    r = run_script("phonon_irreps_K.py", tmp_path, POSITIONS_UNITS="alat", **user)
    assert r.returncode == 0, r.stderr
    assert "Ga (0.6667, 0.3333)" in r.stdout and "atom on the axis: N" in r.stdout
    flex = [l for l in r.stdout.splitlines() if " -1.00 " in l]
    assert any("A''" in l and "N 100%" in l for l in flex)
    assert any("E''_" in l and "Ga 100%" in l for l in flex)
