"""irreps_check.py on synthetic excitons (the yambo loaders are replaced)."""
import pathlib
import re
import runpy

import numpy as np
import pytest

SCRIPT = pathlib.Path(__file__).parent.parent / "scripts" / "irreps_check.py"


def _run(tmp_path, monkeypatch, A, ops, **settings):
    import xphd.excsym as xs
    import xphd.yambo as xy
    nk = A.shape[1]
    kf = np.zeros((nk, 3))
    kf[1:, 0] = np.arange(1, nk) / nk
    E = np.arange(A.shape[0], dtype=float) * 0.1
    E[1] = E[0] + 1e-4                                      # states 1-2: one manifold
    monkeypatch.setattr(xy, "lattice", lambda s: None)
    monkeypatch.setattr(xy, "load_dmats", lambda p: None)
    monkeypatch.setattr(xy, "full_zone_kpoints", lambda y: kf)
    monkeypatch.setattr(xy, "ibz_parents", lambda y: (np.zeros(nk, int), np.array([0])))
    monkeypatch.setattr(xy, "spatial_ops", lambda s, k, D, nv: ops)
    monkeypatch.setattr(xy, "load_excitons", lambda p, y, n: (None, A[:n], E[:n]))
    monkeypatch.setattr(xs, "stabilizer", lambda o, Q: o)
    monkeypatch.setattr(xs, "group_name", lambda s: "test")
    (tmp_path / "bse").mkdir(exist_ok=True)
    (tmp_path / "bse" / "ndb.BS_diago_Q1").write_text("")
    src = SCRIPT.read_text()
    settings = {"BSE_DIR": str(tmp_path / "bse"), "NSTATES": A.shape[0], **settings}
    for k, v in settings.items():
        src, n = re.subn(rf"^{k} = .*$", f"{k} = {v!r}", src, count=1, flags=re.M)
        assert n == 1, k
    p = tmp_path / "chk.py"
    p.write_text(src)
    runpy.run_path(str(p), run_name="__main__")


def _states(n=4, nk=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(nk, n)) + 1j * rng.normal(size=(nk, n))
    q, _ = np.linalg.qr(X)                                  # orthonormal columns
    return q.T.reshape(n, nk, 1, 1)


def _op(label, kmap, nk):
    I = np.ones((nk, 1, 1), complex)
    return (label, np.eye(3), I, I, np.asarray(kmap))


def test_clean_identity(tmp_path, monkeypatch, capsys):
    A = _states()
    _run(tmp_path, monkeypatch, A, [_op("E", np.arange(6), 6)])
    out = capsys.readouterr().out
    assert "unitary for every operation" in out


def test_rotation_out_of_the_space_is_flagged(tmp_path, monkeypatch, capsys):
    A = _states()
    _run(tmp_path, monkeypatch, A, [_op("E", np.arange(6), 6), _op("C3", [1, 2, 0, 4, 5, 3], 6)])
    out = capsys.readouterr().out
    assert "defect on: C3" in out


def test_non_orthonormal_vectors_are_named(tmp_path, monkeypatch, capsys):
    A = _states()
    A[1] *= 1.3
    _run(tmp_path, monkeypatch, A, [_op("E", np.arange(6), 6)])
    out = capsys.readouterr().out
    assert "NOT orthonormal" in out and "not orthonormal: the defect reflects that" in out
