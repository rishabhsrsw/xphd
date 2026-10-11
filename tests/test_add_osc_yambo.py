"""add_osc.py with SOURCE = "yambo": residuals shared within manifolds; energies checked."""
import pathlib
import re
import runpy

import numpy as np
import pytest

SCRIPT = pathlib.Path(__file__).parent.parent / "scripts" / "add_osc.py"


def _run(tmp_path, monkeypatch, E_db, resid, E_arc):
    import xphd.yambo as xy
    exc = type("Exc", (), {"r_residual": np.asarray(resid)})()
    monkeypatch.setattr(xy, "lattice", lambda s: None)
    monkeypatch.setattr(xy, "load_excitons", lambda p, y, n=None: (exc, None, np.asarray(E_db)))
    np.savez(tmp_path / "a.npz", E_n_grid=np.asarray(E_arc), Q_red=np.zeros(3))
    src = SCRIPT.read_text()
    for k, v in dict(ARCHIVE=str(tmp_path / "a.npz"), SOURCE="yambo").items():
        src, n = re.subn(rf"^{k} = .*$", f"{k} = {v!r}", src, count=1, flags=re.M)
        assert n == 1
    p = tmp_path / "osc.py"
    p.write_text(src)
    runpy.run_path(str(p), run_name="__main__")
    return np.load(tmp_path / "a.npz")["osc_Q0"]


def test_doublet_strength_is_shared_equally(tmp_path, monkeypatch, capsys):
    E = [5.4752, 5.4754, 5.90, 6.3659, 6.3661]
    # one field direction: each doublet's strength sits on one member
    resid = np.sqrt([5.0e-8, 0.0, 0.0, 0.0, 1.33e-8])
    osc = _run(tmp_path, monkeypatch, E, resid, E)
    assert osc[0] == pytest.approx(osc[1]) and osc[3] == pytest.approx(osc[4])
    assert osc[2] == 0
    out = capsys.readouterr().out
    assert "receive" in out and "state 1" in out


def test_different_bse_is_refused(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        _run(tmp_path, monkeypatch, [5.40, 5.41, 5.9], [1.0, 1.0, 0.0], [5.4752, 5.4754, 5.90])
