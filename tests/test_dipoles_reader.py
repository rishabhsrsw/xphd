"""read_cv_dipoles against planted ndb.dipoles files (needs yambopy)."""
import numpy as np
import pytest

yambopy = pytest.importorskip("yambopy")
netCDF4 = pytest.importorskip("netCDF4")
from yambopy.dbs.latticedb import YamboLatticeDB  # noqa: E402

from xphd.dipoles import read_cv_dipoles  # noqa: E402

NOCC, NK = 9, 4


def planted(v_band, c_band, i, k):
    return 1000 * v_band + 10 * c_band + i + 1j * (k + 1)


def lattice():
    lat = YamboLatticeDB.__new__(YamboLatticeDB)
    lat.spinor_components = 1
    lat.nelectrons = 2 * NOCC
    return lat


def write(path, bmin, bmax, ordered):
    nv, nc = NOCC - bmin + 1, bmax - NOCC
    n1, n2 = (nv, nc) if ordered else (bmax - bmin + 1,) * 2
    with netCDF4.Dataset(path, "w") as db:
        for n, s in (("four", 4), ("one", 1), ("pars", 9), ("sp", 1),
                     ("nk", NK), ("n1", n1), ("n2", n2), ("x", 3), ("c", 2)):
            db.createDimension(n, s)
        db.createVariable("HEAD_R_LATT", "f8", ("four",))[:] = [1, 1, NK, NK]
        db.createVariable("SPIN_VARS", "f8", ("one",))[:] = [1]
        db.createVariable("PARS", "f8", ("pars",))[:] = [
            bmin, bmax, NOCC, NOCC + 1, 0, 0, 0, 0, int(ordered)]
        a = np.zeros((1, NK, n1, n2, 3, 2))
        for k in range(NK):
            for b1 in range(n1):
                for b2 in range(n2):
                    vb = bmin + b1
                    cb = (NOCC + 1 + b2) if ordered else (bmin + b2)
                    for i in range(3):
                        z = planted(vb, cb, i, k)
                        a[0, k, b1, b2, i] = [z.real, z.imag]
        db.createVariable("DIP_iR", "f8",
                          ("sp", "nk", "n1", "n2", "x", "c"))[:] = a


@pytest.mark.parametrize("bmin, ordered", [(7, False), (1, False), (7, True)])
def test_reader_returns_the_bse_window(tmp_path, bmin, ordered):
    """DipBandsAll from band 7 is the case yambopy's own reader rejects as
    'open-shell'; every returned element must be the one planted there."""
    path = tmp_path / "ndb.dipoles"
    write(path, bmin, 14, ordered)
    d = read_cv_dipoles(lattice(), str(path), [7, 12])
    assert d.shape == (NK, 3, 3, 3)
    want = np.array([[[[planted(7 + v, NOCC + 1 + c, i, k) for v in range(3)]
                       for c in range(3)] for i in range(3)] for k in range(NK)])
    assert np.array_equal(d, want)
