"""The dipole check: agreement means a constant ratio, not only the same ranking."""
from xphd.dipoles import residual_verdict


def test_constant_ratio_agrees():
    v, spread = residual_verdict([3.0, 2.0, 1.0], [3e-7, 2e-7, 1e-7])
    assert v == "AGREE" and abs(spread - 1) < 1e-9


def test_same_ranking_but_varying_ratio_disagrees():
    # the hBN case: ratios 63, 139, 247 with the same three manifolds brightest
    mine = [3.1404e-06, 1.8491e-06, 8.0320e-07]
    ref = [5.0026e-08, 1.3298e-08, 3.2489e-09]
    v, spread = residual_verdict(mine, ref)
    assert v.startswith("DISAGREE") and "NOT proportional" in v
    assert 3.9 < spread < 4.0


def test_different_ranking_disagrees():
    v, _ = residual_verdict([1.0, 2.0, 3.0], [3.0, 2.0, 1.0], k=1)
    assert v.startswith("DISAGREE -- different")
