import numpy as np
from handcore.one_euro import OneEuro


def test_converges_to_constant():
    f = OneEuro(2, 1.0, 0.0, 1.0)
    for i in range(300):
        y = f(np.array([100.0, 50.0]), i / 30)
    assert np.allclose(y, [100, 50], atol=1e-3)


def test_duplicate_and_backwards_timestamps_ignored():
    f = OneEuro(1, 1.0, 0.0, 1.0)
    f(np.array([0.0]), 1.0)
    a = f(np.array([10.0]), 1.1)
    assert np.allclose(f(np.array([99.0]), 1.1), a)      # duplicate stamp
    assert np.allclose(f(np.array([99.0]), 0.5), a)      # clock went backwards
    assert np.isfinite(f(np.array([10.0]), 1.13)).all()


def test_timestamp_jitter_makes_no_velocity_spike():
    rng = np.random.default_rng(0)
    f = OneEuro(1, 1.0, 0.0, 5.0)
    t, peak = 0.0, 0.0
    for i in range(600):
        t += 1 / 30 + rng.normal(0, 0.0005)
        f(np.array([200.0]), t)                          # perfectly still signal
        peak = max(peak, abs(f.dx[0]))
    assert peak < 1e-6


def test_long_gap_does_not_blow_up():
    f = OneEuro(1, 1.0, 0.0, 1.0)
    f(np.array([0.0]), 0.0)
    y = f(np.array([100.0]), 50.0)                       # 50 s stall: dt is clamped, result stays in range
    assert 0.0 <= y[0] <= 100.0
