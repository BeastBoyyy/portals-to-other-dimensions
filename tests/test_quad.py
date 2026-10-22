import numpy as np
from handcore.config import Quad
from handcore import quad


def test_unit_square_valid_any_input_order():
    sq = np.array([[100, 100], [400, 100], [400, 400], [100, 400]], float)
    for perm in ([0, 1, 2, 3], [0, 2, 1, 3], [3, 1, 0, 2]):
        order, ok, _ = quad.build(sq[perm], 1280, 720, Quad())
        assert ok


def test_never_returns_a_bowtie():
    """Property: whenever build() says ok, the drawn polygon is convex (no twist), for random points."""
    rng = np.random.default_rng(1)
    for _ in range(3000):
        p = rng.uniform(0, 1, (4, 2)) * [1280, 720]
        order, ok, _ = quad.build(p, 1280, 720, Quad())
        if ok:
            q = p[list(order)]
            e = np.roll(q, -1, 0) - q
            cr = e[:, 0] * np.roll(e, -1, 0)[:, 1] - e[:, 1] * np.roll(e, -1, 0)[:, 0]
            assert np.all(cr > 0) or np.all(cr < 0)


def test_degenerate_rejected():
    p = np.array([[10, 10], [11, 10], [11, 11], [10, 11]], float)
    assert not quad.build(p, 1280, 720, Quad())[1]
    line = np.array([[0, 0], [300, 0], [600, 0], [900, 0]], float)
    assert not quad.build(line, 1280, 720, Quad())[1]
