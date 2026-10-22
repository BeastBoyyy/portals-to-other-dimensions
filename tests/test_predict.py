import math
import numpy as np
from handcore.config import Predict
from handcore.predict import Presenter, draw_order

C = [100, 100, 300, 100, 300, 300, 100, 300]


def feed(p, f, t_end, delay=0.03, hz=30, until=None):
    outs, ai, ts = [], 0, np.arange(0, t_end, 1 / hz)
    for tk in np.arange(0, (until or t_end), 1 / 60):
        while ai < len(ts) and ts[ai] + delay <= tk:
            c, v = f(ts[ai])
            p.push(ts[ai] + delay, ts[ai], True, c, v); ai += 1
        outs.append((tk,) + p.render(tk))
    return outs


def test_moving_target_is_tracked_and_continuous():
    p = Presenter(Predict(), 1 / 30)
    f = lambda t: ([x + 200 * t for x in C], [200.0] * 8)
    outs = feed(p, f, 4)
    err = [abs(c[0] - (100 + 200 * t)) for t, c, a in outs if t > 1]
    assert max(err) < 12                                       # latency (30 ms * 200 px/s = 6 px) is compensated
    steps = np.abs(np.diff([c[0] for t, c, a in outs if t > 1]))
    assert steps.max() < 200 / 60 * 1.6                        # no pops


def test_stalled_stream_does_not_fly_away_and_fades():
    p = Presenter(Predict(), 1 / 30)
    f = lambda t: ([x + 1000 * t for x in C], [1000.0] * 8)
    outs = feed(p, f, 2.0, until=5.0)                          # samples stop at 2 s, rendering continues to 5 s
    last_ok = [c[0] for t, c, a in outs if 2.0 < t < 2.1][-1]
    end = outs[-1]
    assert abs(end[1][0] - last_ok) < 120                      # bounded extrapolation
    assert end[2] == 0.0                                       # faded out after the stall


def test_invalid_sample_fades_out_over_about_200ms_then_back_in():
    p = Presenter(Predict(), 1 / 30)
    for i in range(30):
        p.push(i / 30, i / 30, True, C, [0] * 8); p.render(i / 30)
    p.push(1.0, 1.0, False, C, [0] * 8)
    a = [p.render(1.0 + k / 60)[1] for k in range(1, 20)]
    assert a[0] < 1.0 and a[-1] == 0.0 and all(x >= y for x, y in zip(a, a[1:]))
    assert 8 <= sum(1 for x in a if x > 0) <= 14               # ~0.2 s at 60 Hz
    p.push(1.4, 1.4, True, C, [0] * 8)
    assert p.render(1.42)[1] > 0


def test_duplicate_timestamp_dropped():
    p = Presenter(Predict(), 1 / 30)
    p.push(1.0, 1.0, True, C, [0] * 8)
    p.push(1.05, 1.0, True, [x + 500 for x in C], [0] * 8)     # same capture stamp: ignored
    assert abs(p.render(1.06)[0][0] - 100) < 1e-6


def test_draw_order_is_convex_winding():
    c = [0, 0, 100, 100, 100, 0, 0, 100]                       # scrambled
    o = draw_order(c)
    assert sorted(o) == [0, 1, 2, 3]
