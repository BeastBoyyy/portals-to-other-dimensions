"""Crop tracker logic (fake network, no model files) + the shared-memory frame hand-off between the tracker and render processes."""
import numpy as np
from app.roi_tracker import RoiTracker
from app.tracker_proc import FrameShare


class FakeNet:
    """Returns a fixed hand in the crop: 'landmarks' on a 21-point ring so every ROI maps back to a hand around the ROI centre."""
    size, name = 224, "fake"

    def run(self, crops):
        a = np.linspace(0, 2 * np.pi, 21, endpoint=False)
        lm = np.column_stack([112 + 40 * np.cos(a), 112 + 40 * np.sin(a)])
        return [{"lm": lm, "z": np.zeros(21), "conf": 0.9, "world": np.zeros((21, 3))} for _ in crops]


def hand_at(cx, cy, r=60):
    a = np.linspace(0, 2 * np.pi, 21, endpoint=False)
    return np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])


def test_seed_ignores_the_same_hand_and_caps_at_two():
    t = RoiTracker(FakeNet())
    assert t.seed(hand_at(300, 300)) and not t.seed(hand_at(305, 302)) and t.seed(hand_at(800, 300)) and not t.seed(hand_at(500, 500))
    assert len(t.tracks) == 2


def test_step_follows_and_drops_after_patience():
    t = RoiTracker(FakeNet(), patience=2)
    t.seed(hand_at(300, 300))
    img = np.zeros((720, 1280, 3), np.uint8)
    out = t.step(img)
    assert len(out) == 1 and np.isfinite(out[0].lm_px).all()
    t.net.run = lambda crops: [{"lm": np.zeros((21, 2)), "z": np.zeros(21), "conf": 0.0, "world": np.zeros((21, 3))} for _ in crops]
    assert t.step(img) == [] and len(t.tracks) == 1            # one bad frame: kept, nothing emitted
    t.step(img)
    assert t.tracks == []                                      # second one: dropped


def test_frame_share_roundtrip():
    w = FrameShare(64, 36)
    r = FrameShare(64, 36, w.name)
    assert r.read(0) is None
    img = (np.random.default_rng(0).random((36, 64, 3)) * 255).astype(np.uint8)
    w.write(img, 1.5)
    f, seq, t = r.read(0)
    assert seq == 1 and t == 1.5 and (f == img).all() and r.read(seq) is None
    r.close(); w.close(unlink=True)
