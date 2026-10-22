"""handcore end-to-end on synthetic clips with exact ground truth (see tools/synth.py for the noise model)."""
import copy
import time
import numpy as np
import pytest
from handcore import config as hcfg
from handcore import quad as Q
from handcore.pipeline import HandCore
from handcore.types import Frame
from tools import synth, evaluate as ev

CFG = hcfg.load()


def run(clip, cfg=CFG):
    core = HandCore(cfg)
    return [core.update(f) for f in clip.frames]


def errs(clip, states, mirror=True):
    out = []
    for s in states:
        g = clip.gt_corners(s.t, mirror)
        if g is not None and s.valid:
            out.append(np.hypot(*(s.corners - g).T).max())
    return np.array(out)


@pytest.mark.parametrize("mirror", [True, False])
def test_still_hands_accurate_in_both_mirror_modes(mirror):
    cfg = hcfg.with_overrides(CFG, {"camera": {"mirror": mirror}})
    clip = synth.make_clip("still", seed=0, dur=10)
    e = errs(clip, run(clip, cfg), mirror)
    assert len(e) > 250 and np.mean(e < 6) > 0.97 and np.mean(e > 25) < 0.02


def test_mirroring_is_decided_in_one_place():
    clip = synth.make_clip("still", seed=0, dur=2, noise=synth.Noise(miss_p=0, teleport_p=0, thumb_snap_p=0))
    f = clip.frames[5]
    raw_x = float(np.mean([f.hands[0].lm[[0, 5, 9, 13, 17], 0].mean(), f.hands[1].lm[[0, 5, 9, 13, 17], 0].mean()]))
    for mirror in (True, False):
        core = HandCore(hcfg.with_overrides(CFG, {"camera": {"mirror": mirror}}))
        dets = core.ingest(f)
        mean_x = np.mean([d.centre[0] for d in dets]) / 1280
        assert abs(mean_x - ((1 - raw_x) if mirror else raw_x)) < 1e-6


def test_no_hands_clip_with_ghosts_gives_no_portal():
    """With the (optional) person-mask gate in strict mode, poster / flicker / random ghost hands never make a portal."""
    strict = hcfg.with_overrides(CFG, {"gates": {"person_mask": True, "mask_acquire_only": False}})
    for gh in (["poster", "flicker", "random"], ["poster2", "flicker", "random"]):
        clip = synth.make_clip("no_hands", seed=1, dur=10, ghosts=gh)
        assert not any(s.valid for s in run(clip, strict))


def test_poster_hands_make_a_portal_without_person_mask_gate():
    """Documents WHY gate 5 exists: two persistent hands on a poster pass gates 1-4."""
    cfg = hcfg.with_overrides(CFG, {"gates": {"person_mask": False}})
    clip = synth.make_clip("no_hands", seed=1, dur=10, ghosts=["poster2"])
    assert sum(s.valid for s in run(clip, cfg)) > 100


def test_one_hand_never_gives_portal():
    clip = synth.make_clip("one_hand", seed=0, dur=10)
    assert not any(s.valid for s in run(clip))


def test_flipping_handedness_label_never_swaps_slots():
    noise = synth.Noise(label_flip_p=0.5)
    for kind in ("crossing", "near_cross", "slow"):
        for seed in range(3):
            clip = synth.make_clip(kind, seed=seed, dur=20, noise=noise)
            assert ev.count_swaps(clip, ev.run_clip(clip, CFG), True) == 0, (kind, seed)


def test_crossing_hands_no_swaps_over_many_seeds():
    total = 0
    for kind in ("crossing", "near_cross"):
        for seed in range(12):
            clip = synth.make_clip(kind, seed=seed, dur=20)
            total += ev.count_swaps(clip, ev.run_clip(clip, CFG), True)
    assert total == 0


def test_random_single_frame_outliers_never_move_output():
    """Property: ONE teleported detection (hand shifted 60..400 px for one frame) moves the output < 25 px."""
    rng = np.random.default_rng(5)
    quiet = synth.Noise(teleport_p=0, thumb_snap_p=0, miss_p=0, frame_drop_p=0)
    clip = synth.make_clip("still", seed=2, dur=12, noise=quiet)
    base = run(clip)
    worst = 0.0
    for _ in range(15):
        k = int(rng.integers(60, len(clip.frames) - 8))
        fr = copy.deepcopy(clip.frames)
        h = fr[k].hands[int(rng.integers(0, 2))]
        ang, d = rng.uniform(0, 6.28), rng.uniform(60, 400)
        h.lm[:, 0] += d * np.cos(ang) / 1280
        h.lm[:, 1] += d * np.sin(ang) / 720
        core = HandCore(CFG)
        st = [core.update(f) for f in fr]
        dev = max(np.hypot(*(a.corners - b.corners).T).max() for a, b in zip(st[k:k + 6], base[k:k + 6]) if a.valid and b.valid)
        worst = max(worst, dev)
    assert worst < 25, worst


def test_output_continuous_across_short_dropout():
    """Hands vanish for 4 frames (inside the lose_m=6 window): bridged by Kalman prediction, no flicker, no jump."""
    clip = synth.make_clip("slow", seed=0, dur=10, noise=synth.Noise(miss_p=0))
    for f in clip.frames[150:154]:
        f.hands = []
    st = run(clip)
    c = np.array([s.corners for s in st if s.valid])
    assert np.abs(np.diff(c, axis=0)).max() < 30
    assert all(s.valid for s in st[150:154])


def test_long_dropout_goes_invalid_then_recovers_fast():
    clip = synth.make_clip("slow", seed=0, dur=10, noise=synth.Noise(miss_p=0))
    for f in clip.frames[150:200]:
        f.hands = []
    st = run(clip)
    assert not any(s.valid for s in st[170:200])    # lose_m=15 frames of prediction bridging first
    after = [s for s in st[200:] if s.valid]
    assert after and after[0].t - clip.frames[200].t < 0.2          # re-acquired within the 3-frame temporal gate


def test_non_monotonic_and_duplicate_timestamps_do_not_corrupt():
    clip = synth.make_clip("slow", seed=0, dur=6)
    fr = clip.frames
    fr.insert(50, Frame(fr[49].t, fr[49].hands))                    # duplicate stamp
    fr.insert(80, Frame(fr[70].t, fr[70].hands))                    # clock went backwards
    core = HandCore(CFG)
    out = [core.update(f) for f in fr]
    assert all(np.isfinite(s.corners).all() for s in out if s is not None)
    assert sum(s.valid for s in out if s is not None) > len(fr) * 0.9


def test_clock_jump_resets_cleanly():
    clip = synth.make_clip("still", seed=0, dur=6)
    for f in clip.frames[90:]:
        f.t += 1000.0
    out = run(clip)
    assert out[-1].valid and np.isfinite(out[-1].corners).all()


def test_hand_partly_outside_frame_is_rejected():
    clip = synth.make_clip("still", seed=0, dur=2, noise=synth.Noise(miss_p=0, teleport_p=0, thumb_snap_p=0))
    f = copy.deepcopy(clip.frames[5])
    f.hands[0].lm[:, 0] -= 0.45                                     # half of hand 0 off the left edge
    core = HandCore(CFG)
    dets = core.ingest(f)
    assert len(dets) == 1 and core.rejects.get("edge", 0) + core.rejects.get("tip_outside", 0) == 1


def test_quad_is_never_twisted_on_extreme_tilt():
    clip = synth.make_clip("tilt", seed=0, dur=20)
    for s in run(clip):
        if s.valid:
            assert Q.validate_quad(s.corners, s.order, 1280, 720, CFG.quad)[0]


def test_cpu_budget_per_camera_frame():
    clip = synth.make_clip("fast", seed=0, dur=10)
    core = HandCore(CFG)
    t0 = time.perf_counter()
    for f in clip.frames:
        core.update(f)
    per = (time.perf_counter() - t0) / len(clip.frames)
    assert per < 4e-3, per                                          # spec: <= 4 ms CPU per frame
