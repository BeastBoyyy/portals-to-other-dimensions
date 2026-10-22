"""Whole-hand pose: openness (pinch <-> open), twist dial, box always on thumb/index/middle/ring, opening animation, 3D twist of the box."""
import numpy as np
import pytest
from handcore import config as hcfg
from handcore.pipeline import HandCore
from handcore.pose import hand_openness, hand_twist_deg, smoothstep
from handcore.types import Frame, Hand
from handcore.validate import palm_size_px, to_px
from handcore.view import ViewState, TwistDial, ALL_EFFECTS
from tools import synth

CFG = hcfg.with_overrides(hcfg.load(), {"gates": {"person_mask": False}})
G = CFG.gesture


def _hand(curl, thumb_alpha=(60.0, 75.0, 80.0), yaw=0.0, x=0.0, splay=0.0):
    local = synth.hand_local(1, curl=curl, thumb_alpha=thumb_alpha, splay=splay)
    lm, world, _ = synth.project(local, (x, -0.15, 0.70, 0, yaw, 0))
    return lm, world


def test_openness_pinch_vs_open():
    open_lm, _ = _hand((0, 0, 0, 0), thumb_alpha=(60, 75, 80), splay=1.6)       # fingers fanned out like a real open hand
    pinch_lm, _ = _hand((1, 1, 1, 1), thumb_alpha=(25, 20, 15))          # fingers curled in, thumb folded over: a claw
    vals = {}
    for name, lm in (("open", open_lm), ("pinch", pinch_lm)):
        px = to_px(lm, 1280, 720)
        vals[name] = hand_openness(px, palm_size_px(px))
    assert vals["open"] > 0.7 and vals["pinch"] < 0.55 and vals["open"] > 1.8 * vals["pinch"], vals


@pytest.mark.parametrize("yaw_deg", [0, 30, 60, 85])
def test_twist_measures_the_palm_angle_to_the_camera(yaw_deg):
    _, world = _hand((0, 0, 0, 0), yaw=np.radians(yaw_deg))
    assert abs(hand_twist_deg(world) - yaw_deg) < 3.0


def _dial():
    return TwistDial(G.dial_effects, G.dial_step_deg, G.dial_commit_s, G.dial_base_tau_s, G.dial_ease_s)


def _run_dial(d, twists, dt=1 / 30, active=True):
    for tw in twists:
        d.update(float(tw), dt, active)
    return d


@pytest.mark.parametrize("rest", [30.0, 65.0, 80.0])
def test_dial_ignores_noise_around_any_resting_twist(rest):
    """Wherever the hands rest (the recorded user rests near 65 deg) +-8 deg of estimator noise for 30 s never steps the effect."""
    rng = np.random.default_rng(0)
    d = _run_dial(_dial(), rest + rng.normal(0, 4.0, 900))                 # sigma 4 deg ~ +-8 deg peaks, smoothed like the app does
    assert d.k == 0 and d.w[G.dial_effects[0]] > 0.9


def test_twist_ratchet_steps_forward_and_back_and_wraps():
    d = _run_dial(_dial(), [60.0] * 30)
    _run_dial(d, [60.0 + 24.0] * 12)                                       # twist 24 deg past rest, hold 0.4 s
    assert G.dial_effects[d.k] == G.dial_effects[1]
    _run_dial(d, [84.0] * 10)                                              # re-based: staying put does not step again
    assert G.dial_effects[d.k] == G.dial_effects[1]
    _run_dial(d, [84.0 + 24.0] * 12)                                       # twist forward again
    assert G.dial_effects[d.k] == G.dial_effects[2]
    _run_dial(d, [84.0] * 12)                                              # twist back
    assert G.dial_effects[d.k] == G.dial_effects[1]
    d = _run_dial(_dial(), [60.0] * 30 + [36.0] * 12)                      # backwards from the first effect wraps to the last
    assert G.dial_effects[d.k] == G.dial_effects[-1]


def test_partial_twist_previews_the_next_effect_and_a_brief_spike_does_not_commit():
    d = _run_dial(_dial(), [60.0] * 30 + [60.0 + 12.0] * 15)
    assert d.k == 0 and 0.1 < d.w[G.dial_effects[1]] < 0.9 and d.w[G.dial_effects[0]] > 0.1                  # half-way: both visible
    d = _run_dial(_dial(), [60.0] * 30 + [95.0] * 3 + [60.0] * 20)                           # 0.1 s spike
    assert d.k == 0 and d.w[G.dial_effects[1]] < 0.05


def test_slow_drift_of_the_resting_pose_is_followed_not_stepped():
    d = _run_dial(_dial(), np.linspace(60.0, 82.0, 30 * 40))               # 22 deg of drift over 40 s
    assert d.k == 0


def test_dial_does_not_move_while_the_portal_is_closed():
    d = _run_dial(_dial(), [60.0] * 30)
    _run_dial(d, [100.0] * 30, active=False)
    assert d.k == 0


def test_effect_weights_change_smoothly():
    d = _run_dial(_dial(), [60.0] * 30)
    prev = dict(d.w)
    for tw in [84.0] * 30:
        cur = dict(d.update(tw, 1 / 30, True))
        assert max(abs(cur[k] - prev[k]) for k in ALL_EFFECTS) < 0.35                    # ~0.25 s ease, never a cut
        prev = cur
    assert d.w[G.dial_effects[1]] > 0.95


def _frames(curl, n=60, spread=0.20, yaw=0.0, thumb_alpha=(60.0, 75.0, 80.0), splay=0.0):
    out = []
    for i in range(n):
        hands = []
        for hid, sx in ((0, -spread), (1, spread)):
            local = synth.hand_local(1 if hid == 0 else -1, curl=curl, thumb_alpha=thumb_alpha, splay=splay)
            lm, world, _ = synth.project(local, (sx, -0.15, 0.70, 0, yaw, 0))
            hands.append(Hand(lm, world, "Left" if hid == 0 else "Right", 0.97, 1.0, hid))
        out.append(Frame(i / 30.0, hands))
    return out


def _run(frames):
    core = HandCore(CFG)
    st = None
    for f in frames:
        st = core.update(f)
    return st


def test_pipeline_openness_open_hand_vs_pinch():
    assert _run(_frames((0, 0, 0, 0), splay=1.6)).openness > 0.95
    assert _run(_frames((1, 1, 1, 1), thumb_alpha=(25, 20, 15))).openness < 0.2


def test_pipeline_twist_follows_hand_rotation():
    for yaw in (0.0, 0.8):
        st = _run(_frames((0, 0, 0, 0), yaw=yaw))
        assert abs(st.twist - np.degrees(yaw)) < 6.0, (yaw, st.twist)


@pytest.mark.parametrize("curl", [(0, 0, 0, 0), (1, 0, 0, 1), (0, 1, 1, 1), (1, 1, 1, 1)])
def test_box_stays_attached_to_the_same_fingertips_whatever_fingers_are_raised(curl):
    """Raising / curling fingers must NOT move the anchors: corners = thumb tips + INDEX tips, lid = mean(middle, ring)."""
    st = _run(_frames(curl))
    for hid, slot in ((1, 0), (0, 1)):                                    # mirror: raw hand 1 is the display-left slot 0
        local = synth.hand_local(1 if hid == 0 else -1, curl=curl)
        _, _, px = synth.project(local, ((-0.20 if hid == 0 else 0.20), -0.15, 0.70, 0, 0, 0))
        px = px.copy(); px[:, 0] = 1280 - px[:, 0]
        assert np.linalg.norm(st.corners[1 if slot == 0 else 2] - px[8]) < 8.0
        assert np.linalg.norm(st.corners[0 if slot == 0 else 3] - px[4]) < 8.0
        assert np.linalg.norm(st.lid[slot] - (px[12] + px[16]) / 2) < 8.0


def test_small_closed_boxes_are_valid_so_the_portal_can_grow_out_of_a_pinch():
    st = _run(_frames((1, 1, 1, 1), thumb_alpha=(25, 20, 15)))
    assert st.valid


def _pose(o=1.0, tw=30.0, tws=None, speed=0.0, motion=(0.0, 0.0)):
    return {"openness": o, "twist": tw, "twists": tws or (tw, tw), "speed": speed, "motion": motion}


C = [215, 560, 235, 235, 1070, 215, 1050, 540, 205, 205, 1085, 190]


def test_view_opens_with_the_hand_and_closes_on_pinch():
    v = ViewState(G)
    for _ in range(30):
        vw = v.update(C, 1.0, 1 / 30, _pose(o=0.0), 1280, 720)
    assert vw["open"] < 0.01 and vw["alpha"] < 0.01
    seq = [v.update(C, 1.0, 1 / 30, _pose(o=1.0), 1280, 720)["open"] for _ in range(12)]
    assert seq[0] < 0.4 < 0.99 < seq[-1] and all(b >= a for a, b in zip(seq, seq[1:]))      # smooth opening in ~0.2 s
    seq = [v.update(C, 1.0, 1 / 30, _pose(o=0.0), 1280, 720)["open"] for _ in range(12)]
    assert seq[-1] < 0.01 and all(b <= a for a, b in zip(seq, seq[1:]))                     # and a smooth close


def test_twisting_one_hand_pushes_its_side_of_the_box_deeper():
    v = ViewState(G)
    for _ in range(40):
        flat = v.update(C, 1.0, 1 / 30, _pose(tw=30.0, tws=(30.0, 30.0)), 1280, 720)
    for _ in range(12):                                             # one hand twists 40 deg past its resting twist (a brief twist, no step)
        tw = v.update(C, 1.0, 1 / 30, _pose(tw=35.0, tws=(70.0, 30.0)), 1280, 720)
    from handcore.predict import draw_order
    order = draw_order(C[:8])
    ins = dict(zip(order, tw["insets"]))
    ref = dict(zip(order, flat["insets"]))
    assert ins[0] < ref[0] - 0.05 and ins[1] < ref[1] - 0.05                 # hand 0 (ids 0,1) twisted: its side recedes
    assert abs(ins[2] - ref[2]) < 0.02 and abs(ins[3] - ref[3]) < 0.02       # the other hand is untouched
