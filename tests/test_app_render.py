"""The standalone app's renderer, offscreen: every effect renders and is distinct, outside the box is the untouched camera,
the box is invisible while closed, effects react to hand speed."""
import numpy as np
import pytest

moderngl = pytest.importorskip("moderngl")
from app.renderer import Renderer, EFFECT_SRC       # noqa: E402
from handcore import config as hcfg                  # noqa: E402
from handcore.view import ViewState, ALL_EFFECTS     # noqa: E402

G = hcfg.load().gesture
W, H = 640, 360
C = [107, 280, 117, 117, 535, 107, 525, 270, 102, 102, 542, 95]     # thumb0, anchor0, anchor1, thumb1, lid0, lid1 (px, top-left origin)


@pytest.fixture(scope="module")
def rig(gl):
    R = Renderer(gl, W, H, 4.0)
    rng = np.random.default_rng(1)
    cam = (rng.random((H, W, 3)) * 120 + 60).astype(np.uint8)
    R.set_camera(cam)
    yy, xx = np.mgrid[0:144, 0:256]
    R.set_mask(((xx - 128) ** 2 / 50 ** 2 + (yy - 90) ** 2 / 60 ** 2 < 1).astype(np.float32))
    return R, cam, gl.simple_framebuffer((W, H), 3)


def render(rig, name, openness=1.0, speed=0.0, t=1.3):
    R, cam, fbo = rig
    v = ViewState(G)
    twist = 35.0                                                              # = the ViewState resting twist, so the ratchet does not step
    v.dial.force(name)
    pose = {"openness": openness, "twist": twist, "twists": (twist, twist), "speed": speed, "motion": (1.0, 0.0) if speed else (0.0, 0.0)}
    for _ in range(60):
        vw = v.update(C, 1.0, 1 / 30, pose, W, H)
    R.hist_filled = False; R._mosh_frame = -10
    for k in range(12):                                                         # a moving camera: slit scan history and datamosh need motion
        R.set_camera(np.roll(cam, 6 * (k - 11), axis=1))
        R.draw(fbo, (W, H), vw, t - (11 - k) / 30.0, show_hud=False)
    return np.frombuffer(fbo.read(components=3), np.uint8).reshape(H, W, 3)[::-1].astype(float), cam.astype(float)


@pytest.mark.parametrize("name", list(G.dial_effects))
def test_every_dial_effect_renders_and_leaves_the_outside_untouched(rig, name):
    img, cam = render(rig, name)
    assert np.isfinite(img).all()
    assert np.abs(img[330:355, 5:60] - cam[330:355, 5:60]).max() < 2          # far from the box: pure camera
    inside = img[170:230, 200:420]
    assert np.abs(inside - cam[170:230, 200:420]).mean() > 6                  # something other than the plain room is drawn in the box


def test_effects_are_visibly_different_from_each_other(rig):
    imgs = {n: render(rig, n)[0][150:260, 150:450] for n in G.dial_effects}
    names = list(imgs)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert np.abs(imgs[a] - imgs[b]).mean() > 4.0, (a, b)


def test_every_registered_effect_shader_compiles(rig):
    R, _, _ = rig
    assert set(EFFECT_SRC) <= set(ALL_EFFECTS) and all(n in R.progs for n in EFFECT_SRC)


def test_portal_is_invisible_while_closed_and_opens_with_the_hand(rig):
    closed, cam = render(rig, "moire", openness=0.0)
    assert np.abs(closed - cam).max() < 2                                      # pinched hands: no box at all
    opened, _ = render(rig, "moire", openness=1.0)
    assert np.abs(opened - cam).mean() > 3


@pytest.mark.parametrize("name", ["glass", "moire", "pixelsort", "crt", "slitscan"])
def test_effects_react_to_hand_speed(rig, name):
    calm, _ = render(rig, name, speed=0.0)
    fast, _ = render(rig, name, speed=0.9)
    assert np.abs(calm - fast)[150:260, 150:450].mean() > 3.0, name
