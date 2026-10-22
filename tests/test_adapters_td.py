"""Things around handcore that CAN be verified without TouchDesigner: wire protocol, self-check maths, script syntax, shaders."""
import importlib.util
import json
import pathlib
import py_compile
import struct

import numpy as np
import pytest

from adapters import protocol as P

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_state_packet_roundtrip_and_rejects_foreign_datagrams():
    c = list(np.arange(8) * 10.5)
    v = list(np.arange(8) * -3.25)
    b = P.pack_state(7, 12.345678, True, False, 2, 0.93, c, v, infer_ms=14.2, tracker_fps=29.7, dropped=3, cpu_us=620,
                     lid=[1, 2, 3, 4], lid_vel=[5, 6, 7, 8], pose=(0.5, 61.0, 50.0, 72.0, 0.25, 0.6, -0.8))
    s = P.unpack_state(b)
    assert s["seq"] == 7 and s["valid"] and not s["held"] and s["n_hands"] == 2 and s["dropped"] == 3
    assert s["lid"] == [1, 2, 3, 4] and s["lid_vel"] == [5, 6, 7, 8]
    assert np.allclose(s["pose"], [0.5, 61.0, 50.0, 72.0, 0.25, 0.6, -0.8], atol=1e-5)
    assert abs(s["t"] - 12.345678) < 1e-9 and np.allclose(s["corners"], c, atol=1e-3) and np.allclose(s["vel"], v, atol=1e-3)
    assert P.unpack_state(b[:-1]) is None and P.unpack_state(b"x" * len(b)) is None


def test_shared_memory_headers_have_documented_sizes():
    assert P.FRAME_HDR.size == 24 and P.MASK_HDR.size == 12 and P.STATE.size == struct.calcsize(P.STATE_FMT)


def test_frame_shm_protocol_end_to_end():
    from multiprocessing import shared_memory
    w, h = 64, 36
    shm = shared_memory.SharedMemory(create=True, size=P.FRAME_HDR.size + w * h * 3)
    try:
        img = (np.random.default_rng(0).integers(0, 255, (h, w, 3))).astype(np.uint8)
        np.frombuffer(shm.buf, np.uint8, w * h * 3, P.FRAME_HDR.size)[:] = img.ravel()
        P.FRAME_HDR.pack_into(shm.buf, 0, P.FRAME_MAGIC, 5, 3.25, w, h, 0)
        magic, seq, t, w2, h2, _ = P.FRAME_HDR.unpack_from(shm.buf, 0)
        got = np.frombuffer(shm.buf, np.uint8, w2 * h2 * 3, P.FRAME_HDR.size).reshape(h2, w2, 3)
        assert magic == P.FRAME_MAGIC and seq == 5 and t == 3.25 and np.array_equal(got, img)
        del got
    finally:
        shm.close(); shm.unlink()


def _load_selfcheck():
    spec = importlib.util.spec_from_file_location("selfcheck", ROOT / "td" / "selfcheck.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_selfcheck_hp_rms_separates_drift_from_jitter():
    sc = _load_selfcheck()
    t = np.arange(0, 5, 1 / 60)
    drift = 20 * np.sin(2 * np.pi * 0.5 * t)                                     # slow real motion: below 3 Hz
    noise = np.random.default_rng(0).normal(0, 0.3, len(t))
    assert sc.hp_rms(drift, 60) < 0.05
    assert 0.2 < sc.hp_rms(drift + noise, 60) < 0.45                              # recovers ~0.3 px of jitter
    assert sc.hp_rms(noise[:10], 60) != sc.hp_rms(noise[:10], 60)                 # too short -> NaN


def test_selfcheck_frame_stats_and_kink():
    sc = _load_selfcheck()
    dts = [1 / 60] * 980 + [0.030] * 20
    s = sc.frame_stats(dts)
    assert s["n_late"] == 20 and abs(s["late_pct"] - 2.0) < 1e-9 and s["avg_fps"] > 58 and s["p1_fps"] < 40
    smooth = np.cumsum(np.ones((50, 8)) * 5, axis=0)
    assert sc.max_kink(smooth) < 1e-9
    jump = smooth.copy(); jump[25:] += 100
    assert sc.max_kink(jump) > 90


@pytest.mark.parametrize("path", sorted(str(p.relative_to(ROOT)) for p in list((ROOT / "td").rglob("*.py")) + list((ROOT / "adapters").glob("*.py"))
                                        + list((ROOT / "tools").glob("*.py"))))
def test_every_script_compiles(path, tmp_path):
    py_compile.compile(str(ROOT / path), cfile=str(tmp_path / "x.pyc"), doraise=True)


def test_td_config_json_is_current():
    import dataclasses
    from handcore import config as hcfg
    p = ROOT / "td" / "config.json"
    assert p.exists(), "run: python tools/export_td_config.py"
    j = json.loads(p.read_text())
    cfg = dataclasses.asdict(hcfg.load(str(ROOT / "config.yaml")))
    for k, v in cfg.items():
        assert j[k] == json.loads(json.dumps(v)), f"td/config.json is stale in section {k}: re-run tools/export_td_config.py"


def test_shaders_compile_and_render_with_td_preamble(gl):
    pytest.importorskip("moderngl")
    from tools import shader_preview as sp
    W, H = 320, 180
    rng = np.random.default_rng(0)
    cam = np.dstack([rng.random((H, W, 3), dtype="f4"), np.ones((H, W), "f4")])
    mask = np.dstack([np.full((H, W, 3), 0.8, "f4"), np.ones((H, W), "f4")])
    a = sp.render("scene_a.glsl", W, H, [cam, mask], {"uA": (1, 4, 6, 1), "uB": (0.9, 2.5, 1.3, 0)}, ctx=gl)
    b = sp.render("scene_b.glsl", W, H, [cam], {"uA": (1, 0.42, 0.15, 0), "uB": (0, 1, 0, 0)}, ctx=gl)
    bl = sp.render("scene_blend.glsl", W, H, [a, b], {"uMix": (0.5, 0, 0, 0)}, ctx=gl)
    quad = (0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75, 0.25)
    for qs in (quad, tuple(reversed(quad))):                                   # either winding must work
        c = sp.render("composite.glsl", W, H, [cam, bl, bl], {"uQA": qs[:4], "uQB": qs[4:], "uP1": (1, 0.6, 0, 1), "uP2": (0, 0, 0.3, 2)}, ctx=gl)
        inside = c[H // 2, W // 2, :3]
        outside = c[5, 5, :3]
        assert np.allclose(outside, cam[5, 5, :3], atol=1e-3)                 # outside the portal: untouched camera
        assert not np.allclose(inside, cam[H // 2, W // 2, :3], atol=1e-2)     # inside: the other dimension
    assert b.max() > 0.5 and b[0, 0, :3].max() == 0.0                         # asteroid lit, black background
    assert (a[..., :3].max(axis=2) > 0.2).mean() > 0.02                       # particles exist


def test_plugin_bridge_maps_fabricated_channels_and_drops_partial_hands():
    from adapters.td_plugin_bridge import frame_from_channels, LANDMARK_NAMES
    vals = {}
    for h in (1, 2):
        for i, n in enumerate(LANDMARK_NAMES):
            for ax, v in zip("xyz", (0.1 * h + 0.01 * i, 0.5 + 0.005 * i, 0.0)):
                vals[f"h{h}:{n}:{ax}"] = v
    del vals["h2:pinky_tip:y"]                                   # hand 2 is incomplete -> must be dropped, not guessed
    fr = frame_from_channels(vals, 1.5)
    assert fr.t == 1.5 and len(fr.hands) == 1 and fr.hands[0].gt_id == 1
    assert abs(fr.hands[0].lm[4, 0] - (0.1 + 0.04)) < 1e-9 and abs(fr.hands[0].lm[8, 1] - (0.5 + 0.04)) < 1e-9


def test_new_effect_shaders_compile_and_behave(gl):
    pytest.importorskip("moderngl")
    from tools import shader_preview as sp
    W, H = 320, 180
    cam = np.dstack([np.full((H, W, 3), 0.5, "f4"), np.ones((H, W), "f4")])
    mask = np.dstack([np.zeros((H, W, 3), "f4"), np.ones((H, W), "f4")])
    moon = sp.render("scene_moon.glsl", W, H, [mask], {"uM": (0.5, 0.5, 0.2, 0.0), "uB": (0.55, 0.95, 0, 0)}, ctx=gl)
    assert moon[H // 2, W // 2, :3].min() > 0.3 and moon[2, 2, :3].max() < 0.02          # a lit disc on black
    mask[H // 2 - 20:H // 2 + 20, W // 2 - 20:W // 2 + 20, :3] = 1.0                     # person in front of the moon hides it
    hidden = sp.render("scene_moon.glsl", W, H, [mask], {"uM": (0.5, 0.5, 0.2, 0.0), "uB": (0.55, 0.95, 0, 0)}, ctx=gl)
    assert hidden[H // 2, W // 2, :3].max() < 0.08
    ghost = sp.render("scene_ghost.glsl", W, H, [cam, mask], {"uA": (0.0, 0.05, 0.9, 0.0)}, ctx=gl)
    assert ghost.max() > 0.3 and ghost[2, 2, :3].max() < 0.02                            # contour lines only
    a = np.dstack([np.ones((H, W, 3), "f4"), np.ones((H, W), "f4")]); b = np.dstack([np.zeros((H, W, 3), "f4"), np.ones((H, W), "f4")])
    mix = sp.render("scene_mix4.glsl", W, H, [a, b, b, b], {"uW": (0.25, 0.5, 0.0, 0.25)}, ctx=gl)
    assert abs(mix[10, 10, 0] - 0.25) < 0.01
