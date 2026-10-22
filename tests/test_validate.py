"""Gate 2 against REAL MediaPipe output (fixtures/real_images_landmarks.jsonl, made by tools/extract_landmarks.py)."""
import json, pathlib
import numpy as np
from handcore.config import Gates as _G


def Gates():
    return _G(geometry=True, self_intersect=True)       # the strict gates (OFF in the live config: they rejected real twisted hands)
from handcore.types import Hand
from handcore.validate import check_hand

FIX = pathlib.Path(__file__).resolve().parent.parent / "fixtures" / "real_images_landmarks.jsonl"


def real_hands():
    for line in FIX.read_text().splitlines():
        r = json.loads(line)
        for h in r["hands"]:
            yield r, Hand(np.array(h["lm"]), np.array(h["world"]), h["label"], h["score"])


def test_real_hands_pass_geometry_gate():
    n = 0
    for r, h in real_hands():
        ok, why = check_hand(h, r["w"], r["h"], Gates())
        assert ok, (r["src"], why)
        n += 1
    assert n >= 5


def test_non_hand_images_have_no_detections():
    for line in FIX.read_text().splitlines():
        r = json.loads(line)
        if r["src"] in ("burger.jpg", "cat.jpg", "portrait.jpg"):
            assert r["hands"] == []


def test_garbage_skeletons_rejected():
    rng = np.random.default_rng(0)
    base = next(real_hands())
    rejected = 0
    for _ in range(200):
        lm = base[1].lm * rng.uniform(0.3, 2.0, (21, 3))
        w = base[1].world * rng.uniform(0.3, 2.0, (21, 3))
        ok, _ = check_hand(Hand(lm, w, "Left", 0.95), base[0]["w"], base[0]["h"], Gates())
        rejected += (not ok)
    assert rejected > 180


def test_hand_at_frame_edge_and_low_score_rejected():
    r, h = next(real_hands())
    lm = h.lm.copy(); lm[:, 0] += 0.6                       # half the hand outside the frame
    assert check_hand(Hand(lm, h.world, "Left", 0.95), r["w"], r["h"], Gates())[1] in ("edge", "tip_outside")
    assert check_hand(Hand(h.lm, h.world, "Left", 0.3), r["w"], r["h"], Gates())[1] == "score"
