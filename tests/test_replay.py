"""Landmark replay: JSONL -> handcore, and the Section 9 metrics on the synthetic suite (thresholds that hold)."""
import json
from handcore import config as hcfg
from tools import synth, evaluate as ev


def test_jsonl_roundtrip_replay(tmp_path):
    clip = synth.make_clip("still", seed=0, dur=5)
    p = tmp_path / "clip.jsonl"
    with open(p, "w") as f:
        for fr in clip.frames:
            f.write(json.dumps({"t": fr.t, "hands": [{"lm": h.lm.tolist(), "world": h.world.tolist(),
                                                      "label": h.label, "score": h.score} for h in fr.hands]}) + "\n")
    rep = ev.replay_report(str(p), None, hcfg.load())
    assert rep["frames"] == len(clip.frames) and rep["portal_valid_pct"] > 90 and rep["jumps_gt_20px"] <= 6    # synthetic thumb snaps of ~30 px pass the relaxed live gates


def test_measured_metrics_on_synthetic_suite():
    strict = hcfg.with_overrides(hcfg.load(), {"gates": {"geometry": True, "self_intersect": True, "person_mask": True, "mask_acquire_only": False}})
    suite = ev.run_suite(strict, seeds=(0, 1), dur=15)
    s = ev.summarize(suite)
    assert s["false_hand"] <= 0.1                      # incl. poster hands, flicker and the one-hand clip
    assert s["swaps"] == 0
    assert s["lag_frames"] <= 1.0
    assert s["corner_acc_by_kind"]["still"] >= 99.0
    assert s["corner_acc_by_kind"]["slow"] >= 99.0
    assert s["cpu_us"] < 4000
