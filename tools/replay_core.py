"""Replay logs/live_frames.jsonl (recorded by the app) through HandCore under different configs and compare how often the portal is tracked.
    python tools/replay_core.py [logs/live_frames.jsonl]"""
import json, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from handcore import config as hcfg                              # noqa: E402
from handcore.pipeline import HandCore                           # noqa: E402
from handcore.types import Frame, Hand                           # noqa: E402

path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "logs" / "live_frames.jsonl"
frames = []
for line in path.read_text().splitlines():
    try:
        r = json.loads(line)
    except Exception:
        continue
    frames.append(Frame(r["t"], [Hand(np.array(h["lm"]), np.array(h["world"]), h["label"], h["score"]) for h in r["hands"]]))
print(f"{len(frames)} frames, {sum(len(f.hands) == 2 for f in frames)} with two detections")
base = hcfg.load(str(ROOT / "config.yaml"))
variants = {"current config": {},
            "previous (geometry gate on, lose 6, score .7, hold 6)": {"gates": {"geometry": True, "lose_m": 6, "min_score": 0.70, "self_intersect": True}, "quad": {"hold_frames": 6}}}
for name, ov in variants.items():
    ov = {**ov, "gates": {"person_mask": False, **ov.get("gates", {})}}
    cfg = hcfg.with_overrides(base, ov)
    core = HandCore(cfg)
    st = [core.update(f) for f in frames]
    st = [s for s in st if s is not None]
    v = np.array([s.valid for s in st]); two = np.array([s.n_hands == 2 for s in st])
    c = np.array([s.corners for s in st])
    j = np.hypot(*np.diff(c, axis=0).transpose(2, 0, 1)).max(1)
    both = v[1:] & v[:-1]
    print(f"{name:30s} portal valid {100 * v.mean():5.1f} %   two hands tracked {100 * two.mean():5.1f} %   "
          f"1-frame corner jumps >40 px: {int(np.sum(j[both] > 40))}   dropouts (valid->invalid): {int(np.sum(v[:-1] & ~v[1:]))}")
