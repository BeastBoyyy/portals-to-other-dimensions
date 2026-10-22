"""Replay logs/live_frames.jsonl through the per-detection gates and report WHY real hands are rejected (with twist / openness at that moment).
    python tools/replay_gates.py [logs/live_frames.jsonl]"""
import collections, json, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from handcore import config as hcfg                                 # noqa: E402
from handcore.pose import hand_twist_deg, hand_openness            # noqa: E402
from handcore.types import Hand                                      # noqa: E402
from handcore.validate import check_hand, palm_size_px, to_px       # noqa: E402

cfg = hcfg.load(str(ROOT / "config.yaml"))
g = cfg.gates
g.person_mask = False                                                # the mask is not in the recording: judge geometry gates only
path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "logs" / "live_frames.jsonl"
why_count = collections.Counter()
by_reason = collections.defaultdict(list)
n = 0
for line in path.read_text().splitlines():
    try:
        r = json.loads(line)
    except Exception:
        continue
    for h in r["hands"]:
        lm = np.array(h["lm"]); w = np.array(h["world"])
        lm[:, 0] = 1.0 - lm[:, 0]; w[:, 0] = -w[:, 0]                # mirror like HandCore.ingest
        hh = Hand(lm, w, h["label"], h["score"])
        ok, why = check_hand(hh, cfg.camera.width, cfg.camera.height, g)
        px = to_px(lm, cfg.camera.width, cfg.camera.height)
        n += 1
        why_count[why or "ok"] += 1
        by_reason[why or "ok"].append((hand_twist_deg(w), hand_openness(px, palm_size_px(px))))
print(f"{n} detections replayed")
for k, v in why_count.most_common():
    a = np.array(by_reason[k])
    print(f"  {k:16s} {v:5d} {100 * v / n:5.1f} %   twist mean {a[:, 0].mean():5.1f}  openness mean {a[:, 1].mean():4.2f}")
