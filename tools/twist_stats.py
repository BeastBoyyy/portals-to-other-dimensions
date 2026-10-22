"""Per twist angle bin: how reliable are detections? (from raw recordings written by the app)
    python tools/twist_stats.py logs/live_frames.jsonl [more.jsonl ...]"""
import collections, json, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from handcore import config as hcfg                               # noqa: E402
from handcore.pose import hand_twist_deg                          # noqa: E402
from handcore.types import Hand                                   # noqa: E402
from handcore.validate import check_hand, palm_size_px, to_px     # noqa: E402

cfg = hcfg.load(str(ROOT / "config.yaml"))
cfg.gates.geometry = True                                         # what the OLD geometry gate would have said
BINS = [(0, 30), (30, 50), (50, 65), (65, 78), (78, 91)]
paths = [pathlib.Path(p) for p in sys.argv[1:]] or [ROOT / "logs" / "live_frames.jsonl"]
stat = {b: collections.Counter() for b in BINS}
jit = {b: [] for b in BINS}
gap = {b: [0, 0] for b in BINS}
for path in paths:
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    prev_tw = None
    for i, r in enumerate(rows):
        nxt = rows[i + 1] if i + 1 < len(rows) else None
        for h in r["hands"]:
            lm = np.array(h["lm"]); w = np.array(h["world"])
            tw = hand_twist_deg(w)
            b = next(b for b in BINS if b[0] <= tw < b[1] or tw >= 90)
            stat[b]["detections"] += 1
            px = to_px(lm, 1280, 720)
            ok, why = check_hand(Hand(lm, w, h["label"], h["score"]), 1280, 720, cfg.gates)
            stat[b]["old geometry gate rejects"] += (not ok)
            stat[b]["score<0.7"] += h["score"] < 0.7
            gap[b][1] += 1
            gap[b][0] += (nxt is not None and len(nxt["hands"]) < len(r["hands"]))     # a hand vanished on the next frame
            if nxt is not None:
                for h2 in nxt["hands"]:
                    lm2 = np.array(h2["lm"])
                    if np.linalg.norm((lm2[0, :2] - lm[0, :2])) < 0.08:                  # same hand (wrist moved < 8 % of frame)
                        d = np.linalg.norm((lm2[[4, 8, 12], :2] - lm2[0, :2]) - (lm[[4, 8, 12], :2] - lm[0, :2]), axis=1).mean() * 1280
                        jit[b].append(d)
                        break
print("twist bin | detections | hand lost next frame | tip jitter relative to wrist (px/frame, median, p90) | old geometry gate rejects | score<0.7")
for b in BINS:
    n = stat[b]["detections"]
    if not n:
        continue
    j = np.array(jit[b]) if jit[b] else np.array([0.0])
    print(f" {b[0]:2d}-{b[1]:2d} deg | {n:5d} | {100 * gap[b][0] / max(gap[b][1], 1):5.1f} % | {np.median(j):5.1f}  {np.percentile(j, 90):5.1f} | "
          f"{100 * stat[b]['old geometry gate rejects'] / n:5.1f} % | {100 * stat[b]['score<0.7'] / n:5.1f} %")
