"""Replay a recorded session through HandCore + the twist ratchet: how often does it step, and when?
    python tools/replay_dial.py logs/live_frames.jsonl"""
import json, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from handcore import config as hcfg                              # noqa: E402
from handcore.pipeline import HandCore                           # noqa: E402
from handcore.types import Frame, Hand                           # noqa: E402
from handcore.view import TwistDial                              # noqa: E402

path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "logs" / "live_frames.jsonl"
cfg = hcfg.load(str(ROOT / "config.yaml"))
g = cfg.gesture
core, dial = HandCore(cfg), TwistDial(g.dial_effects, g.dial_step_deg, g.dial_commit_s, g.dial_base_tau_s, g.dial_ease_s)
tw_s, t_prev, k_prev, steps, t0, active_t = None, None, 0, [], None, 0.0
rels = []
for line in path.read_text().splitlines():
    try:
        r = json.loads(line)
    except Exception:
        continue
    st = core.update(Frame(r["t"], [Hand(np.array(h["lm"]), np.array(h["world"]), h["label"], h["score"]) for h in r["hands"]]))
    if st is None:
        continue
    t0 = r["t"] if t0 is None else t0
    dt = 0.033 if t_prev is None else max(r["t"] - t_prev, 1e-3)
    t_prev = r["t"]
    tw_s = st.twist if tw_s is None else tw_s + (st.twist - tw_s) * min(1.0, dt * 8.0)        # same smoothing as ViewState
    act = st.openness > 0.5 and st.n_hands >= 1
    dial.update(tw_s, dt, active=act)
    active_t += dt if act else 0.0
    if act:
        rels.append(dial.rel)
    if dial.k != k_prev:
        steps.append((r["t"] - t0, g.dial_effects[dial.k], dial.rel))
        k_prev = dial.k
print(f"{path.name}: {active_t:.0f} s with the portal open; ratchet stepped {len(steps)} times ({60 * len(steps) / max(active_t, 1):.1f} per minute open)")
print("rel twist while open: median %.1f  p5 %.1f  p95 %.1f deg" % (np.median(rels), np.percentile(rels, 5), np.percentile(rels, 95)))
for t, name, rel in steps[:25]:
    print(f"  t={t:7.1f}s -> {name}")
