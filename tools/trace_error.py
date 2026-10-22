"""From logs/trace.jsonl: how far are the portal corners (core output) from the raw detected fingertips? Reveals lag, wrong fingers, swapped hands.
    python tools/trace_error.py [last_seconds]"""
import json, pathlib, sys
import numpy as np

p = pathlib.Path(__file__).resolve().parent.parent / "logs" / "trace.jsonl"
rows = []
for line in p.read_text().splitlines():
    try:
        rows.append(json.loads(line))
    except Exception:
        pass
if len(sys.argv) > 1 and rows:
    rows = [r for r in rows if r["t"] >= rows[-1]["t"] - float(sys.argv[1])]
err, swaps, n2 = [], 0, 0
for r in rows:
    if not r["valid"] or r["det"] != 2 or r["held"]:
        continue
    n2 += 1
    hs = sorted(r["tips"], key=lambda h: np.mean([pt[0] for pt in h[:4]]))      # left hand first (display x)
    c = np.array(r["corners"])                                                   # [thumb0, idx0, idx1, thumb1]
    L, Rr = np.array(hs[0][:4]), np.array(hs[1][:4])
    ref = np.array([L[0], L[1], Rr[1], Rr[0]])
    e = np.hypot(*(c - ref).T)
    swapped = np.hypot(*(c - np.array([Rr[0], Rr[1], L[1], L[0]])).T)
    if swapped.mean() < e.mean():
        swaps += 1
    err.append(e)
if not err:
    sys.exit("no frames with a valid portal and two detections yet")
e = np.array(err)
print(f"{len(e)} frames (valid portal, two detections)")
for i, name in enumerate(("left thumb", "left index", "right index", "right thumb")):
    print(f"  {name:12s} error vs detected fingertip: median {np.median(e[:, i]):5.1f} px   p90 {np.percentile(e[:, i], 90):5.1f}   >40 px {100 * np.mean(e[:, i] > 40):4.1f} %")
print(f"frames where the corners match the OTHER hand better (swap): {swaps}")
