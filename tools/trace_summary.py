"""Summarize logs/trace.jsonl written by app/portal_app.py while you use it: how often hands were detected / tracked / valid, and why not.
    python tools/trace_summary.py [last_seconds]"""
import collections, json, pathlib, sys

p = pathlib.Path(__file__).resolve().parent.parent / "logs" / "trace.jsonl"
rows = []
for line in p.read_text().splitlines():
    try:
        rows.append(json.loads(line))
    except Exception:
        pass
if not rows:
    sys.exit("no trace yet")
if len(sys.argv) > 1:
    t_end = rows[-1]["t"]
    rows = [r for r in rows if r["t"] >= t_end - float(sys.argv[1])]
n = len(rows)
c, rej, qr = collections.Counter(), collections.Counter(), collections.Counter()
for r in rows:
    c[f"detections={r['det']}"] += 1
    c[f"tracked slots={r['n_hands']}"] += 1
    c["portal valid"] += r["valid"]
    c["held"] += r["held"]
    for k, v in r["rej"].items():
        rej[k] += v
    if r["quad_reject"]:
        qr[r["quad_reject"]] += 1
print(f"{n} frames over {rows[-1]['t'] - rows[0]['t']:.1f} s")
for k, v in sorted(c.items()):
    print(f"  {k:24s} {v:5d}  {100 * v / n:5.1f} %")
print("gate rejects (per detection):", dict(rej))
print("quad rejects:", dict(qr))
op = [r["open"] for r in rows if r["n_hands"] == 2]
if op:
    print(f"openness when 2 hands tracked: min {min(op):.2f} mean {sum(op) / len(op):.2f} max {max(op):.2f}")
