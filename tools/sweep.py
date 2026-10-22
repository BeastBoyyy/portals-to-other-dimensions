"""Hyper-parameter and ablation sweeps over the synthetic suite (parallel).

    python tools/sweep.py tune        # tip-offset One Euro grid  (also: tune_palm, tune_floor, tune_gate, tune_spike)
    python tools/sweep.py compare     # spec starting values vs the final filter design
    python tools/sweep.py ablate      # feature on/off table -> ABLATION.md
"""
from __future__ import annotations
import itertools, json, pathlib, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from handcore import config as hcfg          # noqa: E402
from tools import synth, evaluate as ev      # noqa: E402


def job(args):
    kind, seed, dur, overrides, ghosts = args
    cfg = hcfg.load(str(ROOT / "config.yaml"), overrides)
    return kind, ev.evaluate_clip(synth.make_clip(kind, seed=seed, dur=dur, ghosts=ghosts), cfg)


def run(overrides, kinds, seeds=(0, 1, 2), dur=15.0, pool=None, ghosts=None):
    jobs = [(k, s, dur, overrides, ghosts if k == "no_hands" else None) for k in kinds for s in seeds]
    res = list(pool.map(job, jobs)) if pool else [job(j) for j in jobs]
    out = {}
    for k, m in res:
        out.setdefault(k, []).append(m)
    return out


def agg(out, kind, key, f=np.mean):
    v = [m[key] for m in out.get(kind, []) if key in m and m[key] == m[key]]
    return float(f(v)) if v else float("nan")


def quality(overrides, pool, kinds=("still", "slow", "fast", "tilt", "crossing")):
    o = run(overrides, kinds, pool=pool)
    return {"jitter": agg(o, "still", "jitter"), "lag": float(np.mean([agg(o, k, "lag_frames") for k in ("slow", "fast", "tilt")])),
            "acc_still": agg(o, "still", "acc"), "acc_fast": agg(o, "fast", "acc"), "acc_tilt": agg(o, "tilt", "acc"),
            "acc_all": float(np.mean([agg(o, k, "acc") for k in kinds])), "render_acc_fast": agg(o, "fast", "render_acc"),
            "r50_slow": agg(o, "slow", "render_p50"), "r50_fast": agg(o, "fast", "render_p50"),
            "outlier": float(np.mean([agg(o, k, "outlier") for k in kinds])), "swaps": sum(m.get("swaps", 0) for m in o.get("crossing", [])),
            "pops": sum(m["pops"] for k in kinds for m in o[k])}


def fmt(d):
    return " ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items())


def tune():
    grid = list(itertools.product((0.7, 1.5), (0.3, 1.0, 3.0), (6.0, 15.0, 40.0)))
    rows = []
    with ProcessPoolExecutor() as pool:
        for mc, beta, dc in grid:
            ov = {"filter": {"min_cutoff": mc, "beta": beta, "d_cutoff": dc}}
            q = quality(ov, pool, kinds=("still", "slow", "fast", "tilt"))
            rows.append((mc, beta, dc, q))
            print(f"mc={mc} beta={beta} dc={dc} :: {fmt(q)}", flush=True)
    json.dump([[a, b, c, q] for a, b, c, q in rows], open(ROOT / "tune_filter.json", "w"), indent=1)


def tune_floor():
    grid = list(itertools.product((0.5, 1.0), (0.3, 1.0), (40.0, 80.0, 150.0), (20.0, 60.0)))
    with ProcessPoolExecutor() as pool:
        for mc, b, fl, pfl in grid:
            ov = {"filter": {"min_cutoff": mc, "beta": b, "speed_floor": fl, "palm_speed_floor": pfl}}
            print(f"mc={mc} b={b} floor={fl} pfloor={pfl} :: {fmt(quality(ov, pool, kinds=('still', 'slow', 'fast', 'tilt')))}", flush=True)


def tune_gate():
    grid = list(itertools.product((6000.0, 10000.0, 15000.0, 30000.0), (9.0, 16.0, 25.0)))
    with ProcessPoolExecutor() as pool:
        for a, th in grid:
            ov = {"kalman": {"accel_sigma": a}, "gates": {"maha_thresh": th}}
            print(f"accel={a} maha={th} :: {fmt(quality(ov, pool, kinds=('still', 'slow', 'fast', 'tilt', 'reentry', 'crossing')))}", flush=True)


def tune_spike():
    grid = list(itertools.product((0.15, 0.2, 0.3), (0.5, 1.0, 2.0), (2, 3)))
    with ProcessPoolExecutor() as pool:
        for fr, g, per in grid:
            ov = {"constraint": {"spike_palm_frac": fr, "spike_speed_gain": g, "spike_persist": per}}
            print(f"frac={fr} gain={g} persist={per} :: {fmt(quality(ov, pool, kinds=('still', 'slow', 'fast', 'tilt', 'occluded', 'lowlight')))}", flush=True)


def compare():
    """Spec starting values vs the final filter design (the number behind DECISIONS.md D7)."""
    variants = {
        "final: decomposed palm + tip-offset 1-euro (tuned)": {},
        "spec start values on tips (min_cutoff 1.0, beta 0.007, d_cutoff 1.0)": {"filter": {"decompose": False, "min_cutoff": 1.0, "beta": 0.007, "d_cutoff": 1.0, "speed_floor": 0.0}},
        "plain 1-euro on tips, tuned (0.7 / 0.3 / 6, floor 80)": {"filter": {"decompose": False}},
    }
    with ProcessPoolExecutor() as pool:
        for name, ov in variants.items():
            q = quality(ov, pool, kinds=("still", "slow", "fast", "tilt"))
            print(name, "::", fmt({k: q[k] for k in ("jitter", "acc_still", "acc_fast", "acc_tilt", "acc_all", "outlier", "r50_slow", "r50_fast")}), flush=True)


def tune_palm():
    rels = ((0.7, 0.05, 2.0), (0.7, 0.3, 6.0))
    grid = list(itertools.product(rels, (1.0, 3.0), (0.1, 0.5, 2.0), (5.0, 15.0, 40.0)))
    with ProcessPoolExecutor() as pool:
        for (rmc, rb, rdc), pmc, pb, pdc in grid:
            ov = {"filter": {"min_cutoff": rmc, "beta": rb, "d_cutoff": rdc, "palm_min_cutoff": pmc, "palm_beta": pb, "palm_d_cutoff": pdc}}
            q = quality(ov, pool, kinds=("still", "slow", "fast", "tilt"))
            print(f"rel=({rmc},{rb},{rdc}) palm=({pmc},{pb},{pdc}) :: {fmt(q)}", flush=True)


FEATURES = {
    "baseline (all on)": {},
    "no gates (4.3 off)": {"gates": {"enable": False, "temporal": False, "motion": False}},
    "no temporal gate": {"gates": {"temporal": False}},
    "no motion gate": {"gates": {"motion": False}},
    "no geometry gate": {"gates": {"geometry": False}},
    "no identity tracking (x-order)": {"identity": {"enable": False}},
    "kinematic constraint ON": {"constraint": {"enable": True}},
    "no anchor-spike gate": {"constraint": {"anchor_spike": False}},
    "filter: none": {"filter": {"kind": "none"}},
    "filter: moving average (3)": {"filter": {"kind": "moving_average", "ma_window": 3}},
    "prediction off": {"predict": {"enable": False}},
}


def ablate(base_over=None):
    base_over = base_over or {}
    rows = []
    with ProcessPoolExecutor() as pool:
        for name, ov in FEATURES.items():
            merged = json.loads(json.dumps(base_over))
            for sec, d in ov.items():
                merged.setdefault(sec, {}).update(d)
            q = quality(merged, pool, kinds=("still", "slow", "fast", "tilt", "crossing", "near_cross", "reentry", "occluded", "lowlight"))
            o = run(merged, ("no_hands",), pool=pool, ghosts=["poster2", "flicker", "random"])
            q["false_hand"] = agg(o, "no_hands", "false_hand", lambda v: float(np.mean([x["cam"] for x in v])))
            rows.append((name, q)); print(name, "::", fmt(q), flush=True)
        # person-mask gate on/off for the poster scenario
        for name, ov in (("person-mask gate OFF (poster x2)", {"gates": {"person_mask": False}}),
                         ("person-mask gate ON  (poster x2)", {"gates": {"person_mask": True}})):
            merged = json.loads(json.dumps(base_over))
            for sec, d in ov.items():
                merged.setdefault(sec, {}).update(d)
            o = run(merged, ("no_hands",), pool=pool, ghosts=["poster2", "flicker", "random"])
            rows.append((name, {"false_hand": agg(o, "no_hands", "false_hand", lambda v: float(np.mean([x["cam"] for x in v])))}))
            print(name, rows[-1][1], flush=True)
    L = ["# Ablation (synthetic suite, 3 seeds x 15 s per clip; see REPORT.md for caveats)", "",
         "| Variant | jitter px | acc% still | acc% fast | acc% tilt | acc% mean | outlier% | swaps | render err px slow / fast (median) | pops | false-hand % |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, q in rows:
        g = lambda k, nd=2: (f"{q[k]:.{nd}f}" if k in q and q[k] == q[k] else "-")
        L.append(f"| {name} | {g('jitter')} | {g('acc_still')} | {g('acc_fast')} | {g('acc_tilt')} | {g('acc_all')} | {g('outlier',3)} | {q.get('swaps','-')} | {g('r50_slow',1)} / {g('r50_fast',1)} | {q.get('pops','-')} | {g('false_hand',3)} |")
    (ROOT / "ABLATION.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote ABLATION.md")


if __name__ == "__main__":
    {"tune": tune, "tune_palm": tune_palm, "compare": compare, "tune_spike": tune_spike, "tune_gate": tune_gate, "tune_floor": tune_floor, "ablate": ablate}[sys.argv[1]]()
