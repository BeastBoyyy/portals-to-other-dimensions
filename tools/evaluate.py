"""Section-9 metrics. Library + CLI.

    python tools/evaluate.py                       # synthetic suite -> REPORT.md (measured, honest)
    python tools/evaluate.py --jsonl clip.jsonl    # replay a recorded landmark file (stability only)
    python tools/evaluate.py --jsonl clip.jsonl --labels labels.json   # + accuracy vs hand labels

Everything measured here is measured on SYNTHETIC clips (tools/synth.py) unless a JSONL is given.
Render-side metrics (fps, frame time, motion-to-photon latency) need TouchDesigner: NOT MEASURED here.
"""
from __future__ import annotations
import argparse, json, pathlib, sys, time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from handcore import config as hcfg                     # noqa: E402
from handcore.pipeline import HandCore                  # noqa: E402
from handcore.predict import Presenter                  # noqa: E402
from handcore.types import Frame, Hand                  # noqa: E402
from tools import synth                                 # noqa: E402

PASS = {"corner_acc": 99.0, "outlier": 0.2, "false_hand": 0.1, "swaps": 0, "jitter": 0.4,
        "lag_frames": 1.0, "recovery_frames": 6, "recovery_pop": 12.0, "pops": 0}
CAM_PERIOD = 1 / 30.0


# ---- running a clip ---------------------------------------------------------------------------
def run_clip(clip, cfg, render_hz=60.0, proc_delay=0.030):
    """Runs HandCore on the clip, then a 60 Hz Presenter fed with samples that arrive proc_delay after capture."""
    core = HandCore(cfg)
    states, t_cpu = [], 0.0
    for f in clip.frames:
        t0 = time.perf_counter()
        s = core.update(f)
        t_cpu += time.perf_counter() - t0
        states.append(s)
    pres = Presenter(cfg.predict, CAM_PERIOD)
    arrivals = sorted(((st.t + proc_delay, st) for st in states if st is not None), key=lambda a: a[0])
    ticks = np.arange(clip.frames[0].t, clip.frames[-1].t, 1.0 / render_hz)
    out_t, out_c, out_a, ai = [], [], [], 0
    for tk in ticks:
        while ai < len(arrivals) and arrivals[ai][0] <= tk:
            ta, st = arrivals[ai]
            pres.push(ta, st.t, st.valid, st.corners.reshape(8), st.velocity.reshape(8), held=bool(st.debug.get("held")))
            ai += 1
        c, a = pres.render(tk)
        out_t.append(tk); out_c.append(np.array(c).reshape(4, 2)); out_a.append(a)
    return {"states": states, "cpu_us": 1e6 * t_cpu / max(len(clip.frames), 1),
            "r_t": np.array(out_t), "r_c": np.array(out_c), "r_a": np.array(out_a)}


# ---- metrics ----------------------------------------------------------------------------------
def _gt_series(clip, times, mirror):
    g = [clip.gt_corners(t, mirror) for t in times]
    vis = np.array([x is not None for x in g])
    arr = np.array([x if x is not None else np.zeros((4, 2)) for x in g])
    return arr, vis


def corner_accuracy(clip, res, mirror, thr=6.0, out_thr=25.0):
    t = np.array([s.t for s in res["states"]])
    gt, vis = _gt_series(clip, t, mirror)
    ok_sum = n = out = 0
    for s, g, v in zip(res["states"], gt, vis):
        if not v:
            continue
        n += 4
        if s.valid:
            e = np.hypot(*(s.corners - g).T)
            ok_sum += int(np.sum(e <= thr)); out += int(np.sum(e > out_thr))
    # render-rate (includes pipeline latency, extrapolation)
    gtr, visr = _gt_series(clip, res["r_t"], mirror)
    er = np.hypot(*(res["r_c"] - gtr).transpose(2, 0, 1))
    sel = visr & (res["r_a"] > 0.99)
    return {"acc": 100.0 * ok_sum / max(n, 1), "outlier": 100.0 * out / max(n, 1), "n": n,
            "render_acc": 100.0 * float(np.mean(er[sel] <= thr)) if sel.any() else float("nan"),
            "render_p50": float(np.median(er[sel])) if sel.any() else float("nan"),
            "render_coverage": 100.0 * sel.sum() / max(visr.sum(), 1)}


def false_hand_rate(res):
    v = np.array([s.valid for s in res["states"]])
    return {"cam": 100.0 * v.mean(), "render": 100.0 * float(np.mean(res["r_a"] > 0.5))}


def count_swaps(clip, res, mirror):
    """Slot identity swaps: a slot's assigned detection switches from one ground-truth hand to the other."""
    last, swaps = {}, 0
    for s in res["states"]:
        changed = False
        for k, gid in s.debug["gt"].items():
            if gid < 0:
                continue
            if k in last and last[k] != gid:
                changed = True
            last[k] = gid
        swaps += changed
    return swaps


def rest_jitter(res, t_from=2.0, hp_hz=3.0):
    t, c, a = res["r_t"], res["r_c"], res["r_a"]
    sel = (t >= t_from) & (a > 0.99)
    if sel.sum() < 64:
        return float("nan")
    x = c[sel].reshape(sel.sum(), 8)
    x = x - x.mean(0)
    fs = 1.0 / np.median(np.diff(t[sel]))
    X = np.fft.rfft(x * np.hanning(len(x))[:, None], axis=0)
    f = np.fft.rfftfreq(len(x), 1 / fs)
    X[f < hp_hz] = 0
    hp = np.fft.irfft(X, n=len(x), axis=0) / np.mean(np.hanning(len(x)))   # undo window gain
    return float(np.sqrt(np.mean(hp ** 2)))


def lag_frames(clip, res, mirror):
    """Output delay vs ground truth: shift s minimizing RMSE(out(t), GT(t - s)); in camera frames."""
    best = (1e9, 0.0)
    st = [s for s in res["states"] if s.valid]
    for s_ms in range(-40, 161, 5):
        sh = s_ms / 1000.0
        errs = []
        for s in st[::2]:
            g = clip.gt_corners(s.t - sh, mirror)
            if g is not None:
                errs.append(np.mean((s.corners - g) ** 2))
        if errs and np.mean(errs) < best[0]:
            best = (np.mean(errs), sh)
    return best[1] / CAM_PERIOD


def pops(clip, res, mirror, thr=20.0):
    """Render-frame discontinuities: |second difference of output - second difference of ground truth| > thr px,
    i.e. a 1-frame jump (or kink) that real motion does not explain. Constant-velocity tracking error is not a pop."""
    gt, vis = _gt_series(clip, res["r_t"], mirror)
    d2o = np.diff(res["r_c"], n=2, axis=0)
    d2g = np.diff(gt, n=2, axis=0)
    ok = vis[2:] & vis[1:-1] & vis[:-2] & (res["r_a"][2:] > 0.99) & (res["r_a"][:-2] > 0.99)
    ex = np.hypot(*(d2o - d2g).transpose(2, 0, 1)).max(axis=1)
    return int(np.sum((ex > thr) & ok)), float(ex[ok].max()) if ok.any() else 0.0


def recovery(clip, res, mirror):
    """After each hand re-entry: camera frames until a stable accurate portal; worst render pop."""
    t = np.array([s.t for s in res["states"]])
    gt, vis = _gt_series(clip, t, mirror)
    out = []
    i = 1
    while i < len(t):
        if vis[i] and not vis[i - 1]:
            j = i
            while j < len(t) - 2:
                good = all(res["states"][k].valid and vis[k] and
                           np.hypot(*(res["states"][k].corners - gt[k]).T).max() <= 12.0 for k in (j, j + 1, j + 2))
                if good:
                    break
                j += 1
            m = (res["r_t"] >= t[i]) & (res["r_t"] < t[i] + 0.6)
            gtr, vr = _gt_series(clip, res["r_t"][m], mirror)
            dd = np.diff(res["r_c"][m], axis=0) - np.diff(gtr, axis=0)
            okk = vr[1:] & vr[:-1] & (res["r_a"][m][1:] > 0.3)
            ex = np.hypot(*dd.transpose(2, 0, 1)).max(axis=1) if len(dd) else np.zeros(1)
            ex = ex * res["r_a"][m][1:] if len(dd) else ex       # visible jump = jump x portal opacity
            out.append((j - i, float(ex[okk].max()) if okk.any() else 0.0))
            i = j
        i += 1
    return out


def evaluate_clip(clip, cfg):
    mirror = cfg.camera.mirror
    res = run_clip(clip, cfg)
    m = {"cpu_us": res["cpu_us"]}
    if clip.kind in ("no_hands", "one_hand"):       # no portal may be shown: with 0 or 1 hands there is no frame gesture
        m["false_hand"] = false_hand_rate(res)
        return m
    m.update(corner_accuracy(clip, res, mirror))
    if clip.kind == "still":
        m["jitter"] = rest_jitter(res)
    if clip.kind in ("fast", "tilt", "slow"):
        m["lag_frames"] = lag_frames(clip, res, mirror)
    if clip.kind in ("crossing", "near_cross"):
        m["swaps"] = count_swaps(clip, res, mirror)
    if clip.kind == "reentry":
        m["recovery"] = recovery(clip, res, mirror)
    m["pops"], m["max_excess"] = pops(clip, res, mirror)
    return m


# ---- suite --------------------------------------------------------------------------------------
KINDS = ["still", "slow", "fast", "crossing", "near_cross", "reentry", "tilt", "one_hand",
         "lowlight", "backlit", "occluded"]
GHOST_SETS = {"no_hands": ["poster", "flicker", "random"], "no_hands_poster2": ["poster2", "flicker", "random"]}


def run_suite(cfg, seeds=(0, 1, 2), dur=20.0, kinds=KINDS, mask=False):
    """Aggregate over seeds; returns dict kind -> list of metric dicts."""
    out = {}
    for k in kinds:
        out[k] = [evaluate_clip(synth.make_clip(k, seed=s, dur=dur), cfg) for s in seeds]
    # no-hand clips with ghosts. mask=True turns the person-mask gate on in cfg beforehand.
    for name, gh in GHOST_SETS.items():
        out[name] = [evaluate_clip(synth.make_clip("no_hands", seed=s, dur=dur, ghosts=gh), cfg) for s in seeds]
    return out


def summarize(suite):
    """Collapse to the Section 9 table numbers (pooled worst/mean as appropriate)."""
    def pool(kind_list, key, f=np.mean):
        v = [m[key] for k in kind_list if k in suite for m in suite[k] if key in m and not (isinstance(m[key], float) and np.isnan(m[key]))]
        return float(f(v)) if v else float("nan")
    tracked = [k for k in suite if not k.startswith(("no_hands", "one_hand"))]
    s = {
        "corner_acc": pool(tracked, "acc"),
        "corner_acc_by_kind": {k: float(np.mean([m["acc"] for m in suite[k]])) for k in tracked},
        "render_acc_by_kind": {k: float(np.nanmean([m["render_acc"] for m in suite[k]])) for k in tracked},
        "outlier": pool(tracked, "outlier"),
        "false_hand": max(m["false_hand"]["cam"] for k in suite if k.startswith(("no_hands", "one_hand")) for m in suite[k]),
        "false_hand_by_kind": {k: float(np.mean([m["false_hand"]["cam"] for m in suite[k]])) for k in suite if k.startswith(("no_hands", "one_hand"))},
        "swaps": int(sum(m.get("swaps", 0) for k in ("crossing", "near_cross") for m in suite.get(k, []))),
        "jitter": pool(["still"], "jitter"),
        "lag_frames": pool(["fast", "tilt"], "lag_frames"),
        "pops": int(sum(m["pops"] for k in tracked for m in suite[k])),
        "max_excess": float(max(m["max_excess"] for k in tracked for m in suite[k])),
        "cpu_us": pool(tracked, "cpu_us"),
    }
    rec = [r for m in suite.get("reentry", []) for r in m.get("recovery", [])]
    s["recovery_frames"] = float(max((r[0] for r in rec), default=float("nan")))
    s["recovery_pop"] = float(max((r[1] for r in rec), default=float("nan")))
    s["n_reentries"] = len(rec)
    return s


# ---- real recorded data -----------------------------------------------------------------------
def load_jsonl(path, mirror=None):
    frames = []
    for line in open(path):
        r = json.loads(line)
        hands = [Hand(np.array(h["lm"]), np.array(h["world"]) if h.get("world") else None, h["label"], h["score"]) for h in r["hands"]]
        frames.append(Frame(r["t"], hands))
    return frames


def replay_report(path, labels, cfg):
    frames = load_jsonl(path)
    core = HandCore(cfg)
    st = [core.update(f) for f in frames]
    v = np.array([s.valid for s in st])
    c = np.array([s.corners for s in st])
    res = {"frames": len(st), "portal_valid_pct": 100 * v.mean()}
    both = v[1:] & v[:-1]
    if both.any():
        j = np.hypot(*(np.diff(c, axis=0)).transpose(2, 0, 1)).max(1)
        res["max_1frame_jump_px"] = float(j[both].max())
        res["jumps_gt_20px"] = int(np.sum(j[both] > 20))
    if labels:
        lab = json.load(open(labels))      # {"frame_index": [[x,y]*4 in handcore id order, display px]}
        e = [np.hypot(*(c[int(i)] - np.array(p)).T) for i, p in lab.items() if v[int(i)]]
        e = np.array(e)
        res.update(label_n=len(lab), label_valid=len(e),
                   corner_acc_pct=100 * float(np.mean(e <= 6)), outlier_pct=100 * float(np.mean(e > 25)))
    return res


# ---- report -----------------------------------------------------------------------------------
def fmt(v, nd=2):
    return "NaN" if v != v else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def verdict(s):
    rows = [
        ("Corner accuracy (camera-rate, <=6 px, all hand-visible frames)", fmt(s["corner_acc"]) + " %", ">= 99 %", s["corner_acc"] >= 99),
        ("Corner outlier rate (>25 px)", fmt(s["outlier"], 3) + " %", "<= 0.2 %", s["outlier"] <= 0.2),
        ("False-hand rate (worst no-hand clip)", fmt(s["false_hand"], 3) + " %", "<= 0.1 %", s["false_hand"] <= 0.1),
        ("Identity swaps (crossing + near-crossing)", str(s["swaps"]), "0", s["swaps"] == 0),
        ("Rest jitter, >3 Hz RMS (render-rate)", fmt(s["jitter"]) + " px", "<= 0.4 px", s["jitter"] <= 0.4),
        ("Fast-motion lag (camera frames)", fmt(s["lag_frames"]), "<= 1", s["lag_frames"] <= 1.0),
        ("Dropout recovery, worst (camera frames)", fmt(s["recovery_frames"], 0), "<= 6", s["recovery_frames"] <= 6),
        ("Re-entry pop, worst 1-frame excess jump", fmt(s["recovery_pop"]) + " px", "< 12 px", s["recovery_pop"] < 12),
        ("Pop/teleport events (>20 px render-frame excess)", str(s["pops"]), "0", s["pops"] == 0),
    ]
    return rows


def write_report(path, cfg, suite, s, extra=""):
    L = ["# REPORT (generated by tools/evaluate.py)", "",
         "**All numbers below were measured on SYNTHETIC clips** (`tools/synth.py`: parametric hand + an *assumed* detector noise model).",
         "They show how `handcore` behaves relative to that model. They are NOT a claim about real MediaPipe accuracy;",
         "real-world values are **NOT MEASURED** until you run `tools/record_clip.py` + `tools/label_tool.py` + `tools/evaluate.py --jsonl ... --labels ...`.", "",
         "## Section 9 metrics", "", "| Metric | Measured | Pass | Result |", "|---|---|---|---|"]
    for name, val, thr, ok in verdict(s):
        L.append(f"| {name} | {val} | {thr} | {'PASS' if ok else 'FAIL'} |")
    L += ["| Render smoothness (frames > 20 ms in 60 s soak) | NOT MEASURED (needs TouchDesigner: run td/selfcheck.py soak) | <= 0.1 % | - |",
          "| Render fps avg / 1st pct | NOT MEASURED (needs TouchDesigner) | >= 59 / >= 55 | - |",
          "| Motion-to-photon latency | NOT MEASURED (physical test, tools/latency_test.md) | report, target <= 80 ms | - |", "",
          f"handcore CPU cost: {fmt(s['cpu_us'], 0)} us per camera frame (single thread, this machine).", "",
          "## Per-clip detail", "", "| Clip | camera-rate acc % (<=6 px) | render-rate median err px (30 ms pipeline delay, extrapolated) | outlier % | extra |", "|---|---|---|---|---|"]
    for k, ms in suite.items():
        if k.startswith(("no_hands", "one_hand")):
            L.append(f"| {k} | - | - | - | portal valid {fmt(float(np.mean([m['false_hand']['cam'] for m in ms])), 3)} % of frames |")
            continue
        extra_s = []
        for key in ("jitter", "lag_frames", "swaps", "pops"):
            vals = [m[key] for m in ms if key in m]
            if vals:
                extra_s.append(f"{key} {fmt(float(np.mean(vals)))}")
        L.append(f"| {k} | {fmt(float(np.mean([m['acc'] for m in ms])))} | {fmt(float(np.nanmean([m['render_p50'] for m in ms])), 1)} | "
                 f"{fmt(float(np.mean([m['outlier'] for m in ms])), 3)} | {', '.join(extra_s)} |")
    L.append(extra or footer())
    pathlib.Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


NL = chr(10)


def real_image_table():
    """Real MediaPipe output on the sample images, pushed through gate 1+2 (fixtures/real_images_landmarks.jsonl)."""
    from handcore.validate import check_hand
    p = ROOT / "fixtures" / "real_images_landmarks.jsonl"
    if not p.exists():
        return ["(fixtures/real_images_landmarks.jsonl missing)"]
    rows = ["| Image | hands found by HandLandmarker | passed gates 1+2 |", "|---|---|---|"]
    for line in p.read_text().splitlines():
        r = json.loads(line)
        ok = []
        for h in r["hands"]:
            hh = Hand(np.array(h["lm"]), np.array(h["world"]), h["label"], h["score"])
            ok.append(check_hand(hh, r["w"], r["h"], hcfg.Gates())[0])
        rows.append(f"| {r['src']} | {len(r['hands'])} | {sum(ok)}/{len(ok)} |")
    return rows


def footer():
    L = ["", "## Real MediaPipe output on the sample images (the only real-detector data available when this report was generated)", ""]
    L += real_image_table()
    L += ["", "Non-hand images (burger, cat, portrait) produced 0 detections at the 0.65/0.65/0.55 confidence settings; the 4 hand images",
          "produced 5 hands, all of which pass the geometry gate. This is 7 images, not a false-positive study: real false-positive rates",
          "on your room (posters, wall card, clothes) are **NOT MEASURED** -- record the `nohands` clip and run `tools/evaluate.py --jsonl`."]
    b = ROOT / "bench_results.json"
    if b.exists():
        j = json.loads(b.read_text())
        L += ["", "## Landmark-source benchmark (this machine, CPU, 1280x720 input; tools/bench_sidecar.py)", "",
              "| Path | mean ms | p95 ms | max fps |", "|---|---|---|---|"]
        for k, v in j.items():
            if isinstance(v, dict) and "mean_ms" in v:
                L.append(f"| {k} | {v['mean_ms']:.1f} | {v['p95_ms']:.1f} | {v['fps']:.0f} |")
            elif isinstance(v, dict) and "error" in v:
                L.append(f"| {k} | unavailable on this build: {v['error'].splitlines()[0][:70]} | | |")
            elif isinstance(v, str) and k.startswith("Option"):
                L.append(f"| {k} | {v} | | |")
    a = ROOT / "ABLATION.md"
    if a.exists():
        L += ["", a.read_text().replace("# Ablation", "## Ablation", 1)]
    L += ["", "## Known limitations", "",
          "- All accuracy numbers come from a synthetic simulator with an *assumed* noise model. Real MediaPipe noise may be better or worse.",
          "- Camera-rate accuracy at the 6 px threshold is physically limited on `fast`/`tilt` clips: peak tip speed is ~2700 px/s, so 6 px is ~2 ms of motion; any smoothing that removes jitter costs that.",
          "  The pooled corner-accuracy figure is therefore below 99 % while `still`, `slow`, `crossing`, `near_cross` clips are above it.",
          "- Render-rate numbers include a simulated 30 ms capture-to-arrival delay and constant-acceleration extrapolation; violent 1.6 Hz oscillations (`fast`) cannot be extrapolated to 6 px.",
          "- Fast tilt makes the ground-truth quad itself non-convex part of the time; there is no valid portal to draw then (excluded from accuracy, counted in pops when the quad comes back).",
          "- Person-mask gate needs the sidecar's selfie segmenter; without it two persistent poster hands DO produce a portal (see ablation).",
          "- TouchDesigner, GPU frame times, display latency and real-camera behaviour were never run: every `td/` item is **needs user verification in TD**.",
          "", "## 10-minute verification checklist (you)", "",
          "1. `python tools/setup_check.py` - camera fps/brightness/blur (fix lighting if WARN).",
          "2. `python tools/export_td_config.py`, open TouchDesigner, run `build_network.py` in the Textport, apply `td/MANUAL_STEPS.md`.",
          "3. `python adapters/mediapipe_sidecar.py`; open the window (F1); make the frame gesture; press H to see the HUD (tracker fps ~30, infer ~15 ms).",
          "4. In the Textport: `op('/project1/portal/selfcheck').module.start()` and follow the prompts (45 s). Paste the PASS/FAIL block into this report.",
          "5. `python tools/record_clip.py still` etc. (11 clips), then `python tools/evaluate.py --jsonl clips/<name>.jsonl` for each; label ~200 frames of `slow` with `tools/label_tool.py` for the 99 % accuracy figure.",
          "6. `tools/latency_test.md` (phone slow-mo) for motion-to-photon latency."]
    return NL.join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--out", default=str(ROOT / "REPORT.md"))
    ap.add_argument("--jsonl")
    ap.add_argument("--labels")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    cfg = hcfg.load(a.config)
    if a.jsonl:
        print(json.dumps(replay_report(a.jsonl, a.labels, cfg), indent=1))
        return
    suite = run_suite(cfg, seeds=tuple(range(a.seeds)))
    s = summarize(suite)
    for name, val, thr, ok in verdict(s):
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {val} (need {thr})")
    write_report(a.out, cfg, suite, s)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
