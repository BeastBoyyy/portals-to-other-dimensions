"""Measure REAL jitter and latency on a recorded session (logs/live_frames.jsonl): raw MediaPipe fingertips vs the filtered / presented portal corners,
in the windows where the hands are (nearly) still.
    python tools/real_jitter.py logs/live_frames.jsonl [more.jsonl]"""
import json, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from handcore import config as hcfg                              # noqa: E402
from handcore.pipeline import HandCore                           # noqa: E402
from handcore.predict import Presenter                           # noqa: E402
from handcore.types import Frame, Hand                           # noqa: E402


def hp_rms(x, fs, cut=3.0):
    x = np.asarray(x, float)
    x = x - x.mean(0)
    w = np.hanning(len(x))[:, None]
    X = np.fft.rfft(x * w, axis=0)
    X[np.fft.rfftfreq(len(x), 1 / fs) < cut] = 0
    return float(np.sqrt(np.mean((np.fft.irfft(X, n=len(x), axis=0) / w.mean()) ** 2)))


cfg = hcfg.load(str(ROOT / "config.yaml"))
paths = [pathlib.Path(p) for p in sys.argv[1:]] or [ROOT / "logs" / "live_frames.jsonl"]
for path in paths:
    frames = []
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        frames.append(Frame(r["t"], [Hand(np.array(h["lm"]), np.array(h["world"]), h["label"], h["score"]) for h in r["hands"]]))
    core, pres = HandCore(cfg), Presenter(cfg.predict, 1 / 30.0, n=12)
    T, RAW, CAM, OUT, PALM = [], [], [], [], []
    t_last_render = None
    for f in frames:
        st = core.update(f)
        if st is None or not st.valid or len(f.hands) != 2:
            continue
        hs = sorted(f.hands, key=lambda h: 1 - h.lm[:, 0].mean())
        raw = np.array([[(1 - h.lm[i, 0]) * 1280, h.lm[i, 1] * 720] for h in hs for i in (4, 8)])      # thumb0, idx0, thumb1, idx1 (display px)
        T.append(f.t); RAW.append(raw[[0, 1, 3, 2]].reshape(8)); CAM.append(st.corners.reshape(8))
        PALM.append(np.mean([np.linalg.norm((1 - h.lm[5, 0] - (1 - h.lm[17, 0])) * np.array([1280, 720][:1])) for h in hs]))
        pres.push(f.t + 0.03, f.t, True, np.concatenate([st.corners.reshape(8), st.lid.reshape(4)]), np.concatenate([st.velocity.reshape(8), st.lid_vel.reshape(4)]))
    T, RAW, CAM = np.array(T), np.array(RAW), np.array(CAM)
    if len(T) < 120:
        print(path.name, "not enough frames with two valid hands")
        continue
    # find still windows: 1 s windows where the raw palm-relative motion is tiny
    fs = 1.0 / np.median(np.diff(T))
    win = int(fs)
    rows = []
    for i in range(0, len(T) - win, win // 2):
        seg = slice(i, i + win)
        span = np.ptp(RAW[seg], axis=0).max()
        if span < 40 and np.ptp(T[seg]) < 1.5:                       # corners moved < 40 px over a second: "held still"
            rows.append((hp_rms(RAW[seg], fs), hp_rms(CAM[seg], fs), span))
    print(f"{path.name}: {len(T)} valid frames at {fs:.1f} Hz tracking; {len(rows)} still windows (corners moved < 40 px in 1 s)")
    if rows:
        a = np.array(rows)
        print(f"  RAW MediaPipe fingertips  HP>3Hz RMS : median {np.median(a[:, 0]):.2f} px   p90 {np.percentile(a[:, 0], 90):.2f} px")
        print(f"  filtered portal corners   HP>3Hz RMS : median {np.median(a[:, 1]):.2f} px   p90 {np.percentile(a[:, 1], 90):.2f} px   (spec target <= 0.4 px)")
    d = np.hypot(*(CAM - RAW).reshape(-1, 4, 2).transpose(2, 0, 1)).mean(1)
    print(f"  filtered vs raw tip distance: median {np.median(d):.1f} px  p90 {np.percentile(d, 90):.1f} px")
