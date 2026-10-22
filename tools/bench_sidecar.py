"""Phase-1 benchmark of the sidecar landmark path on THIS machine (Option B/C front half).

    python tools/bench_sidecar.py            # writes bench_results.json and prints a markdown table

Input is a moving 1280x720 test video built from the sample hand images (translated + slightly scaled each frame), so the
detector sees a hand every frame. Option A (in-TouchDesigner Web Render TOP plugin) cannot be run outside TouchDesigner:
it is reported as NOT MEASURED.
"""
from __future__ import annotations
import json, pathlib, sys, threading, time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import cv2, mediapipe as mp                                      # noqa: E402
from mediapipe.tasks.python import vision, BaseOptions          # noqa: E402
from tools.extract_landmarks import make_landmarker, MODEL      # noqa: E402

W, H, N = 1280, 720, 240


def make_frames():
    img = cv2.imread(str(ROOT / "fixtures" / "images" / "woman_hands.jpg"))
    base = cv2.resize(img, (W, H))
    frames = []
    for i in range(N):
        M = np.float32([[1, 0, 25 * np.sin(i / 15)], [0, 1, 12 * np.cos(i / 11)]])
        frames.append(cv2.cvtColor(cv2.warpAffine(base, M, (W, H), borderMode=cv2.BORDER_REPLICATE), cv2.COLOR_BGR2RGB))
    return frames


def stats(ms):
    a = np.asarray(ms)
    return {"mean_ms": float(a.mean()), "p95_ms": float(np.percentile(a, 95)), "fps": float(1000.0 / a.mean())}


def bench_sync(frames, mode):
    lmk = make_landmarker(mode)
    ms = []
    for i, f in enumerate(frames):
        im = mp.Image(image_format=mp.ImageFormat.SRGB, data=f)
        t0 = time.perf_counter()
        lmk.detect(im) if mode == vision.RunningMode.IMAGE else lmk.detect_for_video(im, i * 33 + 1)
        ms.append((time.perf_counter() - t0) * 1000)
    return stats(ms[10:])


def bench_live(frames, delegate=None):
    done = []
    sub = {}
    lock = threading.Lock()

    def cb(res, img, t):
        with lock:
            done.append((time.perf_counter() - sub.pop(t, time.perf_counter())) * 1000)

    try:
        o = vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(MODEL), delegate=delegate),
                                         running_mode=vision.RunningMode.LIVE_STREAM, num_hands=2,
                                         min_hand_detection_confidence=0.65, min_hand_presence_confidence=0.65,
                                         min_tracking_confidence=0.55, result_callback=cb)
        lmk = vision.HandLandmarker.create_from_options(o)
    except Exception as e:
        return {"error": str(e)[:120]}
    t_start = time.perf_counter()
    for i, f in enumerate(frames):
        ts = i * 33 + 1
        with lock:
            sub[ts] = time.perf_counter()
        lmk.detect_async(mp.Image(image_format=mp.ImageFormat.SRGB, data=f), ts)
        time.sleep(max(0.0, t_start + (i + 1) / 30.0 - time.perf_counter()))      # paced at 30 fps like a camera
    time.sleep(0.5)
    r = stats(done[10:])
    r["delivered"] = len(done)
    r["submitted"] = len(frames)
    return r


def main():
    frames = make_frames()
    out = {"machine": cv2.getBuildInformation().split("\n")[0][:60] if False else "this machine", "n_frames": N}
    out["IMAGE (blocking)"] = bench_sync(frames, vision.RunningMode.IMAGE)
    out["VIDEO (blocking)"] = bench_sync(frames, vision.RunningMode.VIDEO)
    out["LIVE_STREAM CPU, paced 30 fps"] = bench_live(frames)
    try:
        out["LIVE_STREAM GPU delegate"] = bench_live(frames, BaseOptions.Delegate.GPU)
    except Exception as e:
        out["LIVE_STREAM GPU delegate"] = {"error": str(e)[:120]}
    out["Option A (Web Render TOP plugin)"] = "NOT MEASURED (runs only inside TouchDesigner)"
    json.dump(out, open(ROOT / "bench_results.json", "w"), indent=1)
    print("| Path | mean ms | p95 ms | max fps |\n|---|---|---|---|")
    for k, v in out.items():
        if isinstance(v, dict) and "mean_ms" in v:
            print(f"| {k} | {v['mean_ms']:.1f} | {v['p95_ms']:.1f} | {v['fps']:.0f} |")
        elif isinstance(v, (dict, str)) and k.startswith(("LIVE", "Option")):
            print(f"| {k} | {v if isinstance(v, str) else v.get('error')} | | |")


if __name__ == "__main__":
    main()
