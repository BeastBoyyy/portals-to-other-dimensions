"""Camera / lighting setup check (spec 4.2).  python tools/setup_check.py [--camera 0] [--seconds 4] [--lock-exposure]

Reports negotiated resolution + MEASURED fps, mean brightness, clipping, a blur estimate of the hand region (falls back to
the frame centre when no hand is visible) and warns when conditions will hurt tracking. Exit code 1 if any FAIL.
"""
import argparse, pathlib, sys, time
import numpy as np
import cv2

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def hand_roi(rgb, lmk):
    import mediapipe as mp
    r = lmk.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not r.hand_landmarks:
        return None
    pts = np.array([[p.x, p.y] for p in r.hand_landmarks[0]]) * [rgb.shape[1], rgb.shape[0]]
    x0, y0 = np.clip(pts.min(0) - 10, 0, None).astype(int)
    x1, y1 = pts.max(0).astype(int) + 10
    return x0, y0, x1, y1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--lock-exposure", action="store_true", help="try to switch the camera to manual exposure (driver permitting)")
    a = ap.parse_args()
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(a.camera, backend)
    if not cap.isOpened():
        print("FAIL  cannot open camera", a.camera); sys.exit(1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))     # MJPEG is what delivers 720p60 on most webcams
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720); cap.set(cv2.CAP_PROP_FPS, 60)
    if a.lock_exposure:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)                     # DirectShow: 0.25 = manual
        cap.set(cv2.CAP_PROP_EXPOSURE, -7)                            # ~1/128 s on most DirectShow cameras
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    for _ in range(10):
        cap.read()
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"negotiated: {w}x{h} @ {cap.get(cv2.CAP_PROP_FPS):.0f} fps (driver-reported), "
          f"auto_exposure={cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)}, exposure={cap.get(cv2.CAP_PROP_EXPOSURE)}, autofocus={cap.get(cv2.CAP_PROP_AUTOFOCUS)}")
    frames, t0 = [], time.perf_counter()
    while time.perf_counter() - t0 < a.seconds:
        ok, f = cap.read()
        if ok:
            frames.append((time.perf_counter(), f))
    cap.release()
    if len(frames) < 5:
        print("FAIL  camera delivered no frames"); sys.exit(1)
    ts = np.array([t for t, _ in frames])
    fps = (len(ts) - 1) / (ts[-1] - ts[0])
    gray = cv2.cvtColor(frames[-1][1], cv2.COLOR_BGR2GRAY)
    means = [float(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean()) for _, f in frames[::max(1, len(frames) // 20)]]
    clip = float(np.mean(gray >= 250) * 100)
    roi = None
    try:
        from tools.extract_landmarks import make_landmarker
        from mediapipe.tasks.python import vision
        roi = hand_roi(cv2.cvtColor(frames[-1][1], cv2.COLOR_BGR2RGB), make_landmarker(vision.RunningMode.IMAGE, 0.5, 0.5))
    except Exception:
        pass
    x0, y0, x1, y1 = roi if roi else (w // 3, h // 3, 2 * w // 3, 2 * h // 3)
    blur = float(cv2.Laplacian(gray[y0:y1, x0:x1], cv2.CV_64F).var())
    bright = float(np.mean(means)); flicker = float(np.std(means))
    rows = [("measured fps", fps, fps >= 25, ">= 25 (30 ok, 60 better)"),
            ("resolution", w, w >= 960, ">= 960 wide (1280 requested)"),
            ("mean brightness (0-255)", bright, 70 <= bright <= 200, "70..200"),
            ("clipped highlights (%)", clip, clip <= 5, "<= 5 % (backlight / window?)"),
            ("brightness pulsing (std over time)", flicker, flicker <= 3.0, "<= 3 (auto-exposure hunting if higher)"),
            (f"blur (Laplacian var, {'hand' if roi else 'centre'} region)", blur, blur >= 60, ">= 60 (motion blur / focus / noise)")]
    bad = 0
    for name, v, ok, need in rows:
        print(f"{'PASS' if ok else 'WARN'}  {name:46s} {v:8.1f}   need {need}")
        bad += (not ok)
    if bad:
        print("\nFixes: face the main light (window in front, not behind you); lock exposure/white-balance/focus in the camera app or use --lock-exposure;"
              " pick a short shutter (~1/120 s) and add light instead of raising gain; use MJPEG 720p; avoid USB hubs.")
    sys.exit(1 if rows[0][2] is False else 0)


if __name__ == "__main__":
    main()
