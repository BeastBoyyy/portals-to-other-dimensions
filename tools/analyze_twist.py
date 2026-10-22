"""Per-frame hand features on a video: openness (fingertip spread), twist (palm normal vs camera), box roll. Used to calibrate the rules
from the reference clip.   python tools/analyze_twist.py clip.mp4 [--crop-top 20 --crop-bottom 650]"""
import argparse, pathlib, sys
import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import mediapipe as mp                                                  # noqa: E402
from mediapipe.tasks.python import vision                                # noqa: E402
from tools.extract_landmarks import make_landmarker, result_to_hands     # noqa: E402

TIPS = (4, 8, 12, 16, 20)


def features(h, W, H):
    lm = np.array(h["lm"]); w = np.array(h["world"])
    px = lm[:, :2] * [W, H]
    palm = max(np.linalg.norm(px[5] - px[17]), np.linalg.norm(px[0] - px[9]))
    tips = px[list(TIPS)]
    spread = float(np.mean(np.linalg.norm(tips - tips.mean(0), axis=1)) / max(palm, 1e-6))
    n = np.cross(w[5] - w[0], w[17] - w[0]); n /= max(np.linalg.norm(n), 1e-9)
    twist = float(np.degrees(np.arccos(np.clip(abs(n[2]), 0, 1))))         # 0 = palm/back faces the camera, 90 = edge-on
    return spread, twist, px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("--crop-top", type=int, default=0); ap.add_argument("--crop-bottom", type=int, default=0)
    ap.add_argument("--step", type=float, default=0.25)
    a = ap.parse_args()
    lmk = make_landmarker(vision.RunningMode.VIDEO, 0.4, 0.4, 0.4)
    cap = cv2.VideoCapture(a.video); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i, nxt = 0, 0.0
    print("  t   | left hand: spread twist | right hand: spread twist | roll(deg between hand centres)")
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if a.crop_bottom:
            bgr = bgr[a.crop_top:a.crop_bottom]
        H, W = bgr.shape[:2]; t = i / fps; i += 1
        r = lmk.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), int(t * 1000) + 1)
        if t < nxt:
            continue
        nxt += a.step
        hs = sorted([features(h, W, H) + (np.array(h["lm"])[:, :2].mean(0),) for h in result_to_hands(r)], key=lambda f: f[3][0])
        txt = "  |  ".join(f"{f[0]:.2f} {f[1]:5.1f}" for f in hs)
        roll = ""
        if len(hs) == 2:
            d = (hs[1][3] - hs[0][3]) * [W, H]
            roll = f"{np.degrees(np.arctan2(d[1], d[0])):6.1f}"
        print(f"{t:5.2f} | {txt:34s} | {roll}")


if __name__ == "__main__":
    main()
