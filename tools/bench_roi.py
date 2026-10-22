"""Full-res GPU ROI tracker vs MediaPipe Tasks on a video: speed, frame-to-frame smoothness, agreement, and track continuity.

    python tools/bench_roi.py "clip.mp4" [--crop-top 20 --crop-bottom 650] [--frames 300] [--tta 1 3]
Smoothness = mean |2nd difference| of the fingertip trajectories (px / frame^2): jitter AND real acceleration, so only the ratio between engines means anything.
"""
import argparse, pathlib, sys, time
import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import mediapipe as mp                                          # noqa: E402
from mediapipe.tasks.python import vision                       # noqa: E402
from app import hand_models as hm                              # noqa: E402
from app.roi_tracker import RoiTracker                          # noqa: E402
from tools.extract_landmarks import make_landmarker             # noqa: E402

TIPS = [4, 8, 12, 16, 20]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("--crop-top", type=int, default=0); ap.add_argument("--crop-bottom", type=int, default=0)
    ap.add_argument("--frames", type=int, default=300); ap.add_argument("--tta", type=int, nargs="+", default=[1, 3]); ap.add_argument("--min-conf", type=float, default=0.5); ap.add_argument("--patience", type=int, default=2)
    a = ap.parse_args()
    cap = cv2.VideoCapture(a.video)
    frames = []
    while len(frames) < a.frames:
        ok, bgr = cap.read()
        if not ok:
            break
        if a.crop_bottom:
            bgr = bgr[a.crop_top:a.crop_bottom]
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    H, W = frames[0].shape[:2]
    print(f"{len(frames)} frames {W}x{H}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    lmk = make_landmarker(vision.RunningMode.VIDEO, 0.5, 0.5, 0.5)
    ref, t_tasks = [], 0.0
    for i, rgb in enumerate(frames):
        small = cv2.resize(rgb, (640, int(640 * H / W)), interpolation=cv2.INTER_AREA)
        t0 = time.perf_counter()
        r = lmk.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=small), int(i * 1000 / fps) + 1)
        t_tasks += time.perf_counter() - t0
        ref.append([(np.array([[p.x * W, p.y * H] for p in lm]), hd[0].category_name) for lm, hd in zip(r.hand_landmarks, r.handedness)])
    print(f"Tasks (CPU, 640 px): {1000 * t_tasks / len(frames):.1f} ms / frame, hands/frame {np.mean([len(x) for x in ref]):.2f}")

    def smooth(series):
        s = np.array(series)                                   # (T, 5, 2)
        return float(np.mean(np.linalg.norm(s[2:] - 2 * s[1:-1] + s[:-2], axis=-1)))

    for k in a.tta:
        trk = RoiTracker(hm.MpTflite(threads=4), tta=k, min_conf=a.min_conf, patience=a.patience)
        out, t_roi, seeds = [], 0.0, 0
        for i, rgb in enumerate(frames):
            t0 = time.perf_counter()
            hs = trk.step(rgb)
            t_roi += time.perf_counter() - t0
            if len(trk.tracks) < 2:                            # the detector (Tasks, runs on its own thread in the app) re-seeds lost hands
                for lm, label in ref[i]:
                    seeds += trk.seed(lm, label)
            out.append(hs)
        n = [len(h) for h in out]
        # agreement with Tasks on frames where both see the same number of hands; per-hand nearest match
        diffs, sm_roi, sm_ref = [], [], []
        for i in range(1, len(frames)):
            for h in out[i]:
                if not ref[i]:
                    continue
                m = min(ref[i], key=lambda r: np.linalg.norm(r[0][0] - h.lm_px[0]))
                palm = np.linalg.norm(m[0][5] - m[0][17])
                if np.linalg.norm(m[0][0] - h.lm_px[0]) < palm:
                    diffs.append(np.linalg.norm(h.lm_px[TIPS] - m[0][TIPS], axis=1) / palm * 100)
        # smoothness on a single continuously-tracked hand (index/hand 0 of consecutive frames with exactly one hand in both)
        a_roi = [o[0].lm_px[TIPS] for o, p in zip(out[1:], out[:-1]) if len(o) == 1 and len(p) == 1]
        a_ref = [r[0][0][TIPS] for r, p in zip(ref[1:], ref[:-1]) if len(r) == 1 and len(p) == 1]
        d = np.array(diffs)
        print(f"ROI tta={k}: {1000 * t_roi / len(frames):.1f} ms / frame ({trk.n_calls / len(frames):.1f} net calls), hands/frame {np.mean(n):.2f}, "
              f"reseeds {seeds}, tip disagreement vs Tasks: median {np.median(d):.1f}% p90 {np.percentile(d, 90):.1f}% of palm width"
              + (f", smoothness ROI {smooth(a_roi):.2f} vs Tasks {smooth(a_ref):.2f} px/frame^2" if len(a_roi) > 20 and len(a_ref) > 20 else ""))


if __name__ == "__main__":
    main()
