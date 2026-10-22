"""Label-free precision + speed benchmark of hand-landmark models on REAL hand images.

    python tools/bench_models.py "clip.mp4" [--crop-top 20 --crop-bottom 650] [--n 60] [--k 16]

Precision (random error) is measured by EQUIVARIANCE: every hand is re-cropped K times with random shift / scale / rotation, the model is run on each
crop, the points are mapped back to the image, and the per-landmark scatter (std, px) is reported. A model that is precise returns the same point whatever
the crop. (Systematic bias is not measured here: only labelled data can do that.)  Speed = time per hand for K crops, on the GPU (DirectML) and CPU.
"""
import argparse, pathlib, sys, time
import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import mediapipe as mp                                                         # noqa: E402
from mediapipe.tasks.python import vision                                       # noqa: E402
from app import hand_models as hm                                               # noqa: E402
from tools.extract_landmarks import make_landmarker                             # noqa: E402

TIPS = {"thumb tip": 4, "index tip": 8, "middle tip": 12, "ring tip": 16, "pinky tip": 20}


def collect(video, top, bottom, n, step):
    lmk = make_landmarker(vision.RunningMode.VIDEO, 0.6, 0.6, 0.5)
    cap = cv2.VideoCapture(video); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    items, i = [], 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        i += 1
        if bottom:
            bgr = bgr[top:bottom]
        if i % step:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        r = lmk.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(i * 1000 / fps) + 1)
        H, W = rgb.shape[:2]
        for lm, hd in zip(r.hand_landmarks, r.handedness):
            if hd[0].score > 0.9:
                items.append((rgb, np.array([[p.x * W, p.y * H] for p in lm])))
    keep = np.linspace(0, len(items) - 1, min(n, len(items))).astype(int)          # spread evenly over the whole clip
    return [items[k] for k in keep]


def run_model(model, rgb, rois):
    crops, As = zip(*[hm.crop_rgb(rgb, c, s, a, model.size) for c, s, a in rois])
    res = model.run(list(crops))
    return np.stack([hm.landmarks_in_image(model, r, A) for r, A in zip(res, As)]), res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("--crop-top", type=int, default=0); ap.add_argument("--crop-bottom", type=int, default=0)
    ap.add_argument("--n", type=int, default=60); ap.add_argument("--k", type=int, default=16); ap.add_argument("--step", type=int, default=3)
    a = ap.parse_args()
    items = collect(a.video, a.crop_top, a.crop_bottom, a.n, a.step)
    print(f"{len(items)} real hand instances; K={a.k} random crops each\n")
    rng = np.random.default_rng(0)
    models = {"mediapipe-onnx": hm.MpLandmark, "rtmpose-m-hand": hm.RtmPoseHand}
    table, means = {}, {}
    for name, cls in models.items():
        for prefer in ("dml", "cpu"):
            m = cls(prefer)
            hm_rng = np.random.default_rng(1)
            # warm up
            rgb, seed = items[0]
            run_model(m, rgb, [hm.hand_roi(seed)] * 2)
            scat, t_tot, tip_means = [], 0.0, []
            for rgb, seed in items:
                c, s, ang = hm.hand_roi(seed)
                rois = [(c, s, ang)] + [hm.perturb_roi(c, s, ang, hm_rng) for _ in range(a.k - 1)]
                t0 = time.perf_counter()
                pts, res = run_model(m, rgb, rois)
                t_tot += time.perf_counter() - t0
                palm = np.linalg.norm(seed[5] - seed[17])
                scat.append(pts.std(0) / palm * 100.0)                                # (21,2): % of palm width, per axis
                tip_means.append(pts.mean(0))
            S = np.array(scat)                                                         # (N,21,2)
            key = f"{name} [{prefer.upper()}]"
            table[key] = {"scatter": np.hypot(S[..., 0], S[..., 1]), "ms": 1000 * t_tot / len(items)}
            if prefer == "dml":
                means[name] = np.array(tip_means)
    print("Random error (scatter across crops, % of palm width; lower = more precise) and cost per hand for K crops")
    print(f"{'model':28s} {'thumb':>7s} {'index':>7s} {'middle':>7s} {'ring':>7s} {'pinky':>7s}  {'all 21':>7s}   {'ms / hand (K crops)':>20s}")
    for key, v in table.items():
        sc = v["scatter"]
        tips = [np.median(sc[:, i]) for i in TIPS.values()]
        print(f"{key:28s} " + " ".join(f"{t:7.2f}" for t in tips) + f"  {np.median(sc):7.2f}   {v['ms']:20.1f}")
    names = list(means)
    if len(names) == 2:
        palm = np.array([np.linalg.norm(s[5] - s[17]) for _, s in items])
        d = np.linalg.norm(means[names[0]] - means[names[1]], axis=2) / palm[:, None] * 100
        print("\nDisagreement between the two models' crop-averaged landmarks (% of palm width): "
              + "  ".join(f"{k} {np.median(d[:, i]):.2f}" for k, i in TIPS.items()))
    print("\nAgreement of each model's nominal-crop landmarks with MediaPipe Tasks (% of palm width, median over all 21): see below")
    for name, cls in models.items():
        m = cls("dml"); errs = []
        for rgb, seed in items:
            c, s, ang = hm.hand_roi(seed)
            pts, _ = run_model(m, rgb, [(c, s, ang)])
            errs.append(np.median(np.linalg.norm(pts[0] - seed, axis=1)) / np.linalg.norm(seed[5] - seed[17]) * 100)
        print(f"  {name:18s} median {np.median(errs):.1f} %  p90 {np.percentile(errs, 90):.1f} %")
    print("\nIdeal single-crop vs averaged-K: the table above is single-crop random error; averaging K crops divides it by about sqrt(K).")


if __name__ == "__main__":
    main()
