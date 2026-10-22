"""Run HandCore + the whole-hand pose logic over a video and print what the portal would do at each moment.

    python tools/analyze_clip.py "clip.mp4" [--mirror] [--crop-top 20 --crop-bottom 650]

Columns: portal valid, openness (0 pinched .. 1 open), mean palm twist (deg), the effect the twist dial is on, hand speed.
Used to check the open/pinch and twist rules against the reference video.
"""
import argparse, pathlib, sys
import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import mediapipe as mp                                                            # noqa: E402
from mediapipe.tasks.python import vision                                          # noqa: E402
from handcore import config as hcfg                                                # noqa: E402
from handcore.pipeline import HandCore                                             # noqa: E402
from handcore.types import Frame, Hand                                             # noqa: E402
from handcore.view import TwistDial                                             # noqa: E402
from tools.extract_landmarks import make_landmarker, result_to_hands               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--crop-top", type=int, default=0); ap.add_argument("--crop-bottom", type=int, default=0)
    ap.add_argument("--step", type=float, default=0.25)
    a = ap.parse_args()
    cfg = hcfg.load(str(ROOT / "config.yaml"))
    cfg.camera.mirror = a.mirror
    cfg.gates.person_mask = False
    core = HandCore(cfg)
    g = cfg.gesture
    dial = TwistDial(g.dial_effects, g.dial_step_deg, g.dial_commit_s, g.dial_base_tau_s, g.dial_ease_s)
    t_prev = 0.0
    lmk = make_landmarker(vision.RunningMode.VIDEO, 0.5, 0.5, 0.5)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i, next_print = 0, 0.0
    print(" t     portal  open   twist  dial effect        speed  hands")
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if a.crop_bottom:
            bgr = bgr[a.crop_top:a.crop_bottom]
        t = i / fps
        r = lmk.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), int(t * 1000) + 1)
        hands = [Hand(np.array(h["lm"]), np.array(h["world"]), h["label"], h["score"]) for h in result_to_hands(r)]
        st = core.update(Frame(t, hands))
        if st is not None:
            dial.update(st.twist, max(t - t_prev, 1e-3), active=st.openness > 0.5)
        t_prev = t
        if t >= next_print and st is not None:
            next_print += a.step
            top = g.dial_effects[dial.k]
            print(f"{t:5.2f}  {'valid' if st.valid else '-----'}  {st.openness:4.2f}  {st.twist:5.1f}  {top:15s}  {st.speed:4.2f}   {st.n_hands}")
        i += 1


if __name__ == "__main__":
    main()
