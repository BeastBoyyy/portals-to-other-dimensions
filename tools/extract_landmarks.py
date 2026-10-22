"""Run MediaPipe HandLandmarker over a video or image files and write a landmark JSONL (Phase 1 spike).

Usage: python tools/extract_landmarks.py OUT.jsonl INPUT [INPUT...]   (video: VIDEO mode, images: IMAGE mode)
JSONL line: {"t": sec, "w": W, "h": H, "src": name, "hands": [{"lm": 21x3, "world": 21x3, "label": str, "score": f}]}
"""
import json, sys, time, pathlib
import cv2, mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

MODEL = pathlib.Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def make_landmarker(mode, det=0.65, pres=0.65, trk=0.55, model=MODEL):
    o = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)), running_mode=mode, num_hands=2,
        min_hand_detection_confidence=det, min_hand_presence_confidence=pres, min_tracking_confidence=trk)
    return vision.HandLandmarker.create_from_options(o)


def result_to_hands(r):
    return [{"lm": [[p.x, p.y, p.z] for p in lm], "world": [[p.x, p.y, p.z] for p in w],
             "label": hd[0].category_name, "score": hd[0].score}
            for lm, w, hd in zip(r.hand_landmarks, r.hand_world_landmarks, r.handedness)]


def main(out, inputs):
    n = 0
    with open(out, "w") as fo:
        for inp in inputs:
            p = pathlib.Path(inp)
            if p.suffix.lower() in IMG_EXT:
                lmk = make_landmarker(vision.RunningMode.IMAGE, det=0.5, pres=0.5)
                bgr = cv2.imread(str(p)); h, w = bgr.shape[:2]
                r = lmk.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
                fo.write(json.dumps({"t": float(n), "w": w, "h": h, "src": p.name, "hands": result_to_hands(r)}) + "\n"); n += 1
            else:
                lmk = make_landmarker(vision.RunningMode.VIDEO)
                cap = cv2.VideoCapture(str(p)); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0; i = 0; last_ms = -1
                t0 = time.perf_counter()
                while True:
                    ok, bgr = cap.read()
                    if not ok: break
                    h, w = bgr.shape[:2]; ms = int(i * 1000 / fps)
                    if ms <= last_ms: ms = last_ms + 1          # strictly monotonic
                    last_ms = ms
                    r = lmk.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), ms)
                    fo.write(json.dumps({"t": ms / 1000, "w": w, "h": h, "src": p.name, "hands": result_to_hands(r)}) + "\n")
                    i += 1; n += 1
                print(f"{p.name}: {i} frames, {i/(time.perf_counter()-t0):.1f} fps inference (CPU)")
    print("wrote", n, "records to", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
