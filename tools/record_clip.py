"""Record a test clip: video + raw landmarks (JSONL) with capture timestamps.

    python tools/record_clip.py still --seconds 45            # -> clips/still.mp4 + clips/still.jsonl
Clip names to record (spec 9.1): still slow fast crossing reentry nohands onehand lowlight backlit tilt occluded
Then:  python tools/evaluate.py --jsonl clips/still.jsonl
       python tools/label_tool.py clips/still.mp4 clips/still.jsonl     # ~200 hand-labelled frames for accuracy
Landmarks are stored RAW (un-mirrored, as the camera delivered them); handcore mirrors on ingest per config.yaml.
"""
import argparse, json, pathlib, sys, time
import cv2

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import mediapipe as mp                                                    # noqa: E402
from mediapipe.tasks.python import vision                                  # noqa: E402
from tools.extract_landmarks import make_landmarker, result_to_hands      # noqa: E402

PROMPTS = {"still": "hold both hands still in the frame gesture", "slow": "move the frame slowly", "fast": "move the frame FAST",
           "crossing": "cross hands past each other and back", "reentry": "hands out of view for ~1 s, then back, repeat",
           "nohands": "NO hands: face, poster, wall card, clothes only", "onehand": "one hand only", "lowlight": "dim the room",
           "backlit": "sit with the window behind you", "tilt": "tilt / rotate the frame quickly", "occluded": "overlap your fingers"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name"); ap.add_argument("--seconds", type=float, default=45); ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "clips"))
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(exist_ok=True)
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(a.camera, backend)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720); cap.set(cv2.CAP_PROP_FPS, 60)
    lmk = make_landmarker(vision.RunningMode.VIDEO)
    vw, fj = None, open(out / f"{a.name}.jsonl", "w")
    print(f"Recording '{a.name}' for {a.seconds:.0f} s: {PROMPTS.get(a.name, '')}.  Starts in 3 s ...")
    time.sleep(3)
    t0, last_ms = time.perf_counter(), -1
    while time.perf_counter() - t0 < a.seconds:
        ok, bgr = cap.read()
        if not ok:
            continue
        t = time.perf_counter() - t0
        h, w = bgr.shape[:2]
        if vw is None:
            vw = cv2.VideoWriter(str(out / f"{a.name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
        ms = max(int(t * 1000), last_ms + 1); last_ms = ms
        r = lmk.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), ms)
        vw.write(bgr)
        fj.write(json.dumps({"t": t, "w": w, "h": h, "hands": result_to_hands(r)}) + "\n")
        cv2.imshow("recording (q aborts)", cv2.flip(bgr, 1))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release(); vw and vw.release(); fj.close(); cv2.destroyAllWindows()
    print("saved", out / f"{a.name}.mp4", "and", out / f"{a.name}.jsonl")


if __name__ == "__main__":
    main()
