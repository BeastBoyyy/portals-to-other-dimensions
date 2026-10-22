"""Minimal ground-truth fingertip annotator.

    python tools/label_tool.py clips/still.mp4 [--n 200] [--out clips/still.labels.json]

For ~200 random frames the DISPLAY image (mirrored per config.yaml) is shown. Click the four fingertips in this order:
  1 thumb tip of the hand on the LEFT of the screen   2 its index tip
  3 index tip of the hand on the RIGHT of the screen  4 its thumb tip
Keys: n = hands not both visible (skip)   u = undo clicks   q = save and quit.
Output {frame_index: [[x,y]*4]} in display pixels -> `python tools/evaluate.py --jsonl clips/still.jsonl --labels clips/still.labels.json`.
Frame indices refer to the JSONL / video frame order from tools/record_clip.py.
"""
import argparse, json, pathlib, random, sys
import cv2

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from handcore import config as hcfg      # noqa: E402

NAMES = ["left thumb tip", "left index tip", "right index tip", "right thumb tip"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("--n", type=int, default=200); ap.add_argument("--out")
    ap.add_argument("--config", default=str(ROOT / "config.yaml")); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    mirror = hcfg.load(a.config).camera.mirror
    cap = cv2.VideoCapture(a.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = sorted(random.Random(a.seed).sample(range(total), min(a.n, total)))
    out_path = pathlib.Path(a.out or pathlib.Path(a.video).with_suffix(".labels.json"))
    labels = json.load(open(out_path)) if out_path.exists() else {}
    pts = []

    def on_click(ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append([x, y])

    cv2.namedWindow("label"); cv2.setMouseCallback("label", on_click)
    for k, i in enumerate(idx):
        if str(i) in labels:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, bgr = cap.read()
        if not ok:
            continue
        img = cv2.flip(bgr, 1) if mirror else bgr
        pts.clear()
        while True:
            disp = img.copy()
            for j, p in enumerate(pts):
                cv2.circle(disp, tuple(p), 4, (0, 255, 0), -1); cv2.putText(disp, str(j + 1), (p[0] + 6, p[1]), 0, 0.6, (0, 255, 0), 2)
            msg = f"[{k + 1}/{len(idx)}] frame {i}: click {NAMES[len(pts)] if len(pts) < 4 else 'done (any key = next)'}   n=skip u=undo q=quit"
            cv2.putText(disp, msg, (10, 24), 0, 0.6, (255, 255, 255), 2)
            cv2.imshow("label", disp)
            key = cv2.waitKey(20) & 0xFF
            if key == ord("q"):
                json.dump(labels, open(out_path, "w")); print("saved", len(labels), "labels to", out_path); return
            if key == ord("u") and pts:
                pts.pop()
            if key == ord("n"):
                break
            if len(pts) == 4 and key != 255:
                labels[str(i)] = [p[:] for p in pts]; break
    json.dump(labels, open(out_path, "w")); print("saved", len(labels), "labels to", out_path)


if __name__ == "__main__":
    main()
