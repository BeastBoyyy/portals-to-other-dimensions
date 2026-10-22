"""Download the MediaPipe model files the app needs into models/ (they are not stored in the git repository).

    python tools/get_models.py [--out models]
Fetches hand_landmarker.task and selfie_segmenter.tflite from Google's public MediaPipe model bucket, and unpacks the landmark network
(hand_landmarks_detector.tflite) from the .task file (a .task file is a zip archive) into <out>/tflite/. Safe to re-run: existing files are kept.
"""
import argparse, pathlib, urllib.request, zipfile

BASE = "https://storage.googleapis.com/mediapipe-models"
FILES = {
    "hand_landmarker.task": f"{BASE}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "selfie_segmenter.tflite": f"{BASE}/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(pathlib.Path(__file__).resolve().parent.parent / "models"))
    out = pathlib.Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dst = out / name
        if dst.exists() and dst.stat().st_size > 1000:
            print("have", dst)
            continue
        print("downloading", url)
        tmp = dst.with_suffix(dst.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dst)
        print("saved", dst, f"({dst.stat().st_size / 1e6:.1f} MB)")
    tfl = out / "tflite"
    tfl.mkdir(exist_ok=True)
    with zipfile.ZipFile(out / "hand_landmarker.task") as z:
        for member in ("hand_landmarks_detector.tflite", "hand_detector.tflite"):
            if not (tfl / member).exists():
                z.extract(member, tfl)
                print("unpacked", tfl / member)
    print("models ready in", out)


if __name__ == "__main__":
    main()
