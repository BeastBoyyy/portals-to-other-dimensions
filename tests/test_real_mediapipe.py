"""Needs the model files (models/*.task) and mediapipe: runs the REAL HandLandmarker on the sample images."""
import pathlib
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
pytest.importorskip("mediapipe")
if not (ROOT / "models" / "hand_landmarker.task").exists():
    pytest.skip("models/hand_landmarker.task missing (see README)", allow_module_level=True)


def test_real_detector_on_sample_images():
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from tools.extract_landmarks import make_landmarker
    lmk = make_landmarker(vision.RunningMode.IMAGE, det=0.65, pres=0.65)

    def n_hands(name):
        bgr = cv2.imread(str(ROOT / "fixtures" / "images" / name))
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return len(lmk.detect(img).hand_landmarks)

    assert [n_hands(n) for n in ("thumb_up.jpg", "victory.jpg", "pointing_up.jpg")] == [1, 1, 1]
    assert n_hands("woman_hands.jpg") == 2
    assert [n_hands(n) for n in ("burger.jpg", "cat.jpg", "portrait.jpg")] == [0, 0, 0]
