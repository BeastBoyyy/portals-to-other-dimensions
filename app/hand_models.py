"""GPU hand-landmark refinement models (ONNX Runtime, DirectML on the NVIDIA GPU) working on FULL-RESOLUTION rotated crops.

Why: MediaPipe Tasks (CPU) sees a downscaled frame and gives one noisy estimate per frame. Here the landmark model runs on a crop cut from
the full camera frame (more pixels on the fingertips) and, optionally, on several slightly different crops that are averaged (test-time
augmentation), which cuts random error by ~1/sqrt(K). Two interchangeable models:
  * MpLandmark  : MediaPipe hand landmark (TFLite -> ONNX, OpenCV zoo 2023feb), 224x224, regresses 21 (x,y,z) + world coords
  * RtmPoseHand : RTMPose-m hand (OpenMMLab family; community distilled ONNX), 256x256, SimCC heads decoded with soft-argmax (sub-bin precision)
Both use the 21-point MediaPipe / COCO-WholeBody hand order (0 wrist, 4 thumb tip, 8 index tip, 12 middle tip, 16 ring tip, 20 pinky tip).
"""
from __future__ import annotations
import pathlib
import numpy as np
import cv2

ONNX = pathlib.Path(__file__).resolve().parent.parent / "models" / "onnx"


def make_session(path, prefer="dml"):
    import onnxruntime as ort
    avail = ort.get_available_providers()
    order = (["DmlExecutionProvider"] if prefer == "dml" and "DmlExecutionProvider" in avail else []) + ["CPUExecutionProvider"]
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if "DmlExecutionProvider" in order:
        so.enable_mem_pattern = False                    # required by the DirectML EP
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), so, providers=order)


# ---- rotated square ROI around a hand (the same idea as MediaPipe's, computed from landmarks) ---------------------------------------------
def hand_roi(lm_px, scale=2.0, shift_up=0.10):
    """lm_px (21,2) pixel landmarks -> (centre(2,), side, angle_rad). The hand is rotated so wrist->middle-MCP points up."""
    v = lm_px[9] - lm_px[0]
    ang = np.arctan2(v[0], -v[1])                          # 0 = already pointing up; positive = tilted clockwise
    c, s = np.cos(-ang), np.sin(-ang)
    R = np.array([[c, -s], [s, c]])
    rot = (lm_px - lm_px.mean(0)) @ R.T
    lo, hi = rot.min(0), rot.max(0)
    side = max(hi[0] - lo[0], hi[1] - lo[1]) * scale
    centre_rot = (lo + hi) / 2 + np.array([0.0, -shift_up * side])
    centre = lm_px.mean(0) + R.T @ centre_rot
    return centre, float(side), float(ang)


def roi_affine(centre, side, ang, out):
    """2x3 matrix mapping IMAGE pixels -> crop pixels (out x out), rotating by -ang about the ROI centre."""
    s = out / side
    c, sn = np.cos(-ang) * s, np.sin(-ang) * s
    A = np.array([[c, -sn, 0.0], [sn, c, 0.0]])
    A[:, 2] = np.array([out / 2.0, out / 2.0]) - A[:, :2] @ centre
    return A


def invert_affine(A):
    return cv2.invertAffineTransform(A)


def apply_affine(M, pts):
    return pts @ M[:, :2].T + M[:, 2]


def perturb_roi(centre, side, ang, rng, shift=0.08, scale=(0.92, 1.15), rot_deg=15.0):
    """A random nearby ROI (for test-time augmentation / the equivariance precision test)."""
    c = centre + rng.uniform(-shift, shift, 2) * side
    return c, side * rng.uniform(*scale), ang + np.radians(rng.uniform(-rot_deg, rot_deg))


# ---- models ---------------------------------------------------------------------------------------------------------------------------------
class MpLandmark:
    size = 224
    name = "mediapipe-onnx"

    def __init__(self, prefer="dml"):
        self.sess = make_session(ONNX / "handpose_estimation_mediapipe_2023feb.onnx", prefer)
        self.inp = self.sess.get_inputs()[0].name

    def run(self, crops):
        """crops: list of (224,224,3) uint8 RGB -> list of dict(lm (21,2) crop px, conf, z (21,))"""
        out = []
        for c in crops:
            blob = (c.astype(np.float32) / 255.0)[None]
            lm, conf, hand, world = self.sess.run(None, {self.inp: blob})
            p = lm.reshape(21, 3)
            out.append({"lm": p[:, :2].copy(), "z": p[:, 2].copy(), "conf": float(conf.ravel()[0]), "world": world.reshape(21, 3)})
        return out


class MpTflite(MpLandmark):
    """The EXACT landmark network MediaPipe Tasks uses (hand_landmarks_detector.tflite, unpacked from models/hand_landmarker.task), run through
    LiteRT/XNNPACK on the CPU on full-resolution crops. Same weights as Tasks, so no accuracy is traded for the speed-up."""
    name = "mediapipe-tflite"

    def __init__(self, prefer="cpu", threads=2):
        from ai_edge_litert.interpreter import Interpreter
        self.it = Interpreter(str(ONNX.parent / "tflite" / "hand_landmarks_detector.tflite"), num_threads=threads)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]["index"]
        self.outs = {d["name"]: d["index"] for d in self.it.get_output_details()}

    def run(self, crops):
        out = []
        for c in crops:
            self.it.set_tensor(self.inp, (c.astype(np.float32) / 255.0)[None])
            self.it.invoke()
            g = lambda n: self.it.get_tensor(self.outs[n])
            p = g("Identity").reshape(21, 3)
            out.append({"lm": p[:, :2].copy(), "z": p[:, 2].copy(), "conf": float(g("Identity_1").ravel()[0]), "world": g("Identity_3").reshape(21, 3)})
        return out


class RtmPoseHand:
    size = 256
    name = "rtmpose-m-hand"
    MEAN = np.array([123.675, 116.28, 103.53], np.float32)
    STD = np.array([58.395, 57.12, 57.375], np.float32)

    def __init__(self, prefer="dml"):
        self.sess = make_session(ONNX / "rtmpose-m-distill-256x256.onnx", prefer)
        self.inp = self.sess.get_inputs()[0].name

    @staticmethod
    def _soft_argmax(sim, half=4):
        """sim (N,21,bins) -> sub-bin peak via a local softmax-weighted mean around the argmax (bins are 0.5 px of the 256 crop)."""
        idx = sim.argmax(-1)
        out = np.empty(idx.shape, np.float32)
        for n in range(idx.shape[0]):
            for k in range(idx.shape[1]):
                i = idx[n, k]
                lo, hi = max(0, i - half), min(sim.shape[-1], i + half + 1)
                w = np.exp(sim[n, k, lo:hi] - sim[n, k, i])
                out[n, k] = (np.arange(lo, hi) * w).sum() / w.sum()
        return out

    def run(self, crops):
        x = (np.stack(crops).astype(np.float32) - self.MEAN) / self.STD
        sx, sy = self.sess.run(None, {self.inp: np.ascontiguousarray(x.transpose(0, 3, 1, 2))})
        ax, ay = self._soft_argmax(sx) / 2.0, self._soft_argmax(sy) / 2.0
        conf = np.minimum(sx.max(-1), sy.max(-1))
        return [{"lm": np.stack([ax[i], ay[i]], 1), "conf": float(conf[i].mean())} for i in range(len(crops))]


def crop_rgb(rgb, centre, side, ang, out):
    A = roi_affine(centre, side, ang, out)
    return cv2.warpAffine(rgb, A, (out, out), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)), A


def landmarks_in_image(model, result, A):
    return apply_affine(invert_affine(A), result["lm"])
