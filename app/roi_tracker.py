"""Full-resolution hand tracker: MediaPipe's own landmark network run through ONNX Runtime (DirectML GPU) on rotated crops cut from the
FULL camera frame, one crop track per hand (the same crop->landmarks->next crop loop MediaPipe uses internally, but at full resolution, on the GPU,
with an optional test-time-augmentation average that divides the random landmark error by ~sqrt(K)).

  RoiTracker.step(rgb_full)   -> list[TrackedHand]  advance every track by one frame
  RoiTracker.seed(lm_px, ..)  -> start a track from a detection (the MediaPipe Tasks palm detector, run elsewhere, finds hands; this follows them)

No threads here (so it can be benchmarked and tested synchronously); app/gpu_landmarker.py puts it behind the Landmarker interface.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from . import hand_models as hm

# small fixed ROI perturbations (dx, dy in side units, scale, rotation deg) for test-time augmentation; entry 0 is the nominal crop
TTA = [(0.0, 0.0, 1.0, 0.0), (0.03, -0.02, 1.05, 4.0), (-0.03, 0.02, 0.95, -4.0), (0.02, 0.03, 1.02, -3.0), (-0.02, -0.03, 0.98, 3.0)]


@dataclass
class TrackedHand:
    lm_px: np.ndarray                  # (21,2) image px
    z: np.ndarray                      # (21,) relative depth in image px (same scale as x / y)
    world: np.ndarray                  # (21,3) metres, image orientation (rotated back out of the crop frame)
    label: str
    score: float


@dataclass
class _Track:
    lm_px: np.ndarray
    label: str
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    score: float = 1.0
    age: int = 0
    lost: int = 0
    hand: object = None


class RoiTracker:
    def __init__(self, model=None, prefer="dml", tta=1, min_conf=0.5, max_tracks=2, patience=2):
        self.net = model or hm.MpLandmark(prefer)
        self.tta, self.min_conf, self.max_tracks, self.patience = max(1, int(tta)), min_conf, max_tracks, patience
        self.tracks: list[_Track] = []
        self.n_calls = 0

    # -- seeding -----------------------------------------------------------------------------------------------------------------------
    def seed(self, lm_px, label="Left"):
        """Start (or re-aim) a track from detected landmarks (21,2) in image px. Returns True if a new track was created."""
        c = lm_px[[0, 5, 9, 13, 17]].mean(0)
        palm = float(np.linalg.norm(lm_px[5] - lm_px[17]))
        for t in self.tracks:                                                   # already following this hand
            tc, tp = t.lm_px[[0, 5, 9, 13, 17]].mean(0), float(np.linalg.norm(t.lm_px[5] - t.lm_px[17]))
            if np.linalg.norm(tc - c) < 0.9 * max(palm, tp):
                return False
        if len(self.tracks) >= self.max_tracks:
            return False
        self.tracks.append(_Track(lm_px.copy(), label))
        return True

    def reset(self):
        self.tracks = []

    # -- one frame ---------------------------------------------------------------------------------------------------------------------
    def step(self, rgb):
        H, W = rgb.shape[:2]
        keep = []
        for tr in self.tracks:
            c, side, ang = hm.hand_roi(tr.lm_px)
            c = c + tr.vel                                                     # constant-velocity guess where the hand is now
            rois = [(c + np.array([dx, dy]) * side, side * sc, ang + np.radians(rot)) for dx, dy, sc, rot in TTA[:self.tta]]
            crops, As = zip(*[hm.crop_rgb(rgb, rc, rs, ra, self.net.size) for rc, rs, ra in rois])
            res = self.net.run(list(crops))
            self.n_calls += len(crops)
            good = [(r, A, ra, rs) for r, A, (_, rs, ra) in zip(res, As, rois) if r["conf"] >= self.min_conf]
            if not good:
                tr.lost += 1
                if tr.lost >= self.patience:
                    continue                                                    # gone: drop the track (the detector will re-seed it)
                keep.append(tr)
                continue
            pts = np.mean([hm.landmarks_in_image(self.net, r, A) for r, A, _, _ in good], axis=0)
            z = np.mean([r["z"] * rs / self.net.size for r, _, _, rs in good], axis=0)
            world = np.mean([self._world_to_image(r["world"], ra) for r, _, ra, _ in good], axis=0)
            conf = float(np.mean([r["conf"] for r, _, _, _ in good]))
            bb = pts.max(0) - pts.min(0)
            if not np.isfinite(pts).all() or bb.max() < 12 or bb.max() > 1.5 * max(W, H):
                continue
            old_c = tr.lm_px[[0, 5, 9, 13, 17]].mean(0)
            new_c = pts[[0, 5, 9, 13, 17]].mean(0)
            tr.vel = 0.5 * tr.vel + 0.5 * (new_c - old_c)                      # px / frame, lightly smoothed (only steers the next crop)
            tr.lm_px, tr.score, tr.lost, tr.age = pts, conf, 0, tr.age + 1
            keep.append(tr)
            tr.hand = TrackedHand(pts, z, world, tr.label, conf)
        self.tracks = self._dedupe(keep)
        return [t.hand for t in self.tracks if t.lost == 0 and t.hand is not None]

    @staticmethod
    def _world_to_image(world, ang):
        """The network works in the rotated crop frame; rotate its metric (x, y) back into image orientation (z is unchanged)."""
        c, s = np.cos(ang), np.sin(ang)
        w = world.copy()
        w[:, 0], w[:, 1] = c * world[:, 0] - s * world[:, 1], s * world[:, 0] + c * world[:, 1]
        return w

    def _dedupe(self, tracks):
        """Two tracks that converged on the same hand: keep the better one."""
        keep = []
        for t in sorted(tracks, key=lambda t: -t.score):
            c, p = t.lm_px[[0, 5, 9, 13, 17]].mean(0), float(np.linalg.norm(t.lm_px[5] - t.lm_px[17]))
            if all(np.linalg.norm(c - k.lm_px[[0, 5, 9, 13, 17]].mean(0)) > 0.6 * p for k in keep):
                keep.append(t)
        return keep
