"""Drop-in replacement for adapters.mediapipe_sidecar.Landmarker that tracks at the CAMERA rate on FULL-resolution crops.

  * a worker thread runs RoiTracker.step() on every camera frame (MediaPipe's own landmark network, 2-4 crops per frame, ~10-15 ms) and emits one
    Frame per camera frame to on_frame -- no dropped frames, no 640 px downscale;
  * the stock MediaPipe Tasks HandLandmarker (palm detector) runs on its own thread at whatever rate the CPU allows, only to FIND hands: its detections
    seed new crop tracks. When both hands are being followed it is nearly idle (a sanity pass every few frames).
"""
from __future__ import annotations
import threading, time
import numpy as np
from adapters.mediapipe_sidecar import Landmarker
from handcore.types import Frame, Hand
from . import hand_models as hm
from .roi_tracker import RoiTracker


class RoiLandmarker:
    every_frame = True                                     # the camera thread should submit every frame, not every other one

    def __init__(self, cfg, on_frame, tta=2, threads=4, sanity_every=6):
        self.on_frame, self.sanity_every = on_frame, sanity_every
        self.tracker = RoiTracker(hm.MpTflite(threads=threads), tta=tta, min_conf=0.35, patience=4)
        self.seeder = Landmarker(cfg, self._on_seed, use_mask=False)
        self.delegate = f"ROI/{self.tracker.net.name} tta={tta} + Tasks seeds"
        self.seg, self.mask = None, None
        self.lock = threading.Lock()
        self._latest, self._event, self._stop = None, threading.Event(), False
        self._n, self._shape = 0, (720, 1280)
        self.dropped, self.infer_ms, self.done_times, self.last_t = 0, 0.0, [], -1.0
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, small, t, full=None):
        """small: 640 px frame for the detector, full: full-resolution RGB for the crops. Latest wins; an unprocessed older frame counts as dropped."""
        full = small if full is None else full
        self._n += 1
        if len(self.tracker.tracks) < 2 or self._n % self.sanity_every == 0:
            self.seeder.submit(small, t)
        if self._latest is not None:
            self.dropped += 1
        self._latest = (full, t)
        self._event.set()
        return True

    def _on_seed(self, frame):
        H, W = self._shape
        with self.lock:
            for h in frame.hands:
                self.tracker.seed(h.lm[:, :2] * [W, H], h.label)

    def _run(self):
        while not self._stop:
            self._event.wait(0.5)
            self._event.clear()
            item, self._latest = self._latest, None
            if item is None:
                continue
            full, t = item
            if t <= self.last_t:
                continue
            self.last_t = t
            H, W = full.shape[:2]
            self._shape = (H, W)
            t0 = time.perf_counter()
            with self.lock:
                hands = self.tracker.step(full)
            out = [Hand(np.column_stack([h.lm_px[:, 0] / W, h.lm_px[:, 1] / H, h.z / W]), h.world, h.label, h.score) for h in hands]
            now = time.perf_counter()
            self.infer_ms += 0.1 * ((now - t0) * 1000.0 - self.infer_ms)
            self.done_times = (self.done_times + [now])[-30:]
            self.on_frame(Frame(t, out, None))

    @property
    def fps(self):
        d = self.done_times
        return (len(d) - 1) / (d[-1] - d[0]) if len(d) > 2 and d[-1] > d[0] else 0.0
