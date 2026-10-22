"""Landmark source + HandCore + UDP sender (Option B; with --source camera it is Option C's front half).

    python adapters/mediapipe_sidecar.py                       # frames from TouchDesigner shared memory (default)
    python adapters/mediapipe_sidecar.py --source camera       # sidecar opens the camera itself (preview window)
    python adapters/mediapipe_sidecar.py --source video clip.mp4 --log out.jsonl

HandLandmarker runs in LIVE_STREAM mode (async). Timestamps handed to MediaPipe are strictly monotonic
milliseconds; late/duplicate frames are dropped, never queued. HandCore runs in the result callback, so the
only latency added to the pipeline is MediaPipe inference itself.
"""
from __future__ import annotations
import argparse, json, pathlib, socket, sys, threading, time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from adapters import protocol as P                       # noqa: E402
from handcore import config as hcfg                      # noqa: E402
from handcore.pipeline import HandCore                   # noqa: E402
from handcore.types import Frame, Hand                   # noqa: E402

MASK_W, MASK_H = 256, 144


class Landmarker:
    """Thin wrapper over MediaPipe Tasks HandLandmarker (LIVE_STREAM) + optional selfie segmenter."""

    def __init__(self, cfg, on_frame, use_mask=True, model_dir=ROOT / "models"):
        import mediapipe as mp
        from mediapipe.tasks.python import vision, BaseOptions
        self.mp, self.on_frame = mp, on_frame
        d = cfg.detector
        self._pending = {}                                 # ms timestamp -> (t_capture, t_submit)
        self._lock = threading.Lock()
        self.last_ms, self.dropped, self.infer_ms, self.done_times = -1, 0, 0.0, []
        self.n_submitted = self.n_results = 0
        self.mask = None
        delegate = None
        try:                                               # GPU delegate where the platform supports it
            delegate = BaseOptions.Delegate.GPU
        except Exception:
            pass
        for dl in (delegate, None):
            try:
                opts = vision.HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(model_dir / "hand_landmarker.task"), delegate=dl),
                    running_mode=vision.RunningMode.LIVE_STREAM, num_hands=d.num_hands,
                    min_hand_detection_confidence=d.detection_confidence,
                    min_hand_presence_confidence=d.presence_confidence,
                    min_tracking_confidence=d.tracking_confidence, result_callback=self._on_result)
                self.hands = vision.HandLandmarker.create_from_options(opts)
                self.delegate = "GPU" if dl else "CPU"
                break
            except Exception as e:                         # GPU unavailable -> CPU
                if dl is None:
                    raise
                print("GPU delegate unavailable (", str(e)[:80], ") -> CPU", flush=True)
        self.seg = None
        seg_path = model_dir / "selfie_segmenter.tflite"
        if use_mask and seg_path.exists():
            self.seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
                base_options=BaseOptions(model_asset_path=str(seg_path)), running_mode=vision.RunningMode.VIDEO,
                output_category_mask=False, output_confidence_masks=True))
        self._seg_n = 0

    def submit(self, rgb, t_capture):
        """rgb: (h,w,3) uint8 RAW camera orientation. Returns False if the frame was dropped."""
        ms = int(t_capture * 1000.0)
        if ms <= self.last_ms:
            self.dropped += 1
            return False
        self.last_ms = ms
        with self._lock:
            self._pending[ms] = (t_capture, time.perf_counter())
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        self.hands.detect_async(img, ms)
        self.n_submitted += 1
        if self.seg is not None:                           # selfie mask at ~1/2 the camera rate
            self._seg_n += 1
            if self._seg_n % 2 == 0:
                try:
                    r = self.seg.segment_for_video(img, ms)
                    m = r.confidence_masks[-1].numpy_view()
                    import cv2
                    self.mask = cv2.resize(np.asarray(m, dtype=np.float32), (MASK_W, MASK_H), interpolation=cv2.INTER_AREA)
                except Exception:
                    pass
        return True

    def _on_result(self, result, image, ms):
        with self._lock:
            t_cap, t_sub = self._pending.pop(ms, (ms / 1000.0, time.perf_counter()))
            for k in [k for k in self._pending if k < ms - 2000]:
                del self._pending[k]
        self.n_results += 1
        now = time.perf_counter()
        self.infer_ms += 0.1 * ((now - t_sub) * 1000.0 - self.infer_ms)
        self.done_times = (self.done_times + [now])[-30:]
        hands = [Hand(np.array([[p.x, p.y, p.z] for p in lm]), np.array([[p.x, p.y, p.z] for p in w]),
                      hd[0].category_name, hd[0].score)
                 for lm, w, hd in zip(result.hand_landmarks, result.hand_world_landmarks, result.handedness)]
        self.on_frame(Frame(t_cap, hands, self.mask))

    @property
    def fps(self):
        d = self.done_times
        return (len(d) - 1) / (d[-1] - d[0]) if len(d) > 2 and d[-1] > d[0] else 0.0


class Sender:
    def __init__(self, cfg, host, port, log=None):
        self.core = HandCore(cfg)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addr = (host, port)
        self.seq, self.lm = 0, None
        self.log = open(log, "w") if log else None

    def on_frame(self, frame):
        import gc
        t0 = time.perf_counter()
        st = self.core.update(frame)
        cpu_us = int((time.perf_counter() - t0) * 1e6)
        if st is None:
            return
        self.seq += 1
        pkt = P.pack_state(self.seq, frame.t, st.valid, st.debug.get("held", False), st.n_hands, st.confidence,
                           st.corners.reshape(8), st.velocity.reshape(8), self.lm.infer_ms if self.lm else 0.0,
                           self.lm.fps if self.lm else 0.0, self.lm.dropped if self.lm else 0, cpu_us,
                           lid=st.lid.reshape(4), lid_vel=st.lid_vel.reshape(4),
                           pose=(st.openness, st.twist, st.twists[0], st.twists[1], st.speed, st.motion[0], st.motion[1]))
        self.sock.sendto(pkt, self.addr)
        if self.log:
            self.log.write(json.dumps({"t": frame.t, "hands": [{"lm": h.lm.tolist(), "world": h.world.tolist(),
                                                                 "label": h.label, "score": h.score} for h in frame.hands]}) + "\n")


def run_shm(args, cfg, sender):
    from multiprocessing import shared_memory
    print("waiting for TouchDesigner shared memory '" + P.FRAME_SHM_NAME + "' ...", flush=True)
    shm = None
    while shm is None:
        try:
            shm = shared_memory.SharedMemory(name=P.FRAME_SHM_NAME)
        except FileNotFoundError:
            time.sleep(0.5)
    mask_shm = None
    last_seq, n, t_stat = -1, 0, time.perf_counter()
    while True:
        if time.perf_counter() - t_stat > 3.0:
            t_stat = time.perf_counter()
            print(f"status: frames in {n}, submitted {sender.lm.n_submitted}, results {sender.lm.n_results}, packets out {sender.seq}, "
                  f"infer {sender.lm.infer_ms:.1f} ms, dropped {sender.lm.dropped}", flush=True)
        magic, seq, t, w, h, flags = P.FRAME_HDR.unpack_from(shm.buf, 0)
        if magic != P.FRAME_MAGIC or seq == last_seq or min(w, h) < P.MIN_SIDE or w * h * 3 > P.FRAME_MAX_BYTES:
            time.sleep(0.002)               # nothing yet, or a bogus frame (e.g. camera not started): ignore
            continue
        rgb = np.frombuffer(shm.buf, np.uint8, w * h * 3, P.FRAME_HDR.size).reshape(h, w, 3).copy()
        _, seq2, *_ = P.FRAME_HDR.unpack_from(shm.buf, 0)
        if seq2 != seq:                                    # torn read (writer was mid-frame): drop it
            sender.lm.dropped += 1
            continue
        last_seq = seq
        try:
            sender.lm.submit(rgb, t)
        except Exception as e:                # never let one bad frame kill the tracker
            print("frame rejected:", str(e)[:120], flush=True)
            continue
        n += 1
        if sender.lm.mask is not None and n % 2 == 0:
            if mask_shm is None:
                try:
                    mask_shm = shared_memory.SharedMemory(name=P.MASK_SHM_NAME, create=True, size=P.MASK_HDR.size + MASK_W * MASK_H)
                except FileExistsError:
                    mask_shm = shared_memory.SharedMemory(name=P.MASK_SHM_NAME)
            P.MASK_HDR.pack_into(mask_shm.buf, 0, P.MASK_MAGIC, n, MASK_W, MASK_H)
            np.frombuffer(mask_shm.buf, np.uint8, MASK_W * MASK_H, P.MASK_HDR.size)[:] = \
                (np.clip(sender.lm.mask, 0, 1) * 255).astype(np.uint8).ravel()


def run_opencv(args, cfg, sender, source):
    import cv2
    cap = cv2.VideoCapture(source if source != "camera" else args.camera, cv2.CAP_ANY)
    if source == "camera":
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.height)
        cap.set(cv2.CAP_PROP_FPS, cfg.camera.fps)
    t0, vid_fps = time.perf_counter(), cap.get(cv2.CAP_PROP_FPS) or 30.0
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        t = (time.perf_counter() - t0) if source == "camera" else i / vid_fps
        i += 1
        sender.lm.submit(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), t)
        if source != "camera":
            time.sleep(1.0 / vid_fps)
        if args.preview:
            st = sender.core.last_state
            if st is not None and st.valid:
                pts = st.corners[list(st.order)].astype(int)
                disp = cv2.flip(bgr, 1) if cfg.camera.mirror else bgr
                cv2.polylines(disp, [pts], True, (255, 255, 255), 2)
            else:
                disp = cv2.flip(bgr, 1) if cfg.camera.mirror else bgr
            cv2.imshow("portal sidecar (q quits)", disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["shm", "camera", "video"], default="shm")
    ap.add_argument("video", nargs="?")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--log")
    ap.add_argument("--no-mask", action="store_true")
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()
    cfg = hcfg.load(args.config)
    sender = Sender(cfg, args.host, args.port, args.log)
    sender.lm = Landmarker(cfg, sender.on_frame, use_mask=not args.no_mask and cfg.gates.person_mask)
    print(f"HandLandmarker LIVE_STREAM, delegate={sender.lm.delegate}, mask={'on' if sender.lm.seg else 'off'}, "
          f"sending UDP {args.host}:{args.port}", flush=True)
    try:
        if args.source == "shm":
            run_shm(args, cfg, sender)
        else:
            run_opencv(args, cfg, sender, "camera" if args.source == "camera" else args.video)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
