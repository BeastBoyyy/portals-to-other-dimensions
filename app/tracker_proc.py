"""The tracking side of the app, in its OWN PROCESS (so the render loop's Python / GL work can never starve it, and the reverse):

   camera thread -> landmark engine (roi | tasks) -> HandCore -> state messages (Pipe) to the render process
                 \\-> full-resolution camera frames into a shared-memory double buffer, for the render process to display

time.perf_counter() is one system-wide clock on Windows, so timestamps taken here are valid in the render process.
"""
from __future__ import annotations
import json, multiprocessing as mp, pathlib, struct, sys, threading, time
from multiprocessing import shared_memory
import numpy as np
import cv2

ROOT = pathlib.Path(__file__).resolve().parent.parent
HDR = struct.Struct("<qqd")                                # seq, slot index, capture time


class FrameShare:
    """Two full-size RGB buffers + a header; the writer fills the idle buffer and then publishes (seq, index). One writer, one reader."""

    def __init__(self, W, H, name=None):
        self.W, self.H, self.size = W, H, W * H * 3
        if name is None:
            self.shm = shared_memory.SharedMemory(create=True, size=HDR.size + 2 * self.size)
            HDR.pack_into(self.shm.buf, 0, 0, 0, 0.0)
        else:
            self.shm = shared_memory.SharedMemory(name=name)
        self.name, self.seq = self.shm.name, 0
        self.views = [np.ndarray((H, W, 3), np.uint8, buffer=self.shm.buf, offset=HDR.size + i * self.size) for i in (0, 1)]

    def write(self, rgb, t):
        self.seq += 1
        i = self.seq & 1
        self.views[i][:] = rgb if rgb.shape[:2] == (self.H, self.W) else cv2.resize(rgb, (self.W, self.H))
        HDR.pack_into(self.shm.buf, 0, self.seq, i, t)

    def read(self, last_seq):
        """-> (frame copy, seq, t) or None when nothing new."""
        seq, i, t = HDR.unpack_from(self.shm.buf, 0)
        if seq == last_seq or seq == 0:
            return None
        return self.views[i].copy(), seq, t

    def close(self, unlink=False):
        self.views = []
        self.shm.close()
        if unlink:
            self.shm.unlink()


def child_main(a, shm_name, conn, stop):
    sys.path.insert(0, str(ROOT))
    from app.portal_app import Camera
    from handcore import config as hcfg
    from handcore.pipeline import HandCore

    cfg = hcfg.load(a["config"])
    if a["no_mirror"]:
        cfg.camera.mirror = False
    W, H = cfg.camera.width, cfg.camera.height
    core = HandCore(cfg)
    share = FrameShare(W, H, shm_name)
    logdir = ROOT / "logs"; logdir.mkdir(exist_ok=True)
    f_raw, f_trace = open(logdir / "live_frames.jsonl", "w"), open(logdir / "trace.jsonl", "w")
    n_out = [0]
    holder = [None]
    cam = None

    def det_tips(frame):
        out = []
        for h in frame.hands:
            xs = (1.0 - h.lm[:, 0]) if cfg.camera.mirror else h.lm[:, 0]
            out.append([[float(xs[i] * W), float(h.lm[i, 1] * H)] for i in (4, 8, 12, 16)] + [round(float(h.score), 2), h.label])
        return out

    def on_frame(frame):
        f_raw.write(json.dumps({"t": frame.t, "hands": [{"lm": np.round(h.lm, 4).tolist(), "world": np.round(h.world, 4).tolist(),
                                                         "label": h.label, "score": round(float(h.score), 3)} for h in frame.hands]}) + "\n")
        st = core.update(frame)
        if st is None:
            return
        n_out[0] += 1
        sl = core.tracker.slots
        tips = det_tips(frame)
        f_trace.write(json.dumps({"t": round(frame.t, 3), "det": len(frame.hands), "rej": st.debug.get("rejects", {}), "assigned": st.debug.get("assigned"),
                                  "slots": [s.state for s in sl], "n_hands": st.n_hands, "valid": st.valid, "held": bool(st.debug.get("held")),
                                  "quad_reject": st.debug.get("quad_reject"), "open": round(st.openness, 2), "twist": round(st.twist, 1),
                                  "spread": [None if s.pose.spread is None else round(s.pose.spread, 2) for s in sl],
                                  "corners": np.round(st.corners, 0).tolist(), "tips": tips}) + "\n")
        if n_out[0] % 15 == 0:
            f_raw.flush(); f_trace.flush()
        lm = holder[0]
        try:
            conn.send({"now": time.perf_counter(), "t": st.t, "valid": st.valid, "held": bool(st.debug.get("held")),
                       "c12": np.concatenate([st.corners.reshape(8), st.lid.reshape(4)]), "v12": np.concatenate([st.velocity.reshape(8), st.lid_vel.reshape(4)]),
                       "pose": {"openness": st.openness, "twist": st.twist, "twists": st.twists, "speed": st.speed, "motion": st.motion},
                       "n_hands": st.n_hands, "conf": st.confidence, "rejects": dict(st.debug.get("rejects", {})), "dbg": tips,
                       "fps": lm.fps, "infer_ms": lm.infer_ms, "dropped": lm.dropped, "cam_fps": cam.fps if cam else 0.0})
        except (BrokenPipeError, OSError):
            stop.set()

    rec = None
    if a.get("record"):                                    # opt-in: raw camera video + per-frame timestamps, for offline accuracy evaluation
        import queue
        rq = queue.Queue(maxsize=90)
        vpath = logdir / "session_video.mp4"

        def writer():
            vw, ts = None, []
            while True:
                item = rq.get()
                if item is None:
                    break
                rgb, t = item
                if vw is None:
                    vw = cv2.VideoWriter(str(vpath), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (rgb.shape[1], rgb.shape[0]))
                vw.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)); ts.append(t)
            if vw:
                vw.release()
            (logdir / "session_video_t.json").write_text(json.dumps(ts))
        rec = threading.Thread(target=writer, daemon=True); rec.start()

    def sink(rgb, t):
        share.write(rgb, t)
        if rec is not None:
            try:
                rq.put_nowait((rgb.copy(), t))
            except queue.Full:
                pass

    cam = Camera(a["source"] if a["source"] else a["camera"], cfg, holder, sink=sink)
    if not cam.ok:
        conn.send({"error": "cannot open the camera / video (try --camera 1; close other apps that use the webcam)"})
        return
    if a["engine"] == "roi":
        from app.roi_landmarker import RoiLandmarker
        holder[0] = RoiLandmarker(cfg, on_frame, tta=a["tta"])
    else:
        from adapters.mediapipe_sidecar import Landmarker
        holder[0] = Landmarker(cfg, on_frame, use_mask=False)
    conn.send({"ready": f"landmarks: engine={a['engine']} {holder[0].delegate}"})
    cam.start()
    while not stop.is_set():
        time.sleep(0.1)
    cam.stop = True
    if rec is not None:
        rq.put(None); rec.join(10)
    f_raw.close(); f_trace.close()


class TrackerProcess:
    def __init__(self, a, W, H):
        ctx = mp.get_context("spawn")
        self.share = FrameShare(W, H)
        self.stop = ctx.Event()
        recv, send = ctx.Pipe(duplex=False)
        self.conn = recv
        self.proc = ctx.Process(target=child_main, args=(a, self.share.name, send, self.stop), daemon=True)
        self.proc.start()

    def close(self):
        self.stop.set()
        self.proc.join(3)
        if self.proc.is_alive():
            self.proc.terminate()
        self.share.close(unlink=True)
