"""Standalone "Portals to Other Dimensions": NO TouchDesigner needed. Two processes:

  tracker process (app/tracker_proc.py): camera thread -> landmark engine -> handcore.HandCore -> state messages + shared-memory camera frames
  this process                         : glfw window, vsync: Presenter.render(now) -> GLSL passes (td/shaders/*.glsl) -> screen
(separate processes so Python / GL work on one side can never starve the other: no shared GIL)

    python app/portal_app.py                      # webcam 0, mirrored, fullscreen toggle F
    python app/portal_app.py --camera 1 --cell 6  # other camera, fewer particles
    python app/portal_app.py --source clip.mp4    # play a video instead of a webcam
    python app/portal_app.py --engine tasks       # the stock MediaPipe Tasks tracker instead of the full-resolution crop tracker
NO key presses are needed. The box is attached to your thumb / index / middle / ring fingertips; an OPEN hand opens it, a PINCH closes it;
TWISTING your hands twists the box and turns the effect dial (particles+moon > glass > ASCII > bitmap > web > neon); moving fast makes it glitch.
Utility keys only: H toggles the HUD, F fullscreen, Q / Esc quit.
"""
from __future__ import annotations
import argparse, pathlib, sys, threading, time
import numpy as np
import cv2

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from handcore import config as hcfg                               # noqa: E402
from handcore.predict import Presenter                            # noqa: E402
from handcore.view import ViewState                               # noqa: E402


class Camera(threading.Thread):
    def __init__(self, source, cfg, lm_holder, sink=None):
        super().__init__(daemon=True)
        self.cfg, self.lm_holder, self.sink = cfg, lm_holder, sink
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self.is_file = isinstance(source, str)
        self.cap = cv2.VideoCapture(source, cv2.CAP_ANY if self.is_file else backend)
        if not self.is_file:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.height)
            self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.ok = self.cap.isOpened()
        self.latest, self.seq, self.t = None, 0, 0.0
        self.fps, self._times = 0.0, []
        self.stop = False
        self.lock = threading.Lock()

    def run(self):
        vid_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        while not self.stop:
            ok, bgr = self.cap.read()
            if not ok:
                if self.is_file:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
                time.sleep(0.01); continue
            t = time.perf_counter()
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if self.sink:
                self.sink(rgb, t)
            with self.lock:
                self.latest, self.seq, self.t = rgb, self.seq + 1, t
            self._times = (self._times + [t])[-30:]
            if len(self._times) > 2:
                self.fps = (len(self._times) - 1) / (self._times[-1] - self._times[0])
            small = cv2.resize(rgb, (640, 360), interpolation=cv2.INTER_AREA)   # detector input size
            lm = self.lm_holder[0]
            every = getattr(lm, "every_frame", False)
            if lm is not None and self.seq % (1 if every else self.cfg_divider()) == 0:
                try:
                    lm.submit(small, t, rgb) if every else lm.submit(small, t)
                except Exception as e:
                    print("frame rejected:", str(e)[:100], flush=True)
            if self.is_file:
                time.sleep(max(0.0, 1.0 / vid_fps - (time.perf_counter() - t)))

    def cfg_divider(self):
        return 2 if self.fps > 45 else 1                  # 60 fps camera -> 30 Hz tracking

    def get(self):
        with self.lock:
            return self.latest, self.seq, self.t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--source", help="video file instead of a webcam")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--cell", type=float, default=3.0, help="particle grid cell in px (3 = 102,400 points at 1280x720; 4 = 57,600)")
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--engine", choices=("roi", "tasks"), default="roi",
                    help="roi = full-resolution crop tracker at camera rate (default); tasks = stock MediaPipe Tasks on a 640 px frame")
    ap.add_argument("--tta", type=int, default=2, help="roi engine: crops averaged per hand per frame (1-5; more = steadier, slower)")
    ap.add_argument("--record-video", action="store_true", help="OPT-IN: save the raw camera video to logs/session_video.mp4 (local only) for offline accuracy tests")
    ap.add_argument("--effect", help="start on this effect of the dial (testing); the twist ratchet still works from there")
    ap.add_argument("--hud", action="store_true", help="show the debug HUD from the start")
    ap.add_argument("--seconds", type=float, default=0.0, help="quit after N seconds (testing)")
    ap.add_argument("--screenshot", help="save a screenshot after 5 s and continue")
    a = ap.parse_args()

    import glfw, moderngl
    from app.renderer import Renderer
    from app.tracker_proc import TrackerProcess

    cfg = hcfg.load(a.config)
    if a.no_mirror:
        cfg.camera.mirror = False
    W, H = cfg.camera.width, cfg.camera.height
    pres = Presenter(cfg.predict, 1 / 30.0, n=12)             # 4 corners + 2 glass-lid points
    plock = threading.Lock()
    state = {"pose": {"openness": 0.0, "twist": 35.0, "twists": (35.0, 35.0), "speed": 0.0, "motion": (0.0, 0.0)}, "n_hands": 0, "valid": False, "conf": 0.0,
             "fps": 0.0, "infer_ms": 0.0, "dropped": 0, "cam_fps": 0.0, "rejects": {}, "dbg": []}

    # tracking runs in its own process (no GIL shared with this render loop); states arrive over a pipe
    trk = TrackerProcess({"config": a.config, "source": a.source, "camera": a.camera, "no_mirror": a.no_mirror, "engine": a.engine, "tta": a.tta, "record": a.record_video}, W, H)

    def receive():
        while True:
            try:
                m = trk.conn.recv()
            except (EOFError, OSError):
                return
            if "error" in m:
                print(m["error"], flush=True); continue
            if "ready" in m:
                print(m["ready"], flush=True); continue
            with plock:
                pres.push(m["now"], m["t"], m["valid"], m["c12"], m["v12"], held=m["held"])
                state.update(pose=m["pose"], n_hands=m["n_hands"], valid=m["valid"], conf=m["conf"], rejects=m["rejects"], dbg=m["dbg"],
                             fps=m["fps"], infer_ms=m["infer_ms"], dropped=m["dropped"], cam_fps=m["cam_fps"])
    threading.Thread(target=receive, daemon=True).start()

    if not glfw.init():
        sys.exit("glfw init failed")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3); glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE); glfw.window_hint(glfw.RESIZABLE, True)
    win = glfw.create_window(W, H, "Portals to Other Dimensions", None, None)
    glfw.make_context_current(win)
    glfw.swap_interval(1)
    ctx = moderngl.create_context()
    print("GL:", ctx.info["GL_RENDERER"], flush=True)
    R = Renderer(ctx, W, H, a.cell)
    view = ViewState(cfg.gesture)
    if a.effect:
        view.dial.force(a.effect)
    R.hud_on = a.hud
    ui = {"fs": False, "quit": False}

    def on_key(w, key, sc, action, mods):
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_H: R.hud_on = not R.hud_on
        elif key in (glfw.KEY_Q, glfw.KEY_ESCAPE): ui["quit"] = True
        elif key == glfw.KEY_F:
            ui["fs"] = not ui["fs"]
            if ui["fs"]:
                mon = glfw.get_primary_monitor(); vm = glfw.get_video_mode(mon)
                glfw.set_window_monitor(win, mon, 0, 0, vm.size.width, vm.size.height, vm.refresh_rate)
            else:
                glfw.set_window_monitor(win, None, 100, 100, W, H, 0)
    glfw.set_key_callback(win, on_key)

    last_seq, t_prev, dts, t0 = 0, time.perf_counter(), [], time.perf_counter()
    hud_t = 0.0
    shot_done = False
    while not glfw.window_should_close(win) and not ui["quit"]:
        glfw.poll_events()
        now = time.perf_counter()
        dt = now - t_prev; t_prev = now
        dts = (dts + [dt])[-120:]
        got = trk.share.read(last_seq)
        if got is not None:
            frame, last_seq, _ = got
            disp = np.ascontiguousarray(frame[:, ::-1] if cfg.camera.mirror else frame)
            if R.hud_on:                      # debug markers: detected thumb (red) / index (green) / middle (yellow) / ring (cyan) tips
                for hand in state.get("dbg", []):
                    for (x, y), col in zip(hand[:4], ((255, 60, 60), (60, 255, 60), (255, 255, 60), (60, 255, 255))):
                        cv2.circle(disp, (int(x), int(y)), 9, col, 2)
            R.set_camera(disp)
        with plock:
            c, alpha = pres.render(now)
            pose = dict(state["pose"])
        v = view.update(c, alpha, dt, pose, W, H)
        top = max(v["weights"].items(), key=lambda kv: kv[1])[0]
        if now - hud_t > 0.2:
            hud_t = now
            fps = 1.0 / max(np.mean(dts), 1e-6)
            R.set_hud([f"render {fps:5.1f} fps  dt max {max(dts) * 1000:4.1f} ms   camera {state['cam_fps']:4.1f} fps",
                       f"tracker {state['fps']:4.1f} fps  infer {state['infer_ms']:4.1f} ms  dropped {state['dropped']}",
                       f"hands {state['n_hands']}/{len(state.get('dbg', []))}det valid {state['valid']} open {v['open']:.2f} twist {v['twist']:3.0f} spd {v['speed']:.2f} fx {top} rej {state.get('rejects', {})}",
                       ("NO CAMERA FRAMES - another app (TouchDesigner?) may be using the webcam" if last_seq == 0 and now - t0 > 3 else
                        "no keys: pinch = close, open hand = open, twist = effect   H hud  Esc quit")])
            if last_seq == 0 and now - t0 > 8 and not ui.get("warned"):
                ui["warned"] = True
                print("WARNING: the camera produced no frames. Close other apps using the webcam (TouchDesigner's cam node), or try --camera 1.", flush=True)
        fb_w, fb_h = glfw.get_framebuffer_size(win)
        if fb_w < 2 or fb_h < 2:                    # minimized: nothing to draw
            time.sleep(0.05); continue
        ctx.screen.viewport = (0, 0, fb_w, fb_h)
        R.draw(ctx.screen, (fb_w, fb_h), v, now - t0)
        if a.screenshot and not shot_done and now - t0 > 5:
            shot_done = True
            raw = np.frombuffer(ctx.screen.read(components=3), np.uint8).reshape(fb_h, fb_w, 3)[::-1]
            cv2.imwrite(a.screenshot, cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)); print("saved", a.screenshot, flush=True)
        glfw.swap_buffers(win)
        if a.seconds and now - t0 > a.seconds:
            break
    why = "window closed" if glfw.window_should_close(win) else ("Q/Esc pressed" if ui["quit"] else "time limit")
    trk.close()
    glfw.terminate()
    print("exit reason:", why, flush=True)
    print(f"exit: avg render fps {1.0 / np.mean(dts):.1f}, worst frame {max(dts) * 1000:.1f} ms", flush=True)


if __name__ == "__main__":
    main()
