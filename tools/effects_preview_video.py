"""Render the dial effects offscreen while PLAYING a video through the renderer (slit scan and datamosh need real motion), one PNG per effect.

    python tools/effects_preview_video.py "clip.mp4" out_prefix [--start 60] [--frames 70] [--speed 0.0] [--effects glass moire ...] [--crop-top 20 --crop-bottom 650]
Writes <prefix>_<effect>.png (full 1280x720) and <prefix>_sheet.png (contact sheet). Also prints the GPU time per frame of each effect.
"""
import argparse, pathlib, sys, time
import numpy as np
import cv2
import moderngl

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.renderer import Renderer                       # noqa: E402
from handcore import config as hcfg                     # noqa: E402
from handcore.view import ViewState                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("prefix")
    ap.add_argument("--start", type=int, default=60); ap.add_argument("--frames", type=int, default=70)
    ap.add_argument("--speed", type=float, default=0.0); ap.add_argument("--effects", nargs="*")
    ap.add_argument("--crop-top", type=int, default=0); ap.add_argument("--crop-bottom", type=int, default=0)
    a = ap.parse_args()
    W, H = 1280, 720
    g = hcfg.load(str(ROOT / "config.yaml")).gesture
    names = a.effects or list(g.dial_effects)
    ctx = moderngl.create_standalone_context()
    print("GL:", ctx.info["GL_RENDERER"])
    R = Renderer(ctx, W, H, 3.0)
    fbo = ctx.simple_framebuffer((W, H), 3)
    c = [215, 560, 235, 235, 1070, 215, 1050, 540, 205, 205, 1085, 190]
    tiles = []
    for name in names:
        cap = cv2.VideoCapture(a.video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, a.start)
        R.hist_filled = False
        R._mosh_frame = -10
        v = ViewState(g)
        v.dial.force(name)
        pose = {"openness": 1.0, "twist": 35.0, "twists": (35.0, 35.0), "speed": a.speed, "motion": (1.0, 0.0) if a.speed else (0.0, 0.0)}
        for _ in range(60):
            vw = v.update(c, 1.0, 1 / 30, pose, W, H)
        dts = []
        for i in range(a.frames):
            ok, bgr = cap.read()
            if not ok:
                break
            if a.crop_bottom:
                bgr = bgr[a.crop_top:a.crop_bottom]
            R.set_camera(cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (W, H)))
            t0 = time.perf_counter()
            R.draw(fbo, (W, H), vw, 2.0 + i / 30.0, show_hud=False)
            fbo.read(components=1)                                                # force the GPU to finish
            dts.append(time.perf_counter() - t0)
        im = np.frombuffer(fbo.read(components=3), np.uint8).reshape(H, W, 3)[::-1].copy()
        cv2.imwrite(f"{a.prefix}_{name}.png", cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        print(f"{name:10s} draw {1000 * np.median(dts):6.1f} ms median, {1000 * np.percentile(dts, 95):6.1f} ms p95 (offscreen, this GPU)")
        t = im.copy(); cv2.putText(t, name, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        tiles.append(cv2.resize(t, (640, 360), interpolation=cv2.INTER_AREA))
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(f"{a.prefix}_sheet.png", cv2.cvtColor(np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]), cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
