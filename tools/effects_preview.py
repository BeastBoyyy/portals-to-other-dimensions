"""Render the twist dial offscreen on one camera image -> contact sheet (one tile per effect; second sheet = same at high hand speed).

    python tools/effects_preview.py [camera_image.png] [out_prefix]
Corner/lid geometry mimics the reference clip. Each tile sets the hand twist to the middle of an effect's sector on the dial.
"""
import pathlib, sys
import numpy as np
import cv2
import moderngl

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.renderer import Renderer                       # noqa: E402
from handcore import config as hcfg                     # noqa: E402
from handcore.view import ViewState                     # noqa: E402
from tools import shader_preview as sp                  # noqa: E402


def main():
    img = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "fixtures" / "images" / "portrait.jpg")
    prefix = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "fixtures" / "preview" / "effects")
    W, H = 1280, 720
    g = hcfg.load(str(ROOT / "config.yaml")).gesture
    cam = sp.load_rgba(img, W, H)
    mask = sp.selfie_mask(cam)[..., 0]
    ctx = moderngl.create_standalone_context()
    R = Renderer(ctx, W, H, 3.0)
    R.set_camera((cam[..., :3] * 255).astype(np.uint8))
    R.set_mask(cv2.resize(mask, (256, 144), interpolation=cv2.INTER_AREA))
    fbo = ctx.simple_framebuffer((W, H), 3)
    c = [215, 560, 235, 235, 1070, 215, 1050, 540, 205, 205, 1085, 190]       # thumb0, anchor0, anchor1, thumb1, lid0, lid1 (px)
    centres = [60.0] * len(g.dial_effects)
    for tag, speed, motion in (("calm", 0.0, (0.0, 0.0)), ("fast", 0.9, (1.0, 0.0))):
        tiles = []
        for name, tw in zip(g.dial_effects, centres):
            v = ViewState(g)
            v.dial.force(name)
            pose = {"openness": 1.0, "twist": tw, "twists": (tw, tw), "speed": speed, "motion": motion}
            for _ in range(80):
                vw = v.update(c, 1.0, 1 / 30, pose, W, H)
            R.draw(fbo, (W, H), vw, 2.3, show_hud=False)
            im = np.frombuffer(fbo.read(components=3), np.uint8).reshape(H, W, 3)[::-1].copy()
            cv2.putText(im, f"{name}  twist {tw:.0f}  speed {speed}", (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
            if len(sys.argv) > 3 and sys.argv[3] == "full":
                cv2.imwrite(f"{prefix}_{tag}_{name}.png", cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            tiles.append(cv2.resize(im, (640, 360), interpolation=cv2.INTER_AREA))
        sheet = np.vstack([np.hstack(tiles[0:3]), np.hstack(tiles[3:6])])
        cv2.imwrite(f"{prefix}_{tag}.png", cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        print("wrote", f"{prefix}_{tag}.png")


if __name__ == "__main__":
    main()
