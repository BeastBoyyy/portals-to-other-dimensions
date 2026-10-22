"""Compile and RUN the td/shaders/*.glsl files offline (moderngl) with a TouchDesigner-compatible preamble.

    python tools/shader_preview.py            # writes fixtures/preview/*.png from the sample images

TouchDesigner's GLSL TOP injects: vUV, sTD2DInputs[], uTDOutputInfo.res, TDOutputSwizzle(). This script injects the
same names, so a shader that compiles/renders here uses only TD-provided symbols. It does NOT prove that the TD
operator parameter wiring is right (that is build_network.py + MANUAL_STEPS.md, verified by td/selfcheck.py).
"""
from __future__ import annotations
import pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
SH = ROOT / "td" / "shaders"

PREAMBLE = """#version 330
in vec3 vUV;
uniform sampler2D sTD2DInputs[4];
struct TDTexInfo { vec4 res; vec4 depth; };
uniform TDTexInfo uTDOutputInfo;
vec4 TDOutputSwizzle(vec4 c) { return c; }
"""
VERT = """#version 330
in vec2 in_pos;
out vec3 vUV;
void main() { vUV = vec3(in_pos * 0.5 + 0.5, 0.0); gl_Position = vec4(in_pos, 0.0, 1.0); }
"""


def render(name, w, h, textures=(), uniforms=None, ctx=None):
    import moderngl
    own = ctx is None
    if own:
        ctx = moderngl.create_standalone_context()
    src = PREAMBLE + (SH / name).read_text()
    prog = ctx.program(vertex_shader=VERT, fragment_shader=src)
    vbo = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], "f4").tobytes())
    vao = ctx.simple_vertex_array(prog, vbo, "in_pos")
    tex = []
    for i, t in enumerate(textures):
        h_, w_ = t.shape[:2]
        tx = ctx.texture((w_, h_), 4, np.ascontiguousarray(t[::-1], "f4").tobytes(), dtype="f4")  # bottom-left origin
        tx.filter = (moderngl.LINEAR, moderngl.LINEAR); tx.use(i); tex.append(tx)
    if "sTD2DInputs" in prog:
        prog["sTD2DInputs"].value = list(range(len(textures))) + [0] * (4 - len(textures))
    if "uTDOutputInfo.res" in prog:
        prog["uTDOutputInfo.res"].value = (1.0 / w, 1.0 / h, float(w), float(h))
    for k, v in (uniforms or {}).items():
        if k in prog:
            prog[k].value = v
    fbo = ctx.simple_framebuffer((w, h), 4, dtype="f4")
    fbo.use(); ctx.clear(0, 0, 0, 1)
    vao.render(moderngl.TRIANGLE_STRIP)
    out = np.frombuffer(fbo.read(components=4, dtype="f4"), "f4").reshape(h, w, 4)[::-1]
    if own:
        ctx.release()
    return out


def load_rgba(path, w, h):
    import cv2
    bgr = cv2.resize(cv2.imread(str(path)), (w, h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype("f4") / 255.0
    return np.dstack([rgb, np.ones((h, w), "f4")])


def selfie_mask(rgba):
    import cv2, mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=str(ROOT / "models" / "selfie_segmenter.tflite")),
        running_mode=vision.RunningMode.IMAGE, output_category_mask=False, output_confidence_masks=True))
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray((rgba[..., :3] * 255).astype(np.uint8)))
    m = seg.segment(img).confidence_masks[-1].numpy_view()
    m = cv2.resize(np.asarray(m, "f4"), (rgba.shape[1], rgba.shape[0]))
    return np.dstack([m, m, m, np.ones_like(m)])


def save(path, rgba):
    import cv2
    cv2.imwrite(str(path), cv2.cvtColor((np.clip(rgba[..., :3], 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def main():
    out = ROOT / "fixtures" / "preview"
    out.mkdir(parents=True, exist_ok=True)
    W, H = 1280, 720
    cam = load_rgba(ROOT / "fixtures" / "images" / "portrait.jpg", W, H)
    mask = selfie_mask(cam)
    a = render("scene_a.glsl", W, H, [cam, mask], {"uA": (1.3, 4.0, 6.0, 1.0), "uB": (0.9, 2.5, 1.3, 0.0)})
    b = render("scene_b.glsl", W, H, [cam], {"uA": (1.0, 0.42, 0.15, 0.0), "uB": (0.0, 1.0, 0, 0)})
    blend = render("scene_blend.glsl", W, H, [a, b], {"uMix": (0.0, 0, 0, 0)})
    quad = (0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75, 0.25)           # q0..q3 in uv, bottom-left origin
    comp = render("composite.glsl", W, H, [cam, blend, blend],
                  {"uQA": quad[:4], "uQB": quad[4:], "uP1": (1.0, 0.6, 0.0, 1.0), "uP2": (0.0, 0.0, 0.3, 2.0)})
    comp_b = render("composite.glsl", W, H, [cam, b, b],
                    {"uQA": (0.3, 0.2, 0.2, 0.8), "uQB": (0.8, 0.75, 0.7, 0.25), "uP1": (1.0, 0.6, 0.0, 0.0), "uP2": (0.02, 0.0, 0.3, 2.0)})
    for n, im in (("scene_a", a), ("scene_b", b), ("composite_a", comp), ("composite_b_skewed", comp_b)):
        save(out / f"{n}.png", im)
    print("wrote", [p.name for p in sorted(out.glob('*.png'))])


if __name__ == "__main__":
    main()
