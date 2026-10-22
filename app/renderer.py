"""OpenGL renderer for the standalone app (moderngl). Uses the SAME GLSL files as the TouchDesigner build (td/shaders/*.glsl),
wrapped in a small preamble that provides the symbols TouchDesigner would inject (vUV, sTD2DInputs, uTDOutputInfo, TDOutputSwizzle).

Texture convention (identical to TD): uv (0,0) is the BOTTOM-left, so images are uploaded with their rows reversed.
Effects are drawn into their own buffers and blended additively with the weights of the twist dial (cross-fades), only if visible.
"""
from __future__ import annotations
import pathlib
import numpy as np
import moderngl
import cv2

SH = pathlib.Path(__file__).resolve().parent.parent / "td" / "shaders"

PREAMBLE = """#version 330
in vec3 vUV;
uniform sampler2D sTD2DInputs[4];
struct TDTexInfo { vec4 res; vec4 depth; };
uniform TDTexInfo uTDOutputInfo;
vec4 TDOutputSwizzle(vec4 c) { return c; }
"""
VERT = """#version 330
in vec2 in_pos;
uniform vec4 uRect;                       // x0,y0,x1,y1 in NDC (full screen by default)
out vec3 vUV;
void main() { vUV = vec3(in_pos * 0.5 + 0.5, 0.0);
              vec2 p = mix(uRect.xy, uRect.zw, vUV.xy); gl_Position = vec4(p, 0.0, 1.0); }
"""
BLUR = """
uniform vec2 uDir;
layout(location = 0) out vec4 fragColor;
void main() {
    vec2 uv = vUV.st; vec3 s = vec3(0.0);
    float w[5] = float[5](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
    s += texture(sTD2DInputs[0], uv).rgb * w[0];
    for (int i = 1; i < 5; i++) { s += texture(sTD2DInputs[0], uv + uDir * float(i) * 1.5).rgb * w[i];
                                  s += texture(sTD2DInputs[0], uv - uDir * float(i) * 1.5).rgb * w[i]; }
    fragColor = vec4(s, 1.0);
}
"""
COPY = """
layout(location = 0) out vec4 fragColor;
void main() { fragColor = texture(sTD2DInputs[0], vUV.st); }
"""
ACC = """
uniform vec4 uW;                          // x = weight: additive cross-fade accumulation
layout(location = 0) out vec4 fragColor;
void main() { fragColor = vec4(texture(sTD2DInputs[0], vUV.st).rgb * uW.x, 1.0); }
"""

EFFECT_SRC = {"particles": "scene_a.glsl", "moon": "scene_moon.glsl", "glass": "fx_glass.glsl", "ascii": "fx_ascii.glsl",
              "bitmap": "fx_bitmap.glsl", "web": "fx_web.glsl", "neon": "fx_neon.glsl", "asteroid": "scene_b.glsl",
              "ghost": "scene_ghost.glsl", "slitscan": "fx_slitscan.glsl", "moire": "fx_moire.glsl", "pixelsort": "fx_pixelsort.glsl",
              "crt": "fx_crt.glsl", "datamosh": "fx_datamosh.glsl"}
EXTRA_SRC = {"mosh_mv": "fx_datamosh_mv.glsl"}                       # helper passes (not dial effects)
HIST_W, HIST_H, HIST_N = 768, 432, 48                               # frame history for slit scan: 48 frames of 768x432 (about 48 MB)
MOSH_BLOCK = 16
EFFECT_SCALE = {"pixelsort": 0.5}                               # effects whose detail is low-frequency render at a fraction of the camera size (then linearly upsampled)
RAMP = " .`'^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"      # dark -> dense
GLYPH_W, GLYPH_H = 20, 40


def build_glyph_atlas():
    """One row of RAMP glyphs, white on black, (GLYPH_H, GLYPH_W*len) uint8. Uses Consolas / Courier if present, else OpenCV text."""
    from PIL import Image, ImageDraw, ImageFont
    font = None
    for p in ("C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/cour.ttf", "/System/Library/Fonts/Menlo.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        try:
            font = ImageFont.truetype(p, 34)
            break
        except Exception:
            continue
    img = Image.new("L", (GLYPH_W * len(RAMP), GLYPH_H), 0)
    d = ImageDraw.Draw(img)
    for i, ch in enumerate(RAMP):
        if ch == " ":
            continue
        if font is not None:
            bb = d.textbbox((0, 0), ch, font=font)
            d.text((i * GLYPH_W + (GLYPH_W - (bb[2] - bb[0])) // 2 - bb[0], (GLYPH_H - (bb[3] - bb[1])) // 2 - bb[1]), ch, fill=255, font=font)
        else:
            cv2.putText(np.asarray(img), ch, (i * GLYPH_W + 2, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2)
    return np.asarray(img)


class Renderer:
    def __init__(self, ctx: moderngl.Context, W=1280, H=720, cell_px=3.0):
        self.ctx, self.W, self.H, self.cell = ctx, W, H, cell_px
        self.vbo = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], "f4").tobytes())
        self.progs = {}
        srcs = {k: (SH / v).read_text() for k, v in {**EFFECT_SRC, **EXTRA_SRC}.items()}
        srcs.update(composite=(SH / "composite.glsl").read_text(), blur=BLUR, hud=COPY, acc=ACC)
        for name, src in srcs.items():
            p = ctx.program(vertex_shader=VERT, fragment_shader=PREAMBLE + src)
            self.progs[name] = (p, ctx.simple_vertex_array(p, self.vbo, "in_pos"))
        lin = (moderngl.LINEAR, moderngl.LINEAR)
        mip = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.cam = ctx.texture((W, H), 3, dtype="f1"); self.cam.filter = mip                 # mip chain: cheap, flicker-free blurs for the effects
        self.cam_prev = ctx.texture((W, H), 3, dtype="f1"); self.cam_prev.filter = mip
        for t_ in (self.cam, self.cam_prev):                                                  # complete mip chains from the start
            t_.write(bytes(W * H * 3)); t_.build_mipmaps()
        self.cam_seq = 0                                                                      # counts uploaded camera frames
        # two identical history arrays: draws read one while the new frame is written into the other (writing a texture the GPU is still sampling stalls the driver)
        self.hists = [ctx.texture_array((HIST_W, HIST_H, HIST_N), 3, dtype="f1") for _ in (0, 1)]
        for h_ in self.hists:
            h_.filter = lin; h_.repeat_x = h_.repeat_y = False
        self.hcur, self.hist_n, self.hist_seen, self.hist_recent, self.hist_filled = 0, 0, [0, 0], [], False
        self.hist_head = 0                                                                     # layer (frame number % N) of the newest frame in the array being read
        self.frame_id, self._mosh_frame, self._mosh_seq = 0, -10, -1
        self.mask = ctx.texture((256, 144), 4, np.full((144, 256, 4), 255, np.uint8).tobytes(), dtype="f1"); self.mask.filter = lin
        self.hud = ctx.texture((560, 120), 4, dtype="f1"); self.hud.filter = lin
        atlas = build_glyph_atlas()
        self.atlas = ctx.texture((atlas.shape[1], atlas.shape[0]), 3, np.ascontiguousarray(np.dstack([atlas] * 3)[::-1]).tobytes(), dtype="f1")
        self.atlas.filter = lin
        self.hud_on = True
        self.fbo = {}
        for name in ("blend",) + tuple(EFFECT_SRC):
            sc = EFFECT_SCALE.get(name, 1.0)
            self._make_fbo(name, int(W * sc), int(H * sc), lazy=name != "blend")
        self._make_fbo("datamosh_b", W, H, lazy=True)
        self._make_fbo("mv", W // MOSH_BLOCK, H // MOSH_BLOCK, lazy=True)
        self._make_fbo("small1", W // 4, H // 4)
        self._make_fbo("small2", W // 4, H // 4)

    def _make_fbo(self, name, w, h, lazy=False):
        if lazy:
            self.fbo[name] = None
            self._dims = getattr(self, "_dims", {})
            self._dims[name] = (w, h)
            return
        t = self.ctx.texture((w, h), 4, dtype="f2")
        f = moderngl.NEAREST if name == "mv" else moderngl.LINEAR                              # block vectors must not be interpolated
        t.filter = (f, f); t.repeat_x = t.repeat_y = False
        self.fbo[name] = (t, self.ctx.framebuffer([t]))

    def _get_fbo(self, name):
        if self.fbo[name] is None:
            w, h = self._dims[name]
            self._make_fbo(name, w, h)
        return self.fbo[name]

    # ---- uploads ----------------------------------------------------------------------------------------------
    def set_camera(self, rgb_display):
        """(H,W,3) uint8 in DISPLAY orientation (already mirrored if configured), top-left origin."""
        if rgb_display.shape[:2] != (self.H, self.W):
            rgb_display = cv2.resize(rgb_display, (self.W, self.H))
        self.cam, self.cam_prev = self.cam_prev, self.cam                                      # keep the previous frame for motion estimation
        self.cam.write(np.ascontiguousarray(rgb_display[::-1]).tobytes())
        self.cam.build_mipmaps()
        self.cam_seq += 1
        small = np.ascontiguousarray(cv2.resize(rgb_display, (HIST_W, HIST_H), interpolation=cv2.INTER_AREA)[::-1])
        self.hist_n += 1
        n = self.hist_n
        self.hist_recent = (self.hist_recent + [small])[-2:]
        if not self.hist_filled:                                                               # first frame fills both arrays (no black frames)
            tile = np.tile(small[None], (HIST_N, 1, 1, 1)).tobytes()
            for h_ in self.hists:
                h_.write(tile)
            self.hist_seen, self.hist_filled = [n, n], True
        else:
            idle = 1 - self.hcur
            for m in range(self.hist_seen[idle] + 1, n + 1):                                    # frames the idle array missed (at most the last two)
                self.hists[idle].write(self.hist_recent[m - n - 1].tobytes(), viewport=(0, 0, m % HIST_N, HIST_W, HIST_H, 1))
            self.hist_seen[idle] = n
            self.hcur = idle
        self.hist_head = n % HIST_N

    def set_mask(self, m_display):
        """(144,256) float 0..1 in DISPLAY orientation."""
        u = (np.clip(m_display, 0, 1)[::-1] * 255).astype(np.uint8)
        self.mask.write(np.dstack([u, u, u, np.full_like(u, 255)]).tobytes())

    def set_hud(self, lines):
        img = np.zeros((120, 560, 4), np.uint8)
        img[..., 3] = 150
        for i, ln in enumerate(lines[:4]):
            cv2.putText(img, ln, (8, 24 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255, 255), 1, cv2.LINE_AA)
        self.hud.write(np.ascontiguousarray(img[::-1]).tobytes())

    # ---- drawing ----------------------------------------------------------------------------------------------
    def _pass(self, name, target, inputs, uniforms=None, size=None, rect=(-1, -1, 1, 1), blend=None):
        prog, vao = self.progs[name]
        w, h = size
        for i, t in enumerate(inputs):
            t.use(i)
        if "sTD2DInputs" in prog:
            prog["sTD2DInputs"].value = list(range(len(inputs))) + [0] * (4 - len(inputs))
        if "uTDOutputInfo.res" in prog:
            prog["uTDOutputInfo.res"].value = (1.0 / w, 1.0 / h, float(w), float(h))
        prog["uRect"].value = rect
        for k, v in (uniforms or {}).items():
            if k in prog:
                prog[k].value = v
        target.use()
        target.viewport = (0, 0, w, h)
        if blend == "alpha":
            self.ctx.enable(moderngl.BLEND); self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        elif blend == "add":
            self.ctx.enable(moderngl.BLEND); self.ctx.blend_func = moderngl.ONE, moderngl.ONE
        else:
            self.ctx.disable(moderngl.BLEND)
        vao.render(moderngl.TRIANGLE_STRIP)

    def _effect(self, name, view, t):
        """Render one effect into its own buffer; returns its texture."""
        W, H = self.W, self.H
        tex, fbo = self._get_fbo(name)
        sp, tw = view["speed"], min(1.0, view["twist"] / 90.0)
        mx, my = view["motion"]
        common = {"uR": (t, sp, view["open"], tw), "uD": (mx, -my, sp, tw)}
        ins, uni = [self.cam, self.mask], dict(common)
        if name == "particles":
            uni.update(uA=(t, self.cell, 6.0, 1.0), uB=(0.9, 1.1, 1.0, 0.0))
        elif name == "moon":
            mxx, myy, mr = view["moon"]
            ins, uni = [self.mask], {"uM": (mxx, myy, mr, t), "uB": (0.55, 0.95, 0.0, 0.0)}
        elif name == "ascii":
            ins = [self.cam, self.mask, self.atlas]
            uni.update(uA=(10.0, 20.0, float(len(RAMP)), 0.0))
        elif name == "bitmap":
            uni.update(uA=(3.0, 1.6, 0.0, 0.0))
        elif name == "web":
            uni.update(uA=(13.0, 42.0, 30.0, 0.0))
        elif name == "asteroid":
            ins, uni = [self.cam], {"uA": (t, 0.46, 0.12, 0.0), "uB": (0.0, 1.25, -0.16, -0.02)}
        elif name == "ghost":
            uni = {"uA": (t, 0.05, 0.9, 0.0)}
        q = view["q"]
        uni["uC"] = (sum(q[0::2]) / 4.0, sum(q[1::2]) / 4.0, view["width"], 0.0)              # box centre (uv) for effects that radiate from it
        if name == "slitscan":
            self.hists[self.hcur].use(5)
            uni.update(sHist=5, uH=(float(self.hist_head), float(HIST_N), 0.0, 0.0))
        elif name == "datamosh":
            return self._datamosh(view, t, uni)
        self._pass(name, fbo, ins, uni, self._dims.get(name, (W, H)))
        return tex

    def _datamosh(self, view, t, uni):
        """Codec-style feedback: runs once per camera frame; the output buffer persists between frames (ping-pong)."""
        W, H = self.W, self.H
        if self._mosh_frame < self.frame_id - 1:                                                # was not visible last frame: restart from the live picture
            st_t, st_f = self._get_fbo("datamosh")
            self._pass("hud", st_f, [self.cam], None, (W, H))
            self._get_fbo("mv")[1].clear(0.0, 0.0, 0.0, 1.0)
            self._mosh_seq = self.cam_seq
        if self.cam_seq != self._mosh_seq:
            mv_t, mv_f = self._get_fbo("mv")
            self._pass("mosh_mv", mv_f, [self.cam, self.cam_prev], None, (W // MOSH_BLOCK, H // MOSH_BLOCK))
            st_t, st_f = self._get_fbo("datamosh")
            nb_t, nb_f = self._get_fbo("datamosh_b")
            uni = dict(uni, uM=(0.012, (t / 9.0) % 1.2 - 0.1, min(1.0, view["speed"] * 1.3), 0.0))
            self._pass("datamosh", nb_f, [st_t, self.cam, mv_t], uni, (W, H))
            self.fbo["datamosh"], self.fbo["datamosh_b"] = self.fbo["datamosh_b"], self.fbo["datamosh"]
            self._mosh_seq = self.cam_seq
        self._mosh_frame = self.frame_id
        return self.fbo["datamosh"][0]

    def draw(self, screen, screen_size, view, t, show_hud=True):
        """view: dict from handcore.view.ViewState.update. Renders the whole frame into `screen` (a framebuffer)."""
        W, H = self.W, self.H
        self.frame_id += 1
        bl_t, bl_f = self.fbo["blend"]
        bl_f.use(); self.ctx.disable(moderngl.BLEND); bl_f.clear(0.0, 0.0, 0.0, 1.0)
        for name, w in view["weights"].items():
            if w > 0.002:                                                    # only what is visible; heavy shaders cost nothing when off
                tex = self._effect(name, view, t)
                self._pass("acc", bl_f, [tex], {"uW": (w, 0, 0, 0)}, (W, H), blend="add")
        s1_t, s1_f = self.fbo["small1"]; s2_t, s2_f = self.fbo["small2"]
        sw, sh = W // 4, H // 4                                              # bloom source: blurred 1/4-res copy of the scene
        self._pass("hud", s1_f, [bl_t], None, (sw, sh))
        self._pass("blur", s2_f, [s1_t], {"uDir": (1.0 / sw, 0.0)}, (sw, sh))
        self._pass("blur", s1_f, [s2_t], {"uDir": (0.0, 1.0 / sh)}, (sw, sh))
        q = view["q"]
        self._pass("composite", screen, [self.cam, bl_t, s1_t],
                   {"uQA": tuple(q[0:4]), "uQB": tuple(q[4:8]), "uLA": tuple(view["a"]), "uLB": tuple(view["l"]),
                    "uP1": (view["alpha"], 0.0, t, 1.0), "uP2": (view["par"][0], view["par"][1], 0.35, 2.0),
                    "uP3": (0.0, view["lid_on"], 0.0, 0.0), "uIn": tuple(view["insets"]), "uG": (view["speed"], t, 0.0, 0.0)}, screen_size)
        if show_hud and self.hud_on:
            sw_, sh_ = screen_size
            x1 = -1 + 2 * 560 / sw_; y0 = 1 - 2 * 120 / sh_
            self._pass("hud", screen, [self.hud], None, screen_size, rect=(-1, y0, x1, 1), blend="alpha")
