"""Per-frame view parameters derived from the tracker + presenter output (used by the standalone app).

Pure Python. Corners come in display pixels (top-left origin); everything returned is in GL/TD uv (bottom-left origin).
Purely visual smoothing (opening, twist dial cross-fades, 3D twist of the box) lives here: lerps are fine for visuals, never for tracking.

Controls (no keys):
  openness (whole-hand pinch <-> open) opens / closes the box;
  twist    (palm normal vs camera) is a DIAL through the effects, cross-faded; it also pushes each hand's side of the box deeper, so
           twisting a hand visibly twists the box in 3D;
  speed    (hand velocity) is a reactivity signal for the effects (glitch, streaks, ripples).
"""
from __future__ import annotations
import math
from .predict import draw_order

ALL_EFFECTS = ("particles", "moon", "glass", "ascii", "bitmap", "web", "neon", "asteroid", "ghost", "slitscan", "moire", "pixelsort", "crt", "datamosh")


def smoothstep(a, b, x):
    t = min(1.0, max(0.0, (x - a) / (b - a)))
    return t * t * (3 - 2 * t)


def expand(name):
    return ("particles", "moon") if name == "particles_moon" else (name,)


class TwistDial:
    """Relative twist ratchet. `rel = twist - base`; |rel| past `step` (held `commit_s`) commits to the neighbouring effect and re-bases, so
    each further twist steps once more. Partial twists PREVIEW the neighbour (analog cross-fade). `base` follows slow drift when the hands rest."""

    def __init__(self, names, step=20.0, commit_s=0.25, base_tau=6.0, ease_s=0.25):
        self.names, self.step, self.commit_s, self.base_tau, self.ease_s = list(names), step, commit_s, base_tau, ease_s
        self.k = 0
        self.base = None
        self.t_over = 0.0
        self.w = {n: 0.0 for n in ALL_EFFECTS}
        for n in expand(self.names[0]):
            self.w[n] = 1.0
        self.rel = 0.0

    def force(self, name):
        """Jump straight to an effect (tests / previews)."""
        self.k = self.names.index(name)
        self.w = {n: 0.0 for n in ALL_EFFECTS}
        for n in expand(name):
            self.w[n] = 1.0

    def update(self, twist, dt, active=True):
        if self.base is None:
            self.base = twist
        if active:
            rel = twist - self.base
            if abs(rel) < 0.5 * self.step:
                self.base += (twist - self.base) * min(1.0, dt / self.base_tau)       # follow slow drift of the resting pose
            if abs(rel) >= self.step:
                self.t_over += dt
                if self.t_over >= self.commit_s:
                    self.k = (self.k + (1 if rel > 0 else -1)) % len(self.names)
                    self.base, self.t_over, rel = twist, 0.0, 0.0
            else:
                self.t_over = 0.0
            self.rel = rel
        n = len(self.names)
        p = smoothstep(0.35, 1.0, min(1.0, abs(self.rel) / self.step))               # dead-zone, then analog preview of the neighbour
        nb = self.names[(self.k + (1 if self.rel > 0 else -1)) % n]
        tgt = {m: 0.0 for m in ALL_EFFECTS}
        for m in expand(self.names[self.k]):
            tgt[m] = max(tgt[m], 1.0 - p)
        for m in expand(nb):
            tgt[m] = max(tgt[m], p)
        step = min(1.0, dt / self.ease_s)
        self.w = {m: self.w[m] + (tgt[m] - self.w[m]) * step for m in ALL_EFFECTS}
        return self.w


class ViewState:
    def __init__(self, gesture=None, fade_s=0.3, parallax_pos=0.06, parallax_tilt=0.05):
        from .config import Gesture
        self.g = gesture or Gesture()
        self.twist = 35.0
        self.dial = TwistDial(self.g.dial_effects, self.g.dial_step_deg, self.g.dial_commit_s, self.g.dial_base_tau_s, self.g.dial_ease_s)
        self.openness = 0.0
        self.speed = 0.0
        self.motion = [0.0, 0.0]
        self.inset = 0.8
        self.pp, self.pt = parallax_pos, parallax_tilt
        self.tw = [35.0, 35.0]

    @staticmethod
    def _approach(cur, target, step):
        return cur + max(-step, min(step, target - cur))

    def update(self, c, alpha, dt, pose, W, H):
        """c: 12 floats = 4 corners in id order [thumb0, anchor0, anchor1, thumb1] + lid points [lid0, lid1] (px, top-left origin).
        pose: dict(openness, twist, twists, speed, motion) from PortalState."""
        g = self.g
        order = draw_order(c[:8])
        uv = lambda i: (c[2 * i] / W, 1.0 - c[2 * i + 1] / H)
        q = []
        for i in order:
            q += list(uv(i))
        a0, a1 = uv(1), uv(2)
        l0, l1 = (c[8] / W, 1.0 - c[9] / H), (c[10] / W, 1.0 - c[11] / H)
        width = math.hypot((a1[0] - a0[0]) * W, (a1[1] - a0[1]) * H) / W

        # opening: follows the whole-hand openness (pinch = closed); a touch faster to open than to close
        tgt = pose["openness"]
        rate = 1.0 / (0.16 if tgt > self.openness else 0.10)
        self.openness = self._approach(self.openness, tgt, dt * rate)
        o = self.openness
        # reactivity signals, lightly smoothed
        self.speed += (pose["speed"] - self.speed) * min(1.0, dt * 8.0)
        self.motion = [m + (n - m) * min(1.0, dt * 6.0) for m, n in zip(self.motion, pose["motion"])]
        self.twist += (pose["twist"] - self.twist) * min(1.0, dt * 8.0)
        self.tw = [t + (n - t) * min(1.0, dt * 8.0) for t, n in zip(self.tw, pose["twists"])]
        weights = self.dial.update(self.twist, dt, active=self.openness > 0.5)
        base = self.dial.base if self.dial.base is not None else self.twist

        cx, cy = sum(q[0::2]) / 4.0, sum(q[1::2]) / 4.0
        area = 0.5 * abs(sum(q[2 * i] * q[2 * ((i + 1) % 4) + 1] - q[2 * ((i + 1) % 4)] * q[2 * i + 1] for i in range(4)))

        def d(i, j):
            return math.hypot(q[2 * i] - q[2 * j], q[2 * i + 1] - q[2 * j + 1])
        e = [d(0, 1), d(1, 2), d(2, 3), d(3, 0)]
        tilt_x = (e[1] - e[3]) / max(e[1] + e[3], 1e-6)
        tilt_y = (e[0] - e[2]) / max(e[0] + e[2], 1e-6)
        base = min(0.9, max(0.62, 0.55 + 1.0 * area))
        self.inset += (base - self.inset) * min(1.0, dt * 6.0)
        # 3D twist of the box: a hand turned edge-on pushes ITS side of the box deeper (ids 0,1 = hand 0; ids 2,3 = hand 1)
        per_id = []
        for hid in (0, 1):
            s = min(0.95, max(0.4, self.inset - g.twist_depth * max(-1.0, min(1.5, (self.tw[hid] - base) / 60.0))))     # twisted PAST your resting twist: that side recedes
            per_id += [s, s]
        insets = [1.0 - (1.0 - per_id[i]) * o for i in order]                    # draw order, folded by the opening
        lid_len = math.hypot((l0[0] - a0[0]) * W, (l0[1] - a0[1]) * H) + math.hypot((l1[0] - a1[0]) * W, (l1[1] - a1[1]) * H)
        lid_on = smoothstep(6.0, 30.0, lid_len / 2.0)
        R = 0.07 * min(1.5, max(0.6, width / 0.5))
        am = ((a0[0] + a1[0]) / 2, (a0[1] + a1[1]) / 2)
        moon = (am[0], am[1] - 1.15 * R, R)
        return {"q": q, "a": [a0[0], a0[1], a1[0], a1[1]], "l": [l0[0], l0[1], l1[0], l1[1]], "alpha": alpha * smoothstep(0.02, 0.35, o),
                "insets": insets, "lid_on": lid_on * o, "weights": weights, "moon": moon, "open": o,
                "par": (self.pp * (cx - 0.5) + self.pt * tilt_x, self.pp * (cy - 0.5) + self.pt * tilt_y),
                "speed": self.speed, "motion": list(self.motion), "twist": self.twist, "rel_twist": self.dial.rel, "effect_index": self.dial.k, "twists": list(self.tw), "width": width}
