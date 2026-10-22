"""Synthetic landmark clips with exact ground truth (no webcam needed).

A parametric 21-landmark hand (metric, anatomically proportioned from real MediaPipe output) is posed in
the "frame gesture" (index up, thumb out = L shape), moved along scripted trajectories, projected with a
pinhole camera, then corrupted by a *detector noise model* (correlated jitter, thumb snaps, teleporting
re-detections, missed detections, handedness flips, ghosts, timestamp jitter).

IMPORTANT: the noise model is an assumption, not a measurement of MediaPipe. Numbers produced on these
clips measure handcore *relative to that model*; they are not real-world accuracy claims. See REPORT.md.
Coordinates are RAW camera coordinates (un-mirrored); handcore mirrors on ingest.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
import numpy as np
from handcore.types import Frame, Hand
from handcore.config import Quad as QuadCfg
from handcore import quad as quadmod

W, H = 1280, 720
HFOV = math.radians(65)
F = W / (2 * math.tan(HFOV / 2))

# ---- hand model -------------------------------------------------------------------------------
_SX = {5: 0.036, 9: 0.012, 13: -0.012, 17: -0.036}
_MCP_Y = {5: 0.085, 9: 0.090, 13: 0.085, 17: 0.075}
_SEG = {5: (0.043, 0.026, 0.027), 9: (0.048, 0.028, 0.024), 13: (0.044, 0.027, 0.024), 17: (0.034, 0.019, 0.019)}


def hand_local(chirality=1, curl=(0.0, 1.0, 1.0, 1.0), thumb_alpha=(60.0, 75.0, 80.0), splay=0.0):
    """(21,3) metric landmarks, wrist at origin, +y toward fingers, thumb on the +x*chirality side."""
    p = np.zeros((21, 3))
    c = chirality
    for k in (5, 9, 13, 17):
        p[k] = (c * _SX[k], _MCP_Y[k], 0.0)
    # fingers: index extended (curl[0]=0), others curled by curl factor
    for fi, k in enumerate((5, 9, 13, 17)):
        cu = curl[fi]
        phi, pos = 0.0, p[k].copy()
        for j, ln in enumerate(_SEG[k]):
            phi += math.radians(75.0 * cu if j else 85.0 * cu)
            pos = pos + ln * np.array([0.0, math.cos(phi), math.sin(phi)])
            p[k + 1 + j] = pos
    # thumb: CMC(1) near wrist, MCP(2), IP(3), tip(4) abducted into an L
    p[1] = (c * 0.022, 0.022, 0.0)
    p[2] = (c * 0.038, 0.047, 0.004)
    pos, lens = p[2].copy(), (0.030, 0.030)
    for j, ln in enumerate(lens):
        a = math.radians(thumb_alpha[j])
        pos = pos + ln * np.array([c * math.sin(a), math.cos(a), 0.0])
        p[3 + j] = pos
    if splay:                                   # fan the four fingers out in the palm plane (index toward the thumb side)
        for fi, k in enumerate((5, 9, 13, 17)):
            a = -math.radians((12.0, 4.0, -4.0, -14.0)[fi]) * splay * c
            R2 = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
            for j in range(k + 1, k + 4):
                p[j, :2] = p[k, :2] + R2 @ (p[j, :2] - p[k, :2])
    return p


def _rot(roll, yaw, pitch):
    cr, sr, cy, sy, cp, sp = math.cos(roll), math.sin(roll), math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    return Rz @ Ry @ Rx


def project(local, pose):
    """pose=(X,Y,Z,roll,yaw,pitch) -> normalized lm (21,3), world (21,3) metres, px (21,2)."""
    X, Y, Z, roll, yaw, pitch = pose
    R = _rot(roll, yaw, pitch)
    P = local @ R.T + np.array([X, Y, Z])
    u = W / 2 + F * P[:, 0] / P[:, 2]
    v = H / 2 - F * P[:, 1] / P[:, 2]
    lm = np.stack([u / W, v / H, (P[:, 2] - P[0, 2]) * F / P[0, 2] / W * -1.0], axis=1)
    world = (local @ R.T)
    world = world - world.mean(0)
    return lm, world, np.stack([u, v], 1)


# ---- trajectories -----------------------------------------------------------------------------
def _smooth(a, b, t0, t1, t):
    s = np.clip((t - t0) / (t1 - t0), 0, 1)
    return a + (b - a) * (0.5 - 0.5 * math.cos(math.pi * s))


def make_pose_fn(kind, hid, dur):
    sgn = -1 if hid == 0 else 1       # hand 0 raw-left, hand 1 raw-right
    base = (sgn * 0.20, -0.15, 0.70)

    def still(t): return (*base, 0, 0, 0)

    def slow(t):
        return (base[0] + 0.08 * math.sin(2 * math.pi * 0.3 * t + hid), base[1] + 0.04 * math.sin(2 * math.pi * 0.2 * t),
                base[2] + 0.05 * math.sin(2 * math.pi * 0.15 * t), 0.15 * math.sin(2 * math.pi * 0.25 * t), 0, 0)

    def fast(t):
        w = 2 * math.pi * 1.6
        return (base[0] + 0.12 * math.sin(w * t + 0.7 * hid), base[1] + 0.07 * math.sin(1.3 * w * t),
                base[2] + 0.06 * math.sin(0.7 * w * t), 0.3 * math.sin(w * t), 0.2 * math.sin(0.9 * w * t), 0)

    def tilt(t):
        w = 2 * math.pi * 0.9
        return (base[0], base[1], base[2], 1.0 * math.sin(w * t + hid), 0.5 * math.sin(0.7 * w * t), 0.4 * math.sin(0.5 * w * t))

    def crossing(t):
        T0, T1 = 3.0, 4.5
        x = _smooth(base[0], -base[0], T0, T1, t)
        x = _smooth(x, base[0], dur * 0.65, dur * 0.65 + 1.5, t) if t > dur * 0.65 else x
        return (x, base[1], base[2] + (0.04 if hid == 0 else -0.04), 0, 0, 0)

    def near(t):
        T0 = 3.0
        gap = 0.03
        x = base[0] if t < T0 else _smooth(base[0], sgn * gap, T0, T0 + 1.2, t)
        if t > T0 + 1.6:
            x = _smooth(sgn * gap, base[0], T0 + 1.6, T0 + 2.8, t)
        return (x, base[1], base[2] + (0.03 if hid == 0 else -0.03), 0, 0, 0)

    def reentry(t):
        # leave through the side, absent 1 s, return; two cycles
        out = 0.75 * sgn
        x = base[0]
        for t0 in (dur * 0.2, dur * 0.6):
            if t0 <= t < t0 + 0.8:
                x = _smooth(base[0], out, t0, t0 + 0.8, t)
            elif t0 + 0.8 <= t < t0 + 1.8:
                x = out
            elif t0 + 1.8 <= t < t0 + 2.6:
                x = _smooth(out, base[0], t0 + 1.8, t0 + 2.6, t)
        return (x, base[1], base[2], 0, 0, 0)

    return {"still": still, "slow": slow, "fast": fast, "tilt": tilt, "crossing": crossing, "near_cross": near,
            "reentry": reentry, "one_hand": slow, "no_hands": still, "lowlight": slow, "backlit": slow,
            "occluded": slow}[kind]


# ---- noise model --------------------------------------------------------------------------------
@dataclass
class Noise:
    sigma_tip: float = 1.0        # px at palm size 100 px
    sigma_other: float = 0.6
    rho: float = 0.6              # AR(1) temporal correlation
    thumb_snap_p: float = 0.01
    thumb_snap_px: tuple = (25, 55)
    teleport_p: float = 0.003
    miss_p: float = 0.01
    label_flip_p: float = 0.08
    score: tuple = (0.88, 1.0)
    fps: float = 30.0
    t_jitter: float = 0.0015
    frame_drop_p: float = 0.005
    cross_miss_p: float = 0.5     # missed-detection prob when hands overlap

    def scaled(self, k, **kw):
        n = Noise(**{**self.__dict__, **kw})
        n.sigma_tip *= k; n.sigma_other *= k
        return n


PRESETS = {
    "lowlight": dict(k=2.5, miss_p=0.05, score=(0.72, 0.95), thumb_snap_p=0.03),
    "backlit": dict(k=1.6, miss_p=0.03, score=(0.8, 0.97)),
    "occluded": dict(k=1.0, thumb_snap_p=0.06),
}


@dataclass
class Clip:
    kind: str
    frames: list
    dur: float
    pose: dict
    local: dict
    hidden: dict
    present: tuple
    noise: Noise
    meta: dict = field(default_factory=dict)

    def gt(self, t, hid):
        """(tips (2,2) raw px [thumb,index], visible) for hand hid at time t."""
        if hid not in self.present:
            return np.zeros((2, 2)), False
        _, _, px = project(self.local[hid], self.pose[hid](t))
        tips = px[[4, 8]]
        vis = bool(np.all((px[:, 0] > 0) & (px[:, 0] < W) & (px[:, 1] > 0) & (px[:, 1] < H)))
        vis = vis and not any(a <= t < b for a, b in self.hidden.get(hid, ()))
        return tips, vis

    def gt_corners(self, t, mirror=True):
        """Ground-truth portal corners in HANDCORE id order, display coordinates; None unless both visible."""
        out = []
        for hid in ((1, 0) if mirror else (0, 1)):    # slot 0 = display-left hand at acquisition
            tips, vis = self.gt(t, hid)
            if not vis:
                return None
            out.append(tips)
        c = np.array([out[0][0], out[0][1], out[1][1], out[1][0]])
        if mirror:
            c[:, 0] = W - c[:, 0]
        order, ok, _ = quadmod.build(c, W, H, QuadCfg())     # a non-convex ground-truth quad is not a valid portal
        return c if ok else None


def _plausible_ghost(rng, centre_px, scale=1.0, distort=False):
    local = hand_local(1, curl=(1, 1, 1, 1))
    if distort:
        local = local * rng.uniform(0.4, 1.9, size=(21, 3))
    pose = (0, 0, 1.0, rng.uniform(-.3, .3), 0, 0)
    lm, world, _ = project(local, pose)
    lm[:, 0] += centre_px[0] / W - lm[:, 0].mean()
    lm[:, 1] += centre_px[1] / H - lm[:, 1].mean()
    return Hand(lm, world, "Left", float(rng.uniform(0.8, 0.97)), 1.0, -1)


def make_clip(kind, seed=0, dur=20.0, noise: Noise | None = None, ghosts=None) -> Clip:
    """ghosts: iterable of 'poster' | 'poster2' | 'flicker' | 'random' adding false hands."""
    rng = np.random.default_rng(seed)
    n = noise or Noise()
    if kind in PRESETS:
        pr = dict(PRESETS[kind]); k = pr.pop("k")
        n = n.scaled(k, **pr)
    present = (0,) if kind == "one_hand" else (() if kind == "no_hands" else (0, 1))
    pose = {h: make_pose_fn(kind, h, dur) for h in (0, 1)}
    local = {0: hand_local(1), 1: hand_local(-1)}      # chirality: hand 0 'Left'-like, hand 1 'Right'-like
    hidden = {}
    if kind == "occluded":
        hidden = {0: [(dur * 0.4, dur * 0.4 + 0.3)], 1: []}
    ghosts = list(ghosts or ())
    poster_px = [(190, 200), (1120, 170)]
    poster = [_plausible_ghost(rng, poster_px[0]), _plausible_ghost(rng, poster_px[1])]
    ar = {h: np.zeros((21, 3)) for h in (0, 1)}
    snap = {h: 0 for h in (0, 1)}
    snap_vec = {h: np.zeros(2) for h in (0, 1)}
    clip = Clip(kind, [], dur, pose, local, hidden, present, n, {"ghosts": ghosts, "poster_px": poster_px})
    t_nom = 0.0
    last_t = -1.0
    while t_nom < dur:
        t = t_nom + rng.normal(0, n.t_jitter)
        t_nom += (2.0 if rng.random() < n.frame_drop_p else 1.0) / n.fps
        if t <= last_t:
            continue
        last_t = t
        hands = []
        centres = []
        for hid in present:
            tips, vis = clip.gt(t, hid)
            if not vis:
                continue
            lm, world, px = project(local[hid], pose[hid](t))
            centres.append(px[[0, 5, 9, 13, 17]].mean(0))
            palm = np.linalg.norm(px[5] - px[17])
            k = palm / 100.0
            eps = rng.normal(size=(21, 3)) * np.sqrt(1 - n.rho ** 2)
            ar[hid] = n.rho * ar[hid] + eps
            sig = np.full((21, 1), n.sigma_other)
            sig[[4, 8, 3, 7, 12, 16, 20]] = n.sigma_tip
            d_px = ar[hid][:, :2] * sig * k
            if snap[hid] > 0:
                snap[hid] -= 1
                d_px[4] += snap_vec[hid]
            elif rng.random() < n.thumb_snap_p:
                snap[hid] = int(rng.integers(0, 3))
                ang = rng.uniform(0, 2 * math.pi)
                snap_vec[hid] = rng.uniform(*n.thumb_snap_px) * k * np.array([math.cos(ang), math.sin(ang)])
                d_px[4] += snap_vec[hid]
            lm2 = lm.copy()
            lm2[:, 0] += d_px[:, 0] / W
            lm2[:, 1] += d_px[:, 1] / H
            if rng.random() < n.teleport_p:
                off = rng.uniform(80, 300) * np.array([math.cos(a := rng.uniform(0, 6.28)), math.sin(a)])
                lm2[:, 0] += off[0] / W
                lm2[:, 1] += off[1] / H
            if rng.random() < n.miss_p:
                continue
            label = "Left" if hid == 0 else "Right"
            if rng.random() < n.label_flip_p:
                label = "Right" if label == "Left" else "Left"
            hands.append(Hand(lm2, world + rng.normal(0, 0.0005, world.shape), label,
                              float(rng.uniform(*n.score)), 1.0, hid))
        if len(hands) == 2:
            c0, c1 = centres[0], centres[1]
            if np.hypot(*(c0 - c1)) < 1.2 * 100 and rng.random() < n.cross_miss_p:
                hands.pop(int(rng.integers(0, 2)))
        for g in ghosts:
            if g == "poster":
                gh = poster[0]
                hands.append(_jit(gh, rng))
            elif g == "poster2":
                hands += [_jit(poster[0], rng), _jit(poster[1], rng)]
            elif g == "flicker" and rng.random() < 0.15:
                hands.append(_plausible_ghost(rng, (rng.uniform(100, 1180), rng.uniform(100, 620)), distort=True))
            elif g == "random" and rng.random() < 0.06:
                hands.append(_plausible_ghost(rng, (rng.uniform(100, 1180), rng.uniform(100, 620))))
        mask = body_mask(centres, poster_px) if centres or kind == "no_hands" else body_mask([], poster_px)
        clip.frames.append(Frame(t, hands, mask))
    return clip


def _jit(h, rng):
    lm = h.lm.copy()
    lm[:, :2] += rng.normal(0, 0.0008, (21, 2))
    return Hand(lm, h.world, h.label, h.score, 1.0, -1)


def body_mask(hand_centres_px, poster_px, h=36, w=64):
    """Low-res 'selfie segmentation': a torso blob in the lower centre + blobs around real hands. Posters stay 0."""
    yy, xx = np.mgrid[0:h, 0:w]
    m = ((xx - w / 2) / (w * 0.30)) ** 2 + ((yy - h * 0.95) / (h * 0.65)) ** 2 < 1
    for c in hand_centres_px:
        cx, cy = c[0] / W * w, c[1] / H * h
        m |= ((xx - cx) / (w * 0.12)) ** 2 + ((yy - cy) / (h * 0.30)) ** 2 < 1
    return m.astype(np.float32)
