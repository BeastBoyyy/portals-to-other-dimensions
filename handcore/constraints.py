"""Kinematic tip constraint (spec 4.5) + palm-anchor reliability.

A tip cannot be farther from its previous joint (thumb IP / index DIP) than the bone length
allows; foreshortening only ever *shortens* the projected bone, so the constraint is an upper
bound on the 2D distance. Pixels-per-metre is estimated from the least-foreshortened bone.
"""
from __future__ import annotations
import numpy as np

SCALE_BONES = ((0, 5), (0, 17), (5, 9), (9, 13), (13, 17), (0, 9), (5, 17))
DEFAULT_PALM_M = 0.075   # only used when no world landmarks are available


def px_per_metre(px, world):
    if world is None:
        return float(max(np.linalg.norm(px[a, :2] - px[b, :2]) for a, b in SCALE_BONES) / DEFAULT_PALM_M)
    r = [np.linalg.norm(px[a, :2] - px[b, :2]) / max(np.linalg.norm(world[a] - world[b]), 1e-6) for a, b in SCALE_BONES]
    return float(max(r))


class TipConstraint:
    """Holds (optionally calibrated) segment length / palm-width ratios for thumb and index."""

    def __init__(self, cfg):
        self.c = cfg
        self.ratios = {"thumb": cfg.thumb_ratio, "index": cfg.index_ratio}
        self._calib = {"thumb": [], "index": []}

    def calibrate(self, px, world):
        """Feed valid open-hand frames during the optional calibration pose; learns segment/palm ratios."""
        if world is None:
            return
        pw = np.linalg.norm(world[5] - world[17])
        self._calib["thumb"].append(np.linalg.norm(world[4] - world[3]) / pw)
        self._calib["index"].append(np.linalg.norm(world[8] - world[7]) / pw)
        for k, v in self._calib.items():
            if len(v) >= max(self.c.calibrate_frames, 1):
                self.ratios[k] = float(np.median(v))

    def apply(self, px, world, score):
        """Returns (thumb_xy, index_xy, flags) after constraint; px is (21,3) pixel landmarks."""
        out = [px[4, :2].copy(), px[8, :2].copy()]
        flags = [False, False]
        if not self.c.enable:
            return out[0], out[1], flags
        s = px_per_metre(px, world)
        pw = np.linalg.norm(world[5] - world[17]) if world is not None else DEFAULT_PALM_M
        w = float(np.clip(score, 0.0, 1.0))
        for k, (tip, joint, name) in enumerate(((4, 3, "thumb"), (8, 7, "index"))):
            R = self.ratios[name] * pw * s * (1.0 + self.c.slack)
            v = px[tip, :2] - px[joint, :2]
            d = float(np.hypot(*v))
            if d > R > 0:
                proj = px[joint, :2] + v * (R / d)
                out[k] = px[tip, :2] + w * (proj - px[tip, :2])
                flags[k] = True
        return out[0], out[1], flags


def anchor_gate(slot, raw_tips, centre, palm_px, c, t, loosen=1.0):
    """Palm-anchor reliability (spec 4.5). A tip's offset from the palm centre moves smoothly, so a deviation from
    its constant-velocity prediction larger than `spike_palm_frac` palm sizes is a spike (thumb snap under
    self-occlusion), not motion. While a spike lasts the tip follows the prediction (still tracking the palm). A
    spike that persists at the same new place for more than `spike_persist` frames is real and is accepted.
    Fast steady motion (hand rotation) is predicted, so it is never mistaken for a spike.
    raw_tips (4,) -> (4,) gated tips."""
    rel = raw_tips.reshape(2, 2) - centre
    if not c.anchor_spike or slot.rel_good is None:
        slot.rel_good, slot.rel_raw, slot.strikes = rel.copy(), rel.copy(), [0, 0]
        slot.rel_vel, slot.t_gate = np.zeros((2, 2)), t
        return raw_tips
    dt = min(max(t - slot.t_gate, 1e-3), 0.1)
    slot.t_gate = t
    pred = slot.rel_good + slot.rel_vel * dt
    out = rel.copy()
    for k in (0, 1):
        # allowance grows with the offset's own speed: fast steady motion is predicted less exactly
        thr = (c.spike_palm_frac * palm_px + c.spike_speed_gain * float(np.hypot(*slot.rel_vel[k])) * dt) * loosen
        alt = slot.alt[k]
        if alt is not None:                                     # remember where the tip was before an accepted spike
            alt[0] = alt[0] + alt[1] * dt
            alt[2] -= 1
            if alt[2] <= 0:
                slot.alt[k] = alt = None
        dev = np.hypot(*(rel[k] - pred[k]))
        if dev <= thr:
            if dev <= 0.5 * thr:                                # only trust clearly smooth samples for velocity
                slot.rel_vel[k] += c.spike_vel_gain * ((rel[k] - slot.rel_good[k]) / dt - slot.rel_vel[k])
            else:
                slot.rel_vel[k] *= 0.6
            slot.rel_good[k] = rel[k]
            slot.strikes[k] = 0
            slot.alt[k] = None
        elif alt is not None and np.hypot(*(rel[k] - alt[0])) <= thr:   # the "spike" we accepted was a snap: roll back
            slot.rel_good[k], slot.rel_vel[k], slot.strikes[k], slot.alt[k] = rel[k], alt[1].copy(), 0, None
        elif np.hypot(*(rel[k] - slot.rel_raw[k])) <= thr:      # settled at a new place: counting towards "real"
            slot.strikes[k] += 1
            if slot.strikes[k] > c.spike_persist:
                slot.alt[k] = [pred[k].copy(), slot.rel_vel[k].copy(), 6]
                slot.rel_good[k] = rel[k]
                slot.rel_vel[k] = 0.0
                slot.strikes[k] = 0
            else:
                slot.rel_good[k] = out[k] = pred[k]
                slot.rel_vel[k] *= 0.6
        else:                                                   # still jumping around: follow the prediction
            slot.strikes[k] = 1
            slot.rel_good[k] = out[k] = pred[k]
            slot.rel_vel[k] *= 0.6
    slot.rel_raw = rel.copy()
    return (out + centre).reshape(4)
