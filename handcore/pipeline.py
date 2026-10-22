"""HandCore: Frame -> PortalState at camera rate.
validate -> identify -> constrain -> filter -> build quad. Deterministic; no I/O."""
from __future__ import annotations
import numpy as np
from .config import Config
from .types import Frame, Hand, PortalState
from .validate import check_hand, palm_centre, palm_size_px, to_px
from .identity import Tracker, Det, EMPTY
from .constraints import TipConstraint, anchor_gate
from . import quad as quadmod
from .pose import smoothstep


def _dilate(m, r):
    """Separable square max-filter (radius r) of a boolean array, numpy only."""
    out = m.copy()
    for ax in (0, 1):
        acc = out.copy()
        n = out.shape[ax]
        for k in range(1, r + 1):
            a = [slice(None)] * 2; b = [slice(None)] * 2
            a[ax], b[ax] = slice(k, n), slice(0, n - k)
            acc[tuple(a)] |= out[tuple(b)]
            acc[tuple(b)] |= out[tuple(a)]
        out = acc
    return out


class HandCore:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.W, self.H = self.cfg.camera.width, self.cfg.camera.height
        self.reset()

    def reset(self):
        self.tracker = Tracker(self.cfg)
        self.constraint = TipConstraint(self.cfg.constraint)
        self.t_last = None
        self.last_valid = None        # (corners, vel, order)
        self.invalid_run = 0
        self.rejects = {}
        self.calib_n = 0
        self.last_state = None
        self.last_lid = None

    # -- the one place mirroring is decided -------------------------------------------------
    def ingest(self, frame: Frame):
        """Mirror, convert to pixels, run per-detection gates. Returns up to two Det."""
        cfg, g = self.cfg, self.cfg.gates
        mask = frame.person_mask if g.person_mask else None
        if mask is not None and cfg.camera.mirror:
            mask = mask[:, ::-1]                  # masks arrive in raw-camera orientation, like the landmarks
        if mask is not None and g.mask_dilate > 0:
            r = max(1, round(g.mask_dilate * mask.shape[1] / 256))      # mask_dilate is in cells of a 256-wide mask
            mask = _dilate(mask > 0.5, r).astype(np.float32)
        dets, self.rejects = [], {}
        for h in frame.hands:
            lm = np.asarray(h.lm, dtype=float).copy()
            world = None if h.world is None else np.asarray(h.world, dtype=float).copy()
            label = h.label
            if cfg.camera.mirror:
                lm[:, 0] = 1.0 - lm[:, 0]
                if world is not None:
                    world[:, 0] = -world[:, 0]
                label = {"Left": "Right", "Right": "Left"}.get(label, label)
            hh = Hand(lm, world, label, h.score, h.presence, h.gt_id)
            mask_ok = True
            if g.enable:
                ok, why = check_hand(hh, self.W, self.H, g, mask)
                if not ok and why == "person_mask" and g.mask_acquire_only:
                    mask_ok = False                       # keep the detection: it may only CONTINUE a track that already exists
                    self.rejects["off_mask"] = self.rejects.get("off_mask", 0) + 1
                elif not ok:
                    self.rejects[why] = self.rejects.get(why, 0) + 1
                    continue
            px = to_px(lm, self.W, self.H)
            sign = {"Left": 1.0, "Right": -1.0}.get(label, 0.0)
            dets.append(Det(hh, px, palm_centre(px), float(palm_size_px(px)), sign, mask_ok))
        dets.sort(key=lambda d: -d.hand.score)        # single-user rule: best two only
        return dets[:2]

    # --------------------------------------------------------------------------------------
    def update(self, frame: Frame) -> PortalState:
        t = frame.t
        if self.t_last is not None:
            if t <= self.t_last:                        # duplicate / non-monotonic: drop frame
                return self.last_state
            if t - self.t_last > 1.0:                   # clock jump / long stall: start clean
                self.reset()
        self.t_last = t
        cfg = self.cfg
        dets = self.ingest(frame)
        if cfg.constraint.calibrate_frames > 0 and self.calib_n < cfg.constraint.calibrate_frames:
            for d in dets:
                self.constraint.calibrate(d.px, d.hand.world)
                self.calib_n += 1
        assign = self.tracker.update(t, dets)
        flags = {}
        for k, d in assign.items():
            s = self.tracker.slots[k]
            th, ix, fl = self.constraint.apply(d.px, d.hand.world, d.hand.score)
            # the box is ALWAYS attached to the same fingers: thumb tip (bottom), index tip (top corner), middle+ring tips (glass lid)
            anchor = ix
            lid = (d.px[12, :2] + d.px[16, :2]) / 2.0
            if s.missed > 4 or (s.missed and np.hypot(*(d.centre - s.kf.pos)) > 1.5 * d.palm):
                s.rearm(t, d)
            s.pose.update(t, d.px, d.hand.world, d.palm)
            raw = np.array([th[0], th[1], anchor[0], anchor[1]])
            # pinching / opening moves the fingertips fast relative to the palm ON PURPOSE: do not mistake that for a snap
            raw = anchor_gate(s, raw, d.centre, d.palm, cfg.constraint, t, loosen=1.0 + 3.0 * min(1.0, s.pose.rate / 1.5))
            s.measure(t, d, raw)
            s.measure_lid(t, d, lid)
            flags[k] = fl
        tips, vels, lids = {}, {}, {}
        for s in self.tracker.slots:
            if s.state == EMPTY:
                continue
            if s.idx in assign:
                tips[s.idx], vels[s.idx], lids[s.idx] = s.tips, s.vel, s.lid
            else:
                tips[s.idx], vels[s.idx] = s.coast()
                lids[s.idx] = s.coast_lid()
        state = self._quad_state(t, tips, vels, assign, flags)
        if 0 in lids and 1 in lids and state.valid:
            self.last_lid = np.array([lids[0], lids[1]])
            state.lid_vel = np.array([self.tracker.slots[0].vc, self.tracker.slots[1].vc])
        state.lid = self.last_lid.copy() if self.last_lid is not None else state.corners[[1, 2]].copy()
        if state.lid_vel is None:
            state.lid_vel = np.zeros((2, 2))
        self._pose_outputs(state)
        self.last_state = state
        return state

    def _pose_outputs(self, state):
        """Whole-hand pose of the tracked hands -> openness / twist / speed (visual drivers; the tracking above never depends on them)."""
        g = self.cfg.gesture
        poses = [(s, s.pose) for s in self.tracker.slots if s.state != EMPTY and s.pose.spread is not None]
        if not poses:
            return
        spread = min(p.spread for _, p in poses)                        # BOTH hands must open for the portal to open
        state.openness = smoothstep(g.spread_closed, g.spread_open, spread)
        state.twists = tuple(float(p.twist) for _, p in poses) + (poses[0][1].twist,) * (2 - len(poses))
        state.twist = float(np.mean([p.twist for _, p in poses]))
        v = np.mean([s.kf.vel for s, _ in poses], axis=0)
        sp = float(np.hypot(*v))
        state.speed = float(min(1.0, sp / max(g.speed_ref * self.W, 1e-6)))
        state.motion = (float(v[0] / sp), float(v[1] / sp)) if sp > 1e-3 else (0.0, 0.0)

    def _quad_state(self, t, tips, vels, assign, flags):
        cfg = self.cfg
        dbg = {"rejects": dict(self.rejects), "assigned": sorted(assign), "gt": {k: d.hand.gt_id for k, d in assign.items()}, "constraint": flags,
               "events": self.tracker.events[-4:]}
        centres = tuple(None if s.state == EMPTY else s.kf.pos.copy() for s in self.tracker.slots)
        if 0 in tips and 1 in tips:
            a, b = tips[0], tips[1]
            corners = np.array([a[0:2], a[2:4], b[2:4], b[0:2]])
            va, vb = vels[0], vels[1]
            vel = np.array([va[0:2], va[2:4], vb[2:4], vb[0:2]])
            order, ok, why = quadmod.build(corners, self.W, self.H, cfg.quad)
            slots = self.tracker.slots
            conf = float(np.mean([slots[0].score, slots[1].score])) * (0.6 ** max(slots[0].missed, slots[1].missed))
            if ok:
                self.last_valid, self.invalid_run = (corners, vel, order), 0
                return PortalState(t, True, corners, vel, order, conf, 2, centres, dbg)
            self.invalid_run += 1
            dbg["quad_reject"] = why
            if self.last_valid is not None and self.invalid_run <= cfg.quad.hold_frames:
                dbg["held"] = True
                c, v, o = self.last_valid                     # hold the last valid quad briefly
                return PortalState(t, True, c, np.zeros_like(v), o, conf * 0.5, 2, centres, dbg)
            return PortalState(t, False, corners, vel, order, 0.0, 2, centres, dbg)
        n = sum(1 for s in self.tracker.slots if s.state != EMPTY)
        z = np.zeros((4, 2))
        c, v, o = self.last_valid if self.last_valid is not None else (z, z, (0, 1, 2, 3))
        return PortalState(t, False, c, np.zeros_like(v), o, 0.0, n, centres, dbg)
