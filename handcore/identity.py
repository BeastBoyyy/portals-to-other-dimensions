"""Persistent two-slot identity tracking (spec 4.4) with temporal acquisition / loss (gate 3)
and motion gating (gate 4). Handedness labels are only a weak vote; slots are our own."""
from __future__ import annotations
import math
from dataclasses import dataclass
from itertools import permutations
import numpy as np
from .kalman import CVKalman
from .one_euro import OneEuro
from .pose import HandPose

EMPTY, TRACKING, COASTING = "empty", "tracking", "coasting"
INF = 1e9


@dataclass
class Det:
    hand: object
    px: np.ndarray          # (21,3) pixel landmarks
    centre: np.ndarray      # palm centre px
    palm: float             # 2D palm size px
    label_sign: float       # +1 'Left', -1 'Right', 0 unknown (after mirror handling)
    mask_ok: bool = True    # False: not on the person mask -> may continue an existing track but cannot start one


def _shrink2(v, d):
    """Soft-threshold each 2D vector in v (shape (2k,)) by d px/s."""
    o = v.reshape(-1, 2)
    m = np.maximum(np.hypot(o[:, 0], o[:, 1]), 1e-9)
    return (o * (np.maximum(m - d, 0.0) / m)[:, None]).reshape(-1)


class Slot:
    def __init__(self, idx, cfg):
        self.idx, self.cfg = idx, cfg
        self.filt = None
        self.reset()

    def reset(self):
        self.state = EMPTY
        self.kf = CVKalman(self.cfg.kalman.accel_sigma, self.cfg.kalman.meas_sigma, self.cfg.kalman.coast_decay_s)
        f = self.cfg.filter
        self.filt = OneEuro(4, f.min_cutoff, f.beta, f.d_cutoff, f.speed_floor)
        self.cfilt = OneEuro(2, f.palm_min_cutoff, f.palm_beta, f.palm_d_cutoff, f.palm_speed_floor)
        self.t_meas = None          # time of last measurement
        self.t_kf = None            # time the KF was last advanced
        self.missed = 0
        self.centre_meas = None
        self.kf_at_meas = None
        self.palm = None
        self.label_vote = 0.0
        self.tips = None            # (4,) filtered [thumb x,y, index x,y]
        self.vel = np.zeros(4)
        self.rel_good = None
        self.rel_raw = None
        self.rel_vel = None
        self.t_gate = None
        self.alt = [None, None]
        self.strikes = [0, 0]
        self.hist = []
        self.score = 0.0
        self.swap_pending = 0
        self.t_prev_filter = None
        self.cf_prev = self.rf_prev = None
        self.vc = np.zeros(2)
        self.vr = np.zeros(4)
        g = self.cfg.gesture
        self.pose = HandPose(g.tau_open_s, g.tau_twist_s)
        self.lid = None
        self.lid_filt = OneEuro(2, f.min_cutoff, f.beta, f.d_cutoff, f.speed_floor)

    # -- lifecycle ----------------------------------------------------------------------
    def start(self, t, det):
        self.reset()
        self.state = TRACKING
        self.kf.init(det.centre)
        self.t_meas = self.t_kf = t
        self.centre_meas = det.centre.copy()
        self.kf_at_meas = det.centre.copy()
        self.palm = det.palm
        self.label_vote = det.label_sign

    def rearm(self, t, det):
        """A hand comes back after a long coast (or far from where we predicted it): every filter / gate remembers a different hand pose -> start clean."""
        keep = (self.state, self.pose, self.label_vote, self.palm)
        self.reset()
        self.state, self.pose, self.label_vote, self.palm = keep
        self.kf.init(det.centre)
        self.t_meas = self.t_kf = t
        self.centre_meas = self.kf_at_meas = det.centre.copy()

    def predict(self, t):
        dt = t - self.t_kf
        self.kf.predict(dt, coasting=self.missed > 0)
        self.t_kf = t

    def measure(self, t, det, tips_raw):
        self.kf.update(det.centre)
        self.state, self.missed = TRACKING, 0
        self.t_meas = t
        self.centre_meas = det.centre.copy()
        self.kf_at_meas = self.kf.pos.copy()
        self.palm = 0.8 * self.palm + 0.2 * det.palm
        self.label_vote = 0.9 * self.label_vote + 0.1 * det.label_sign
        self.score = det.hand.score
        f = self.cfg.filter
        prev = self.tips
        if f.kind == "none":
            self.tips = tips_raw.copy()
        elif f.kind == "moving_average":
            self.hist.append(tips_raw.copy())
            self.hist = self.hist[-f.ma_window:]
            self.tips = np.mean(self.hist, axis=0)
        elif f.decompose:
            c = np.tile(det.centre, 2)
            cf, rf = self.cfilt(det.centre, t), self.filt(tips_raw - c, t)
            self.tips = rf + np.tile(cf, 2)
            if self.t_prev_filter is not None and self.cf_prev is not None:
                dt = max(t - self.t_prev_filter, 1e-3)
                a = 1.0 / (1.0 + (1.0 / (2.0 * math.pi * f.vel_cutoff)) / dt)
                self.vc += a * ((cf - self.cf_prev) / dt - self.vc)
                self.vr += a * ((rf - self.rf_prev) / dt - self.vr)
                self.vel = np.tile(self.vc, 2) + _shrink2(self.vr, f.vel_rel_deadband)   # clean palm + dead-zoned finger motion
            self.cf_prev, self.rf_prev = cf, rf
            self.t_prev_filter = t
            return self.tips
        else:
            self.tips = self.filt(tips_raw, t)
        if prev is not None and self.t_prev_filter is not None:
            dt = max(t - self.t_prev_filter, 1e-3)
            a = 1.0 / (1.0 + (1.0 / (2.0 * math.pi * f.vel_cutoff)) / dt)
            self.vel = self.vel + a * ((self.tips - prev) / dt - self.vel)
        self.t_prev_filter = t
        return self.tips

    def measure_lid(self, t, det, lid_raw):
        """Glass-lid far-edge point: filtered like the tips (offset from the palm, plus the filtered palm centre)."""
        base = self.cf_prev if self.cf_prev is not None else det.centre
        self.lid = self.lid_filt(lid_raw - det.centre, t) + base
        return self.lid

    def coast_lid(self):
        shift = self.kf.pos - self.kf_at_meas
        return (self.lid if self.lid is not None else self.tips[2:4]) + shift

    def coast(self):
        self.missed += 1
        self.state = COASTING
        shift = self.kf.pos - self.kf_at_meas
        v = np.tile(self.kf.vel, 2)
        return self.tips + np.tile(shift, 2), v


class Tracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.slots = [Slot(0, cfg), Slot(1, cfg)]
        self.cands = []              # [centre, palm, count, t]
        self.events = []

    # ------------------------------------------------------------------------------------
    def _gate_ok(self, det, slot, t):
        g = self.cfg.gates
        if not g.motion:
            return True
        dt = max(t - slot.t_meas, 1e-3)
        if np.hypot(*(det.centre - slot.centre_meas)) > g.vmax_px_s * dt + 0.5 * slot.palm:
            return False
        # deviation from the constant-velocity prediction is bounded by hand acceleration (+ a palm-relative slack)
        if np.hypot(*(det.centre - slot.kf.pos)) > g.jump_palm_frac * slot.palm + 0.5 * self.cfg.kalman.accel_sigma * dt * dt:
            return False
        return slot.kf.maha2(det.centre) <= g.maha_thresh

    def _cost(self, det, slot):
        i = self.cfg.identity
        c = slot.kf.maha2(det.centre) / self.cfg.gates.maha_thresh
        c += i.w_appearance * abs(math.log(max(det.palm, 1e-6) / max(slot.palm, 1e-6)))
        if det.label_sign * slot.label_vote < 0:
            c += i.w_label * min(1.0, abs(slot.label_vote))
        return c

    def update(self, t, dets):
        """dets: validated Det list (<=2). Returns {slot_idx: Det} assigned this frame."""
        cfg, g = self.cfg, self.cfg.gates
        live = [s for s in self.slots if s.state != EMPTY]
        for s in live:
            s.predict(t)
        assign = {}
        self.ambiguous = False
        if not cfg.identity.enable:
            order = sorted(range(len(dets)), key=lambda j: dets[j].centre[0])
            for k, j in enumerate(order[:2]):
                assign[k] = dets[j]
        else:
            assign = self._assign(t, dets, live)
        used = {id(d) for d in assign.values()} | ({id(d) for d in dets} if self.ambiguous else set())
        # naive mode may assign into EMPTY slots directly (no temporal gate): start them here
        for k, d in list(assign.items()):
            if self.slots[k].state == EMPTY:
                self.slots[k].start(t, d)
        rest = [d for d in dets if id(d) not in used]
        self._candidates(t, rest, assign)
        # slots that got nothing this frame
        for s in live:
            if s.idx not in assign:
                if s.missed + 1 > g.lose_m:
                    s.reset()
                    self.events.append(("lost", s.idx, t))
        return assign

    # ------------------------------------------------------------------------------------
    def _assign(self, t, dets, live):
        cfg = self.cfg
        if not live or not dets:
            return {}
        n = len(dets)
        best, best_key = {}, None
        opts = []
        # every injective map from live slots to dets-or-None
        slots = live
        choices = list(range(n)) + [None] * len(slots)
        seen = set()
        for perm in permutations(choices, len(slots)):
            if perm in seen:
                continue
            seen.add(perm)
            if len({p for p in perm if p is not None}) != len([p for p in perm if p is not None]):
                continue
            cost, matched, ok = 0.0, 0, True
            for s, p in zip(slots, perm):
                if p is None:
                    continue
                if not self._gate_ok(dets[p], s, t):
                    ok = False
                    break
                cost += self._cost(dets[p], s)
                matched += 1
            if ok:
                opts.append((-matched, cost, perm))
        if not opts:
            return {}
        opts.sort(key=lambda o: (o[0], o[1]))
        best = opts[0]
        # fewer detections than live slots and the detection fits several slots about equally well: ambiguous
        # (crossing hands, one detection) -> assign nobody; both slots coast rather than risk a swap.
        if best[0] > -len(slots) and len(opts) > 1:
            alt = next((o for o in opts[1:] if o[0] == best[0] and o[2] != best[2]), None)
            if alt is not None and alt[1] - best[1] < cfg.identity.swap_margin:
                self.ambiguous = True
                return {}
        # swap hysteresis (only meaningful with two tracked slots and two detections)
        if len(slots) == 2 and n == 2 and best[0] == -2 and len(opts) >= 2 and all(s.state == TRACKING for s in slots):
            ref = min((o for o in opts if o[0] == -2), key=lambda o: sum(
                np.hypot(*(dets[p].centre - s.centre_meas)) for s, p in zip(slots, o[2])))
            if ref[2] != best[2]:
                margin = ref[1] - best[1]
                pend = slots[0].swap_pending + 1
                if margin >= cfg.identity.swap_margin or pend >= cfg.identity.swap_confirm:
                    slots[0].swap_pending = 0
                else:
                    slots[0].swap_pending = pend
                    best = ref
            else:
                slots[0].swap_pending = 0
        return {s.idx: dets[p] for s, p in zip(slots, best[2]) if p is not None}

    def _candidates(self, t, rest, assign):
        g = self.cfg.gates
        need = g.acquire_n if g.temporal else 1
        old, self.cands = self.cands, []
        for d in rest:
            if not d.mask_ok and not any(s.state != EMPTY for s in self.slots):
                continue                      # poster / background hand cannot start the FIRST track; a second hand may join a live one
            cnt = 1
            for c in old:
                dt = max(t - c[3], 1e-3)
                if np.hypot(*(d.centre - c[0])) <= (g.vmax_px_s * dt if g.motion else 1e9) + 0.5 * c[1] \
                        and (not g.motion or np.hypot(*(d.centre - c[0])) <= 0.35 * c[1] + 1500 * dt + 30):
                    cnt = c[2] + 1
                    break
            self.cands.append([d.centre.copy(), d.palm, cnt, t, d])
        ready = sorted([c for c in self.cands if c[2] >= need], key=lambda c: c[0][0])
        free = [s for s in self.slots if s.state == EMPTY and s.idx not in assign]
        free += sorted([s for s in self.slots if s.state == COASTING and s.idx not in assign], key=lambda s: -s.missed)
        for c in ready:
            if not free:
                break
            idx = 0
            if len(free) == 2 and all(x.state == EMPTY for x in free):    # nothing to anchor on: left half -> slot 0
                idx = 0 if c[0][0] < 0.5 * self.cfg.camera.width else 1
                if len(ready) > 1:
                    idx = 0                                              # two at once: x-sorted order already
            s = free.pop(min(idx, len(free) - 1))
            s.start(t, c[4])
            assign[s.idx] = c[4]
            self.cands = [x for x in self.cands if x is not c]
            self.events.append(("acquired", s.idx, t))
