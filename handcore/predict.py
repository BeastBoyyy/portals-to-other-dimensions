"""Render-rate presenter (spec 4.7 / 4.8): turns ~30 Hz samples into a continuous 60 Hz portal.

Pure Python floats (no numpy, no allocation beyond tiny lists) so it can run in a TouchDesigner
Script CHOP every frame. Call `push()` when a camera-rate sample arrives and `render()` once per
render tick with the render clock; both take the same clock as the sample timestamps.
"""
from __future__ import annotations
import math


def draw_order(c):
    """c: flat list of 8 floats [x0,y0,...x3,y3] -> indices sorted by angle around centroid."""
    cx = sum(c[0::2]) / 4.0
    cy = sum(c[1::2]) / 4.0
    return sorted(range(4), key=lambda i: math.atan2(c[2 * i + 1] - cy, c[2 * i] - cx))


def _shrink(x, y, d):
    """Soft-threshold a 2D vector's magnitude by d (0 inside the dead-zone, continuous outside)."""
    m = math.hypot(x, y)
    if m <= d:
        return 0.0, 0.0
    f = (m - d) / m
    return x * f, y * f


class Presenter:
    def __init__(self, p, camera_period=1 / 30.0, stall_s=0.5, n=8):
        self.n = n                # floats per sample (x,y pairs): 8 = four corners; the app adds 4 for the lid points
        self.p = p
        self.period = camera_period
        self.stall_s = stall_s
        self.reset()

    def reset(self):
        self.ts = None            # capture time of the model sample
        self.ta = None            # render-clock time the sample arrived
        self.c = [0.0] * self.n
        self.v = [0.0] * self.n
        self.target_valid = False
        self.alpha = 0.0
        self.off = [0.0] * self.n      # critically-damped error-decay offset (position)
        self.offv = [0.0] * self.n     # ... and its velocity, so the output stays C1-continuous when a sample lands
        self.off_tau = self.p.blend_s
        self.vout = [0.0] * self.n
        self.a = [0.0] * self.n        # acceleration estimate from successive sample velocities
        self.out = [0.0] * self.n
        self.have_out = False
        self.t_render = None      # last render() time (alpha slew, output velocity)
        self.t_off = None         # time at which (off, offv) are valid
        self.t_last_push = None

    # -- model -----------------------------------------------------------------------------
    def _model(self, t):
        if self.ts is None or not self.p.enable:
            return list(self.c)
        # latency (arrival - capture) is compensated in full; only the time *since arrival* is clamped, so a
        # stalled stream cannot fly away
        lat = max(self.ta - self.ts, 0.0)
        te = lat + min(max(t - self.ta, 0.0), self.p.extrap_max_periods * self.period)
        tau = self.p.tau_damp_s
        k = tau * (1.0 - math.exp(-te / tau)) if tau > 1e-6 else te
        db, adb = self.p.vel_deadband, self.p.accel_deadband
        out = []
        for i in range(0, self.n, 2):
            vx, vy = _shrink(self.v[i], self.v[i + 1], db)           # dead-zone: sensor noise is not motion
            ax, ay = _shrink(self.a[i], self.a[i + 1], adb) if self.p.use_accel else (0.0, 0.0)
            k2 = 0.5 * te * te * math.exp(-te / tau) if tau > 1e-6 else 0.5 * te * te
            out += [self.c[i] + vx * k + ax * k2, self.c[i + 1] + vy * k + ay * k2]
        return out

    def _decayed(self, t):
        """(offset, offset velocity) advanced to time t (critically damped spring toward zero, w = 1/tau)."""
        if self.t_off is None:
            return self.off, self.offv
        dt = max(t - self.t_off, 0.0)
        w = 1.0 / self.off_tau
        e = math.exp(-w * dt)
        off = [(o + (v + w * o) * dt) * e for o, v in zip(self.off, self.offv)]
        offv = [(v - w * (v + w * o) * dt) * e for o, v in zip(self.off, self.offv)]
        return off, offv

    def _current(self, t):
        m = self._model(t)
        off, _ = self._decayed(t)
        return [a + o for a, o in zip(m, off)]

    # -- inputs ----------------------------------------------------------------------------
    def push(self, t_now, t_capture, valid, corners, vel, held=False):
        """A new camera-rate sample arrived at render-clock time t_now. held=True: core is holding its last valid
        quad (current pose unusable); keep extrapolating the last good sample instead of freezing on stale corners."""
        if self.t_last_push is not None and t_capture > self.t_last_push:
            dt = t_capture - self.t_last_push
            if 0.005 < dt < 0.2:
                self.period += 0.1 * (dt - self.period)
        if self.t_last_push is not None and t_capture <= self.t_last_push:
            return                                   # duplicate / out-of-order stamp: drop
        self.t_last_push = t_capture
        self.target_valid = bool(valid)
        if not valid:
            return                                   # keep the old model; alpha fades out in render()
        if held:
            self.was_held = True                     # model untouched: carries on along its damped trajectory
            return
        prev_visible = self.have_out and self.alpha > 0.02
        before = self._current(t_now) if self.have_out else None
        _, offv_now = self._decayed(t_now)
        vbefore = [m + o for m, o in zip(self._model_vel(t_now), offv_now)] if self.have_out else None
        newv = [float(x) for x in vel]
        if self.ts is not None and prev_visible and 0.005 < t_capture - self.ts < 0.2:
            dts = t_capture - self.ts
            self.a = [a + self.p.accel_gain * ((nv - ov) / dts - a) for a, nv, ov in zip(self.a, newv, self.v)]
        else:
            self.a = [0.0] * self.n
        self.ts, self.ta = t_capture, t_now
        self.c, self.v = [float(x) for x in corners], newv
        if before is not None and prev_visible:
            m = self._model(t_now)
            mv = self._model_vel(t_now)
            vis = min(1.0, self.alpha)         # an invisible portal needs no easing: it just appears at the right place
            self.off = [(b - a) * vis for b, a in zip(before, m)]
            self.offv = [(b - a) * vis for b, a in zip(vbefore, mv)]
            # slower ease when we were (nearly) gone: re-acquisition; else the 2-3 frame error decay
            self.off_tau = self.p.reacq_s if (self.alpha < 0.98 or self.was_held) else self.p.blend_s
        else:
            self.off, self.offv = [0.0] * self.n, [0.0] * self.n
        self.t_off = t_now
        self.was_held = False

    def _model_vel(self, t, eps=1e-3):
        a, b = self._model(t), self._model(t + eps)
        return [(y - x) / eps for x, y in zip(a, b)]

    def render(self, t_now):
        """Returns (corners[8], alpha) at render time t_now."""
        if self.ts is None:
            return list(self.out), 0.0
        dt = 0.0 if self.t_render is None else max(t_now - self.t_render, 0.0)
        stalled = (t_now - self.t_last_push) > self.stall_s if self.t_last_push is not None else True
        target = 1.0 if (self.target_valid and not stalled) else 0.0
        if target > self.alpha:
            self.alpha = min(target, self.alpha + dt / max(self.p.fade_in_s, 1e-3))
        else:
            self.alpha = max(target, self.alpha - dt / max(self.p.fade_out_s, 1e-3))
        new = self._current(t_now)
        if self.have_out and dt > 1e-6:
            self.vout = [(n - o) / dt for n, o in zip(new, self.out)]
        self.out = new
        self.have_out = True
        self.off, self.offv = self._decayed(t_now)    # advance the stored offset to now
        self.t_off = self.t_render = t_now
        return list(self.out), self.alpha
