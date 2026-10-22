# Script CHOP callbacks for 'presenter'  (needs user verification in TD)
# The ONLY place portal geometry is computed per frame: one Script CHOP -> Null CHOP 'portal_null' is the single
# source of truth. Reads ~30 numbers, no numpy, no allocations beyond tiny lists. Cooks every frame because it
# reads absTime.
import math

S = op('portal_state').module
CFG = S.CFG
W, H = CFG['camera']['width'], CFG['camera']['height']

_mix = 0.0
_inset = 0.6
_last = None


def onSetupParameters(scriptOp):
    return


def onCook(scriptOp):
    global _mix, _inset, _last
    now = absTime.seconds
    dt = S.tick_clock(now)
    for s in S.drain():
        S.presenter.push(now, s['t'], s['valid'], s['corners'], s['vel'], s['held'])
    c, alpha = S.presenter.render(now)
    last = S.last or {}

    order = S.draw_order(c)                          # angular sort of the 4 (id-stable) corners
    uv = []
    for i in order:
        uv += [c[2 * i] / W, 1.0 - c[2 * i + 1] / H]    # TD textures are bottom-left origin
    cx = sum(uv[0::2]) / 4.0
    cy = sum(uv[1::2]) / 4.0
    area = 0.5 * abs(sum(uv[2 * i] * uv[2 * ((i + 1) % 4) + 1] - uv[2 * ((i + 1) % 4)] * uv[2 * i + 1] for i in range(4)))

    def d(i, j):
        return math.hypot(uv[2 * i] - uv[2 * j], uv[2 * i + 1] - uv[2 * j + 1])
    e = [d(0, 1), d(1, 2), d(2, 3), d(3, 0)]
    tilt_x = (e[1] - e[3]) / max(e[1] + e[3], 1e-6)       # side lengths differ -> yaw-like tilt
    tilt_y = (e[0] - e[2]) / max(e[0] + e[2], 1e-6)

    # purely visual smoothing (allowed): scene cross-fade 300 ms, tunnel inset follows quad size
    step = dt / CFG['td']['scene_fade_s']
    _mix += max(-step, min(step, S.scene_target - _mix))
    _inset += (min(0.9, max(0.62, 0.55 + 1.0 * area)) - _inset) * min(1.0, dt * 6.0)
    px = CFG['td']['parallax_pos'] * (cx - 0.5) + CFG['td']['parallax_tilt'] * tilt_x
    py = CFG['td']['parallax_pos'] * (cy - 0.5) + CFG['td']['parallax_tilt'] * tilt_y

    scriptOp.clear()
    scriptOp.numSamples = 1
    vals = {'alpha': alpha, 'conf': last.get('conf', 0.0), 'valid': 1.0 if last.get('valid') else 0.0,
            'n_hands': last.get('n_hands', 0), 'cx': cx, 'cy': cy, 'area': area, 'tilt_x': tilt_x, 'tilt_y': tilt_y,
            'par_x': px, 'par_y': py, 'inset': _inset, 'scene_mix': _mix, 'hud': 1.0 if S.hud_on else 0.0,
            'time': now, 'render_dt_ms': dt * 1000.0, 'tracker_fps': S.tracker_fps(),
            'infer_ms': last.get('infer_ms', 0.0), 'age_ms': (now - last['t']) * 1000.0 if last else 0.0,
            'dropped': last.get('dropped', 0), 'cpu_us': last.get('cpu_us', 0)}
    for k, v in vals.items():
        scriptOp.appendChan(k)[0] = v
    for j in range(4):
        scriptOp.appendChan('q%dx' % j)[0] = uv[2 * j]
        scriptOp.appendChan('q%dy' % j)[0] = uv[2 * j + 1]
    for i in range(8):                                   # id-order pixel corners (for selfcheck / debugging)
        scriptOp.appendChan('c%d' % i)[0] = c[i]
    return
