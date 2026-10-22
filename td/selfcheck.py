# In-app self-check (needs user verification in TD). Stored as Text DAT 'selfcheck' by build_network.py.
#
# In the Textport:       op('/project1/portal/selfcheck').module.start()          # ~45 s, guided
#                        op('/project1/portal/selfcheck').module.start(soak_s=60)  # full 60 s soak
# Then follow the prompts it prints. It never blocks TouchDesigner: a frame callback advances a small state machine.
#
# The analysis functions at the top are plain Python/numpy (no `op`, no `absTime`) and are unit-tested outside TD.
import math

# ---- pure analysis (tested in tests/test_selfcheck.py) -------------------------------------------------------------


def hp_rms(samples, fs, cutoff_hz=3.0):
    """RMS of the part of `samples` (list of rows, one value per channel) above cutoff_hz."""
    import numpy as np
    x = np.asarray(samples, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if len(x) < 32:
        return float('nan')
    x = x - x.mean(0)
    w = np.hanning(len(x))[:, None]
    X = np.fft.rfft(x * w, axis=0)
    X[np.fft.rfftfreq(len(x), 1.0 / fs) < cutoff_hz] = 0
    hp = np.fft.irfft(X, n=len(x), axis=0) / w.mean()
    return float(np.sqrt(np.mean(hp ** 2)))


def frame_stats(dts):
    """dts: seconds per frame. -> dict(avg_fps, p1_fps, late_pct, n_late, n)."""
    import numpy as np
    d = np.asarray(dts, dtype=float)
    if len(d) < 10:
        return {'avg_fps': float('nan'), 'p1_fps': float('nan'), 'late_pct': float('nan'), 'n_late': 0, 'n': len(d)}
    return {'avg_fps': float(1.0 / d.mean()), 'p1_fps': float(1.0 / np.percentile(d, 99)),
            'late_pct': float(100.0 * np.mean(d > 0.020)), 'n_late': int(np.sum(d > 0.020)), 'n': len(d)}


def max_kink(series):
    """Largest |second difference| (px) over a list of 8-vectors sampled at the render rate: a 1-frame jump."""
    import numpy as np
    x = np.asarray(series, dtype=float)
    if len(x) < 5:
        return float('nan')
    return float(np.abs(np.diff(x, n=2, axis=0)).max())


# ---- state machine inside TD ------------------------------------------------------------------------------------------
_S = {}
CH = ['c%d' % i for i in range(8)]


def _chan(name):
    return op('portal_null')[name].eval()


def start(soak_s=10.0):
    _S.clear()
    _S.update(phase=0, t0=None, soak_s=float(soak_s), data={}, results=[])
    ex = op('selfcheck_exec') or op('/project1/portal').create(executeDAT, 'selfcheck_exec')
    ex.text = "def onFrameStart(frame):\n    op('selfcheck').module.tick()\n    return\n"
    ex.par.framestart = True
    ex.par.active = True
    print('\n=== PORTAL SELF-CHECK ===  (keep both hands in the frame gesture where the camera sees them)')


PHASES = [
    ('fps', 8.0, 'Phase 1/5: camera + tracker rate. Hold both hands in the frame pose and move slowly.'),
    ('still', 5.0, 'Phase 2/5: HOLD BOTH HANDS COMPLETELY STILL in the frame pose for 5 seconds.'),
    ('cross', 10.0, 'Phase 3/5: slowly CROSS your hands past each other and back, twice.'),
    ('latency', 5.0, 'Phase 4/5: wave the frame left and right at a normal pace.'),
    ('soak', None, 'Phase 5/5: frame-time soak. Keep the frame in view; do nothing special.'),
]


def _end():
    ex = op('selfcheck_exec')
    if ex:
        ex.par.active = False
    _verdict()


def tick():
    if not _S:
        return
    now = absTime.seconds
    ph = _S['phase']
    if ph >= len(PHASES):
        _end()
        _S.clear()
        return
    name, dur, msg = PHASES[ph]
    dur = _S['soak_s'] if dur is None else dur
    if _S['t0'] is None:
        _S['t0'] = now
        _S['data'] = {'dt': [], 'c': [], 'age': [], 'trk': [], 'cam': [], 'n': [], 'alpha': [], 'infer': [], 'cpu': []}
        print('\n' + msg)
    d = _S['data']
    d['dt'].append(_chan('render_dt_ms') / 1000.0)
    d['c'].append([_chan(k) for k in CH])
    d['age'].append(_chan('age_ms'))
    d['trk'].append(_chan('tracker_fps'))
    d['cam'].append(op('portal_state').module.camera_fps())
    d['n'].append(_chan('n_hands'))
    d['alpha'].append(_chan('alpha'))
    d['infer'].append(_chan('infer_ms'))
    d['cpu'].append(_chan('cpu_us'))
    if now - _S['t0'] >= dur:
        _S['results'].append((name, _analyse(name, d, dur)))
        _S['phase'] += 1
        _S['t0'] = None


def _analyse(name, d, dur):
    import numpy as np
    if name == 'fps':
        cam, trk = float(np.median(d['cam'][len(d['cam']) // 2:])), float(np.median(d['trk'][len(d['trk']) // 2:]))
        return [('camera fps (new frames/s)', cam, cam >= 25, '>= 25 (30 expected)'),
                ('tracker fps', trk, trk >= 0.9 * max(cam, 1.0), '>= 90% of camera fps'),
                ('sidecar inference ms', float(np.median(d['infer'])), float(np.median(d['infer'])) <= 25, '<= 25 ms'),
                ('handcore CPU us/frame', float(np.median(d['cpu'])), float(np.median(d['cpu'])) <= 4000, '<= 4000 us')]
    if name == 'still':
        ok = [i for i, (a, n) in enumerate(zip(d['alpha'], d['n'])) if a > 0.9 and n == 2]
        stab = 100.0 * len(ok) / max(len(d['alpha']), 1)
        fs = 1.0 / max(float(np.mean(d['dt'])), 1e-6)
        j = hp_rms([d['c'][i] for i in ok], fs) if len(ok) > 64 else float('nan')
        return [('portal visible with 2 hands (% of frames)', stab, stab >= 99.0, '>= 99 %'),
                ('rest jitter >3 Hz RMS (px)', j, j == j and j <= 0.4, '<= 0.4 px')]
    if name == 'cross':
        k = max_kink(d['c']) if len(d['c']) > 5 else float('nan')
        return [('largest 1-frame corner kink while crossing (px)', k, k == k and k <= 60.0,
                 '<= 60 px (a left/right slot swap shows up as a >100 px kink)')]
    if name == 'latency':
        a = np.asarray(d['age'])
        return [('tracker->render age, mean (ms)', float(a.mean()), a.mean() <= 80, '<= 80 ms'),
                ('tracker->render age, p95 (ms)', float(np.percentile(a, 95)), np.percentile(a, 95) <= 110, '<= 110 ms')]
    if name == 'soak':
        s = frame_stats(d['dt'])
        return [('render fps average', s['avg_fps'], s['avg_fps'] >= 59, '>= 59'),
                ('render fps, 1st percentile', s['p1_fps'], s['p1_fps'] >= 55, '>= 55'),
                ('frames over 20 ms (%)', s['late_pct'], s['late_pct'] <= 0.1, '<= 0.1 %  (%d of %d)' % (s['n_late'], s['n']))]
    return []


def _verdict():
    print('\n=== RESULTS ===')
    fails = []
    for name, rows in _S['results']:
        for label, val, ok, thr in rows:
            print('  %-4s %-48s %8.2f   (need %s)' % ('PASS' if ok else 'FAIL', label, val, thr))
            if not ok:
                fails.append((name, label))
    print()
    if not fails:
        print('VERDICT: tracking and rendering look healthy. The portal should feel attached to your hands.')
        return
    hints = {'fps': 'Camera/tracker rate low: lower camera load (close other apps), request 720p30 in the sidecar, check the sidecar console for "GPU delegate unavailable".',
             'still': 'Jitter or lost hands: improve lighting (face a window), lock camera exposure/white balance, shorten shutter; see README "Camera setup".',
             'cross': 'Corner kinks while crossing: keep hands apart in depth, or raise identity.swap_margin in config.yaml; run tools/evaluate.py on a recorded clip.',
             'latency': 'High latency: close heavy apps, set the camera to 30 fps MJPEG, make sure Video Device In is not buffering extra frames.',
             'soak': 'Frame drops: lower quality_cell_px (td/config.json: 4 -> 6), turn the HUD off (H), reduce resolution; check Performance Monitor (F2) for the slow operator.'}
    print('VERDICT: %d check(s) failed.' % len(fails))
    for n in sorted({f[0] for f in fails}):
        print('  next step [%s]: %s' % (n, hints[n]))
