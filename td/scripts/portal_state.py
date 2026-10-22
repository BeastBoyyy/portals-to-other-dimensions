# Text DAT 'portal_state' (used as a module: op('portal_state').module)   -- needs user verification in TD
# Holds the latest sidecar packets and the Presenter. UDP callback calls on_packet(); the Script CHOP calls
# drain()/presenter once per frame. Nothing here parses DAT cells: packets are fixed-size binary structs.
import sys
import json
import socket
import threading
import time
import builtins

CFGM = op('config_json').module
CFG = CFGM.CFG
if CFG['td']['repo'] not in sys.path:
    sys.path.insert(0, CFG['td']['repo'])
from handcore.predict import Presenter, draw_order      # noqa: E402  (pure Python, no numpy)
from handcore.config import Predict                       # noqa: E402

P = op('protocol').module

_pc = Predict()
for k, v in CFG['predict'].items():
    setattr(_pc, k, v)
presenter = Presenter(_pc, 1.0 / CFG['td']['camera_fps'])

pending = []              # packets received since the last cook
last = None               # last accepted packet dict
n_recv = 0
n_bad = 0
frames_written = 0
recv_times = []
scene_target = 0.0        # 0 = particle self-view, 1 = asteroid
hud_on = True
render_dt = 1.0 / 60
render_dt_max = 0.0
_last_now = None
_late_frames = 0


def on_packet(b):
    global n_recv, n_bad
    s = P.unpack_state(bytes(b))
    if s is None:
        n_bad += 1
        return
    n_recv += 1
    s['t_recv'] = time.perf_counter()        # NOT absTime: this runs on the receiver thread
    pending.append(s)
    recv_times.append(s['t_recv'])
    if len(recv_times) > 90:
        del recv_times[0]


cam_new_times = []        # capture times of frames that were actually new camera frames


# ---- UDP receiver thread (replaces the UDP In DAT: plain socket, no operator settings to get wrong) -------------------
_stop = False
_sock = None


def _rx():
    global _sock
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0.2)
    s.bind(('127.0.0.1', CFG['td']['udp_port']))
    _sock = s
    while not _stop:
        try:
            data = s.recv(2048)
        except socket.timeout:
            continue
        except OSError:
            break
        on_packet(data)
    s.close()


def _restart_receiver():
    old = getattr(builtins, '_portal_rx', None)          # a previous incarnation of this module (DAT edited / rebuilt)
    if old is not None:
        old[0]()                                          # ask it to stop
        old[1].join(1.0)
    th = threading.Thread(target=_rx, name='portal_udp_rx', daemon=True)

    def stop():
        global _stop
        _stop = True
    builtins._portal_rx = (stop, th)
    th.start()


_restart_receiver()


def note_frame_written(t, is_new):
    global frames_written
    frames_written += 1
    if is_new:
        cam_new_times.append(t)
        if len(cam_new_times) > 120:
            del cam_new_times[0]


def camera_fps():
    if len(cam_new_times) < 3 or cam_new_times[-1] <= cam_new_times[0]:
        return 0.0
    return (len(cam_new_times) - 1) / (cam_new_times[-1] - cam_new_times[0])


def drain():
    global last
    out = pending[:]
    del pending[:]
    for s in out:
        last = s
    return out


def tick_clock(now):
    """Frame-time statistics (dt from absTime). Returns dt."""
    global _last_now, render_dt, render_dt_max, _late_frames
    if _last_now is not None:
        dt = now - _last_now
        render_dt += 0.05 * (dt - render_dt)
        render_dt_max = max(render_dt_max * 0.995, dt)
        if dt > 0.020:
            _late_frames += 1
    _last_now = now
    return render_dt


def tracker_fps():
    if len(recv_times) < 3 or recv_times[-1] <= recv_times[0]:
        return 0.0
    return (len(recv_times) - 1) / (recv_times[-1] - recv_times[0])


def key(ch):
    global scene_target, hud_on
    if ch == '1':
        scene_target = 0.0
    elif ch == '2':
        scene_target = 1.0
    elif ch == 'h':
        hud_on = not hud_on


def hud_string():
    s = last or {}
    return ("render %5.1f fps  dt %4.1f ms (max %4.1f)  late>20ms: %d\n"
            "tracker %4.1f fps  infer %4.1f ms  core %d us  age %4.1f ms\n"
            "hands %d  conf %.2f  valid %s  held %s  dropped %d  bad pkts %d") % (
        1.0 / max(render_dt, 1e-6), render_dt * 1000, render_dt_max * 1000, _late_frames,
        tracker_fps(), s.get('infer_ms', 0), s.get('cpu_us', 0), (absTime.seconds - s['t']) * 1000 if s else 0,
        s.get('n_hands', 0), s.get('conf', 0), s.get('valid'), s.get('held'), s.get('dropped', 0), n_bad)
