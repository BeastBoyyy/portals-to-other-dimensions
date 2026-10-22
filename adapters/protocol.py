"""Wire formats shared by the sidecar and TouchDesigner (pure Python, struct only; no numpy needed to unpack).

Frame shared memory (TD -> sidecar), name `portal_frame`:
    header  <IIdHHI  : magic, seq, t_capture (TD absTime.seconds), width, height, flags      (24 bytes)
    pixels  w*h*3 uint8 RGB, top-left origin, RAW camera orientation (TD flips bottom-up before writing)

Mask shared memory (sidecar -> TD), name `portal_mask`:
    header  <IIHH    : magic, seq, width, height                                              (12 bytes)
    pixels  w*h uint8 (0..255), top-left origin, RAW camera orientation

State packet (sidecar -> TD) over UDP, little endian, one datagram per camera frame:
    see STATE_FMT. Capture time is TD's own clock (it was stamped in TD), so TD needs no clock sync.
"""
from __future__ import annotations
import struct

FRAME_MAGIC = 0x50465231      # 'PFR1'
MASK_MAGIC = 0x504D4B31       # 'PMK1'
STATE_MAGIC = 0x504F5254      # 'PORT'

FRAME_SHM_NAME = "portal_frame_v2"      # v2: fixed-size block (v1 was sized by the first frame and could be tiny)
MASK_SHM_NAME = "portal_mask_v2"
FRAME_MAX_BYTES = 1280 * 720 * 3
MIN_SIDE = 64                          # frames smaller than this are not camera images (TD cooks a 1x1 TOP before the camera starts)

FRAME_HDR = struct.Struct("<IIdHHI")
MASK_HDR = struct.Struct("<IIHH")

# magic, seq, t_capture, valid, held, n_hands, pad, confidence, corners[8], vel[8], lid[4], lid_vel[4],
# pose[7] = (openness, twist, twist_slot0, twist_slot1, speed, motion_x, motion_y), infer_ms, tracker_fps, dropped, cpu_us
STATE_FMT = "<IIdBBBBf8f8f4f4f7fffII"
STATE = struct.Struct(STATE_FMT)


def pack_state(seq, t_capture, valid, held, n_hands, confidence, corners, vel,
               infer_ms=0.0, tracker_fps=0.0, dropped=0, cpu_us=0, lid=(0,) * 4, lid_vel=(0,) * 4, pose=(1.0, 35.0, 35.0, 35.0, 0.0, 0.0, 0.0)):
    return STATE.pack(STATE_MAGIC, seq, t_capture, int(valid), int(held), int(n_hands), 0, confidence,
                      *[float(x) for x in corners], *[float(x) for x in vel],
                      *[float(x) for x in lid], *[float(x) for x in lid_vel], *[float(x) for x in pose],
                      infer_ms, tracker_fps, dropped, cpu_us)


def unpack_state(b):
    """-> dict, or None for a malformed / foreign datagram."""
    if len(b) != STATE.size:
        return None
    f = STATE.unpack(b)
    if f[0] != STATE_MAGIC:
        return None
    return {"seq": f[1], "t": f[2], "valid": bool(f[3]), "held": bool(f[4]), "n_hands": f[5], "conf": f[7],
            "corners": list(f[8:16]), "vel": list(f[16:24]), "lid": list(f[24:28]), "lid_vel": list(f[28:32]),
            "pose": list(f[32:39]), "infer_ms": f[39], "tracker_fps": f[40], "dropped": f[41], "cpu_us": f[42]}
