# Execute DAT 'frame_writer'  (needs user verification in TouchDesigner)
# Copies the camera frame (TOP 'cam_small', 640x360, RAW un-mirrored orientation) into shared memory for the
# sidecar at ~30 Hz, stamped with TD's own clock (absTime.seconds), so no clock sync is ever needed.
import numpy as np
from multiprocessing import shared_memory

P = op('protocol').module                 # adapters/protocol.py text, loaded into DAT 'protocol'
CFG = op('config_json').module.CFG        # td/config.json, loaded by build_network.py

_shm = None
_seq = 0
_sig = None


def _get(w, h):
    global _shm
    if _shm is None:
        size = P.FRAME_HDR.size + P.FRAME_MAX_BYTES
        try:
            _shm = shared_memory.SharedMemory(name=P.FRAME_SHM_NAME, create=True, size=size)
        except FileExistsError:
            _shm = shared_memory.SharedMemory(name=P.FRAME_SHM_NAME)
    return _shm


def onFrameEnd(frame):
    global _seq, _sig
    if frame % CFG['td']['frame_divider']:        # 60 fps project / 2 = 30 Hz camera-rate work
        return
    top = op('cam_small')
    arr = top.numpyArray(delayed=True)           # float32 (h,w,4) 0..1, bottom-up. delayed=True avoids a GPU stall
    if arr is None:
        return
    h, w = arr.shape[:2]
    if min(h, w) < P.MIN_SIDE or w * h * 3 > P.FRAME_MAX_BYTES:
        return                                   # camera not running yet (or wrong size): publish nothing
    shm = _get(w, h)
    rgb = (arr[::-1, :, :3] * 255.0).astype(np.uint8)       # flip to top-left origin
    np.frombuffer(shm.buf, np.uint8, w * h * 3, P.FRAME_HDR.size)[:] = rgb.ravel()
    _seq += 1
    t = absTime.seconds - CFG['td']['frame_delay_s']          # the delayed readback is ~1 frame old
    P.FRAME_HDR.pack_into(shm.buf, 0, P.FRAME_MAGIC, _seq, t, w, h, 0)   # header (seq) written LAST
    sig = int(rgb[::37, ::37].sum())              # did the camera deliver a NEW frame since last time?
    op('portal_state').module.note_frame_written(t, sig != _sig)
    _sig = sig
    return
