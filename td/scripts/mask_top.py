# Script TOP callbacks for 'mask_in'  (needs user verification in TD)
# Uploads the selfie mask (written by the sidecar into shared memory 'portal_mask') whenever it changes.
# Falls back to an all-white mask (particles everywhere) until the sidecar publishes one.
import numpy as np
from multiprocessing import shared_memory

P = op('protocol').module
_shm = None
_seq = -1


def onSetupParameters(scriptOp):
    return


def onCook(scriptOp):
    global _shm, _seq
    _ = absTime.frame                      # make the Script TOP time-dependent so it polls every frame
    if _shm is None:
        try:
            _shm = shared_memory.SharedMemory(name=P.MASK_SHM_NAME)
        except FileNotFoundError:
            if _seq == -1:
                scriptOp.copyNumpyArray(np.ones((9, 16, 4), np.float32))
                _seq = -2
            return
    magic, seq, w, h = P.MASK_HDR.unpack_from(_shm.buf, 0)
    if magic != P.MASK_MAGIC or seq == _seq:
        return
    _seq = seq
    m = np.frombuffer(_shm.buf, np.uint8, w * h, P.MASK_HDR.size).reshape(h, w).astype(np.float32) / 255.0
    rgba = np.empty((h, w, 4), np.float32)
    rgba[..., 0] = rgba[..., 1] = rgba[..., 2] = rgba[..., 3] = m[::-1]      # TD is bottom-up
    scriptOp.copyNumpyArray(rgba)
    return
