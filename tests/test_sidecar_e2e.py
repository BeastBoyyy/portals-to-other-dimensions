"""End to end on this machine, no TouchDesigner: a fake 'TD' writes frames into shared memory (same layout as
td/scripts/frame_writer.py), the REAL sidecar runs MediaPipe LIVE_STREAM + HandCore, and we receive its UDP state packets."""
import pathlib
import socket
import subprocess
import sys
import time

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
pytest.importorskip("mediapipe")
cv2 = pytest.importorskip("cv2")
if not (ROOT / "models" / "hand_landmarker.task").exists():
    pytest.skip("models missing", allow_module_level=True)

from adapters import protocol as P   # noqa: E402


def test_shm_to_sidecar_to_udp_roundtrip():
    from multiprocessing import shared_memory
    src = cv2.cvtColor(cv2.imread(str(ROOT / "fixtures" / "images" / "woman_hands.jpg")), cv2.COLOR_BGR2RGB)
    w, h = 640, 360
    k = h / src.shape[0]                                                 # letterbox: keep the aspect ratio (squashing fakes bad hands)
    small = cv2.resize(src, (int(src.shape[1] * k), h))
    img = np.zeros((h, w, 3), np.uint8)
    x0 = (w - small.shape[1]) // 2
    img[:, x0:x0 + small.shape[1]] = small
    try:
        old = shared_memory.SharedMemory(name=P.FRAME_SHM_NAME)      # a leftover from a crashed run
        old.close(); old.unlink()
    except FileNotFoundError:
        pass
    shm = shared_memory.SharedMemory(name=P.FRAME_SHM_NAME, create=True, size=P.FRAME_HDR.size + P.FRAME_MAX_BYTES)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]; sock.settimeout(0.05)
    proc = subprocess.Popen([sys.executable, "-I", str(ROOT / "adapters" / "mediapipe_sidecar.py"), "--port", str(port), "--no-mask"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=str(ROOT))
    pkts, seq = [], 0
    try:
        t0 = time.perf_counter()
        nxt = t0
        while time.perf_counter() - t0 < 9.0:
            now = time.perf_counter()
            if now >= nxt:
                nxt += 1 / 30
                seq += 1
                M = np.float32([[1, 0, 6 * np.sin(seq / 12)], [0, 1, 3 * np.cos(seq / 9)]])
                frame = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
                np.frombuffer(shm.buf, np.uint8, w * h * 3, P.FRAME_HDR.size)[:] = frame.ravel()
                P.FRAME_HDR.pack_into(shm.buf, 0, P.FRAME_MAGIC, seq, now - t0, w, h, 0)
            try:
                d = P.unpack_state(sock.recv(2048))
                if d:
                    pkts.append(d)
            except socket.timeout:
                pass
    finally:
        proc.kill(); proc.wait(timeout=10)
        shm.close(); shm.unlink(); sock.close()
    assert len(pkts) > 100, f"only {len(pkts)} packets"                 # ~30 Hz for ~8 s after model start-up
    seqs = [p["seq"] for p in pkts]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    ts = [p["t"] for p in pkts]
    assert all(b > a for a, b in zip(ts, ts[1:])) and ts[-1] < 9.5        # TD-clock capture stamps pass through untouched
    assert max(p["n_hands"] for p in pkts) >= 1                           # real hands were seen by real MediaPipe
    assert np.mean([p["infer_ms"] for p in pkts[20:]]) > 0
    assert all(np.isfinite(p["corners"]).all() for p in pkts)
