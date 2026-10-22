"""Option A glue: feed the in-TouchDesigner `mediapipe-touchdesigner` plugin's hand landmarks into handcore.

STATUS: UNVERIFIED against the plugin. The plugin's CHOP/DAT channel naming could not be confirmed from its public README
(it documents a Hand tracking tox that outputs landmark data but not the channel names) and TouchDesigner was not available.
The mapping below is therefore configurable (CHANNEL_PATTERN). The pure function `frame_from_channels` is unit-tested with
fabricated channel names; the Script CHOP text at the bottom is what you would paste into a Script CHOP that reads the plugin.

Why Option A was not chosen (DECISIONS.md D1): the plugin runs MediaPipe in a Chromium Web Render TOP and returns data over a
local websocket; its own docs note extra frame delay from the Web Render TOP and a 720p input cap, and it competes with the
render for CPU/GPU. It cannot be benchmarked outside TouchDesigner, so by the spec's rule ("first candidate that passes the
thresholds") it stays an *option*: the same HandCore runs behind it unchanged.
"""
from __future__ import annotations
import re
import numpy as np

# adjust to the plugin's real naming, e.g. "h1:thumb_tip:x"; groups: hand index, landmark name, axis
CHANNEL_PATTERN = re.compile(r"^h(?P<hand>\d+)[:_](?P<lm>[a-z_0-9]+)[:_](?P<axis>[xyz])$")

LANDMARK_NAMES = [
    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip", "index_finger_mcp", "index_finger_pip", "index_finger_dip",
    "index_finger_tip", "middle_finger_mcp", "middle_finger_pip", "middle_finger_dip", "middle_finger_tip", "ring_finger_mcp",
    "ring_finger_pip", "ring_finger_dip", "ring_finger_tip", "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"]
ALIASES = {"thumb_tip": 4, "index_tip": 8}


def frame_from_channels(values: dict, t: float, pattern=CHANNEL_PATTERN):
    """values: {channel_name: float}. Returns handcore Frame (hands with all 21 landmarks present; others dropped)."""
    from handcore.types import Frame, Hand
    hands = {}
    index = {n: i for i, n in enumerate(LANDMARK_NAMES)}
    index.update(ALIASES)
    for name, v in values.items():
        m = pattern.match(name)
        if not m or m["lm"] not in index:
            continue
        h = hands.setdefault(int(m["hand"]), np.full((21, 3), np.nan))
        h[index[m["lm"]], "xyz".index(m["axis"])] = v
    out = []
    for hid in sorted(hands):
        lm = hands[hid]
        if np.isfinite(lm).all():                      # partial skeletons are dropped, never guessed
            out.append(Hand(lm, None, "", 1.0, 1.0, hid))
    return Frame(t, out)


SCRIPT_CHOP_TEXT = '''
# Script CHOP 'plugin_bridge' (Option A). Reads the plugin's landmark CHOP 'hands' every frame, runs HandCore in-process
# (about 0.7 ms) and outputs the same channels as the sidecar path would, so the rest of the network is unchanged.
import sys
sys.path.insert(0, CFG['td']['repo'])
from adapters.td_plugin_bridge import frame_from_channels
from handcore import config as hcfg
from handcore.pipeline import HandCore
_core = HandCore(hcfg.load(CFG['td']['repo'] + '/config.yaml'))

def onCook(scriptOp):
    src = op('hands')                                   # the plugin's hand-landmark CHOP (name may differ)
    vals = {c.name: c[0] for c in src.chans()}
    st = _core.update(frame_from_channels(vals, absTime.seconds))
    if st is not None:
        S = op('portal_state').module
        S.pending.append({'t': st.t, 'valid': st.valid, 'held': st.debug.get('held', False), 'n_hands': st.n_hands,
                          'conf': st.confidence, 'corners': list(st.corners.reshape(8)), 'vel': list(st.velocity.reshape(8)),
                          'infer_ms': 0.0, 'cpu_us': 0, 'dropped': 0, 't_recv': absTime.seconds})
'''
