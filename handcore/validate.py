"""Per-detection gates (spec 4.3 gates 1, 2, 5). Temporal and motion gates live in identity.py."""
from __future__ import annotations
import numpy as np

CHAINS = {"thumb": (1, 2, 3, 4), "index": (5, 6, 7, 8), "middle": (9, 10, 11, 12),
          "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
PALM_POLY = (0, 5, 9, 13, 17)
ANCHORS = (0, 5, 9, 13, 17)


def _seg_cross(a, b, c, d):
    def ccw(p, q, r):
        return (r[1] - p[1]) * (q[0] - p[0]) - (q[1] - p[1]) * (r[0] - p[0])
    return ccw(a, c, d) * ccw(b, c, d) < 0 and ccw(a, b, c) * ccw(a, b, d) < 0


def palm_self_intersects(px2d):
    poly = [px2d[i] for i in PALM_POLY]
    n = len(poly)
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            if _seg_cross(poly[i], poly[(i + 1) % n], poly[j], poly[(j + 1) % n]):
                return True
    return False


def palm_centre(px):
    return px[list(ANCHORS), :2].mean(axis=0)


def palm_size_px(px):
    """2D palm size robust to edge-on views: max of palm width and wrist->middle-MCP length."""
    return max(np.linalg.norm(px[5, :2] - px[17, :2]), np.linalg.norm(px[0, :2] - px[9, :2]))


def to_px(lm, W, H):
    return lm * np.array([W, H, W])


def check_hand(hand, W, H, g, mask=None):
    """Returns (ok, reason). reason is '' when ok."""
    if not np.all(np.isfinite(hand.lm)) or hand.lm.shape != (21, 3):
        return False, "nonfinite"
    if hand.score < g.min_score:
        return False, "score"
    if hand.presence < g.min_presence:
        return False, "presence"
    px = to_px(hand.lm, W, H)
    m = g.inside_margin
    inside = np.mean((hand.lm[:, 0] > -m) & (hand.lm[:, 0] < 1 + m) & (hand.lm[:, 1] > -m) & (hand.lm[:, 1] < 1 + m))
    if inside < g.min_inside_frac:
        return False, "edge"
    for i in (4, 8):                      # portal tips must be inside the frame
        if not (-m < hand.lm[i, 0] < 1 + m and -m < hand.lm[i, 1] < 1 + m):
            return False, "tip_outside"
    if g.geometry:
        ps = palm_size_px(px)
        if not (g.palm_w_frac[0] <= ps / W <= g.palm_w_frac[1]):
            return False, "palm_size"
        bb = px[:, :2].max(0) - px[:, :2].min(0)
        if max(bb) / max(min(bb), 1e-6) > g.max_aspect:
            return False, "aspect"
        if g.self_intersect and palm_self_intersects(px[:, :2]):
            return False, "self_intersect"
        P = hand.world if hand.world is not None else px        # metric when available
        pw = np.linalg.norm(P[5] - P[17])
        if pw < 1e-9:
            return False, "degenerate"
        lo, hi = g.palm_len_ratio
        if not lo <= np.linalg.norm(P[0] - P[9]) / pw <= hi:
            return False, "palm_len"
        for name, ch in CHAINS.items():
            segs = [np.linalg.norm(P[ch[i + 1]] - P[ch[i]]) / pw for i in range(3)]
            total = sum(segs) + (np.linalg.norm(P[ch[0]] - P[0]) / pw if name == "thumb" else 0.0)
            lo, hi = g.thumb_len_ratio if name == "thumb" else g.finger_len_ratio
            if not lo <= total <= hi:
                return False, f"{name}_len"
            if min(segs) < g.segment_ratio[0] or max(segs) > g.segment_ratio[1]:
                return False, f"{name}_seg"
        # knuckle order: MCPs must advance monotonically along the 5->17 axis
        ax = P[17] - P[5]
        proj = [float((P[i] - P[5]) @ ax) for i in (5, 9, 13, 17)]
        if not all(proj[i] < proj[i + 1] for i in range(3)):
            return False, "knuckle_order"
    if mask is not None:
        h, w = mask.shape[:2]
        ix = np.clip((hand.lm[list(ANCHORS), 0] * w).astype(int), 0, w - 1)
        iy = np.clip((hand.lm[list(ANCHORS), 1] * h).astype(int), 0, h - 1)
        if np.mean(mask[iy, ix] > 0.5) < g.mask_min_frac:
            return False, "person_mask"
    return True, ""
