"""Quad builder: order around the centroid, validate convexity / area / edges. Never returns a bowtie."""
from __future__ import annotations
import math
import numpy as np


def order_around_centroid(pts):
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    return tuple(int(i) for i in np.argsort(ang))


def validate_quad(pts, order, W, H, q):
    """pts (4,2) in id order, order = draw order. Returns (ok, reason)."""
    p = pts[list(order)]
    if not np.all(np.isfinite(p)):
        return False, "nonfinite"
    edges = np.roll(p, -1, axis=0) - p
    if np.min(np.hypot(edges[:, 0], edges[:, 1])) < q.min_edge_px:
        return False, "edge_short"
    cr = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    if not (np.all(cr > 0) or np.all(cr < 0)):
        return False, "not_convex"
    area = 0.5 * abs(float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])))
    if area < q.min_area_frac * W * H:
        return False, "area"
    return True, ""


def build(pts, W, H, q):
    order = order_around_centroid(pts)
    ok, why = validate_quad(pts, order, W, H, q)
    return order, ok, why
