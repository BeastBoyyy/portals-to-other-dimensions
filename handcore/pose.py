"""Whole-hand pose features that drive the portal (no finger-raise logic; the box is always attached to thumb / index / middle / ring tips):

  openness : fingertip spread / palm size.  Pinched claw ~0.2-0.4, open hand ~0.8-1.2 (measured on the reference clip).
             The portal OPENS as the hand opens and CLOSES when the hand pinches.
  twist    : angle between the palm normal and the camera axis, 0..90 deg (0 = palm or back faces the camera, 90 = edge-on).
             Twisting the hands twists the box AND turns the effect dial (see handcore/view.py).

Both are low-passed per hand; the portal uses the values of both hands combined.
"""
from __future__ import annotations
import math
import numpy as np

TIPS = (4, 8, 12, 16, 20)


def hand_openness(px, palm):
    """px (21,>=2) pixel landmarks, palm = 2D palm size px. Mean distance of the 5 fingertips from their centroid, in palm sizes."""
    tips = px[list(TIPS), :2]
    return float(np.mean(np.linalg.norm(tips - tips.mean(0), axis=1)) / max(palm, 1e-6))


def hand_twist_deg(world):
    """Angle of the palm normal to the camera axis from metric 3D landmarks (rotation about any axis counts)."""
    n = np.cross(world[5] - world[0], world[17] - world[0])
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        return 0.0
    return math.degrees(math.acos(min(1.0, abs(float(n[2])) / nn)))


def smoothstep(a, b, x):
    t = min(1.0, max(0.0, (x - a) / (b - a)))
    return t * t * (3.0 - 2.0 * t)


class HandPose:
    """Smoothed pose of one tracked hand (slot)."""

    def __init__(self, tau_open=0.06, tau_twist=0.15):
        self.tau_open, self.tau_twist = tau_open, tau_twist
        self.spread = None
        self.twist = None
        self.rate = 0.0                      # sustained |d openness / dt| (spread units per second): how fast the hand is pinching / opening
        self._raw_prev = None
        self._deltas = []
        self.t = None

    def update(self, t, px, world, palm):
        dt = 0.033 if self.t is None else min(max(t - self.t, 1e-3), 0.2)
        s = hand_openness(px, palm)
        tw = hand_twist_deg(world) if world is not None else (self.twist if self.twist is not None else 35.0)
        if self.t is None:
            self.spread, self.twist = s, tw
        else:
            self.spread += dt / (self.tau_open + dt) * (s - self.spread)
            # a real pinch changes the spread EVERY frame; a one-frame landmark snap changes it once and then returns: take the minimum of the last 3 steps
            self._deltas = (self._deltas + [abs(s - self._raw_prev) / dt])[-3:]
            self.rate = min(self._deltas) if len(self._deltas) == 3 else 0.0
            self.twist += dt / (self.tau_twist + dt) * (tw - self.twist)
        self._raw_prev = s
        self.t = t
