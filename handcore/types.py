"""Plain data types. `Frame` is what every adapter produces; `PortalState` is what comes out."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Hand:
    lm: np.ndarray                       # (21,3) normalized image x,y (0..1), z (same scale as x)
    world: Optional[np.ndarray] = None   # (21,3) metres, MediaPipe hand_world_landmarks
    label: str = ""                      # "Left"/"Right" as reported (weak vote only)
    score: float = 1.0                   # handedness score
    presence: float = 1.0
    gt_id: int = -1                      # ground-truth hand id (synthetic / evaluation only)


@dataclass
class Frame:
    t: float                             # capture time, seconds (monotonic clock)
    hands: list = field(default_factory=list)
    person_mask: Optional[np.ndarray] = None   # low-res (h,w) selfie mask, 0..1


@dataclass
class PortalState:
    t: float
    valid: bool
    corners: np.ndarray                  # (4,2) px, FIXED id order [s0.thumb, s0.index, s1.index, s1.thumb]
    velocity: np.ndarray                 # (4,2) px/s
    order: tuple                         # draw order (indices into corners): convex winding
    confidence: float
    n_hands: int
    slot_centres: tuple = (None, None)   # per-slot palm centre px or None
    debug: dict = field(default_factory=dict)
    lid: Optional[np.ndarray] = None          # (2,2) px far-edge points of the glass lid (mean of middle + ring fingertips), per slot
    lid_vel: Optional[np.ndarray] = None
    openness: float = 1.0                     # 0 = both hands pinched (portal closed) .. 1 = both open
    twist: float = 35.0                       # mean palm twist of the tracked hands, deg (drives the effect dial)
    twists: tuple = (35.0, 35.0)              # per slot
    speed: float = 0.0                        # hand speed, 0..1 (1 = fast; reactivity of glitch / streaks)
    motion: tuple = (0.0, 0.0)                # unit direction of hand motion in px space
