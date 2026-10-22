"""Config dataclasses mirroring config.yaml. `load()` needs PyYAML; defaults work without it."""
from __future__ import annotations
from dataclasses import dataclass, field, is_dataclass
import copy


@dataclass
class Detector:
    num_hands: int = 2
    detection_confidence: float = 0.5
    presence_confidence: float = 0.5
    tracking_confidence: float = 0.5


@dataclass
class Camera:
    width: int = 1280
    height: int = 720
    fps: int = 60
    mirror: bool = True


@dataclass
class Gates:
    enable: bool = True
    min_score: float = 0.35
    min_presence: float = 0.50
    geometry: bool = False         # bone-proportion / palm-size / aspect checks: OFF (rejected real twisted + pinched hands, measured on live frames)
    palm_w_frac: tuple = (0.015, 0.45)
    finger_len_ratio: tuple = (0.55, 2.3)
    thumb_len_ratio: tuple = (0.9, 3.4)
    segment_ratio: tuple = (0.12, 1.2)
    palm_len_ratio: tuple = (0.85, 2.2)
    max_aspect: float = 5.0
    min_inside_frac: float = 0.95
    inside_margin: float = 0.01
    temporal: bool = True
    acquire_n: int = 3
    lose_m: int = 15               # a hand missing for up to 0.5 s is bridged by prediction instead of losing the portal
    motion: bool = True
    vmax_px_s: float = 12000.0
    maha_thresh: float = 16.0
    jump_palm_frac: float = 0.25   # max deviation from the Kalman prediction, in palm sizes (+ accel allowance)
    self_intersect: bool = False   # palm-polygon self-intersection test: OFF (it rejected pinched / strongly twisted REAL hands)
    person_mask: bool = False      # OFF by default: the selfie segmenter often leaves hands held out at arm's length off the mask (live trace)
    mask_min_frac: float = 0.3
    mask_dilate: int = 10          # dilate the low-res person mask by this many cells before testing hands (arms held out)
    mask_acquire_only: bool = True # off-mask hands cannot start the FIRST track; once one hand is tracked the second joins without the mask test


@dataclass
class Kalman:
    accel_sigma: float = 15000.0
    meas_sigma: float = 4.0
    coast_decay_s: float = 0.15


@dataclass
class Identity:
    enable: bool = True
    w_label: float = 0.6
    w_appearance: float = 2.0
    swap_margin: float = 0.6
    swap_confirm: int = 4


@dataclass
class Constraint:
    enable: bool = False
    thumb_ratio: float = 0.45
    index_ratio: float = 0.42
    slack: float = 0.35
    calibrate_frames: int = 0
    anchor_spike: bool = True
    spike_palm_frac: float = 0.2
    spike_persist: int = 3
    spike_speed_gain: float = 1.0
    spike_vel_gain: float = 0.9


@dataclass
class Filter:
    kind: str = "one_euro"
    min_cutoff: float = 0.7
    beta: float = 0.3
    d_cutoff: float = 6.0
    ma_window: int = 3
    speed_floor: float = 80.0     # px/s dead-zone on the tip-offset filter's speed signal
    palm_speed_floor: float = 20.0
    decompose: bool = True        # filter palm centre and tip-offset-from-palm separately
    palm_min_cutoff: float = 1.0
    palm_beta: float = 0.5
    palm_d_cutoff: float = 15.0
    vel_rel_deadband: float = 150.0  # px/s soft threshold on the finger-relative velocity (noise, not motion)
    vel_cutoff: float = 8.0       # Hz, low-pass on the output velocity used for extrapolation


@dataclass
class Predict:
    enable: bool = True
    extrap_max_periods: float = 1.5
    tau_damp_s: float = 0.5
    use_accel: bool = True        # constant-acceleration term (damped) on top of velocity extrapolation
    accel_gain: float = 0.5
    accel_deadband: float = 1500.0  # px/s^2
    vel_deadband: float = 20.0    # px/s: extrapolation fades out below this speed (kills rest jitter)
    blend_s: float = 0.03
    reacq_s: float = 0.10
    fade_out_s: float = 0.20
    fade_in_s: float = 0.10


@dataclass
class Quad:
    min_area_frac: float = 0.0001
    min_edge_px: float = 3.0
    hold_frames: int = 12


@dataclass
class Gesture:
    enable: bool = True
    tau_open_s: float = 0.06           # openness smoothing
    tau_twist_s: float = 0.15          # twist smoothing (twist estimates are noisy: +-10 deg)
    spread_closed: float = 0.35        # fingertip spread / palm size at which the hand counts as pinched (portal closed) ...
    spread_open: float = 0.80          # ... and fully open (portal open); measured on the reference clip: pinch 0.2-0.4, open 0.8-1.2
    speed_ref: float = 0.6             # hand speed (frame widths / s) that counts as "fast" (full reactivity)
    # the twist RATCHET: twisting the hands ~dial_step_deg beyond where they rest steps to the next effect (twist back = previous).
    # It is RELATIVE to your own resting twist (which follows slow drift), so extreme, edge-on angles - where MediaPipe is unreliable - are never needed.
    dial_effects: tuple = ("glass", "moire", "slitscan", "pixelsort", "crt", "datamosh")
    dial_step_deg: float = 20.0
    dial_commit_s: float = 0.25        # the twist must be held past the step this long to commit (no accidental steps)
    dial_base_tau_s: float = 6.0       # the resting twist follows slow drift with this time constant
    dial_ease_s: float = 0.25          # cross-fade time between effects
    twist_depth: float = 0.25          # how much each hand's twist pushes its side of the box deeper (3D twist of the box)


@dataclass
class Config:
    detector: Detector = field(default_factory=Detector)
    camera: Camera = field(default_factory=Camera)
    gates: Gates = field(default_factory=Gates)
    kalman: Kalman = field(default_factory=Kalman)
    identity: Identity = field(default_factory=Identity)
    constraint: Constraint = field(default_factory=Constraint)
    filter: Filter = field(default_factory=Filter)
    predict: Predict = field(default_factory=Predict)
    quad: Quad = field(default_factory=Quad)
    gesture: Gesture = field(default_factory=Gesture)


def _apply(obj, d: dict):
    for k, v in d.items():
        if not hasattr(obj, k):
            raise KeyError(f"unknown config key {k!r} on {type(obj).__name__}")
        cur = getattr(obj, k)
        if is_dataclass(cur):
            _apply(cur, v)
        else:
            setattr(obj, k, tuple(v) if isinstance(cur, tuple) else v)


def load(path: str | None = None, overrides: dict | None = None) -> Config:
    cfg = Config()
    if path:
        import yaml
        with open(path) as f:
            _apply(cfg, yaml.safe_load(f) or {})
    if overrides:
        _apply(cfg, overrides)
    return cfg


def with_overrides(cfg: Config, overrides: dict) -> Config:
    c = copy.deepcopy(cfg)
    _apply(c, overrides)
    return c
