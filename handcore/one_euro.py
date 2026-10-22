"""One Euro filter (Casiez et al. 2012) over a vector of coordinates, in pixel units, real dt."""
import math
import numpy as np

DT_MIN, DT_MAX = 1e-3, 0.25


class OneEuro:
    def __init__(self, n, min_cutoff=1.0, beta=0.007, d_cutoff=1.0, speed_floor=0.0):
        self.n, self.min_cutoff, self.beta, self.d_cutoff = n, min_cutoff, beta, d_cutoff
        self.speed_floor = speed_floor      # px/s dead-zone: derivative noise below it does not open the cutoff
        self.reset()

    def reset(self):
        self.x = None
        self.dx = np.zeros(self.n)
        self.t = None

    def __call__(self, x, t, strength=None):
        """x: (n,) raw. strength: optional (n,) in (0,1]; <1 lowers cutoff (heavier smoothing)."""
        x = np.asarray(x, dtype=float)
        if self.x is None:
            self.x, self.t = x.copy(), t
            return self.x.copy()
        dt = t - self.t
        if dt <= 0.0:                       # duplicate / non-monotonic stamp: ignore the sample
            return self.x.copy()
        dt = min(max(dt, DT_MIN), DT_MAX)   # clamp across clock jumps / long gaps
        self.t = t
        dx = (x - self.x) / dt
        a_d = 1.0 / (1.0 + (1.0 / (2.0 * math.pi * self.d_cutoff)) / dt)
        self.dx = self.dx + a_d * (dx - self.dx)
        s = 1.0 if strength is None else strength
        cutoff = (self.min_cutoff + self.beta * np.maximum(np.abs(self.dx) - self.speed_floor, 0.0)) * s
        a = 1.0 / (1.0 + (1.0 / (2.0 * math.pi * cutoff)) / dt)
        self.x = self.x + a * (x - self.x)
        return self.x.copy()
