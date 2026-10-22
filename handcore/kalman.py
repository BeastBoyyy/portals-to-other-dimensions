"""Constant-velocity 2D Kalman filter for the palm centre: gating, prediction, dropout bridging."""
import math
import numpy as np

_H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)


class CVKalman:
    def __init__(self, accel_sigma=30000.0, meas_sigma=4.0, coast_decay_s=0.15):
        self.qa, self.r, self.tau = accel_sigma, meas_sigma, coast_decay_s
        self.x = None   # [x, y, vx, vy]
        self.P = None

    def init(self, z):
        self.x = np.array([z[0], z[1], 0.0, 0.0])
        self.P = np.diag([self.r ** 2, self.r ** 2, 1e6, 1e6])

    def predict(self, dt, coasting=False):
        dt = max(dt, 1e-3)
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        if coasting:
            F[2, 2] = F[3, 3] = math.exp(-dt / self.tau)
        G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + (G @ G.T) * self.qa ** 2
        return self.x[:2].copy()

    def innovation(self, z):
        S = _H @ self.P @ _H.T + np.eye(2) * self.r ** 2
        return np.asarray(z, float) - self.x[:2], S

    def maha2(self, z):
        y, S = self.innovation(z)
        return float(y @ np.linalg.solve(S, y))

    def update(self, z):
        y, S = self.innovation(z)
        d2 = float(y @ np.linalg.solve(S, y))
        if d2 > 9.0:                      # Huber-style: a far-off measurement is down-weighted, not trusted
            S = S * (d2 / 9.0)
        K = self.P @ _H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ _H) @ self.P

    @property
    def pos(self):
        return self.x[:2]

    @property
    def vel(self):
        return self.x[2:]
