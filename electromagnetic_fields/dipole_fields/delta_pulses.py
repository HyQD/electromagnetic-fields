"""Finite-width approximations to electric-field delta pulses."""

import numpy as np

from .base import TemporalField, finite_real, positive_real


class DiscreteDeltaPulse(TemporalField):
    r"""Right-sided rectangular approximation to ``F_str*delta(t - t0)``.

    The electric field equals ``F_str/dt`` on ``[t0, t0 + dt)`` and vanishes
    elsewhere, so its time integral is ``F_str``. In velocity gauge the
    vector potential is chosen to vanish before the pulse, decrease linearly
    during the pulse, and equal ``-F_str`` afterward. Consequently,
    ``E(t) = -dA(t)/dt`` away from the interval endpoints.
    """

    supported_gauges = ("length", "velocity")

    def __init__(self, F_str, dt, t0=0.0, gauge="length"):
        self.F_str = finite_real(F_str, "F_str")
        self.dt = positive_real(dt, "dt")
        self.t0 = finite_real(t0, "t0")
        super().__init__(gauge, time_support=(self.t0, self.t0 + self.dt))

    def electric_field(self, time):
        values = np.asarray(time)
        inside = (values >= self.t0) & (values < self.t0 + self.dt)
        return np.where(inside, self.F_str / self.dt, 0.0)

    def vector_potential(self, time):
        values = np.asarray(time, dtype=float)
        elapsed_fraction = np.clip((values - self.t0) / self.dt, 0.0, 1.0)
        return -self.F_str * elapsed_fraction


class GaussianDeltaPulse(TemporalField):
    r"""Normalized Gaussian approximation with time integral ``F_str``."""

    supported_gauges = ("length",)

    def __init__(
        self,
        F_str=1e-3,
        t_c=5.0,
        gamma=5.0,
        gauge="length",
    ):
        self.F_str = finite_real(F_str, "F_str")
        self.t_c = finite_real(t_c, "t_c")
        self.gamma = positive_real(gamma, "gamma")
        super().__init__(gauge, time_support=(None, None))

    def electric_field(self, time):
        return (
            self.F_str
            * np.sqrt(self.gamma / np.pi)
            * np.exp(-self.gamma * (np.asarray(time) - self.t_c) ** 2)
        )
