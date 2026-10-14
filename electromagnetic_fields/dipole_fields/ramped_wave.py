"""Continuously driven electric fields with smooth switch-on envelopes."""

import numpy as np

from .base import TemporalField, finite_real, positive_real


class AdiabaticField(TemporalField):
    """Continuous electric field ramped on over ``n_switch`` cycles.

    The field continues after the switch-on interval and is currently
    available only in length gauge.
    """

    supported_gauges = ("length",)

    def __init__(
        self,
        F_str,
        omega,
        phase=0.0,
        n_switch=1,
        switch="quadratic",
        delta=None,
        gauge="length",
    ):
        self.F_str = finite_real(F_str, "F_str")
        self.omega = positive_real(omega, "omega")
        self.n_switch = positive_real(n_switch, "n_switch")
        self.t_cycle = 2 * np.pi / self.omega
        self.phase = finite_real(phase, "phase")
        self._envelope = self._select_envelope(switch.lower(), delta)
        super().__init__(gauge, time_support=(0.0, None))

    def _continuous_wave(self, time):
        return self.F_str * np.cos(self.omega * np.asarray(time) + self.phase)

    def _select_envelope(self, switch, delta):
        t0 = 0.0
        t1 = self.n_switch * self.t_cycle
        if switch == "fermi":
            epsilon = 1e-4 if delta is None else positive_real(delta, "delta")
            if epsilon >= 0.5:
                raise ValueError("delta must be smaller than 0.5.")
            tau = -0.5 * (t1 - t0) / np.log(epsilon / (1 - epsilon))
            return FermiSwitch(t0, t1, tau)
        if switch == "linear":
            return LinearSwitch(t0, t1)
        if switch == "quadratic":
            return QuadraticSwitch(t0, t1)
        raise ValueError(f"Illegal switch: {switch}")

    def electric_field(self, time):
        return self._continuous_wave(time) * self._envelope(time)


class FermiSwitch:
    """Fermi-like switch from zero to one on a finite interval."""

    def __init__(self, t0, t1, tau):
        self.t0 = finite_real(t0, "t0")
        self.t1 = finite_real(t1, "t1")
        self.tau = positive_real(tau, "tau")
        if self.t1 <= self.t0:
            raise ValueError("t1 must be larger than t0.")

    def name(self):
        return "Fermi"

    def __call__(self, time):
        values = np.asarray(time)
        midpoint = 0.5 * (self.t0 + self.t1)
        exponent = np.clip(-(values - midpoint) / self.tau, -700, 700)
        interior = 1 / (1 + np.exp(exponent))
        return np.where(
            values < self.t0,
            0.0,
            np.where(values <= self.t1, interior, 1.0),
        )


class LinearSwitch:
    """Linear switch from zero to one on a finite interval."""

    def __init__(self, t0, t1):
        self.t0 = finite_real(t0, "t0")
        self.t1 = finite_real(t1, "t1")
        if self.t1 <= self.t0:
            raise ValueError("t1 must be larger than t0.")

    def name(self):
        return "linear"

    def __call__(self, time):
        values = np.asarray(time)
        return np.clip((values - self.t0) / (self.t1 - self.t0), 0.0, 1.0)


class QuadraticSwitch:
    """Piecewise-quadratic switch with a continuous first derivative."""

    def __init__(self, t0, t1):
        self.t0 = finite_real(t0, "t0")
        self.t1 = finite_real(t1, "t1")
        if self.t1 <= self.t0:
            raise ValueError("t1 must be larger than t0.")
        self.t_mid = 0.5 * (self.t0 + self.t1)
        self.alpha = 2 / (self.t1 - self.t0) ** 2

    def name(self):
        return "quadratic"

    def __call__(self, time):
        values = np.asarray(time)
        rising = self.alpha * (values - self.t0) ** 2
        falling = 1 - self.alpha * (values - self.t1) ** 2
        return np.where(
            values < self.t0,
            0.0,
            np.where(
                values <= self.t_mid,
                rising,
                np.where(values <= self.t1, falling, 1.0),
            ),
        )


# Backward-compatible name retained for existing imports.
AdiabaticLaser = AdiabaticField
