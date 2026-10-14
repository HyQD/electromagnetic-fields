"""Finite and Gaussian laser pulses in length and velocity gauge."""

from math import comb

import numpy as np

from .base import TemporalField, finite_real, positive_real
from .envelopes import (
    GaussianEnvelope,
    SineSquareEnvelope,
    TrigonometricGaussianEnvelope,
)


def _integral_sine(frequency, upper, lower, phase):
    """Return the integral of ``sin(frequency*t + phase)``."""
    scale = max(1.0, abs(frequency))
    if abs(frequency) <= 10 * np.finfo(float).eps * scale:
        return (upper - lower) * np.sin(phase)
    return (
        np.cos(frequency * lower + phase)
        - np.cos(frequency * upper + phase)
    ) / frequency


class SineSquareLaser(TemporalField):
    r"""Sine carrier with a finite sine-square envelope.

    ``E0`` is the peak electric-field parameter in both gauges. Length gauge
    returns the electric field, while velocity gauge returns a vector
    potential satisfying ``E(t) = -dA(t)/dt`` and ``A(t0) = 0``.

    ``ncycles`` may be any positive real number. Fractional cycles generally
    produce a nonzero pulse area, so the vector potential can approach a
    nonzero constant after the electric-field pulse has ended.
    """

    supported_gauges = ("length", "velocity")
    required_params = {"E0", "omega", "ncycles", "gauge"}

    def __init__(self, *, E0, omega, ncycles, gauge, phase=0.0, t0=0.0):
        self.E0 = finite_real(E0, "E0")
        self.omega = positive_real(omega, "omega")
        self.ncycles = positive_real(ncycles, "ncycles")
        self.A0 = self.E0 / self.omega
        self.tprime = 2 * self.ncycles * np.pi / self.omega
        self.phase = phase
        self.t0 = finite_real(t0, "t0")
        if not callable(phase):
            self.phase = finite_real(phase, "phase")
        elif gauge == "velocity":
            raise ValueError(
                "A callable phase is supported only in length gauge because "
                "its consistent vector potential is not available analytically."
            )
        self._envelope = SineSquareEnvelope(self.t0, self.tprime)
        super().__init__(
            gauge,
            time_support=(self.t0, self.t0 + self.tprime),
        )

    def _phase(self, time):
        return self.phase(time) if callable(self.phase) else self.phase

    def electric_field(self, time):
        values = np.asarray(time)
        delay = values - self.t0
        return (
            self.E0
            * self._envelope(values)
            * np.sin(self.omega * delay + self._phase(delay))
        )

    def vector_potential(self, time):
        delay = np.asarray(time, dtype=float) - self.t0
        upper = np.clip(delay, 0.0, self.tprime)
        envelope_frequency = 2 * np.pi / self.tprime
        integral = self.E0 * (
            0.5 * _integral_sine(self.omega, upper, 0.0, self.phase)
            - 0.25
            * _integral_sine(
                self.omega + envelope_frequency,
                upper,
                0.0,
                self.phase,
            )
            - 0.25
            * _integral_sine(
                self.omega - envelope_frequency,
                upper,
                0.0,
                self.phase,
            )
        )
        return -integral


class GaussianLaser(TemporalField):
    """Gaussian-envelope laser currently available in length gauge."""

    supported_gauges = ("length",)

    def __init__(self, E0, omega, phase, sigma, tc=0.0, gauge="length"):
        self.E0 = finite_real(E0, "E0")
        self.omega = positive_real(omega, "omega")
        self.phase = finite_real(phase, "phase")
        self.sigma = positive_real(sigma, "sigma")
        self.tc = finite_real(tc, "tc")
        self._envelope = GaussianEnvelope(self.tc, self.sigma)
        super().__init__(gauge, time_support=(None, None))

    def electric_field(self, time):
        return (
            self.E0
            * self._envelope(time)
            * np.sin(self.omega * (np.asarray(time) - self.tc) + self.phase)
        )


class TrigonometricGaussianLaser(TemporalField):
    r"""Compact trigonometric approximation to a Gaussian laser pulse.

    The same electric-field amplitude ``E0`` is used in both gauges. The
    velocity-gauge representation satisfies ``E(t) = -dA(t)/dt`` and chooses
    ``A(t_start) = 0``.
    """

    supported_gauges = ("length", "velocity")

    def __init__(self, E0, omega, phase, sigma, n, tc=None, gauge="length"):
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
            raise TypeError("n must be a positive integer.")
        if n <= 0:
            raise ValueError("n must be a positive integer.")
        self.n = int(n)
        self.E0 = finite_real(E0, "E0")
        self.omega = positive_real(omega, "omega")
        self.phase = finite_real(phase, "phase")
        self.sigma = positive_real(sigma, "sigma")
        self.tau_n = (
            np.pi
            * np.sqrt(np.log(2))
            * self.sigma
            / np.arccos(2 ** (-1 / (2 * self.n)))
        )
        self.tc = self.tau_n / 2 if tc is None else finite_real(tc, "tc")
        self.t_start = self.tc - self.tau_n / 2
        self.t_end = self.tc + self.tau_n / 2
        self._envelope = TrigonometricGaussianEnvelope(
            self.tc, self.sigma, self.n
        )
        super().__init__(gauge, time_support=(self.t_start, self.t_end))

    def electric_field(self, time):
        values = np.asarray(time)
        return (
            self.E0
            * self._envelope(values)
            * np.sin(self.omega * (values - self.tc) + self.phase)
            * np.heaviside(values - self.t_start, 1.0)
            * np.heaviside(self.t_end - values, 1.0)
        )

    def _vector_potential_inside(self, time):
        relative_time = np.asarray(time) - self.tc
        relative_start = self.t_start - self.tc
        result = np.zeros_like(relative_time, dtype=float)
        for k in range(self.n + 1):
            frequency = (self.n - 2 * k) * np.pi / self.tau_n + self.omega
            result -= (
                comb(self.n, k)
                * _integral_sine(
                    frequency,
                    relative_time,
                    relative_start,
                    self.phase,
                )
            )
        return self.E0 * result / (2**self.n)

    def vector_potential(self, time):
        values = np.asarray(time, dtype=float)
        clipped = np.clip(values, self.t_start, self.t_end)
        return self._vector_potential_inside(clipped)
