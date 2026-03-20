import numpy as np
from electromagnetic_fields.dipole_fields.envelopes import (
    SineSquareEnvelope,
    TrigonometricGaussianEnvelope,
)


class SineSquareLaser:
    supported_gauges = ("length", "velocity")
    required_params = {"E0", "omega", "ncycles", "gauge", "phase", "t0"}

    def __init__(self, *, E0, omega, ncycles, gauge, phase=0.0, t0=0.0):
        self.E0 = E0
        self.A0 = E0 / omega
        self.omega = omega
        self.tprime = 2 * ncycles * np.pi / omega
        self.phase = phase
        self.t0 = t0
        self._envelope = SineSquareEnvelope(t0, self.tprime)

        self._ncycles_is_one = True if ncycles == 1 else False
        self._A1_t0 = self._A1(0)

        if gauge == "length":
            self._callable = self.electric_field
        elif gauge == "velocity":
            self._callable = self.vector_potential
        else:
            raise ValueError(f"Gauge '{gauge}' is not supported.")

    def __call__(self, t):
        return self._callable(t)

    def _phase(self, t):
        if callable(self.phase):
            return self.phase(t)
        else:
            return self.phase

    def electric_field(self, t):
        dt = t - self.t0
        pulse = (
            self._envelope(dt)
            * np.sin(self.omega * dt + self._phase(dt))
            * self.E0
        )
        return pulse

    def vector_potential(self, t):
        dt = t - self.t0

        pulse = self.E0 * (self._A1(dt) - self._A1_t0) * self._envelope(dt)
        return pulse

    def _A1(self, t):
        if not self._ncycles_is_one:
            f0 = (
                self.tprime
                * np.cos(
                    t * (self.omega - 2 * np.pi / self.tprime) + self._phase(t)
                )
                / (self.omega * self.tprime - 2 * np.pi)
            )
        else:
            f0 = 0
        f1 = (
            self.tprime
            * np.cos(
                t * (self.omega + 2 * np.pi / self.tprime) + self._phase(t)
            )
            / (self.omega * self.tprime + 2 * np.pi)
        )
        f2 = 2 * np.cos(self._phase(t)) * np.cos(self.omega * t) / self.omega
        f3 = 2 * np.sin(self._phase(t)) * np.sin(self.omega * t) / self.omega
        return (1 / 4.0) * (-f0 - f1 + f2 - f3)
