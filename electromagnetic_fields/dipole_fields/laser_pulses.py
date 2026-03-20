import numpy as np
from scipy.special import comb
from electromagnetic_fields.dipole_fields.envelopes import (
    SineSquareEnvelope,
    TrigonometricGaussianEnvelope,
    GaussianEnvelope,
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


class GaussianLaser:

    def __init__(self, E0, omega, phase, sigma, tc=0.0, gauge="length"):
        self.E0 = E0
        self.omega = omega
        self.phase = phase
        self._envelope = GaussianEnvelope(tc, sigma)

        if gauge == "length":
            self._callable = self.electric_field
        else:
            raise ValueError(f"Gauge '{gauge}' is not supported.")

    def __call__(self, t):
        return self._callable(t)

    def electric_field(self, t):
        return (
            self.E0
            * self._envelope(t)
            * np.sin(self.omega * (t - self._envelope.tc) + self.phase)
        )


class TrigonometricGaussianLaser:
    """
    Approximate a monochromatic wave modulated by a Gaussian envelope
        E_G(t) = E0 * exp(-(t - tc)**2 / (2*sigma**2)) * cos(omega * (t - tc) + phase)
    using a trigonometric envelope of the form
        T(t;n,t_c) = cos( pi * (t - tc) / \tau_n)**n, \tau_n = pi * sqrt(ln(2)) * sigma / arccos(2^(-1/(2n)))
    so that
        E_Tn(t) = E0 * T(t) * cos(omega * (t - tc) + phase), tc-\tau_n/2 <= t <= tc+\tau_n/2
        E_Tn(t) = 0, otherwise.

    The corresponding vector potential is obtained from
        A_Tn(t) = - integral E_Tn(t) dt.
    """

    def __init__(self, E0, omega, phase, sigma, n, tc=None, gauge="length"):
        self.n = n
        self.tau_n = (
            np.pi * np.sqrt(np.log(2)) * sigma / np.arccos(2 ** (-1 / (2 * n)))
        )

        self.E0 = E0
        self.omega = omega
        self.phase = phase
        if tc is None:
            self.tc = self.tau_n / 2
        else:
            self.tc = tc

        self.t_start = self.tc - self.tau_n / 2
        self.t_end = self.tc + self.tau_n / 2

        self._envelope = TrigonometricGaussianEnvelope(self.tc, sigma, n)

        self.At_start = self._At(self.t_start)
        self.At_end = self._At(self.t_end)

        if gauge == "length":
            self._callable = self.electric_field
        elif gauge == "velocity":
            self._callable = self.vector_potential
        else:
            raise ValueError(f"Gauge '{gauge}' is not supported.")

    def __call__(self, t):
        return self._callable(t)

    def electric_field(self, t):
        return (
            self.E0
            * self._envelope(t)
            * np.sin(self.omega * (t - self.tc) + self.phase)
            * np.heaviside(t - self.t_start, 1.0)
            * np.heaviside(self.t_end - t, 1.0)
        )

    def _At(self, t):
        if type(t) == np.ndarray:
            At = np.zeros_like(t)
        else:
            At = 0.0

        for k in range(0, self.n + 1):
            n_choose_k = comb(self.n, k)
            Omega_k = (self.n - 2 * k) * np.pi / self.tau_n + self.omega
            At += (
                n_choose_k
                * np.cos(Omega_k * (t - self.tc) + self.phase)
                / Omega_k
            )
        At *= self.E0 / (2 ** (self.n))

        return At

    def vector_potential(self, t):
        if isinstance(t, np.ndarray):
            result = np.zeros_like(t, dtype=float)
            mask = (t >= self.t_start) & (t <= self.t_end)
            result[mask] = self._At(t[mask]) - self.At_start
            result[~mask] = self.At_end - self.At_start
            return result
        else:
            if self.t_start <= t <= self.t_end:
                return self._At(t) - self.At_start
            else:
                return self.At_end - self.At_start
