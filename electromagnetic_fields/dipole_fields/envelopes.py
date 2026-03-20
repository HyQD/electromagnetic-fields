import numpy as np


class SineSquareEnvelope:

    def __init__(self, t0, t_duration):
        self.t0 = t0
        self.t_duration = t_duration

    def __call__(self, t):
        """
        Evaluate the enevelope
            F(t) = sin^2(pi * (t - t0) / t_duration) for t0 <= t <= t0 + t_duration
                   0 otherwise
        """
        dt = t - self.t0
        return (
            (np.sin(np.pi * dt / self.t_duration) ** 2)
            * np.heaviside(dt, 1.0)
            * np.heaviside(self.t_duration - dt, 1.0)
        )


class GaussianEnvelope:
    def __init__(self, tc, sigma):
        self.tc = tc
        self.sigma = sigma

    def __call__(self, t):
        """
        Evaluate the Gaussian envelope
            F(t) = exp(-(t - tc)**2 / (2*sigma**2))
        """
        return np.exp(-((t - self.tc) ** 2) / (2 * self.sigma**2))


class TrigonometricGaussianEnvelope:
    """
    The trigonometric envelope for a given n
        T(t;n,t_c) = cos( pi * (t - tc) / \tau_n)**n,
    approximates a Gaussian envelope of the form
        G(t) = exp(-(t - tc)**2 / (2*sigma**2))
    where
        \tau_n = pi * sqrt(ln(2)) * sigma / arccos(2^(-1/(2n)))
    -------------------------------------------------
    Ref: DOI, 10.1088/0953-4075/42/23/235101
    """

    def __init__(self, tc, sigma, n):
        """
        Parameters
        ----------
        tc : float
            Center of the Gaussian to approximate.
        sigma : float
            Width of the Gaussian to approximate.
        n : int
            Order of the trigonometric envelope.
        """
        self.tc = tc
        self.n = n
        self.tau_n = (
            np.pi * np.sqrt(np.log(2)) * sigma / np.arccos(2 ** (-1 / (2 * n)))
        )

    def __call__(self, t):
        return np.cos(np.pi * (t - self.tc) / self.tau_n) ** self.n
