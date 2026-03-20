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
