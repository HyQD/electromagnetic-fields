import numpy as np

class DiscreteDeltaPulse:
    def __init__(self, F_str, dt):
        self.F_str = F_str
        self.dt = dt

    def __call__(self, t):
        return np.where(t < self.dt, self.F_str / self.dt, 0)
        
class GaussianDeltaPulse:
    """
    Normalized Gaussian pulse, with the unormalized pulse given by Eq.[17] Ref.[1].
    ---------
    [1]: https://pubs.acs.org/doi/10.1021/ct200137z
    """
    
    def __init__(self, F_str=1e-3, t_c=5, gamma=5.0):

        self.F_str = F_str
        self.t_c = t_c
        self.gamma = gamma

    def __call__(self, t):
        return (
            self.F_str
            * np.sqrt(self.gamma / np.pi)
            * np.exp(-self.gamma * (t - self.t_c) ** 2)
        )