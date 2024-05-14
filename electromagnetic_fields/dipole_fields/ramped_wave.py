import numpy as np

class AdiabaticLaser:

    def __init__(self, F_str, omega, phase=0.0, n_switch=1, switch='quadratic', delta=None):
        self.F_str = F_str
        self.omega = omega
        self.n_switch = n_switch
        self.t_cycle = 2*np.pi/omega
        self.phase = phase
        self._envelope = self._select_envelope(switch.lower(), delta)

    def _continuous_wave(self, t):
        return self.F_str*np.cos(self.omega*t+self.phase)

    def _select_envelope(self, switch, delta):
        t0 = 0.
        t1 = self.n_switch*self.t_cycle
        if switch == 'fermi':
            if delta is None:
                eps = 1e-4*self.F_str
            else:
                eps = delta
            tau = -0.5*t1/np.log(eps/(1-eps))
            return FermiSwitch(t0, t1, tau)
        elif switch == 'linear':
            return LinearSwitch(t0, t1)
        elif switch == 'quadratic':
            return QuadraticSwitch(t0, t1)
        else:
            raise ValueError(f'Illegal switch: {switch}')

    def __call__(self, t):
        return self._continuous_wave(t)*self._envelope(t)

class FermiSwitch:

    def __init__(self, t0, t1, tau):
        self.t0 = t0
        self.t1 = t1
        self.tau = tau

    def name(self):
        return 'Fermi'

    def __call__(self, t):
        if type(t) == np.ndarray:
            # t is assumed in ascending order and uniformly spaced
            t_mid = 0.5*(self.t1 - self.t0)
            dt = t[1] - t[0]
            indx_0 = int(self.t0/dt)
            indx_1 = int(self.t1/dt)
            res = np.empty(t.shape[0])
            res[:indx_0] = 0.
            res[indx_0:indx_1] = 1 - 1/(1 + np.exp((t[indx_0:indx_1] - t_mid)/self.tau))
            res[indx_1:] = 1.
        else:
            if t < self.t0:
                res = 0.
            elif t <= self.t1:
                res = (t - self.t0)/(self.t1 - self.t0)
            else:
                res = 1.
        return res


class LinearSwitch:

    def __init__(self, t0, t1):
        self.t0 = t0
        self.t1 = t1

    def name(self):
        return 'linear'

    def __call__(self, t):
        if type(t) == np.ndarray:
            # t is assumed in ascending order and uniformly spaced
            dt = t[1] - t[0]
            indx_0 = int(self.t0/dt)
            indx_1 = int(self.t1/dt)
            res = np.empty(t.shape[0])
            res[:indx_0] = 0.
            res[indx_0:indx_1] = (t[indx_0:indx_1] - self.t0)/(self.t1 - self.t0)
            res[indx_1:] = 1.
        else:
            if t < self.t0:
                res = 0.
            elif t <= self.t1:
                res = (t - self.t0)/(self.t1 - self.t0)
            else:
                res = 1.
        return res


class QuadraticSwitch:

    def __init__(self, t0, t1):
        self.t0 = t0
        self.t1 = t1
        self.t_mid = 0.5*(t1 - t0)
        self.alpha = 2/(t1-3*t0)**2

    def name(self):
        return 'quadratic'

    def __call__(self, t):
        if type(t) == np.ndarray:
            # t is assumed in ascending order and uniformly spaced
            dt = t[1] - t[0]
            indx_0 = int(self.t0/dt)
            indx_m = int(self.t_mid/dt)
            indx_1 = int(self.t1/dt)
            res = np.empty(t.shape[0])
            res[:indx_0] = 0.
            res[indx_0:indx_m] = self.alpha*(t[indx_0:indx_m] - self.t0)**2
            res[indx_m:indx_1] = -self.alpha*(t[indx_m:indx_1] - self.t1)**2 + 1
            res[indx_1:] = 1.
        else:
            if t < self.t0:
                res = 0.
            elif t <= self.t_mid:
                res = self.alpha*(t - self.t0)**2
            elif t <= self.t1:
                res = -self.alpha*(t - self.t1)**2 + 1
            else:
                res = 1.
        return res
                

