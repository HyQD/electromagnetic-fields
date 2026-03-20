import numpy as np
import matplotlib.pyplot as plt
from electromagnetic_fields.dipole_fields.delta_pulses import DiscreteDeltaPulse, GaussianDeltaPulse

F0 = 1e-3

time_points = np.linspace(0, 5, 100)
dt = time_points[1] - time_points[0]
discrete_dp = DiscreteDeltaPulse(F0, dt)

time_points_gaussian = np.linspace(-10, 10, 1000)
gaussian_dp = GaussianDeltaPulse(F_str=F0, t_c=0, gamma=1.0)

int_gaussian_dp = np.trapz(gaussian_dp(time_points_gaussian), time_points_gaussian)
print(f"Integral of Gaussian Delta Pulse / F0: {int_gaussian_dp / F0:.6f}")

plt.figure(1)
plt.plot(time_points, discrete_dp(time_points), '-o', label='Discrete Delta Pulse')
plt.axvline(x=dt, color='r', linestyle='--', label='dt')
plt.grid()
plt.legend()

title = r"$G(t) = \frac{F_0}{{\sqrt{{\pi / \gamma}}}} e^{{-\gamma (t - t_c)^2}}$" + "\n"
title += rf"$F_0$ = {F0:.1e}, $t_c$ = 0, $\gamma$ = 1.0"
plt.figure(2)
plt.title(title)
plt.plot(time_points_gaussian, gaussian_dp(time_points_gaussian))
plt.fill_between(time_points_gaussian, gaussian_dp(time_points_gaussian), alpha=0.3, label=r"$\frac{1}{F_0}\int G(t) dt = %.1f$" % (int_gaussian_dp/F0))
plt.grid()
plt.legend()

plt.show()
