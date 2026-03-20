import numpy as np
import matplotlib.pyplot as plt
from electromagnetic_fields.dipole_fields.laser_pulses import GaussianLaser, TrigonometricGaussianLaser

t_au = 2.418884326505e-17  # seconds
t_1fs = 1e-15 / t_au  # atomic units

E0 = 1.0
omega = 0.08
phase = 0.0
tc = 0.0
sigma = 15 * t_1fs

time_points = np.linspace(-60*t_1fs, 60*t_1fs, 10000)
gaussian_laser = GaussianLaser(E0, omega, phase, sigma, tc)
trig_laser_2 = TrigonometricGaussianLaser(E0, omega, phase, sigma, n=2, tc=tc)
trig_laser_4 = TrigonometricGaussianLaser(E0, omega, phase, sigma, n=4, tc=tc)
trig_laser_8 = TrigonometricGaussianLaser(E0, omega, phase, sigma, n=8, tc=tc)

title = "Comparison of Gaussian and Trigonometric Gaussian Lasers \n"
title += rf"$E(t) = E_0 \exp(-t^2 / (2\sigma^2)) \sin(\omega (t-t_c) + \phi)$" + "\n"
title += rf"$E_0 = {E0}, \omega = {omega}, \phi = {phase}, t_c={tc}, \sigma = {sigma/t_1fs:.2f} $" + "fs"

fig, axes = plt.subplots(4, 1, sharex=True)
fig.suptitle(title)
axes[0].plot(time_points / t_1fs, gaussian_laser(time_points), label="Gaussian Laser")
axes[0].legend()
axes[1].plot(time_points / t_1fs, trig_laser_2(time_points), label="Trigonometric Gaussian Laser (n=2)")
axes[1].legend()
axes[2].plot(time_points / t_1fs, trig_laser_4(time_points), label="Trigonometric Gaussian Laser (n=4)")
axes[2].legend()
axes[3].plot(time_points / t_1fs, trig_laser_8(time_points), label="Trigonometric Gaussian Laser (n=8)")
axes[3].legend()
for ax in axes:
    ax.grid()
axes[-1].set_xlabel("Time (fs)")
plt.show()