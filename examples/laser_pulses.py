import numpy as np
from matplotlib import pyplot as plt
from electromagnetic_fields.dipole_fields.laser_pulses import SineSquareLaser

E0 = 1.0
omega = 1.0
ncycles = 3
tc = 2*np.pi/omega
td = ncycles * tc

e_field = SineSquareLaser(E0=E0, omega=omega, ncycles=ncycles, gauge="length", phase=0.0, t0=0.0)
a_field = SineSquareLaser(E0=E0, omega=omega, ncycles=ncycles, gauge="velocity", phase=0.0, t0=0.0)
time_points = np.linspace(0, 10*np.pi, 1000)

title = r"$E(t) = E_0\sin^2(\pi t / t_d) \sin(\omega t + \varphi), t_0 \leq t \leq t_d; 0$" + f" else \n"
title += rf"$E_0={E0}, \omega={omega}, n_c={ncycles}, \varphi=0, t_0=0, t_d=n_c t_c, t_c = 2\pi/\omega$"
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.set_title(title, fontsize=16)
ax1.plot(time_points, e_field(time_points), color='red')
ax1.axvline(x=td, color='black', linestyle='--', label=r"$t_d$")
ax1.grid()
ax1.legend()
ax2.plot(time_points, a_field(time_points), color='blue', label=f"Corresponding vector potential " + r"$A(t)$")
ax2.axvline(x=td, color='black', linestyle='--', label=r"$t_d$")
ax2.grid()
ax2.legend()
ax2.set_xlabel(f"time (a.u.)")
plt.show()

