import numpy as np
from matplotlib import pyplot as plt
from electromagnetic_fields.dipole_fields import AdiabaticLaser

E0 = 1.0
omega = 0.057
t_c = 2 * np.pi / omega
n_ramp_cycles = 2
n_post_ramp_cycles = 1
tfinal = n_ramp_cycles * t_c + n_post_ramp_cycles * t_c
time_points = np.linspace(0, tfinal, 1000)

switch_functions = ["linear", "quadratic", "fermi"]

plt.figure()

for switch in switch_functions:
    electric_field = AdiabaticLaser(
        F_str=E0,
        omega=omega,
        n_switch=n_ramp_cycles,
        switch=switch,
    )
    plt.plot(
        time_points, electric_field(time_points), label=f"Switch: {switch}"
    )


plt.xlabel("Time")
plt.ylabel("Electric field")
plt.legend()
plt.axvline(x=n_ramp_cycles * t_c, color="black", linestyle="--")
plt.show()
