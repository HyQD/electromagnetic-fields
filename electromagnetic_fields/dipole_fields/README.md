# Dipole fields

These classes provide scalar temporal fields for spatially uniform
dipole-approximation interactions. Every field exposes:

- `gauge`, the representation returned by `field(t)`;
- `supported_gauges`, the representations implemented by its class;
- `electric_field(t)`;
- `vector_potential(t)` when velocity gauge is supported;
- `time_support`, a `(start, end)` tuple in which `None` denotes an open end.

## Available profiles

| Class | Length gauge | Velocity gauge | Nominal time support |
| --- | --- | --- | --- |
| `SineSquareLaser` | yes | yes | finite |
| `TrigonometricGaussianLaser` | yes | yes | finite |
| `GaussianLaser` | yes | no | unbounded |
| `DiscreteDeltaPulse` | yes | no | finite |
| `GaussianDeltaPulse` | yes | no | unbounded |
| `AdiabaticField` | yes | no | begins at zero, no finite end |

`AdiabaticLaser` remains available as an alias of `AdiabaticField`.

For profiles supporting both gauges, the same physical parameters construct
either representation:

```python
from electromagnetic_fields import SineSquareLaser

electric_field = SineSquareLaser(
    E0=1e-4,
    omega=0.057,
    ncycles=1,
    phase=0.0,
    gauge="length",
)
vector_potential = SineSquareLaser(
    E0=1e-4,
    omega=0.057,
    ncycles=1,
    phase=0.0,
    gauge="velocity",
)
```

Here `E0` is the electric-field amplitude in both constructions and
$E(t)=-dA(t)/dt$.

`SineSquareLaser.ncycles` may be any positive real number. Fractional cycles
generally produce a nonzero pulse area, $\int E(t)\,dt\ne 0$. Consequently,
the corresponding vector potential can remain at a nonzero constant after the
electric-field pulse has ended.

`DiscreteDeltaPulse(F_str, dt, t0)` is the right-sided approximation

$$E(t)=\frac{F_0}{\Delta t},\qquad t_0\le t<t_0+\Delta t,$$

and zero otherwise. Its time integral is `F_str`, so it approaches
$F_0\delta(t-t_0)$ as `dt` decreases.
