# electromagnetic-fields

Temporal electromagnetic-field profiles for quantum-dynamics calculations.

The dipole-field classes describe a physical electric field in atomic units.
Their `gauge` parameter selects the value returned when the object is called:

- `gauge="length"` returns the electric field $E(t)$;
- `gauge="velocity"` returns the corresponding vector potential $A(t)$.

When both gauges are supported, the implementations use the convention

$$E(t)=-\frac{dA(t)}{dt}.$$

Field-strength parameters such as `E0` always specify an electric-field
amplitude, including when the object returns a vector potential.

See [the dipole-field documentation](electromagnetic_fields/dipole_fields/README.md)
for the available profiles and their supported gauges.
