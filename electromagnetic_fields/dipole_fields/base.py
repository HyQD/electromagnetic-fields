"""Common interface and validation for temporal dipole fields."""

from numbers import Real

import numpy as np


def finite_real(value, name):
    """Return *value* as a finite float."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real scalar.")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real scalar.")
    return float(value)


def positive_real(value, name):
    """Return *value* as a positive finite float."""
    result = finite_real(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result


class TemporalField:
    r"""Gauge-selected scalar field used in the electric-dipole approximation.

    Subclasses describe a physical electric field. Calling an instance returns
    its electric field in length gauge and its consistent vector potential in
    velocity gauge. Field-strength parameters therefore always refer to the
    electric field, irrespective of the selected representation.
    """

    supported_gauges = ()

    def __init__(self, gauge, *, time_support=(None, None)):
        if gauge not in self.supported_gauges:
            choices = ", ".join(self.supported_gauges)
            raise ValueError(
                f"Gauge {gauge!r} is not supported by {type(self).__name__}; "
                f"choose one of: {choices}."
            )
        start, end = time_support
        if start is not None:
            start = finite_real(start, "time-support start")
        if end is not None:
            end = finite_real(end, "time-support end")
        if start is not None and end is not None and end < start:
            raise ValueError("The end of the time support cannot precede its start.")
        self.gauge = gauge
        self._time_support = (start, end)

    @property
    def time_support(self):
        """Nominal ``(start, end)`` support; ``None`` denotes an open end."""
        return self._time_support

    def __call__(self, time):
        if self.gauge == "length":
            return self.electric_field(time)
        return self.vector_potential(time)
