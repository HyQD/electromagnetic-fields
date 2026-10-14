"""Time-dependent electromagnetic-field profiles."""

from .dipole_fields import ramped_wave
from .dipole_fields import (
    AdiabaticField,
    AdiabaticLaser,
    DiscreteDeltaPulse,
    GaussianDeltaPulse,
    GaussianLaser,
    SineSquareLaser,
    TemporalField,
    TrigonometricGaussianLaser,
)

__all__ = [
    "AdiabaticField",
    "AdiabaticLaser",
    "DiscreteDeltaPulse",
    "GaussianDeltaPulse",
    "GaussianLaser",
    "SineSquareLaser",
    "TemporalField",
    "TrigonometricGaussianLaser",
    "ramped_wave",
]
