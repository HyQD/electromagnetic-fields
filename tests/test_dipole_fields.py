import numpy as np
import pytest

from electromagnetic_fields import (
    AdiabaticField,
    AdiabaticLaser,
    DiscreteDeltaPulse,
    GaussianDeltaPulse,
    GaussianLaser,
    SineSquareLaser,
    TrigonometricGaussianLaser,
)
from electromagnetic_fields.dipole_fields.ramped_wave import (
    FermiSwitch,
    LinearSwitch,
    QuadraticSwitch,
)


@pytest.mark.parametrize(
    "factory",
    [
        lambda gauge: SineSquareLaser(
            E0=0.2,
            omega=0.7,
            ncycles=2,
            phase=0.3,
            t0=0.2,
            gauge=gauge,
        ),
        lambda gauge: TrigonometricGaussianLaser(
            E0=0.2,
            omega=0.7,
            phase=0.3,
            sigma=3.0,
            n=4,
            tc=5.0,
            gauge=gauge,
        ),
    ],
)
def test_length_and_velocity_gauges_describe_same_electric_field(factory):
    length = factory("length")
    velocity = factory("velocity")
    start, end = length.time_support
    times = np.linspace(start + 0.1, end - 0.1, 31)
    spacing = 1e-5
    derivative = (
        velocity.vector_potential(times + spacing)
        - velocity.vector_potential(times - spacing)
    ) / (2 * spacing)
    np.testing.assert_allclose(length(times), -derivative, atol=2e-9, rtol=2e-8)
    np.testing.assert_allclose(length(times), length.electric_field(times))
    np.testing.assert_allclose(velocity(times), velocity.vector_potential(times))


def test_one_cycle_sine_square_handles_zero_frequency_term():
    field = SineSquareLaser(
        E0=0.2,
        omega=0.7,
        ncycles=1,
        phase=0.3,
        t0=0.2,
        gauge="velocity",
    )
    start, end = field.time_support
    times = np.linspace(start + 0.1, end - 0.1, 31)
    spacing = 1e-5
    derivative = (
        field.vector_potential(times + spacing)
        - field.vector_potential(times - spacing)
    ) / (2 * spacing)
    length = SineSquareLaser(
        E0=0.2,
        omega=0.7,
        ncycles=1,
        phase=0.3,
        t0=0.2,
        gauge="length",
    )
    np.testing.assert_allclose(length(times), -derivative, atol=2e-9, rtol=2e-8)


def test_sine_square_respects_nonzero_start_time():
    field = SineSquareLaser(
        E0=1.0,
        omega=0.8,
        ncycles=2,
        phase=0.0,
        t0=3.0,
        gauge="length",
    )
    start, end = field.time_support
    assert start == 3.0
    np.testing.assert_allclose(field([start - 1, start, end, end + 1]), 0, atol=1e-28)
    assert abs(field(start + 0.25 * (end - start))) > 0


def test_discrete_delta_uses_t0_as_interval_start_and_has_requested_area():
    field = DiscreteDeltaPulse(F_str=0.3, dt=0.2, t0=1.5)
    assert field.gauge == "length"
    assert field.time_support == (1.5, 1.7)
    np.testing.assert_allclose(
        field(np.array([1.49, 1.5, 1.6, 1.7])),
        np.array([0.0, 1.5, 1.5, 0.0]),
    )
    assert field.F_str == pytest.approx(field.dt * field(1.5))


@pytest.mark.parametrize(
    "factory",
    [
        lambda: GaussianLaser(1.0, 0.5, 0.0, 2.0, gauge="velocity"),
        lambda: DiscreteDeltaPulse(1.0, 0.1, gauge="velocity"),
        lambda: GaussianDeltaPulse(gauge="velocity"),
        lambda: AdiabaticField(1.0, 0.5, gauge="velocity"),
    ],
)
def test_length_only_fields_reject_velocity_gauge(factory):
    with pytest.raises(ValueError, match="not supported"):
        factory()


def test_callable_sine_square_phase_is_length_only():
    phase = lambda time: 0.1 * np.asarray(time)
    field = SineSquareLaser(
        E0=0.2,
        omega=0.7,
        ncycles=2,
        phase=phase,
        gauge="length",
    )
    assert np.all(np.isfinite(field(np.linspace(0, field.tprime, 11))))
    with pytest.raises(ValueError, match="callable phase"):
        SineSquareLaser(
            E0=0.2,
            omega=0.7,
            ncycles=2,
            phase=phase,
            gauge="velocity",
        )


def test_adiabatic_laser_is_backward_compatible_alias():
    assert AdiabaticLaser is AdiabaticField
    field = AdiabaticLaser(1.0, 0.5)
    assert field.gauge == "length"
    assert field.time_support == (0.0, None)


@pytest.mark.parametrize(
    "switch",
    [
        FermiSwitch(1.0, 3.0, 0.2),
        LinearSwitch(1.0, 3.0),
        QuadraticSwitch(1.0, 3.0),
    ],
)
def test_switch_scalar_and_array_evaluation_agree(switch):
    times = np.array([0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
    expected = np.array([switch(float(time)) for time in times])
    np.testing.assert_allclose(switch(times), expected)
    assert switch(0.0) == 0
    assert switch(4.0) == 1
