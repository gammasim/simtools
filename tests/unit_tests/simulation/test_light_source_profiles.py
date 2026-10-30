#!/usr/bin/python3

import math

import numpy as np
import pytest

from simtools.simulation.light_source_profiles import angular_profile, pulse_profile


def test_angular_profile():
    angles, values = angular_profile("Isotropic", 0.0)
    assert angles.tolist() == [0.0, 180.0]
    assert values.tolist() == [1.0, 1.0]
    angles, values = angular_profile("Gauss", 2.0, samples=101, n_sigma=5.0)
    assert len(angles) == 101
    assert math.isclose(angles[-1], 10.0, rel_tol=1e-6, abs_tol=1e-12)
    assert values[20] == pytest.approx(math.exp(-0.5))  # At 2 deg, one standard deviation.
    angles, values = angular_profile("FilledCone", 11.0)
    assert math.isclose(angles[1], 11.0, rel_tol=1e-6, abs_tol=1e-12)
    assert values.tolist() == [1.0, 1.0, 0.0]
    angles, values = angular_profile("Lambertian", 60.0)
    assert math.isclose(angles[-1], 60.0, rel_tol=1e-6, abs_tol=1e-12)
    assert math.isclose(values[-1], 0.5, rel_tol=1e-6, abs_tol=1e-12)
    with pytest.raises(ValueError, match="positive width"):
        angular_profile("Gauss", 0.0)
    with pytest.raises(ValueError, match="no tabulated profile"):
        angular_profile("Rayleigh", 5.0)


def test_pulse_profile():
    times, rates = pulse_profile("TopHat", 2.5, 0.0)
    assert times.tolist() == [-1.25, 1.25]
    assert rates.tolist() == [1.0, 1.0]
    times, rates = pulse_profile("Gauss", 1.5, 0.0)
    assert math.isclose(np.average(times, weights=rates), 0.0, rel_tol=1e-6, abs_tol=1e-12)
    assert math.isclose(
        math.sqrt(np.average(times**2, weights=rates)), 1.5, rel_tol=1e-3, abs_tol=1e-12
    )
    assert pulse_profile("Gauss", 0.0, 0.0) is None
    times, rates = pulse_profile("Exponential", 0.0, 2.0)
    assert np.allclose((times[0], rates[0]), (0.0, 1.0), rtol=1e-6, atol=1e-12)
    # Mean of the density, linear between the nodes.
    mean = np.trapezoid(times * rates, times) / np.trapezoid(rates, times)
    assert math.isclose(mean, 2.0, rel_tol=1e-2, abs_tol=1e-12)
    assert pulse_profile("Exponential", 0.0, 0.0) is None
    times, rates = pulse_profile("Gauss-Exponential", 1.59, 0.4)
    assert math.isclose(times[np.argmax(rates)], 0.0, rel_tol=1e-6, abs_tol=1e-12)
    assert math.isclose(rates.max(), 1.0, rel_tol=1e-6, abs_tol=1e-12)
    assert np.all(rates >= 0.0)
    assert np.all(np.diff(times) > 0.0)
    above = times[rates >= 0.1]
    assert above[0] < 0.0 < above[-1]
    with pytest.raises(ValueError, match="rise and fall"):
        pulse_profile("Gauss-Exponential", 1.0, 0.0)
    with pytest.raises(ValueError, match="no tabulated profile"):
        pulse_profile("Lorentz", 1.0, 0.0)
