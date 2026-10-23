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
    assert angles[-1] == pytest.approx(10.0)
    assert values[20] == pytest.approx(math.exp(-0.5))  # At 2 deg, one standard deviation.
    angles, values = angular_profile("FilledCone", 11.0)
    assert angles[1] == pytest.approx(11.0)
    assert values.tolist() == [1.0, 1.0, 0.0]
    angles, values = angular_profile("Lambertian", 60.0)
    assert angles[-1] == pytest.approx(60.0)
    assert values[-1] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="positive width"):
        angular_profile("Gauss", 0.0)
    with pytest.raises(ValueError, match="no tabulated profile"):
        angular_profile("Rayleigh", 5.0)


def test_pulse_profile():
    times, rates = pulse_profile("TopHat", 2.5, 0.0)
    assert times.tolist() == [-1.25, 1.25]
    assert rates.tolist() == [1.0, 1.0]
    times, rates = pulse_profile("Gauss", 1.5, 0.0)
    assert np.average(times, weights=rates) == pytest.approx(0.0, abs=1e-12)
    assert math.sqrt(np.average(times**2, weights=rates)) == pytest.approx(1.5, rel=1e-3)
    assert pulse_profile("Gauss", 0.0, 0.0) is None
    times, rates = pulse_profile("Exponential", 0.0, 2.0)
    assert (times[0], rates[0]) == pytest.approx((0.0, 1.0))
    # Mean of the density, linear between the nodes.
    mean = np.trapezoid(times * rates, times) / np.trapezoid(rates, times)
    assert mean == pytest.approx(2.0, rel=1e-2)
    assert pulse_profile("Exponential", 0.0, 0.0) is None
    times, rates = pulse_profile("Gauss-Exponential", 1.59, 0.4)
    assert times[np.argmax(rates)] == pytest.approx(0.0)
    assert rates.max() == pytest.approx(1.0)
    assert np.all(rates >= 0.0)
    assert np.all(np.diff(times) > 0.0)
    above = times[rates >= 0.1]
    assert above[0] < 0.0 < above[-1]
    with pytest.raises(ValueError, match="rise and fall"):
        pulse_profile("Gauss-Exponential", 1.0, 0.0)
    with pytest.raises(ValueError, match="no tabulated profile"):
        pulse_profile("Lorentz", 1.0, 0.0)
