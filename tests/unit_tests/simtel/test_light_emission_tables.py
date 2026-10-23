#!/usr/bin/python3

import math

import numpy as np
import pytest

from simtools.simtel.light_emission_tables import angular_profile, pulse_profile


def test_angular_profile(tmp_test_directory):
    angles, values = angular_profile("isotropic")
    assert angles.tolist() == [0.0, 180.0]
    assert values.tolist() == [1.0, 1.0]
    angles, values = angular_profile("gauss:2.0")
    assert angles[1] == pytest.approx(0.1)
    assert angles[-1] == pytest.approx(9.9)
    assert values[20] == pytest.approx(math.exp(-0.5))  # At 2 deg, one rms.
    angles, values = angular_profile("filledcone:11.0")
    assert angles[1] == pytest.approx(11.0)
    assert values.tolist() == [1.0, 1.0, 0.0]
    table = tmp_test_directory / "angular.dat"
    table.write_text("# angle intensity\n0 1.0\n10 0.5\n20 0.0\n", encoding="utf-8")
    angles, values = angular_profile(str(table))
    assert angles.tolist() == [0.0, 10.0, 20.0]
    assert values.tolist() == [1.0, 0.5, 0.0]
    with pytest.raises(ValueError, match="not supported"):
        angular_profile("cone:5")


def test_pulse_profile(tmp_test_directory):
    times, rates = pulse_profile("simple:2.5")
    assert times.tolist() == [-1.25, 1.25]
    assert rates.tolist() == [1.0, 1.0]
    times, rates = pulse_profile("gauss:1.5")
    assert np.average(times, weights=rates) == pytest.approx(0.0, abs=1e-12)
    assert math.sqrt(np.average(times**2, weights=rates)) == pytest.approx(1.5, rel=1e-3)
    times, rates = pulse_profile("exponential:2.0")
    assert times[0] == pytest.approx(-2.0)
    assert rates[0] == pytest.approx(1.0)
    assert pulse_profile("gauss:0.0") is None
    table = tmp_test_directory / "pulse.dat"
    table.write_text("0 0\n1 1\n2 0\n", encoding="utf-8")
    assert pulse_profile(str(table))[0].tolist() == [0.0, 1.0, 2.0]
    with pytest.raises(ValueError, match="not supported"):
        pulse_profile("lorentz:1")
