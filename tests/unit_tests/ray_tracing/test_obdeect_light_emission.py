#!/usr/bin/python3

import math

import numpy as np
import pytest

from simtools.ray_tracing import obdeect_light_emission as light


def test_angular_profile(tmp_test_directory):
    angles, values = light.angular_profile("isotropic")
    assert angles.tolist() == [0.0, 180.0]
    assert values.tolist() == [1.0, 1.0]
    angles, values = light.angular_profile("gauss:2.0")
    assert angles[1] == pytest.approx(0.1)
    assert angles[-1] == pytest.approx(9.9)
    assert values[20] == pytest.approx(math.exp(-0.5))  # At 2 deg, one rms.
    angles, values = light.angular_profile("filledcone:11.0")
    assert angles[1] == pytest.approx(11.0)
    assert values.tolist() == [1.0, 1.0, 0.0]
    table = tmp_test_directory / "angular.dat"
    table.write_text("# angle intensity\n0 1.0\n10 0.5\n20 0.0\n", encoding="utf-8")
    angles, values = light.angular_profile(str(table))
    assert angles.tolist() == [0.0, 10.0, 20.0]
    assert values.tolist() == [1.0, 0.5, 0.0]
    with pytest.raises(ValueError, match="not supported"):
        light.angular_profile("cone:5")


def test_pulse_profile(tmp_test_directory):
    times, rates = light.pulse_profile("simple:2.5")
    assert times.tolist() == [-1.25, 1.25]
    assert rates.tolist() == [1.0, 1.0]
    times, rates = light.pulse_profile("gauss:1.5")
    assert np.average(times, weights=rates) == pytest.approx(0.0, abs=1e-12)
    assert math.sqrt(np.average(times**2, weights=rates)) == pytest.approx(1.5, rel=1e-3)
    times, rates = light.pulse_profile("exponential:2.0")
    assert times[0] == pytest.approx(-2.0)
    assert rates[0] == pytest.approx(1.0)
    assert light.pulse_profile("gauss:0.0") is None
    table = tmp_test_directory / "pulse.dat"
    table.write_text("0 0\n1 1\n2 0\n", encoding="utf-8")
    assert light.pulse_profile(str(table))[0].tolist() == [0.0, 1.0, 2.0]
    with pytest.raises(ValueError, match="not supported"):
        light.pulse_profile("lorentz:1")


def test_atmospheric_transmission(tmp_test_directory):
    table = tmp_test_directory / "atm_trans.dat"
    table.write_text(
        "# Optical depth between altitudes\n"
        "# H2= 2.000, H1= 3.000 12.000\n"
        "300 0.10 0.40\n"
        "301 0.20 0.80\n",
        encoding="utf-8",
    )
    atmosphere = light.AtmosphericTransmission(table)
    assert atmosphere.observation_level_m == pytest.approx(2000.0)
    assert atmosphere.optical_depth(300.0, 2000.0) == pytest.approx(0.0)
    assert atmosphere.optical_depth(300.4, 3000.0) == pytest.approx(0.10)  # Integer wavelength.
    # Linear in log10(altitude) between the levels, clamped beyond them.
    middle = 10 ** ((math.log10(3000.0) + math.log10(12000.0)) / 2)
    assert atmosphere.optical_depth(301.0, middle) == pytest.approx(0.50)
    assert atmosphere.optical_depth(900.0, 50000.0) == pytest.approx(0.80)
    # The depth between emitter and telescope, scaled by the airmass.
    transmission = atmosphere.transmission(300.0, 3000.0, 2000.0, 2.0)
    assert transmission == pytest.approx(math.exp(-0.2))
    assert atmosphere.transmission(300.0, 1999.0, 2000.0, 1.0) == pytest.approx(0.0)


def test_ground_to_telescope_follows_the_star_geometry():
    """Telescope pointing as in the validated star-light runs (obdeect)."""
    pytest.importorskip("obdeect")
    # pylint: disable=import-outside-toplevel
    from simtools.ray_tracing.backends.obdeect import star_direction
    from simtools.simtel.simulator_ray_tracing import telescope_pointing

    zenith = math.radians(20.0)
    towards_star = np.array([math.sin(zenith), 0.0, math.cos(zenith)])
    theta, phi = telescope_pointing(20.0, 1.0, 0.5)
    direction = light.ground_direction_to_telescope(towards_star, theta, phi)
    assert direction == pytest.approx(star_direction(20.0, 1.0, 0.5), abs=1e-15)
    # Points rotate like directions; the axes offsets shift the frame.
    assert light.ground_to_telescope(towards_star, theta, phi) == pytest.approx(direction)
    vertical = light.ground_to_telescope([0.0, 0.0, 5.0], 0.0, 0.0, [1.0, 0.25])
    assert vertical == pytest.approx([-0.75, 0.0, 5.0])


def test_camera_bunches_turn_the_frame_about_x():
    arrivals = {
        "position_m": np.array([[0.1, 0.2, 16.0]]),
        "direction": np.array([[0.01, 0.02, 0.99975]]),
        "time_ns": np.array([55.0]),
        "weight": np.array([0.7]),
    }
    bunch = light.camera_bunches(arrivals, np.array([385.0]), 2147.0)[0]
    assert (bunch["x"], bunch["y"]) == pytest.approx((10.0, -20.0))
    assert (bunch["cx"], bunch["cy"]) == pytest.approx((0.01, -0.02))
    assert bunch["zem"] == pytest.approx(214700.0)
    assert bunch["photons"] == pytest.approx(0.7)
    assert bunch["wavelength"] == pytest.approx(385.0)


def test_light_source_events(mocker):
    pytest.importorskip("obdeect")
    model = mocker.Mock()
    model.emitter_photons.side_effect = lambda photons, **_: {"wavelength_nm": np.full(3, 385.0)}
    model.trace.return_value = {
        "status": np.array([0, 4, 0], dtype=np.uint8),  # detected, blocked, detected
        "position_m": np.zeros((3, 3)),
        "direction": np.tile([0.0, 0.0, 1.0], (3, 1)),
        "time_ns": np.zeros(3),
        "weight": np.array([1.0, 0.0, 0.0]),
    }

    events = list(
        light.light_source_events(
            model,
            [1e6, 2e6],
            seed=5,
            position_m=(0.0, 0.0, -0.5),
            profile=light.angular_profile("isotropic"),
            pulse=None,
            wavelength_nm=385.0,
            emission_altitude_m=2163.5,
            direct=True,
            transmission=0.5,
        )
    )

    assert len(events) == 2
    assert len(events[0]) == 1
    assert events[0]["photons"][0] == pytest.approx(0.5)
    first, second = model.emitter_photons.call_args_list
    assert first.args == (1e6,)
    assert first.kwargs["first_id"] == 0
    assert second.kwargs["first_id"] == 1 << 40
    assert model.trace.call_args.kwargs == {"seed": 5, "direct": True}
