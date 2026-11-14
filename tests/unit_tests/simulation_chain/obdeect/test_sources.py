"""Tests for the commands of the obdeect light sources."""

import numpy as np
import pytest

from simtools.simulation_chain import light_sources
from simtools.simulation_chain.light_sources import (
    AngularDistribution,
    FlasherRequest,
    IlluminatorRequest,
    PulseShape,
)
from simtools.simulation_chain.obdeect import sources
from simtools.simulation_chain.programs import program_command


def _flasher_request(**options):
    return FlasherRequest(
        **{
            "position_m": (0.02, 0.0, -0.5),
            "camera_radius_m": 1.2,
            "events": (2, 1),
            "photons": (1000, 2000),
            "wavelength_nm": 355.0,
            "angular": AngularDistribution("FilledCone", 2.5),
            "pulse": PulseShape("Gauss", 1.5, 0.0),
            "observation_level_m": 2158.0,
            "telescope": "LSTN-01",
            "site": "North",
            "model_version": "6.0.2",
            "run_number": 4,
            "seed": 9,
            "instrument_seed": 3,
        }
        | options
    )


def test_flasher_command(tmp_path):
    tokens = sources.flasher_command(
        _flasher_request(), tmp_path, output=tmp_path / "f.pst", file_token="_x", executable="py"
    )
    assert tokens == [
        "py",
        "-m",
        "simtools.simulation_chain.flasher_stream",
        *("--request", str(tmp_path / "light_source_request_x.json")),
        *("--output-path", str(tmp_path)),
        *("--output", str(tmp_path / "f.pst")),
    ]
    assert light_sources.flasher_request_from_json(tokens[4]) == _flasher_request()
    assert "--output" not in sources.flasher_command(_flasher_request(), tmp_path)


def _illuminator_request(**options):
    return IlluminatorRequest(
        **{
            "position_m": (110.0, 0.0, 5.0),
            "pointing": (-1.0, 0.0, 0.0),
            "telescope_position_m": (10.0, 0.0, 5.0),
            "telescope_radius_m": 12.0,
            "telescope_pointing_deg": (90.0, 0.0),
            "events": 2,
            "photons": (1_000_000,),
            "bunch_size": 1.0,
            "wavelength_nm": 355.0,
            "angular": AngularDistribution("Lambertian", 20.0),
            "pulse": PulseShape("Gauss-Exponential", 0.4, 1.59),
            "observation_level_m": 2158.0,
            "run_number": 3,
            "seed": 8,
        }
        | options
    )


def test_illuminator_command_and_tables(tmp_path):
    tokens = sources.illuminator_command(
        _illuminator_request(), tmp_path, output=tmp_path / "i.pst", file_token="_x"
    )
    positions = tmp_path / "light_source_telescope_position_x.dat"
    angular = tmp_path / "light_source_angular_distribution_x.dat"
    pulse = tmp_path / "light_source_pulse_shape_x.dat"
    assert tokens == [
        *program_command("obdeect", "obdeect-sources-point-source"),
        *("-R", "3", "-N", "2"),
        *("-x", "11000", "-y", "0", "-z", "500", "-d", "-1,0,0"),
        *("-n", "1000000", "-b", "1", "-s", "355"),
        *("-p", str(pulse), "-a", str(angular), "-h", "2158"),
        *("--telpos-file", str(positions), "--seed", "8", "-o", str(tmp_path / "i.pst")),
    ]
    assert positions.read_text(encoding="utf-8").split() == ["1000", "0", "500", "1200"]
    angles, intensity = np.loadtxt(angular, unpack=True)
    assert angles[0] == pytest.approx(0.0)
    assert angles[-1] == pytest.approx(20.0)
    assert intensity[0] == pytest.approx(1.0)
    assert np.all(np.diff(intensity) < 0.0)
    times, rates = np.loadtxt(pulse, unpack=True)
    assert np.all(np.diff(times) > 0.0)
    assert rates.max() > 0.0
    assert np.all(rates >= 0.0)


def test_illuminator_command_with_program_tokens(tmp_path):
    request = _illuminator_request(
        angular=AngularDistribution("FilledCone", 2.0), pulse=PulseShape("Exponential", 0.0, 3.0)
    )
    tokens = sources.illuminator_command(request, tmp_path)
    assert tokens[tokens.index("-a") + 1] == "FilledCone:2"
    assert tokens[tokens.index("-p") + 1] == "Exp:3"
    assert "-o" not in tokens
    assert not (tmp_path / "light_source_angular_distribution.dat").exists()


def test_illuminator_tokens_without_tables(tmp_path):
    assert (
        sources.illuminator_angular_token(AngularDistribution("isotropic", 0.0), tmp_path)
        == "isotropic"
    )
    assert (
        sources.illuminator_angular_token(AngularDistribution("Gauss", 3.0), tmp_path) == "Gauss:3"
    )
    assert sources.illuminator_pulse_token(PulseShape("TopHat", 2.0, 0.0), tmp_path) == ("Simple:2")
    assert sources.illuminator_pulse_token(PulseShape("Gauss", 0.0, 0.0), tmp_path) == ("Gauss:0")
    # An instantaneous pulse needs no table; an unknown shape with a width has none.
    assert sources.illuminator_pulse_token(PulseShape("Triangle", 0.0, 0.0), tmp_path) == "Simple:0"
    with pytest.raises(ValueError, match="has no tabulated profile"):
        sources.illuminator_pulse_token(PulseShape("Triangle", 1.0, 0.0), tmp_path)


def test_illuminator_command_executable_override(tmp_path):
    tokens = sources.illuminator_command(_illuminator_request(), tmp_path, executable="my-source")
    assert tokens[0] == "my-source"
    assert tokens[1] == "-R"
