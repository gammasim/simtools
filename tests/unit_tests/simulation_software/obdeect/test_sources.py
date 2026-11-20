"""Tests for the commands of the obdeect light sources."""

import json

import numpy as np
import pytest

from simtools.simulation_software import light_sources
from simtools.simulation_software.light_sources import (
    AngularDistribution,
    FlasherRequest,
    IlluminatorRequest,
    PulseShape,
)
from simtools.simulation_software.obdeect import sources
from simtools.simulation_software.programs import program_command


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
        "simtools.simulation_software.flasher_stream",
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
    angular = tmp_path / "light_source_angular_distribution_x.dat"
    pulse = tmp_path / "light_source_pulse_shape_x.dat"
    assert tokens == [
        *program_command("obdeect", "obdeect"),
        "sources",
        "run",
        str(tmp_path / "light_source_run_x.json"),
        "-o",
        str(tmp_path / "i.pst"),
    ]
    description = json.loads((tmp_path / "light_source_run_x.json").read_text())
    emission = description["events"][0]["emissions"][0]
    assert emission["position_m"] == [110, 0, 5]
    assert emission["target"]["normal"] == [-1, 0, 0]
    assert emission["target"]["radius_m"] > 12
    assert description["telescopes"][0]["position_m"] == [10, 0, 5]
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
    emission = json.loads((tmp_path / "light_source_run.json").read_text())["events"][0][
        "emissions"
    ][0]
    assert emission["angular"] == {"kind": "filled_cone", "angle_deg": 2}
    assert emission["pulse"] == {"kind": "exponential", "decay_ns": 3, "mean_ns": 3}
    assert "-o" not in tokens
    assert not (tmp_path / "light_source_angular_distribution.dat").exists()


def test_illuminator_tokens_without_tables(tmp_path):
    assert (
        sources.illuminator_angular_token(AngularDistribution("isotropic", 0.0), tmp_path)
        == "isotropic"
    )
    assert (
        sources.illuminator_angular_token(AngularDistribution("Gauss", 3.0), tmp_path)
        == "gauss:sigma_deg=3"
    )
    assert sources.illuminator_pulse_token(PulseShape("TopHat", 2.0, 0.0), tmp_path) == (
        "tophat:width_ns=2"
    )
    assert sources.illuminator_pulse_token(PulseShape("Gauss", 0.0, 0.0), tmp_path) == (
        "fixed:time_ns=0"
    )
    # An instantaneous pulse needs no table; an unknown shape with a width has none.
    assert (
        sources.illuminator_pulse_token(PulseShape("Triangle", 0.0, 0.0), tmp_path)
        == "fixed:time_ns=0"
    )
    with pytest.raises(ValueError, match="has no tabulated profile"):
        sources.illuminator_pulse_token(PulseShape("Triangle", 1.0, 0.0), tmp_path)


@pytest.mark.parametrize(
    ("position", "angular", "target"),
    [
        ((10.0, 0.0, 5.0), AngularDistribution("isotropic", 0), "none"),
        ((110.0, 0.0, 5.0), AngularDistribution("isotropic", 0), "sphere"),
        ((110.0, 0.0, 5.0), AngularDistribution("Gauss", 3), "disc"),
    ],
)
def test_illuminator_target_geometry(tmp_path, position, angular, target):
    request = _illuminator_request(position_m=position, angular=angular, photons=(100, 200))
    sources.illuminator_command(request, tmp_path, executable="obdeect")
    description = json.loads((tmp_path / "light_source_run.json").read_text())
    emissions = [event["emissions"][0] for event in description["events"]]
    assert [emission["photons"] for emission in emissions] == [100, 200]
    actual = emissions[0]["target"]
    assert (actual if isinstance(actual, str) else actual["kind"]) == target


def test_illuminator_command_executable_override(tmp_path):
    tokens = sources.illuminator_command(_illuminator_request(), tmp_path, executable="my-source")
    assert tokens[0] == "my-source"
    assert tokens[1:3] == ["sources", "run"]


def test_flasher_command_passes_simulation_models_path(tmp_path):
    """The flasher subprocess reads the telescope model from the models path of the run."""
    command = sources.flasher_command(
        _flasher_request(),
        tmp_path,
        output=tmp_path / "out.stream",
        executable="python",
        simulation_models_path="/models",
    )

    index = command.index("--simulation-models-path")
    assert command[index + 1] == "/models"
    assert command.index("--output-path") < index


def test_flasher_command_without_models_path_has_no_option(tmp_path):
    command = sources.flasher_command(_flasher_request(), tmp_path, executable="python")

    assert "--simulation-models-path" not in command


def test_source_command_passes_models_path_to_flasher_only(tmp_path):
    flasher = light_sources.source_command(
        "flasher",
        _flasher_request(),
        tmp_path,
        executable="python",
        simulation_models_path="/models",
    )

    assert flasher[flasher.index("--simulation-models-path") + 1] == "/models"
