from pathlib import Path

import astropy.units as u
import pytest

from simtools.simtel.simulator_obdeect import SimulatorObdeect


def test_obdeect_command_requires_model_derived_optical_model(mocker, tmp_test_directory):
    telescope = mocker.Mock(name="MSTN-01", label="test")
    telescope.name = "MSTN-01"
    output_file = Path(tmp_test_directory) / "arrivals.csv"
    mocker.patch(
        "simtools.simtel.simulator_obdeect.settings.config",
        mocker.Mock(obdeect_exe=output_file.parent / "obdeect"),
    )
    with pytest.raises(ValueError, match="obdeect_optical_model_file is required"):
        SimulatorObdeect(
            telescope_model=telescope,
            config_data={"source_distance": 12 * u.km, "number_of_photons": 32},
            output_file=output_file,
        )


def test_obdeect_command_accepts_provenance_bound_optical_model_file(mocker, tmp_test_directory):
    telescope = mocker.Mock(name="MSTN-01", label="test")
    telescope.name = "MSTN-01"
    output_file = Path(tmp_test_directory) / "arrivals.csv"
    optical_model_file = Path(tmp_test_directory) / "MSTN.optical_model"
    optical_model_file.write_text("obdeect.compiled-optical-model.v1\n")
    mocker.patch(
        "simtools.simtel.simulator_obdeect.settings.config",
        mocker.Mock(obdeect_exe=output_file.parent / "obdeect"),
    )
    simulator = SimulatorObdeect(
        telescope_model=telescope,
        config_data={"obdeect_optical_model_file": optical_model_file, "number_of_photons": 4},
        output_file=output_file,
    )

    command = simulator.make_run_command()

    assert command[:3] == [
        str(output_file.parent / "obdeect"),
        "--optical-model",
        str(optical_model_file),
    ]


@pytest.mark.parametrize(
    ("config", "output", "error", "message"),
    [
        ({"number_of_photons": 0}, True, ValueError, "number_of_photons"),
        ({}, False, ValueError, "output_file"),
        ({"obdeect_optical_model_file": "absent"}, True, FileNotFoundError, "does not exist"),
        ({"single_mirror_mode": True}, True, ValueError, "single_mirror_mode"),
    ],
)
def test_obdeect_rejects_invalid_configuration(tmp_test_directory, config, output, error, message):
    """Reject unsupported studies and missing inputs before starting transport."""
    root = Path(tmp_test_directory)
    model = root / "model.json"
    model.write_text("{}")
    with pytest.raises(error, match=message):
        SimulatorObdeect(
            telescope_model=None,
            config_data={"obdeect_optical_model_file": model} | config,
            output_file=root / "arrivals.csv" if output else None,
        )


def test_obdeect_rejects_empty_native_output(mocker, tmp_test_directory):
    """A successful process exit without arrivals does not qualify as a result."""
    root = Path(tmp_test_directory)
    model = root / "model.json"
    model.write_text("{}")
    output = root / "arrivals.csv"
    simulator = SimulatorObdeect(
        telescope_model=None,
        config_data={"obdeect_optical_model_file": model},
        output_file=output,
    )
    mocker.patch.object(simulator, "make_run_command", return_value=["obdeect-simtools-raytrace"])
    mocker.patch(
        "simtools.simtel.simulator_obdeect.subprocess.run",
        side_effect=lambda *_args, **_kwargs: output.touch(),
    )
    with pytest.raises(RuntimeError, match="did not write an arrival file"):
        simulator.run()
