import astropy.units as u
import pytest

from simtools.simtel.simulator_obdeect import SimulatorObdeect


def test_obdeect_command_requires_model_derived_scene(mocker, tmp_test_directory):
    telescope = mocker.Mock(name="MSTN-01", label="test")
    telescope.name = "MSTN-01"
    output_file = tmp_test_directory / "arrivals.csv"
    mocker.patch(
        "simtools.simtel.simulator_obdeect.settings.config",
        mocker.Mock(obdeect_exe=output_file.parent / "obdeect"),
    )
    with pytest.raises(ValueError, match="obdeect_scene_file is required"):
        SimulatorObdeect(
            telescope_model=telescope,
            config_data={"source_distance": 12 * u.km, "number_of_photons": 32},
            output_file=output_file,
        )


def test_obdeect_command_accepts_provenance_bound_scene_file(mocker, tmp_test_directory):
    telescope = mocker.Mock(name="MSTN-01", label="test")
    telescope.name = "MSTN-01"
    output_file = tmp_test_directory / "arrivals.csv"
    scene_file = tmp_test_directory / "MSTN.scene"
    scene_file.write_text("obdeect-scene-v1\n")
    mocker.patch(
        "simtools.simtel.simulator_obdeect.settings.config",
        mocker.Mock(obdeect_exe=output_file.parent / "obdeect"),
    )
    simulator = SimulatorObdeect(
        telescope_model=telescope,
        config_data={"obdeect_scene_file": scene_file, "number_of_photons": 4},
        output_file=output_file,
    )

    command = simulator.make_run_command()

    assert command[:3] == [str(output_file.parent / "obdeect"), "--scene-file", str(scene_file)]
