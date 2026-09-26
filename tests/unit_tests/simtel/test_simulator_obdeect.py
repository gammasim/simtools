import astropy.units as u

from simtools.simtel.simulator_obdeect import SimulatorObdeect


def test_obdeect_command_uses_packaged_executable(mocker, tmp_test_directory):
    telescope = mocker.Mock(name="MSTN-01", label="test")
    telescope.name = "MSTN-01"
    output_file = tmp_test_directory / "arrivals.csv"
    mocker.patch(
        "simtools.simtel.simulator_obdeect.settings.config",
        mocker.Mock(obdeect_exe=output_file.parent / "obdeect"),
    )
    simulator = SimulatorObdeect(
        telescope_model=telescope,
        config_data={
            "off_axis_x": 1.5,
            "off_axis_y": -0.5,
            "source_distance": 12 * u.km,
            "number_of_photons": 32,
        },
        output_file=output_file,
    )

    command = simulator.make_run_command()

    assert command[:5] == [
        str(output_file.parent / "obdeect"),
        "--telescope",
        "MST",
        "--source",
        "star",
    ]
    assert "--photons" in command
    assert command[command.index("--photons") + 1] == "32"
    assert command[command.index("--distance-m") + 1] == "12000.0"


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
    assert "--telescope" not in command
