#!/usr/bin/python3

import copy
import gzip
import logging
import math
import shutil
import tarfile
from pathlib import Path

import pytest
from astropy import units as u

from simtools.model.array_model import ArrayModel
from simtools.runners.corsika_runner import CorsikaRunner
from simtools.runners.corsika_simtel_runner import CorsikaSimtelRunner
from simtools.simtel.simulator_array import SimulatorArray
from simtools.simulator import InvalidRunsToSimulateError, Simulator

logger = logging.getLogger()

CORSIKA_CONFIG_MOCK_PATCH = "simtools.simulator.CorsikaConfig"
INITIALIZE_RUN_LIST_ERROR_MSG = (
    "Error in initializing run list "
    "(missing 'run_number', 'run_number_offset' or 'number_of_runs')."
)


@pytest.fixture
def input_file_list():
    return ["run1", "abc_run22", "def_run02_and"]


@pytest.fixture
def corsika_file():
    return "run1_proton_za20deg_azm0deg_North_1LST_test.corsika.zst"


@pytest.fixture
def simulations_args_dict(corsika_config_data, model_version, simtel_path):
    """Return a dictionary with the simulation command line arguments."""
    args_dict = copy.deepcopy(corsika_config_data)
    args_dict["simulation_software"] = "sim_telarray"
    args_dict["simtel_path"] = simtel_path
    args_dict["model_version"] = model_version
    args_dict["label"] = "test-array-simulator"
    args_dict["array_layout_name"] = "test_layout"
    args_dict["site"] = "North"
    args_dict["keep_seeds"] = False
    args_dict["run_number"] = 1
    args_dict["run_number_offset"] = 0
    args_dict["nshow"] = 10
    args_dict["extra_commands"] = None
    return args_dict


@pytest.fixture
def array_simulator(io_handler, db_config, simulations_args_dict):
    args_dict = copy.deepcopy(simulations_args_dict)
    args_dict["simulation_software"] = "sim_telarray"

    return Simulator(
        label=args_dict["label"],
        args_dict=args_dict,
        db_config=db_config,
    )


@pytest.fixture
def shower_simulator(io_handler, db_config, simulations_args_dict):
    args_dict = copy.deepcopy(simulations_args_dict)
    args_dict["simulation_software"] = "corsika"
    args_dict["label"] = "test-shower-simulator"

    return Simulator(
        label=args_dict["label"],
        args_dict=args_dict,
        db_config=db_config,
    )


@pytest.fixture
def shower_array_simulator(io_handler, db_config, simulations_args_dict):
    args_dict = copy.deepcopy(simulations_args_dict)
    args_dict["simulation_software"] = "corsika_sim_telarray"
    args_dict["label"] = "test-shower-array-simulator"
    args_dict["sequential"] = True
    return Simulator(
        label=args_dict["label"],
        args_dict=args_dict,
        db_config=db_config,
    )


@pytest.fixture
def calibration_simulator(io_handler, db_config, simulations_args_dict):
    args_dict = copy.deepcopy(simulations_args_dict)
    args_dict["simulation_software"] = "corsika_sim_telarray"
    args_dict["label"] = "test-calibration-shower-array-simulator"
    args_dict["sequential"] = True
    args_dict["run_mode"] = "nsb_only_pedestals"
    return Simulator(
        label=args_dict["label"],
        args_dict=args_dict,
        db_config=db_config,
    )


def test_init_simulator(shower_simulator, array_simulator, shower_array_simulator):
    assert isinstance(shower_simulator._simulation_runner, CorsikaRunner)
    assert isinstance(shower_array_simulator._simulation_runner, CorsikaSimtelRunner)
    assert isinstance(array_simulator._simulation_runner, SimulatorArray)


def test_simulation_software(array_simulator, shower_simulator, shower_array_simulator):
    assert array_simulator.simulation_software == "sim_telarray"
    assert shower_simulator.simulation_software == "corsika"
    assert shower_array_simulator.simulation_software == "corsika_sim_telarray"

    # setting
    test_array_simulator = copy.deepcopy(array_simulator)
    test_array_simulator.simulation_software = "corsika"
    assert test_array_simulator.simulation_software == "corsika"

    with pytest.raises(
        ValueError, match="Invalid simulation software: this_simulator_is_not_there"
    ):
        test_array_simulator.simulation_software = "this_simulator_is_not_there"


def test_initialize_run_list(shower_simulator):
    assert shower_simulator._initialize_run_list() == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def test_initialize_run_list_valid_cases(shower_simulator):
    # Test case where number_of_runs <= 1
    shower_simulator.args_dict["number_of_runs"] = 1
    shower_simulator.args_dict["run_number"] = 5
    shower_simulator.args_dict["run_number_offset"] = 10
    result = shower_simulator._initialize_run_list()
    assert result == [15]  # run_number_offset + run_number

    # Test case where number_of_runs > 1
    shower_simulator.args_dict["number_of_runs"] = 3
    shower_simulator.args_dict["run_number"] = 5
    shower_simulator.args_dict["run_number_offset"] = 10
    result = shower_simulator._initialize_run_list()
    assert result == [15, 16, 17]  # range from 15 to 17


def test_prepare_run_list_and_range(shower_simulator, shower_array_simulator):
    for simulator_now in [shower_simulator, shower_array_simulator]:
        assert not simulator_now._prepare_run_list_and_range(None, None)

        run_list = [1, 24, 3]

        assert simulator_now._prepare_run_list_and_range(run_list=run_list, run_range=None) == [
            1,
            3,
            24,
        ]

        with pytest.raises(InvalidRunsToSimulateError):
            simulator_now._prepare_run_list_and_range(run_list=[1, "a", 4], run_range=None)

        assert simulator_now._prepare_run_list_and_range(run_list=None, run_range=[3, 6]) == [
            3,
            4,
            5,
        ]

        assert simulator_now._prepare_run_list_and_range(run_list=None, run_range=[6, 3]) == []

        with pytest.raises(InvalidRunsToSimulateError):
            simulator_now._prepare_run_list_and_range(run_list=None, run_range=[3, "b"])

        with pytest.raises(InvalidRunsToSimulateError):
            simulator_now._prepare_run_list_and_range(run_list=None, run_range=[3, 4, 5])

        assert simulator_now._prepare_run_list_and_range(run_list=5, run_range=None) == [5]


def test_fill_results_without_run(array_simulator, input_file_list):
    array_simulator._fill_results_without_run(input_file_list=[])
    assert isinstance(array_simulator.runs, list)

    array_simulator.runs = []
    array_simulator._fill_results_without_run(input_file_list=input_file_list)
    assert array_simulator.runs == [1, 22, 2]


def test_simulate_shower_simulator(shower_simulator):
    shower_simulator._test = True
    shower_simulator.simulate()
    assert len(shower_simulator._results["simtel_output"]) > 0
    assert len(shower_simulator._results["sub_out"]) > 0
    run_script = shower_simulator._simulation_runner.prepare_run_script(run_number=2)
    assert Path(run_script).exists()


def test_simulate_array_simulator(array_simulator, corsika_file):
    array_simulator._test = True
    array_simulator.simulate(input_file_list=corsika_file)

    assert len(array_simulator._results["simtel_output"]) > 0
    assert len(array_simulator._results["sub_out"]) > 0


def test_simulate_shower_array_simulator(shower_array_simulator):
    shower_array_simulator._test = True
    shower_array_simulator.simulate()

    assert len(shower_array_simulator._results["simtel_output"]) > 0
    assert len(shower_array_simulator._results["sub_out"]) > 0


def test_get_runs_and_files_to_submit(
    array_simulator, shower_simulator, shower_array_simulator, input_file_list
):
    with pytest.raises(ValueError, match=r"No runs to submit."):
        array_simulator._get_runs_and_files_to_submit(input_file_list=None)

    assert array_simulator._get_runs_and_files_to_submit(input_file_list=input_file_list) == {
        1: "run1",
        2: "def_run02_and",
        22: "abc_run22",
    }

    for simulator_now in [shower_simulator, shower_array_simulator]:
        assert simulator_now._get_runs_and_files_to_submit(input_file_list=None) == {
            1: None,
            2: None,
            3: None,
            4: None,
            5: None,
            6: None,
            7: None,
            8: None,
            9: None,
            10: None,
        }


def test_guess_run_from_file(array_simulator, caplog):
    assert array_simulator._guess_run_from_file("run12345_bla_ble") == 12345

    assert array_simulator._guess_run_from_file("run5test2_bla_ble") == 5

    assert array_simulator._guess_run_from_file("abc-run12345_bla_ble") == 12345

    assert array_simulator._guess_run_from_file("run10") == 10

    # Invalid run number - returns 1
    with caplog.at_level(logging.WARNING):
        assert array_simulator._guess_run_from_file("abc-ran12345_bla_ble") == 1
    assert "Run number could not be guessed from abc-ran12345_bla_ble using run = 1" in caplog.text


def test_fill_results(array_simulator, shower_simulator, shower_array_simulator, input_file_list):
    for simulator_now in [array_simulator, shower_array_simulator]:
        for run_number in [1, 2, 22]:
            simulator_now._fill_results(input_file_list[1], run_number=run_number)
        assert len(simulator_now.get_file_list("simtel_output")) == 3
        assert len(simulator_now._results["sub_out"]) == 3
        assert len(simulator_now.get_file_list("log")) == 3
        assert len(simulator_now.get_file_list("input")) == 3
        assert len(simulator_now.get_file_list("hist")) == 3
        logger.error(simulator_now.get_file_list("input"))
        assert simulator_now.get_file_list("input")[1] == "abc_run22"

    shower_simulator._fill_results(input_file_list[1], run_number=5)
    assert len(shower_simulator.get_file_list("simtel_output")) == 1
    assert len(shower_simulator.get_file_list("corsika_log")) == 1
    assert len(shower_simulator.get_file_list("hist")) == 0


def test_get_list_of_files(shower_simulator):
    test_shower_simulator = copy.deepcopy(shower_simulator)
    test_shower_simulator._results["simtel_output"] = ["file_name"] * 10
    assert len(test_shower_simulator.get_file_list("simtel_output")) == len(shower_simulator.runs)
    assert len(test_shower_simulator.get_file_list("not_a_valid_file_type")) == 0


def test_resources(shower_simulator, capsys):
    shower_simulator.resources(input_file_list=None)
    print_text = capsys.readouterr()
    assert "Wall time/run [sec] = nan" in print_text.out


def test_make_resources_report(shower_simulator):
    _resources_1 = shower_simulator._make_resources_report(input_file_list=None)
    assert math.isnan(_resources_1["Wall time/run [sec]"])

    # Copying the corsika log file to the expected location in the test directory.
    log_file_name = "log_sub_corsika_run000001_gamma_North_test_layout_test-production.out"
    shutil.copy(
        f"tests/resources/{log_file_name}",
        shower_simulator._simulation_runner.get_file_name(
            file_type="sub_log",
            run_number=1,
            mode="out",
        ),
    )
    test_shower_simulator = copy.deepcopy(shower_simulator)
    test_shower_simulator.runs = [1]
    _resources_1 = test_shower_simulator._make_resources_report(input_file_list=log_file_name)
    assert _resources_1["Wall time/run [sec]"] == 6

    test_shower_simulator.runs = [4]
    with pytest.raises(FileNotFoundError):
        test_shower_simulator._make_resources_report(input_file_list)


def test_get_runs_to_simulate(shower_simulator):
    assert len(shower_simulator.runs) == len(
        shower_simulator._get_runs_to_simulate(run_list=None, run_range=None)
    )

    assert 3 == len(shower_simulator._get_runs_to_simulate(run_list=[2, 5, 7]))

    assert 3 == len(shower_simulator._get_runs_to_simulate(run_range=[1, 4]))

    shower_simulator.runs = None
    assert isinstance(shower_simulator._get_runs_to_simulate(), list)


def test_save_file_lists(shower_simulator, mocker, caplog):
    with caplog.at_level(logging.DEBUG):
        shower_simulator.save_file_lists()
    assert "No files to save for simtel_output files." in caplog.text

    mock_shower_simulator = copy.deepcopy(shower_simulator)
    mocker.patch.object(mock_shower_simulator, "get_file_list", return_value=["file1", "file2"])

    with caplog.at_level(logging.INFO):
        mock_shower_simulator.save_file_lists()
    assert "Saving list of simtel_output files to" in caplog.text

    mocker.patch.object(mock_shower_simulator, "get_file_list", return_value=[None, None])
    with caplog.at_level(logging.DEBUG):
        mock_shower_simulator.save_file_lists()
    assert "No files to save for simtel_output files." in caplog.text


def test_pack_for_register(array_simulator, mocker, model_version, caplog, tmp_test_directory):
    mocker.patch.object(
        array_simulator,
        "get_file_list",
        side_effect=[
            [f"output_file_{model_version}_simtel.zst"],
            [f"log_file_{model_version}_simtel.log.gz"],
            [f"log_file_corsika_{model_version}.log.gz"],
            [f"hist_file_{model_version}_hist_log.zst"],
        ],
    )
    mocker.patch("shutil.move")
    mocker.patch("tarfile.open")
    mocker.patch("pathlib.Path.exists", return_value=True)
    mocker.patch("pathlib.Path.is_file", return_value=True)

    directory_for_grid_upload = tmp_test_directory / "directory_for_grid_upload"
    with caplog.at_level(logging.INFO):
        array_simulator.pack_for_register(str(directory_for_grid_upload))

    assert "Overwriting existing file" in caplog.text
    assert "Packing the output files for registering on the grid" in caplog.text
    assert "Output files for the grid placed in" in caplog.text
    tarfile.open.assert_called_once()
    shutil.move.assert_any_call(
        Path(f"output_file_{model_version}_simtel.zst"),
        directory_for_grid_upload / Path(f"output_file_{model_version}_simtel.zst"),
    )


def test_initialize_array_models_with_single_version(shower_simulator, model_version):
    # Test with a single model version
    array_models = shower_simulator._initialize_array_models()
    assert len(array_models) == 1
    assert isinstance(array_models[0], ArrayModel)
    assert array_models[0].model_version == model_version
    assert array_models[0].site == shower_simulator.args_dict.get("site")
    assert array_models[0].layout_name == shower_simulator.args_dict.get("array_layout_name")


def test_initialize_array_models_with_multiple_versions(shower_simulator):
    # Test with multiple model versions
    model_versions = ["5.0.0", "6.0.1"]
    shower_simulator.args_dict["model_version"] = model_versions
    array_models = shower_simulator._initialize_array_models()
    assert len(array_models) == 2
    for i, model_version in enumerate(model_versions):
        assert isinstance(array_models[i], ArrayModel)
        assert array_models[i].model_version == model_version
        assert array_models[i].site == shower_simulator.args_dict.get("site")
        assert array_models[i].layout_name == shower_simulator.args_dict.get("array_layout_name")


def test_validate_metadata(array_simulator, mocker, caplog, model_version):
    # Test when simulation software is not simtel
    array_simulator.simulation_software = "corsika"
    with caplog.at_level(logging.INFO):
        array_simulator.validate_metadata()
    assert "No sim_telarray files to validate." in caplog.text

    # Test when simulation software is simtel and there are output files
    array_simulator.simulation_software = "sim_telarray"
    mocker.patch.object(
        array_simulator, "get_file_list", return_value=[f"output_file1_{model_version}"]
    )
    mock_assert_sim_telarray_metadata = mocker.patch(
        "simtools.simulator.assert_sim_telarray_metadata"
    )
    with caplog.at_level(logging.INFO):
        array_simulator.validate_metadata()
    assert f"Validating metadata for output_file1_{model_version}" in caplog.text
    assert f"Metadata for sim_telarray file output_file1_{model_version} is valid." in caplog.text
    assert mock_assert_sim_telarray_metadata.call_count == 1

    mocker.patch.object(array_simulator, "get_file_list", return_value=["output_file1_5.0.0"])
    mocker.patch("simtools.simulator.assert_sim_telarray_metadata")
    with caplog.at_level(logging.WARNING):
        array_simulator.validate_metadata()
    assert "No sim_telarray file found for model version" in caplog.text


def test_pack_for_register_with_multiple_versions(
    io_handler, simulations_args_dict, db_config, mocker, caplog, tmp_test_directory
):
    args_dict = copy.deepcopy(simulations_args_dict)
    args_dict["simulation_software"] = "corsika_sim_telarray"
    args_dict["label"] = "local-test-shower-array-simulator"
    model_versions = ["5.0.0", "6.0.1"]
    args_dict["model_version"] = model_versions
    local_shower_array_simulator = Simulator(
        label=args_dict["label"],
        args_dict=args_dict,
        db_config=db_config,
    )

    # Define file patterns
    file_patterns = {
        "output": "output_file_{}_simtel.zst",
        "log": "log_file_{}_simtel.log.gz",
        "corsika_log": "log_file_corsika_{}.log.gz",
        "hist": "hist_file_{}_hist_log.zst",
    }

    # Generate file lists for side effects
    side_effects = [
        [file_patterns["output"].format(v) for v in model_versions],
        [file_patterns["log"].format(v) for v in model_versions],
        [file_patterns["corsika_log"].format(model_versions[0])],
        [file_patterns["hist"].format(v) for v in model_versions],
    ]

    # Mocking methods and objects
    mocker.patch.object(local_shower_array_simulator, "get_file_list", side_effect=side_effects)
    mocker.patch("shutil.move")

    # Create a mock for tarfile.open that returns a mock context manager
    mock_tar = mocker.MagicMock()
    mock_tar_cm = mocker.MagicMock()
    mock_tar_cm.__enter__ = mocker.MagicMock(return_value=mock_tar)
    mock_tar_cm.__exit__ = mocker.MagicMock(return_value=None)
    mock_tarfile_open = mocker.patch("tarfile.open", return_value=mock_tar_cm)

    mocker.patch("pathlib.Path.is_file", return_value=True)
    mocker.patch("pathlib.Path.exists", return_value=True)
    mocker.patch.object(local_shower_array_simulator, "_copy_corsika_log_file_for_all_versions")

    directory_for_grid_upload = tmp_test_directory / "directory_for_grid_upload"
    with caplog.at_level(logging.INFO):
        local_shower_array_simulator.pack_for_register(str(directory_for_grid_upload))

    # Assertions
    assert "Packing the output files for registering on the grid" in caplog.text
    assert "Output files for the grid placed in" in caplog.text
    local_shower_array_simulator._copy_corsika_log_file_for_all_versions.assert_called_once()

    # Verify tarfile operations
    assert mock_tarfile_open.call_count > 0

    # Generate expected additions using the same patterns
    expected_additions = []
    for version in model_versions:
        for file_type in ["log", "hist"]:
            filename = file_patterns[file_type].format(version)
            expected_additions.append((filename, filename))

    # Check tarball additions
    for filepath, arcname in expected_additions:
        mock_tar.add.assert_any_call(Path(filepath), arcname=arcname)

    # Check file movement
    for version in model_versions:
        output_file = file_patterns["output"].format(version)
        shutil.move.assert_any_call(
            Path(output_file),
            directory_for_grid_upload / Path(output_file),
        )


def test_copy_corsika_log_file_for_all_versions(array_simulator, mocker, tmp_test_directory):
    original_content = b"Original CORSIKA log content."
    expected_content = "Original CORSIKA log content."
    helper_test_copy_corsika_log_file(
        array_simulator, mocker, tmp_test_directory, original_content, expected_content
    )


def test_copy_corsika_log_file_for_all_versions_with_non_unicode(
    array_simulator, mocker, tmp_test_directory
):
    original_content = b"Valid line 1\nValid line 2\nInvalid line \x80\x81\n"
    expected_content = "Valid line 1\nValid line 2\nInvalid line"
    helper_test_copy_corsika_log_file(
        array_simulator, mocker, tmp_test_directory, original_content, expected_content
    )


def helper_test_copy_corsika_log_file(
    array_simulator, mocker, tmp_test_directory, original_content, expected_content
):
    """
    Helper function to test _copy_corsika_log_file_for_all_versions.

    Parameters
    ----------
    array_simulator: Simulator
        The simulator instance.
    mocker: pytest-mocker
        The mocker instance for mocking objects.
    tmp_test_directory: Path
        Temporary directory for creating test files.
    original_content: bytes
        The content to write to the original log file.
    expected_content: str
        The expected content in the new log file.
    """
    # Mock array_models with multiple versions
    array_simulator.array_models = [
        mocker.Mock(model_version="5.0.0"),
        mocker.Mock(model_version="6.0.0"),
    ]

    # Create a temporary directory for log files
    original_log_dir = tmp_test_directory / "logs"
    original_log_dir.mkdir()

    # Create a mock original log file
    original_log_file = original_log_dir / "log_file_5.0.0.log.gz"
    with gzip.open(original_log_file, "wb") as f:
        f.write(original_content)

    # Mock the input corsika_log_files list
    corsika_log_files = [str(original_log_file)]

    # Call the method
    array_simulator._copy_corsika_log_file_for_all_versions(corsika_log_files)

    # Check that the new log file for the second model version was created
    new_log_file = original_log_dir / "log_file_6.0.0.log.gz"
    assert new_log_file.exists()

    # Verify the content of the new log file
    with gzip.open(new_log_file, "rt", encoding="utf-8") as f:
        content = f.read()
        assert "Copy of CORSIKA log file from model version 5.0.0." in content
        assert "Applicable also for 6.0.0" in content
        assert expected_content in content

    # Ensure the new log file was added to the corsika_log_files list
    assert str(new_log_file) in corsika_log_files


def test_get_seed_for_random_instrument_instances(shower_simulator):
    # Test with a seed provided in the configuration
    shower_simulator.sim_telarray_seeds["seed"] = "12345, 67890"
    seed = shower_simulator._get_seed_for_random_instrument_instances(
        shower_simulator.sim_telarray_seeds["seed"], model_version="6.0.1"
    )
    assert seed == 12345

    # Test without a seed provided in the configuration
    shower_simulator.sim_telarray_seeds["seed"] = None
    shower_simulator.args_dict["site"] = "North"
    shower_simulator.args_dict["zenith_angle"] = 20 * u.deg
    shower_simulator.args_dict["azimuth_angle"] = 180 * u.deg
    seed = shower_simulator._get_seed_for_random_instrument_instances(
        shower_simulator.sim_telarray_seeds["seed"], model_version="6.0.1"
    )
    assert seed == 600010000000 + 2000000 + 20 * 1000 + 180

    shower_simulator.args_dict["site"] = "South"
    seed = shower_simulator._get_seed_for_random_instrument_instances(
        shower_simulator.sim_telarray_seeds["seed"], model_version="6.0.1"
    )
    assert seed == 600010000000 + 1000000 + 20 * 1000 + 180


def test_initialize_simulation_runner_with_corsika(shower_simulator, db_config, mocker):
    # Mock CorsikaConfig to avoid actual initialization
    mock_corsika_config = mocker.patch(CORSIKA_CONFIG_MOCK_PATCH, autospec=True)
    mock_corsika_runner = mocker.patch("simtools.simulator.CorsikaRunner", autospec=True)

    # Call the method
    simulation_runner = shower_simulator._initialize_simulation_runner()

    # Assertions
    assert isinstance(simulation_runner, CorsikaRunner)
    mock_corsika_config.assert_called_once_with(
        array_model=shower_simulator.array_models[0],
        label=shower_simulator.label,
        args_dict=shower_simulator.args_dict,
        db_config=db_config,
        dummy_simulations=False,
    )
    mock_corsika_runner.assert_called_once_with(
        label=shower_simulator.label,
        corsika_config=mock_corsika_config.return_value,
        simtel_path=shower_simulator.args_dict.get("simtel_path"),
        use_multipipe=False,
        keep_seeds=shower_simulator.args_dict.get("corsika_test_seeds", False),
        curved_atmosphere_min_zenith_angle=65 * u.deg,
    )


def test_initialize_simulation_runner_with_sim_telarray(array_simulator, db_config, mocker):
    # Mock SimulatorArray to avoid actual initialization
    mock_simulator_array = mocker.patch("simtools.simulator.SimulatorArray", autospec=True)
    mock_corsika_config = mocker.patch(CORSIKA_CONFIG_MOCK_PATCH, autospec=True)

    # Call the method
    simulation_runner = array_simulator._initialize_simulation_runner()

    # Assertions
    assert isinstance(simulation_runner, SimulatorArray)
    mock_simulator_array.assert_called_once_with(
        label=array_simulator.label,
        corsika_config=mock_corsika_config.return_value,
        simtel_path=array_simulator.args_dict.get("simtel_path"),
        use_multipipe=False,
        sim_telarray_seeds=array_simulator.sim_telarray_seeds,
    )


def test_initialize_simulation_runner_with_corsika_sim_telarray(
    shower_array_simulator, db_config, mocker
):
    # Mock CorsikaConfig and CorsikaSimtelRunner to avoid actual initialization
    mock_corsika_config = mocker.patch(CORSIKA_CONFIG_MOCK_PATCH, autospec=True)
    mock_corsika_simtel_runner = mocker.patch(
        "simtools.simulator.CorsikaSimtelRunner", autospec=True
    )

    # Call the method
    simulation_runner = shower_array_simulator._initialize_simulation_runner()

    # Assertions
    assert isinstance(simulation_runner, CorsikaSimtelRunner)
    mock_corsika_config.assert_called()
    mock_corsika_simtel_runner.assert_called_once_with(
        label=shower_array_simulator.label,
        corsika_config=[mock_corsika_config.return_value]
        * len(shower_array_simulator.array_models),
        simtel_path=shower_array_simulator.args_dict.get("simtel_path"),
        use_multipipe=True,
        keep_seeds=shower_array_simulator.args_dict.get("corsika_test_seeds", False),
        curved_atmosphere_min_zenith_angle=65 * u.deg,
        sim_telarray_seeds=shower_array_simulator.sim_telarray_seeds,
        sequential=shower_array_simulator.args_dict.get("sequential", False),
        calibration_config=None,
    )


def test_initialize_simulation_runner_with_calibration_simulator(
    calibration_simulator, db_config, mocker
):
    # Mock CorsikaConfig and CorsikaSimtelRunner to avoid actual initialization
    mock_corsika_config = mocker.patch(CORSIKA_CONFIG_MOCK_PATCH, autospec=True)
    mock_corsika_simtel_runner = mocker.patch(
        "simtools.simulator.CorsikaSimtelRunner", autospec=True
    )

    # Call the method
    simulation_runner = calibration_simulator._initialize_simulation_runner()

    # Assertions
    assert isinstance(simulation_runner, CorsikaSimtelRunner)
    mock_corsika_config.assert_called()
    mock_corsika_simtel_runner.assert_called_once_with(
        label=calibration_simulator.label,
        corsika_config=[mock_corsika_config.return_value] * len(calibration_simulator.array_models),
        simtel_path=calibration_simulator.args_dict.get("simtel_path"),
        use_multipipe=True,
        keep_seeds=calibration_simulator.args_dict.get("corsika_test_seeds", False),
        curved_atmosphere_min_zenith_angle=65 * u.deg,
        sim_telarray_seeds=calibration_simulator.sim_telarray_seeds,
        sequential=calibration_simulator.args_dict.get("sequential", False),
        calibration_config=calibration_simulator.args_dict,
    )


def test_save_reduced_event_lists_not_sim_telarray(shower_simulator, caplog):
    with caplog.at_level(logging.WARNING):
        shower_simulator.save_reduced_event_lists()
    assert "Reduced event lists can only be saved for sim_telarray simulations." in caplog.text


def test_save_reduced_event_lists_sim_telarray(array_simulator, mocker):
    mock_output_files = ["output_file1.simtel.zst", "output_file2.simtel.zst"]
    mock_event_data_files = [
        "output_file1.reduced_event_data.hdf5",
        "output_file2.reduced_event_data.hdf5",
    ]
    mocker.patch.object(
        array_simulator,
        "get_file_list",
        side_effect=lambda file_type: mock_output_files
        if file_type == "simtel_output"
        else mock_event_data_files,
    )

    mock_generator = mocker.MagicMock()
    mock_simtel_io_writer = mocker.patch(
        "simtools.simulator.SimtelIOEventDataWriter", return_value=mock_generator
    )
    mock_table_handler = mocker.patch("simtools.simulator.table_handler")

    array_simulator.save_reduced_event_lists()

    assert mock_simtel_io_writer.call_count == 2
    mock_simtel_io_writer.assert_any_call(["output_file1.simtel.zst"])
    mock_simtel_io_writer.assert_any_call(["output_file2.simtel.zst"])

    assert mock_table_handler.write_tables.call_count == 2
    mock_table_handler.write_tables.assert_any_call(
        tables=mock_generator.process_files.return_value,
        output_file=Path("output_file1.reduced_event_data.hdf5"),
        overwrite_existing=True,
    )
    mock_table_handler.write_tables.assert_any_call(
        tables=mock_generator.process_files.return_value,
        output_file=Path("output_file2.reduced_event_data.hdf5"),
        overwrite_existing=True,
    )


def test_save_reduced_event_lists_no_output_files(array_simulator, mocker):
    mocker.patch.object(array_simulator, "get_file_list", return_value=[])
    mock_simtel_io_writer = mocker.patch("simtools.simulator.SimtelIOEventDataWriter")
    mock_io_table_handler = mocker.patch("simtools.simulator.table_handler")

    array_simulator.save_reduced_event_lists()

    mock_simtel_io_writer.assert_not_called()
    mock_io_table_handler.write_tables.assert_not_called()


def test_is_calibration_run():
    assert Simulator._is_calibration_run("nsb_only_pedestals") is True
    assert Simulator._is_calibration_run(None) is False
    assert Simulator._is_calibration_run("not_a_calibration_run") is False


def test_get_calibration_device_types():
    assert Simulator._get_calibration_device_types("direct_injection") == ["flat_fielding"]
    assert Simulator._get_calibration_device_types("what_ever") == []
