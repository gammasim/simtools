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

import simtools.utils.general as gen
from simtools.model.array_model import ArrayModel
from simtools.runners.corsika_runner import CorsikaRunner
from simtools.runners.corsika_simtel_runner import CorsikaSimtelRunner
from simtools.simtel.simulator_array import SimulatorArray
from simtools.simulator import InvalidRunsToSimulateError, Simulator

logger = logging.getLogger()

CORSIKA_CONFIG_MOCK_PATCH = "simtools.simulator.CorsikaConfig"


@pytest.fixture
def input_file_list():
    return ["run1", "abc_run22", "def_run02_and"]


@pytest.fixture
def corsika_file():
    return "run1_proton_za20deg_azm0deg_North_1LST_test.corsika.zst"


@pytest.fixture
def submit_engine():
    return "local"


@pytest.fixture
def simulations_args_dict(corsika_config_data, model_version, simtel_path, submit_engine):
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
    args_dict["submit_engine"] = submit_engine
    args_dict["extra_commands"] = None
    return args_dict


@pytest.fixture
def array_simulator(io_handler, db_config, simulations_args_dict):
    args_dict = copy.deepcopy(simulations_args_dict)
    args_dict["simulation_software"] = "sim_telarray"

    return Simulator(
        label=args_dict["label"],
        args_dict=args_dict,
        mongo_db_config=db_config,
    )


@pytest.fixture
def shower_simulator(io_handler, db_config, simulations_args_dict):
    args_dict = copy.deepcopy(simulations_args_dict)
    args_dict["simulation_software"] = "corsika"
    args_dict["label"] = "test-shower-simulator"

    return Simulator(
        label=args_dict["label"],
        args_dict=args_dict,
        mongo_db_config=db_config,
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
        mongo_db_config=db_config,
    )


def test_init_simulator(shower_simulator, array_simulator, shower_array_simulator):
    assert isinstance(shower_simulator._simulation_runner, CorsikaRunner)
    assert isinstance(shower_array_simulator._simulation_runner, CorsikaSimtelRunner)
    assert isinstance(array_simulator._simulation_runner, SimulatorArray)


def test_simulation_software(array_simulator, shower_simulator, shower_array_simulator, caplog):
    assert array_simulator.simulation_software == "sim_telarray"
    assert shower_simulator.simulation_software == "corsika"
    assert shower_array_simulator.simulation_software == "corsika_sim_telarray"

    # setting
    test_array_simulator = copy.deepcopy(array_simulator)
    test_array_simulator.simulation_software = "corsika"
    assert test_array_simulator.simulation_software == "corsika"

    with pytest.raises(gen.InvalidConfigDataError):
        with caplog.at_level(logging.ERROR):
            test_array_simulator.simulation_software = "this_simulator_is_not_there"
    assert "Invalid simulation software" in caplog.text


def test_initialize_run_list(shower_simulator, caplog):
    assert shower_simulator._initialize_run_list() == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    test_shower_simulator = copy.deepcopy(shower_simulator)
    test_shower_simulator.args_dict.pop("run_number_offset", None)
    with caplog.at_level(logging.ERROR):
        with pytest.raises(KeyError):
            test_shower_simulator._initialize_run_list()
    assert (
        "Error in initializing run list "
        "(missing 'run_number', 'run_number_offset' or 'number_of_runs')."
    ) in caplog.text


def test_validate_run_list_and_range(shower_simulator, shower_array_simulator):
    for simulator_now in [shower_simulator, shower_array_simulator]:
        assert not simulator_now._validate_run_list_and_range(None, None)

        run_list = [1, 24, 3]

        assert simulator_now._validate_run_list_and_range(run_list=run_list, run_range=None) == [
            1,
            3,
            24,
        ]

        with pytest.raises(InvalidRunsToSimulateError):
            simulator_now._validate_run_list_and_range(run_list=[1, "a", 4], run_range=None)

        assert simulator_now._validate_run_list_and_range(run_list=None, run_range=[3, 6]) == [
            3,
            4,
            5,
        ]

        assert simulator_now._validate_run_list_and_range(run_list=None, run_range=[6, 3]) == []

        with pytest.raises(InvalidRunsToSimulateError):
            simulator_now._validate_run_list_and_range(run_list=None, run_range=[3, "b"])

        with pytest.raises(InvalidRunsToSimulateError):
            simulator_now._validate_run_list_and_range(run_list=None, run_range=[3, 4, 5])

        assert simulator_now._validate_run_list_and_range(run_list=5, run_range=None) == [5]


def test_fill_results_without_run(array_simulator, input_file_list):
    array_simulator._fill_results_without_run(input_file_list=[])
    assert isinstance(array_simulator.runs, list)

    array_simulator.runs = []
    array_simulator._fill_results_without_run(input_file_list=input_file_list)
    assert array_simulator.runs == [1, 22, 2]


def test_simulate_shower_simulator(shower_simulator, submit_engine):
    shower_simulator._test = True
    shower_simulator._submit_engine = submit_engine
    shower_simulator.simulate()
    assert len(shower_simulator._results["output"]) > 0
    assert len(shower_simulator._results["sub_out"]) > 0
    run_script = shower_simulator._simulation_runner.prepare_run_script(run_number=2)
    assert Path(run_script).exists()


def test_simulate_array_simulator(array_simulator, corsika_file, submit_engine):
    array_simulator._test = True
    array_simulator._submit_engine = submit_engine
    array_simulator.simulate(input_file_list=corsika_file)

    assert len(array_simulator._results["output"]) > 0
    assert len(array_simulator._results["sub_out"]) > 0


def test_simulate_shower_array_simulator(shower_array_simulator, submit_engine):
    shower_array_simulator._test = True
    shower_array_simulator._submit_engine = submit_engine
    shower_array_simulator.simulate()

    assert len(shower_array_simulator._results["output"]) > 0
    assert len(shower_array_simulator._results["sub_out"]) > 0


def test_get_runs_and_files_to_submit(
    array_simulator, shower_simulator, shower_array_simulator, input_file_list
):
    with pytest.raises(ValueError, match="No runs to submit."):
        assert array_simulator._get_runs_and_files_to_submit(input_file_list=None) == {}

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


def test_enforce_list_type(array_simulator):
    assert array_simulator._enforce_list_type(None) == []
    assert array_simulator._enforce_list_type([1, 2, 3]) == [1, 2, 3]
    assert array_simulator._enforce_list_type(5) == [5]


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
        assert len(simulator_now.get_file_list("output")) == 3
        assert len(simulator_now._results["sub_out"]) == 3
        assert len(simulator_now.get_file_list("log")) == 3
        assert len(simulator_now.get_file_list("input")) == 3
        assert len(simulator_now.get_file_list("hist")) == 3
        logger.error(simulator_now.get_file_list("input"))
        assert simulator_now.get_file_list("input")[1] == "abc_run22"

    shower_simulator._fill_results(input_file_list[1], run_number=5)
    assert len(shower_simulator.get_file_list("output")) == 1
    assert len(shower_simulator.get_file_list("corsika_log")) == 1
    assert len(shower_simulator.get_file_list("hist")) == 0


def test_get_list_of_files(shower_simulator):
    test_shower_simulator = copy.deepcopy(shower_simulator)
    test_shower_simulator._results["output"] = ["file_name"] * 10
    assert len(test_shower_simulator.get_file_list("output")) == len(shower_simulator.runs)
    assert len(test_shower_simulator.get_file_list("not_a_valid_file_type")) == 0


def test_print_list_of_files(array_simulator, input_file_list, capsys):
    array_simulator._fill_results_without_run(input_file_list)
    array_simulator.print_list_of_files("log")
    captured = capsys.readouterr()
    assert "log.gz" in captured.out
    assert captured.out.count("log.gz") == 3
    array_simulator.print_list_of_files("blabla")
    assert captured.out.count("blabal") == 0


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
    assert "No files to save for output files." in caplog.text

    mock_shower_simulator = copy.deepcopy(shower_simulator)
    mocker.patch.object(mock_shower_simulator, "get_file_list", return_value=["file1", "file2"])

    with caplog.at_level(logging.INFO):
        mock_shower_simulator.save_file_lists()
    assert "Saving list of output files to" in caplog.text

    mocker.patch.object(mock_shower_simulator, "get_file_list", return_value=[None, None])
    with caplog.at_level(logging.DEBUG):
        mock_shower_simulator.save_file_lists()
    assert "No files to save for output files." in caplog.text


def test_pack_for_register(array_simulator, mocker, model_version, caplog):
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

    with caplog.at_level(logging.INFO):
        array_simulator.pack_for_register("directory_for_grid_upload")

    assert "Overwriting existing file" in caplog.text
    assert "Packing the output files for registering on the grid" in caplog.text
    assert "Output files for the grid placed in" in caplog.text
    tarfile.open.assert_called_once()
    shutil.move.assert_any_call(
        Path(f"output_file_{model_version}_simtel.zst"),
        Path(f"directory_for_grid_upload/output_file_{model_version}_simtel.zst"),
    )


def test_initialize_array_models_with_single_version(shower_simulator, db_config, model_version):
    # Test with a single model version
    array_models = shower_simulator._initialize_array_models(mongo_db_config=db_config)
    assert len(array_models) == 1
    assert isinstance(array_models[0], ArrayModel)
    assert array_models[0].model_version == model_version
    assert array_models[0].site == shower_simulator.args_dict.get("site")
    assert array_models[0].layout_name == shower_simulator.args_dict.get("array_layout_name")


def test_initialize_array_models_with_multiple_versions(shower_simulator, db_config):
    # Test with multiple model versions
    model_versions = ["5.0.0", "6.0.0"]
    shower_simulator.args_dict["model_version"] = model_versions
    array_models = shower_simulator._initialize_array_models(mongo_db_config=db_config)
    assert len(array_models) == 2
    for i, model_version in enumerate(model_versions):
        assert isinstance(array_models[i], ArrayModel)
        assert array_models[i].model_version == model_version
        assert array_models[i].site == shower_simulator.args_dict.get("site")
        assert array_models[i].layout_name == shower_simulator.args_dict.get("array_layout_name")


def test_validate_metadata(array_simulator, mocker, caplog):
    # Test when simulation software is not simtel
    array_simulator.simulation_software = "corsika"
    with caplog.at_level(logging.INFO):
        array_simulator.validate_metadata()
    assert "No sim_telarray files to validate." in caplog.text

    # Test when simulation software is simtel and there are output files
    array_simulator.simulation_software = "sim_telarray"
    mocker.patch.object(array_simulator, "get_file_list", return_value=["output_file1_6.0.0"])
    mock_assert_sim_telarray_metadata = mocker.patch(
        "simtools.simulator.assert_sim_telarray_metadata"
    )
    with caplog.at_level(logging.INFO):
        array_simulator.validate_metadata()
    assert "Validating metadata for output_file1_6.0.0" in caplog.text
    assert "Metadata for sim_telarray file output_file1_6.0.0 is valid." in caplog.text
    assert mock_assert_sim_telarray_metadata.call_count == 1

    mocker.patch.object(array_simulator, "get_file_list", return_value=["output_file1_5.0.0"])
    mocker.patch("simtools.simulator.assert_sim_telarray_metadata")
    with caplog.at_level(logging.WARNING):
        array_simulator.validate_metadata()
    assert "No sim_telarray file found for model version 6.0.0:" in caplog.text


def test_pack_for_register_with_multiple_versions(
    io_handler, simulations_args_dict, db_config, mocker, caplog
):
    args_dict = copy.deepcopy(simulations_args_dict)
    args_dict["simulation_software"] = "corsika_sim_telarray"
    args_dict["label"] = "local-test-shower-array-simulator"
    model_versions = ["5.0.0", "6.0.0"]
    args_dict["model_version"] = model_versions
    local_shower_array_simulator = Simulator(
        label=args_dict["label"],
        args_dict=args_dict,
        mongo_db_config=db_config,
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

    mocker.patch("pathlib.Path.exists", return_value=True)
    mocker.patch.object(local_shower_array_simulator, "_copy_corsika_log_file_for_all_versions")

    # Call the method
    with caplog.at_level(logging.INFO):
        local_shower_array_simulator.pack_for_register("directory_for_grid_upload")

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
        mock_tar.add.assert_any_call(filepath, arcname=arcname)

    # Check file movement
    for version in model_versions:
        output_file = file_patterns["output"].format(version)
        shutil.move.assert_any_call(
            Path(output_file),
            Path(f"directory_for_grid_upload/{output_file}"),
        )


def test_copy_corsika_log_file_for_all_versions(array_simulator, mocker, tmp_test_directory):
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
    with gzip.open(original_log_file, "wt") as f:
        f.write("Original CORSIKA log content.")

    # Mock the input corsika_log_files list
    corsika_log_files = [str(original_log_file)]

    # Call the method
    array_simulator._copy_corsika_log_file_for_all_versions(corsika_log_files)

    # Check that the new log file for the second model version was created
    new_log_file = original_log_dir / "log_file_6.0.0.log.gz"
    assert new_log_file.exists()

    # Verify the content of the new log file
    with gzip.open(new_log_file, "rt") as f:
        content = f.read()
        assert "Copy of CORSIKA log file from model version 5.0.0." in content
        assert "Applicable also for 6.0.0" in content
        assert "Original CORSIKA log content." in content

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
    simulation_runner = shower_simulator._initialize_simulation_runner(db_config)

    # Assertions
    assert isinstance(simulation_runner, CorsikaRunner)
    mock_corsika_config.assert_called_once_with(
        array_model=shower_simulator.array_models[0],
        label=shower_simulator.label,
        args_dict=shower_simulator.args_dict,
        db_config=db_config,
    )
    mock_corsika_runner.assert_called_once_with(
        label=shower_simulator.label,
        corsika_config=mock_corsika_config.return_value,
        simtel_path=shower_simulator.args_dict.get("simtel_path"),
        use_multipipe=False,
        keep_seeds=shower_simulator.args_dict.get("corsika_test_seeds", False),
    )


def test_initialize_simulation_runner_with_sim_telarray(array_simulator, db_config, mocker):
    # Mock SimulatorArray to avoid actual initialization
    mock_simulator_array = mocker.patch("simtools.simulator.SimulatorArray", autospec=True)
    mock_corsika_config = mocker.patch(CORSIKA_CONFIG_MOCK_PATCH, autospec=True)

    # Call the method
    simulation_runner = array_simulator._initialize_simulation_runner(db_config)

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
    simulation_runner = shower_array_simulator._initialize_simulation_runner(db_config)

    # Assertions
    assert isinstance(simulation_runner, CorsikaSimtelRunner)
    mock_corsika_config.assert_called()
    mock_corsika_simtel_runner.assert_called_once_with(
        label=shower_array_simulator.label,
        corsika_config=[mock_corsika_config.return_value]
        * len(shower_array_simulator.array_models),
        simtel_path=shower_array_simulator.args_dict.get("simtel_path"),
        use_multipipe=True,
        sim_telarray_seeds=shower_array_simulator.sim_telarray_seeds,
        sequential=shower_array_simulator.args_dict.get("sequential", False),
        keep_seeds=shower_array_simulator.args_dict.get("corsika_test_seeds", False),
    )
