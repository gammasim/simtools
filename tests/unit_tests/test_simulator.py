#!/usr/bin/python3

import copy
import logging
import math
import shutil
import tarfile
from pathlib import Path

import pytest

import simtools.utils.general as gen
from simtools.model.array_model import ArrayModel
from simtools.runners.corsika_runner import CorsikaRunner
from simtools.runners.corsika_simtel_runner import CorsikaSimtelRunner
from simtools.simtel.simulator_array import SimulatorArray
from simtools.simulator import InvalidRunsToSimulateError, Simulator

logger = logging.getLogger()


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
    args_dict["simulation_software"] = "simtel"
    args_dict["simtel_path"] = simtel_path
    args_dict["model_version"] = model_version
    args_dict["label"] = "test-array-simulator"
    args_dict["array_layout_name"] = "test_layout"
    args_dict["site"] = "North"
    args_dict["keep_seeds"] = False
    args_dict["run_number_start"] = 1
    args_dict["nshow"] = 10
    args_dict["submit_engine"] = submit_engine
    args_dict["extra_commands"] = None
    return args_dict


@pytest.fixture
def array_simulator(io_handler, db_config, simulations_args_dict):
    args_dict = copy.deepcopy(simulations_args_dict)
    args_dict["simulation_software"] = "simtel"

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
    args_dict["simulation_software"] = "corsika_simtel"
    args_dict["label"] = "test-shower-array-simulator"
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
    assert array_simulator.simulation_software == "simtel"
    assert shower_simulator.simulation_software == "corsika"
    assert shower_array_simulator.simulation_software == "corsika_simtel"

    # setting
    test_array_simulator = copy.deepcopy(array_simulator)
    test_array_simulator.simulation_software = "corsika"
    assert test_array_simulator.simulation_software == "corsika"

    with pytest.raises(gen.InvalidConfigDataError):
        with caplog.at_level(logging.ERROR):
            test_array_simulator.simulation_software = "this_simulator_is_not_there"
    assert "Invalid simulation software" in caplog.text


def test_initialize_array_model(shower_simulator, db_config):
    assert isinstance(
        shower_simulator._initialize_array_model(mongo_db_config=db_config),
        ArrayModel,
    )


def test_initialize_run_list(shower_simulator, caplog):
    assert shower_simulator._initialize_run_list() == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    test_shower_simulator = copy.deepcopy(shower_simulator)
    test_shower_simulator.args_dict.pop("run_number_start", None)
    with caplog.at_level(logging.ERROR):
        with pytest.raises(KeyError):
            test_shower_simulator._initialize_run_list()
    assert (
        "Error in initializing run list (missing 'run_number_start' or 'number_of_runs')"
        in caplog.text
    )


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
    assert shower_simulator.get_file_list("hist")[0] is None


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


def test_pack_for_register(array_simulator, mocker, caplog):
    mocker.patch.object(
        array_simulator,
        "get_file_list",
        side_effect=[
            ["output_file1", "output_file2"],
            ["log_file1", "log_file2"],
            ["hist_file1", "hist_file2"],
            ["corsika_log_file1", "corsika_log_file2"],
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
        Path("output_file1"), Path("directory_for_grid_upload/output_file1")
    )
    shutil.move.assert_any_call(
        Path("output_file2"), Path("directory_for_grid_upload/output_file2")
    )
