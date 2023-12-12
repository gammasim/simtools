#!/usr/bin/python3

import logging
import math
import shutil
from copy import copy
from pathlib import Path

import pytest

import simtools.utils.general as gen
from simtools.corsika.corsika_runner import CorsikaRunner
from simtools.corsika_simtel.corsika_simtel_runner import CorsikaSimtelRunner
from simtools.model.array_model import ArrayModel
from simtools.simtel.simtel_runner_array import SimtelRunnerArray
from simtools.simulator import InvalidRunsToSimulate, Simulator

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def label():
    return "test"


@pytest.fixture
def input_file_list():
    return ["run1", "abc_run22", "def_run02_and"]


@pytest.fixture
def shower_array_config_data(shower_config_data, array_config_data):
    # Any common entries are taken from the array_config_data
    return shower_config_data | array_config_data


@pytest.fixture
def corsika_file():
    return "run1_proton_za20deg_azm0deg_North_1LST_test.corsika.zst"


@pytest.fixture
def array_simulator(label, array_config_data, io_handler, db_config, simtel_path):
    array_simulator = Simulator(
        label=label,
        simulator="simtel",
        simulator_source_path=simtel_path,
        config_data=array_config_data,
        mongo_db_config=db_config,
    )
    return array_simulator


@pytest.fixture
def shower_simulator(label, shower_config_data, io_handler, db_config, simtel_path):
    shower_simulator = Simulator(
        label=label,
        simulator="corsika",
        simulator_source_path=simtel_path,
        config_data=shower_config_data,
        mongo_db_config=db_config,
    )
    return shower_simulator


@pytest.fixture
def shower_array_simulator(label, simulator_config_data, io_handler, db_config, simtel_path):
    shower_array_simulator = Simulator(
        label=label,
        simulator="corsika_simtel",
        simulator_source_path=simtel_path,
        config_data=simulator_config_data,
        mongo_db_config=db_config,
    )
    return shower_array_simulator


def test_guess_run_from_file(array_simulator):
    assert array_simulator._guess_run_from_file("run12345_bla_ble") == 12345

    # Invalid run number - returns 1
    assert array_simulator._guess_run_from_file("run1test2_bla_ble") == 1

    assert array_simulator._guess_run_from_file("abc-run12345_bla_ble") == 12345

    assert array_simulator._guess_run_from_file("run10") == 10

    assert array_simulator._guess_run_from_file("abc-ran12345_bla_ble") == 1

    # TODO add test for 'Error creating output directory'
    # (not sure how to test a failed mkdir)


def test_set_simulator(array_simulator, shower_simulator, shower_array_simulator):
    array_simulator.simulator = "simtel"
    assert array_simulator.simulator == "simtel"

    shower_simulator.simulator = "corsika"
    assert shower_simulator.simulator == "corsika"

    shower_array_simulator.simulator = "corsika_simtel"
    assert shower_array_simulator.simulator == "corsika_simtel"

    with pytest.raises(gen.InvalidConfigData):
        shower_simulator.simulator = "this_simulator_is_not_there"


def test_load_configuration_and_simulation_model(array_simulator):
    with pytest.raises(gen.InvalidConfigData):
        array_simulator._load_configuration_and_simulation_model()


def test_load_corsika_config_and_model(shower_simulator, shower_config_data):
    assert shower_simulator.site == "North"

    assert "site" not in shower_simulator._corsika_config_data

    _temp_shower_data = copy(shower_config_data)
    _temp_shower_data.pop("site")
    with pytest.raises(KeyError):
        shower_simulator._load_corsika_config_and_model(config_data=_temp_shower_data)


def test_load_sim_tel_config_and_model(
    array_simulator, array_config_data, shower_array_simulator, shower_array_config_data, caplog
):
    with caplog.at_level(logging.DEBUG):
        array_simulator._load_sim_tel_config_and_model(array_config_data)
    assert "in config_data cannot be identified" not in caplog.text

    assert isinstance(array_simulator.array_model, ArrayModel)

    with caplog.at_level(logging.DEBUG):
        shower_array_simulator._load_sim_tel_config_and_model(config_data=shower_array_config_data)
    assert "in config_data cannot be identified" in caplog.text

    assert isinstance(shower_array_simulator.array_model, ArrayModel)


def test_load_shower_array_config_and_model(shower_array_simulator, shower_array_config_data):
    assert shower_array_simulator.site == "North"

    assert "site" not in shower_array_simulator._corsika_config_data
    assert shower_array_simulator.config.primary == "gamma"
    assert (
        shower_array_simulator.config.data_directory == shower_array_config_data["data_directory"]
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

        with pytest.raises(InvalidRunsToSimulate):
            simulator_now._validate_run_list_and_range(run_list=[1, "a", 4], run_range=None)

        assert simulator_now._validate_run_list_and_range(run_list=None, run_range=[3, 6]) == [
            3,
            4,
            5,
            6,
        ]

        assert simulator_now._validate_run_list_and_range(run_list=None, run_range=[6, 3]) == []

        with pytest.raises(InvalidRunsToSimulate):
            simulator_now._validate_run_list_and_range(run_list=None, run_range=[3, "b"])

        with pytest.raises(InvalidRunsToSimulate):
            simulator_now._validate_run_list_and_range(run_list=None, run_range=[3, 4, 5])


def test_collect_array_model_parameters(array_simulator, array_config_data):
    _array_model_data, _rest_data = array_simulator._collect_array_model_parameters(
        config_data=array_config_data
    )

    assert isinstance(_array_model_data, dict)
    assert isinstance(_rest_data, dict)
    assert _array_model_data["site"] == "North"
    assert _array_model_data["LST-01"] == "1"
    new_array_config_data = copy(array_config_data)
    new_array_config_data.pop("site")

    with pytest.raises(KeyError):
        _, _ = array_simulator._collect_array_model_parameters(config_data=new_array_config_data)


def test_set_simulation_runner(array_simulator, shower_simulator, shower_array_simulator):
    assert isinstance(array_simulator._simulation_runner, SimtelRunnerArray)

    assert isinstance(shower_simulator._simulation_runner, CorsikaRunner)

    assert isinstance(shower_array_simulator._simulation_runner, CorsikaSimtelRunner)


def test_fill_results_without_run(array_simulator, input_file_list):
    array_simulator._fill_results_without_run(input_file_list=[])
    assert array_simulator.runs == list()

    array_simulator._fill_results_without_run(input_file_list=input_file_list)
    assert array_simulator.runs == [1, 22, 2]


def test_submitting(shower_simulator, array_simulator, corsika_file):
    shower_simulator.test = True
    shower_simulator._submit_command = "local"
    shower_simulator.simulate()

    shower_simulator.print_list_of_output_files()
    shower_simulator.print_list_of_log_files()
    shower_simulator.print_list_of_input_files()

    run_script = shower_simulator._simulation_runner.prepare_run_script(run_number=2)

    assert Path(run_script).exists()

    array_simulator.test = True
    array_simulator._submit_command = "local"
    array_simulator.simulate(input_file_list=corsika_file)

    array_simulator.print_list_of_output_files()
    array_simulator.print_list_of_log_files()
    array_simulator.print_list_of_input_files()

    input_files = array_simulator.get_list_of_input_files()
    assert str(input_files[0]) == str(corsika_file)


def test_get_runs_and_files_to_submit(
    array_simulator, shower_simulator, shower_array_simulator, input_file_list
):
    assert array_simulator._get_runs_and_files_to_submit(input_file_list=None) == dict()

    assert array_simulator._get_runs_and_files_to_submit(input_file_list=input_file_list) == {
        1: "run1",
        2: "def_run02_and",
        22: "abc_run22",
    }

    for simulator_now in [shower_simulator, shower_array_simulator]:
        assert simulator_now._get_runs_and_files_to_submit(input_file_list=None) == {
            3: None,
            4: None,
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


def test_fill_results(array_simulator, shower_simulator, shower_array_simulator, input_file_list):
    for simulator_now in [array_simulator, shower_array_simulator]:
        simulator_now._fill_results_without_run(input_file_list)
        assert len(simulator_now.get_list_of_output_files()) == 3
        assert len(simulator_now._results["sub_out"]) == 3
        assert len(simulator_now.get_list_of_log_files()) == 3
        assert len(simulator_now.get_list_of_input_files()) == 3
        assert len(simulator_now.get_list_of_histogram_files()) == 3
        assert simulator_now.get_list_of_input_files()[1] == "abc_run22"

    shower_simulator._fill_results_without_run(input_file_list)
    assert len(shower_simulator.get_list_of_output_files()) == 3
    assert len(shower_simulator.get_list_of_log_files()) == 3
    assert shower_simulator.get_list_of_histogram_files()[1] is None


def test_get_list_of_files(shower_simulator):
    assert len(shower_simulator.get_list_of_output_files()) == len(shower_simulator.runs)
    assert len(shower_simulator.get_list_of_output_files(run_list=[2, 5, 7])) == 10
    assert len(shower_simulator.get_list_of_output_files(run_range=[1, 4])) == 14


def test_no_corsika_data(shower_config_data, label, simtel_path, io_handler, db_config):
    new_shower_config_data = copy(shower_config_data)
    new_shower_config_data.pop("data_directory", None)
    new_shower_simulator = Simulator(
        label=label,
        simulator="corsika",
        config_data=new_shower_config_data,
        simulator_source_path=simtel_path,
        mongo_db_config=db_config,
    )
    files = new_shower_simulator.get_list_of_output_files(run_list=[3])

    assert "/" + label + "/" in files[0]


def test_make_resources_report(label, shower_config_data, io_handler, db_config, simtel_path):
    shower_config_data["run_list"] = 1
    shower_config_data["run_range"] = None
    shower_simulator = Simulator(
        label="corsika-test",
        simulator="corsika",
        config_data=shower_config_data,
        simulator_source_path=simtel_path,
        mongo_db_config=db_config,
    )
    _resources_1 = shower_simulator._make_resources_report(input_file_list=None)
    assert math.isnan(_resources_1["Walltime/run [sec]"])

    # Copying the corsika log file to the expected location in the test directory.
    # This should not affect the efficacy of this test.
    log_file_name = "log_sub_corsika_run000001_gamma_North_TestLayout_test-production.out"
    shutil.copy(
        f"tests/resources/{log_file_name}",
        shower_simulator._simulation_runner.get_file_name(
            file_type="sub_log",
            **shower_simulator._simulation_runner.get_info_for_file_name(1),
            mode="out",
        ),
    )
    _resources_1 = shower_simulator._make_resources_report(input_file_list=log_file_name)
    assert _resources_1["Walltime/run [sec]"] == 6

    with pytest.raises(FileNotFoundError):
        shower_simulator.runs = [4]
        shower_simulator._make_resources_report(input_file_list)


def test_get_runs_to_simulate(shower_config_data, simtel_path, io_handler, db_config):
    shower_simulator = Simulator(
        label="corsika-test",
        simulator="corsika",
        config_data=shower_config_data,
        simulator_source_path=simtel_path,
        mongo_db_config=db_config,
    )
    assert len(shower_simulator.runs) == len(
        shower_simulator._get_runs_to_simulate(run_list=None, run_range=None)
    )

    assert 3 == len(shower_simulator._get_runs_to_simulate(run_list=[2, 5, 7]))

    assert 4 == len(shower_simulator._get_runs_to_simulate(run_range=[1, 4]))

    shower_simulator.runs = None
    assert shower_simulator._get_runs_to_simulate() == list()


def test_print_list_of_files(array_simulator, input_file_list):
    array_simulator._fill_results_without_run(input_file_list)
    with pytest.raises(KeyError):
        array_simulator._print_list_of_files("blabla")
    array_simulator._print_list_of_files("log")
