#!/usr/bin/python3

import copy
import logging
import shutil
import tarfile
import warnings
from pathlib import Path
from unittest import mock

import pytest

from simtools.sim_events import file_info
from simtools.simulator import Simulator

logger = logging.getLogger()

CORSIKA_CONFIG_MOCK_PATCH = "simtools.simulator.CorsikaConfig"
INITIALIZE_RUN_LIST_ERROR_MSG = (
    "Error in initializing run list "
    "(missing 'run_number', 'run_number_offset' or 'number_of_runs')."
)


@pytest.fixture
def simulations_args_dict(corsika_config_data, model_version):
    """Return a dictionary with the simulation command line arguments."""
    args_dict = copy.deepcopy(corsika_config_data)
    args_dict["simulation_software"] = "sim_telarray"
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
def mock_array_model(model_version):
    """Create a mock ArrayModel for testing without database access."""
    array_model = mock.MagicMock()
    array_model.layout_name = "test_layout"
    array_model.site = "North"
    array_model.model_version = model_version
    array_model.site_model = mock.MagicMock()
    array_model.site_model._parameters = {"geomag_rotation": -4.533}

    def mock_get_parameter_value(par_name):
        return array_model.site_model._parameters.get(par_name)

    array_model.site_model.get_parameter_value.side_effect = mock_get_parameter_value
    array_model.pack_model_files.return_value = []

    return array_model


@pytest.fixture
def patch_simulator_core(mocker, mock_array_model):
    """Patch core simulator dependencies to avoid DB and heavy init."""

    def _apply():
        mocker.patch("simtools.simulator.ArrayModel", return_value=mock_array_model)
        mocker.patch("simtools.simulator.CorsikaConfig")
        mock_runner_service = mocker.patch("simtools.simulator.runner_services.RunnerServices")
        mock_runner_service.return_value.load_files.return_value = {}
        mock_runner_service.return_value.get_file_name.side_effect = (
            lambda file_type, run_number=1, **kwargs: f"{file_type}_{run_number}.txt"
        )

    return _apply


@pytest.fixture
def configure_runner_mock(io_handler):
    """Configure a runner patch with common behaviors used in tests."""

    def _configure(runner_patch, add_resources=False):
        runner_patch.return_value.prepare_run.return_value = str(
            io_handler.get_output_directory() / "test_run_script.sh"
        )
        runner_patch.return_value.get_file_name.side_effect = lambda file_type, **kwargs: str(
            io_handler.get_output_directory() / f"{file_type}_{kwargs.get('run_number', 1)}.txt"
        )
        if add_resources:

            def mock_get_resources(run_number):
                file_path = io_handler.get_output_directory() / f"sub_log_{run_number}.txt"
                if not file_path.exists():
                    raise FileNotFoundError(f"Log file not found: {file_path}")
                return {"runtime": 6, "n_events": 100}

            runner_patch.return_value.get_resources.side_effect = mock_get_resources

    return _configure


@pytest.fixture
def array_simulator(
    io_handler,
    simulations_args_dict,
    patch_simulator_core,
    configure_runner_mock,
    mocker,
):
    args_dict = copy.deepcopy(simulations_args_dict)
    args_dict["simulation_software"] = "sim_telarray"

    mock_config = mocker.Mock()
    mock_config.args = args_dict
    mocker.patch("simtools.settings.config", mock_config)

    patch_simulator_core()
    mock_runner = mocker.patch("simtools.simulator.SimulatorArray")
    configure_runner_mock(mock_runner)

    return Simulator(
        label=args_dict["label"],
    )


@pytest.fixture
def shower_simulator(
    io_handler,
    simulations_args_dict,
    patch_simulator_core,
    configure_runner_mock,
    mocker,
):
    args_dict = copy.deepcopy(simulations_args_dict)
    args_dict["simulation_software"] = "corsika"
    args_dict["label"] = "test-shower-simulator"

    mock_config = mocker.Mock()
    mock_config.args = args_dict
    mocker.patch("simtools.settings.config", mock_config)

    patch_simulator_core()
    mock_runner = mocker.patch("simtools.runners.corsika_runner.CorsikaRunner")
    configure_runner_mock(mock_runner, add_resources=True)

    return Simulator(
        label=args_dict["label"],
    )


@pytest.fixture
def shower_array_simulator(
    io_handler,
    simulations_args_dict,
    patch_simulator_core,
    configure_runner_mock,
    mocker,
):
    args_dict = copy.deepcopy(simulations_args_dict)
    args_dict["simulation_software"] = "corsika_sim_telarray"
    args_dict["label"] = "test-shower-array-simulator"
    args_dict["sequential"] = True

    # Mock the entire settings.config object with a Mock that has an args property
    mock_config = mocker.Mock()
    mock_config.args = args_dict
    mocker.patch("simtools.settings.config", mock_config)

    patch_simulator_core()
    mock_runner = mocker.patch("simtools.runners.corsika_simtel_runner.CorsikaSimtelRunner")
    configure_runner_mock(mock_runner)

    return Simulator(
        label=args_dict["label"],
    )


@pytest.fixture
def calibration_simulator(
    io_handler,
    simulations_args_dict,
    patch_simulator_core,
    configure_runner_mock,
    mocker,
):
    args_dict = copy.deepcopy(simulations_args_dict)
    args_dict["simulation_software"] = "corsika_sim_telarray"
    args_dict["label"] = "test-calibration-shower-array-simulator"
    args_dict["sequential"] = True
    args_dict["run_mode"] = "pedestals_nsb_only"

    mock_config = mocker.Mock()
    mock_config.args = args_dict
    mocker.patch("simtools.settings.config", mock_config)

    patch_simulator_core()
    mock_runner = mocker.patch("simtools.runners.corsika_simtel_runner.CorsikaSimtelRunner")
    configure_runner_mock(mock_runner)

    return Simulator(
        label=args_dict["label"],
    )


def test_init_simulator(shower_simulator, array_simulator, shower_array_simulator):
    assert shower_simulator._simulation_runner is not None
    assert shower_array_simulator._simulation_runner is not None
    assert array_simulator._simulation_runner is not None


def test_simulation_software(array_simulator, shower_simulator, shower_array_simulator):
    assert array_simulator.simulation_software == "sim_telarray"
    assert shower_simulator.simulation_software == "corsika"
    assert shower_array_simulator.simulation_software == "corsika_sim_telarray"

    test_array_simulator = copy.deepcopy(array_simulator)
    test_array_simulator.simulation_software = "corsika"
    assert test_array_simulator.simulation_software == "corsika"

    with pytest.raises(
        ValueError, match="Invalid simulation software: this_simulator_is_not_there"
    ):
        test_array_simulator.simulation_software = "this_simulator_is_not_there"


def test_pack_for_register(array_simulator, mocker, model_version, caplog, tmp_test_directory):
    mocker.patch.object(
        array_simulator,
        "get_files",
        side_effect=[
            [f"output_file_{model_version}_simtel.zst"],
            [f"log_file_{model_version}_simtel.log.gz"],
            [f"log_file_corsika_{model_version}.log.gz"],
            [f"hist_file_{model_version}_hist_log.zst"],
        ],
    )
    mocker.patch("shutil.move")
    mocker.patch("tarfile.open")  # NOSONAR
    mocker.patch("pathlib.Path.exists", return_value=True)
    mocker.patch("pathlib.Path.is_file", return_value=True)

    directory_for_grid_upload = tmp_test_directory / "directory_for_grid_upload"
    with caplog.at_level(logging.INFO):
        array_simulator.pack_for_register(str(directory_for_grid_upload))

    assert "Overwriting existing file" in caplog.text
    assert "Packing output files for registering on the grid" in caplog.text
    assert "Grid output files grid placed in" in caplog.text
    tarfile.open.assert_called_once()  # NOSONAR
    shutil.move.assert_any_call(
        f"output_file_{model_version}_simtel.zst",
        directory_for_grid_upload / Path(f"output_file_{model_version}_simtel.zst"),
    )


def test_initialize_array_models_with_single_version(
    shower_simulator, model_version, mock_array_model
):
    array_models, corsika_configurations = shower_simulator._initialize_array_models()
    assert len(array_models) == 1
    assert array_models[0] is not None
    assert array_models[0] == mock_array_model
    assert corsika_configurations is not None


def test_initialize_from_tool_configuration_with_corsika_file(shower_simulator, mocker):
    """Test initialization when a corsika file is provided."""
    corsika_file = "test_corsika.corsika.gz"

    mock_config = mocker.Mock()
    mock_config.args = {"corsika_file": corsika_file}
    mocker.patch("simtools.settings.config", mock_config)

    mocker.patch("simtools.sim_events.file_info.get_corsika_run_number", return_value=42)

    shower_simulator.run_number = shower_simulator._initialize_from_tool_configuration()
    assert shower_simulator.run_number == 42
    file_info.get_corsika_run_number.assert_called_once_with(corsika_file)


def test_initialize_array_models_with_multiple_versions(shower_simulator, mocker):
    model_versions = ["5.0.0", "6.0.1"]
    mock_models = []
    for version in model_versions:
        m = mocker.MagicMock()
        m.model_version = version
        mock_models.append(m)

    mocker.patch("simtools.simulator.ArrayModel", side_effect=mock_models)
    mock_config = mocker.Mock()
    mock_config.args = {"model_version": model_versions}
    mocker.patch("simtools.settings.config", mock_config)
    shower_simulator.model_version = model_versions
    array_models, _ = shower_simulator._initialize_array_models()
    assert len(array_models) == 2
    for i, model_version in enumerate(model_versions):
        assert array_models[i] is not None
        assert array_models[i] == mock_models[i]


def test_pack_for_register_with_multiple_versions(
    io_handler, simulations_args_dict, mocker, caplog, tmp_test_directory, model_version
):
    args_dict = copy.deepcopy(simulations_args_dict)
    args_dict["simulation_software"] = "corsika_sim_telarray"
    args_dict["label"] = "local-test-shower-array-simulator"
    model_versions = ["5.0.0", "6.0.1"]
    args_dict["model_version"] = model_versions

    mock_array_models = []
    for version in model_versions:
        mock_model = mocker.MagicMock()
        mock_model.model_version = version
        mock_model.pack_model_files.return_value = []
        mock_array_models.append(mock_model)

    mocker.patch("simtools.simulator.ArrayModel", side_effect=mock_array_models)

    mock_corsika_config = mocker.MagicMock()
    mock_corsika_config.get_config_parameter.side_effect = lambda param: {
        "VIEWCONE": [0, 10],
        "THETAP": [20, 20],
    }.get(param, [0, 0])
    mock_corsika_config.azimuth_angle = 0  # from args
    mock_corsika_config.array_model.site = "North"  # from args
    mock_corsika_config.array_model.layout_name = "test_layout"  # from args
    mock_corsika_config.array_model.model_version = model_versions[0]
    mock_corsika_config.run_mode = None
    mock_corsika_config.primary_particle.name = "proton"  # from args

    mocker.patch("simtools.simulator.CorsikaConfig", return_value=mock_corsika_config)
    mocker.patch("simtools.runners.corsika_simtel_runner.CorsikaSimtelRunner")

    mock_config = mocker.Mock()
    mock_config.args = args_dict
    mocker.patch("simtools.settings.config", mock_config)

    local_shower_array_simulator = Simulator(label=args_dict["label"])

    file_patterns = {
        "output": "output_file_{}_simtel.zst",
        "log": "log_file_{}_simtel.log.gz",
        "corsika_log": "log_file_corsika_{}.log.gz",
        "histogram": "hist_file_{}_hist_log.zst",
    }

    def mock_get_files(file_type):
        if file_type == "sim_telarray_output":
            return [file_patterns["output"].format(v) for v in model_versions]
        if file_type == "sim_telarray_log":
            return [file_patterns["log"].format(v) for v in model_versions]
        if file_type == "corsika_log":
            return [file_patterns["corsika_log"].format(model_versions[0])]
        if file_type == "sim_telarray_histogram":
            return [file_patterns["histogram"].format(v) for v in model_versions]
        if file_type == "sim_telarray_event_data":
            return []  # Empty since save_reduced_event_lists is not set
        return []

    mocker.patch.object(local_shower_array_simulator, "get_files", side_effect=mock_get_files)
    mocker.patch("shutil.move")
    mocker.patch("pathlib.Path.is_file", return_value=True)
    mocker.patch("pathlib.Path.exists", return_value=True)
    mocker.patch("simtools.utils.general.pack_tar_file")

    directory_for_grid_upload = tmp_test_directory / "directory_for_grid_upload"
    with caplog.at_level(logging.INFO):
        local_shower_array_simulator.pack_for_register(str(directory_for_grid_upload))

    assert "Packing output files for registering on the grid" in caplog.text
    assert "Grid output files grid placed in" in caplog.text

    for version in model_versions:
        output_file = file_patterns["output"].format(version)
        shutil.move.assert_any_call(
            output_file,
            directory_for_grid_upload / Path(output_file),
        )


def test_get_seed_for_random_instrument_instances(shower_simulator):
    # Test with a seed provided in the configuration
    shower_simulator.sim_telarray_seeds["seed"] = "12345, 67890"
    seed = shower_simulator._get_seed_for_random_instrument_instances(
        shower_simulator.sim_telarray_seeds["seed"],
        model_version="6.0.1",
        zenith_angle=20.0,
        azimuth_angle=180.0,
    )
    assert seed == 12345

    shower_simulator.sim_telarray_seeds["seed"] = None
    shower_simulator.model_version = "6.0.1"
    shower_simulator.site = "North"
    seed = shower_simulator._get_seed_for_random_instrument_instances(
        shower_simulator.sim_telarray_seeds["seed"],
        model_version="6.0.1",
        zenith_angle=20.0,
        azimuth_angle=180.0,
    )
    assert seed == 600010000000 + 1000000 + 20 * 1000 + 180

    shower_simulator.site = "South"
    seed = shower_simulator._get_seed_for_random_instrument_instances(
        shower_simulator.sim_telarray_seeds["seed"],
        model_version="6.0.1",
        zenith_angle=20.0,
        azimuth_angle=180.0,
    )
    assert seed == 600010000000 + 2000000 + 20 * 1000 + 180


def test_initialize_simulation_runner_with_corsika(shower_simulator):
    simulation_runner = shower_simulator._initialize_simulation_runner()
    assert simulation_runner is not None


def test_initialize_simulation_runner_with_sim_telarray(array_simulator):
    simulation_runner = array_simulator._initialize_simulation_runner()
    assert simulation_runner is not None


def test_initialize_simulation_runner_with_corsika_sim_telarray(
    shower_array_simulator,
):
    simulation_runner = shower_array_simulator._initialize_simulation_runner()
    assert simulation_runner is not None


def test_initialize_simulation_runner_with_calibration_simulator(
    calibration_simulator,
):
    simulation_runner = calibration_simulator._initialize_simulation_runner()
    assert simulation_runner is not None


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
        "get_files",
        side_effect=lambda file_type: mock_output_files
        if file_type == "sim_telarray_output"
        else mock_event_data_files,
    )

    mock_generator = mocker.MagicMock()
    mock_simtel_io_writer = mocker.patch(
        "simtools.sim_events.writer.EventDataWriter", return_value=mock_generator
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
    mocker.patch.object(array_simulator, "get_files", return_value=[])
    mock_simtel_io_writer = mocker.patch("simtools.sim_events.writer.EventDataWriter")
    mock_io_table_handler = mocker.patch("simtools.simulator.table_handler")

    array_simulator.save_reduced_event_lists()

    mock_simtel_io_writer.assert_not_called()
    mock_io_table_handler.write_tables.assert_not_called()


def test_get_calibration_device_types():
    assert Simulator._get_calibration_device_types("direct_injection") == ["flat_fielding"]
    assert Simulator._get_calibration_device_types("what_ever") == []


def test_validate_metadata(array_simulator, mocker, caplog):
    """Test validate_metadata method for complete coverage."""

    # Test case 1: simulation_software does not contain "sim_telarray"
    array_simulator.simulation_software = "corsika"
    with caplog.at_level(logging.INFO):
        array_simulator.validate_metadata()
    assert "No sim_telarray files to validate." in caplog.text
    caplog.clear()

    # Test case 2: simulation_software contains "sim_telarray" with files
    array_simulator.simulation_software = "sim_telarray"
    mock_files = ["output_file_6.0.2_simtel.zst", "output_file_6.0.1_simtel.zst"]
    mocker.patch.object(array_simulator, "get_files", return_value=mock_files)
    mocker.patch("simtools.simulator.assert_sim_telarray_metadata")

    mock_model1 = mocker.Mock()
    mock_model1.model_version = "6.0.2"
    mock_model2 = mocker.Mock()
    mock_model2.model_version = "6.0.0"  # No matching file
    array_simulator.array_models = [mock_model1, mock_model2]

    with caplog.at_level(logging.INFO):
        array_simulator.validate_metadata()

    assert "Validating metadata for output_file_6.0.2_simtel.zst" in caplog.text
    assert "Metadata for sim_telarray file output_file_6.0.2_simtel.zst is valid." in caplog.text
    assert "No sim_telarray file found for model version 6.0.0" in caplog.text


def test_report(array_simulator, mocker, caplog):
    """Test report method for complete coverage."""

    mock_corsika_config = mocker.Mock()
    mock_corsika_config.primary_particle = "gamma"
    mock_corsika_config.azimuth_angle = 180.0
    mock_corsika_config.zenith_angle = 20.0
    mocker.patch.object(
        array_simulator, "_get_first_corsika_config", return_value=mock_corsika_config
    )

    mock_resources_report = "Mean wall time/run [sec]: 123.45, #events/run: 1000"
    mocker.patch.object(
        array_simulator, "_make_resources_report", return_value=mock_resources_report
    )

    array_simulator.site = "North"
    array_simulator.model_version = "6.0.2"
    array_simulator.simulation_software = "sim_telarray"

    with caplog.at_level(logging.INFO):
        array_simulator.report()

    expected_production_msg = (
        "Production run complete for primary gamma showers "
        "from 180.0 azimuth and 20.0 zenith "
        "at North site, using 6.0.2 model."
    )
    assert expected_production_msg in caplog.text

    expected_computing_msg = f"Computing for sim_telarray Simulations: {mock_resources_report}"
    assert expected_computing_msg in caplog.text


def test_make_resources_report(array_simulator, mocker):
    """Test _make_resources_report method for complete coverage."""
    # Test case 1: With runtime and n_events > 0
    mock_sub_out_files = ["sub_out_file.txt"]
    mocker.patch.object(array_simulator, "get_files", return_value=mock_sub_out_files)

    mock_resources = {"runtime": 123.45, "n_events": 1000}
    mocker.patch.object(
        array_simulator._simulation_runner, "get_resources", return_value=mock_resources
    )

    result = array_simulator._make_resources_report()
    expected = "Mean wall time/run [sec]: 123.45, #events/run: 1000"
    assert result == expected

    # Test case 2: With runtime but n_events <= 0
    mock_resources = {"runtime": 67.89, "n_events": 0}
    mocker.patch.object(
        array_simulator._simulation_runner, "get_resources", return_value=mock_resources
    )

    result = array_simulator._make_resources_report()
    expected = "Mean wall time/run [sec]: 67.89"
    assert result == expected

    # Test case 3: With runtime but no n_events key
    mock_resources = {"runtime": 99.99}
    mocker.patch.object(
        array_simulator._simulation_runner, "get_resources", return_value=mock_resources
    )

    result = array_simulator._make_resources_report()
    expected = "Mean wall time/run [sec]: 99.99"
    assert result == expected

    # Test case 4: No runtime available (empty runtime list)
    mock_resources = {"n_events": 500}
    mocker.patch.object(
        array_simulator._simulation_runner, "get_resources", return_value=mock_resources
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = array_simulator._make_resources_report()
        # np.mean([]) returns nan
        assert "Mean wall time/run [sec]: nan" in result
        assert ", #events/run: 500" in result


def test_get_corsika_file(array_simulator, mocker):
    """Test the _get_corsika_file method for different simulation software types."""
    # Mock settings.config to return various corsika file scenarios
    mock_config = mocker.Mock()
    mocker.patch("simtools.settings.config", mock_config)

    # Test case 1: sim_telarray with corsika_file in args
    array_simulator.simulation_software = "sim_telarray"
    mock_config.args.get.return_value = "/path/to/corsika_file.corsika"

    result = array_simulator._get_corsika_file()
    assert result == "/path/to/corsika_file.corsika"
    mock_config.args.get.assert_called_with("corsika_file", None)

    # Test case 2: sim_telarray without corsika_file in args
    mock_config.args.get.return_value = None

    result = array_simulator._get_corsika_file()
    assert result is None

    # Test case 3: Non-sim_telarray simulation software
    array_simulator.simulation_software = "corsika"

    result = array_simulator._get_corsika_file()
    assert result is None

    # Test case 4: corsika_sim_telarray simulation software
    array_simulator.simulation_software = "corsika_sim_telarray"

    result = array_simulator._get_corsika_file()
    assert result is None


def test_simulate(array_simulator, mocker):
    """Test the simulate method orchestrating the simulation process."""
    mock_simulation_runner = mocker.Mock()
    array_simulator._simulation_runner = mock_simulation_runner

    mock_runner_service = mocker.Mock()
    array_simulator.runner_service = mock_runner_service

    array_simulator.run_number = 42
    array_simulator._extra_commands = ["echo test"]

    mock_runner_service.get_file_name.side_effect = lambda file_type, run_num: {
        "sub_script": f"script_{run_num}.sh",
        "sub_out": f"output_{run_num}.out",
    }[file_type]

    mocker.patch.object(array_simulator, "_get_corsika_file", return_value="/path/to/corsika.file")
    mocker.patch.object(array_simulator, "update_file_lists")

    mock_submit = mocker.patch("simtools.job_execution.job_manager.submit")

    array_simulator.simulate()

    # Verify the simulation runner prepared the run
    mock_simulation_runner.prepare_run.assert_called_once_with(
        run_number=42,
        corsika_file="/path/to/corsika.file",
        sub_script="script_42.sh",
        extra_commands=["echo test"],
    )

    # Verify the job manager submitted the job
    mock_submit.assert_called_once_with(
        command="script_42.sh",
        out_file="output_42.out",
        err_file="output_42.out",
        env={"SIM_TELARRAY_CONFIG_PATH": ""},
    )


def test_save_file_lists(array_simulator, mocker, tmp_path, caplog):
    """Test the save_file_lists method for saving various file types to text files."""
    mock_io_handler = mocker.Mock()
    array_simulator.io_handler = mock_io_handler
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    mock_io_handler.get_output_directory.return_value = output_dir

    # Test case 1: Mixed file types with some files, some empty, some None
    array_simulator.file_list = {
        "simtel_output": ["/path/to/file1.simtel.gz", "/path/to/file2.simtel.gz"],
        "log": ["/path/to/logfile.log"],
        "corsika_log": [],  # Empty list
        "histogram": [None, "/path/to/hist.hist"],  # Contains None
        "empty_type": [],  # Empty
    }

    with caplog.at_level(logging.DEBUG):
        array_simulator.save_file_lists()

    simtel_file = output_dir / "simtel_output_files.txt"
    assert simtel_file.exists()
    content = simtel_file.read_text(encoding="utf-8")
    assert "/path/to/file1.simtel.gz\n/path/to/file2.simtel.gz\n" == content

    log_file = output_dir / "log_files.txt"
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "/path/to/logfile.log\n" == content

    corsika_log_file = output_dir / "corsika_log_files.txt"
    assert not corsika_log_file.exists()

    histogram_file = output_dir / "histogram_files.txt"
    assert not histogram_file.exists()

    empty_type_file = output_dir / "empty_type_files.txt"
    assert not empty_type_file.exists()

    assert "Saving list of simtel_output files to" in caplog.text
    assert "Saving list of log files to" in caplog.text
    assert "No files to save for corsika_log files." in caplog.text
    assert "No files to save for histogram files." in caplog.text
    assert "No files to save for empty_type files." in caplog.text

    # Test case 2: All files are None or empty
    caplog.clear()
    array_simulator.file_list = {"all_none": [None, None], "empty_list": [], "mixed_none": [None]}

    with caplog.at_level(logging.DEBUG):
        array_simulator.save_file_lists()

    all_none_file = output_dir / "all_none_files.txt"
    assert not all_none_file.exists()

    empty_list_file = output_dir / "empty_list_files.txt"
    assert not empty_list_file.exists()

    mixed_none_file = output_dir / "mixed_none_files.txt"
    assert not mixed_none_file.exists()

    assert "No files to save for all_none files." in caplog.text
    assert "No files to save for empty_list files." in caplog.text
    assert "No files to save for mixed_none files." in caplog.text

    # Test case 3: Valid files with Path objects (should convert to strings)

    caplog.clear()
    array_simulator.file_list = {
        "path_objects": [Path("/path/to/file1.path"), Path("/path/to/file2.path")]
    }

    with caplog.at_level(logging.INFO):
        array_simulator.save_file_lists()

    path_objects_file = output_dir / "path_objects_files.txt"
    assert path_objects_file.exists()
    content = path_objects_file.read_text(encoding="utf-8")
    assert "/path/to/file1.path\n/path/to/file2.path\n" == content
    assert "Saving list of path_objects files to" in caplog.text


def test_verify_simulated_events_in_reduced_event_lists(shower_simulator, mocker):
    mock_config = mocker.Mock()
    mock_config.args = {"save_reduced_event_lists": True}
    mocker.patch("simtools.settings.config", mock_config)

    mock_file_list = ["event_data_file1.hdf5", "event_data_file2.hdf5"]
    mocker.patch.object(shower_simulator, "get_files", return_value=mock_file_list)

    mock_tables = {"SHOWERS": [{"event_id": 1}, {"event_id": 2}, {"event_id": 3}]}
    mocker.patch("simtools.simulator.table_handler.read_tables", return_value=mock_tables)

    shower_simulator._verify_simulated_events_in_reduced_event_lists(expected_mc_events=3)


def test_verify_simulated_events_in_reduced_event_lists_mismatch(shower_simulator, mocker):
    mock_file_list = ["event_data_file1.hdf5"]
    mocker.patch.object(shower_simulator, "get_files", return_value=mock_file_list)

    mock_tables = {"SHOWERS": [{"event_id": 1}, {"event_id": 2}]}
    mocker.patch("simtools.simulator.table_handler.read_tables", return_value=mock_tables)

    with pytest.raises(ValueError, match="Inconsistent event counts found in reduced event lists"):
        shower_simulator._verify_simulated_events_in_reduced_event_lists(expected_mc_events=5)


def test_verify_simulated_events_in_reduced_event_lists_missing_table(shower_simulator, mocker):
    mock_file_list = ["event_data_file1.hdf5"]
    mocker.patch.object(shower_simulator, "get_files", return_value=mock_file_list)

    mock_tables = {"OTHER_TABLE": []}
    mocker.patch("simtools.simulator.table_handler.read_tables", return_value=mock_tables)

    with pytest.raises(ValueError, match="SHOWERS table not found"):
        shower_simulator._verify_simulated_events_in_reduced_event_lists(expected_mc_events=3)


def test_verify_simulated_events_in_sim_telarray(shower_array_simulator, mocker):
    mock_file_list = ["output_file1.simtel.zst", "output_file2.simtel.zst"]
    mocker.patch.object(shower_array_simulator, "get_files", return_value=mock_file_list)

    mocker.patch("simtools.simulator.file_info.get_simulated_events", return_value=(100, 500))

    shower_array_simulator._verify_simulated_events_in_sim_telarray(
        expected_shower_events=100, expected_mc_events=500
    )


def test_verify_simulated_events_in_sim_telarray_shower_mismatch(shower_array_simulator, mocker):
    mock_file_list = ["output_file1.simtel.zst"]
    mocker.patch.object(shower_array_simulator, "get_files", return_value=mock_file_list)

    mocker.patch("simtools.simulator.file_info.get_simulated_events", return_value=(80, 500))

    with pytest.raises(ValueError, match="Inconsistent event counts found"):
        shower_array_simulator._verify_simulated_events_in_sim_telarray(
            expected_shower_events=100, expected_mc_events=500
        )


def test_verify_simulated_events_in_sim_telarray_mc_mismatch(shower_array_simulator, mocker):
    mock_file_list = ["output_file1.simtel.zst"]
    mocker.patch.object(shower_array_simulator, "get_files", return_value=mock_file_list)

    mocker.patch("simtools.simulator.file_info.get_simulated_events", return_value=(100, 400))

    with pytest.raises(ValueError, match="Inconsistent event counts found"):
        shower_array_simulator._verify_simulated_events_in_sim_telarray(
            expected_shower_events=100, expected_mc_events=500
        )


def test_verify_simulations(shower_array_simulator, mocker):
    mock_config = mocker.Mock()
    mock_config.args = {"nshow": 100, "core_scatter": [5]}
    mocker.patch("simtools.settings.config", mock_config)

    mock_corsika_config = mocker.MagicMock()
    mock_corsika_config.shower_events = 100
    mock_corsika_config.mc_events = 500
    shower_array_simulator.corsika_configurations = [mock_corsika_config]

    mock_verify_simtel = mocker.patch.object(
        shower_array_simulator, "_verify_simulated_events_in_sim_telarray"
    )

    shower_array_simulator.verify_simulations()

    mock_verify_simtel.assert_called_once_with(100, 500)


def test_verify_simulations_with_reduced_event_lists(shower_array_simulator, mocker):
    mock_config = mocker.Mock()
    mock_config.args = {"nshow": 100, "core_scatter": [5], "save_reduced_event_lists": True}
    mocker.patch("simtools.settings.config", mock_config)

    mock_corsika_config = mocker.MagicMock()
    mock_corsika_config.shower_events = 100
    mock_corsika_config.mc_events = 500
    shower_array_simulator.corsika_configurations = [mock_corsika_config]

    mocker.patch.object(shower_array_simulator, "_verify_simulated_events_in_sim_telarray")
    mock_verify_reduced = mocker.patch.object(
        shower_array_simulator, "_verify_simulated_events_in_reduced_event_lists"
    )

    shower_array_simulator.verify_simulations()

    mock_verify_reduced.assert_called_once_with(500)


def test_verify_simulated_events_corsika(shower_simulator, mocker):
    mock_file_list = ["corsika_output_file1.zst", "corsika_output_file2.zst"]
    mocker.patch.object(shower_simulator, "get_files", return_value=mock_file_list)
    mocker.patch("simtools.simulator.file_info.get_simulated_events", return_value=(100, 0))

    shower_simulator._verify_simulated_events_corsika(expected_mc_events=100)


def test_verify_simulated_events_corsika_mismatch(shower_simulator, mocker):
    mock_file_list = ["corsika_output_file1.zst"]
    mocker.patch.object(shower_simulator, "get_files", return_value=mock_file_list)
    mocker.patch("simtools.simulator.file_info.get_simulated_events", return_value=(80, 0))

    with pytest.raises(ValueError, match="Inconsistent event counts found in CORSIKA output"):
        shower_simulator._verify_simulated_events_corsika(expected_mc_events=100)


def test_verify_simulated_events_corsika_tolerance(shower_simulator, mocker, caplog):
    mock_file_list = ["corsika_output_file1.zst"]
    mocker.patch.object(shower_simulator, "get_files", return_value=mock_file_list)
    mocker.patch("simtools.simulator.file_info.get_simulated_events", return_value=(9999, 0))

    with caplog.at_level(logging.WARNING):
        shower_simulator._verify_simulated_events_corsika(expected_mc_events=10000, tolerance=0.001)
    assert "Small mismatch in number of events" in caplog.text


def test_verify_simulations_corsika(shower_simulator, mocker):
    mock_config = mocker.Mock()
    mock_config.args = {"nshow": 100, "core_scatter": [5]}
    mocker.patch("simtools.settings.config", mock_config)

    mock_corsika_config = mocker.MagicMock()
    mock_corsika_config.shower_events = 100
    mock_corsika_config.mc_events = 500
    shower_simulator.corsika_configurations = mock_corsika_config

    mock_verify_corsika = mocker.patch.object(shower_simulator, "_verify_simulated_events_corsika")

    shower_simulator.verify_simulations()

    mock_verify_corsika.assert_called_once_with(500)


def test_get_seed_for_random_instrument_instances_with_unknown_site(shower_simulator):
    shower_simulator.sim_telarray_seeds["seed"] = None
    shower_simulator.site = "UnknownSite"
    seed = shower_simulator._get_seed_for_random_instrument_instances(
        shower_simulator.sim_telarray_seeds["seed"],
        model_version="6.0.1",
        zenith_angle=20.0,
        azimuth_angle=180.0,
    )
    assert seed == 600010000000 + 1000000 + 20 * 1000 + 180


def test_get_first_corsika_config_error(shower_simulator):
    shower_simulator.corsika_configurations = []
    with pytest.raises(ValueError, match="CORSIKA configuration not found for verification"):
        shower_simulator._get_first_corsika_config()
