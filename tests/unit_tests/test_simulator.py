#!/usr/bin/python3

import copy
import gzip
import logging
import shutil
import tarfile
from pathlib import Path
from unittest import mock

import pytest

from simtools.io import eventio_handler
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

    return _apply


@pytest.fixture
def configure_runner_mock(io_handler):
    """Configure a runner patch with common behaviors used in tests."""

    def _configure(runner_patch, add_resources=False):
        runner_patch.return_value.prepare_run_script.return_value = str(
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

    patch_simulator_core()
    mock_runner = mocker.patch("simtools.simulator.SimulatorArray")
    configure_runner_mock(mock_runner)

    return Simulator(
        label=args_dict["label"],
        args_dict=args_dict,
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

    patch_simulator_core()
    mock_runner = mocker.patch("simtools.simulator.CorsikaRunner")
    configure_runner_mock(mock_runner, add_resources=True)

    return Simulator(
        label=args_dict["label"],
        args_dict=args_dict,
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

    patch_simulator_core()
    mock_runner = mocker.patch("simtools.simulator.CorsikaSimtelRunner")
    configure_runner_mock(mock_runner)

    return Simulator(
        label=args_dict["label"],
        args_dict=args_dict,
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

    patch_simulator_core()
    mock_runner = mocker.patch("simtools.simulator.CorsikaSimtelRunner")
    configure_runner_mock(mock_runner)

    return Simulator(
        label=args_dict["label"],
        args_dict=args_dict,
    )


def test_init_simulator(shower_simulator, array_simulator, shower_array_simulator):
    assert shower_simulator._simulation_runner is not None
    assert shower_array_simulator._simulation_runner is not None
    assert array_simulator._simulation_runner is not None


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


def test_simulate_shower_simulator(shower_simulator, io_handler):
    shower_simulator._test = True
    shower_simulator.simulate()
    assert len(shower_simulator._results["simtel_output"]) > 0
    assert len(shower_simulator._results["sub_out"]) > 0
    run_script = shower_simulator._simulation_runner.prepare_run_script(run_number=2)
    Path(run_script).parent.mkdir(parents=True, exist_ok=True)
    Path(run_script).touch()
    assert Path(run_script).exists()


def test_simulate_array_simulator(array_simulator):
    array_simulator._test = True
    array_simulator.simulate()

    assert len(array_simulator._results["simtel_output"]) > 0
    assert len(array_simulator._results["sub_out"]) > 0


def test_simulate_shower_array_simulator(shower_array_simulator):
    shower_array_simulator._test = True
    shower_array_simulator.simulate()

    assert len(shower_array_simulator._results["simtel_output"]) > 0
    assert len(shower_array_simulator._results["sub_out"]) > 0


def test_fill_list_of_generated_files(array_simulator, shower_simulator, shower_array_simulator):
    for simulator_now in [array_simulator, shower_array_simulator]:
        simulator_now._fill_list_of_generated_files()
        assert len(simulator_now.get_file_list("simtel_output")) == 1
        assert len(simulator_now._results["sub_out"]) == 1
        assert len(simulator_now.get_file_list("log")) == 1
        assert len(simulator_now.get_file_list("histogram")) == 1

    shower_simulator._fill_list_of_generated_files()
    assert len(shower_simulator.get_file_list("simtel_output")) == 1
    assert len(shower_simulator.get_file_list("corsika_log")) == 1
    assert len(shower_simulator.get_file_list("histogram")) == 0


def test_get_list_of_files(shower_simulator):
    test_shower_simulator = copy.deepcopy(shower_simulator)
    test_shower_simulator._results["simtel_output"] = ["file_name"]
    assert len(test_shower_simulator.get_file_list("simtel_output")) == 1
    assert len(test_shower_simulator.get_file_list("not_a_valid_file_type")) == 0


def test_report(shower_simulator, caplog):
    with caplog.at_level(logging.INFO):
        shower_simulator.report(input_file_list=None)

    assert "Mean wall time/run [sec]: np.nan" in caplog.text


def test_make_resources_report(shower_simulator):
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
    test_shower_simulator.run_number = 1
    _resources_1 = test_shower_simulator._make_resources_report(input_file_list=log_file_name)
    assert "Mean wall time/run [sec]: 6" in _resources_1

    test_shower_simulator.run_number = 4


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
    mocker.patch("tarfile.open")  # NOSONAR
    mocker.patch("pathlib.Path.exists", return_value=True)
    mocker.patch("pathlib.Path.is_file", return_value=True)

    directory_for_grid_upload = tmp_test_directory / "directory_for_grid_upload"
    with caplog.at_level(logging.INFO):
        array_simulator.pack_for_register(str(directory_for_grid_upload))

    assert "Overwriting existing file" in caplog.text
    assert "Packing the output files for registering on the grid" in caplog.text
    assert "Output files for the grid placed in" in caplog.text
    tarfile.open.assert_called_once()  # NOSONAR
    shutil.move.assert_any_call(
        Path(f"output_file_{model_version}_simtel.zst"),
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
    shower_simulator.args_dict["corsika_file"] = corsika_file
    mocker.patch("simtools.io.eventio_handler.get_corsika_run_number", return_value=42)

    shower_simulator._initialize_from_tool_configuration()
    assert shower_simulator.run_number == 42
    eventio_handler.get_corsika_run_number.assert_called_once_with(corsika_file)


def test_run_number_from_corsika_file_missing(shower_simulator, io_handler):
    # Test the KeyError when corsika_file is not in args_dict
    shower_simulator.args_dict.pop("corsika_file", None)  # Ensure key is not present
    with pytest.raises(KeyError, match="corsika_file"):
        shower_simulator.run_number = eventio_handler.get_corsika_run_number(
            shower_simulator.args_dict["corsika_file"]
        )


def test_initialize_array_models_with_multiple_versions(shower_simulator, mocker):
    model_versions = ["5.0.0", "6.0.1"]
    mock_models = []
    for version in model_versions:
        m = mocker.MagicMock()
        m.model_version = version
        mock_models.append(m)

    mocker.patch("simtools.simulator.ArrayModel", side_effect=mock_models)
    shower_simulator.args_dict["model_version"] = model_versions
    shower_simulator.model_version = model_versions
    array_models, _ = shower_simulator._initialize_array_models()
    assert len(array_models) == 2
    for i, model_version in enumerate(model_versions):
        assert array_models[i] is not None
        assert array_models[i] == mock_models[i]


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
    io_handler, simulations_args_dict, mocker, caplog, tmp_test_directory, model_version
):
    args_dict = copy.deepcopy(simulations_args_dict)
    args_dict["simulation_software"] = "corsika_sim_telarray"
    args_dict["label"] = "local-test-shower-array-simulator"
    model_versions = ["5.0.0", "6.0.1"]
    args_dict["model_version"] = model_versions

    # Create mock array models for each version
    mock_array_models = []
    for version in model_versions:
        mock_model = mocker.MagicMock()
        mock_model.model_version = version
        mock_model.pack_model_files.return_value = []
        mock_array_models.append(mock_model)

    mocker.patch("simtools.simulator.ArrayModel", side_effect=mock_array_models)
    mocker.patch("simtools.simulator.CorsikaConfig")
    mocker.patch("simtools.simulator.CorsikaSimtelRunner")

    local_shower_array_simulator = Simulator(label=args_dict["label"], args_dict=args_dict)

    # Define file patterns
    file_patterns = {
        "output": "output_file_{}_simtel.zst",
        "log": "log_file_{}_simtel.log.gz",
        "corsika_log": "log_file_corsika_{}.log.gz",
        "histogram": "hist_file_{}_hist_log.zst",
    }

    # Generate file lists for side effects
    side_effects = [
        [file_patterns["output"].format(v) for v in model_versions],
        [file_patterns["log"].format(v) for v in model_versions],
        [file_patterns["corsika_log"].format(model_versions[0])],
        [file_patterns["histogram"].format(v) for v in model_versions],
    ]

    # Mocking methods and objects
    mocker.patch.object(local_shower_array_simulator, "get_file_list", side_effect=side_effects)
    mocker.patch("shutil.move")

    # Create a mock for tarfile.open that returns a mock context manager
    mock_tar = mocker.MagicMock()
    mock_tar_cm = mocker.MagicMock()
    mock_tar_cm.__enter__ = mocker.MagicMock(return_value=mock_tar)
    mock_tar_cm.__exit__ = mocker.MagicMock(return_value=None)
    mock_tarfile_open = mocker.patch("tarfile.open", return_value=mock_tar_cm)  # NOSONAR

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
        for file_type in ["log", "histogram"]:
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
        shower_simulator.sim_telarray_seeds["seed"],
        model_version="6.0.1",
        zenith_angle=20.0,
        azimuth_angle=180.0,
    )
    assert seed == 12345

    # Test without a seed provided in the configuration
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
    assert Simulator._is_calibration_run("pedestals_nsb_only") is True
    assert Simulator._is_calibration_run(None) is False
    assert Simulator._is_calibration_run("not_a_calibration_run") is False


def test_get_calibration_device_types():
    assert Simulator._get_calibration_device_types("direct_injection") == ["flat_fielding"]
    assert Simulator._get_calibration_device_types("what_ever") == []


def test_verify_simulated_events_in_reduced_event_lists(shower_simulator, mocker):
    shower_simulator.args_dict["save_reduced_event_lists"] = True

    mock_file_list = ["event_data_file1.hdf5", "event_data_file2.hdf5"]
    mocker.patch.object(shower_simulator, "get_file_list", return_value=mock_file_list)

    mock_tables = {"SHOWERS": [{"event_id": 1}, {"event_id": 2}, {"event_id": 3}]}
    mocker.patch("simtools.simulator.table_handler.read_tables", return_value=mock_tables)

    shower_simulator._verify_simulated_events_in_reduced_event_lists(expected_mc_events=3)


def test_verify_simulated_events_in_reduced_event_lists_mismatch(shower_simulator, mocker):
    mock_file_list = ["event_data_file1.hdf5"]
    mocker.patch.object(shower_simulator, "get_file_list", return_value=mock_file_list)

    mock_tables = {"SHOWERS": [{"event_id": 1}, {"event_id": 2}]}
    mocker.patch("simtools.simulator.table_handler.read_tables", return_value=mock_tables)

    with pytest.raises(ValueError, match="Inconsistent event counts found in reduced event lists"):
        shower_simulator._verify_simulated_events_in_reduced_event_lists(expected_mc_events=5)


def test_verify_simulated_events_in_reduced_event_lists_missing_table(shower_simulator, mocker):
    mock_file_list = ["event_data_file1.hdf5"]
    mocker.patch.object(shower_simulator, "get_file_list", return_value=mock_file_list)

    mock_tables = {"OTHER_TABLE": []}
    mocker.patch("simtools.simulator.table_handler.read_tables", return_value=mock_tables)

    with pytest.raises(ValueError, match="SHOWERS table not found"):
        shower_simulator._verify_simulated_events_in_reduced_event_lists(expected_mc_events=3)


def test_verify_simulated_events_in_sim_telarray(shower_array_simulator, mocker):
    mock_file_list = ["output_file1.simtel.zst", "output_file2.simtel.zst"]
    mocker.patch.object(shower_array_simulator, "get_file_list", return_value=mock_file_list)

    mocker.patch("simtools.simulator.eventio_handler.get_simulated_events", return_value=(100, 500))

    shower_array_simulator._verify_simulated_events_in_sim_telarray(
        expected_shower_events=100, expected_mc_events=500
    )


def test_verify_simulated_events_in_sim_telarray_shower_mismatch(shower_array_simulator, mocker):
    mock_file_list = ["output_file1.simtel.zst"]
    mocker.patch.object(shower_array_simulator, "get_file_list", return_value=mock_file_list)

    mocker.patch("simtools.simulator.eventio_handler.get_simulated_events", return_value=(80, 500))

    with pytest.raises(ValueError, match="Inconsistent event counts found"):
        shower_array_simulator._verify_simulated_events_in_sim_telarray(
            expected_shower_events=100, expected_mc_events=500
        )


def test_verify_simulated_events_in_sim_telarray_mc_mismatch(shower_array_simulator, mocker):
    mock_file_list = ["output_file1.simtel.zst"]
    mocker.patch.object(shower_array_simulator, "get_file_list", return_value=mock_file_list)

    mocker.patch("simtools.simulator.eventio_handler.get_simulated_events", return_value=(100, 400))

    with pytest.raises(ValueError, match="Inconsistent event counts found"):
        shower_array_simulator._verify_simulated_events_in_sim_telarray(
            expected_shower_events=100, expected_mc_events=500
        )


def test_verify_simulations(shower_array_simulator, mocker):
    shower_array_simulator.args_dict["nshow"] = 100
    shower_array_simulator.args_dict["core_scatter"] = [5]

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
    shower_array_simulator.args_dict["nshow"] = 100
    shower_array_simulator.args_dict["core_scatter"] = [5]
    shower_array_simulator.args_dict["save_reduced_event_lists"] = True

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
    mocker.patch.object(shower_simulator, "get_file_list", return_value=mock_file_list)
    mocker.patch("simtools.simulator.eventio_handler.get_simulated_events", return_value=(100, 0))

    shower_simulator._verify_simulated_events_corsika(expected_mc_events=100)


def test_verify_simulated_events_corsika_mismatch(shower_simulator, mocker):
    mock_file_list = ["corsika_output_file1.zst"]
    mocker.patch.object(shower_simulator, "get_file_list", return_value=mock_file_list)
    mocker.patch("simtools.simulator.eventio_handler.get_simulated_events", return_value=(80, 0))

    with pytest.raises(ValueError, match="Inconsistent event counts found in CORSIKA output"):
        shower_simulator._verify_simulated_events_corsika(expected_mc_events=100)


def test_verify_simulated_events_corsika_tolerance(shower_simulator, mocker, caplog):
    mock_file_list = ["corsika_output_file1.zst"]
    mocker.patch.object(shower_simulator, "get_file_list", return_value=mock_file_list)
    mocker.patch("simtools.simulator.eventio_handler.get_simulated_events", return_value=(9999, 0))

    with caplog.at_level(logging.WARNING):
        shower_simulator._verify_simulated_events_corsika(expected_mc_events=10000, tolerance=0.001)

    assert "Small mismatch in number of events" in caplog.text


def test_verify_simulations_corsika(shower_simulator, mocker):
    shower_simulator.args_dict["nshow"] = 100
    shower_simulator.args_dict["core_scatter"] = [5]

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
