#!/usr/bin/python3

import copy
import gzip
import logging
import shutil
from pathlib import Path
from unittest import mock
from unittest.mock import call

import pytest
from astropy import units as u

from simtools.corsika.corsika_config import CorsikaConfig
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
    args_dict["showers_per_run"] = 10
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


def test_get_files_returns_empty_list_for_unknown_type(array_simulator):
    assert array_simulator.get_files("unknown_file_type") == []


def test_pack_for_register(array_simulator, mocker, model_version, caplog, tmp_test_directory):
    source_dir = Path(str(tmp_test_directory)) / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    output_file = source_dir / f"output_file_{model_version}_simtel.zst"
    output_file.write_text("output", encoding="utf-8")
    log_file = source_dir / f"log_file_{model_version}_simtel.log.gz"
    with gzip.open(log_file, "wt", encoding="utf-8") as handle:
        handle.write("log")
    histogram_file = source_dir / f"hist_file_{model_version}_hist_log.zst"
    histogram_file.write_text("hist", encoding="utf-8")
    corsika_log_file = source_dir / f"corsika_{model_version}.log.gz"
    with gzip.open(corsika_log_file, "wt", encoding="utf-8") as handle:
        handle.write("corsika")
    model_archive = source_dir / f"model_files_{model_version}.tar.gz"
    model_archive.write_text("model", encoding="utf-8")
    simtools_log_file = source_dir / "simtools.log"
    simtools_log_file.write_text("simtools", encoding="utf-8")

    files_by_type = {
        "sim_telarray_output": [str(output_file)],
        "sim_telarray_log": [str(log_file)],
        "sim_telarray_histogram": [str(histogram_file)],
        "corsika_log": [str(corsika_log_file)],
        "sim_telarray_event_data": [],
    }
    mocker.patch.object(
        array_simulator,
        "get_files",
        side_effect=lambda file_type: files_by_type.get(file_type, []),
    )
    array_simulator.array_models = [mocker.Mock(model_version=model_version)]
    array_simulator.array_models[0].pack_model_files.return_value = str(model_archive)
    mocker.patch(
        "simtools.utils.general.get_simtools_log_file", return_value=str(simtools_log_file)
    )

    directory_for_grid_upload = tmp_test_directory / "directory_for_grid_upload"
    with caplog.at_level(logging.INFO):
        array_simulator.pack_for_register(str(directory_for_grid_upload))

    assert "Packing output files for registering on the grid" in caplog.text
    assert "Grid output files grid placed in" in caplog.text
    with gzip.open(directory_for_grid_upload / log_file.name, "rt", encoding="utf-8") as handle:
        assert handle.read() == "log"
    assert (directory_for_grid_upload / histogram_file.name).read_text(encoding="utf-8") == "hist"
    with gzip.open(
        directory_for_grid_upload / corsika_log_file.name, "rt", encoding="utf-8"
    ) as handle:
        assert handle.read() == "corsika"
    assert (directory_for_grid_upload / model_archive.name).read_text(encoding="utf-8") == "model"
    assert not output_file.exists()
    assert (directory_for_grid_upload / output_file.name).read_text(encoding="utf-8") == "output"
    with gzip.open(directory_for_grid_upload / "simtools.log.gz", "rt", encoding="utf-8") as handle:
        assert handle.read() == "simtools"


def test_initialize_from_tool_configuration_with_corsika_file(shower_simulator, mocker):
    corsika_file = "test_corsika.corsika.gz"

    mock_config = mocker.Mock()
    mock_config.args = {"corsika_file": corsika_file}
    mocker.patch("simtools.settings.config", mock_config)

    mocker.patch("simtools.sim_events.file_info.get_corsika_run_number", return_value=42)

    shower_simulator.run_number = shower_simulator._initialize_from_tool_configuration()
    assert shower_simulator.run_number == 42
    file_info.get_corsika_run_number.assert_called_once_with(corsika_file)


def test_pack_for_register_with_multiple_versions(
    io_handler, simulations_args_dict, mocker, caplog, tmp_test_directory, model_version
):
    args_dict = copy.deepcopy(simulations_args_dict)
    args_dict["simulation_software"] = "corsika_sim_telarray"
    args_dict["label"] = "local-test-shower-array-simulator"
    args_dict["save_corsika_output"] = True
    model_versions = ["5.0.0", "6.0.1"]
    args_dict["model_version"] = model_versions

    mock_array_models = []
    for version in model_versions:
        mock_model = mocker.MagicMock()
        mock_model.model_version = version
        mock_model.pack_model_files.return_value = []
        mock_array_models.append(mock_model)

    mocker.patch("simtools.simulator.ArrayModel", side_effect=mock_array_models)

    mock_corsika_config = mocker.MagicMock(CorsikaConfig, instance=True)
    mock_corsika_config.array_model = mocker.MagicMock()
    mock_corsika_config.get_config_parameter.side_effect = lambda param: {
        "VIEWCONE": [0, 10],
        "THETAP": [20, 20],
    }.get(param, [0, 0])
    mock_corsika_config.azimuth_angle = 0  # from args
    mock_corsika_config.zenith_angle = 20  # from args
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
        "corsika_output": "output_file_corsika_{}.corsika.zst",
        "histogram": "hist_file_{}_hist_log.zst",
    }

    def mock_get_files(file_type):
        if file_type == "sim_telarray_output":
            return [file_patterns["output"].format(v) for v in model_versions]
        if file_type == "sim_telarray_log":
            return [file_patterns["log"].format(v) for v in model_versions]
        if file_type == "corsika_log":
            return [file_patterns["corsika_log"].format(model_versions[0])]
        if file_type == "corsika_output":
            return [file_patterns["corsika_output"].format(model_versions[0])]
        if file_type == "sim_telarray_histogram":
            return [file_patterns["histogram"].format(v) for v in model_versions]
        if file_type == "sim_telarray_event_data":
            return []  # Empty since reduced_event_lists is not set
        return []

    mocker.patch.object(local_shower_array_simulator, "get_files", side_effect=mock_get_files)
    mocker.patch("shutil.move")
    mocker.patch("shutil.copy2")
    mocker.patch("pathlib.Path.is_file", return_value=True)
    mocker.patch("pathlib.Path.exists", return_value=True)
    mocker.patch("simtools.utils.general.get_simtools_log_file", return_value=None)

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
        shutil.copy2.assert_any_call(
            file_patterns["log"].format(version),
            directory_for_grid_upload / Path(file_patterns["log"].format(version)),
        )

    corsika_output = file_patterns["corsika_output"].format(model_versions[0])
    shutil.move.assert_any_call(
        corsika_output,
        directory_for_grid_upload / Path(corsika_output),
    )


@pytest.mark.parametrize(
    ("fixture_name", "expected_software"),
    [
        ("shower_simulator", "corsika"),
        ("array_simulator", "sim_telarray"),
        ("shower_array_simulator", "corsika_sim_telarray"),
        ("calibration_simulator", "corsika_sim_telarray"),
    ],
)
def test_initialize_simulation_runner(fixture_name, expected_software, request):
    """Test simulation runner initialization for different simulation software types."""
    simulator = request.getfixturevalue(fixture_name)
    simulation_runner = simulator._initialize_simulation_runner()
    assert simulation_runner is not None
    assert simulator.simulation_software == expected_software


def test_reduced_event_lists_not_sim_telarray(shower_simulator, caplog):
    with caplog.at_level(logging.WARNING):
        shower_simulator.save_reduced_event_lists()
    assert "Reduced event lists can only be saved for sim_telarray simulations." in caplog.text


def _mock_reduced_event_table_writer(mocker):
    """Mock table writing while preserving the output-file contract."""
    table_handler = mocker.patch("simtools.simulator.table_handler")

    def _touch_output(*_args, **kwargs):
        output_file = Path(kwargs["output_file"])
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.touch()

    table_handler.write_table_chunks.side_effect = _touch_output
    return table_handler


def test_reduced_event_lists_sim_telarray(array_simulator, mocker, tmp_test_directory):
    mock_output_files = ["output_file1.simtel.zst", "output_file2.simtel.zst"]
    mock_event_data_files = [
        tmp_test_directory / "output_file1.reduced_event_data.hdf5",
        tmp_test_directory / "output_file2.reduced_event_data.hdf5",
    ]
    mocker.patch.object(
        array_simulator,
        "get_files",
        side_effect=lambda file_type: (
            mock_output_files if file_type == "sim_telarray_output" else mock_event_data_files
        ),
    )

    mock_generator = mocker.MagicMock()
    mock_simtel_io_writer = mocker.patch(
        "simtools.sim_events.writer.EventDataWriter", return_value=mock_generator
    )
    mock_table_handler = _mock_reduced_event_table_writer(mocker)

    array_simulator.save_reduced_event_lists()

    assert mock_simtel_io_writer.call_count == 2
    mock_simtel_io_writer.assert_any_call(["output_file1.simtel.zst"])
    mock_simtel_io_writer.assert_any_call(["output_file2.simtel.zst"])

    assert mock_table_handler.write_table_chunks.call_count == 2
    output_files = {
        call.kwargs["output_file"] for call in mock_table_handler.write_table_chunks.call_args_list
    }
    assert output_files == {
        Path(tmp_test_directory) / "output_file1.reduced_event_data.hdf5",
        Path(tmp_test_directory) / "output_file2.reduced_event_data.hdf5",
    }
    assert all(
        "metadata_documents" in call.kwargs
        for call in mock_table_handler.write_table_chunks.call_args_list
    )


def test_write_reduced_event_lists_derives_output_files(mocker, tmp_test_directory):
    tmp_base = Path(str(tmp_test_directory))
    data_dir = tmp_base / "data"
    output_dir = tmp_base / "reduced"
    input_files = [
        str(data_dir / "output_file1.simtel.zst"),
        str(data_dir / "output_file2.simtel.gz"),
    ]
    output_path = str(output_dir)

    mock_generator = mocker.MagicMock()
    mock_simtel_io_writer = mocker.patch(
        "simtools.sim_events.writer.EventDataWriter", return_value=mock_generator
    )
    mock_table_handler = _mock_reduced_event_table_writer(mocker)

    Simulator.write_reduced_event_lists(input_files=input_files, output_path=output_path)

    assert mock_simtel_io_writer.call_count == 2
    mock_simtel_io_writer.assert_any_call([str(data_dir / "output_file1.simtel.zst")])
    mock_simtel_io_writer.assert_any_call([str(data_dir / "output_file2.simtel.gz")])

    assert mock_table_handler.write_table_chunks.call_count == 2
    output_files = {
        call.kwargs["output_file"] for call in mock_table_handler.write_table_chunks.call_args_list
    }
    assert output_files == {
        output_dir / "output_file1.reduced_event_data.hdf5",
        output_dir / "output_file2.reduced_event_data.hdf5",
    }


def test_write_reduced_event_lists_passes_activity_id_to_hdf5_writer(mocker, tmp_test_directory):
    """Pass the application activity ID to retained incomplete HDF5 filenames."""
    input_file = Path(tmp_test_directory) / "output_file.simtel.zst"
    activity_id = "019d85b6-1f98-715b-b92b-bfbcd06d7cd8"
    mocker.patch("simtools.sim_events.writer.EventDataWriter", return_value=mocker.MagicMock())
    mock_table_handler = _mock_reduced_event_table_writer(mocker)

    Simulator.write_reduced_event_lists(
        input_files=[input_file],
        metadata_args={"activity_id": activity_id},
    )

    assert mock_table_handler.write_table_chunks.call_args.kwargs["activity_id"] == activity_id


def test_write_reduced_event_lists_derives_output_to_input_directory(mocker, tmp_test_directory):
    data_dir = Path(str(tmp_test_directory)) / "data"
    input_file = str(data_dir / "output_file3.simtel")

    mock_generator = mocker.MagicMock()
    mock_simtel_io_writer = mocker.patch(
        "simtools.sim_events.writer.EventDataWriter", return_value=mock_generator
    )
    mock_table_handler = _mock_reduced_event_table_writer(mocker)

    Simulator.write_reduced_event_lists(input_files=[input_file])

    mock_simtel_io_writer.assert_called_once_with([input_file])
    call = mock_table_handler.write_table_chunks.call_args
    assert call.kwargs["output_file"] == data_dir / "output_file3.reduced_event_data.hdf5"
    assert "metadata_documents" in call.kwargs


def test_write_reduced_event_lists_raises_for_mismatched_explicit_output_files(mocker):
    input_files = ["output_file1.simtel.zst", "output_file2.simtel.zst"]
    output_files = ["output_file1.reduced_event_data.hdf5"]

    mock_simtel_io_writer = mocker.patch("simtools.sim_events.writer.EventDataWriter")
    mock_table_handler = _mock_reduced_event_table_writer(mocker)

    with pytest.raises(ValueError, match="Length mismatch between input_files and output_files"):
        Simulator.write_reduced_event_lists(input_files=input_files, output_files=output_files)

    mock_simtel_io_writer.assert_not_called()
    mock_table_handler.write_table_chunks.assert_not_called()


def test_write_reduced_event_lists_from_file_list_in_batches(mocker, tmp_test_directory):
    tmp_base = Path(str(tmp_test_directory))
    input_file_list = tmp_base / "simtel_files.txt"
    input_files = [f"input_file{index}.simtel.zst" for index in range(1, 6)]
    input_file_list.write_text("\n".join(input_files) + "\n", encoding="utf-8")
    output_dir = tmp_base / "reduced"

    mock_generator = mocker.MagicMock()
    mock_simtel_io_writer = mocker.patch(
        "simtools.sim_events.writer.EventDataWriter", return_value=mock_generator
    )
    mock_table_handler = _mock_reduced_event_table_writer(mocker)

    Simulator.write_reduced_event_lists(
        input_file_list=input_file_list,
        files_per_reduced_event_file=2,
        output_path=output_dir,
    )

    assert mock_simtel_io_writer.call_args_list == [
        mocker.call(input_files[0:2]),
        mocker.call(input_files[2:4]),
        mocker.call(input_files[4:5]),
    ]
    assert [
        call.kwargs["output_file"] for call in mock_table_handler.write_table_chunks.call_args_list
    ] == [
        output_dir / f"simtel_files.part{index:04d}.reduced_event_data.hdf5"
        for index in range(1, 4)
    ]
    assert all(
        "metadata_documents" in call.kwargs
        for call in mock_table_handler.write_table_chunks.call_args_list
    )


def test_write_reduced_event_lists_from_multiple_file_lists(mocker, tmp_test_directory):
    """Process multiple input lists in one execution submission."""
    tmp_base = Path(str(tmp_test_directory))
    first_list = tmp_base / "first.txt"
    second_list = tmp_base / "second.txt"
    first_inputs = ["first_file1.simtel.zst", "first_file2.simtel.zst"]
    second_inputs = ["second_file1.simtel.zst"]
    first_list.write_text("\n".join(first_inputs) + "\n", encoding="utf-8")
    second_list.write_text("\n".join(second_inputs) + "\n", encoding="utf-8")
    output_dir = tmp_base / "reduced"

    mock_generator = mocker.MagicMock()
    mock_simtel_io_writer = mocker.patch(
        "simtools.sim_events.writer.EventDataWriter", return_value=mock_generator
    )
    mock_table_handler = _mock_reduced_event_table_writer(mocker)

    Simulator.write_reduced_event_lists(
        input_file_lists=[first_list, second_list],
        files_per_reduced_event_file=2,
        output_path=output_dir,
    )

    assert mock_simtel_io_writer.call_args_list == [
        mocker.call(first_inputs),
        mocker.call(second_inputs),
    ]
    assert [
        call.kwargs["output_file"] for call in mock_table_handler.write_table_chunks.call_args_list
    ] == [
        output_dir / "first.part0001.reduced_event_data.hdf5",
        output_dir / "second.part0001.reduced_event_data.hdf5",
    ]


def test_write_reduced_event_lists_parallelizes_output_batches(mocker):
    """Execute independent output batches through the shared execution facade."""
    mock_execute = mocker.patch("simtools.simulator.execute_jobs")

    Simulator.write_reduced_event_lists(
        input_files=["input1.simtel.zst", "input2.simtel.zst"],
        max_workers=2,
    )

    mock_execute.assert_called_once()
    jobs, options = mock_execute.call_args.args
    assert options.max_workers == 2
    assert len(jobs) == 2
    assert all(job.output_paths for job in jobs)


def test_write_reduced_event_lists_submits_htcondor_without_waiting(mocker, tmp_test_directory):
    """HTCondor reduced-event batches can be submitted without waiting."""
    mock_submit = mocker.patch("simtools.simulator.submit_jobs")
    output_files = [
        Path(tmp_test_directory) / "part0001.hdf5",
        Path(tmp_test_directory) / "part0002.hdf5",
    ]

    Simulator.write_reduced_event_lists(
        input_files=["input1.simtel.zst", "input2.simtel.zst"],
        output_files=output_files,
        backend="htcondor",
        wait_for_completion=False,
    )

    mock_submit.assert_called_once()
    assert len(mock_submit.call_args.args[0]) == 2


@pytest.mark.parametrize("files_per_reduced_event_file", [0, -1])
def test_write_reduced_event_lists_rejects_invalid_batch_size(
    files_per_reduced_event_file,
):
    with pytest.raises(ValueError, match="must be greater than zero"):
        Simulator.write_reduced_event_lists(
            input_files=["input.simtel.zst"],
            files_per_reduced_event_file=files_per_reduced_event_file,
        )


@pytest.mark.parametrize(
    ("run_mode", "expected_devices"),
    [
        ("direct_injection", ["flat_fielding"]),
        ("pedestals_nsb_only", []),
        ("what_ever", []),
        (None, []),
    ],
)
def test_get_calibration_device_types(run_mode, expected_devices):
    assert Simulator._get_calibration_device_types(run_mode) == expected_devices


def test_overwrite_flasher_photons_for_direct_injection(mocker):
    simulator = Simulator.__new__(Simulator)
    simulator.run_mode = "direct_injection"
    simulator.logger = mocker.Mock()

    calib_1 = mocker.Mock()
    calib_2 = mocker.Mock()
    array_model = mocker.Mock()
    array_model.calibration_models = {
        "TEL01": {"CAL01": calib_1},
        "TEL02": {"CAL02": calib_2},
    }
    simulator.array_models = [array_model]

    mock_settings = mocker.Mock()
    mock_settings.config.args = {"flasher_photons": 1234567}
    mocker.patch("simtools.simulator.settings", mock_settings)

    for calib in (calib_1, calib_2):
        calib.name = "CAL"
        calib.site = "North"
        calib.parameters = {
            "flasher_photons_at_pixel": {
                "value": 100000,
                "parameter": "flasher_photons_at_pixel",
            },
        }
        calib.get_parameter_value.side_effect = [100000, 1234567]

    simulator._overwrite_flasher_photons_for_direct_injection()

    expected_calls = [call("flasher_photons_at_pixel", 1234567)]
    assert calib_1.overwrite_model_parameter.call_args_list == expected_calls
    assert calib_2.overwrite_model_parameter.call_args_list == expected_calls


def test_overwrite_flasher_photons_for_direct_injection_noop_without_value(mocker):
    simulator = Simulator.__new__(Simulator)
    simulator.run_mode = "direct_injection"
    simulator.logger = mocker.Mock()

    calib = mocker.Mock()
    array_model = mocker.Mock()
    array_model.calibration_models = {"TEL01": {"CAL01": calib}}
    simulator.array_models = [array_model]

    mock_settings = mocker.Mock()
    mock_settings.config.args = {"flasher_photons": None}
    mocker.patch("simtools.simulator.settings", mock_settings)

    simulator._overwrite_flasher_photons_for_direct_injection()

    calib.overwrite_model_parameter.assert_not_called()


def test_simulate_direct_injection_sequence_reloads_config_per_run(mocker):
    base_args = {
        "run_mode": "direct_injection",
        "run_number": 10,
        "number_of_events": [2, 1, 1],
        "flasher_photons": ["1e6", "2e6", "3e6"],
    }
    base_db_config = {"db_api_user": "user"}

    mock_config = mocker.Mock()
    mock_config.args = base_args
    mock_config.db_config = base_db_config
    mocker.patch("simtools.simulator.settings", mocker.Mock(config=mock_config))

    mock_init = mocker.patch.object(Simulator, "__init__", return_value=None)
    mock_simulate = mocker.patch.object(Simulator, "simulate")
    mock_validate = mocker.patch.object(Simulator, "validate_simulations")

    Simulator.simulate_direct_injection_sequence(label="test-label")

    assert mock_init.call_count == 3
    assert mock_simulate.call_count == 3
    assert mock_validate.call_count == 3

    assert mock_config.load.call_count == 4
    run0_args = mock_config.load.call_args_list[0].kwargs["args"]
    run1_args = mock_config.load.call_args_list[1].kwargs["args"]
    run2_args = mock_config.load.call_args_list[2].kwargs["args"]
    restore_args = mock_config.load.call_args_list[3].kwargs["args"]

    assert run0_args["run_number"] == 10
    assert run1_args["run_number"] == 11
    assert run2_args["run_number"] == 12
    assert run0_args["number_of_events"] == 2
    assert run1_args["number_of_events"] == 1
    assert run2_args["number_of_events"] == 1
    assert run0_args["flasher_photons"] == 1000000
    assert run1_args["flasher_photons"] == 2000000
    assert run2_args["flasher_photons"] == 3000000
    assert restore_args == base_args


def test_simulate_direct_injection_sequence_defaults_events_and_photons_when_missing(mocker):
    base_args = {
        "run_mode": "direct_injection",
        "run_number": 10,
        "number_of_events": None,
        "flasher_photons": None,
    }
    base_db_config = {"db_api_user": "user"}

    mock_config = mocker.Mock()
    mock_config.args = base_args
    mock_config.db_config = base_db_config
    mocker.patch("simtools.simulator.settings", mocker.Mock(config=mock_config))

    mocker.patch.object(Simulator, "__init__", return_value=None)
    mocker.patch.object(Simulator, "simulate")
    mocker.patch.object(Simulator, "validate_simulations")

    Simulator.simulate_direct_injection_sequence(label="test-label")

    run_args = mock_config.load.call_args_list[0].kwargs["args"]
    assert run_args["run_number"] == 10
    assert run_args["number_of_events"] == 1
    assert run_args.get("flasher_photons") is None


def test_simulate_direct_injection_sequence_expands_single_event_for_multiple_photon_runs(mocker):
    base_args = {
        "run_mode": "direct_injection",
        "run_number": 20,
        "number_of_events": [3],
        "flasher_photons": [100, 200],
    }
    base_db_config = {"db_api_user": "user"}

    mock_config = mocker.Mock()
    mock_config.args = base_args
    mock_config.db_config = base_db_config
    mocker.patch("simtools.simulator.settings", mocker.Mock(config=mock_config))

    mocker.patch.object(Simulator, "__init__", return_value=None)
    mocker.patch.object(Simulator, "simulate")
    mocker.patch.object(Simulator, "validate_simulations")

    Simulator.simulate_direct_injection_sequence(label="test-label")

    run0_args = mock_config.load.call_args_list[0].kwargs["args"]
    run1_args = mock_config.load.call_args_list[1].kwargs["args"]
    assert run0_args["number_of_events"] == 3
    assert run1_args["number_of_events"] == 3


def test_simulate_direct_injection_sequence_raises_for_invalid_event_list_length(mocker):
    base_args = {
        "run_mode": "direct_injection",
        "run_number": 10,
        "number_of_events": [1, 2],
        "flasher_photons": [100, 200, 300],
    }
    base_db_config = {"db_api_user": "user"}

    mock_config = mocker.Mock()
    mock_config.args = base_args
    mock_config.db_config = base_db_config
    mocker.patch("simtools.simulator.settings", mocker.Mock(config=mock_config))

    with pytest.raises(
        ValueError,
        match="Invalid number_of_events list length for direct_injection",
    ):
        Simulator.simulate_direct_injection_sequence(label="test-label")


def test_simulate_direct_injection_sequence_raises_for_invalid_photon_list_length(mocker):
    base_args = {
        "run_mode": "direct_injection",
        "run_number": 10,
        "number_of_events": [1, 2, 3],
        "flasher_photons": [100, 200],
    }
    base_db_config = {"db_api_user": "user"}

    mock_config = mocker.Mock()
    mock_config.args = base_args
    mock_config.db_config = base_db_config
    mocker.patch("simtools.simulator.settings", mocker.Mock(config=mock_config))

    with pytest.raises(
        ValueError,
        match="Invalid flasher_photons list length for direct_injection",
    ):
        Simulator.simulate_direct_injection_sequence(label="test-label")


def test_simulate_direct_injection_sequence_restores_config_after_failure(mocker):
    base_args = {
        "run_mode": "direct_injection",
        "run_number": 10,
        "number_of_events": 3,
        "flasher_photons": 100,
    }
    base_db_config = {"db_api_user": "user"}

    mock_config = mocker.Mock()
    mock_config.args = base_args
    mock_config.db_config = base_db_config
    mocker.patch("simtools.simulator.settings", mocker.Mock(config=mock_config))

    mocker.patch.object(Simulator, "__init__", return_value=None)
    mocker.patch.object(Simulator, "simulate", side_effect=RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        Simulator.simulate_direct_injection_sequence(label="test-label")

    assert mock_config.load.call_count == 2
    assert mock_config.load.call_args_list[-1].kwargs["args"] == base_args


def test_report(array_simulator, mocker, caplog):

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

    # Test case 4: Runtime of 0 is a valid value
    mock_resources = {"runtime": 0, "n_events": 500}
    mocker.patch.object(
        array_simulator._simulation_runner, "get_resources", return_value=mock_resources
    )

    result = array_simulator._make_resources_report()
    expected = "Mean wall time/run [sec]: 0.0, #events/run: 500"
    assert result == expected

    # Test case 5: No runtime available (empty runtime list)
    mock_resources = {"n_events": 500}
    mocker.patch.object(
        array_simulator._simulation_runner, "get_resources", return_value=mock_resources
    )

    result = array_simulator._make_resources_report()
    assert "Mean wall time/run [sec]: nan" in result
    assert ", #events/run: 500" in result


def test_get_corsika_file(array_simulator, mocker):
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
    mock_simulation_runner = mocker.Mock()
    array_simulator._simulation_runner = mock_simulation_runner

    mock_runner_service = mocker.Mock()
    array_simulator.runner_service = mock_runner_service

    array_simulator.run_number = 42
    array_simulator._extra_commands = ["echo test"]

    mock_runner_service.get_file_name.side_effect = lambda file_type, run_num: {
        "sub_script": f"script_{run_num}.sh",
        "sub_out": f"output_{run_num}.out",
        "sub_err": f"output_{run_num}.err",
    }[file_type]

    mocker.patch.object(array_simulator, "_get_corsika_file", return_value="/path/to/corsika.file")
    mocker.patch.object(array_simulator, "update_file_lists")
    mock_submit = mocker.patch(
        "simtools.job_execution.job_manager.submit", return_value=(None, 2.5)
    )

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
        err_file="output_42.err",
        env={"SIM_TELARRAY_CONFIG_PATH": ""},
        return_runtime=True,
    )
    assert array_simulator._runtime == pytest.approx(2.5)


@pytest.mark.parametrize("transition_energy", [None, 120 * u.GeV])
def test_simulate_exports_resolved_corsika_configuration(
    shower_simulator, mocker, transition_energy
):
    shower_simulator._simulation_runner = mocker.Mock()
    shower_simulator.runner_service = mocker.Mock()
    shower_simulator.runner_service.get_file_name.side_effect = lambda file_type, run_number: (
        f"{file_type}_{run_number}"
    )
    mocker.patch.object(shower_simulator, "update_file_lists")

    mock_config = mocker.Mock()
    mock_config.args = {"corsika_hadronic_transition_energy": transition_energy}
    mock_config.corsika_interaction_models = ("qgs3", "urqmd")
    mocker.patch("simtools.settings.config", mock_config)
    mock_submit = mocker.patch(
        "simtools.job_execution.job_manager.submit", return_value=(None, 2.5)
    )

    shower_simulator.simulate()

    expected_environment = {
        "SIM_TELARRAY_CONFIG_PATH": "",
        "SIMTOOLS_CORSIKA_HE_INTERACTION": "qgs3",
        "SIMTOOLS_CORSIKA_LE_INTERACTION": "urqmd",
    }
    if transition_energy is not None:
        expected_environment["SIMTOOLS_CORSIKA_HADRONIC_TRANSITION_ENERGY_GEV"] = "120.0"
    assert mock_submit.call_args.kwargs["env"] == expected_environment


def test_save_file_lists(array_simulator, mocker, tmp_test_directory, caplog):
    mock_io_handler = mocker.Mock()
    array_simulator.io_handler = mock_io_handler
    output_dir = Path(str(tmp_test_directory)) / "output"
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


def test_get_first_corsika_config_error(shower_simulator):
    shower_simulator.corsika_configurations = []
    with pytest.raises(ValueError, match="CORSIKA configuration not found for verification"):
        shower_simulator._get_first_corsika_config()


def test_update_file_lists_with_none_file_list(array_simulator, mocker):
    mock_runner_file_list = {
        "sim_telarray_output": ["output1.simtel.zst"],
        "sim_telarray_log": ["log1.log.gz"],
    }
    array_simulator.file_list = None
    array_simulator._simulation_runner.file_list = mock_runner_file_list

    array_simulator.update_file_lists()

    assert array_simulator.file_list == mock_runner_file_list


def test_update_file_lists_with_existing_file_list(array_simulator, mocker):
    existing_file_list = {
        "sim_telarray_output": ["existing_output.simtel.zst"],
        "corsika_log": ["existing_log.log"],
    }
    new_file_list = {
        "sim_telarray_output": ["new_output.simtel.zst"],
        "sim_telarray_log": ["new_log.log.gz"],
    }

    array_simulator.file_list = existing_file_list.copy()
    array_simulator._simulation_runner.file_list = new_file_list

    array_simulator.update_file_lists()

    assert array_simulator.file_list["sim_telarray_output"] == ["new_output.simtel.zst"]
    assert array_simulator.file_list["sim_telarray_log"] == ["new_log.log.gz"]
    assert array_simulator.file_list["corsika_log"] == ["existing_log.log"]


def test_validate_simulations_sim_telarray(array_simulator, mocker):
    mock_corsika_config = mocker.Mock()
    mock_corsika_config.shower_events = 1000
    mock_corsika_config.mc_events = 500
    mock_corsika_config.use_curved_atmosphere = False

    mocker.patch.object(
        array_simulator, "_get_first_corsika_config", return_value=mock_corsika_config
    )

    mock_simtel_validator = mocker.patch(
        "simtools.simulator.simtel_output_validator.validate_sim_telarray"
    )
    mock_corsika_validator = mocker.patch(
        "simtools.simulator.corsika_output_validator.validate_corsika_output"
    )

    mock_output_files = ["output_file1.simtel.zst", "output_file2.simtel.zst"]
    mock_log_files = ["log_file1.log.gz", "log_file2.log.gz"]

    mocker.patch.object(
        array_simulator,
        "get_files",
        side_effect=lambda file_type: {
            "sim_telarray_output": mock_output_files,
            "sim_telarray_log": mock_log_files,
        }.get(file_type, []),
    )

    array_simulator.validate_simulations()

    mock_simtel_validator.assert_called_once_with(
        data_files=mock_output_files,
        log_files=mock_log_files,
        array_models=array_simulator.array_models,
        expected_mc_events=500,
        expected_shower_events=1000,
        curved_atmo=False,
        allow_for_changes=["nsb_scaling_factor", "stars"],
    )
    mock_corsika_validator.assert_not_called()


def test_validate_simulations_corsika(shower_simulator, mocker):
    mock_corsika_config = mocker.Mock()
    mock_corsika_config.shower_events = 2000
    mock_corsika_config.mc_events = 1000
    mock_corsika_config.use_curved_atmosphere = True

    mocker.patch.object(
        shower_simulator, "_get_first_corsika_config", return_value=mock_corsika_config
    )

    mock_simtel_validator = mocker.patch(
        "simtools.simulator.simtel_output_validator.validate_sim_telarray"
    )
    mock_corsika_validator = mocker.patch(
        "simtools.simulator.corsika_output_validator.validate_corsika_output"
    )

    mock_corsika_log_files = ["corsika_run_001.log"]

    mocker.patch.object(
        shower_simulator,
        "get_files",
        side_effect=lambda file_type: {
            "corsika_log": mock_corsika_log_files,
        }.get(file_type, []),
    )

    shower_simulator.validate_simulations()

    mock_corsika_validator.assert_called_once_with(
        data_files=[],
        log_files=mock_corsika_log_files,
        expected_shower_events=2000,
        curved_atmo=True,
    )
    mock_simtel_validator.assert_not_called()


@pytest.mark.parametrize(
    ("save_corsika_output", "expected_corsika_data_files"),
    [(False, None), (True, ["output_file.corsika.zst"])],
)
def test_validate_simulations_corsika_sim_telarray(
    shower_array_simulator,
    mocker,
    caplog,
    save_corsika_output,
    expected_corsika_data_files,
):
    mock_corsika_config = mocker.Mock()
    mock_corsika_config.shower_events = 1500
    mock_corsika_config.mc_events = 750
    mock_corsika_config.use_curved_atmosphere = False

    mocker.patch.object(
        shower_array_simulator, "_get_first_corsika_config", return_value=mock_corsika_config
    )

    mock_simtel_validator = mocker.patch(
        "simtools.simulator.simtel_output_validator.validate_sim_telarray"
    )
    mock_corsika_validator = mocker.patch(
        "simtools.simulator.corsika_output_validator.validate_corsika_output"
    )

    mock_simtel_output_files = ["output_file.simtel.zst"]
    mock_simtel_log_files = ["log_file.log.gz"]
    mock_corsika_log_files = ["corsika_run_001.log"]
    mock_corsika_output_files = ["output_file.corsika.zst"]

    mocker.patch.object(
        shower_array_simulator,
        "get_files",
        side_effect=lambda file_type: {
            "sim_telarray_output": mock_simtel_output_files,
            "sim_telarray_log": mock_simtel_log_files,
            "corsika_log": mock_corsika_log_files,
            "corsika_output": mock_corsika_output_files,
        }.get(file_type, []),
    )
    mock_config = mocker.patch("simtools.simulator.settings.config")
    mock_config.args = {"save_corsika_output": save_corsika_output}

    with caplog.at_level(logging.INFO):
        shower_array_simulator.validate_simulations()

    assert "Validating simulations" in caplog.text
    assert "750 MC events and 1500 shower events" in caplog.text

    mock_simtel_validator.assert_called_once()
    mock_corsika_validator.assert_called_once_with(
        data_files=expected_corsika_data_files,
        log_files=mock_corsika_log_files,
        expected_shower_events=1500,
        curved_atmo=False,
    )


def test_validate_simulations_with_reduced_event_lists(array_simulator, mocker, caplog):
    mock_corsika_config = mocker.Mock()
    mock_corsika_config.shower_events = 1000
    mock_corsika_config.mc_events = 500
    mock_corsika_config.use_curved_atmosphere = False

    mocker.patch.object(
        array_simulator, "_get_first_corsika_config", return_value=mock_corsika_config
    )

    mocker.patch("simtools.simulator.simtel_output_validator.validate_sim_telarray")
    mocker.patch("simtools.simulator.corsika_output_validator.validate_corsika_output")
    mock_output_validator = mocker.patch("simtools.simulator.output_validator.validate_sim_events")

    mock_output_files = ["output_file.simtel.zst"]
    mock_log_files = ["log_file.log.gz"]
    mock_event_data_files = ["output_file.reduced_event_data.hdf5"]

    mocker.patch.object(
        array_simulator,
        "get_files",
        side_effect=lambda file_type: {
            "sim_telarray_output": mock_output_files,
            "sim_telarray_log": mock_log_files,
            "sim_telarray_event_data": mock_event_data_files,
        }.get(file_type, []),
    )

    mock_config = mocker.Mock()
    mock_config.args.get.return_value = True
    mocker.patch("simtools.simulator.settings.config", mock_config)

    with caplog.at_level(logging.INFO):
        array_simulator.validate_simulations()

    mock_output_validator.assert_called_once_with(
        data_files=mock_event_data_files,
        expected_mc_events=500,
    )
