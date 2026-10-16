"""Tests for sim_telarray trigger-patch mapping."""

from pathlib import Path
from unittest.mock import Mock

import astropy.units as u
import numpy as np
import pytest
from astropy.table import Table

import simtools.simtel.trigger_patch_mapping as mapping
from simtools.simtel import trigger_patch_validation as validation

PIXLED_LOG_FILE = "pixled.log"
FIRING_PAIR = "Firing #0 (pixel 0) #1 (pixel 1)\n"
OUT_FILE = "out_file"
RETAINED_REPORT = "retained report"

ATMPROF1_DAT = "atmprof1.dat"
GET_MODEL_WRITER = "get_model_writer"
PREVIOUS_SITE_PROFILE = "previous site profile"
SELECTED_SITE_ATMOSPHERE = "selected site atmosphere"

INVALID_CAMERA_GEOMETRY = "Invalid camera geometry"
ADC_SAMPLES = "adc_samples"
EVENT_ID = "event_id"
MC_EVENT = "mc_event"
PLOT_PIXEL_LAYOUT_WITH_IMAGE = "plot_pixel_layout_with_image"
SHOWER_NUMBER = "shower_num"
TELESCOPE_EVENTS = "telescope_events"
HEADER = "header"
LIST_TRGSECT = "list_trgsect"
PHOTOELECTRONS = "photoelectrons"
PIXEL_LIST = "pixel_list"
PIXEL_LISTS = "pixel_lists"
PIXEL_SHAPE = "pixel_shape"
PIXEL_DIAMETER = "pixel_diameter"
SAVEFIG = "savefig"
LIMIT = "limit"
IMAGE = "image"

EMPTY = "empty"

ALTITUDE = "--altitude"
SLAVES = "--slaves"
LSTN_01 = "LSTN-01"
MISSING_OR_EMPTY_OUTPUT = "Missing or empty output"
PDF_FILENAME = "PDF filename"
SIM_TELARRAY_PATH = "_sim_telarray_path"
ATMOSPHERE_ECSV = "atmosphere.ecsv"
ATMOSPHERIC_TRANSMISSION = "atmospheric_transmission"
CAMERA_TEST_DAT = "camera-test.dat"
CUSTOM_PDF = "custom.pdf"
EXTERNAL_FAILURE = "external failure"
PIXLED = "pixled"
RESULT_PS = "result.ps"
SIM_TELARRAY = "sim_telarray"
SUBMIT = "submit"
UTF_8 = "utf-8"
VALUE = "value"


def _models(tmp_test_directory):
    """Return models whose exported configuration defines a camera file."""
    tmp_test_directory = Path(tmp_test_directory)
    config_file = tmp_test_directory / "telescope.cfg"
    camera_file = tmp_test_directory / CAMERA_TEST_DAT
    camera_file.write_text(
        "Pixel 0 1 0 0\nPixel 1 1 5 0\nMajorityTrigger 1 of 0 1\n", encoding=UTF_8
    )
    telescope = Mock(name=LSTN_01)
    telescope.name = LSTN_01
    telescope.site = "North"
    telescope.model_version = "7.0.0"
    telescope.get_parameter_value.return_value = 0
    telescope.get_parameter_table.side_effect = lambda key: {
        "camera_trigger_groups": Table(
            rows=[
                {
                    validation.GROUP_ID: 0,
                    validation.KIND: "majority",
                    validation.USE_DEFAULT_MULTIPLICITY: False,
                    validation.MULTIPLICITY: 1,
                }
            ]
        ),
        "camera_trigger_members": Table(
            rows=[
                {
                    validation.GROUP_ID: 0,
                    validation.MEMBER_ORDER: pixel,
                    validation.PIXEL_ORDER: 0,
                    validation.PIXEL_ID: pixel,
                    validation.REQUIRED: False,
                }
                for pixel in (0, 1)
            ]
        ),
    }[key]
    telescope.config_file_path = config_file
    telescope.config_file_directory = tmp_test_directory

    def write_config(additional_models):
        assert additional_models is site
        config_file.write_text(
            'camera_config_file = "camera-test.dat" # camera\n'
            'atmospheric_transmission = "atmosphere-North.dat"\n',
            encoding=UTF_8,
        )

    telescope.write_sim_telarray_config_file.side_effect = write_config
    site = Mock()
    site.parameters = {ATMOSPHERIC_TRANSMISSION: {VALUE: ATMOSPHERE_ECSV}}
    site.get_parameter_value.side_effect = lambda name: {ATMOSPHERIC_TRANSMISSION: ATMOSPHERE_ECSV}[
        name
    ]
    site.get_parameter_value_with_unit.return_value = 2150 * u.m
    writer = Mock()
    writer.get_atmospheric_profile_file_name.return_value = "selected-site-profile.dat"
    writer.export_atmospheric_transmission_file.side_effect = lambda directory: (
        Path(directory) / writer.get_atmospheric_profile_file_name()
    ).write_text(SELECTED_SITE_ATMOSPHERE, encoding=UTF_8)
    site.configuration_writers = {SIM_TELARRAY: writer}
    return telescope, site


def _simtel_installation(tmp_test_directory, monkeypatch):
    """Configure fake sim_telarray executables."""
    tmp_test_directory = Path(tmp_test_directory)
    simtel_path = tmp_test_directory / SIM_TELARRAY
    pixled = simtel_path / "LightEmission" / PIXLED
    simtel = simtel_path / "bin" / SIM_TELARRAY
    for executable in (pixled, simtel):
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.touch()
        executable.chmod(0o755)
    monkeypatch.setattr(mapping.settings.config, SIM_TELARRAY_PATH, str(simtel_path))
    monkeypatch.setattr(mapping.settings.config, "_sim_telarray_exe", SIM_TELARRAY)
    monkeypatch.setattr(
        mapping,
        "_write_pdf",
        lambda files, _name, _limit: files.pdf_file.write_bytes(b"%PDF-1.4\n"),
    )
    return pixled, simtel


def _produce_output(command, **kwargs):
    """Stand in for each program by writing its requested output."""
    option = {SIM_TELARRAY: "-r", PIXLED: "-o"}[Path(command[0]).name]
    Path(command[command.index(option) + 1]).write_bytes(b"generated output\n")
    if Path(command[0]).name == PIXLED:
        Path(kwargs[OUT_FILE]).write_text(FIRING_PAIR, encoding=UTF_8)


def test_run_trigger_patch_mapping_runs_simtel_tools(tmp_test_directory, monkeypatch, mocker):
    telescope, site = _models(tmp_test_directory)
    pixled, simtel = _simtel_installation(tmp_test_directory, monkeypatch)
    submit = mocker.patch.object(mapping.job_manager, SUBMIT, side_effect=_produce_output)

    files = mapping.run_trigger_patch_mapping(
        telescope,
        site,
        tmp_test_directory / "output",
        include_presum_slaves=True,
        photons_per_pixel=700,
        events_per_combination=2,
        run_number=3,
    )

    assert (
        files.camera_file == tmp_test_directory / "output/trigger-patches-LSTN-01-run3.camera.dat"
    )
    assert (
        files.camera_file.read_bytes() == (Path(tmp_test_directory) / CAMERA_TEST_DAT).read_bytes()
    )
    assert files.pdf_file.name == "trigger-patches-LSTN-01-run3.pdf"
    assert len(files.log_files) == 4
    assert submit.call_count == 2
    pixled_command = submit.call_args_list[0].args[0]
    assert pixled_command == [
        str(pixled),
        f"-I{files.iact_file.parent}",
        "--atmosphere",
        "1",
        "--camera-file",
        str(files.pdf_file.with_suffix(validation.ILLUMINATION_SUFFIX)),
        "--use-trg-all",
        "--require",
        "1",
        "--photons",
        "700",
        "--events",
        "1",
        "--run",
        "3",
        ALTITUDE,
        "2150.0",
        "-o",
        str(files.iact_file),
    ]
    manifest = Table.read(files.pdf_file.with_suffix(validation.EVENTS_SUFFIX))
    assert len(manifest) == 2
    simtel_command = submit.call_args_list[1].args[0]
    assert simtel_command[:6] == [
        str(simtel),
        "-c",
        str(telescope.config_file_path),
        f"-I{telescope.config_file_directory}",
        "-DNUM_TELESCOPES=1",
        "-C",
    ]
    assert "Altitude=2150.0" in simtel_command
    assert "atmospheric_transmission=atmosphere-North.dat" in simtel_command
    assert f"histogram_file={files.pdf_file.with_suffix('.hdata')}" in simtel_command
    for setting in ("trigger_telescopes=1", "random_mono_prob=0"):
        assert setting in simtel_command
        assert simtel_command[simtel_command.index(setting) - 1] == "-C"
    assert simtel_command[-3:] == ["-r", str(files.simtel_file), str(files.iact_file)]
    assert not any(token.startswith("input_file=") for token in simtel_command)
    for index, call in enumerate(submit.call_args_list):
        assert call.kwargs == {
            OUT_FILE: files.log_files[index * 2],
            "err_file": files.log_files[index * 2 + 1],
            "env": mapping.SIM_TELARRAY_ENV,
        }


def test_run_trigger_patch_mapping_rejects_invalid_options(tmp_test_directory):
    telescope, site = _models(tmp_test_directory)

    with pytest.raises(ValueError, match="positive"):
        mapping.run_trigger_patch_mapping(telescope, site, tmp_test_directory, required_pixels=0)
    with pytest.raises(ValueError, match=PDF_FILENAME):
        mapping.run_trigger_patch_mapping(
            telescope, site, tmp_test_directory, output_name=RESULT_PS
        )


@pytest.mark.parametrize(LIMIT, [-1, True, 1.5, None, 2147483648])
def test_mapping_rejects_invalid_event_page_limit(tmp_test_directory, limit):
    telescope, site = _models(tmp_test_directory)
    with pytest.raises(ValueError, match="max_plotted_events must be a nonnegative"):
        mapping.run_trigger_patch_mapping(
            telescope, site, tmp_test_directory, max_plotted_events=limit
        )


@pytest.mark.parametrize("radius", [0, -1, float("nan"), float("inf"), True, "invalid"])
def test_mapping_rejects_invalid_radius_before_execution(tmp_test_directory, radius):
    telescope, site = _models(tmp_test_directory)
    with pytest.raises(ValueError, match="max_pixel_radius must be finite and positive"):
        mapping.run_trigger_patch_mapping(
            telescope, site, tmp_test_directory, max_pixel_radius=radius
        )


def test_event_cap_includes_repetitions_and_radius_limits_inputs(
    tmp_test_directory, monkeypatch, mocker
):
    telescope, site = _models(tmp_test_directory)
    _simtel_installation(tmp_test_directory, monkeypatch)
    mocker.patch.object(mapping.job_manager, SUBMIT, side_effect=_produce_output)
    files = mapping.run_trigger_patch_mapping(
        telescope,
        site,
        tmp_test_directory,
        required_pixels=1,
        events_per_combination=5,
        max_events=3,
        max_pixel_radius=0.01,
    )
    manifest = Table.read(files.pdf_file.with_suffix(validation.EVENTS_SUFFIX))
    assert len(manifest) == 3
    assert list(manifest[validation.INPUT_PIXELS]) == ["[0]"] * 3
    groups, _pixels, _rotation = validation.read_camera(
        files.pdf_file.with_suffix(validation.ILLUMINATION_SUFFIX)
    )
    assert len(groups) == 3
    assert all(group[validation.REQUIRED] == [0] for group in groups)


def test_legacy_pixled_log_recovers_complete_and_slave_patterns(tmp_test_directory):
    log = Path(tmp_test_directory) / PIXLED_LOG_FILE
    log.write_text(
        "Using 2 pixels: +0 1\nFiring all selected pixels.\n"
        "    Slaves pixels also firing for pixel 1: 2\n",
        encoding=UTF_8,
    )
    assert mapping._read_pixled_input_patterns(log) == [[0, 1, 2]]


def test_run_trigger_patch_mapping_requires_executables(tmp_test_directory, monkeypatch):
    telescope, site = _models(tmp_test_directory)
    monkeypatch.setattr(
        mapping.settings.config, SIM_TELARRAY_PATH, str(tmp_test_directory / validation.MISSING)
    )

    with pytest.raises(FileNotFoundError, match=PIXLED):
        mapping.run_trigger_patch_mapping(telescope, site, tmp_test_directory)


def test_export_camera_file_requires_generated_camera(tmp_test_directory):
    telescope, site = _models(tmp_test_directory)
    (Path(tmp_test_directory) / CAMERA_TEST_DAT).unlink()

    with pytest.raises(FileNotFoundError, match="Generated camera"):
        mapping._export_camera_file(telescope, site)


@pytest.mark.parametrize(VALUE, [0, -1, 1.5, "2", True, None, 2147483648])
def test_numeric_options_require_positive_integers(value):
    with pytest.raises(ValueError, match="positive 32-bit integers"):
        mapping._validate_options(value)


@pytest.mark.parametrize("name", ["../result.pdf", "nested/result.pdf", "/result.pdf", RESULT_PS])
def test_output_name_rejects_directories_and_wrong_suffix(tmp_test_directory, name):
    with pytest.raises(ValueError, match=PDF_FILENAME):
        mapping._output_file(tmp_test_directory, name)


@pytest.mark.parametrize("missing_index", [0, 1])
def test_missing_executable_fails_before_model_export(
    tmp_test_directory, monkeypatch, mocker, missing_index
):
    telescope, site = _models(tmp_test_directory)
    executables = _simtel_installation(tmp_test_directory, monkeypatch)
    executables[missing_index].unlink()
    submit = mocker.patch.object(mapping.job_manager, SUBMIT)
    with pytest.raises(FileNotFoundError):
        mapping.run_trigger_patch_mapping(telescope, site, tmp_test_directory)
    telescope.write_sim_telarray_config_file.assert_not_called()
    submit.assert_not_called()


def test_nonexecutable_program_fails_early(tmp_test_directory, monkeypatch):
    telescope, site = _models(tmp_test_directory)
    pixled, _ = _simtel_installation(tmp_test_directory, monkeypatch)
    pixled.chmod(0o644)
    with pytest.raises(PermissionError, match=str(pixled)):
        mapping.run_trigger_patch_mapping(telescope, site, tmp_test_directory)


@pytest.mark.parametrize("failed_step", [0, 1])
def test_failed_step_stops_pipeline_and_retains_context(
    tmp_test_directory, monkeypatch, mocker, failed_step
):
    telescope, site = _models(tmp_test_directory)
    _simtel_installation(tmp_test_directory, monkeypatch)
    failure = mapping.job_manager.JobExecutionError(EXTERNAL_FAILURE)
    outputs = [_produce_output] * failed_step + [failure]

    def submit_step(command, **kwargs):
        step = outputs.pop(0)
        if isinstance(step, Exception):
            raise step
        step(command, **kwargs)

    submit = mocker.patch.object(mapping.job_manager, SUBMIT, side_effect=submit_step)
    with pytest.raises(mapping.job_manager.JobExecutionError, match=EXTERNAL_FAILURE) as error:
        mapping.run_trigger_patch_mapping(telescope, site, tmp_test_directory)
    assert error.value.__cause__ is failure
    assert submit.call_count == failed_step + 1
    for suffix in (
        CAMERA_TEST_DAT,
        ".iact.gz",
        ".simtel.gz",
        ".pdf",
        ".stdout.log",
        ".stderr.log",
    ):
        assert suffix in str(error.value)


@pytest.mark.parametrize(EMPTY, [False, True])
def test_successful_exit_without_output_is_an_error(tmp_test_directory, monkeypatch, mocker, empty):
    telescope, site = _models(tmp_test_directory)
    _simtel_installation(tmp_test_directory, monkeypatch)

    def submit_step(command, **_kwargs):
        if empty:
            Path(command[command.index("-o") + 1]).touch()

    submit = mocker.patch.object(mapping.job_manager, SUBMIT, side_effect=submit_step)
    with pytest.raises(mapping.job_manager.JobExecutionError, match=MISSING_OR_EMPTY_OUTPUT):
        mapping.run_trigger_patch_mapping(telescope, site, tmp_test_directory)
    submit.assert_called_once()


def test_defaults_and_site_altitude(tmp_test_directory, monkeypatch, mocker):
    telescope, site = _models(tmp_test_directory)
    site.get_parameter_value_with_unit.return_value = 1.8 * u.km
    _simtel_installation(tmp_test_directory, monkeypatch)
    submit = mocker.patch.object(mapping.job_manager, SUBMIT, side_effect=_produce_output)
    files = mapping.run_trigger_patch_mapping(
        telescope, site, tmp_test_directory, output_name=CUSTOM_PDF
    )
    assert files.camera_file == tmp_test_directory / CAMERA_TEST_DAT
    assert files.pdf_file == tmp_test_directory / CUSTOM_PDF
    command = submit.call_args_list[0].args[0]
    assert SLAVES not in command
    assert command[command.index(ALTITUDE) + 1] == "1800.0"
    assert "Altitude=1800.0" in submit.call_args_list[1].args[0]


@pytest.mark.parametrize("missing_step", [0, 1])
def test_repeat_run_cannot_accept_previous_outputs(
    tmp_test_directory, monkeypatch, mocker, missing_step
):
    telescope, site = _models(tmp_test_directory)
    _simtel_installation(tmp_test_directory, monkeypatch)
    submit = mocker.patch.object(mapping.job_manager, SUBMIT, side_effect=_produce_output)
    files = mapping.run_trigger_patch_mapping(telescope, site, tmp_test_directory)
    outputs = (files.iact_file, files.simtel_file)
    for output in outputs:
        output.write_bytes(b"previous run\n")
    submit.reset_mock()

    def submit_step(command, **kwargs):
        index = submit.call_count - 1
        assert not outputs[index].exists()
        if index != missing_step:
            _produce_output(command, **kwargs)

    submit.side_effect = submit_step
    with pytest.raises(mapping.job_manager.JobExecutionError, match=MISSING_OR_EMPTY_OUTPUT):
        mapping.run_trigger_patch_mapping(telescope, site, tmp_test_directory)
    assert submit.call_count == missing_step + 1
    assert not outputs[missing_step].exists()
    for output in outputs[:missing_step]:
        assert output.stat().st_size > 0
    for output in outputs[missing_step + 1 :]:
        assert output.read_bytes() == b"previous run\n"


@pytest.mark.parametrize("text", ["# no camera\n", "camera_config_file =\n"])
def test_exported_config_requires_camera_setting(tmp_test_directory, text):
    telescope, site = _models(tmp_test_directory)
    telescope.write_sim_telarray_config_file.side_effect = None
    telescope.config_file_path.write_text(text, encoding=UTF_8)
    with pytest.raises(ValueError, match="camera_config_file"):
        mapping._export_camera_file(telescope, site)


@pytest.mark.parametrize(
    validation.KIND, ["MajorityTrigger", "DigitalSumTrigger", "AnalogSumTrigger"]
)
def test_mapping_rejects_group_that_pixled_would_skip(
    tmp_test_directory, monkeypatch, mocker, kind
):
    telescope, site = _models(tmp_test_directory)
    _simtel_installation(tmp_test_directory, monkeypatch)
    camera_file = tmp_test_directory / CAMERA_TEST_DAT
    camera_file.write_text(f"{kind} 1 of 0 1\n{kind} 1 of 0\n", encoding=UTF_8)
    submit = mocker.patch.object(mapping.job_manager, SUBMIT)
    with pytest.raises(ValueError, match="has 1 master pixels, but required_pixels=2"):
        mapping.run_trigger_patch_mapping(telescope, site, tmp_test_directory, required_pixels=2)
    submit.assert_not_called()


@pytest.mark.parametrize("members", ["+0 1", "0[2,3] 1", "0 1 # comment"])
def test_group_size_counts_masters_only(tmp_test_directory, members):
    camera_file = tmp_test_directory / CAMERA_TEST_DAT
    camera_file.write_text(f"# camera\nMajorityTrigger * of {members}\n", encoding=UTF_8)
    mapping._validate_trigger_group_sizes(camera_file, 2)
    with pytest.raises(ValueError, match="has 2 master pixels"):
        mapping._validate_trigger_group_sizes(camera_file, 3)


def test_group_size_requires_trigger_definitions(tmp_test_directory):
    camera_file = tmp_test_directory / CAMERA_TEST_DAT
    camera_file.write_text("# empty camera\n\nPixel 0\n", encoding=UTF_8)
    with pytest.raises(ValueError, match="No trigger groups found"):
        mapping._validate_trigger_group_sizes(camera_file, 1)


def test_parse_trigger_groups_and_event_context(tmp_test_directory):
    camera_file = Path(tmp_test_directory) / CAMERA_TEST_DAT
    camera_file.write_text(
        "AnalogSumTrigger * of 4 16 7\nMajorityTrigger * of +4 20[21,22] 16\n",
        encoding=UTF_8,
    )
    native_groups, _pixels, _rotation = validation.read_camera(camera_file)
    groups = [set(group[validation.MASTERS]) for group in native_groups]
    log_file = Path(tmp_test_directory) / PIXLED_LOG_FILE
    log_file.write_text("Firing #0 (pixel 4) #1 (pixel 16)\n", encoding=UTF_8)
    input_patterns = mapping._read_pixled_input_patterns(log_file)
    photoelectrons = np.zeros(17, dtype=int)
    photoelectrons[[4, 16]] = [3, 4]
    context = mapping._trigger_event_context(
        {
            MC_EVENT: {SHOWER_NUMBER: 1},
            PHOTOELECTRONS: {0: {PHOTOELECTRONS: photoelectrons}},
        },
        {HEADER: {LIST_TRGSECT: np.array([0, 1])}},
        1,
        groups,
        input_patterns,
    )
    assert context == (
        [4, 16],
        [4, 16],
        [0, 1],
        [0, 1],
    )


def test_trigger_event_context_handles_missing_input_metadata(tmp_test_directory):
    missing_log = Path(tmp_test_directory) / "missing.log"
    assert mapping._read_pixled_input_patterns(missing_log) == []
    context = mapping._trigger_event_context({}, {HEADER: {}}, 1, [{0}], [])
    assert context == ([], [], [], [])


def _pdf_files(tmp_test_directory):
    tmp_test_directory = Path(tmp_test_directory)
    camera_file = tmp_test_directory / CAMERA_TEST_DAT
    camera_file.write_text("AnalogSumTrigger * of 0 1\n", encoding=UTF_8)
    pixled_log = tmp_test_directory / PIXLED_LOG_FILE
    pixled_log.write_text(FIRING_PAIR * 3, encoding=UTF_8)
    return mapping.TriggerPatchMappingFiles(
        camera_file=camera_file,
        iact_file=tmp_test_directory / "events.iact.gz",
        simtel_file=tmp_test_directory / "events.simtel.gz",
        pdf_file=tmp_test_directory / CUSTOM_PDF,
        log_files=(pixled_log,),
    )


def _simulated_events(mocker, events):
    reader = mocker.MagicMock()
    reader.telescope_descriptions = {
        1: {
            "camera_organization": {
                "n_sectors": 1,
                "sectors": [[0], [0]],
                "sector_type": [1],
                "sector_threshold": [10.0],
                "sector_pixel_threshold": [0.0],
            },
            "camera_settings": {
                "focal_length": 28.0,
                "pixel_size": np.array([0.05, 0.05]),
                "common_pixel_shape": 1,
                "pixel_x": np.array([0.0, 0.05]),
                "pixel_y": np.array([0.0, 0.0]),
            },
        }
    }
    reader.__iter__.side_effect = lambda: iter(events)
    reader.__enter__.return_value = reader
    mocker.patch.object(mapping, "SimTelFile", return_value=reader)
    return reader


def test_manifest_matches_events_by_identity_and_rejects_duplicates(tmp_test_directory):
    files = _pdf_files(tmp_test_directory)
    path = files.pdf_file.with_suffix(validation.EVENTS_SUFFIX)
    table = Table({validation.EVENT_NUMBER: [2, 1]}, meta={"origin": "planned"})
    table.write(path)
    rows, metadata = mapping._inspection_manifest(files, [])
    assert [row[validation.EVENT_NUMBER] for row in rows] == [1, 2]
    assert metadata == table.meta
    table[validation.EVENT_NUMBER] = [1, 1]
    table.write(path, overwrite=True)
    with pytest.raises(ValueError, match="unique and contiguous"):
        mapping._inspection_manifest(files, [])


def test_write_pdf_uses_simtools_analysis_and_plotting(tmp_test_directory, mocker):
    files = _pdf_files(tmp_test_directory)
    files.pdf_file.write_bytes(b"stale PDF")
    samples = np.full((1, 2, 40), 10.0)
    samples[0, 0, 8:12] = 30.0
    _simulated_events(
        mocker,
        [
            {EVENT_ID: 1, MC_EVENT: {SHOWER_NUMBER: 1}, TELESCOPE_EVENTS: {}},
            {
                EVENT_ID: 2,
                MC_EVENT: {SHOWER_NUMBER: 2},
                PHOTOELECTRONS: {0: {PHOTOELECTRONS: np.array([5, 5])}},
                TELESCOPE_EVENTS: {
                    1: {
                        ADC_SAMPLES: samples,
                        PIXEL_LISTS: {1: {PIXEL_LIST: np.array([0])}},
                        HEADER: {LIST_TRGSECT: np.array([0])},
                    }
                },
            },
            {
                EVENT_ID: 3,
                MC_EVENT: {SHOWER_NUMBER: 3},
                PHOTOELECTRONS: {0: {PHOTOELECTRONS: np.array([5, 5])}},
                TELESCOPE_EVENTS: {
                    1: {
                        ADC_SAMPLES: samples,
                        PIXEL_LISTS: {1: {PIXEL_LIST: np.array([0])}},
                        HEADER: {LIST_TRGSECT: np.array([0])},
                    }
                },
            },
        ],
    )
    plot = mocker.spy(mapping, PLOT_PIXEL_LAYOUT_WITH_IMAGE)
    overlay = mocker.spy(mapping, "_overlay_trigger_groups")
    pages = mocker.spy(mapping.PdfPages, SAVEFIG)
    open_figures = mapping.plt.get_fignums()
    mapping._write_pdf(files, LSTN_01)
    assert files.pdf_file.read_bytes().startswith(b"%PDF-")
    assert pages.call_count == 6
    assert plot.call_count == 5
    assert plot.call_args_list[0].kwargs[IMAGE] == pytest.approx([1, 1])
    assert plot.call_args_list[1].kwargs[IMAGE] == pytest.approx([2, 2])
    assert plot.call_args.kwargs[IMAGE] == pytest.approx([80.0, 0.0])
    assert overlay.call_count == 3
    assert overlay.call_args.args[3:] == ([0], [0, 1])
    assert mapping.plt.get_fignums() == open_figures


@pytest.mark.parametrize(LIMIT, [0, 1, 2, 5])
def test_pdf_event_page_limit_does_not_limit_accounting(tmp_test_directory, mocker, limit):
    files = _pdf_files(tmp_test_directory)
    _simulated_events(
        mocker,
        [{EVENT_ID: number * 100, MC_EVENT: {SHOWER_NUMBER: number}} for number in (1, 2, 3)],
    )
    pages = mocker.spy(mapping.PdfPages, SAVEFIG)
    mapping._write_pdf(files, LSTN_01, max_plotted_events=limit)
    assert pages.call_count == 3 + min(limit, 3)
    assert len(Table.read(files.pdf_file.with_suffix(validation.EVENTS_SUFFIX))) == 3


def test_fired_patch_outlines_enclose_whole_patches_without_pixel_edges():
    camera = Mock(
        pixels={
            mapping.PIXEL_IDS: [0, 1, 2],
            "x": [0, 1, 2],
            "y": [0, 0, 0],
            PIXEL_SHAPE: 2,
            PIXEL_DIAMETER: 1,
        }
    )
    figure, axis = mapping.plt.subplots()
    try:
        mapping._overlay_trigger_groups(axis, camera, [{0, 1}, {1, 2}], [0, 1, 99], [1])
        assert len(axis.patches) == 2
        assert axis.patches[0].get_edgecolor() != axis.patches[1].get_edgecolor()
        assert [patch.get_label() for patch in axis.patches] == ["Patch 0", "Patch 1"]
        for patch, left in zip(axis.patches, [-0.5, 0.5]):
            vertices = patch.get_xy()
            assert len(vertices) == 5
            assert vertices[:, 0].min() == pytest.approx(left)
            assert vertices[:, 0].max() == pytest.approx(left + 2)
            assert vertices[:, 1].min() == pytest.approx(-0.5)
            assert vertices[:, 1].max() == pytest.approx(0.5)
        assert len(axis.collections) == 1
        assert np.asarray(axis.collections[0].get_offsets()) == pytest.approx(np.array([[1, 0]]))
    finally:
        mapping.plt.close(figure)


def test_patch_outline_skips_empty_or_unsupported_pixel_shapes():
    camera = Mock(pixels={mapping.PIXEL_IDS: [0], "x": [0], "y": [0], PIXEL_SHAPE: 0})
    figure, axis = mapping.plt.subplots()
    try:
        mapping._outline_patch(axis, camera, {0})
        mapping._outline_patch(axis, camera, set())
        assert not axis.patches
    finally:
        mapping.plt.close(figure)


def test_patch_palette_cycles_with_distinct_line_styles():
    camera = Mock(
        pixels={
            mapping.PIXEL_IDS: [0],
            "x": [0],
            "y": [0],
            PIXEL_SHAPE: 2,
            PIXEL_DIAMETER: 1,
        }
    )
    figure, axis = mapping.plt.subplots()
    try:
        mapping._overlay_trigger_groups(axis, camera, [{0}] * 6, range(6), [])
        assert len({patch.get_edgecolor() for patch in axis.patches}) == 3
        assert axis.patches[0].get_linestyle() != axis.patches[3].get_linestyle()
        assert axis.get_legend() is not None
    finally:
        mapping.plt.close(figure)


def test_observed_patch_efficiency_excludes_untested_groups():
    groups = [
        {validation.GROUP_ID: 0, validation.OBSERVED: 4, validation.INTENDED_SECTOR_FIRED: 3},
        {validation.GROUP_ID: 1, validation.OBSERVED: 0, validation.INTENDED_SECTOR_FIRED: 0},
        {validation.GROUP_ID: 2, validation.OBSERVED: 2, validation.INTENDED_SECTOR_FIRED: 0},
    ]
    assert mapping._observed_patch_efficiency(groups) == pytest.approx(
        [0.75, np.nan, 0], nan_ok=True
    )


@pytest.mark.parametrize("description_count", [0, 2])
def test_pdf_rejects_invalid_telescope_count(tmp_test_directory, mocker, description_count):
    files = _pdf_files(tmp_test_directory)
    reader = _simulated_events(mocker, [])
    reader.telescope_descriptions = {index: {} for index in range(description_count)}
    with pytest.raises(mapping.job_manager.JobExecutionError, match="exactly one telescope"):
        mapping._write_pdf(files, LSTN_01)
    assert not files.pdf_file.exists()


def test_pdf_rejects_missing_events_and_removes_stale_pdf(tmp_test_directory, mocker):
    files = _pdf_files(tmp_test_directory)
    files.pdf_file.write_bytes(b"stale PDF")
    _simulated_events(mocker, [{EVENT_ID: 1, TELESCOPE_EVENTS: {}}])
    with pytest.raises(mapping.job_manager.JobExecutionError, match="No MC events"):
        mapping._write_pdf(files, LSTN_01)
    assert not files.pdf_file.exists()


def test_pdf_retains_missing_trials_and_report_on_integrity_failure(tmp_test_directory, mocker):
    files = _pdf_files(tmp_test_directory)
    _simulated_events(mocker, [{EVENT_ID: 100, MC_EVENT: {SHOWER_NUMBER: 1}}])
    with pytest.raises(mapping.job_manager.JobExecutionError, match=RETAINED_REPORT):
        mapping._write_pdf(files, LSTN_01)
    assert files.pdf_file.exists()
    rows = Table.read(files.pdf_file.with_suffix(validation.EVENTS_SUFFIX))
    assert list(rows[validation.STATE]) == [
        validation.NON_TRIGGERED,
        validation.MISSING,
        validation.MISSING,
    ]


def test_all_non_triggered_events_produce_a_valid_compact_report(tmp_test_directory, mocker):
    files = _pdf_files(tmp_test_directory)
    _simulated_events(
        mocker,
        [{EVENT_ID: number * 100, MC_EVENT: {SHOWER_NUMBER: number}} for number in (1, 2, 3)],
    )
    mapping._write_pdf(files, LSTN_01)
    rows = Table.read(files.pdf_file.with_suffix(validation.EVENTS_SUFFIX))
    assert list(rows[validation.STATE]) == [validation.NON_TRIGGERED] * 3


def test_incomplete_stream_does_not_turn_absent_triggers_into_known_negatives(
    tmp_test_directory, mocker
):
    files = _pdf_files(tmp_test_directory)
    reader = _simulated_events(
        mocker,
        [{EVENT_ID: number * 100, MC_EVENT: {SHOWER_NUMBER: number}} for number in (1, 2, 3)],
    )
    reader.histograms = None
    with pytest.raises(mapping.job_manager.JobExecutionError, match=RETAINED_REPORT):
        mapping._write_pdf(files, LSTN_01)
    rows = Table.read(files.pdf_file.with_suffix(validation.EVENTS_SUFFIX))
    assert list(rows[validation.STATE]) == [validation.UNRESOLVED] * 3


def test_pdf_plot_failure_closes_figure_and_removes_partial_pdf(tmp_test_directory, mocker):
    files = _pdf_files(tmp_test_directory)
    _simulated_events(
        mocker,
        [
            {
                EVENT_ID: 1,
                MC_EVENT: {SHOWER_NUMBER: 1},
                TELESCOPE_EVENTS: {1: {ADC_SAMPLES: np.ones((1, 2, 40))}},
            }
        ],
    )
    failure = ValueError(INVALID_CAMERA_GEOMETRY)
    mocker.patch.object(mapping, PLOT_PIXEL_LAYOUT_WITH_IMAGE, side_effect=failure)
    open_figures = mapping.plt.get_fignums()
    with pytest.raises(mapping.job_manager.JobExecutionError, match=INVALID_CAMERA_GEOMETRY) as exc:
        mapping._write_pdf(files, LSTN_01)
    assert exc.value.__cause__ is failure
    assert not files.pdf_file.exists()
    assert mapping.plt.get_fignums() == open_figures


def test_pixled_finds_atmosphere_outside_installation_directory(
    tmp_test_directory, monkeypatch, mocker
):
    telescope, site = _models(tmp_test_directory)
    pixled, _simtel = _simtel_installation(tmp_test_directory, monkeypatch)
    atmosphere_directory = Path(tmp_test_directory)
    atmosphere_file = atmosphere_directory / ATMPROF1_DAT
    working_directory = Path(tmp_test_directory) / "unrelated working directory"
    working_directory.mkdir()
    monkeypatch.chdir(working_directory)

    def submit_step(command, **kwargs):
        if Path(command[0]) == pixled:
            include_directories = [Path(token[2:]) for token in command if token.startswith("-I")]
            assert atmosphere_file.parent in include_directories
            assert any((path / atmosphere_file.name).is_file() for path in include_directories)
            assert atmosphere_file.read_text(encoding=UTF_8) == SELECTED_SITE_ATMOSPHERE
        _produce_output(command, **kwargs)

    mocker.patch.object(mapping.job_manager, SUBMIT, side_effect=submit_step)
    files = mapping.run_trigger_patch_mapping(telescope, site, tmp_test_directory)
    assert files.pdf_file.exists()


@pytest.mark.parametrize("native_name", [ATMPROF1_DAT, "site-profile.dat"])
def test_pixled_atmosphere_uses_exported_native_site_profile(
    tmp_test_directory, mocker, native_name
):
    output_directory = Path(tmp_test_directory)
    profile_file = output_directory / native_name
    writer = mocker.Mock()
    writer.get_atmospheric_profile_file_name.return_value = native_name
    writer.export_atmospheric_transmission_file.side_effect = lambda directory: (
        directory / native_name
    ).write_text("native profile for selected site and version", encoding=UTF_8)
    model_writer = mocker.patch.object(mapping, GET_MODEL_WRITER, return_value=writer)
    site = mocker.sentinel.site
    mapping._prepare_pixled_atmosphere(site, output_directory)
    model_writer.assert_called_once_with(site)
    writer.export_atmospheric_transmission_file.assert_called_once_with(output_directory)
    assert (output_directory / ATMPROF1_DAT).read_bytes() == profile_file.read_bytes()


def test_pixled_atmosphere_missing_export_fails(tmp_test_directory, mocker):
    writer = mocker.Mock()
    writer.get_atmospheric_profile_file_name.return_value = "missing-profile.dat"
    mocker.patch.object(mapping, GET_MODEL_WRITER, return_value=writer)
    with pytest.raises(FileNotFoundError, match="Exported site atmospheric profile not found"):
        mapping._prepare_pixled_atmosphere(mocker.sentinel.site, Path(tmp_test_directory))


def test_pixled_atmosphere_replaces_alias_without_modifying_previous_source(
    tmp_test_directory, mocker
):
    output_directory = Path(tmp_test_directory)
    previous_profile = output_directory / "previous-profile.dat"
    previous_profile.write_text(PREVIOUS_SITE_PROFILE, encoding=UTF_8)
    alias = output_directory / ATMPROF1_DAT
    alias.symlink_to(previous_profile)
    _telescope, site = _models(tmp_test_directory)
    mapping._prepare_pixled_atmosphere(site, output_directory)
    assert alias.read_text(encoding=UTF_8) == SELECTED_SITE_ATMOSPHERE
    assert previous_profile.read_text(encoding=UTF_8) == PREVIOUS_SITE_PROFILE
    assert not alias.is_symlink()
