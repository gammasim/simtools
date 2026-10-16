"""Tests for the trigger-patch mapping application."""

from types import SimpleNamespace

import pytest

from simtools.applications import plot_trigger_patches

VALUE_7_0_0 = "7.0.0"
LSTN_01 = "LSTN-01"
NORTH = "North"
CAMERA_PIXELS = "camera_pixels"
CORSIKA_OBSERVATION_LEVEL = "corsika_observation_level"
EVENTS_PER_COMBINATION = "events_per_combination"
MODEL_VERSION = "model_version"
OVERWRITE_MODEL_PARAMETER_DICT = "overwrite_model_parameter_dict"
PHOTONS_PER_PIXEL = "photons_per_pixel"
REQUIRED_PIXELS = "required_pixels"
RUN_NUMBER = "run_number"
TELESCOPE = "telescope"


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        (None, {}),
        ({CAMERA_PIXELS: 3}, {CAMERA_PIXELS: 3}),
        ('{"camera_pixels": 3}', {CAMERA_PIXELS: 3}),
        ({"changes": {CORSIKA_OBSERVATION_LEVEL: 1800}}, {CORSIKA_OBSERVATION_LEVEL: 1800}),
    ],
)
def test_run_creates_models_and_delegates_to_mapping(
    tmp_test_directory, mocker, overrides, expected
):
    output_directory = tmp_test_directory / "output"
    context = SimpleNamespace(
        args={
            "site": NORTH,
            TELESCOPE: LSTN_01,
            MODEL_VERSION: "6.0.0",
            REQUIRED_PIXELS: 2,
            "include_presum_slaves": False,
            PHOTONS_PER_PIXEL: 500,
            EVENTS_PER_COMBINATION: 1,
            RUN_NUMBER: 1,
            "output_name": None,
            "overwrite_model_parameters": overrides,
        },
        io_handler=mocker.Mock(),
        model_reader=mocker.sentinel.reader,
        logger=mocker.Mock(),
    )
    context.io_handler.get_output_directory.return_value = output_directory
    telescope_class = mocker.patch.object(plot_trigger_patches, "TelescopeModel")
    telescope = telescope_class.return_value
    site_class = mocker.patch.object(plot_trigger_patches, "SiteModel")
    files = mocker.Mock(
        pdf_file=output_directory / "patches.pdf",
        simtel_file=output_directory / "patches.simtel.gz",
    )
    runner = mocker.patch.object(
        plot_trigger_patches, "run_trigger_patch_mapping", return_value=files
    )

    assert plot_trigger_patches.run(context) is files
    assert telescope_class.call_args.kwargs["model_directory"] == output_directory
    assert telescope_class.call_args.kwargs[OVERWRITE_MODEL_PARAMETER_DICT] == expected
    assert site_class.call_args.kwargs[OVERWRITE_MODEL_PARAMETER_DICT] == expected
    runner.assert_called_once_with(
        telescope,
        mocker.ANY,
        output_directory,
        required_pixels=2,
        include_presum_slaves=False,
        photons_per_pixel=500,
        events_per_combination=1,
        run_number=1,
        output_name=None,
        max_events=None,
        max_pixel_radius=None,
        exhaustive=False,
    )


_MODEL_ARGS = ["--site", NORTH, "--telescope", LSTN_01, "--model_version", VALUE_7_0_0]


def test_cli_defaults():
    args = plot_trigger_patches.APPLICATION.build_parser().parse_args(_MODEL_ARGS)
    assert args.model_version == VALUE_7_0_0
    assert args.required_pixels is None
    assert args.photons_per_pixel == 500
    assert args.events_per_combination == 1
    assert args.run_number == 1
    assert args.include_presum_slaves is False
    assert args.output_name is None
    assert args.max_events is None
    assert args.max_pixel_radius is None
    assert args.exhaustive is False


def test_cli_limits_select_a_small_central_exhaustive_sample():
    args = plot_trigger_patches.APPLICATION.build_parser().parse_args(
        [
            *_MODEL_ARGS,
            "--max_events",
            "5",
            "--max_pixel_radius",
            "0.2",
            "--exhaustive",
            "--required_pixels",
            "2",
        ]
    )
    assert args.max_events == 5
    assert args.max_pixel_radius == pytest.approx(0.2)
    assert args.exhaustive


@pytest.mark.parametrize("missing", ["site", TELESCOPE, MODEL_VERSION])
def test_cli_requires_model_selection(missing):
    missing_index = _MODEL_ARGS.index(f"--{missing}")
    args = _MODEL_ARGS[:missing_index] + _MODEL_ARGS[missing_index + 2 :]
    with pytest.raises(SystemExit):
        plot_trigger_patches.APPLICATION.build_parser().parse_args(args)


@pytest.mark.parametrize(
    "option", [REQUIRED_PIXELS, PHOTONS_PER_PIXEL, EVENTS_PER_COMBINATION, RUN_NUMBER]
)
def test_cli_rejects_nonpositive_options(option):
    with pytest.raises(SystemExit):
        plot_trigger_patches.APPLICATION.build_parser().parse_args(
            [*_MODEL_ARGS, f"--{option}", "0"]
        )
