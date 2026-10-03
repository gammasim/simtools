"""Tests for the trigger-patch mapping application."""

from types import SimpleNamespace

import pytest

from simtools.applications import plot_trigger_patches


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        (None, {}),
        ({"camera_pixels": 3}, {"camera_pixels": 3}),
        ('{"camera_pixels": 3}', {"camera_pixels": 3}),
        ({"changes": {"corsika_observation_level": 1800}}, {"corsika_observation_level": 1800}),
    ],
)
def test_run_creates_models_and_delegates_to_mapping(
    tmp_test_directory, mocker, overrides, expected
):
    output_directory = tmp_test_directory / "output"
    context = SimpleNamespace(
        args={
            "site": "North",
            "telescope": "LSTN-01",
            "model_version": "6.0.0",
            "required_pixels": 2,
            "include_presum_slaves": False,
            "photons_per_pixel": 500,
            "events_per_combination": 1,
            "run_number": 1,
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
        postscript_file=output_directory / "patches.ps",
        simtel_file=output_directory / "patches.simtel.gz",
    )
    runner = mocker.patch.object(
        plot_trigger_patches, "run_trigger_patch_mapping", return_value=files
    )

    assert plot_trigger_patches.run(context) is files
    assert telescope_class.call_args.kwargs["model_directory"] == output_directory
    assert telescope_class.call_args.kwargs["overwrite_model_parameter_dict"] == expected
    assert site_class.call_args.kwargs["overwrite_model_parameter_dict"] == expected
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
    )


_MODEL_ARGS = ["--site", "North", "--telescope", "LSTN-01", "--model_version", "7.0.0"]


def test_cli_defaults():
    args = plot_trigger_patches.APPLICATION.build_parser().parse_args(_MODEL_ARGS)
    assert args.model_version == "7.0.0"
    assert args.required_pixels == 2
    assert args.photons_per_pixel == 500
    assert args.events_per_combination == 1
    assert args.run_number == 1
    assert args.include_presum_slaves is False
    assert args.output_name is None


@pytest.mark.parametrize("missing", ["site", "telescope", "model_version"])
def test_cli_requires_model_selection(missing):
    missing_index = _MODEL_ARGS.index(f"--{missing}")
    args = _MODEL_ARGS[:missing_index] + _MODEL_ARGS[missing_index + 2 :]
    with pytest.raises(SystemExit):
        plot_trigger_patches.APPLICATION.build_parser().parse_args(args)


@pytest.mark.parametrize(
    "option", ["required_pixels", "photons_per_pixel", "events_per_combination", "run_number"]
)
def test_cli_rejects_nonpositive_options(option):
    with pytest.raises(SystemExit):
        plot_trigger_patches.APPLICATION.build_parser().parse_args(
            [*_MODEL_ARGS, f"--{option}", "0"]
        )
