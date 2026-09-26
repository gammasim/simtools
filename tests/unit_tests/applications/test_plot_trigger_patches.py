"""Tests for the trigger-patch mapping application."""

from types import SimpleNamespace

from simtools.applications import plot_trigger_patches


def test_run_creates_models_and_delegates_to_mapping(tmp_test_directory, mocker):
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
        },
        io_handler=mocker.Mock(),
        model_reader=mocker.sentinel.reader,
        logger=mocker.Mock(),
    )
    context.io_handler.get_output_directory.return_value = output_directory
    telescope_class = mocker.patch.object(plot_trigger_patches, "TelescopeModel")
    telescope = telescope_class.return_value
    mocker.patch.object(plot_trigger_patches, "SiteModel")
    files = mocker.Mock(
        postscript_file=output_directory / "patches.ps",
        simtel_file=output_directory / "patches.simtel.gz",
    )
    runner = mocker.patch.object(
        plot_trigger_patches, "run_trigger_patch_mapping", return_value=files
    )

    assert plot_trigger_patches.run(context) is files
    assert telescope_class.call_args.kwargs["model_directory"] == output_directory
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
