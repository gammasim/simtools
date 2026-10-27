from unittest.mock import Mock

from astropy.table import Table

from simtools.reporting.model_figures import (
    _get_flasher_readout_models,
    _plot_nsb_altitude_correction,
    _use_exported_model_filenames,
)


def test_get_flasher_readout_models_uses_explicit_device_assignment():
    reader = Mock()
    reader.get_array_elements.return_value = ["MSTx-NectarCam", "MSTx-FlashCam"]
    reader.get_model_parameters.side_effect = [
        {
            "calibration_devices": {"value": {"flat_fielding": "MSFx-NectarCam"}},
            "fadc_sum_bins": {"value": 60},
        },
        {
            "calibration_devices": {"value": {"flat_fielding": "MSFx-FlashCam"}},
            "fadc_sum_bins": {"value": 40},
        },
    ]

    result = _get_flasher_readout_models(reader, "North", "8.0.0", "MSFx-NectarCam")

    assert result == [("MSTx-NectarCam", 60)]


def test_use_exported_model_filenames_matches_qualified_report_assets(tmp_path):
    telescope_model = Mock()
    telescope_model.config_file_directory = tmp_path
    telescope_model.parameters = {
        "mirror_list": {
            "file": True,
            "instrument": "MSTx-NectarCam",
            "value": "mirror_list-1.0.0.ecsv",
        }
    }
    (tmp_path / "mirror_list-1.0.0-MSTx-NectarCam.ecsv").touch()

    _use_exported_model_filenames(telescope_model)

    assert telescope_model.parameters["mirror_list"]["value"] == (
        "mirror_list-1.0.0-MSTx-NectarCam.ecsv"
    )


def test_plot_nsb_altitude_correction_creates_heatmap(tmp_path):
    input_file = tmp_path / "altitude-correction.ecsv"
    Table(
        rows=[(300.0, 2.5, 0.2), (400.0, 2.5, 0.4), (300.0, 5.0, 0.6), (400.0, 5.0, 0.8)],
        names=("wavelength", "altitude", "extinction"),
    ).write(input_file, format="ascii.ecsv")

    plot_names = _plot_nsb_altitude_correction(input_file, tmp_path)

    assert len(plot_names) == 1
    assert plot_names[0].startswith("altitude-correction_")
    assert (tmp_path / f"{plot_names[0]}.png").is_file()
