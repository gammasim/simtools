import logging
from io import StringIO

import astropy.units as u
import numpy as np
import pytest
from astropy.table import Table

from simtools.reporting.model_figures import (
    _generate_pulse_samples,
    _get_flasher_readout_models,
    _supported_flasher_shape,
    plot_nsb_altitude_correction,
    write_flasher_pulse_shape_plots,
    write_telescope_structure_figure,
)


def test_plot_nsb_altitude_correction_creates_heatmap(tmp_path):
    input_file = tmp_path / "altitude-correction.ecsv"
    Table(
        rows=[(300.0, 2.5, 0.2), (400.0, 2.5, 0.4), (300.0, 5.0, 0.6), (400.0, 5.0, 0.8)],
        names=("wavelength", "altitude", "extinction"),
    ).write(input_file, format="ascii.ecsv")

    plot_names = plot_nsb_altitude_correction(input_file, tmp_path)

    assert len(plot_names) == 1
    assert plot_names[0].startswith("nsb_altitude_correction_")
    assert (tmp_path / f"{plot_names[0]}.png").is_file()


def test_mirror_plot_identity_ignores_repository_only_metadata(tmp_path):
    from simtools.reporting.model_figures import _mirror_plot_render_data

    table = Table(
        {
            "mirror_x": [0.0, 10.0] * u.cm,
            "mirror_y": [0.0, 20.0] * u.cm,
            "mirror_diameter": [100.0, 100.0] * u.cm,
            "focal_length": [1000.0, 1000.0] * u.cm,
            "shape_type": [1, 1],
            "mirror_panel_id": [0, 1],
        },
        meta={"context_from_sim_telarray": "Rmax = 2.0 m Total mirror surface area: 4.0 m**2"},
    )
    first_path = tmp_path / "nectarcam.ecsv"
    second_path = tmp_path / "flashcam.ecsv"
    table.write(first_path, format="ascii.ecsv")
    table.meta["instrument"] = "MSTx-FlashCam"
    table.write(second_path, format="ascii.ecsv")

    first = _mirror_plot_render_data(first_path, "mirror_list", "MSTx-NectarCam")
    second = _mirror_plot_render_data(second_path, "mirror_list", "MSTx-FlashCam")

    assert first["statistics"] == second["statistics"]
    assert first["is_mst"] == second["is_mst"]
    for column in ("x_cm", "y_cm", "focal_lengths_cm", "panel_ids"):
        assert np.array_equal(first[column], second[column])


def test_mirror_plot_identity_hashes_non_mirror_input_content(tmp_test_directory):
    from hashlib import sha256
    from pathlib import Path

    from simtools.reporting.model_figures import _mirror_plot_render_data

    source = Path(tmp_test_directory) / "segmentation.ecsv"
    source.write_text("first input\n", encoding="utf-8")
    first = _mirror_plot_render_data(source, "primary_mirror_segmentation", "LSTN-design")
    source.write_text("second input\n", encoding="utf-8")
    second = _mirror_plot_render_data(source, "primary_mirror_segmentation", "LSTN-design")

    assert first == sha256(b"first input\n").hexdigest()
    assert second == sha256(b"second input\n").hexdigest()


def test_write_telescope_structure_figure_renders_and_reuses_existing_image(
    tmp_test_directory, run_context
):
    from pathlib import Path
    from types import SimpleNamespace

    run_context.paths.output_path["default"] = Path(tmp_test_directory)
    telescope_model = SimpleNamespace(
        name="LSTN-design",
        model_version="8.0.0",
        parameters={"focal_length": {"value": 28.0, "unit": "m"}},
        mirrors=SimpleNamespace(
            mirror_table=Table(
                {
                    "mirror_x": [0.0] * u.m,
                    "mirror_y": [0.0] * u.m,
                    "mirror_diameter": [1.0] * u.m,
                    "shape_type": [1],
                }
            )
        ),
        config_file_directory=Path(tmp_test_directory),
    )
    first_output = StringIO()
    second_output = StringIO()

    write_telescope_structure_figure(
        first_output, telescope_model, logging.getLogger(__name__), context=run_context
    )
    telescope_model.name = "LSTS-design"
    write_telescope_structure_figure(
        second_output, telescope_model, logging.getLogger(__name__), context=run_context
    )

    assert "Telescope structure for LSTN-design." in first_output.getvalue()
    assert "Telescope structure for LSTS-design." in second_output.getvalue()
    first_image_path = first_output.getvalue().split("](", maxsplit=1)[1].split(")", maxsplit=1)[0]
    second_image_path = (
        second_output.getvalue().split("](", maxsplit=1)[1].split(")", maxsplit=1)[0]
    )
    assert first_image_path == second_image_path
    image_directory = Path(run_context.paths.get_output_directory("_images"))
    assert len(list(image_directory.glob("telescope_structure_*.png"))) == 1


@pytest.mark.parametrize(
    ("shape", "width", "decay", "expected_first_time"),
    [
        ("TopHat", 2.0, 0.0, -1.0),
        ("Gauss", 2.0, 0.0, -10.0),
        ("Exponential", 0.0, 2.0, 0.0),
    ],
)
def test_generate_flasher_pulse_samples_for_analytic_shapes(
    shape, width, decay, expected_first_time
):
    times, rates = _generate_pulse_samples(shape, width, decay, 60)

    assert np.isclose(times[0], expected_first_time)
    assert np.isclose(rates.max(), 1.0)
    assert len(times) == len(rates)
    if shape == "Gauss":
        assert np.isclose(rates[np.flatnonzero(np.isclose(times, 0.0))[0]], 1.0)
        assert np.isclose(rates[np.argmin(np.abs(times - width))], np.exp(-0.5), atol=0.01)
    elif shape == "TopHat":
        assert np.allclose(rates, np.ones_like(rates))
    else:
        sample_at_one_decay = np.argmin(np.abs(times - decay))
        assert np.isclose(rates[sample_at_one_decay], np.exp(-1.0), atol=0.01)


def test_generate_flasher_pulse_samples_rejects_unavailable_shape():
    with pytest.raises(ValueError, match="Pulse shape 'Unknown' has no tabulated profile"):
        _generate_pulse_samples("Unknown", 1.0, 1.0, 60)

    with pytest.raises(ValueError, match="Pulse shape Exponential has no finite-width profile"):
        _generate_pulse_samples("Exponential", 0.0, 0.0, 60)


@pytest.mark.parametrize(
    "value",
    [
        ["Gauss", 1.0, 0.0],
        ["TopHat", 1.0, 0.0],
        ["Exponential", 0.0, 1.0],
        ["Gauss-Exponential", 1.0, 1.0],
    ],
)
def test_supported_flasher_shape_accepts_supported_profiles(value):
    assert _supported_flasher_shape(value) == tuple(value)


def test_supported_flasher_shape_skips_zero_width_and_unknown_shapes():
    assert _supported_flasher_shape(["Gauss", 0.0, 0.0]) is None
    assert _supported_flasher_shape(["Exponential", 1.0, 0.0]) is None
    assert _supported_flasher_shape(["Gauss-Exponential", 1.0, -1.0]) is None
    assert _supported_flasher_shape(["Unknown", 1.0, 0.0]) is None


def test_flasher_readout_lookup_skips_other_site_and_missing_parameters(mocker):
    model_reader = mocker.Mock()
    model_reader.read_production_table.return_value = {
        "parameters": {
            "LSTS-design": {},
            "LSTN-design": {},
            "MSTx-NectarCam": {},
            "LSTN-01": {},
        }
    }

    def read_parameters(**kwargs):
        if kwargs["array_element_name"] == "LSTN-design":
            raise ValueError("No parameters found for telescopes: LSTN-design")
        return {
            "calibration_devices": {"value": {"flat_fielding": "MSFx-NectarCam"}},
            "fadc_sum_bins": {"value": 60},
        }

    model_reader.get_model_parameters.side_effect = read_parameters

    result = _get_flasher_readout_models(model_reader, "North", "8.0.0", "MSFx-NectarCam")

    assert result == [("MSTx-NectarCam", 60)]
    assert [
        call.kwargs["array_element_name"]
        for call in model_reader.get_model_parameters.call_args_list
    ] == [
        "LSTN-design",
        "MSTx-NectarCam",
    ]


def test_flasher_readout_lookup_propagates_unexpected_parameter_errors(mocker):
    model_reader = mocker.Mock()
    model_reader.read_production_table.return_value = {"parameters": {"LSTN-design": {}}}
    model_reader.get_model_parameters.side_effect = ValueError("metadata is malformed")

    with pytest.raises(ValueError, match="metadata is malformed"):
        _get_flasher_readout_models(model_reader, "North", "8.0.0", "MSFx-NectarCam")


def test_write_flasher_pulse_shape_plots_uses_width_parameters_for_string_shape(
    tmp_test_directory, run_context
):
    from io import StringIO
    from pathlib import Path
    from unittest.mock import Mock

    model_reader = Mock()
    model_reader.get_model_parameters.side_effect = [
        {
            "flasher_pulse_shape": {"parameter_version": "1.0.0", "value": "Gauss"},
            "flasher_pulse_width": {"value": 2.0},
        },
        {
            "calibration_devices": {"value": {"flat_fielding": "MSFx-NectarCam"}},
            "fadc_sum_bins": {"value": 60},
        },
    ]
    model_reader.read_production_table.return_value = {"parameters": {"LSTN-design": {}}}
    output = StringIO()

    write_flasher_pulse_shape_plots(
        output,
        [{"parameter_version": "1.0.0", "model_version": "8.0.0"}],
        "calibration_devices",
        model_reader,
        "North",
        "MSFx-NectarCam",
        context=run_context,
    )

    assert "Flasher pulse profile for Gauss on LSTN-design" in output.getvalue()
    images = list(
        Path(run_context.paths.get_output_directory("_images")).glob("flasher_pulse_shape_*.png")
    )
    assert len(images) == 1


def test_write_flasher_pulse_shape_plots_skips_unmatched_parameter_version(run_context):
    from io import StringIO
    from unittest.mock import Mock

    model_reader = Mock()
    model_reader.get_model_parameters.return_value = {
        "flasher_pulse_shape": {"parameter_version": "2.0.0", "value": "Gauss"}
    }
    output = StringIO()

    write_flasher_pulse_shape_plots(
        output,
        [{"parameter_version": "1.0.0", "model_version": "8.0.0"}],
        "calibration_devices",
        model_reader,
        "North",
        "MSFx-NectarCam",
        context=run_context,
    )

    assert output.getvalue() == ""
    model_reader.read_production_table.assert_not_called()


def test_generate_gauss_exponential_samples_match_requested_edge_widths():
    times, rates = _generate_pulse_samples("Gauss-Exponential", 1.2879, 2.0961, 60)
    peak = rates.argmax()
    rising_low = np.interp(0.1, rates[: peak + 1], times[: peak + 1])
    rising_high = np.interp(0.9, rates[: peak + 1], times[: peak + 1])
    falling_high = np.interp(0.9, rates[peak:][::-1], times[peak:][::-1])
    falling_low = np.interp(0.1, rates[peak:][::-1], times[peak:][::-1])

    assert np.isclose(rates[peak], 1.0)
    assert np.isclose(rising_high - rising_low, 1.2879, atol=0.15)
    assert np.isclose(falling_low - falling_high, 2.0961, atol=0.15)


def test_supported_gauss_exponential_is_case_insensitive():
    assert _supported_flasher_shape(["gauss-exponential", 2, 3]) == ("Gauss-Exponential", 2, 3)
    assert _supported_flasher_shape([None, 2, 3]) is None
