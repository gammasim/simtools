"""Unit tests for simtools.ray_tracing.optics_validation."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import astropy.units as u
import numpy as np
import pytest
from astropy.table import QTable, Table

from simtools.ray_tracing import optics_validation


def test_load_data_normalizes_and_converts_ecsv_radius(tmp_test_directory):
    data_file = Path(str(tmp_test_directory)) / "measured.ecsv"
    table = Table(
        {
            "radius": [10.0, 20.0] * u.mm,
            "differential_value": [1.0, 2.0],
            "integral_value": [2.0, -4.0],
        }
    )
    table.write(data_file, format="ascii.ecsv")

    data = optics_validation.load_data(data_file)

    np.testing.assert_allclose(data["Radius [cm]"], [1.0, 2.0])
    np.testing.assert_allclose(data["Relative intensity"], [0.5, -1.0])


def test_load_data_normalizes_legacy_ascii_data(tmp_test_directory):
    data_file = Path(str(tmp_test_directory)) / "measured.dat"
    data_file.write_text("10.0 1.0 2.0\n20.0 2.0 -4.0\n", encoding="utf-8")

    data = optics_validation.load_data(data_file)

    np.testing.assert_allclose(data["Radius [cm]"], [1.0, 2.0])
    np.testing.assert_allclose(data["Relative intensity"], [0.5, -1.0])


def test_load_data_raises_for_missing_integral_column(tmp_test_directory):
    data_file = Path(str(tmp_test_directory)) / "measured.dat"
    data_file.write_text("radius differential_value\n10.0 1.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Could not find required PSF data column 'integral'"):
        optics_validation.load_data(data_file)


def test_validate_cumulative_psf_raises_without_radius_data():
    args_dict = {
        "site": "North",
        "telescope": "LSTN-01",
        "model_version": "5.0.0",
        "zenith_angle": 20.0 * u.deg,
        "source_distance": 10.0 * u.km,
        "label": "validate_optics",
        "test": True,
    }
    io_handler = MagicMock()
    app_context = SimpleNamespace(args=args_dict, io_handler=io_handler)

    mock_tel_model = MagicMock()
    mock_tel_model.name = "LSTN-01"
    mock_site_model = MagicMock()
    mock_ray = MagicMock()
    mock_image = MagicMock()
    mock_ray.images.return_value = [mock_image]

    with (
        patch(
            "simtools.ray_tracing.optics_validation.initialize_simulation_models",
            return_value=(mock_tel_model, mock_site_model, None),
        ),
        patch("simtools.ray_tracing.optics_validation.RayTracing", return_value=mock_ray),
    ):
        with pytest.raises(ValueError, match="Radius data is not available"):
            optics_validation.validate_cumulative_psf(app_context)


def test_validate_cumulative_psf_saves_cumulative_and_image_plots(tmp_test_directory):
    args_dict = {
        "site": "North",
        "telescope": "LSTN-01",
        "model_version": "5.0.0",
        "zenith_angle": 20.0 * u.deg,
        "source_distance": 10.0 * u.km,
        "data": "measured.dat",
        "data_search_path": str(tmp_test_directory),
        "label": "validate_optics",
        "test": True,
    }
    io_handler = MagicMock()
    io_handler.get_output_file.return_value = Path(str(tmp_test_directory)) / "output.png"
    app_context = SimpleNamespace(args=args_dict, io_handler=io_handler)

    measured = np.array(
        [(1.0, 0.2), (2.0, 1.0)],
        dtype=[("Radius [cm]", "f8"), ("Relative intensity", "f8")],
    )

    mock_tel_model = MagicMock()
    mock_tel_model.name = "LSTN-01"
    mock_site_model = MagicMock()
    mock_ray = MagicMock()
    mock_image = MagicMock()
    mock_image.get_psf.side_effect = [2.0, 4.0]
    mock_image.get_cumulative_data.return_value = np.array([0.4, 0.9])
    mock_image.get_image_data.return_value = np.array(
        [(0.1, -0.2)], dtype=[("X", "f8"), ("Y", "f8")]
    )
    mock_ray.images.return_value = [mock_image]

    fig_1d = MagicMock()
    fig_1d.gca.return_value = MagicMock()
    fig_2d = MagicMock()

    with (
        patch(
            "simtools.ray_tracing.optics_validation.initialize_simulation_models",
            return_value=(mock_tel_model, mock_site_model, None),
        ),
        patch("simtools.ray_tracing.optics_validation.RayTracing", return_value=mock_ray),
        patch("simtools.ray_tracing.optics_validation.gen.find_file", return_value="measured.dat"),
        patch("simtools.ray_tracing.optics_validation.load_data", return_value=measured),
        patch("simtools.ray_tracing.optics_validation.visualize.plot_1d", return_value=fig_1d),
        patch(
            "simtools.ray_tracing.optics_validation.plot_ray_tracing_psf.create_psf_image_figure",
            return_value=(fig_2d, MagicMock()),
        ) as mock_plot_image,
        patch("simtools.ray_tracing.optics_validation.visualize.save_figure") as mock_save,
    ):
        optics_validation.validate_cumulative_psf(app_context)

    assert mock_ray.simulate.call_count == 1
    assert mock_ray.analyze.call_count == 1
    assert mock_plot_image.call_count == 1
    assert mock_save.call_count == 2
    assert all(call.kwargs["close"] is True for call in mock_save.call_args_list)


def test_validate_optics_no_images(tmp_test_directory):
    args_dict = {
        "site": "North",
        "telescope": "LSTN-01",
        "model_version": "5.0.0",
        "zenith_angle": 20.0 * u.deg,
        "source_distance": 10.0 * u.km,
        "max_offset": 1.0 * u.deg,
        "offset_step": 0.5 * u.deg,
        "offset_file": None,
        "offset_directions": "N,S,E,W",
        "plot_images": False,
        "save_photons": False,
        "label": "validate_optics",
        "test": True,
    }
    io_handler = MagicMock()
    io_handler.get_output_file.return_value = Path(str(tmp_test_directory)) / "output.png"
    app_context = SimpleNamespace(args=args_dict, io_handler=io_handler)

    mock_tel_model = MagicMock()
    mock_tel_model.name = "LSTN-01"
    mock_site_model = MagicMock()
    mock_ray = MagicMock()

    with (
        patch(
            "simtools.ray_tracing.optics_validation.initialize_simulation_models",
            return_value=(mock_tel_model, mock_site_model, None),
        ),
        patch("simtools.ray_tracing.optics_validation.RayTracing", return_value=mock_ray),
        patch("simtools.ray_tracing.optics_validation.visualize.save_figure"),
        patch(
            "simtools.ray_tracing.optics_validation._export_effective_focal_length_model_parameter"
        ) as mock_export_model_parameter,
    ):
        optics_validation.validate_optics(app_context)

        mock_export_model_parameter.assert_called_once()

        mock_ray.simulate.assert_called_once_with(test=True, force=False)
        mock_ray.analyze.assert_called_once_with(force=True, save_photons=False)
        assert mock_ray.plot.call_count == 4
        assert mock_ray.plot.call_args_list[-1].kwargs["error_type"] == "errorbar"


def test_validate_optics_with_images_and_default_label(tmp_test_directory):
    args_dict = {
        "site": "North",
        "telescope": "LSTN-01",
        "model_version": "5.0.0",
        "parameter_version": "5.0.1",
        "zenith_angle": 20.0 * u.deg,
        "source_distance": 10.0 * u.km,
        "max_offset": 1.0 * u.deg,
        "offset_step": 0.5 * u.deg,
        "offset_file": None,
        "offset_directions": None,
        "plot_images": True,
        "save_photons": False,
        "label": None,
        "test": True,
    }
    io_handler = MagicMock()
    io_handler.get_output_file.return_value = Path(str(tmp_test_directory)) / "output.pdf"
    app_context = SimpleNamespace(args=args_dict, io_handler=io_handler)

    mock_tel_model = MagicMock()
    mock_tel_model.name = "LSTN-01"
    mock_site_model = MagicMock()
    mock_ray = MagicMock()

    image_non_empty = MagicMock()
    image_non_empty.get_image_data.return_value = np.array(
        [(-2.0, 3.0), (1.0, -1.0)], dtype=[("X", "f8"), ("Y", "f8")]
    )
    image_non_empty.get_psf.return_value = 2.4

    image_empty = MagicMock()
    image_empty.get_image_data.return_value = np.array([], dtype=[("X", "f8"), ("Y", "f8")])
    image_empty.get_psf.return_value = 1.2

    mock_ray.psf_images = {(0.0, 0.0): image_non_empty, (0.5, 0.0): image_empty}

    with (
        patch(
            "simtools.ray_tracing.optics_validation.initialize_simulation_models",
            return_value=(mock_tel_model, mock_site_model, None),
        ),
        patch(
            "simtools.ray_tracing.optics_validation.RayTracing", return_value=mock_ray
        ) as mock_rt,
        patch(
            "simtools.ray_tracing.optics_validation.plot_ray_tracing_psf."
            "create_annotated_psf_image_figure",
            return_value=MagicMock(),
        ) as mock_create_figure,
        patch(
            "simtools.ray_tracing.optics_validation.visualize.save_figures_to_single_document"
        ) as mock_save_pdf,
        patch("simtools.ray_tracing.optics_validation.visualize.save_figure") as mock_save,
        patch(
            "simtools.ray_tracing.optics_validation._export_effective_focal_length_model_parameter"
        ) as mock_export_model_parameter,
    ):
        optics_validation.validate_optics(app_context)

    rt_kwargs = mock_rt.call_args.kwargs
    assert rt_kwargs["label"] == "validate_optics"
    assert rt_kwargs["offset_directions"] is None
    assert len(rt_kwargs["off_axis_angle"]) == 3

    assert mock_save.call_count == 4
    assert all(call.kwargs["close"] is True for call in mock_save.call_args_list)
    assert mock_create_figure.call_count == 2
    assert mock_save_pdf.call_args.kwargs["close"] is True
    mock_export_model_parameter.assert_called_once()


def test_validate_optics_exports_effective_focal_length_model_parameter(tmp_test_directory):
    args_dict = {
        "site": "North",
        "telescope": "LSTN-01",
        "model_version": "5.0.0",
        "parameter_version": "5.0.1",
        "zenith_angle": 20.0 * u.deg,
        "source_distance": 10.0 * u.km,
        "max_offset": 1.0 * u.deg,
        "offset_step": 0.5 * u.deg,
        "offset_file": None,
        "offset_directions": "N,S,E,W",
        "plot_images": False,
        "export_model_parameter": True,
        "label": "validate_optics",
        "test": True,
    }
    io_handler = MagicMock()
    io_handler.get_output_file.return_value = Path(str(tmp_test_directory)) / "output.png"
    io_handler.get_output_directory.return_value = Path(str(tmp_test_directory))
    app_context = SimpleNamespace(args=args_dict, io_handler=io_handler)

    mock_tel_model = MagicMock()
    mock_tel_model.name = "LSTN-01"
    mock_site_model = MagicMock()
    mock_ray = MagicMock()
    mock_ray._results = QTable(
        {
            "off_x": [0.0, 1.0, -1.0, 0.0, 0.0] * u.deg,
            "off_y": [0.0, 0.0, 0.0, 1.0, -1.0] * u.deg,
            "eff_flen": [np.nan, 2930.0, 2910.0, 2920.0, 2940.0],
        }
    )

    with (
        patch(
            "simtools.ray_tracing.optics_validation.initialize_simulation_models",
            return_value=(mock_tel_model, mock_site_model, None),
        ),
        patch("simtools.ray_tracing.optics_validation.RayTracing", return_value=mock_ray),
        patch("simtools.ray_tracing.optics_validation.visualize.save_figure"),
        patch(
            "simtools.ray_tracing.optics_validation."
            "model_data_writer.ModelDataWriter.write_model_parameter"
        ) as mock_write_model_parameter,
    ):
        optics_validation.validate_optics(app_context)

    mock_write_model_parameter.assert_called_once()
    call_kwargs = mock_write_model_parameter.call_args.kwargs
    assert call_kwargs["parameter_name"] == "effective_focal_length"
    assert call_kwargs["instrument"] == "LSTN-01"
    assert call_kwargs["parameter_version"] == "5.0.1"
    assert call_kwargs["metadata_input_dict"] == args_dict
    assert call_kwargs["unit"] == ["cm", "cm", "cm", "cm", "cm"]
    np.testing.assert_allclose(call_kwargs["value"], [2925.0, 2920.0, 2930.0, 0.0, 0.0])


def test_effective_focal_length_value_from_results_empty_mask_returns_zero():
    results = QTable(
        {
            "off_x": [0.0] * u.deg,
            "off_y": [0.0] * u.deg,
            "eff_flen": [np.nan],
        }
    )

    value = optics_validation._effective_focal_length_value_from_results(results)

    assert value == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_export_effective_focal_length_model_parameter_without_results(caplog):
    mock_ray = MagicMock(spec=[])

    optics_validation._export_effective_focal_length_model_parameter(
        ray=mock_ray,
        telescope_name="LSTN-01",
        model_version="5.0.0",
        parameter_version="5.0.1",
        output_directory=Path("dummy"),
        metadata_input_dict={},
    )

    assert "No ray-tracing results available to export effective_focal_length" in caplog.text


# ---------------------------------------------------------------------------
# _median_effective_focal_length
# ---------------------------------------------------------------------------


def _make_results(off_x, off_y, eff_flen):
    """Build a minimal QTable matching the ray-tracing results schema."""
    return QTable(
        {
            "off_x": off_x * u.deg,
            "off_y": off_y * u.deg,
            "eff_flen": np.asarray(eff_flen, dtype=float),
        }
    )


def test_median_effective_focal_length_returns_median_of_nonzero_rows():
    results = _make_results(
        off_x=[0.0, 1.0, -1.0, 0.0, 0.0],
        off_y=[0.0, 0.0, 0.0, 1.0, -1.0],
        eff_flen=[np.nan, 2900.0, 2950.0, 2920.0, 2940.0],
    )

    value = optics_validation._median_effective_focal_length(results)

    assert value == pytest.approx(2930.0)


def test_median_effective_focal_length_excludes_zero_offset_nan():
    # Only the zero-offset row; eff_flen is NaN there -> no valid values
    results = _make_results(off_x=[0.0], off_y=[0.0], eff_flen=[np.nan])

    value = optics_validation._median_effective_focal_length(results)

    assert value is None


def test_median_effective_focal_length_all_nonzero_nan_returns_none():
    results = _make_results(
        off_x=[1.0, -1.0],
        off_y=[0.0, 0.0],
        eff_flen=[np.nan, np.nan],
    )

    value = optics_validation._median_effective_focal_length(results)

    assert value is None


def test_median_effective_focal_length_single_valid_value():
    results = _make_results(off_x=[0.0, 1.0], off_y=[0.0, 0.0], eff_flen=[np.nan, 2800.0])

    value = optics_validation._median_effective_focal_length(results)

    assert value == pytest.approx(2800.0)


# ---------------------------------------------------------------------------
# _prepare_image_for_plotting
# ---------------------------------------------------------------------------

_IMAGE_DTYPE = [("X", "f8"), ("Y", "f8")]


def _make_image_data(x_vals, y_vals):
    data = np.zeros(len(x_vals), dtype=_IMAGE_DTYPE)
    data["X"] = x_vals
    data["Y"] = y_vals
    return data


def test_prepare_image_for_plotting_returns_cm_when_no_eff_flen():
    image_data = _make_image_data([1.0, -2.0], [3.0, -4.0])
    psf_cm = 2.0
    max_extent_cm = 5.0

    converted, psf_q, cont_r, plot_extent = optics_validation._prepare_image_for_plotting(
        image_data, psf_cm, max_extent_cm
    )

    assert converted is image_data
    assert psf_q.unit == u.cm
    assert psf_q.value == pytest.approx(2.0)
    assert cont_r.unit == u.cm
    assert cont_r.value == pytest.approx(1.0)
    assert plot_extent == pytest.approx(5.0)


def test_prepare_image_for_plotting_converts_to_degrees():
    image_data = _make_image_data([100.0, -200.0], [50.0, -50.0])
    psf_cm = 4.0
    max_extent_cm = 200.0
    eff_flen_cm = 2000.0

    converted, psf_q, cont_r, plot_extent = optics_validation._prepare_image_for_plotting(
        image_data, psf_cm, max_extent_cm, eff_flen_cm
    )

    expected_x = np.rad2deg(np.array([100.0, -200.0]) / eff_flen_cm)
    expected_y = np.rad2deg(np.array([50.0, -50.0]) / eff_flen_cm)
    np.testing.assert_allclose(converted["X"], expected_x)
    np.testing.assert_allclose(converted["Y"], expected_y)
    assert psf_q.unit == u.deg
    assert psf_q.value == pytest.approx(np.rad2deg(psf_cm / eff_flen_cm))
    assert cont_r.unit == u.deg
    assert cont_r.value == pytest.approx(np.rad2deg(psf_cm / 2 / eff_flen_cm))
    assert plot_extent == pytest.approx(np.rad2deg(max_extent_cm / eff_flen_cm))


def test_prepare_image_for_plotting_does_not_mutate_original():
    image_data = _make_image_data([10.0], [20.0])
    original_x = image_data["X"].copy()

    optics_validation._prepare_image_for_plotting(image_data, 1.0, 5.0, eff_flen_cm=1000.0)

    np.testing.assert_array_equal(image_data["X"], original_x)


# ---------------------------------------------------------------------------
# _plot_psf_summary
# ---------------------------------------------------------------------------


def test_plot_psf_summary_saves_four_figures():
    mock_ray = MagicMock()
    mock_ray.plot.return_value = MagicMock()
    mock_io = MagicMock()

    with patch("simtools.ray_tracing.optics_validation.visualize.save_figure") as mock_save:
        optics_validation._plot_psf_summary(mock_ray, "validate_optics", "LSTN-01", mock_io)

    assert mock_ray.plot.call_count == 4
    assert mock_save.call_count == 4
    plotted_keys = [call.args[0] for call in mock_ray.plot.call_args_list]
    assert plotted_keys == ["psf_deg", "psf_cm", "eff_area", "eff_flen"]


def test_plot_psf_summary_adds_errorbar_kwargs_for_eff_flen():
    mock_ray = MagicMock()
    mock_ray.plot.return_value = MagicMock()
    mock_io = MagicMock()

    with patch("simtools.ray_tracing.optics_validation.visualize.save_figure"):
        optics_validation._plot_psf_summary(mock_ray, "label", "TEL", mock_io)

    eff_flen_call = mock_ray.plot.call_args_list[3]
    assert eff_flen_call.kwargs.get("error_type") == "errorbar"


# ---------------------------------------------------------------------------
# _eff_flen_cm_for_degree_conversion
# ---------------------------------------------------------------------------


def test_eff_flen_cm_for_degree_conversion_returns_median():
    mock_ray = MagicMock()
    mock_ray._results = _make_results(
        off_x=[0.0, 1.0, -1.0],
        off_y=[0.0, 0.0, 0.0],
        eff_flen=[np.nan, 2900.0, 2950.0],
    )

    value = optics_validation._eff_flen_cm_for_degree_conversion(mock_ray)

    assert value == pytest.approx(2925.0)


def test_eff_flen_cm_for_degree_conversion_returns_none_when_no_results():
    mock_ray = MagicMock(spec=[])  # no _results attribute

    value = optics_validation._eff_flen_cm_for_degree_conversion(mock_ray)

    assert value is None


def test_eff_flen_cm_for_degree_conversion_returns_none_when_column_missing():
    mock_ray = MagicMock()
    mock_ray._results = QTable({"off_x": [1.0] * u.deg, "off_y": [0.0] * u.deg})

    value = optics_validation._eff_flen_cm_for_degree_conversion(mock_ray)

    assert value is None


def test_eff_flen_cm_for_degree_conversion_returns_none_when_all_nan():
    mock_ray = MagicMock()
    mock_ray._results = _make_results(off_x=[1.0], off_y=[0.0], eff_flen=[np.nan])

    value = optics_validation._eff_flen_cm_for_degree_conversion(mock_ray)

    assert value is None


# ---------------------------------------------------------------------------
# _max_image_extent
# ---------------------------------------------------------------------------


def _make_mock_image(x_vals, y_vals):
    image = MagicMock()
    image.get_image_data.return_value = _make_image_data(x_vals, y_vals)
    return image


def test_max_image_extent_returns_rounded_half_range():
    images_dict = {
        (0.0, 0.0): _make_mock_image([-3.0, 1.0], [2.0, -1.0]),
        (1.0, 0.0): _make_mock_image([0.5, -0.5], [4.1, -0.5]),
    }

    result = optics_validation._max_image_extent(images_dict)

    # max abs is 4.1; ceil(4.1*2)/2 = ceil(8.2)/2 = 9/2 = 4.5
    assert result == pytest.approx(4.5)


def test_max_image_extent_ignores_empty_images():
    images_dict = {
        (0.0, 0.0): _make_mock_image([], []),
        (1.0, 0.0): _make_mock_image([2.0], [-1.0]),
    }

    result = optics_validation._max_image_extent(images_dict)

    # max abs is 2.0; ceil(2.0*2)/2 = ceil(4.0)/2 = 2.0
    assert result == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# _plot_psf_images
# ---------------------------------------------------------------------------


def test_plot_psf_images_calls_create_figure_per_image_and_saves_pdf():
    image_a = _make_mock_image([-1.0, 1.0], [-1.0, 1.0])
    image_a.get_psf.return_value = 2.0
    image_b = _make_mock_image([0.5], [-0.5])
    image_b.get_psf.return_value = 1.5

    mock_ray = MagicMock()
    mock_ray.psf_images = {(0.0, 0.0): image_a, (1.0, 0.0): image_b}
    mock_ray._results = None

    with (
        patch(
            "simtools.ray_tracing.optics_validation.plot_ray_tracing_psf"
            ".create_annotated_psf_image_figure",
            return_value=MagicMock(),
        ) as mock_create,
        patch(
            "simtools.ray_tracing.optics_validation.visualize.save_figures_to_single_document"
        ) as mock_save,
    ):
        optics_validation._plot_psf_images(mock_ray, "LSTN-01", Path("out.pdf"))

    assert mock_create.call_count == 2
    mock_save.assert_called_once()
    assert mock_save.call_args.kwargs["close"] is True


def test_plot_psf_images_passes_telescope_name_to_figure():
    image = _make_mock_image([1.0], [-1.0])
    image.get_psf.return_value = 2.0

    mock_ray = MagicMock()
    mock_ray.psf_images = {(0.5, 0.0): image}
    mock_ray._results = None

    with (
        patch(
            "simtools.ray_tracing.optics_validation.plot_ray_tracing_psf"
            ".create_annotated_psf_image_figure",
            return_value=MagicMock(),
        ) as mock_create,
        patch("simtools.ray_tracing.optics_validation.visualize.save_figures_to_single_document"),
    ):
        optics_validation._plot_psf_images(mock_ray, "MSTS-01", Path("out.pdf"))

    assert mock_create.call_args.kwargs["telescope_name"] == "MSTS-01"


def test_plot_psf_images_in_degrees_uses_eff_flen():
    image = _make_mock_image([100.0, -100.0], [50.0, -50.0])
    image.get_psf.return_value = 4.0

    mock_ray = MagicMock()
    mock_ray.psf_images = {(1.0, 0.0): image}
    mock_ray._results = _make_results(off_x=[0.0, 1.0], off_y=[0.0, 0.0], eff_flen=[np.nan, 2000.0])

    captured = {}

    def capture_figure(data, **kwargs):
        captured["psf_unit"] = kwargs["psf"].unit
        return MagicMock()

    with (
        patch(
            "simtools.ray_tracing.optics_validation.plot_ray_tracing_psf"
            ".create_annotated_psf_image_figure",
            side_effect=capture_figure,
        ),
        patch("simtools.ray_tracing.optics_validation.visualize.save_figures_to_single_document"),
    ):
        optics_validation._plot_psf_images(
            mock_ray, "LSTN-01", Path("out.pdf"), plot_in_degrees=True
        )

    assert captured["psf_unit"] == u.deg
