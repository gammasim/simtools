"""Unit tests for simtools.ray_tracing.optics_validation."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import astropy.units as u
import numpy as np
import pytest

from simtools.ray_tracing import optics_validation


def test_load_data_normalizes_and_converts_radius():
    """Test loading/scaling of cumulative PSF data."""
    input_data = np.array(
        [(10.0, 2.0), (20.0, -4.0)],
        dtype=[("Radius [cm]", "f8"), ("Relative intensity", "f8")],
    )

    with patch("simtools.ray_tracing.optics_validation.np.loadtxt", return_value=input_data):
        data = optics_validation.load_data("dummy.dat")

    np.testing.assert_allclose(data["Radius [cm]"], [1.0, 2.0])
    np.testing.assert_allclose(data["Relative intensity"], [0.5, -1.0])


def test_validate_cumulative_psf_raises_without_radius_data():
    """Test cumulative validation fails if no measured radius data are provided."""
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
    """Test cumulative validation success path and produced plots."""
    args_dict = {
        "site": "North",
        "telescope": "LSTN-01",
        "model_version": "5.0.0",
        "zenith_angle": 20.0 * u.deg,
        "source_distance": 10.0 * u.km,
        "data": "measured.dat",
        "model_path": str(tmp_test_directory),
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
    fig_2d.gca.return_value = MagicMock()

    with (
        patch(
            "simtools.ray_tracing.optics_validation.initialize_simulation_models",
            return_value=(mock_tel_model, mock_site_model, None),
        ),
        patch("simtools.ray_tracing.optics_validation.RayTracing", return_value=mock_ray),
        patch("simtools.ray_tracing.optics_validation.gen.find_file", return_value="measured.dat"),
        patch("simtools.ray_tracing.optics_validation.load_data", return_value=measured),
        patch("simtools.ray_tracing.optics_validation.visualize.plot_1d", return_value=fig_1d),
        patch("simtools.ray_tracing.optics_validation.visualize.plot_hist_2d", return_value=fig_2d),
        patch("simtools.ray_tracing.optics_validation.plt.Circle", return_value=MagicMock()),
        patch("simtools.ray_tracing.optics_validation.visualize.save_figure") as mock_save,
    ):
        optics_validation.validate_cumulative_psf(app_context)

    assert mock_ray.simulate.call_count == 1
    assert mock_ray.analyze.call_count == 1
    assert mock_save.call_count == 2


def test_validate_optics_no_images(tmp_test_directory):
    """Test optics validation without image PDF generation."""
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
        patch("simtools.ray_tracing.optics_validation.plt") as mock_plt,
        patch("simtools.ray_tracing.optics_validation.visualize.save_figure"),
    ):
        mock_plt.figure.return_value = MagicMock()

        optics_validation.validate_optics(app_context)

        mock_ray.simulate.assert_called_once_with(test=True, force=False)
        mock_ray.analyze.assert_called_once_with(force=True)
        assert mock_ray.plot.call_count == 4


def test_validate_optics_with_images_and_default_label(tmp_test_directory):
    """Test optics validation image-PDF branch and default label behavior."""
    args_dict = {
        "site": "North",
        "telescope": "LSTN-01",
        "model_version": "5.0.0",
        "zenith_angle": 20.0 * u.deg,
        "source_distance": 10.0 * u.km,
        "max_offset": 1.0 * u.deg,
        "offset_step": 0.5 * u.deg,
        "offset_file": None,
        "offset_directions": None,
        "plot_images": True,
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

    mock_pdf = MagicMock()
    ax = MagicMock()

    with (
        patch(
            "simtools.ray_tracing.optics_validation.initialize_simulation_models",
            return_value=(mock_tel_model, mock_site_model, None),
        ),
        patch(
            "simtools.ray_tracing.optics_validation.RayTracing", return_value=mock_ray
        ) as mock_rt,
        patch("simtools.ray_tracing.optics_validation.PdfPages", return_value=mock_pdf),
        patch("simtools.ray_tracing.optics_validation.plt.figure", return_value=MagicMock()),
        patch("simtools.ray_tracing.optics_validation.plt.gca", return_value=ax),
        patch("simtools.ray_tracing.optics_validation.plt.close"),
        patch("simtools.ray_tracing.optics_validation.visualize.save_figure") as mock_save,
    ):
        optics_validation.validate_optics(app_context)

    rt_kwargs = mock_rt.call_args.kwargs
    assert rt_kwargs["label"] == "validate_optics"
    assert rt_kwargs["offset_directions"] is None
    assert len(rt_kwargs["off_axis_angle"]) == 3

    assert mock_save.call_count == 4
    assert mock_pdf.savefig.call_count == 2
    mock_pdf.close.assert_called_once()
