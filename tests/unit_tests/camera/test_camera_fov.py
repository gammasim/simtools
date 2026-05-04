"""Unit tests for simtools.camera.camera_fov."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from simtools.camera.camera_fov import parse_pixel_ids_to_print, run_camera_fov_validation


def _mock_camera(n_pixels=1855):
    camera = MagicMock()
    camera.get_number_of_pixels.return_value = n_pixels
    camera.calc_fov.return_value = (4.5, 120.0)
    return camera


def test_parse_pixel_ids_to_print_integer():
    camera = _mock_camera()
    assert parse_pixel_ids_to_print(50, camera) == 50


def test_parse_pixel_ids_to_print_zero_returns_minus_one():
    camera = _mock_camera()
    assert parse_pixel_ids_to_print(0, camera) == -1


def test_parse_pixel_ids_to_print_all():
    camera = _mock_camera(n_pixels=1855)
    assert parse_pixel_ids_to_print("All", camera) == 1855


def test_parse_pixel_ids_to_print_all_case_insensitive():
    camera = _mock_camera(n_pixels=300)
    assert parse_pixel_ids_to_print("all", camera) == 300


def test_parse_pixel_ids_to_print_invalid_raises():
    camera = _mock_camera()
    with pytest.raises(ValueError, match="must be integer or 'All'"):
        parse_pixel_ids_to_print("invalid", camera)


def test_run_camera_fov_validation(tmp_test_directory):
    args_dict = {
        "site": "North",
        "telescope": "LSTN-01",
        "model_version": "5.0.0",
        "camera_in_sky_coor": False,
        "print_pixels_id": 10,
    }
    io_handler = MagicMock()
    io_handler.get_output_directory.return_value = Path(str(tmp_test_directory))
    app_context = SimpleNamespace(args=args_dict, io_handler=io_handler)

    mock_camera = _mock_camera()
    mock_tel_model = MagicMock()
    mock_tel_model.name = "LSTN-01"
    mock_tel_model.camera = mock_camera
    mock_tel_model.get_telescope_effective_focal_length.return_value = 2800.0

    with (
        patch("simtools.camera.camera_fov.TelescopeModel", return_value=mock_tel_model) as mock_tm,
        patch("simtools.camera.camera_fov.plot_camera.plot_pixel_layout") as mock_plot,
        patch("simtools.camera.camera_fov.visualize.save_figure") as mock_save,
    ):
        run_camera_fov_validation(app_context)

        mock_tm.assert_called_once_with(
            site="North",
            telescope_name="LSTN-01",
            model_version="5.0.0",
            label="validate_camera_fov",
        )
        mock_tel_model.export_model_files.assert_called_once()
        mock_camera.calc_fov.assert_called_once()
        mock_plot.assert_called_once_with(mock_camera, False, 10)
        mock_save.assert_called_once()
