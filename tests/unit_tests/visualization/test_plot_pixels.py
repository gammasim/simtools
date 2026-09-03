#!/usr/bin/python3

from pathlib import Path
from unittest import mock

import astropy.units as u
import numpy as np
import pytest
from matplotlib.figure import Figure

from simtools.visualization import plot_pixels


@pytest.mark.parametrize(
    ("rotate_angle", "expected_extra_kwargs"),
    [
        (None, {}),
        (10.0 * u.deg, {"rotate_angle": 10.0 * u.deg}),
    ],
)
@mock.patch("simtools.visualization.plot_pixels.plot_pixel_layout_from_file")
@mock.patch("simtools.visualization.plot_pixels.visualize.save_figure")
def test_plot_rotate_angle_kwarg(mock_save, mock_plot_layout, rotate_angle, expected_extra_kwargs):
    config = {
        "parameter": "pixel_layout",
        "site": "North",
        "telescope": "LSTN-01",
        "parameter_version": "1.0.0",
        "model_version": "6.0.0",
        "file_name": "test.dat",
    }
    if rotate_angle is not None:
        config["rotate_angle"] = rotate_angle

    model_reader = mock.MagicMock()
    mock_fig = mock.MagicMock()
    mock_plot_layout.return_value = mock_fig

    with mock.patch("simtools.visualization.plot_pixels.io_handler.IOHandler") as mock_io:
        mock_io_instance = mock.MagicMock()
        mock_io.return_value = mock_io_instance
        mock_io_instance.get_output_directory.return_value = Path("/test/path")

        plot_pixels.plot(config, "test.png", model_reader=model_reader)

        expected_path = Path("/test/path/test.dat")
        mock_plot_layout.assert_called_once_with(
            expected_path,
            "LSTN-01",
            pixels_id_to_print=80,
            focal_length=1.0,
            **expected_extra_kwargs,
        )
        mock_save.assert_called_once_with(mock_fig, "test.png")


@mock.patch("simtools.visualization.plot_pixels.plot_pixel_layout_from_configuration")
@mock.patch("simtools.visualization.plot_pixels.visualize.save_figure")
def test_plot_camera_configuration(mock_save, mock_plot_layout):
    config = {
        "parameter": "camera_configuration",
        "site": "North",
        "telescope": "LSTN-01",
        "model_version": "6.0.0",
    }
    model_reader = mock.MagicMock()
    model_reader.get_model_parameters.return_value = {
        "camera_configuration": {
            "value": {
                "camera_config_rotate": "camera_config_rotate",
                "camera_pixel_types": "camera_pixel_types",
                "camera_pixel_layout": "camera_pixel_layout",
                "camera_trigger_groups": "camera_trigger_groups",
                "camera_trigger_members": "camera_trigger_members",
            }
        },
        "camera_config_rotate": {"value": 0.0},
        "camera_pixel_types": {"value": [{"funnel_shape": 1, "funnel_diameter_cm": 1.0}]},
        "camera_pixel_layout": {"value": "layout.ecsv"},
        "camera_trigger_groups": {"value": "groups.ecsv"},
        "camera_trigger_members": {"value": "members.ecsv"},
    }
    model_reader.get_parameter_table.side_effect = [
        mock_table([{"pixel_id": 0, "x_cm": 0.0, "y_cm": 0.0, "enabled": 1}]),
        mock_table([]),
        mock_table([]),
    ]
    mock_fig = mock.MagicMock()
    mock_plot_layout.return_value = mock_fig

    with mock.patch("simtools.visualization.plot_pixels.io_handler.IOHandler") as mock_io:
        mock_io.return_value.get_output_directory.return_value = Path("/test/path")
        plot_pixels.plot(config, "test.png", model_reader=model_reader)

    mock_plot_layout.assert_called_once()
    assert mock_plot_layout.call_args.args[0]["pixels"][0]["pixel_id"] == 0
    mock_save.assert_called_once_with(mock_fig, "test.png")


def mock_table(rows):
    """Return a minimal table-like object for component plotting tests."""
    table = mock.MagicMock()
    table.colnames = list(rows[0]) if rows else []
    table.__iter__.return_value = iter([mock_row(row) for row in rows])
    return table


def mock_row(values):
    """Return a row-like mapping for component plotting tests."""
    row = mock.MagicMock()
    row.__getitem__.side_effect = values.__getitem__
    return row


def test_plot_pixel_layout_from_file_smoke():
    camera = mock.MagicMock()
    camera.telescope_name = "LSTN-01"
    camera.pixels = {
        "x": [0.0, 1.0],
        "y": [0.0, 1.0],
        "pix_id": [0, 1],
        "pix_on": [True, True],
        "pixel_shape": 1,
        "pixel_diameter": 1.0,
        "rotate_angle": 0.0,
        "orientation": 0,
    }

    with (
        mock.patch("simtools.visualization.plot_pixels.Camera") as mock_camera_cls,
        mock.patch(
            "simtools.visualization.plot_pixels._apply_telescope_specific_pixel_transform"
        ) as mock_transform,
        mock.patch("simtools.visualization.plot_pixels.create_pixel_patches_by_type") as mock_p,
        mock.patch("simtools.visualization.plot_pixels.add_pixel_patch_collections") as mock_add,
    ):
        mock_camera_cls.return_value = camera
        mock_transform.side_effect = lambda cam, camera_config_file, rotate_angle=None: (
            cam.pixels.__setitem__("plot_rotate_angle", (90.0 * u.deg).to(u.rad).value)
        )
        mock_p.return_value = ([], [], [])

        fig = plot_pixels.plot_pixel_layout_from_file(
            "dummy.dat",
            "LSTN-01",
            pixels_id_to_print=1,
            title="Test",
            xtitle="X",
            ytitle="Y",
        )

        assert isinstance(fig, Figure)
        mock_camera_cls.assert_called_once()
        mock_transform.assert_called_once()
        mock_add.assert_called_once()


@pytest.mark.parametrize(
    (
        "telescope_name",
        "pixel_shape",
        "rotate_angle_arg",
        "expected_rot_deg",
        "expect_y_flip",
        "expected_orientation",
    ),
    [
        ("LSTN-01", 3, None, -100.0, True, 30),
        ("SSTS-01", 1, 10.0, 80.0, False, 0),
    ],
)
def test_apply_telescope_specific_pixel_transform(
    telescope_name,
    pixel_shape,
    rotate_angle_arg,
    expected_rot_deg,
    expect_y_flip,
    expected_orientation,
):
    camera = mock.MagicMock()
    camera.telescope_name = telescope_name
    camera.pixels = {"pixel_shape": pixel_shape}

    raw_pixels = {
        "x": np.array([1.0, 0.0]),
        "y": np.array([0.0, 1.0]),
        "rotate_angle": np.deg2rad(10.0),
        "pix_id": [0, 1],
        "pix_on": [True, True],
        "pixel_shape": pixel_shape,
        "pixel_diameter": 1.0,
    }

    with mock.patch("simtools.visualization.plot_pixels.Camera.read_pixel_list") as mock_read:
        mock_read.return_value = raw_pixels
        plot_pixels._apply_telescope_specific_pixel_transform(
            camera,
            camera_config_file="dummy.dat",
            rotate_angle=rotate_angle_arg,
        )

    rot = (expected_rot_deg * u.deg).to(u.rad).value
    x_pos = np.array([1.0, 0.0])
    y_pos = np.array([0.0, 1.0])
    if expect_y_flip:
        y_pos = -y_pos

    expected_x = x_pos * np.cos(rot) - y_pos * np.sin(rot)
    expected_y = y_pos * np.cos(rot) + x_pos * np.sin(rot)

    assert np.allclose(camera.pixels["x"], expected_x)
    assert np.allclose(camera.pixels["y"], expected_y)
    assert np.isclose(camera.pixels["plot_rotate_angle"], rot)
    assert camera.pixels["orientation"] == expected_orientation
