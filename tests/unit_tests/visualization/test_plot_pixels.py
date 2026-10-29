#!/usr/bin/python3

from unittest import mock

import astropy.units as u
import numpy as np
import pytest
from matplotlib.figure import Figure

from simtools.visualization import plot_pixels


def _camera(telescope_name="LSTN-01", pixel_shape=1):
    camera = mock.MagicMock()
    camera.telescope_name = telescope_name
    camera.pixels = {
        "x": [0.0, 1.0],
        "y": [0.0, 1.0],
        "pix_id": [0, 1],
        "pix_on": [True, True],
        "pixel_shape": pixel_shape,
        "pixel_diameter": 1.0,
        "rotate_angle": 0.0,
        "orientation": 0,
    }
    return camera


def _raw_pixels(pixel_shape=1):
    return {
        "x": np.array([1.0, 0.0]),
        "y": np.array([0.0, 1.0]),
        "rotate_angle": np.deg2rad(10.0),
        "pixel_shape": pixel_shape,
    }


@pytest.mark.parametrize(
    ("rotate_angle", "expected_rotate_angle"),
    [(None, None), (10.0 * u.deg, 10.0 * u.deg)],
)
@mock.patch("simtools.visualization.plot_pixels.visualize.save_figure")
@mock.patch("simtools.visualization.plot_pixels.plt.close")
@mock.patch("simtools.visualization.plot_pixels.plot_pixel_layout")
def test_plot_passes_rotate_angle_and_saves(
    mock_layout, mock_close, mock_save, rotate_angle, expected_rotate_angle
):
    camera, raw_pixels = _camera(), _raw_pixels()
    mock_fig = mock.MagicMock()
    mock_layout.return_value = mock_fig

    plot_pixels.plot("test.png", camera, raw_pixels, rotate_angle=rotate_angle)

    mock_layout.assert_called_once_with(
        camera, raw_pixels, rotate_angle=expected_rotate_angle, pixels_id_to_print=80
    )
    mock_save.assert_called_once_with(mock_fig, "test.png")
    mock_close.assert_called_once_with(mock_fig)


def test_raw_pixels_from_configuration():
    configuration = {
        "rotate": 90.0,
        "pixels": [
            {"pixel_id": 0, "x_cm": 1.0, "y_cm": 2.0, "enabled": 1},
            {"pixel_id": 1, "x_cm": 3.0, "y_cm": 4.0, "enabled": 1},
        ],
    }

    raw = plot_pixels.raw_pixels_from_configuration(configuration)

    assert raw["x"].tolist() == [1.0, 3.0]
    assert raw["y"].tolist() == [2.0, 4.0]
    assert raw["rotate_angle"] == pytest.approx(np.pi / 2)


def test_resolve_camera_components():
    model_reader = mock.MagicMock()
    parameters = {
        "camera_rotate": {"value": 0.0},
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

    components = plot_pixels.resolve_camera_components(model_reader, parameters)

    assert components["pixels"][0]["pixel_id"] == 0
    assert components["rotate"] == pytest.approx(0.0)
    assert model_reader.get_parameter_table.call_args_list[0].args[0] == {"value": "layout.ecsv"}


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


@mock.patch("simtools.visualization.plot_pixels.add_pixel_patch_collections")
@mock.patch("simtools.visualization.plot_pixels.create_pixel_patches_by_type")
def test_plot_pixel_layout_smoke(mock_patches, mock_add):
    camera = _camera()
    mock_patches.return_value = ([], [], [])

    fig = plot_pixels.plot_pixel_layout(camera, _raw_pixels(), pixels_id_to_print=1)

    assert isinstance(fig, Figure)
    mock_add.assert_called_once()
    assert "plot_rotate_angle" in camera.pixels


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
    camera = _camera(telescope_name=telescope_name, pixel_shape=pixel_shape)
    raw_pixels = _raw_pixels(pixel_shape=pixel_shape)

    plot_pixels._apply_telescope_specific_pixel_transform(camera, raw_pixels, rotate_angle_arg)

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
