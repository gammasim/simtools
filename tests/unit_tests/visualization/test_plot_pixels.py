#!/usr/bin/python3

from pathlib import Path
from unittest import mock

import astropy.units as u
from matplotlib.figure import Figure

from simtools.visualization import plot_pixels

# Constants
DUMMY_DAT_PATH = "tests/resources/pixel_layout.dat"


@mock.patch("simtools.visualization.plot_pixels.plot_pixel_layout_from_file")
@mock.patch("simtools.visualization.plot_pixels.visualize.save_figure")
@mock.patch("simtools.visualization.plot_pixels.db_handler.DatabaseHandler")
def test_plot(mock_db_handler, mock_save, mock_plot_layout):
    """Test the main plot function."""
    config = {
        "parameter": "pixel_layout",
        "site": "North",
        "telescope": "LSTN-01",
        "parameter_version": "1.0.0",
        "model_version": "6.0.0",
        "file_name": "test.dat",
    }

    mock_db_instance = mock.MagicMock()
    mock_db_handler.return_value = mock_db_instance

    mock_fig = mock.MagicMock()
    mock_plot_layout.return_value = mock_fig

    with mock.patch("simtools.visualization.plot_pixels.io_handler.IOHandler") as mock_io:
        mock_io_instance = mock.MagicMock()
        mock_io.return_value = mock_io_instance
        mock_io_instance.get_output_directory.return_value = Path("/test/path")

        plot_pixels.plot(config, "test.png")

        mock_db_instance.export_model_file.assert_called_once_with(
            parameter="pixel_layout",
            site="North",
            array_element_name="LSTN-01",
            parameter_version="1.0.0",
            model_version="6.0.0",
            export_file_as_table=False,
        )

        expected_path = Path("/test/path/test.dat")
        mock_plot_layout.assert_called_once_with(
            expected_path,
            "LSTN-01",
            pixels_id_to_print=80,
            focal_length=1.0,
        )
        mock_save.assert_called_once_with(mock_fig, "test.png")


def test_plot_pixel_layout_from_file():
    """Test plot_pixel_layout_from_file using real config file."""
    fig = plot_pixels.plot_pixel_layout_from_file(
        DUMMY_DAT_PATH,
        "SSTS-01",
        pixels_id_to_print=1,
        title="Test",
        xtitle="X",
        ytitle="Y",
    )

    assert isinstance(fig, Figure)


def test_add_coordinate_axes():
    """Test coordinate axes addition."""
    mock_ax = mock.MagicMock()
    mock_ax.get_xlim.return_value = (-10, 10)
    mock_ax.get_ylim.return_value = (-10, 10)
    rotation = 90.0 * u.deg

    plot_pixels._add_coordinate_axes(mock_ax, rotation)

    assert mock_ax.arrow.call_count == 4
    assert mock_ax.text.call_count == 4


def test_configure_plot_calls_setup():
    """Test configure plot calls shared axis setup."""
    camera = mock.MagicMock()
    camera.pixels = {
        "x": [0.0, 1.0],
        "y": [0.0, 1.0],
        "rotate_angle": 0.0,
    }
    ax = mock.MagicMock()
    ax.get_xlim.return_value = (-0.5, 1.5)
    ax.get_ylim.return_value = (-0.5, 1.5)

    with mock.patch(
        "simtools.visualization.plot_pixels.setup_camera_axis_properties"
    ) as mock_setup:
        plot_pixels._configure_plot(ax, camera, title="Test")

    mock_setup.assert_called_once()
