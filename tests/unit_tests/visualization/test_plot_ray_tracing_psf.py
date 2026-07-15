"""Unit tests for simtools.visualization.plot_ray_tracing_psf."""

from unittest.mock import MagicMock, patch

import numpy as np

from simtools.visualization import plot_ray_tracing_psf


def test_create_cumulative_psf_figure_draws_curve_and_markers():
    """Test cumulative PSF helper draws the line and both markers."""
    data = np.array(
        [(0.0, 0.0), (1.0, 0.5), (2.0, 1.0)],
        dtype=[("Radius [cm]", "f8"), ("Cumulative PSF", "f8")],
    )
    mock_fig = MagicMock()
    mock_ax = MagicMock()

    with patch("simtools.visualization.plot_ray_tracing_psf.plt.subplots") as mock_subplots:
        mock_subplots.return_value = (mock_fig, mock_ax)
        fig, ax = plot_ray_tracing_psf.create_cumulative_psf_figure(
            data,
            radius_key="Radius [cm]",
            cumulative_key="Cumulative PSF",
            containment_radius_cm=1.5,
            psf_diameter_cm=4.0,
            color="blue",
        )

    assert fig == mock_fig
    assert ax == mock_ax
    mock_ax.plot.assert_called_once()
    assert mock_ax.axvline.call_count == 2


def test_create_psf_image_figure_draws_histogram_circle_and_axes():
    """Test PSF image helper draws the histogram, circle, and reference axes."""
    data = np.array([(0.0, 0.0), (1.0, -1.0)], dtype=[("X", "f8"), ("Y", "f8")])
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_ax.figure = mock_fig

    mock_fig.gca.return_value = mock_ax
    with patch(
        "simtools.visualization.plot_ray_tracing_psf.visualize.plot_hist_2d",
        return_value=mock_fig,
    ) as mock_plot_hist:
        fig, ax = plot_ray_tracing_psf.create_psf_image_figure(
            data,
            containment_radius_cm=2.0,
            center=(0.0, 0.0),
            bins=80,
            image_range=[[-1.0, 1.0], [-1.0, 1.0]],
            psf_kwargs={"color": "k", "fill": False},
        )

    assert fig == mock_fig
    assert ax == mock_ax
    mock_plot_hist.assert_called_once_with(
        data,
        ax=None,
        bins=80,
        range=[[-1.0, 1.0], [-1.0, 1.0]],
    )
    mock_ax.add_artist.assert_called_once()
    mock_ax.axhline.assert_called_once()
    mock_ax.axvline.assert_called_once()


def test_create_annotated_psf_image_figure_adds_text():
    """Test annotated PSF image helper adds the offset/PSF label."""
    data = np.array([(0.0, 0.0), (1.0, -1.0)], dtype=[("X", "f8"), ("Y", "f8")])
    mock_fig = MagicMock()
    mock_ax = MagicMock()

    with patch("simtools.visualization.plot_ray_tracing_psf.plt.subplots") as mock_subplots:
        mock_subplots.return_value = (mock_fig, mock_ax)
        with patch(
            "simtools.visualization.plot_ray_tracing_psf.visualize.plot_hist_2d",
            return_value=mock_fig,
        ):
            fig = plot_ray_tracing_psf.create_annotated_psf_image_figure(
                data,
                off_x=0.5,
                off_y=-0.5,
                psf_cm=4.2,
                image_range=[[-1.0, 1.0], [-1.0, 1.0]],
                cmap="gist_heat_r",
            )

    assert fig == mock_fig
    mock_ax.text.assert_called_once()


def test_save_and_close_figure(mocker, tmp_test_directory):
    """Test saving and closing a PSF figure."""
    fig = mocker.Mock()
    mock_close = mocker.patch("simtools.visualization.plot_ray_tracing_psf.plt.close")
    file_name = tmp_test_directory / "plot.pdf"

    plot_ray_tracing_psf.save_and_close_figure(fig, file_name)

    fig.savefig.assert_called_once_with(file_name)
    mock_close.assert_called_once_with(fig)
