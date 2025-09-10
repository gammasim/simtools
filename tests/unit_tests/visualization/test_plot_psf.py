#!/usr/bin/python3

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from simtools.visualization import plot_psf

logger = logging.getLogger(__name__)


@pytest.fixture
def sample_psf_data():
    """Create sample PSF data for testing."""
    radius = np.linspace(0, 10, 21)
    cumulative = np.linspace(0, 1, 21)
    dtype = {"names": (plot_psf.RADIUS_CM, plot_psf.CUMULATIVE_PSF), "formats": ("f8", "f8")}
    data = np.empty(21, dtype=dtype)
    data[plot_psf.RADIUS_CM] = radius
    data[plot_psf.CUMULATIVE_PSF] = cumulative
    return data


@pytest.fixture
def sample_parameters():
    """Create sample parameter dictionary for testing."""
    return {
        "mirror_reflection_random_angle": [0.006, 0.15, 0.035],
        "mirror_align_random_horizontal": [0.005, 28.0, 0.0, 0.0],
        "mirror_align_random_vertical": [0.005, 28.0, 0.0, 0.0],
    }


@pytest.fixture
def data_to_plot(sample_psf_data):
    """Create data_to_plot dictionary for testing."""
    return {"measured": sample_psf_data}


def test__get_significance_level():
    """Test p-value significance level classification."""
    assert plot_psf._get_significance_level(0.1) == "GOOD"
    assert plot_psf._get_significance_level(0.03) == "FAIR"
    assert plot_psf._get_significance_level(0.005) == "POOR"


@pytest.mark.parametrize(
    ("use_ks_statistic", "second_metric", "p_value"),
    [(False, None, None), (True, None, 0.05), (False, 0.456, None), (False, 0.456, 0.05)],
)
def test__format_metric_text(use_ks_statistic, second_metric, p_value):
    """Test metric text formatting for different modes."""
    result = plot_psf._format_metric_text(3.5, 0.123, p_value, use_ks_statistic, second_metric)
    assert "D80 = 3.50000 cm" in result
    assert "0.123" in result


def test__create_base_plot_figure(data_to_plot, sample_psf_data):
    """Test base plot figure creation with different scenarios."""
    with patch("simtools.visualization.plot_psf.visualize.plot_1d") as mock_plot:
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_fig.get_axes.return_value = [mock_ax]
        mock_plot.return_value = mock_fig

        # Test with simulated data
        fig, ax = plot_psf._create_base_plot_figure(data_to_plot, simulated_data=sample_psf_data)
        assert fig == mock_fig
        assert ax == mock_ax

        # Test error handling
        mock_plot.side_effect = ValueError("Plot error")
        with pytest.raises(ValueError, match="Plot error"):
            plot_psf._create_base_plot_figure(data_to_plot)


def test__build_parameter_title(sample_parameters):
    """Test parameter title building."""
    title = plot_psf._build_parameter_title(sample_parameters, is_best=True)
    assert "* reflection" in title
    assert "0.00600" in title

    title = plot_psf._build_parameter_title(sample_parameters, is_best=False)
    assert "* reflection" not in title


def test__add_metric_text_box():
    """Test metric text box addition."""
    mock_ax = MagicMock()
    plot_psf._add_metric_text_box(mock_ax, "Test metrics", is_best=True)
    mock_ax.text.assert_called_once()

    mock_ax.reset_mock()
    plot_psf._add_metric_text_box(mock_ax, "Test metrics", is_best=False)
    mock_ax.text.assert_called_once()


def test__add_plot_annotations(sample_parameters):
    """Test plot annotation addition."""
    mock_ax = MagicMock()
    mock_fig = MagicMock()

    # Test best parameter case (includes footnote)
    plot_psf._add_plot_annotations(mock_ax, mock_fig, sample_parameters, 3.5, 0.123, True, 0.05)
    mock_ax.set_title.assert_called_once()
    mock_ax.text.assert_called_once()
    mock_fig.text.assert_called_once()


def test_create_psf_parameter_plot(data_to_plot, sample_parameters):
    """Test PSF parameter plot creation."""
    with (
        patch("simtools.visualization.plot_psf._create_base_plot_figure") as mock_base,
        patch("simtools.visualization.plot_psf._add_plot_annotations") as mock_annotations,
        patch("matplotlib.pyplot.clf"),
    ):
        mock_fig, mock_ax = MagicMock(), MagicMock()
        mock_base.return_value = (mock_fig, mock_ax)
        mock_pdf_pages = MagicMock()

        plot_psf.create_psf_parameter_plot(
            data_to_plot, sample_parameters, 3.5, 0.123, True, mock_pdf_pages
        )

        mock_base.assert_called_once()
        mock_annotations.assert_called_once()
        mock_pdf_pages.savefig.assert_called_once()


def test_create_detailed_parameter_plot(data_to_plot, sample_parameters, sample_psf_data):
    """Test detailed parameter plot creation with error handling."""
    with (
        patch("simtools.visualization.plot_psf._create_base_plot_figure") as mock_base,
        patch("simtools.visualization.plot_psf._add_plot_annotations") as mock_annotations,
        patch("simtools.visualization.plot_psf.logger") as mock_logger,
        patch("matplotlib.pyplot.clf"),
    ):
        mock_fig, mock_ax = MagicMock(), MagicMock()
        mock_base.return_value = (mock_fig, mock_ax)
        mock_pdf_pages = MagicMock()

        # Test successful case
        plot_psf.create_detailed_parameter_plot(
            sample_parameters, 0.123, 3.5, sample_psf_data, data_to_plot, True, mock_pdf_pages, 0.05
        )
        mock_annotations.assert_called_once()

        # Test error case for 100% coverage
        mock_base.side_effect = ValueError("Plot creation failed")
        result = plot_psf.create_detailed_parameter_plot(
            sample_parameters, 0.123, 3.5, sample_psf_data, data_to_plot, True, mock_pdf_pages
        )
        assert result is None
        mock_logger.error.assert_called()

        # Test no simulated data case
        mock_base.side_effect = None
        plot_psf.create_detailed_parameter_plot(
            sample_parameters, 0.123, 3.5, None, data_to_plot, True, mock_pdf_pages
        )
        mock_logger.warning.assert_called()


def test_create_parameter_progression_plots(data_to_plot, sample_parameters, sample_psf_data):
    """Test parameter progression plots creation."""
    results = [
        (sample_parameters, 0.123, 0.05, 3.5, sample_psf_data),
        (sample_parameters, 0.100, None, 3.2, None),  # Test with None data (skipped)
    ]

    with patch("simtools.visualization.plot_psf.create_detailed_parameter_plot") as mock_create:
        mock_pdf_pages = MagicMock()
        plot_psf.create_parameter_progression_plots(
            results, sample_parameters, data_to_plot, mock_pdf_pages
        )
        assert mock_create.call_count == 1  # Only first one called, second skipped due to None data


def test_create_gradient_descent_convergence_plot(tmp_path):
    """Test gradient descent convergence plot creation."""
    gd_results = [
        ({"param1": 0.1}, 0.1, 0.05, 3.5, None),
        ({"param1": 0.05}, 0.05, 0.03, 3.2, None),
    ]
    output_file = tmp_path / "convergence.png"

    with patch("matplotlib.pyplot.savefig") as mock_save, patch("matplotlib.pyplot.close"):
        # Test RMSD mode
        plot_psf.create_gradient_descent_convergence_plot(
            gd_results, 0.01, output_file, use_ks_statistic=False
        )
        mock_save.assert_called_once()

        mock_save.reset_mock()
        # Test KS statistic mode
        plot_psf.create_gradient_descent_convergence_plot(
            gd_results, 0.01, output_file, use_ks_statistic=True
        )
        mock_save.assert_called_once()


def test_create_monte_carlo_uncertainty_plot(tmp_path):
    """Test Monte Carlo uncertainty plot creation."""
    # RMSD mode results (no p-values)
    mc_results_rmsd = (
        0.1,
        0.02,
        [0.08, 0.09, 0.11],
        None,
        0,
        [None, None, None],
        3.2,
        0.1,
        [3.1, 3.2, 3.3],
    )
    # KS mode results (with p-values)
    mc_results_ks = (
        0.1,
        0.02,
        [0.08, 0.09, 0.11],
        0.05,
        0.01,
        [0.04, 0.05, 0.06],
        3.2,
        0.1,
        [3.1, 3.2, 3.3],
    )
    output_file = tmp_path / "monte_carlo"

    with patch("matplotlib.pyplot.savefig") as mock_save, patch("matplotlib.pyplot.close"):
        # Test RMSD mode
        plot_psf.create_monte_carlo_uncertainty_plot(
            mc_results_rmsd, output_file, use_ks_statistic=False
        )
        assert mock_save.call_count == 2  # PDF and PNG

        mock_save.reset_mock()
        # Test KS mode
        plot_psf.create_monte_carlo_uncertainty_plot(
            mc_results_ks, output_file, use_ks_statistic=True
        )
        assert mock_save.call_count == 2


def test_create_d80_vs_offaxis_plot(sample_parameters, tmp_path):
    """Test D80 vs off-axis angle plot creation."""
    mock_telescope_model = MagicMock()
    mock_site_model = MagicMock()
    args_dict = {"simtel_path": "/path/to/simtel", "zenith": 20.0, "src_distance": 10.0}

    # Mock RayTracing and its methods
    with (
        patch("simtools.visualization.plot_psf.RayTracing") as mock_ray_class,
        patch("matplotlib.pyplot.savefig") as mock_save,
        patch("matplotlib.pyplot.close"),
        patch("numpy.linspace") as mock_linspace,
    ):
        mock_ray = MagicMock()
        mock_ray_class.return_value = mock_ray
        mock_ray.images.return_value = [MagicMock()]
        mock_ray.images.return_value[0].get_psf.return_value = 3.5
        mock_linspace.return_value = np.array([0, 1, 2])

        plot_psf.create_d80_vs_offaxis_plot(
            mock_telescope_model, mock_site_model, args_dict, sample_parameters, tmp_path
        )

        # Verify telescope parameters were applied and simulation was run
        mock_telescope_model.change_multiple_parameters.assert_called_once_with(**sample_parameters)
        assert mock_save.call_count >= 1  # At least one save call
