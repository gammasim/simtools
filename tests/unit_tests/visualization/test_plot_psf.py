#!/usr/bin/python3

import logging
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from simtools.visualization import plot_psf

logger = logging.getLogger(__name__)


@pytest.fixture
def sample_psf_data():
    """Create sample PSF data for testing."""
    radius = np.linspace(0, 10, 21)
    cumulative = np.linspace(0, 1, 21)
    dtype = {"names": (plot_psf.RADIUS, plot_psf.CUMULATIVE_PSF), "formats": ("f8", "f8")}
    data = np.empty(21, dtype=dtype)
    data[plot_psf.RADIUS] = radius
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


def test_get_significance_label():
    """Test p-value significance label classification."""
    assert plot_psf.get_significance_label(0.1) == "GOOD"
    assert plot_psf.get_significance_label(0.03) == "FAIR"
    assert plot_psf.get_significance_label(0.005) == "POOR"


def test_get_psf_diameter_label():
    """Test PSF diameter label generation for different fractions and units."""
    # Test D80 with cm
    assert plot_psf.get_psf_diameter_label(0.8, "cm") == "D80 (cm)"

    # Test D95 with degrees
    assert plot_psf.get_psf_diameter_label(0.95, "degrees") == "D95 (degrees)"

    # Test D90 without unit
    assert plot_psf.get_psf_diameter_label(0.9, "") == "D90"

    # Test default cm unit
    assert plot_psf.get_psf_diameter_label(0.85) == "D85 (cm)"


@pytest.mark.parametrize(
    ("use_ks_statistic", "second_metric", "p_value", "fraction"),
    [
        (False, None, None, 0.8),
        (True, None, 0.05, 0.8),
        (False, 0.456, None, 0.8),
        (False, 0.456, 0.05, 0.95),
    ],
)
def test__format_metric_text(use_ks_statistic, second_metric, p_value, fraction):
    """Test metric text formatting for different modes and fractions."""
    result = plot_psf._format_metric_text(
        3.5, 0.123, fraction, p_value, use_ks_statistic, second_metric
    )
    expected_label = f"D{int(fraction * 100)}"
    assert f"{expected_label} = 3.50000 cm" in result
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
            data_to_plot, sample_parameters, 3.5, 0.123, True, mock_pdf_pages, fraction=0.8
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
            sample_parameters,
            0.123,
            3.5,
            sample_psf_data,
            data_to_plot,
            True,
            mock_pdf_pages,
            fraction=0.8,
            p_value=0.05,
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

    with patch("matplotlib.pyplot.savefig") as mock_save:
        # Test RMSD mode
        plot_psf.create_gradient_descent_convergence_plot(
            gd_results, 0.01, output_file, fraction=0.8, use_ks_statistic=False
        )
        plt.close("all")
        mock_save.assert_called_once()

        mock_save.reset_mock()
        # Test KS statistic mode
        plot_psf.create_gradient_descent_convergence_plot(
            gd_results, 0.01, output_file, fraction=0.8, use_ks_statistic=True
        )
        plt.close("all")
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

    with patch("matplotlib.pyplot.savefig") as mock_save:
        # Test RMSD mode
        plot_psf.create_monte_carlo_uncertainty_plot(
            mc_results_rmsd, output_file, fraction=0.8, use_ks_statistic=False
        )
        plt.close("all")
        assert mock_save.call_count == 2  # PDF and PNG

        mock_save.reset_mock()
        # Test KS mode
        plot_psf.create_monte_carlo_uncertainty_plot(
            mc_results_ks, output_file, fraction=0.8, use_ks_statistic=True
        )
        plt.close("all")
        assert mock_save.call_count == 2


def test_create_psf_vs_offaxis_plot(sample_parameters, tmp_path):
    """Test psf vs off-axis angle plot creation."""
    mock_telescope_model = MagicMock()
    mock_telescope_model.name = "LSTN-01"
    mock_site_model = MagicMock()
    args_dict = {
        "fraction": 0.8,
        "zenith": 20,
        "src_distance": 10,
    }

    # Mock RayTracing and its methods
    with (
        patch("simtools.visualization.plot_psf.RayTracing") as mock_ray_class,
        patch("simtools.visualization.plot_psf.visualize.save_figure") as mock_save_figure,
        patch("simtools.visualization.plot_psf.np.linspace") as mock_linspace,
    ):
        mock_ray = MagicMock()
        mock_ray_class.return_value = mock_ray
        mock_linspace.return_value = np.array([0, 1, 2])

        plot_psf.create_psf_vs_offaxis_plot(
            mock_telescope_model, mock_site_model, args_dict, sample_parameters, tmp_path
        )
        plt.close("all")

        # Verify telescope parameters were applied and simulation was run

        mock_telescope_model.overwrite_parameters.assert_called_once_with(
            sample_parameters, flat_dict=True
        )
        assert mock_save_figure.call_count >= 1  # At least one save call


def test_plot_psf_histogram_returns_none_when_not_configured(tmp_path):
    args_dict = {"output_path": str(tmp_path)}
    assert plot_psf.plot_psf_histogram([10.0], [11.0], args_dict) is None


def test_plot_psf_histogram_returns_none_when_empty_after_filtering(tmp_path):
    args_dict = {"output_path": str(tmp_path), "psf_hist": "hist.png"}
    assert plot_psf.plot_psf_histogram([np.nan], [11.0], args_dict) is None
    assert plot_psf.plot_psf_histogram([10.0], [np.nan], args_dict) is None


def test_plot_psf_histogram_returns_none_when_range_invalid(tmp_path):
    args_dict = {"output_path": str(tmp_path), "psf_hist": "hist.png"}
    # with invalid range of x_max <= x_min
    assert plot_psf.plot_psf_histogram([1.0, 1.0], [1.0, 1.0], args_dict) is None


def test_plot_psf_histogram_saves_to_output_path_when_relative(tmp_path):
    args_dict = {
        "output_path": str(tmp_path),
        "psf_hist": "hist.png",
        "telescope": "LSTN-01",
        "model_version": "6.0.0",
    }

    with (
        patch("matplotlib.pyplot.subplots") as mock_subplots,
        patch("matplotlib.pyplot.close"),
    ):
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        out = plot_psf.plot_psf_histogram([10.0, 11.0], [12.0, 13.0], args_dict)
        expected = str(tmp_path / "hist.png")
        assert out == expected
        mock_fig.savefig.assert_called_once()
        saved_path = str(mock_fig.savefig.call_args.args[0])
        assert saved_path == expected


@pytest.mark.parametrize(
    ("plot_all", "expected_result"),
    [
        (True, "not_none"),  # Should return PdfPages object
        (False, None),  # Should return None
    ],
)
def test_setup_pdf_plotting(tmp_path, plot_all, expected_result):
    """Test PDF plotting setup with plot_all enabled and disabled."""
    args_dict = {"plot_all": plot_all, "fraction": 0.8}

    pdf_pages = plot_psf.setup_pdf_plotting(args_dict, tmp_path, "LSTN-01")

    if expected_result == "not_none":
        assert pdf_pages is not None
        pdf_pages.close()
    else:
        assert pdf_pages is expected_result


def test_create_optimization_plots(tmp_path, sample_psf_data, sample_parameters):
    """Test optimization plots creation with save_plots enabled and disabled."""
    mock_telescope_model = MagicMock()
    mock_telescope_model.name = "LSTN-01"
    data_to_plot = {"measured": sample_psf_data}
    gd_results = [(sample_parameters, 0.1, 0.8, 3.5, sample_psf_data)]

    # Test with save_plots=True - should create plots
    args_dict_with_plots = {"save_plots": True, "fraction": 0.8}
    with (
        patch("simtools.visualization.plot_psf.PdfPages") as mock_pdf,
        patch("simtools.visualization.plot_psf.create_psf_parameter_plot") as mock_plot,
    ):
        mock_pdf_instance = MagicMock()
        mock_pdf.return_value = mock_pdf_instance

        plot_psf.create_optimization_plots(
            args_dict_with_plots, gd_results, mock_telescope_model, data_to_plot, tmp_path
        )

        mock_pdf.assert_called_once()
        mock_plot.assert_called_once()
        mock_pdf_instance.close.assert_called_once()

    # Test with save_plots=False - should return early without creating plots
    args_dict_no_plots = {"save_plots": False, "fraction": 0.8}
    result = plot_psf.create_optimization_plots(
        args_dict_no_plots, gd_results, mock_telescope_model, data_to_plot, tmp_path
    )
    assert result is None


def test_create_summary_psf_comparison_plot(tmp_path, sample_psf_data, sample_parameters):
    """Test final PSF comparison plot creation."""
    mock_telescope_model = MagicMock()
    mock_telescope_model.name = "LSTN-01"
    data_to_plot = {"measured": sample_psf_data}
    final_rmsd = 0.023

    with (
        patch("simtools.visualization.plot_psf._create_base_plot_figure") as mock_base,
        patch("matplotlib.pyplot.close") as mock_close,
    ):
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_base.return_value = (mock_fig, mock_ax)

        # Test with all parameters present
        output_file = plot_psf.create_summary_psf_comparison_plot(
            mock_telescope_model,
            sample_parameters,
            data_to_plot,
            tmp_path,
            final_rmsd,
            sample_psf_data,
        )

        assert output_file is not None
        mock_base.assert_called_once()
        mock_ax.set_title.assert_called_once()
        mock_ax.text.assert_called_once()
        mock_fig.savefig.assert_called_once()
        mock_close.assert_called_once()

        # Verify title contains parameter information
        title_call = mock_ax.set_title.call_args[0][0]
        assert "Final Optimized Parameters" in title_call
        assert "mirror_reflection_random_angle" in title_call
        assert "mirror_align_random_vertical" in title_call
        assert "mirror_align_random_horizontal" in title_call

        # Verify RMSD text
        text_call = mock_ax.text.call_args[0][2]
        assert "RMSD" in text_call
        assert "0.0230" in text_call
