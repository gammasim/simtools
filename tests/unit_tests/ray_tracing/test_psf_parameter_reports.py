"""Unit tests for the PSF optimisation output (logs, plots)."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from astropy import units as u

import simtools.ray_tracing.psf_parameter_optimisation as psf_opt
import simtools.ray_tracing.psf_parameter_reports as reports

_MIRROR_ALIGN_RANDOM = (0.004, 28.0, 0.0, 0.0)


def mirror_align_random_values():
    return list(_MIRROR_ALIGN_RANDOM)


def make_mirror_align_random_parameters():
    return {
        "mirror_align_random_horizontal": mirror_align_random_values(),
        "mirror_align_random_vertical": mirror_align_random_values(),
    }


TEST_OUTPUT_DIR = Path("/dummy_test_path")


@pytest.fixture
def mock_telescope_model():
    """Create a mock telescope model."""
    mock_tel = MagicMock()
    mock_tel.name = "LSTN-01"

    def _get_parameter_value(param):
        return {**make_mirror_align_random_parameters(), "focal_length": 28.0}.get(param, None)

    mock_tel.get_parameter_value.side_effect = _get_parameter_value
    return mock_tel


@pytest.fixture
def sample_data():
    """Create sample PSF data."""
    radius = np.linspace(0, 10, 21)
    cumulative = np.linspace(0, 1, 21)
    dtype = {"names": (psf_opt.RADIUS, psf_opt.CUMULATIVE_PSF), "formats": ("f8", "f8")}
    data = np.empty(21, dtype=dtype)
    data[psf_opt.RADIUS] = radius
    data[psf_opt.CUMULATIVE_PSF] = cumulative
    return data


def test_write_gradient_descent_log(mock_telescope_model, sample_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        gd_results = [
            ({"param": 1.0}, 0.1, 0.8, 3.5, sample_data),
            ({"param": 1.1}, 0.08, 0.85, 3.2, sample_data),
        ]

        log_file = reports.write_gradient_descent_log(
            gd_results,
            {"param": 1.1},
            3.2,
            output_dir,
            mock_telescope_model,
            use_ks_statistic=False,
        )
        content = log_file.read_text(encoding="utf-8")
        assert "# Telescope: LSTN-01" in content
        assert "[ITER-00] Iteration 0: RMSD=0.100000" in content
        assert "[FINAL] Iteration 1: RMSD=0.080000" in content
        assert "Best rmsd: 0.080000" in content
        assert "Total iterations: 2" in content


@pytest.mark.parametrize(
    ("use_ks", "mc_results", "expected_content", "not_expected"),
    [
        # Test with KS statistic and p-values
        (
            True,
            (
                0.15,
                0.02,
                [0.14, 0.15, 0.16],
                0.75,
                0.05,
                [0.7, 0.75, 0.8],
                3.5,
                0.1,
                [3.4, 3.5, 3.6],
            ),
            ["KS Statistic", "P-VALUE STATISTICS", "Good fits", "Fair fits", "Poor fits"],
            [],
        ),
        # Test with varied p-values (good, fair, poor mix)
        (
            True,
            (
                0.15,
                0.02,
                [0.14, 0.15, 0.16, 0.17, 0.18],
                0.05,
                0.03,
                [0.08, 0.03, 0.005, 0.001, 0.2],
                3.5,
                0.1,
                [3.4, 3.5, 3.6, 3.7, 3.8],
            ),
            ["GOOD", "FAIR", "POOR"],
            [],
        ),
        # Test without KS statistic (RMSD mode)
        (
            False,
            (
                0.15,
                0.02,
                [0.14, 0.15, 0.16],
                None,
                None,
                [None, None, None],
                3.5,
                0.1,
                [3.4, 3.5, 3.6],
            ),
            ["RMSD"],
            ["KS Statistic", "GOOD", "FAIR", "POOR"],
        ),
    ],
)
def test_write_monte_carlo_analysis(
    mock_telescope_model, tmp_path, use_ks, mc_results, expected_content, not_expected
):
    output_file = reports.write_monte_carlo_analysis(
        mc_results,
        tmp_path,
        mock_telescope_model,
        use_ks_statistic=use_ks,
        fraction=0.8,
    )

    assert output_file.exists()
    content = output_file.read_text()

    for expected in expected_content:
        assert expected in content

    for not_exp in not_expected:
        assert not_exp not in content


_OFFAXIS_PARS = {
    "mirror_align_random_horizontal": [0.1, 0.2, 0.3, 0.4],
    "mirror_align_random_vertical": [0.1, 0.2, 0.3, 0.4],
}


_OFFAXIS_ARGS = {"fraction": 0.8, "zenith_angle": 20 * u.deg, "source_distance": 10 * u.km}


def test_create_psf_vs_offaxis_plot(tmp_path):
    mock_telescope_model = MagicMock()
    mock_telescope_model.name = "LSTN-01"
    with (
        patch("simtools.ray_tracing.psf_parameter_reports.RayTracing") as mock_ray_class,
        patch("simtools.ray_tracing.psf_parameter_reports.visualize.save_figure") as mock_save,
        patch("simtools.ray_tracing.psf_parameter_reports.plt"),
    ):
        reports.create_psf_vs_offaxis_plot(
            mock_telescope_model, MagicMock(), _OFFAXIS_ARGS, _OFFAXIS_PARS, tmp_path
        )
    mock_ray = mock_ray_class.return_value
    mock_ray.analyze.assert_called_once()
    assert mock_ray.analyze.call_args.kwargs["keep_images"] is False
    mock_telescope_model.overwrite_parameters.assert_called_once_with(_OFFAXIS_PARS, flat_dict=True)
    assert mock_save.call_count == 2


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"max_offset": 4.0 * u.deg, "offset_step": 0.0 * u.deg}, "offset_step must be positive"),
        (
            {"max_offset": -1.0 * u.deg, "offset_step": 0.5 * u.deg},
            "max_offset must be non-negative",
        ),
    ],
)
def test_create_psf_vs_offaxis_plot_rejects_invalid_offsets(overrides, message):
    with pytest.raises(ValueError, match=message):
        reports.create_psf_vs_offaxis_plot(
            MagicMock(), MagicMock(), {**_OFFAXIS_ARGS, **overrides}, _OFFAXIS_PARS, None
        )
