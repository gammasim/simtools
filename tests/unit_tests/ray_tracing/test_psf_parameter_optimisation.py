#!/usr/bin/python3

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import astropy.units as u
import numpy as np
import pytest

import simtools.ray_tracing.psf_parameter_optimisation as psf_opt

TEST_OUTPUT_DIR = Path("/dummy_test_path")


@pytest.fixture
def mock_telescope_model():
    """Create a mock telescope model."""
    mock_tel = MagicMock()
    mock_tel.name = "LSTN-01"
    mock_tel.get_parameter_value.side_effect = lambda param: {
        "mirror_reflection_random_angle": [0.005, 0.15, 0.03],
        "mirror_align_random_horizontal": [0.004, 28.0, 0.0, 0.0],
        "mirror_align_random_vertical": [0.004, 28.0, 0.0, 0.0],
        "focal_length": 28.0,
    }.get(param, None)
    return mock_tel


@pytest.fixture
def mock_site_model():
    """Create a mock site model."""
    return MagicMock()


@pytest.fixture
def mock_args_dict():
    """Create mock arguments dictionary."""
    return {
        "data": "test_data.txt",
        "model_path": "/path/to/model",
        "ks_statistic": False,
        "learning_rate": 0.1,
        "test": True,
        "plot_all": False,
        "zenith": 20.0,
        "src_distance": 10.0,
        "monte_carlo_analysis": False,
        "rmsd_threshold": 0.01,
        "fraction": 0.8,
    }


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


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_parameters():
    """Create sample PSF parameters for testing."""
    return {"mirror_reflection_random_angle": [0.005, 0.15, 0.03]}


@pytest.fixture
def sample_results():
    """Create sample optimization results for testing."""
    return [
        ({"param": 1.0}, 0.05, 0.8, 5.0, {"data": "test"}),
        ({"param": 1.1}, 0.06, 0.7, 5.1, {"data": "test2"}),
    ]


@pytest.fixture
def sample_mc_results():
    """Create sample Monte Carlo results for testing."""
    return (
        0.1,
        0.01,
        [0.09, 0.1, 0.11],  # mean_metric, std_metric, metric_values
        0.8,
        0.05,
        [0.75, 0.8, 0.85],  # mean_p_value, std_p_value, p_values
        3.5,
        0.1,
        [3.4, 3.5, 3.6],  # mean_psf, std_psf, psf_values
    )


@pytest.fixture
def optimizer(mock_telescope_model, mock_site_model, mock_args_dict, sample_data):
    """Create a PSFParameterOptimizer instance for testing."""
    radius = sample_data[psf_opt.RADIUS]
    data_to_plot = {"measured": sample_data}
    return psf_opt.PSFParameterOptimizer(
        mock_telescope_model, mock_site_model, mock_args_dict, data_to_plot, radius, TEST_OUTPUT_DIR
    )


@pytest.fixture
def mock_gradient_descent_workflow():
    """Factory fixture for common gradient descent workflow patches."""

    def _create_mocks(
        initial_params=None,
        sim_result=None,
        step_result=None,
        gd_convergence=False,
    ):
        """Create mock objects for gradient descent workflow.

        Parameters
        ----------
        initial_params : dict, optional
            Initial parameters to return
        sim_result : tuple, optional
            Simulation result (metric, p_value, psf_diameter, simulated_data)
        step_result : GradientStepResult, optional
            Step result object
        gd_convergence : bool, optional
            Whether gradient descent should converge

        Returns
        -------
        dict
            Dictionary of mock patches
        """
        if initial_params is None:
            initial_params = {"mirror_reflection_random_angle": [0.005, 0.15, 0.03]}
        if sim_result is None:
            sim_result = (0.1, 0.8, 3.5, None)
        if step_result is None:
            step_result = _create_gradient_step_result(
                metric=0.09 if not gd_convergence else 0.005,
            )

        return {
            "get_params": MagicMock(return_value=initial_params),
            "sim": MagicMock(return_value=sim_result),
            "step": MagicMock(return_value=step_result),
        }

    return _create_mocks


def _create_gradient_step_result(
    params=None,
    metric=0.09,
    p_value=None,
    psf_diameter=3.4,
    simulated_data=None,
    step_accepted=True,
    learning_rate=0.1,
):
    """Helper function to create GradientStepResult objects with default values."""
    if params is None:
        params = {"mirror_reflection_random_angle": [0.004, 0.15, 0.028]}
    return GradientStepResult(
        params=params,
        metric=metric,
        p_value=p_value,
        psf_diameter=psf_diameter,
        simulated_data=simulated_data,
        step_accepted=step_accepted,
        learning_rate=learning_rate,
    )


@pytest.mark.parametrize(
    ("title", "additional_info", "value", "expected_checks"),
    [
        # Test header creation mode with basic info
        ("Test Title", {"key": "value"}, None, ["Test Title", "LSTN-01", "key: value"]),
        # Test value formatting mode with numeric value
        ("Test Title", None, 1.23, ["1.230000"]),
        # Test list formatting
        ("Test Title", None, [1.0, 2.0, 3.0], ["[1.000000, 2.000000, 3.000000]"]),
        # Test with additional info
        (
            "Test Header",
            {"Version": "1.0", "Date": "2023-01-01"},
            None,
            ["Version: 1.0", "Date: 2023-01-01"],
        ),
        # Test non-numeric value
        (None, None, "test_string", ["test_string"]),
    ],
)
def test_create_log_header_and_format_value(
    mock_telescope_model, title, additional_info, value, expected_checks
):
    """Test creation of log header with various inputs."""
    tel_model = mock_telescope_model if title is not None else None
    result = psf_opt._create_log_header_and_format_value(title, tel_model, additional_info, value)

    for expected in expected_checks:
        if expected == result:  # For exact matches like formatted values
            assert result == expected
        else:  # For substring matches like headers
            assert expected in result


def test_calculate_rmsd():
    """Test RMSD calculation between data and simulated arrays."""
    data = np.array([1, 2, 3, 4])
    sim = np.array([1.1, 2.1, 3.1, 4.1])
    rmsd = psf_opt.calculate_rmsd(data, sim)
    assert rmsd == pytest.approx(0.1, rel=1e-3)


def test_calculate_ks_statistic():
    """Test KS statistic calculation."""
    data = np.array([1, 2, 3, 4])
    sim = np.array([1.1, 2.1, 3.1, 4.1])
    ks_stat, p_value = psf_opt.calculate_ks_statistic(data, sim)
    assert isinstance(ks_stat, float)
    assert isinstance(p_value, float)
    assert 0.0 <= p_value <= 1.0


def test_get_previous_values(mock_telescope_model):
    """Test getting previous parameter values from telescope model."""
    values = psf_opt.get_previous_values(mock_telescope_model)
    assert "mirror_reflection_random_angle" in values
    assert "mirror_align_random_horizontal" in values
    assert "mirror_align_random_vertical" in values
    assert len(values["mirror_reflection_random_angle"]) == 3
    assert len(values["mirror_align_random_horizontal"]) == 4


@pytest.mark.parametrize(
    ("data_file", "should_raise_error"),
    [
        # Normal case with data file
        ("test_data.txt", False),
        # Error case with no file
        (None, True),
    ],
)
def test_load_and_process_data(mock_args_dict, data_file, should_raise_error):
    """Test loading and processing PSF data with and without file."""
    mock_args_dict["data"] = data_file

    if should_raise_error:
        with pytest.raises(FileNotFoundError, match="No data file specified"):
            psf_opt.load_and_process_data(mock_args_dict)
    else:
        with (
            patch("simtools.utils.general.find_file") as mock_find,
            patch("astropy.table.Table.read") as mock_read,
        ):
            mock_find.return_value = Path("test.txt")

            # Create proper mock table
            mock_table = MagicMock()
            mock_table.__len__.return_value = 3
            mock_table.colnames = ["radius_mm", "integral_psf"]

            # Mock astropy quantity behavior
            mock_radius_col = MagicMock()
            mock_radius_col.to.return_value.value = np.array([1, 2, 3])
            mock_psf_col = np.array([0.1, 0.2, 0.3])

            mock_table.__getitem__.side_effect = lambda key: (
                mock_radius_col if "radius" in key else mock_psf_col
            )
            mock_read.return_value = mock_table

            data_to_plot, radius = psf_opt.load_and_process_data(mock_args_dict)
            assert "measured" in data_to_plot
            assert len(radius) > 0


@pytest.mark.parametrize(
    ("pars", "should_raise_error", "expected_psf_diameter"),
    [
        # Normal case with parameters
        ({"mirror_reflection_random_angle": [0.005, 0.15, 0.03]}, False, 5.0),
        # Error case with None parameters
        (None, True, None),
    ],
)
def test__run_ray_tracing_simulation(
    mock_telescope_model,
    mock_site_model,
    mock_args_dict,
    pars,
    should_raise_error,
    expected_psf_diameter,
):
    """Test ray tracing simulation execution with normal parameters and error cases."""
    if should_raise_error:
        with pytest.raises(ValueError, match="No best parameters found"):
            psf_opt._run_ray_tracing_simulation(
                mock_telescope_model, mock_site_model, mock_args_dict, pars
            )
    else:
        with patch("simtools.ray_tracing.psf_parameter_optimisation.RayTracing") as mock_rt:
            # Create mock ray tracing instance with proper return values
            mock_instance = MagicMock()
            mock_images = [MagicMock()]
            mock_images[0].get_psf.return_value = expected_psf_diameter
            mock_images[0].get_cumulative_data.return_value = MagicMock()
            mock_instance.images.return_value = mock_images
            mock_rt.return_value = mock_instance

            psf_diameter, _ = psf_opt._run_ray_tracing_simulation(
                mock_telescope_model, mock_site_model, mock_args_dict, pars
            )
            assert psf_diameter == pytest.approx(expected_psf_diameter)
            mock_telescope_model.overwrite_parameters.assert_called_once_with(pars)


@pytest.mark.parametrize(
    ("use_ks_statistic", "plot_all", "radius_none", "should_raise_error", "expected_behavior"),
    [
        # Basic PSF simulation
        (False, False, False, False, "basic"),
        # With KS statistic enabled
        (True, False, False, False, "ks_statistic"),
        # With plotting enabled
        (False, True, False, False, "plotting"),
        # Error case: radius is None
        (False, False, True, True, "radius_error"),
    ],
)
def test_run_psf_simulation(
    mock_telescope_model,
    mock_site_model,
    mock_args_dict,
    sample_data,
    sample_parameters,
    use_ks_statistic,
    plot_all,
    radius_none,
    should_raise_error,
    expected_behavior,
):
    """Test PSF simulation with various configurations."""
    radius = None if radius_none else sample_data[psf_opt.RADIUS]
    data_to_plot = {"measured": sample_data}
    mock_args_dict["plot_all"] = plot_all

    if should_raise_error:
        with patch(
            "simtools.ray_tracing.psf_parameter_optimisation._run_ray_tracing_simulation"
        ) as mock_sim:
            mock_image = MagicMock()
            mock_sim.return_value = (3.5, mock_image)

            with pytest.raises(ValueError, match="Radius data is not available"):
                psf_opt.run_psf_simulation(
                    mock_telescope_model,
                    mock_site_model,
                    mock_args_dict,
                    sample_parameters,
                    data_to_plot,
                    radius,
                )
    else:
        patches = ["simtools.ray_tracing.psf_parameter_optimisation._run_ray_tracing_simulation"]
        if expected_behavior == "plotting":
            patches.append("simtools.visualization.plot_psf.create_psf_parameter_plot")

        with (
            patch(patches[0]) as mock_sim,
            patch(patches[1]) if len(patches) > 1 else patch("builtins.print") as mock_plot,
        ):
            mock_image = MagicMock()
            mock_image.get_cumulative_data.return_value = sample_data
            mock_sim.return_value = (3.5, mock_image)

            if expected_behavior == "plotting":
                mock_pdf = MagicMock()
                psf_opt.run_psf_simulation(
                    mock_telescope_model,
                    mock_site_model,
                    mock_args_dict,
                    sample_parameters,
                    data_to_plot,
                    radius,
                    pdf_pages=mock_pdf,
                )
                mock_plot.assert_called_once()
            else:
                result = psf_opt.run_psf_simulation(
                    mock_telescope_model,
                    mock_site_model,
                    mock_args_dict,
                    sample_parameters,
                    data_to_plot,
                    radius,
                    use_ks_statistic=use_ks_statistic,
                )
                assert len(result) == 4
                if expected_behavior == "ks_statistic":
                    assert result[2] is not None  # p_value should not be None with KS statistic


def test_write_tested_parameters_to_file(
    mock_telescope_model, temp_dir, sample_results, sample_parameters
):
    """Test writing tested parameters to log file."""
    with (
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation._add_units_to_psf_parameters"
        ) as mock_units,
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation._create_log_header_and_format_value"
        ) as mock_header,
    ):
        mock_units.return_value = sample_parameters
        mock_header.return_value = "Test Header\n"

        param_file = psf_opt.write_tested_parameters_to_file(
            sample_results, sample_parameters, 5.0, temp_dir, mock_telescope_model
        )
        assert param_file.exists()


def test__add_units_to_psf_parameters():
    """Test adding astropy units to PSF parameters with multiple scenarios."""
    # Test normal case with known parameters
    best_pars = {
        "mirror_reflection_random_angle": [0.005, 0.15, 0.03],
        "mirror_align_random_horizontal": [0.004, 28.0, 0.0, 0.0],
        "mirror_align_random_vertical": [0.004, 28.0, 0.0, 0.0],
    }
    result = psf_opt._add_units_to_psf_parameters(best_pars)
    assert "mirror_reflection_random_angle" in result

    # Test else branch - parameters without known units
    parameters_no_units = {"param1": [1.5, 2.0]}
    result = psf_opt._add_units_to_psf_parameters(parameters_no_units)
    assert result == {"param1": [1.5, 2.0]}


def test_export_psf_parameters(mock_telescope_model, temp_dir, sample_parameters):
    """Test exporting PSF parameters."""
    with (
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation._add_units_to_psf_parameters"
        ) as mock_units,
        patch("simtools.ray_tracing.psf_parameter_optimisation.writer") as mock_writer,
    ):
        mock_units.return_value = sample_parameters

        psf_opt.export_psf_parameters(
            sample_parameters, mock_telescope_model.name, "1.0.0", temp_dir
        )

        mock_units.assert_called_once_with(sample_parameters)
        assert mock_writer.ModelDataWriter.dump_model_parameter.call_count == len(sample_parameters)

        for call_args in mock_writer.ModelDataWriter.dump_model_parameter.call_args_list:
            _, kwargs = call_args
            assert kwargs["instrument"] == mock_telescope_model.name
            assert kwargs["parameter_version"] == "1.0.0"

    with (
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation._add_units_to_psf_parameters"
        ) as mock_units,
        patch("simtools.ray_tracing.psf_parameter_optimisation.writer") as mock_writer,
        patch("simtools.ray_tracing.psf_parameter_optimisation.logger") as mock_logger,
    ):
        mock_units.return_value = sample_parameters
        mock_writer.ModelDataWriter.dump_model_parameter.side_effect = ValueError("Test error")

        psf_opt.export_psf_parameters(
            sample_parameters, mock_telescope_model.name, "1.0.0", temp_dir
        )

        mock_logger.error.assert_called_once_with(
            "Error exporting simulation parameters: Test error"
        )


def test__calculate_param_gradient_success(optimizer):
    """Test successful parameter gradient calculation."""
    current_params = {"mirror_reflection_random_angle": [0.005, 0.15, 0.03]}
    current_metric = 0.1

    with patch.object(optimizer, "run_simulation") as mock_sim:
        mock_sim.return_value = (None, 0.095, None, None)

        gradient = optimizer._calculate_param_gradient(
            current_params,
            current_metric,
            "mirror_reflection_random_angle",
            [0.005, 0.15, 0.03],
            0.0005,
        )

        assert gradient is not None
        assert isinstance(gradient, list)
        assert len(gradient) == 3
        assert all(isinstance(g, float) for g in gradient)


def test__calculate_param_gradient_simulation_failure(optimizer):
    """Test parameter gradient calculation when simulation fails."""

    current_params = {"mirror_reflection_random_angle": [0.005, 0.15, 0.03]}

    with patch.object(optimizer, "run_simulation") as mock_sim:
        # First call succeeds, second fails
        mock_sim.side_effect = ValueError("Simulation failed")

        gradient = optimizer._calculate_param_gradient(
            current_params,
            0.1,
            "mirror_reflection_random_angle",
            [0.005, 0.15, 0.03],
            0.0005,
        )
        # Should return None when simulation fails
        assert gradient is None


def test_calculate_gradient(optimizer, sample_parameters):
    """Test gradient calculation for all parameters using PSFParameterOptimizer."""

    with patch.object(optimizer, "_calculate_param_gradient") as mock_grad:
        mock_grad.return_value = [-0.1, 0.05, -0.02]

        gradients = optimizer.calculate_gradient(
            sample_parameters,
            0.1,
        )
        assert "mirror_reflection_random_angle" in gradients
        assert gradients["mirror_reflection_random_angle"] == [-0.1, 0.05, -0.02]


def test_calculate_gradient_returns_none_on_failure(
    mock_telescope_model, mock_site_model, mock_args_dict, sample_data, tmp_path
):
    """Test that calculate_gradient returns None if any parameter gradient fails."""
    data_to_plot = {"measured": sample_data}
    radius = sample_data[psf_opt.RADIUS]

    optimizer = psf_opt.PSFParameterOptimizer(
        mock_telescope_model, mock_site_model, mock_args_dict, data_to_plot, radius, tmp_path
    )

    current_params = {
        "mirror_reflection_random_angle": [0.005, 0.15, 0.03],
        "mirror_align_random_horizontal": [0.004, 28.0, 0.0, 0.0],
    }

    with patch.object(optimizer, "_calculate_param_gradient") as mock_grad:
        # First parameter succeeds, second fails
        mock_grad.side_effect = [[-0.1, 0.05, -0.02], None]

        gradients = optimizer.calculate_gradient(current_params, 0.1)
        # Should return None when any gradient calculation fails
        assert gradients is None


def test_apply_gradient_step(optimizer):
    """Test applying gradient descent step with various parameter types and zenith angle preservation."""

    # Test with list parameters
    current_params = {"mirror_reflection_random_angle": [0.005, 0.15, 0.03]}
    gradients = {"mirror_reflection_random_angle": [-0.001, 0.01, -0.002]}
    new_params = optimizer.apply_gradient_step(current_params, gradients, 0.1)
    assert "mirror_reflection_random_angle" in new_params
    assert (
        new_params["mirror_reflection_random_angle"][0]
        != current_params["mirror_reflection_random_angle"][0]
    )

    # Test with single-value parameter
    current_params = {"camera_filter_relative_efficiency": 1.0}
    gradients = {"camera_filter_relative_efficiency": 0.01}
    new_params = optimizer.apply_gradient_step(current_params, gradients, 0.1)
    assert new_params["camera_filter_relative_efficiency"] == pytest.approx(0.999)

    # Test zenith angle preservation for mirror_align parameters
    current_params = {"mirror_align_random_horizontal": [0.005, 0.15, 0.03]}
    gradients = {"mirror_align_random_horizontal": [-0.001, -0.02, -0.003]}
    new_params = optimizer.apply_gradient_step(current_params, gradients, 0.1)
    assert new_params["mirror_align_random_horizontal"][1] == pytest.approx(0.15)

    # Test mirror_align_random_vertical
    current_params = {"mirror_align_random_vertical": [0.005, 0.15, 0.03]}
    gradients = {"mirror_align_random_vertical": [-0.001, -0.02, -0.003]}
    new_params = optimizer.apply_gradient_step(current_params, gradients, 0.1)
    assert new_params["mirror_align_random_vertical"][1] == pytest.approx(0.15)


def test_perform_gradient_step_with_retries(optimizer):
    """Test gradient step with retries using PSFParameterOptimizer."""

    current_params = {"mirror_reflection_random_angle": [0.005]}
    current_metric = 10.0

    with (
        patch.object(optimizer, "calculate_gradient") as mock_grad,
        patch.object(optimizer, "apply_gradient_step") as mock_step,
        patch.object(optimizer, "run_simulation") as mock_sim,
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation._are_all_parameters_within_allowed_range"
        ) as mock_validate,
    ):
        mock_grad.return_value = {"mirror_reflection_random_angle": [0.001]}
        mock_step.return_value = {"mirror_reflection_random_angle": [0.004]}
        mock_sim.return_value = (8.0, 4.5, 0.9, {"data": "test"})  # Better metric
        mock_validate.return_value = True  # Parameters are valid

        result = optimizer.perform_gradient_step_with_retries(
            current_params,
            current_metric,
            0.1,
        )

        assert result is not None
        assert isinstance(result, psf_opt.GradientStepResult)
        assert result.params == {"mirror_reflection_random_angle": [0.004]}
        assert result.step_accepted is True


def test__create_step_plot(sample_data, mock_args_dict, tmp_path):
    """Test creating step plot for optimization iteration."""
    data_to_plot = {"measured": sample_data}
    current_params = {"mirror_reflection_random_angle": [0.005, 0.15, 0.03]}
    mock_args_dict["plot_all"] = True
    mock_args_dict["fraction"] = 0.8

    # Create optimizer instance
    mock_tel = MagicMock()
    mock_site = MagicMock()
    optimizer = psf_opt.PSFParameterOptimizer(
        mock_tel, mock_site, mock_args_dict, data_to_plot, sample_data[psf_opt.RADIUS], tmp_path
    )

    with (
        patch("matplotlib.backends.backend_pdf.PdfPages") as mock_pdf,
        patch("simtools.visualization.plot_psf.create_psf_parameter_plot") as mock_plot,
    ):
        mock_pages = MagicMock()
        mock_pdf.return_value = mock_pages

        optimizer._create_step_plot(mock_pages, current_params, 3.5, 0.1, 0.8, sample_data)

        mock_plot.assert_called_once_with(
            data_to_plot,
            current_params,
            3.5,  # new_psf_diameter
            0.1,  # new_metric
            False,  # is_best
            mock_pages,  # pdf_pages
            fraction=0.8,
            p_value=0.8,
            use_ks_statistic=False,
        )

    # Test early return when pdf_pages is None
    result = optimizer._create_step_plot(None, current_params, 3.5, 0.1, 0.8, sample_data)
    assert result is None

    # Test early return when plot_all is False
    mock_args_dict["plot_all"] = False
    optimizer_no_plot = psf_opt.PSFParameterOptimizer(
        mock_tel, mock_site, mock_args_dict, data_to_plot, sample_data[psf_opt.RADIUS], tmp_path
    )
    result = optimizer_no_plot._create_step_plot(
        mock_pages, current_params, 3.5, 0.1, 0.8, sample_data
    )
    assert result is None

    # Test early return when new_simulated_data is None
    mock_args_dict["plot_all"] = True
    optimizer2 = psf_opt.PSFParameterOptimizer(
        mock_tel, mock_site, mock_args_dict, data_to_plot, sample_data[psf_opt.RADIUS], tmp_path
    )
    result = optimizer2._create_step_plot(mock_pages, current_params, 3.5, 0.1, 0.8, None)
    assert result is None


def test__create_final_plot(optimizer, sample_data):
    """Test creating final optimization result plot."""
    best_params = {"mirror_reflection_random_angle": [0.005, 0.15, 0.03]}
    data_to_plot = optimizer.data_to_plot

    with (
        patch("matplotlib.backends.backend_pdf.PdfPages") as mock_pdf,
        patch("simtools.visualization.plot_psf.create_psf_parameter_plot") as mock_plot,
        patch.object(optimizer, "run_simulation") as mock_sim,
        patch("simtools.ray_tracing.psf_parameter_optimisation.calculate_rmsd") as mock_rmsd,
    ):
        mock_pages = MagicMock()
        mock_pdf.return_value = mock_pages
        mock_sim.return_value = (3.5, 0.08, 0.9, sample_data)
        mock_rmsd.return_value = 0.05

        optimizer._create_final_plot(
            mock_pages,
            best_params,
            3.5,
        )

        mock_sim.assert_called_once_with(
            best_params,
            pdf_pages=None,
            is_best=False,
            use_cache=False,
            use_ks_statistic=True,
        )

        mock_rmsd.assert_called_once()
        call_args = mock_rmsd.call_args[0]
        assert len(call_args) == 2
        np.testing.assert_array_equal(
            call_args[0], data_to_plot["measured"][psf_opt.CUMULATIVE_PSF]
        )
        np.testing.assert_array_equal(call_args[1], sample_data[psf_opt.CUMULATIVE_PSF])

        mock_plot.assert_called_once_with(
            data_to_plot,
            best_params,
            3.5,  # best_psf_diameter
            0.05,  # best_rmsd from mock
            True,  # is_best
            mock_pages,  # pdf_pages
            fraction=0.8,
            p_value=0.9,  # p_value from mock_sim
            use_ks_statistic=False,
            second_metric=0.08,  # ks_stat from mock_sim
        )

        mock_pages.close.assert_called_once()

    # Test early return when pdf_pages is None
    result = optimizer._create_final_plot(
        None,
        best_params,
        3.5,
    )
    assert result is None

    # Test early return when best_params is None
    result = optimizer._create_final_plot(
        mock_pages,
        None,
        3.5,
    )
    assert result is None


def test_run_gradient_descent_optimization(optimizer, sample_data, mock_gradient_descent_workflow):
    """Test complete gradient descent optimization workflow using PSFParameterOptimizer."""
    mocks = mock_gradient_descent_workflow()

    with (
        patch.object(optimizer, "get_initial_parameters", mocks["get_params"]),
        patch.object(optimizer, "run_simulation", mocks["sim"]),
        patch("simtools.visualization.plot_psf.setup_pdf_plotting") as mock_setup,
        patch.object(optimizer, "perform_gradient_step_with_retries", mocks["step"]),
    ):
        mock_setup.return_value = None

        best_pars, best_psf_diameter, gd_results = optimizer.run_gradient_descent(
            rmsd_threshold=0.1,
            learning_rate=0.1,
        )
        assert "mirror_reflection_random_angle" in best_pars
        assert isinstance(best_psf_diameter, float)
        assert len(gd_results) > 0


def test_run_gradient_descent_with_no_data(optimizer):
    """Test that run_gradient_descent returns early when no data is available."""
    optimizer.data_to_plot = None
    optimizer.radius = None

    best_pars, best_psf_diameter, gd_results = optimizer.run_gradient_descent(
        rmsd_threshold=0.1,
        learning_rate=0.1,
    )

    assert best_pars is None
    assert best_psf_diameter is None
    assert gd_results == []


def test__write_log_interpretation():
    """Test writing log interpretation section."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        psf_opt._write_log_interpretation(f, use_ks_statistic=True)
        f.flush()

        with open(f.name) as rf:
            content = rf.read()
            assert "P-VALUE INTERPRETATION" in content

    # Test with RMSD
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        psf_opt._write_log_interpretation(f, use_ks_statistic=False)
        f.flush()

        with open(f.name) as rf:
            content = rf.read()
            assert "RMSD INTERPRETATION" in content


def test__write_iteration_entry():
    """Test writing single iteration entry to log."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        pars = {"mirror_reflection_random_angle": [0.005, 0.15, 0.03]}

        psf_opt._write_iteration_entry(
            f,
            1,
            pars,
            0.1,
            0.8,
            3.5,
            use_ks_statistic=True,
            metric_name="KS-stat",
            total_iterations=10,
        )
        f.flush()

        with open(f.name) as rf:
            content = rf.read()
            assert "Iteration 1:" in content


def test_write_gradient_descent_log(mock_telescope_model, sample_data):
    """Test writing complete gradient descent log file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        gd_results = [
            ({"param": 1.0}, 0.1, 0.8, 3.5, sample_data),
            ({"param": 1.1}, 0.08, 0.85, 3.2, sample_data),
        ]

        with patch(
            "simtools.ray_tracing.psf_parameter_optimisation._create_log_header_and_format_value"
        ) as mock_header:
            mock_header.return_value = "Test Header\n"

            log_file = psf_opt.write_gradient_descent_log(
                gd_results,
                {"param": 1.1},
                3.2,
                output_dir,
                mock_telescope_model,
                use_ks_statistic=False,
            )
            assert log_file.exists()


def test_analyze_monte_carlo_error(
    optimizer, mock_telescope_model, mock_site_model, mock_args_dict, sample_data
):
    """Test Monte Carlo error analysis using PSFParameterOptimizer."""
    radius = sample_data[psf_opt.RADIUS]
    data_to_plot = {"measured": sample_data}

    # Test 1: Normal case
    with (
        patch.object(optimizer, "get_initial_parameters") as mock_prev,
        patch.object(optimizer, "run_simulation") as mock_sim,
    ):
        mock_prev.return_value = {"mirror_reflection_random_angle": [0.005, 0.15, 0.03]}
        mock_sim.return_value = (3.5, 0.1, 0.8, sample_data)

        result = optimizer.analyze_monte_carlo_error(n_simulations=2)
        assert len(result) == 9  # All MC statistics

    # Test 2: No data case
    optimizer_no_data = psf_opt.PSFParameterOptimizer(
        mock_telescope_model, mock_site_model, mock_args_dict, None, None, TEST_OUTPUT_DIR
    )
    result = optimizer_no_data.analyze_monte_carlo_error()
    assert result[0] is None
    assert result[1] is None
    assert result[2] == []

    # Test 3: All simulations fail
    with (
        patch.object(optimizer, "get_initial_parameters") as mock_prev,
        patch.object(optimizer, "run_simulation") as mock_sim,
    ):
        mock_prev.return_value = {"mirror_reflection_random_angle": [0.005, 0.15, 0.03]}
        mock_sim.side_effect = RuntimeError("All simulations failed")

        result = optimizer.analyze_monte_carlo_error(n_simulations=2)

        assert result[0] is None  # mean_metric should be None
        assert result[1] is None  # std_metric should be None
        assert result[2] == []  # metric_values should be empty

    # Test 4: With KS statistic
    mock_args_dict["ks_statistic"] = True
    optimizer_ks = psf_opt.PSFParameterOptimizer(
        mock_telescope_model, mock_site_model, mock_args_dict, data_to_plot, radius, TEST_OUTPUT_DIR
    )
    with (
        patch.object(optimizer_ks, "get_initial_parameters") as mock_prev,
        patch.object(optimizer_ks, "run_simulation") as mock_sim,
    ):
        mock_prev.return_value = {"mirror_reflection_random_angle": [0.005, 0.15, 0.03]}
        mock_sim.return_value = (3.5, 0.1, 0.8, sample_data)  # psf_diameter, metric, p_value, data

        result = optimizer_ks.analyze_monte_carlo_error(n_simulations=2)

        assert len(result) == 9
        assert result[3] is not None


def test_run_simulation_with_caching_and_ks_override(
    optimizer, mock_telescope_model, mock_site_model, mock_args_dict, sample_data
):
    """Test run_simulation caching logic and KS statistic override."""
    params = {"mirror_reflection_random_angle": [0.005, 0.15, 0.03]}
    radius = sample_data[psf_opt.RADIUS]
    data_to_plot = {"measured": sample_data}

    with patch("simtools.ray_tracing.psf_parameter_optimisation.run_psf_simulation") as mock_sim:
        mock_sim.return_value = (3.5, 0.1, 0.8, sample_data)

        # Test caching
        optimizer.run_simulation(params, use_cache=True)
        assert optimizer.cache_misses == 1
        optimizer.run_simulation(params, use_cache=True)
        assert optimizer.cache_hits == 1

        # Test cache bypass with pdf_pages and is_best
        optimizer.run_simulation(params, pdf_pages=MagicMock(), use_cache=True)
        optimizer.run_simulation(params, is_best=True, use_cache=True)
        optimizer.run_simulation(params, use_cache=False)

    # Test KS statistic override
    mock_args_dict["ks_statistic"] = False
    optimizer2 = psf_opt.PSFParameterOptimizer(
        mock_telescope_model, mock_site_model, mock_args_dict, data_to_plot, radius, TEST_OUTPUT_DIR
    )

    with patch("simtools.ray_tracing.psf_parameter_optimisation.run_psf_simulation") as mock_sim:
        mock_sim.return_value = (3.5, 0.1, 0.8, sample_data)
        optimizer2.run_simulation(params, use_ks_statistic=True)
        assert mock_sim.call_args[0][8] is True


def test_perform_gradient_step_comprehensive(optimizer, sample_data):
    """Test perform_gradient_step_with_retries: tuple structure, retries, bounds checking, and LR reset."""
    current_params = {"mirror_reflection_random_angle": [0.005, 0.15, 0.03]}

    # Test 1: Returns dataclass with False when gradient is None
    with patch.object(optimizer, "calculate_gradient") as mock_calc_grad:
        mock_calc_grad.return_value = None
        result = optimizer.perform_gradient_step_with_retries(current_params, 0.1, 0.1)
        assert result.step_accepted is False
        assert result.learning_rate == pytest.approx(0.1)

    # Test 2: Learning rate reduction with retries
    with (
        patch.object(optimizer, "calculate_gradient") as mock_calc_grad,
        patch.object(optimizer, "apply_gradient_step") as mock_apply,
        patch.object(optimizer, "run_simulation") as mock_sim,
    ):
        mock_calc_grad.return_value = {"mirror_reflection_random_angle": [-0.001, 0.01, -0.002]}
        mock_apply.return_value = {"mirror_reflection_random_angle": [0.006, 0.15, 0.031]}
        mock_sim.side_effect = [
            (3.6, 0.15, 0.75, sample_data),  # Worse than current (0.1)
            (3.4, 0.08, 0.85, sample_data),  # Better
        ]

        result = optimizer.perform_gradient_step_with_retries(
            current_params, 0.1, 0.1, max_retries=3
        )
        assert result.step_accepted is True
        assert result.learning_rate == pytest.approx(0.07, rel=1e-2)

    # Test 3: Parameters out of bounds - all retries fail
    with (
        patch.object(optimizer, "calculate_gradient") as mock_calc_grad,
        patch.object(optimizer, "apply_gradient_step") as mock_apply,
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation._are_all_parameters_within_allowed_range"
        ) as mock_validate,
    ):
        mock_calc_grad.return_value = {"mirror_reflection_random_angle": [-0.001, 0.01, -0.002]}
        mock_apply.return_value = {
            "mirror_reflection_random_angle": [999, 999, 999]
        }  # Out of bounds
        mock_validate.return_value = False

        result = optimizer.perform_gradient_step_with_retries(
            current_params, 0.1, 0.1, max_retries=3
        )
        assert result.step_accepted is False

    # Test 4: Learning rate reset when < 1e-6
    with (
        patch.object(optimizer, "calculate_gradient") as mock_calc_grad,
        patch.object(optimizer, "apply_gradient_step") as mock_apply,
        patch.object(optimizer, "run_simulation"),
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation._are_all_parameters_within_allowed_range"
        ) as mock_valid,
    ):
        mock_calc_grad.return_value = {"mirror_reflection_random_angle": [-0.001, 0.01, -0.002]}
        mock_apply.return_value = {"mirror_reflection_random_angle": [0.006, 0.15, 0.031]}
        mock_valid.side_effect = [False, False, True]

        result = optimizer.perform_gradient_step_with_retries(
            current_params, 0.1, 0.0000015, max_retries=3
        )
        assert result.learning_rate == pytest.approx(0.0001)


def test_gradient_descent_convergence_and_tracking(
    optimizer, sample_data, mock_gradient_descent_workflow
):
    """Test gradient descent convergence, max iterations, and best metric tracking."""

    # Test 1: Convergence when threshold is reached
    mocks = mock_gradient_descent_workflow(gd_convergence=True)
    with (
        patch.object(optimizer, "get_initial_parameters", mocks["get_params"]),
        patch.object(optimizer, "run_simulation", mocks["sim"]),
        patch.object(optimizer, "perform_gradient_step_with_retries") as mock_step,
    ):
        mock_step.side_effect = [
            _create_gradient_step_result(metric=0.008, p_value=0.85, simulated_data=sample_data),
        ]

        _, _, results = optimizer.run_gradient_descent(
            rmsd_threshold=0.01, learning_rate=0.1, max_iterations=10
        )
        assert len(results) == 2

    # Test 2: Max iterations reached
    mocks = mock_gradient_descent_workflow()
    with (
        patch.object(optimizer, "get_initial_parameters", mocks["get_params"]),
        patch.object(optimizer, "run_simulation", mocks["sim"]),
        patch.object(optimizer, "perform_gradient_step_with_retries") as mock_step,
    ):
        mock_step.return_value = _create_gradient_step_result(
            metric=0.095, p_value=0.85, simulated_data=sample_data
        )

        _, _, results = optimizer.run_gradient_descent(
            rmsd_threshold=0.01, learning_rate=0.1, max_iterations=2
        )
        assert len(results) == 3

    # Test 3: Best metric tracking across iterations
    mocks = mock_gradient_descent_workflow()
    with (
        patch.object(optimizer, "get_initial_parameters", mocks["get_params"]),
        patch.object(optimizer, "run_simulation", mocks["sim"]),
        patch.object(optimizer, "perform_gradient_step_with_retries") as mock_step,
    ):
        mock_step.side_effect = [
            _create_gradient_step_result(
                params={"mirror_reflection_random_angle": [0.004, 0.15, 0.028]},
                psf_diameter=3.4,
                metric=0.08,
                p_value=0.85,
                simulated_data=sample_data,
            ),
            _create_gradient_step_result(
                params={"mirror_reflection_random_angle": [0.003, 0.15, 0.025]},
                psf_diameter=3.3,
                metric=0.05,
                p_value=0.85,
                simulated_data=sample_data,
            ),
            _create_gradient_step_result(
                params={"mirror_reflection_random_angle": [0.002, 0.15, 0.020]},
                psf_diameter=3.2,
                metric=0.12,
                p_value=0.85,
                simulated_data=sample_data,
            ),
        ]

        _, best_diameter, _ = optimizer.run_gradient_descent(
            rmsd_threshold=0.01, learning_rate=0.1, max_iterations=3
        )
        assert best_diameter == pytest.approx(3.3)


def test_perform_gradient_step_with_retries_learning_rate_reduction(
    mock_telescope_model, mock_site_model, mock_args_dict, sample_data
):
    """Test learning rate reduction logic in gradient step retries."""
    radius = sample_data[psf_opt.RADIUS]
    data_to_plot = {"measured": sample_data}
    current_params = {"mirror_reflection_random_angle": [0.005, 0.15, 0.03]}

    optimizer = psf_opt.PSFParameterOptimizer(
        mock_telescope_model, mock_site_model, mock_args_dict, data_to_plot, radius, TEST_OUTPUT_DIR
    )

    with (
        patch.object(optimizer, "calculate_gradient") as mock_calc_grad,
        patch.object(optimizer, "apply_gradient_step") as mock_apply,
        patch.object(optimizer, "run_simulation") as mock_sim,
    ):
        # Simulate worse results to trigger learning rate reduction
        mock_calc_grad.return_value = {"mirror_reflection_random_angle": [-0.001, 0.01, -0.002]}
        mock_apply.return_value = {"mirror_reflection_random_angle": [0.006, 0.15, 0.031]}
        mock_sim.return_value = (3.5, 0.15, 0.8, sample_data)  # Worse metric

        result = optimizer.perform_gradient_step_with_retries(
            current_params, 0.1, 0.1, max_retries=3
        )

        # Should fail and return GradientStepResult with False for step_accepted
        assert isinstance(result, psf_opt.GradientStepResult)
        assert result.step_accepted is False


def test_parameter_validation():
    """Test parameter range validation and boundary checking."""
    # Test with valid value in known range
    assert (
        psf_opt._is_parameter_within_allowed_range("camera_filter_relative_efficiency", 0, 0.5)
        is True
    )
    # Test boundary values
    assert (
        psf_opt._is_parameter_within_allowed_range("camera_filter_relative_efficiency", 0, 0.0)
        is True
    )
    assert (
        psf_opt._is_parameter_within_allowed_range("camera_filter_relative_efficiency", 0, 1.0)
        is True
    )
    # Test unknown parameter (no schema - returns True by default)
    assert psf_opt._is_parameter_within_allowed_range("unknown_parameter", 0, 0.5) is True

    # Test all parameters validation
    params = {"mirror_reflection_random_angle": [0.005, 0.15, 0.03]}
    assert psf_opt._are_all_parameters_within_allowed_range(params) is True

    # Test with clearly out-of-range value
    invalid_params = {"camera_filter_relative_efficiency": 100.0}
    result = psf_opt._are_all_parameters_within_allowed_range(invalid_params)
    assert isinstance(result, bool)


def test_params_to_cache_key(optimizer):
    """Test _params_to_cache_key with list and non-list parameter values."""

    # Test with mixed list and non-list values
    params = {
        "mirror_reflection_random_angle": [0.005, 0.15, 0.03],
        "camera_filter_relative_efficiency": 1.0,
    }
    cache_key = optimizer._params_to_cache_key(params)
    assert isinstance(cache_key, frozenset)
    assert cache_key == optimizer._params_to_cache_key(params)

    # Test with all non-list values
    params2 = {"param1": 1.0, "param2": 2.0}
    cache_key2 = optimizer._params_to_cache_key(params2)
    assert isinstance(cache_key2, frozenset)
    assert cache_key != cache_key2  # Different params = different keys


def test_workflow_edge_cases(
    mock_telescope_model, mock_site_model, mock_args_dict, sample_data, tmp_path
):
    """Test PSF optimization workflow edge cases: no data, failed optimization, Monte Carlo."""
    # Test 1: No data
    with (
        patch("simtools.ray_tracing.psf_parameter_optimisation.load_and_process_data") as mock_load,
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation.PSFParameterOptimizer"
        ) as mock_opt_cls,
    ):
        mock_load.return_value = (None, None)
        mock_optimizer = MagicMock()
        mock_opt_cls.return_value = mock_optimizer
        mock_optimizer.run_gradient_descent.return_value = (None, None, [])

        psf_opt.run_psf_optimization_workflow(
            mock_telescope_model, mock_site_model, mock_args_dict, tmp_path
        )

    # Test 2: Optimization fails
    with (
        patch("simtools.ray_tracing.psf_parameter_optimisation.load_and_process_data") as mock_load,
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation.PSFParameterOptimizer"
        ) as mock_opt_cls,
    ):
        mock_load.return_value = ({"measured": sample_data}, sample_data[psf_opt.RADIUS])
        mock_optimizer = MagicMock()
        mock_opt_cls.return_value = mock_optimizer
        mock_optimizer.run_gradient_descent.return_value = (None, None, [])

        psf_opt.run_psf_optimization_workflow(
            mock_telescope_model, mock_site_model, mock_args_dict, tmp_path
        )

    # Test 3: Monte Carlo analysis with results
    mock_args_dict["monte_carlo_analysis"] = True
    with (
        patch("simtools.ray_tracing.psf_parameter_optimisation.load_and_process_data") as mock_load,
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation.PSFParameterOptimizer"
        ) as mock_opt_cls,
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation.write_monte_carlo_analysis"
        ) as mock_write,
        patch("simtools.visualization.plot_psf.create_monte_carlo_uncertainty_plot"),
    ):
        mock_load.return_value = ({"measured": sample_data}, sample_data[psf_opt.RADIUS])
        mock_optimizer = MagicMock()
        mock_opt_cls.return_value = mock_optimizer
        mock_optimizer.analyze_monte_carlo_error.return_value = (
            0.1,
            0.01,
            [0.09, 0.11],
            None,
            None,
            [],
            3.5,
            0.05,
            [3.4, 3.6],
        )

        psf_opt.run_psf_optimization_workflow(
            mock_telescope_model, mock_site_model, mock_args_dict, tmp_path
        )
        mock_write.assert_called_once()

    # Test 4: Monte Carlo with no results
    with (
        patch("simtools.ray_tracing.psf_parameter_optimisation.load_and_process_data") as mock_load,
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation.PSFParameterOptimizer"
        ) as mock_opt_cls,
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation.write_monte_carlo_analysis"
        ) as mock_write,
    ):
        mock_load.return_value = ({"measured": sample_data}, sample_data[psf_opt.RADIUS])
        mock_optimizer = MagicMock()
        mock_opt_cls.return_value = mock_optimizer
        mock_optimizer.analyze_monte_carlo_error.return_value = (
            None,
            None,
            [],
            None,
            None,
            [],
            None,
            None,
            [],
        )

        psf_opt.run_psf_optimization_workflow(
            mock_telescope_model, mock_site_model, mock_args_dict, tmp_path
        )
        mock_write.assert_not_called()


def test_perturbed_params_creation():
    """Test _create_perturbed_params with list and non-list parameters."""
    current_params = {
        "mirror_reflection_random_angle": [0.005, 0.15, 0.03],
        "camera_filter_relative_efficiency": 1.0,
    }

    # Test list parameter perturbation
    perturbed = psf_opt._create_perturbed_params(
        current_params, "mirror_reflection_random_angle", [0.005, 0.15, 0.03], 0, 0.005, 0.001
    )
    assert perturbed["mirror_reflection_random_angle"][0] == pytest.approx(0.006)
    assert perturbed["mirror_reflection_random_angle"][1] == pytest.approx(0.15)

    # Test non-list parameter perturbation
    perturbed2 = psf_opt._create_perturbed_params(
        current_params, "camera_filter_relative_efficiency", 1.0, 0, 1.0, 0.01
    )
    assert perturbed2["camera_filter_relative_efficiency"] == pytest.approx(1.01)


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
    """Test write_monte_carlo_analysis with various configurations."""
    output_file = psf_opt.write_monte_carlo_analysis(
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


def test_perform_gradient_step_with_metric_rejection_lr_reset(
    mock_telescope_model, mock_site_model, mock_args_dict, sample_data
):
    """Test learning rate reset when metric gets worse and lr drops below 1e-6."""
    radius = sample_data[psf_opt.RADIUS]
    data_to_plot = {"measured": sample_data}
    current_params = {"mirror_reflection_random_angle": [0.005, 0.15, 0.03]}

    optimizer = psf_opt.PSFParameterOptimizer(
        mock_telescope_model, mock_site_model, mock_args_dict, data_to_plot, radius, TEST_OUTPUT_DIR
    )

    with (
        patch.object(optimizer, "calculate_gradient") as mock_calc_grad,
        patch.object(optimizer, "apply_gradient_step") as mock_apply,
        patch.object(optimizer, "run_simulation") as mock_sim,
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation._are_all_parameters_within_allowed_range"
        ) as mock_valid,
    ):
        mock_calc_grad.return_value = {"mirror_reflection_random_angle": [-0.001, 0.01, -0.002]}
        mock_apply.return_value = {"mirror_reflection_random_angle": [0.006, 0.15, 0.031]}
        mock_valid.return_value = True  # Parameters within bounds

        # First 2 calls: worse metric (triggers rejection and lr reduction)
        # third call: improved metric (accepts)
        mock_sim.side_effect = [
            (3.6, 0.15, 0.8, sample_data),  # Worse: 0.15 > 0.1
            (3.6, 0.15, 0.8, sample_data),  # Worse: 0.15 > 0.1,
            (3.4, 0.08, 0.8, sample_data),  # Better: 0.08 < 0.1, accept
        ]

        result = optimizer.perform_gradient_step_with_retries(
            current_params, 0.1, 0.0000015, max_retries=3
        )

        # Step should be accepted with reset learning rate
        # After 2 rejections: 0.0000015 * 0.7 * 0.7 = 0.000000735 < 1e-6, resets to 0.0001
        assert result.step_accepted is True
        assert result.learning_rate == pytest.approx(0.0001)


def test_get_initial_parameters(optimizer, mock_telescope_model):
    """Test get_initial_parameters method calls get_previous_values with optimize_only."""

    with patch("simtools.ray_tracing.psf_parameter_optimisation.get_previous_values") as mock_get:
        mock_get.return_value = {"mirror_reflection_random_angle": [0.005, 0.15, 0.03]}

        result = optimizer.get_initial_parameters()

        mock_get.assert_called_once_with(mock_telescope_model, optimizer.optimize_only)
        assert result == {"mirror_reflection_random_angle": [0.005, 0.15, 0.03]}


def test_calculate_param_gradient_with_exception(optimizer, sample_data):
    """Test _calculate_param_gradient exception handling with actual logging."""
    current_params = {"mirror_reflection_random_angle": [0.005, 0.15, 0.03]}

    with patch.object(optimizer, "run_simulation") as mock_sim:
        # Trigger exception during simulation - let logger actually log
        mock_sim.side_effect = ValueError("Simulation failed")

        # This should return None when exception occurs
        result = optimizer._calculate_param_gradient(
            current_params, 0.1, "mirror_reflection_random_angle", [0.005, 0.15, 0.03], 0.001
        )

        assert result is None


def test_calculate_param_gradient_with_runtime_error(optimizer):
    """Test _calculate_param_gradient with RuntimeError."""
    current_params = {"mirror_reflection_random_angle": [0.005, 0.15, 0.03]}

    with patch.object(optimizer, "run_simulation") as mock_sim:
        mock_sim.side_effect = RuntimeError("Runtime error in simulation")

        result = optimizer._calculate_param_gradient(
            current_params, 0.1, "mirror_reflection_random_angle", [0.005, 0.15, 0.03], 0.001
        )

        assert result is None


def test_is_parameter_within_allowed_range_schema_errors():
    """Test _is_parameter_within_allowed_range schema access errors."""
    # Test that function handles KeyError and returns True
    with patch("simtools.utils.names.model_parameters") as mock_params:
        mock_params.side_effect = KeyError("Parameter not found")

        result = psf_opt._is_parameter_within_allowed_range("nonexistent_param", 0, 0.5)

        assert result is True


def test_run_gradient_descent_no_step_accepted(
    optimizer, sample_data, mock_gradient_descent_workflow
):
    """Test gradient descent when no step is accepted, increases learning rate."""
    mocks = mock_gradient_descent_workflow()

    with (
        patch.object(optimizer, "get_initial_parameters", mocks["get_params"]),
        patch.object(optimizer, "run_simulation", mocks["sim"]),
        patch.object(optimizer, "perform_gradient_step_with_retries") as mock_step,
    ):
        mock_step.side_effect = [
            _create_gradient_step_result(
                params=None,
                psf_diameter=None,
                metric=None,
                p_value=None,
                simulated_data=None,
                step_accepted=False,
            ),
            _create_gradient_step_result(
                metric=0.008, p_value=0.85, simulated_data=sample_data, learning_rate=0.2
            ),
        ]

        best_params, _, _ = optimizer.run_gradient_descent(
            rmsd_threshold=0.01, learning_rate=0.1, max_iterations=5
        )

        assert best_params is not None


def test_run_gradient_descent_learning_rate_cap(
    optimizer, sample_data, mock_gradient_descent_workflow
):
    """Test that learning rate is capped at maximum threshold when increased."""
    mocks = mock_gradient_descent_workflow()

    with (
        patch.object(optimizer, "get_initial_parameters", mocks["get_params"]),
        patch.object(optimizer, "run_simulation", mocks["sim"]),
        patch.object(optimizer, "perform_gradient_step_with_retries") as mock_step,
    ):
        mock_step.side_effect = [
            _create_gradient_step_result(
                metric=0.04, p_value=0.85, simulated_data=sample_data, learning_rate=0.003
            ),
            _create_gradient_step_result(
                params={"mirror_reflection_random_angle": [0.0038, 0.15, 0.027]},
                psf_diameter=3.3,
                metric=0.03,
                p_value=0.87,
                simulated_data=sample_data,
                learning_rate=0.006,
            ),
            _create_gradient_step_result(
                params={"mirror_reflection_random_angle": [0.0035, 0.15, 0.026]},
                psf_diameter=3.2,
                metric=0.008,
                p_value=0.88,
                simulated_data=sample_data,
                learning_rate=0.01,
            ),
        ]

        best_params, _, _ = optimizer.run_gradient_descent(
            rmsd_threshold=0.01, learning_rate=0.003, max_iterations=5
        )

        assert best_params is not None
        assert mock_step.call_count >= 2


def test_parameter_validation_edge_cases():
    """Test _is_parameter_within_allowed_range edge cases"""
    # Test when data is not a list
    with patch("simtools.utils.names.model_parameters") as mock_params:
        mock_params.return_value = {"test_param": {"data": "not_a_list"}}
        result = psf_opt._is_parameter_within_allowed_range("test_param", 0, 0.5)
        assert result is True

    # Test when no allowed_range specified
    with patch("simtools.utils.names.model_parameters") as mock_params:
        mock_params.return_value = {"test_param": {"data": [{}]}}
        result = psf_opt._is_parameter_within_allowed_range("test_param", 0, 0.5)
        assert result is True

    # Test value below minimum
    with patch("simtools.utils.names.model_parameters") as mock_params:
        mock_params.return_value = {
            "test_param": {"data": [{"allowed_range": {"min": 0.0, "max": 1.0}}]}
        }
        result = psf_opt._is_parameter_within_allowed_range("test_param", 0, -0.1)
        assert result is False

    # Test value above maximum
    with patch("simtools.utils.names.model_parameters") as mock_params:
        mock_params.return_value = {
            "test_param": {"data": [{"allowed_range": {"min": 0.0, "max": 1.0}}]}
        }
        result = psf_opt._is_parameter_within_allowed_range("test_param", 0, 1.5)
        assert result is False


def test_are_all_parameters_list_and_single_value(sample_data):
    """Test _are_all_parameters_within_allowed_range with list and single values"""
    # Test with list parameter out of range
    with patch(
        "simtools.ray_tracing.psf_parameter_optimisation._is_parameter_within_allowed_range"
    ) as mock_check:
        mock_check.return_value = False

        params = {"mirror_reflection_random_angle": [0.005, 999.0, 0.03]}
        result = psf_opt._are_all_parameters_within_allowed_range(params)

        assert result is False

    # Test with single value parameter out of range
    with patch(
        "simtools.ray_tracing.psf_parameter_optimisation._is_parameter_within_allowed_range"
    ) as mock_check:
        mock_check.return_value = False

        params = {"camera_filter_relative_efficiency": 999.0}
        result = psf_opt._are_all_parameters_within_allowed_range(params)

        assert result is False


def test_workflow_with_all_features(
    mock_telescope_model, mock_site_model, mock_args_dict, sample_data, tmp_path
):
    """Test workflow with plotting, parameter export, and all features."""
    mock_args_dict.update(
        {
            "rmsd_threshold": 0.01,
            "learning_rate": 0.1,
            "write_psf_parameters": True,
            "telescope": "LSTN-01",
            "parameter_version": "1.0.0",
        }
    )

    with (
        patch("simtools.ray_tracing.psf_parameter_optimisation.load_and_process_data") as mock_load,
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation.PSFParameterOptimizer"
        ) as mock_opt_cls,
        patch("simtools.visualization.plot_psf.create_optimization_plots") as mock_opt_plot,
        patch(
            "simtools.visualization.plot_psf.create_gradient_descent_convergence_plot"
        ) as mock_conv_plot,
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation.write_gradient_descent_log"
        ) as mock_write_log,
        patch("simtools.visualization.plot_psf.create_psf_vs_offaxis_plot") as mock_offaxis_plot,
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation.export_psf_parameters"
        ) as mock_export,
    ):
        mock_load.return_value = ({"measured": sample_data}, sample_data[psf_opt.RADIUS])

        mock_optimizer = MagicMock()
        mock_opt_cls.return_value = mock_optimizer

        best_params = {"mirror_reflection_random_angle": [0.005, 0.15, 0.03]}
        gd_results = [(best_params, 0.05, 0.8, 3.5, sample_data)]
        mock_optimizer.run_gradient_descent.return_value = (best_params, 3.5, gd_results)
        mock_optimizer.use_ks_statistic = False
        mock_optimizer.fraction = 0.8

        mock_write_log.return_value = tmp_path / "log.txt"

        psf_opt.run_psf_optimization_workflow(
            mock_telescope_model, mock_site_model, mock_args_dict, tmp_path
        )

        mock_opt_plot.assert_called_once()
        mock_conv_plot.assert_called_once()
        mock_write_log.assert_called_once()
        mock_offaxis_plot.assert_called_once()
        mock_export.assert_called_once()


def test_cleanup_intermediate_files(
    tmp_path, mock_telescope_model, mock_site_model, mock_args_dict, sample_data
):
    """Test cleanup of intermediate log and list files via workflow integration."""
    mock_args_dict["cleanup"] = True
    (tmp_path / "test.log").write_text("log content")
    (tmp_path / "sim.lis").write_text("lis content")
    (tmp_path / "sim.lis.gz").write_bytes(b"lis.gz content")
    (tmp_path / "keep_me.png").write_bytes(b"png content")
    (tmp_path / "keep_me.pdf").write_bytes(b"pdf content")

    with (
        patch("simtools.ray_tracing.psf_parameter_optimisation.load_and_process_data") as mock_load,
        patch("simtools.ray_tracing.psf_parameter_optimisation.PSFParameterOptimizer") as mock_opt,
        patch("simtools.visualization.plot_psf.create_optimization_plots"),
        patch("simtools.visualization.plot_psf.create_gradient_descent_convergence_plot"),
        patch("simtools.ray_tracing.psf_parameter_optimisation.write_gradient_descent_log"),
        patch("simtools.visualization.plot_psf.create_psf_vs_offaxis_plot"),
    ):
        mock_load.return_value = ({"measured": sample_data}, sample_data[psf_opt.RADIUS])
        mock_opt.return_value.run_gradient_descent.return_value = (
            {"mirror_reflection_random_angle": [0.005, 0.15, 0.03]},
            3.5,
            [
                (
                    {"mirror_reflection_random_angle": [0.005, 0.15, 0.03]},
                    0.05,
                    0.8,
                    3.5,
                    sample_data,
                )
            ],
        )
        psf_opt.run_psf_optimization_workflow(
            mock_telescope_model, mock_site_model, mock_args_dict, tmp_path
        )

    assert not (tmp_path / "test.log").exists()
    assert not (tmp_path / "sim.lis").exists()
    assert not (tmp_path / "sim.lis.gz").exists()
    assert (tmp_path / "keep_me.png").exists()
    assert (tmp_path / "keep_me.pdf").exists()


def test_optimizer_with_custom_optimize_only(mock_telescope_model, mock_site_model, mock_args_dict):
    """Test PSFParameterOptimizer initialization with custom optimize_only parameter."""
    custom_params = ["mirror_reflection_random_angle"]
    sample_data = np.linspace(0, 5, 100)

    optimizer = psf_opt.PSFParameterOptimizer(
        tel_model=mock_telescope_model,
        site_model=mock_site_model,
        args_dict=mock_args_dict,
        data_to_plot={"measured": sample_data},
        radius=sample_data,
        output_dir=TEST_OUTPUT_DIR,
        optimize_only=custom_params,
    )

    assert optimizer.optimize_only == custom_params


def _setup_ray_tracing_mock():
    """Helper function to set up common RayTracing mock objects."""
    mock_instance = MagicMock()
    mock_im = MagicMock()
    mock_instance.images.return_value = [mock_im]
    mock_im.get_psf.return_value = {"psf_d80": 0.1}
    return mock_instance


def test_run_ray_tracing_simulation_single_mirror_mode(mock_telescope_model, mock_site_model):
    """Test _run_ray_tracing_simulation in single mirror mode."""
    args_dict = {
        "single_mirror_mode": True,
        "mirror_numbers": "0-5",
        "use_random_focal_length": True,
        "random_focal_length_seed": 42,
        "simtel_path": "/path/to/simtel",
        "test": True,
        "fraction": 0.95,
    }
    params = {"mirror_reflection_random_angle": [0.006, 0.14, 0.025]}

    with patch("simtools.ray_tracing.psf_parameter_optimisation.RayTracing") as mock_rt:
        mock_rt.return_value = _setup_ray_tracing_mock()

        result, _ = psf_opt._run_ray_tracing_simulation(
            mock_telescope_model, mock_site_model, args_dict, params
        )

        assert result == {"psf_d80": 0.1}
        mock_telescope_model.overwrite_parameters.assert_called_once_with(params)

        mock_rt.assert_called_once()
        call_kwargs = mock_rt.call_args[1]
        assert call_kwargs["single_mirror_mode"] is True
        assert call_kwargs["mirror_numbers"] == "0-5"
        assert call_kwargs["use_random_focal_length"] is True
        assert call_kwargs["random_focal_length_seed"] == 42


def test_run_ray_tracing_simulation_full_telescope_mode(mock_telescope_model, mock_site_model):
    """Test _run_ray_tracing_simulation in full telescope mode (single_mirror_mode=False)."""
    args_dict = {
        "single_mirror_mode": False,
        "zenith": 25.0,
        "src_distance": 12.0,
        "simtel_path": "/path/to/simtel",
        "test": True,
        "fraction": 0.85,
    }
    params = {"mirror_reflection_random_angle": [0.004, 0.16, 0.035]}

    with patch("simtools.ray_tracing.psf_parameter_optimisation.RayTracing") as mock_rt:
        mock_rt.return_value = _setup_ray_tracing_mock()

        result, _ = psf_opt._run_ray_tracing_simulation(
            mock_telescope_model, mock_site_model, args_dict, params
        )

        assert result == {"psf_d80": 0.1}
        mock_telescope_model.overwrite_parameters.assert_called_once_with(params)

        mock_rt.assert_called_once()
        call_kwargs = mock_rt.call_args[1]
        assert "zenith_angle" in call_kwargs
        assert "source_distance" in call_kwargs
        assert "off_axis_angle" in call_kwargs
        assert call_kwargs["zenith_angle"].value == pytest.approx(25.0)
        assert call_kwargs["zenith_angle"].unit == u.deg
        assert call_kwargs["source_distance"].value == pytest.approx(12.0)
        assert call_kwargs["source_distance"].unit == u.km
        assert "single_mirror_mode" not in call_kwargs or not call_kwargs.get("single_mirror_mode")
