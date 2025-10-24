#!/usr/bin/python3

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

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
        "simtel_path": "/path/to/simtel",
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
@patch("simtools.ray_tracing.psf_parameter_optimisation.RayTracing")
def test__run_ray_tracing_simulation(
    mock_rt,
    mock_telescope_model,
    mock_site_model,
    mock_args_dict,
    pars,
    should_raise_error,
    expected_psf_diameter,
):
    """Test ray tracing simulation execution with normal parameters and error cases."""
    if should_raise_error:
        mock_args_dict["simtel_path"] = "/path/to/simtel"
        with pytest.raises(ValueError, match="No best parameters found"):
            psf_opt._run_ray_tracing_simulation(
                mock_telescope_model, mock_site_model, mock_args_dict, pars
            )
    else:
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


def test__calculate_param_gradient(
    mock_telescope_model, mock_site_model, mock_args_dict, sample_data, sample_parameters
):
    """Test parameter gradient calculation."""
    radius = sample_data[psf_opt.RADIUS]
    data_to_plot = {"measured": sample_data}

    with patch("simtools.ray_tracing.psf_parameter_optimisation.run_psf_simulation") as mock_sim:
        # Need 3 calls for the 3 parameter values
        mock_sim.side_effect = [
            (3.5, 0.1, 0.8, sample_data),
            (3.7, 0.12, 0.7, sample_data),
            (3.6, 0.11, 0.75, sample_data),
        ]

        gradient = psf_opt._calculate_param_gradient(
            mock_telescope_model,
            mock_site_model,
            mock_args_dict,
            sample_parameters,
            data_to_plot,
            radius,
            0.1,
            "mirror_reflection_random_angle",
            [0.005, 0.15, 0.03],
            0.0005,
            False,
        )
        assert isinstance(gradient, list)
        assert len(gradient) == 3


def test_calculate_gradient(
    mock_telescope_model, mock_site_model, mock_args_dict, sample_data, sample_parameters
):
    """Test gradient calculation for all parameters."""
    radius = sample_data[psf_opt.RADIUS]
    data_to_plot = {"measured": sample_data}

    with patch(
        "simtools.ray_tracing.psf_parameter_optimisation._calculate_param_gradient"
    ) as mock_grad:
        mock_grad.return_value = [-0.1, 0.05, -0.02]

        gradients = psf_opt.calculate_gradient(
            mock_telescope_model,
            mock_site_model,
            mock_args_dict,
            sample_parameters,
            data_to_plot,
            radius,
            0.1,
        )
        assert "mirror_reflection_random_angle" in gradients


def test_apply_gradient_step():
    """Test applying gradient descent step to parameters with multiple scenarios."""
    # Basic test case
    current_params = {"mirror_reflection_random_angle": [0.005, 0.15, 0.03]}
    gradients = {"mirror_reflection_random_angle": [-0.001, 0.01, -0.002]}
    learning_rate = 0.1

    new_params = psf_opt.apply_gradient_step(current_params, gradients, learning_rate)
    assert "mirror_reflection_random_angle" in new_params
    # Check that parameters were updated
    assert (
        new_params["mirror_reflection_random_angle"][0]
        != current_params["mirror_reflection_random_angle"][0]
    )

    # Edge case with single parameter (consolidating test_apply_gradient_step_edge_case)
    parameters = {"param1": 1.0}
    gradient = {"param1": 0.1}
    learning_rate = 0.5

    result = psf_opt.apply_gradient_step(parameters, gradient, learning_rate)
    assert result == {"param1": 0.95}  # gradient descent: 1.0 - 0.5 * 0.1 = 0.95

    current_params = {"mirror_reflection_random_angle": [0.005]}
    with (
        patch("simtools.ray_tracing.psf_parameter_optimisation.run_psf_simulation") as mock_sim,
        patch("simtools.ray_tracing.psf_parameter_optimisation.apply_gradient_step") as mock_step,
        patch("simtools.ray_tracing.psf_parameter_optimisation.calculate_gradient") as mock_grad,
    ):
        # Set up mocks to return improved results
        mock_grad.return_value = {"mirror_reflection_random_angle": [0.001]}
        mock_step.return_value = {"mirror_reflection_random_angle": [0.004]}
        mock_sim.return_value = (8.0, 4.5, 0.9, {"data": "test"})  # Better metric and PSF diameter

        result = psf_opt._perform_gradient_step_with_retries(
            None,
            None,
            {},
            current_params,
            10.0,
            {"measured": {"data": "test"}},
            np.array([1, 2, 3]),
            3.0,
        )

        # Should return the improved parameters
        assert result is not None


def test__create_step_plot(sample_data):
    """Test creating step plot for optimization iteration."""
    data_to_plot = {"measured": sample_data}
    current_params = {"mirror_reflection_random_angle": [0.005, 0.15, 0.03]}
    args_dict = {"plot_all": True, "fraction": 0.8}

    with (
        patch("matplotlib.backends.backend_pdf.PdfPages") as mock_pdf,
        patch("simtools.visualization.plot_psf.create_psf_parameter_plot") as mock_plot,
    ):
        mock_pages = MagicMock()
        mock_pdf.return_value = mock_pages

        psf_opt._create_step_plot(
            mock_pages, args_dict, data_to_plot, current_params, 3.5, 0.1, 0.8, sample_data
        )

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
    result = psf_opt._create_step_plot(
        None, args_dict, data_to_plot, current_params, 3.5, 0.1, 0.8, sample_data
    )
    assert result is None

    # Test early return when plot_all is False
    args_dict_no_plot = {"plot_all": False}
    result = psf_opt._create_step_plot(
        mock_pages, args_dict_no_plot, data_to_plot, current_params, 3.5, 0.1, 0.8, sample_data
    )
    assert result is None

    # Test early return when new_simulated_data is None
    result = psf_opt._create_step_plot(
        mock_pages, args_dict, data_to_plot, current_params, 3.5, 0.1, 0.8, None
    )
    assert result is None


def test__create_final_plot(mock_telescope_model, mock_site_model, mock_args_dict, sample_data):
    """Test creating final optimization result plot."""
    radius = sample_data[psf_opt.RADIUS]
    data_to_plot = {"measured": sample_data}
    best_params = {"mirror_reflection_random_angle": [0.005, 0.15, 0.03]}

    with (
        patch("matplotlib.backends.backend_pdf.PdfPages") as mock_pdf,
        patch("simtools.visualization.plot_psf.create_psf_parameter_plot") as mock_plot,
        patch("simtools.ray_tracing.psf_parameter_optimisation.run_psf_simulation") as mock_sim,
        patch("simtools.ray_tracing.psf_parameter_optimisation.calculate_rmsd") as mock_rmsd,
    ):
        mock_pages = MagicMock()
        mock_pdf.return_value = mock_pages
        mock_sim.return_value = (3.5, 0.08, 0.9, sample_data)
        mock_rmsd.return_value = 0.05

        psf_opt._create_final_plot(
            mock_pages,
            mock_telescope_model,
            mock_site_model,
            mock_args_dict,
            best_params,
            data_to_plot,
            radius,
            3.5,
        )

        mock_sim.assert_called_once_with(
            mock_telescope_model,
            mock_site_model,
            mock_args_dict,
            best_params,
            data_to_plot,
            radius,
            pdf_pages=None,
            is_best=False,
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
    result = psf_opt._create_final_plot(
        None,
        mock_telescope_model,
        mock_site_model,
        mock_args_dict,
        best_params,
        data_to_plot,
        radius,
        3.5,
    )
    assert result is None

    # Test early return when best_params is None
    result = psf_opt._create_final_plot(
        mock_pages,
        mock_telescope_model,
        mock_site_model,
        mock_args_dict,
        None,
        data_to_plot,
        radius,
        3.5,
    )
    assert result is None


def test_run_gradient_descent_optimization(
    mock_telescope_model, mock_site_model, mock_args_dict, sample_data
):
    """Test complete gradient descent optimization workflow."""
    radius = sample_data[psf_opt.RADIUS]
    data_to_plot = {"measured": sample_data}

    with (
        patch("simtools.ray_tracing.psf_parameter_optimisation.get_previous_values") as mock_prev,
        patch("simtools.ray_tracing.psf_parameter_optimisation.run_psf_simulation") as mock_sim,
        patch("simtools.visualization.plot_psf.setup_pdf_plotting") as mock_setup,
        patch("simtools.ray_tracing.psf_parameter_optimisation.calculate_gradient") as mock_grad,
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation._perform_gradient_step_with_retries"
        ) as mock_step,
    ):
        mock_prev.return_value = {"mirror_reflection_random_angle": [0.005, 0.15, 0.03]}
        mock_sim.return_value = (3.5, 0.1, 0.8, sample_data)
        mock_setup.return_value = None
        mock_grad.return_value = {"mirror_reflection_random_angle": [-0.001, 0.01, -0.002]}
        # Simulate convergence
        mock_step.side_effect = [
            ({"mirror_reflection_random_angle": [0.004, 0.16, 0.028]}, 0.05, 0.9, 3.2, sample_data)
        ]

        best_pars, best_psf_diameter, gd_results = psf_opt.run_gradient_descent_optimization(
            mock_telescope_model,
            mock_site_model,
            mock_args_dict,
            data_to_plot,
            radius,
            rmsd_threshold=0.1,
            learning_rate=0.1,
            output_dir=TEST_OUTPUT_DIR,
        )
        assert "mirror_reflection_random_angle" in best_pars
        assert isinstance(best_psf_diameter, float)
        assert len(gd_results) > 0


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
    mock_telescope_model, mock_site_model, mock_args_dict, sample_data
):
    """Test Monte Carlo error analysis with multiple scenarios."""
    radius = sample_data[psf_opt.RADIUS]
    data_to_plot = {"measured": sample_data}

    # Test 1: Normal case
    with (
        patch("simtools.ray_tracing.psf_parameter_optimisation.get_previous_values") as mock_prev,
        patch("simtools.ray_tracing.psf_parameter_optimisation.run_psf_simulation") as mock_sim,
    ):
        mock_prev.return_value = {"mirror_reflection_random_angle": [0.005, 0.15, 0.03]}
        mock_sim.return_value = (3.5, 0.1, 0.8, sample_data)

        result = psf_opt.analyze_monte_carlo_error(
            mock_telescope_model,
            mock_site_model,
            mock_args_dict,
            data_to_plot,
            radius,
            n_simulations=2,
        )
        assert len(result) == 9  # All MC statistics

    # Test 2: No data case (consolidating test_analyze_monte_carlo_error_no_data)
    result = psf_opt.analyze_monte_carlo_error(
        mock_telescope_model, mock_site_model, mock_args_dict, None, None
    )
    assert result == (None, None, [])

    # Test 3: All simulations fail (consolidating test_analyze_monte_carlo_error_all_simulations_fail)
    with (
        patch("simtools.ray_tracing.psf_parameter_optimisation.get_previous_values") as mock_prev,
        patch("simtools.ray_tracing.psf_parameter_optimisation.run_psf_simulation") as mock_sim,
    ):
        mock_prev.return_value = {"mirror_reflection_random_angle": [0.005, 0.15, 0.03]}
        mock_sim.side_effect = RuntimeError("All simulations failed")

        result = psf_opt.analyze_monte_carlo_error(
            mock_telescope_model,
            mock_site_model,
            mock_args_dict,
            data_to_plot,
            radius,
            n_simulations=2,
        )

        assert result[0] is None  # mean_metric should be None
        assert result[1] is None  # std_metric should be None
        assert result[2] == []  # metric_values should be empty

    # Test 4: With KS statistic (consolidating test_analyze_monte_carlo_error_with_ks_statistic)
    mock_args_dict["ks_statistic"] = True
    with (
        patch("simtools.ray_tracing.psf_parameter_optimisation.get_previous_values") as mock_prev,
        patch("simtools.ray_tracing.psf_parameter_optimisation.run_psf_simulation") as mock_sim,
    ):
        mock_prev.return_value = {"mirror_reflection_random_angle": [0.005, 0.15, 0.03]}
        mock_sim.return_value = (3.5, 0.1, 0.8, sample_data)  # psf_diameter, metric, p_value, data

        result = psf_opt.analyze_monte_carlo_error(
            mock_telescope_model,
            mock_site_model,
            mock_args_dict,
            data_to_plot,
            radius,
            n_simulations=2,
        )

        assert len(result) == 9  # All MC statistics including KS
        assert result[3] is not None  # mean_p_value should be set


@pytest.mark.parametrize(
    ("monte_carlo_enabled", "expected_result"),
    [
        (True, True),  # Monte Carlo analysis enabled
        (False, False),  # Monte Carlo analysis disabled
    ],
)
def test__handle_monte_carlo_analysis(
    mock_telescope_model,
    mock_site_model,
    mock_args_dict,
    sample_data,
    monte_carlo_enabled,
    expected_result,
    temp_dir,
):
    """Test Monte Carlo analysis handling when enabled and disabled."""
    radius = sample_data[psf_opt.RADIUS]
    data_to_plot = {"measured": sample_data}

    mock_args_dict["monte_carlo_analysis"] = monte_carlo_enabled

    if monte_carlo_enabled:
        with (
            patch(
                "simtools.ray_tracing.psf_parameter_optimisation.analyze_monte_carlo_error"
            ) as mock_mc,
            patch(
                "simtools.ray_tracing.psf_parameter_optimisation.write_monte_carlo_analysis"
            ) as mock_write,
        ):
            mock_mc.return_value = (
                0.1,
                0.01,
                [0.09, 0.1, 0.11],
                0.8,
                0.05,
                [0.75, 0.8, 0.85],
                3.5,
                0.1,
                [3.4, 3.5, 3.6],
            )
            mock_write.return_value = temp_dir / "mc_log.log"

            result = psf_opt._handle_monte_carlo_analysis(
                mock_telescope_model,
                mock_site_model,
                mock_args_dict,
                data_to_plot,
                radius,
                temp_dir,
                False,
            )
    else:
        result = psf_opt._handle_monte_carlo_analysis(
            mock_telescope_model,
            mock_site_model,
            mock_args_dict,
            data_to_plot,
            radius,
            temp_dir,
            False,
        )

    assert result is expected_result


def test__calculate_param_gradient_with_exception(
    mock_telescope_model, mock_site_model, mock_args_dict, sample_data
):
    """Test parameter gradient calculation with simulation exception."""
    radius = sample_data[psf_opt.RADIUS]
    data_to_plot = {"measured": sample_data}
    current_params = {"mirror_reflection_random_angle": 0.005}  # Single value, not list

    with patch("simtools.ray_tracing.psf_parameter_optimisation.run_psf_simulation") as mock_sim:
        mock_sim.side_effect = RuntimeError("Simulation failed")

        gradient = psf_opt._calculate_param_gradient(
            mock_telescope_model,
            mock_site_model,
            mock_args_dict,
            current_params,
            data_to_plot,
            radius,
            0.1,
            "mirror_reflection_random_angle",
            0.005,
            0.0005,
            False,
        )

        assert gradient == pytest.approx(0.0)  # Should return 0.0 on exception


def test_run_gradient_descent_optimization_no_data():
    """Test gradient descent with no data."""
    mock_tel_model = MagicMock()
    mock_site_model = MagicMock()
    args_dict = {}

    result = psf_opt.run_gradient_descent_optimization(
        mock_tel_model, mock_site_model, args_dict, None, None, 0.01, 0.1, TEST_OUTPUT_DIR
    )

    assert result == (None, None, [])


def test_analyze_monte_carlo_error_no_data():
    """Test Monte Carlo error analysis with no data."""
    mock_tel_model = MagicMock()
    mock_site_model = MagicMock()
    args_dict = {}

    result = psf_opt.analyze_monte_carlo_error(
        mock_tel_model, mock_site_model, args_dict, None, None
    )

    assert result == (None, None, [])


def test_analyze_monte_carlo_error_all_simulations_fail(
    mock_telescope_model, mock_site_model, mock_args_dict, sample_data
):
    """Test Monte Carlo error analysis when all simulations fail."""
    radius = sample_data[psf_opt.RADIUS]
    data_to_plot = {"measured": sample_data}

    with (
        patch("simtools.ray_tracing.psf_parameter_optimisation.get_previous_values") as mock_prev,
        patch("simtools.ray_tracing.psf_parameter_optimisation.run_psf_simulation") as mock_sim,
    ):
        mock_prev.return_value = {"mirror_reflection_random_angle": [0.005, 0.15, 0.03]}
        mock_sim.side_effect = RuntimeError("All simulations failed")

        result = psf_opt.analyze_monte_carlo_error(
            mock_telescope_model,
            mock_site_model,
            mock_args_dict,
            data_to_plot,
            radius,
            n_simulations=2,
        )

        assert result[0] is None  # mean_metric should be None
        assert result[1] is None  # std_metric should be None
        assert result[2] == []  # metric_values should be empty


def test_analyze_monte_carlo_error_with_ks_statistic(
    mock_telescope_model, mock_site_model, mock_args_dict, sample_data
):
    """Test Monte Carlo error analysis with KS statistic."""
    radius = sample_data[psf_opt.RADIUS]
    data_to_plot = {"measured": sample_data}
    mock_args_dict["ks_statistic"] = True

    with (
        patch("simtools.ray_tracing.psf_parameter_optimisation.get_previous_values") as mock_prev,
        patch("simtools.ray_tracing.psf_parameter_optimisation.run_psf_simulation") as mock_sim,
    ):
        mock_prev.return_value = {"mirror_reflection_random_angle": [0.005, 0.15, 0.03]}
        mock_sim.return_value = (3.5, 0.1, 0.8, sample_data)  # psf_diameter, metric, p_value, data

        result = psf_opt.analyze_monte_carlo_error(
            mock_telescope_model,
            mock_site_model,
            mock_args_dict,
            data_to_plot,
            radius,
            n_simulations=2,
        )

        assert len(result) == 9
        assert result[3] is not None  # mean_p_value should be set
        assert result[4] is not None  # std_p_value should be set


def test_run_gradient_descent_optimization_no_step_accepted(
    mock_telescope_model, mock_site_model, mock_args_dict, sample_data
):
    """Test gradient descent when no steps are accepted."""
    radius = sample_data[psf_opt.RADIUS]
    data_to_plot = {"measured": sample_data}

    with (
        patch("simtools.ray_tracing.psf_parameter_optimisation.get_previous_values") as mock_prev,
        patch("simtools.visualization.plot_psf.setup_pdf_plotting") as mock_pdf,
        patch("simtools.ray_tracing.psf_parameter_optimisation.run_psf_simulation") as mock_sim,
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation._perform_gradient_step_with_retries"
        ) as mock_step,
    ):
        mock_prev.return_value = {"mirror_reflection_random_angle": [0.005, 0.15, 0.03]}
        mock_pdf.return_value = None
        mock_sim.return_value = (3.5, 0.1, 0.8, sample_data)  # Initial simulation
        # Return no step accepted
        mock_step.return_value = (None, None, None, None, None, False, 0.2)

        best_pars, _, results = psf_opt.run_gradient_descent_optimization(
            mock_telescope_model,
            mock_site_model,
            mock_args_dict,
            data_to_plot,
            radius,
            rmsd_threshold=0.01,
            learning_rate=0.1,
            output_dir=TEST_OUTPUT_DIR,
        )

        assert best_pars is not None
        assert len(results) == 1  # Only initial evaluation


@pytest.mark.parametrize(
    ("use_ks_statistic", "p_values", "expected_content", "not_expected_content"),
    [
        (True, [0.75, 0.8, 0.85], ["KS Statistic=", "p_value=", "GOOD"], []),
        (False, [None, None, None], ["RMSD="], ["p_value="]),
        (True, [0.03, 0.02, 0.04], ["KS Statistic=", "p_value=", "FAIR"], []),
        (True, [0.001, 0.005, 0.003], ["KS Statistic=", "p_value=", "POOR"], []),
    ],
)
def test_write_monte_carlo_analysis(
    mock_telescope_model, use_ks_statistic, p_values, expected_content, not_expected_content
):
    """Test writing Monte Carlo analysis with and without KS statistic."""
    mc_results = (
        0.1,
        0.01,
        [0.09, 0.1, 0.11],  # mean_metric, std_metric, metric_values
        0.8 if use_ks_statistic else None,
        0.05 if use_ks_statistic else None,
        p_values,  # p-value stats
        3.5,
        0.1,
        [3.4, 3.5, 3.6],  # mean_psf, std_psf, psf_values
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)

        result_file = psf_opt.write_monte_carlo_analysis(
            mc_results, output_dir, mock_telescope_model, use_ks_statistic=use_ks_statistic
        )

        assert result_file.exists()
        content = result_file.read_text()

        # Check expected content is present
        for expected in expected_content:
            if expected in ["GOOD", "FAIR", "POOR"] and use_ks_statistic:
                # For significance labels, check at least one is present
                assert any(sig in content for sig in ["GOOD", "FAIR", "POOR"])
            else:
                assert expected in content

        # Check unexpected content is not present
        for not_expected in not_expected_content:
            assert not_expected not in content


def test__perform_gradient_step_with_retries_learning_rate_reset(
    mock_telescope_model, mock_site_model, mock_args_dict, sample_data
):
    """Test gradient step when learning rate becomes very small and resets."""
    radius = sample_data[psf_opt.RADIUS]
    data_to_plot = {"measured": sample_data}
    current_params = {"mirror_reflection_random_angle": [0.005, 0.15, 0.03]}
    current_metric = 0.1

    with (
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation.calculate_gradient"
        ) as mock_calc_grad,
        patch("simtools.ray_tracing.psf_parameter_optimisation.apply_gradient_step") as mock_apply,
        patch("simtools.ray_tracing.psf_parameter_optimisation.run_psf_simulation") as mock_sim,
    ):
        mock_calc_grad.return_value = {"mirror_reflection_random_angle": [-0.001, 0.01, -0.002]}
        mock_apply.return_value = {"mirror_reflection_random_angle": [0.004, 0.16, 0.028]}
        # Always return worse metric to trigger learning rate reduction
        mock_sim.return_value = (3.8, 0.15, 0.7, sample_data)

        result = psf_opt._perform_gradient_step_with_retries(
            mock_telescope_model,
            mock_site_model,
            mock_args_dict,
            current_params,
            current_metric,
            data_to_plot,
            radius,
            1e-6,  # Very small learning rate
        )

        assert len(result) == 7
        assert result[5] is False  # step_accepted should be False
        # Check that learning rate was reduced and potentially reset (just check it's different)
        assert result[6] != pytest.approx(1e-6)  # learning rate should be reset


def test__perform_gradient_step_with_retries_all_exceptions(
    mock_telescope_model, mock_site_model, mock_args_dict, sample_data
):
    """Test gradient step when all attempts raise exceptions."""
    radius = sample_data[psf_opt.RADIUS]
    data_to_plot = {"measured": sample_data}
    current_params = {"mirror_reflection_random_angle": [0.005, 0.15, 0.03]}
    current_metric = 0.1

    with patch(
        "simtools.ray_tracing.psf_parameter_optimisation.calculate_gradient"
    ) as mock_calc_grad:
        # All attempts raise KeyError
        mock_calc_grad.side_effect = KeyError("Parameter not found")

        result = psf_opt._perform_gradient_step_with_retries(
            mock_telescope_model,
            mock_site_model,
            mock_args_dict,
            current_params,
            current_metric,
            data_to_plot,
            radius,
            0.1,
            max_retries=2,
        )

        assert result == (None, None, None, None, None, False, 0.1)


def test_run_gradient_descent_optimization_step_accepted_path(
    mock_telescope_model, mock_site_model, mock_args_dict, sample_data
):
    """Test gradient descent optimization when steps are accepted."""
    radius = sample_data[psf_opt.RADIUS]
    data_to_plot = {"measured": sample_data}

    with (
        patch("simtools.ray_tracing.psf_parameter_optimisation.get_previous_values") as mock_prev,
        patch("simtools.visualization.plot_psf.setup_pdf_plotting") as mock_pdf,
        patch("simtools.ray_tracing.psf_parameter_optimisation.run_psf_simulation") as mock_sim,
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation._perform_gradient_step_with_retries"
        ) as mock_step,
    ):
        mock_prev.return_value = {"mirror_reflection_random_angle": [0.005, 0.15, 0.03]}
        mock_pdf.return_value = None
        mock_sim.return_value = (3.5, 0.05, 0.8, sample_data)  # Initial simulation above threshold

        # Step accepted with improvement - this will meet the threshold and stop
        new_params = {"mirror_reflection_random_angle": [0.004, 0.16, 0.028]}
        mock_step.return_value = (
            new_params,
            3.2,
            0.005,
            0.9,
            sample_data,
            True,
            0.1,
        )  # Lower than threshold

        best_pars, _, results = psf_opt.run_gradient_descent_optimization(
            mock_telescope_model,
            mock_site_model,
            mock_args_dict,
            data_to_plot,
            radius,
            rmsd_threshold=0.01,
            learning_rate=0.1,
            output_dir=TEST_OUTPUT_DIR,
        )

        assert best_pars == new_params
        assert len(results) == 2  # Initial + 1 accepted step


def test_run_psf_optimization_workflow_optimization_failed_no_radius(
    mock_telescope_model, mock_site_model, mock_args_dict, sample_data
):
    """Test PSF optimization workflow when optimization fails due to no radius data."""
    with (
        patch("simtools.ray_tracing.psf_parameter_optimisation.load_and_process_data") as mock_load,
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation._handle_monte_carlo_analysis"
        ) as mock_mc,
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation.run_gradient_descent_optimization"
        ) as mock_gd,
    ):
        mock_load.return_value = ({"measured": sample_data}, None)  # No radius data
        mock_mc.return_value = False
        mock_gd.return_value = (None, None, [])  # Failed optimization

        with patch("logging.Logger.error") as mock_logger:
            psf_opt.run_psf_optimization_workflow(
                mock_telescope_model, mock_site_model, mock_args_dict, TEST_OUTPUT_DIR
            )

            # Should log error about no radius data
            mock_logger.assert_any_call(
                "Possible cause: No PSF measurement data provided. Use --data argument to provide PSF data."
            )


def test_run_psf_optimization_workflow_complete_success_path(
    mock_telescope_model, mock_site_model, mock_args_dict, sample_data
):
    """Test complete successful PSF optimization workflow including final steps."""
    mock_args_dict.update(
        {
            "overwrite_model_files": True,
            "write_psf_parameters": True,
            "save_plots": True,
            "telescope": "LSTN-01",
            "parameter_version": "1.0.0",
        }
    )

    with (
        patch("simtools.ray_tracing.psf_parameter_optimisation.load_and_process_data") as mock_load,
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation._handle_monte_carlo_analysis"
        ) as mock_mc,
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation.run_gradient_descent_optimization"
        ) as mock_gd,
        patch("simtools.visualization.plot_psf.create_optimization_plots") as mock_plots,
        patch(
            "simtools.visualization.plot_psf.create_gradient_descent_convergence_plot"
        ) as mock_conv_plot,
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation.write_gradient_descent_log"
        ) as mock_log,
        patch("simtools.visualization.plot_psf.create_psf_vs_offaxis_plot") as mock_psf_plot,
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation.export_psf_parameters"
        ) as mock_export,
    ):
        radius = np.linspace(0, 10, 21)
        mock_load.return_value = ({"measured": sample_data}, radius)
        mock_mc.return_value = False
        best_pars = {"param": 1.5}
        gd_results = [(best_pars, 0.03, 0.9, 3.2, sample_data)]
        mock_gd.return_value = (best_pars, 3.2, gd_results)
        mock_log.return_value = TEST_OUTPUT_DIR / "log.log"

        psf_opt.run_psf_optimization_workflow(
            mock_telescope_model, mock_site_model, mock_args_dict, TEST_OUTPUT_DIR
        )

        # Verify all key workflow steps were called
        mock_load.assert_called_once_with(mock_args_dict)
        mock_mc.assert_called_once()
        mock_gd.assert_called_once()
        mock_plots.assert_called_once()
        mock_conv_plot.assert_called_once()
        mock_log.assert_called_once()
        mock_psf_plot.assert_called_once()
        mock_export.assert_called_once_with(best_pars, "LSTN-01", "1.0.0", TEST_OUTPUT_DIR)


def test_edge_cases(mock_telescope_model, mock_site_model, mock_args_dict, sample_data):
    """Test miscellaneous edge cases"""
    # 1. Test error handling in export_psf_parameters when writer fails
    with patch("simtools.ray_tracing.psf_parameter_optimisation.writer") as mock_writer:
        mock_writer.ModelDataWriter.dump_model_parameter.side_effect = ValueError("Export failed")
        psf_opt.export_psf_parameters({}, "LSTN-01", "1.0.0", TEST_OUTPUT_DIR)

    # 2. Test learning rate reset in _perform_gradient_step_with_retries with very small learning rate
    with (
        patch("simtools.ray_tracing.psf_parameter_optimisation.calculate_gradient") as mock_grad,
        patch("simtools.ray_tracing.psf_parameter_optimisation.apply_gradient_step") as mock_apply,
        patch("simtools.ray_tracing.psf_parameter_optimisation.run_psf_simulation") as mock_sim,
    ):
        mock_grad.return_value = {"param": [0.1]}
        mock_apply.return_value = {"param": [0.2]}
        mock_sim.return_value = (3.8, 0.15, 0.7, sample_data)  # Worse metric

        result = psf_opt._perform_gradient_step_with_retries(
            mock_telescope_model,
            mock_site_model,
            mock_args_dict,
            {"param": [0.1]},
            0.1,
            {"measured": sample_data},
            sample_data[psf_opt.RADIUS],
            1e-6,
        )
        # Learning rate will be reduced by 0.7 multiple times until it gets reset to 0.001
        assert result[5] is False  # step_accepted
        assert result[6] > 1e-6  # Learning rate was modified

    # 4. Test Monte Carlo analysis file content verification
    mc_results = (0.1, 0.01, [0.09, 0.1], 0.8, 0.05, [0.75, 0.8], 3.5, 0.1, [3.4, 3.5])
    with tempfile.TemporaryDirectory() as temp_dir:
        result_file = psf_opt.write_monte_carlo_analysis(
            mc_results, Path(temp_dir), mock_telescope_model, use_ks_statistic=True
        )
        content = result_file.read_text()
        assert "INDIVIDUAL SIMULATION RESULTS:" in content

    # 6. Test early return in run_psf_optimization_workflow when Monte Carlo analysis returns True
    with (
        patch("simtools.ray_tracing.psf_parameter_optimisation.load_and_process_data") as mock_load,
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation._handle_monte_carlo_analysis"
        ) as mock_mc,
    ):
        mock_load.return_value = ({}, [])
        mock_mc.return_value = True  # Early return
        result = psf_opt.run_psf_optimization_workflow(
            mock_telescope_model, mock_site_model, {}, TEST_OUTPUT_DIR
        )
        assert result is None
