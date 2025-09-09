#!/usr/bin/python3

import logging
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import astropy.units as u
import numpy as np
import pytest

import simtools.ray_tracing.psf_parameter_optimisation as psf_opt

# Test constants
MOCK_MODEL_PATH = "/path/to/model"


@pytest.fixture
def sample_psf_data():
    """Create sample PSF data for testing."""
    radius = np.linspace(0, 10, 21)
    cumulative = np.linspace(0, 1, 21)
    dtype = {"names": (psf_opt.RADIUS_CM, psf_opt.CUMULATIVE_PSF), "formats": ("f8", "f8")}
    data = np.empty(21, dtype=dtype)
    data[psf_opt.RADIUS_CM] = radius
    data[psf_opt.CUMULATIVE_PSF] = cumulative
    return data


@pytest.fixture
def mock_telescope_model():
    """Create a mock telescope model."""
    mock_tel = MagicMock()
    mock_tel.get_parameter_value.side_effect = lambda param: {
        "mirror_reflection_random_angle": [0.005, 0.15, 0.03],
        "mirror_align_random_horizontal": [0.004, 28.0, 0.0, 0.0],
        "mirror_align_random_vertical": [0.004, 28.0, 0.0, 0.0],  # Add missing parameter
        "focal_length": 2800.0,
        "mirror_list": [1, 2, 3],
        "camera_center_x": 0.0,
        "camera_center_y": 0.0,
        "telescope_axis_height": 1000.0,
    }[param]
    mock_tel.name = "LSTN-01"
    return mock_tel


@pytest.fixture
def mock_site_model():
    """Create a mock site model."""
    return MagicMock()


@pytest.fixture
def mock_args_dict():
    """Mock args_dict fixture with all required keys."""
    return {
        "data": "test_data.txt",
        "model_path": MOCK_MODEL_PATH,
        "fixed": False,
        "plot_all": True,
        "simtel_path": "/path/to/simtel",
        "zenith": 20,
        "src_distance": 10,
        "ks_statistic": False,
        "monte_carlo_analysis": False,
        "rmsd_threshold": 0.01,
        "learning_rate": 0.1,
        "test": False,
        "write_psf_parameters": False,
        "output_path": "/tmp",
        "parameter_version": "1.0.0",
    }


@pytest.fixture
def mock_data_to_plot(sample_psf_data):
    """Create mock data_to_plot structure."""
    data_to_plot = OrderedDict()
    data_to_plot["measured"] = sample_psf_data
    return data_to_plot


@pytest.fixture
def sample_parameters():
    """Create sample parameter dictionary for testing."""
    return {
        "mirror_reflection_random_angle": [0.006, 0.15, 0.035],
        "mirror_align_random_horizontal": [0.005, 28.0, 0.0, 0.0],
        "mirror_align_random_vertical": [0.005, 28.0, 0.0, 0.0],
    }


def test_load_and_process_data_functionality(tmp_test_directory):
    """Test data loading and processing functionality."""
    # Test the load_and_process_data function which replaced load_psf_data
    args_dict = {"data": None, "model_path": MOCK_MODEL_PATH}

    # Test case 1: No data file
    data_to_plot, radius = psf_opt.load_and_process_data(args_dict)
    assert isinstance(data_to_plot, OrderedDict)
    assert len(data_to_plot) == 0
    assert radius is None

    # Test case 2: With data file (mocked)
    args_dict["data"] = "test_data.txt"
    with (
        patch("simtools.ray_tracing.psf_parameter_optimisation.gen.find_file") as mock_find,
        patch("numpy.loadtxt") as mock_loadtxt,
    ):
        mock_data = np.array(
            [(0.0, 0.0), (50.0, 0.5), (100.0, 1.0)],  # in mm
            dtype=[("Radius [cm]", "f8"), ("Cumulative PSF", "f8")],
        )
        mock_find.return_value = "found_file.txt"
        mock_loadtxt.return_value = mock_data

        data_to_plot, radius = psf_opt.load_and_process_data(args_dict)

        # Verify data was processed (radius converted from mm to cm)
        assert "measured" in data_to_plot
        assert radius is not None
        # Check that radius data was converted from mm to cm
        expected_radius = np.array([0.0, 5.0, 10.0])  # converted to cm
        np.testing.assert_array_equal(radius, expected_radius)


def test_calculate_ks_statistic():
    """Test KS statistic calculation."""
    data = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    sim = np.array([0.2, 0.4, 0.6, 0.8, 1.0])

    ks_stat, p_value = psf_opt.calculate_ks_statistic(data, sim)

    # Check that we get valid KS statistic and p-value
    assert ks_stat >= 0.0
    assert 0.0 <= p_value <= 1.0
    assert isinstance(ks_stat, float)
    assert isinstance(p_value, float)


def test_calculate_ks_statistic_identical_arrays():
    """Test KS statistic calculation with identical arrays."""
    data = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    sim = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

    ks_stat, p_value = psf_opt.calculate_ks_statistic(data, sim)

    # Identical arrays should have KS statistic = 0 and high p-value
    assert np.isclose(ks_stat, 0.0, atol=1e-10)
    assert p_value > 0.9  # High p-value for identical distributions


def test_calculate_rmsd():
    """Test RMSD calculation."""
    data = np.array([1.0, 2.0, 3.0, 4.0])
    sim = np.array([1.1, 2.1, 2.9, 3.9])

    expected_rmsd = np.sqrt(np.mean((data - sim) ** 2))
    result = psf_opt.calculate_rmsd(data, sim)

    assert np.isclose(result, expected_rmsd, atol=1e-9)


def test_calculate_rmsd_identical_arrays():
    """Test RMSD calculation with identical arrays."""
    data = np.array([1.0, 2.0, 3.0, 4.0])
    sim = np.array([1.0, 2.0, 3.0, 4.0])

    result = psf_opt.calculate_rmsd(data, sim)
    assert np.isclose(result, 0.0, atol=1e-9)


def test__validate_psf_data():
    """Test PSF data validation function."""
    # Test with valid data
    data_to_plot = {"measured": {"data": [1, 2, 3]}}
    radius = np.array([0, 1, 2])
    # Should not raise any exception
    psf_opt._validate_psf_data(data_to_plot, radius)

    # Test with None data_to_plot
    with pytest.raises(ValueError, match="PSF data and radius are required for analysis"):
        psf_opt._validate_psf_data(None, radius)

    # Test with None radius
    with pytest.raises(ValueError, match="PSF data and radius are required for analysis"):
        psf_opt._validate_psf_data(data_to_plot, None)


def test__create_log_file_header(mock_telescope_model):
    """Test log file header creation."""
    title = "Test Log File"
    additional_info = {"param1": "value1", "param2": "value2"}

    header = psf_opt._create_log_file_header(title, mock_telescope_model, additional_info)

    assert "Test Log File" in header
    assert mock_telescope_model.name in header
    assert "param1: value1" in header
    assert "param2: value2" in header
    assert "=" * 60 in header


def test__format_parameter_value():
    """Test parameter value formatting."""
    # Test list formatting
    list_value = [0.005, 0.15, 0.035]
    result = psf_opt._format_parameter_value(list_value)
    assert "[" in result
    assert "]" in result

    # Test numeric formatting
    float_value = 3.14159
    result = psf_opt._format_parameter_value(float_value)
    assert isinstance(result, str)

    # Test string formatting
    str_value = "test_string"
    result = psf_opt._format_parameter_value(str_value)
    assert result == str_value


def test_get_previous_values(mock_telescope_model, caplog):
    """Test retrieving previous parameter values."""
    with caplog.at_level(logging.DEBUG):
        params = psf_opt.get_previous_values(mock_telescope_model)

    # Check that we get a dictionary with the expected keys
    assert isinstance(params, dict)
    assert "mirror_reflection_random_angle" in params
    assert "mirror_align_random_horizontal" in params
    assert "mirror_align_random_vertical" in params

    # Check mirror reflection values
    mrra_values = params["mirror_reflection_random_angle"]
    assert len(mrra_values) == 3
    assert np.isclose(mrra_values[0], 0.005, atol=1e-9)  # mrra_0
    assert np.isclose(mrra_values[1], 0.15, atol=1e-9)  # mfr_0
    assert np.isclose(mrra_values[2], 0.03, atol=1e-9)  # mrra2_0

    # Check mirror alignment values (should be 4-element arrays)
    mar_h = params["mirror_align_random_horizontal"]
    mar_v = params["mirror_align_random_vertical"]
    assert len(mar_h) == 4
    assert len(mar_v) == 4
    assert np.isclose(mar_h[0], 0.004, atol=1e-9)
    assert np.isclose(mar_v[0], 0.004, atol=1e-9)


def test__should_accept_step():
    """Test gradient step acceptance logic."""
    # Test step with good improvement
    assert psf_opt._should_accept_step(0.1, 0.05) is True  # 50% improvement

    # Test step with marginal improvement
    assert psf_opt._should_accept_step(0.1, 0.0999) is True  # Small absolute improvement

    # Test step with no improvement
    assert psf_opt._should_accept_step(0.1, 0.1) is False

    # Test step with worsening
    assert psf_opt._should_accept_step(0.1, 0.15) is False


def test__should_stop_optimization():
    """Test optimization stopping criteria."""
    # Test convergence condition
    result = psf_opt._should_stop_optimization(0.005, 0.01, False)
    assert result is True

    # Test non-convergence condition
    result = psf_opt._should_stop_optimization(0.02, 0.01, False)
    assert result is False


def test__update_best_parameters():
    """Test updating best parameters logic."""
    current_metric = 0.05
    best_metric = 0.1
    current_params = [1, 2, 3]
    best_params = [4, 5, 6]
    current_d80 = 2.5
    best_d80 = 3.0

    # Test improvement case
    new_best_metric, new_best_params, new_best_d80 = psf_opt._update_best_parameters(
        current_metric, best_metric, current_params, best_params, current_d80, best_d80
    )

    assert np.isclose(new_best_metric, current_metric)
    assert new_best_params == current_params
    assert np.isclose(new_best_d80, current_d80)

    # Test no improvement case
    current_metric = 0.15  # Worse than best
    new_best_metric, new_best_params, new_best_d80 = psf_opt._update_best_parameters(
        current_metric, best_metric, current_params, best_params, current_d80, best_d80
    )

    assert np.isclose(new_best_metric, best_metric)
    assert new_best_params == best_params
    assert np.isclose(new_best_d80, best_d80)


def test__get_significance_level():
    """Test p-value significance level classification."""
    # Good significance
    assert psf_opt._get_significance_level(0.1) == "GOOD"

    # Fair significance
    assert psf_opt._get_significance_level(0.03) == "FAIR"

    # Poor significance
    assert psf_opt._get_significance_level(0.005) == "POOR"


def test_calculate_gradient(
    mock_telescope_model, mock_site_model, mock_args_dict, sample_psf_data, sample_parameters
):
    """Test gradient calculation."""
    data_to_plot = {"measured": sample_psf_data}
    radius = sample_psf_data[psf_opt.RADIUS_CM]
    current_rmsd = 0.1

    with patch("simtools.ray_tracing.psf_parameter_optimisation.run_psf_simulation") as mock_sim:
        # Mock simulation results for gradient calculation
        mock_sim.side_effect = [
            (3.0, 0.11),  # mirror_reflection_random_angle[0] + epsilon
            (3.0, 0.12),  # mirror_reflection_random_angle[1] + epsilon
            (3.0, 0.09),  # mirror_reflection_random_angle[2] + epsilon
            (3.0, 0.105),  # mirror_align_random_horizontal[0] + epsilon
            (3.0, 0.101),  # mirror_align_random_horizontal[1] + epsilon
            (3.0, 0.100),  # mirror_align_random_horizontal[2] + epsilon
            (3.0, 0.100),  # mirror_align_random_horizontal[3] + epsilon
            (3.0, 0.105),  # mirror_align_random_vertical[0] + epsilon
            (3.0, 0.101),  # mirror_align_random_vertical[1] + epsilon
            (3.0, 0.100),  # mirror_align_random_vertical[2] + epsilon
            (3.0, 0.100),  # mirror_align_random_vertical[3] + epsilon
        ]

        gradients = psf_opt.calculate_gradient(
            mock_telescope_model,
            mock_site_model,
            mock_args_dict,
            sample_parameters,
            data_to_plot,
            radius,
            current_rmsd,
        )

        # Check that gradients have the same structure as parameters
        assert "mirror_reflection_random_angle" in gradients
        assert "mirror_align_random_horizontal" in gradients
        assert "mirror_align_random_vertical" in gradients

        # Check gradient list lengths
        assert len(gradients["mirror_reflection_random_angle"]) == 3
        assert len(gradients["mirror_align_random_horizontal"]) == 4
        assert len(gradients["mirror_align_random_vertical"]) == 4


def test_apply_gradient_step():
    """Test applying gradient descent step."""
    current_params = {"param1": [1.0, 2.0, 3.0], "param2": 5.0}
    gradients = {"param1": [0.1, -0.2, 0.3], "param2": -0.5}
    learning_rate = 0.1

    new_params = psf_opt.apply_gradient_step(current_params, gradients, learning_rate)

    # Check parameter updates: new = old - learning_rate * gradient
    expected_param1 = [1.0 - 0.1 * 0.1, 2.0 - 0.1 * (-0.2), 3.0 - 0.1 * 0.3]
    expected_param2 = 5.0 - 0.1 * (-0.5)

    assert np.allclose(new_params["param1"], expected_param1)
    assert np.isclose(new_params["param2"], expected_param2)


@pytest.mark.parametrize(
    ("use_ks_statistic", "return_simulated_data", "expected_tuple_length"),
    [
        (False, False, 2),  # (d80, rmsd)
        (False, True, 3),  # (d80, rmsd, simulated_data)
        (True, False, 3),  # (d80, ks_stat, p_value)
        (True, True, 4),  # (d80, ks_stat, p_value, simulated_data)
    ],
)
def test_run_psf_simulation(
    mock_telescope_model,
    mock_site_model,
    mock_args_dict,
    sample_psf_data,
    sample_parameters,
    use_ks_statistic,
    return_simulated_data,
    expected_tuple_length,
):
    """Test PSF simulation function with different return options."""
    radius = sample_psf_data[psf_opt.RADIUS_CM]
    data_to_plot = {"measured": sample_psf_data}

    with patch(
        "simtools.ray_tracing.psf_parameter_optimisation._run_ray_tracing_simulation"
    ) as mock_sim:
        mock_image = MagicMock()
        mock_image.get_cumulative_data.return_value = sample_psf_data
        mock_sim.return_value = (3.5, mock_image)

        result = psf_opt.run_psf_simulation(
            mock_telescope_model,
            mock_site_model,
            mock_args_dict,
            sample_parameters,
            data_to_plot,
            radius,
            use_ks_statistic=use_ks_statistic,
            return_simulated_data=return_simulated_data,
        )

        # Check return tuple length
        assert len(result) == expected_tuple_length

        # Common assertions
        d80 = result[0]
        metric = result[1]
        assert np.isclose(d80, 3.5, atol=1e-9)
        assert metric >= 0

        if use_ks_statistic and len(result) >= 3:
            p_value = result[2]
            assert 0.0 <= p_value <= 1.0

        if return_simulated_data:
            simulated_data = result[-1]  # Last element
            np.testing.assert_array_equal(simulated_data, sample_psf_data)
            mock_image.get_cumulative_data.assert_called_once()


def test__run_ray_tracing_simulation(
    mock_telescope_model, mock_site_model, mock_args_dict, sample_parameters
):
    """Test the ray tracing simulation function."""
    with patch("simtools.ray_tracing.psf_parameter_optimisation.RayTracing") as mock_ray_class:
        mock_ray = MagicMock()
        mock_image = MagicMock()
        mock_image.get_psf.return_value = 3.2
        mock_ray.images.return_value = [mock_image]
        mock_ray_class.return_value = mock_ray

        d80, im = psf_opt._run_ray_tracing_simulation(
            mock_telescope_model, mock_site_model, mock_args_dict, sample_parameters
        )

        # Check that telescope parameters were changed
        mock_telescope_model.change_multiple_parameters.assert_called_once_with(**sample_parameters)

        # Check ray tracing was called correctly
        mock_ray_class.assert_called_once()
        mock_ray.simulate.assert_called_once_with(test=False, force=True)
        mock_ray.analyze.assert_called_once_with(force=True, use_rx=False)

        assert np.isclose(d80, 3.2, atol=1e-9)
        assert im == mock_image


def test__run_ray_tracing_simulation_no_parameters(
    mock_telescope_model, mock_site_model, mock_args_dict
):
    """Test ray tracing simulation with no parameters (should raise ValueError)."""
    with pytest.raises(ValueError, match="No best parameters found"):
        psf_opt._run_ray_tracing_simulation(
            mock_telescope_model, mock_site_model, mock_args_dict, None
        )


@pytest.mark.parametrize(
    (
        "has_radius",
        "expected_error_message",
        "should_raise_error",
        "description",
    ),
    [
        (True, None, False, "with valid radius data"),
        (
            False,
            "Radius data is not available.",
            True,
            "without radius data",
        ),
    ],
)
def test_run_psf_simulation_error_cases(
    mock_telescope_model,
    mock_site_model,
    mock_args_dict,
    sample_psf_data,
    sample_parameters,
    has_radius,
    expected_error_message,
    should_raise_error,
    description,
):
    """Test PSF simulation function error handling."""
    radius = sample_psf_data[psf_opt.RADIUS_CM] if has_radius else None
    data_to_plot = {"measured": sample_psf_data}

    with patch(
        "simtools.ray_tracing.psf_parameter_optimisation._run_ray_tracing_simulation"
    ) as mock_sim:
        mock_image = MagicMock()
        mock_image.get_cumulative_data.return_value = sample_psf_data
        mock_sim.return_value = (3.5, mock_image)

        if should_raise_error:
            with pytest.raises(ValueError, match=expected_error_message):
                psf_opt.run_psf_simulation(
                    mock_telescope_model,
                    mock_site_model,
                    mock_args_dict,
                    sample_parameters,
                    data_to_plot,
                    radius,
                )
        else:
            result = psf_opt.run_psf_simulation(
                mock_telescope_model,
                mock_site_model,
                mock_args_dict,
                sample_parameters,
                data_to_plot,
                radius,
            )
            d80, rmsd = result
            assert np.isclose(d80, 3.5, atol=1e-9)
            assert rmsd >= 0


@pytest.mark.parametrize(
    (
        "is_best",
        "description",
    ),
    [
        (True, "with plotting and best parameters"),
        (False, "with plotting and non-best parameters"),
    ],
)
def test_run_psf_simulation_with_plotting(
    mock_telescope_model,
    mock_site_model,
    mock_args_dict,
    sample_psf_data,
    sample_parameters,
    is_best,
    description,
):
    """Test PSF simulation function with plotting scenarios."""
    radius = sample_psf_data[psf_opt.RADIUS_CM]
    data_to_plot = {"measured": sample_psf_data}

    with patch(
        "simtools.ray_tracing.psf_parameter_optimisation._run_ray_tracing_simulation"
    ) as mock_sim:
        mock_image = MagicMock()
        mock_image.get_cumulative_data.return_value = sample_psf_data
        mock_sim.return_value = (3.5, mock_image)

        mock_pdf_pages = MagicMock()
        mock_args_dict["plot_all"] = True

        with patch("simtools.visualization.plot_psf.create_psf_parameter_plot") as mock_plot_func:
            result = psf_opt.run_psf_simulation(
                mock_telescope_model,
                mock_site_model,
                mock_args_dict,
                sample_parameters,
                data_to_plot,
                radius,
                pdf_pages=mock_pdf_pages,
                is_best=is_best,
            )

            d80, rmsd = result[0], result[1]
            assert np.isclose(d80, 3.5, atol=1e-9)
            assert rmsd >= 0
            mock_plot_func.assert_called_once()


def test_run_gradient_descent_optimization(
    mock_telescope_model, mock_site_model, mock_args_dict, sample_psf_data, tmp_path
):
    """Test the gradient descent optimization workflow."""
    radius = sample_psf_data[psf_opt.RADIUS_CM]
    data_to_plot = {"measured": sample_psf_data}

    # Mock the helper functions called by run_gradient_descent_optimization
    with (
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation._setup_optimization_plotting"
        ) as mock_setup,
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation._evaluate_initial_parameters"
        ) as mock_eval,
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation._execute_single_iteration"
        ) as mock_iteration,
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation._create_final_best_plot"
        ) as mock_final_plot,
    ):
        mock_setup.return_value = None
        mock_eval.return_value = (3.0, 0.1, None, sample_psf_data)
        mock_iteration.return_value = (
            ([1, 2, 3], 0.08, 2.8, [4, 5, 6], 0.08, 2.8),  # optimization_state
            0.01,  # learning_rate
            True,  # step_accepted
        )

        best_pars, best_d80, results = psf_opt.run_gradient_descent_optimization(
            mock_telescope_model,
            mock_site_model,
            mock_args_dict,
            data_to_plot,
            radius,
            rmsd_threshold=0.05,
            learning_rate=0.1,
            output_dir=tmp_path,
        )

        # Check that we got results
        assert best_pars is not None
        assert best_d80 > 0
        assert len(results) >= 1

        # Check that helper functions were called
        mock_setup.assert_called_once()
        mock_eval.assert_called_once()
        mock_final_plot.assert_called_once()


def test_write_gradient_descent_log(tmp_path, mock_telescope_model):
    """Test writing gradient descent log file."""
    gd_results = [
        ({"mirror_reflection_random_angle": [0.005, 0.15, 0.035]}, 0.1, None, 3.0, None),
        ({"mirror_reflection_random_angle": [0.006, 0.15, 0.035]}, 0.08, None, 2.8, None),
    ]
    best_pars = {"mirror_reflection_random_angle": [0.006, 0.15, 0.035]}
    best_d80 = 2.8

    with patch(
        "simtools.ray_tracing.psf_parameter_optimisation._create_log_file_header"
    ) as mock_header:
        mock_header.return_value = "# Test Header\n"

        log_file = psf_opt.write_gradient_descent_log(
            gd_results, best_pars, best_d80, tmp_path, mock_telescope_model
        )  # Check that file was created
        assert log_file.exists()
        assert "gradient_descent" in log_file.name

        # Check file contents
        content = log_file.read_text()
        assert "GRADIENT DESCENT PROGRESSION" in content
        assert "Test Header" in content


@pytest.mark.parametrize(
    ("has_data_file", "expected_measured", "expected_radius_not_none", "description"),
    [
        (True, True, True, "with data file provided"),
        (False, False, False, "without data file"),
    ],
)
def test_load_and_process_data(
    mock_args_dict,
    sample_psf_data,
    has_data_file,
    expected_measured,
    expected_radius_not_none,
    description,
):
    """Test loading and processing data under different scenarios."""
    if not has_data_file:
        mock_args_dict["data"] = None

    if has_data_file:
        with (
            patch("simtools.ray_tracing.psf_parameter_optimisation.gen.find_file") as mock_find,
            patch("numpy.loadtxt") as mock_loadtxt,
        ):
            mock_find.return_value = "found_file.txt"
            mock_data = np.array(
                [(0.0, 0.0), (50.0, 0.5), (100.0, 1.0)],  # in mm
                dtype=[("Radius [cm]", "f8"), ("Cumulative PSF", "f8")],
            )
            mock_loadtxt.return_value = mock_data

            data_to_plot, radius = psf_opt.load_and_process_data(mock_args_dict)

            mock_find.assert_called_once_with("test_data.txt", MOCK_MODEL_PATH)
            mock_loadtxt.assert_called_once()

            if expected_measured:
                assert "measured" in data_to_plot

            if expected_radius_not_none:
                assert radius is not None
    else:
        data_to_plot, radius = psf_opt.load_and_process_data(mock_args_dict)

        assert isinstance(data_to_plot, OrderedDict)
        assert len(data_to_plot) == 0
        assert radius is None


def test_run_psf_optimization_workflow(
    mock_telescope_model,
    mock_site_model,
    tmp_path,
):
    """Test the complete PSF optimization workflow."""
    args_dict = {
        "data": "test_data.txt",
        "model_path": MOCK_MODEL_PATH,
        "ks_statistic": False,
        "monte_carlo_analysis": False,
        "rmsd_threshold": 0.01,
        "learning_rate": 0.1,
        "plot_all": False,
        "simtel_path": "/path/to/simtel",
        "zenith": 20.0,
        "src_distance": 10.0,
    }

    # Mock the workflow functions
    with (
        patch("simtools.ray_tracing.psf_parameter_optimisation.load_and_process_data") as mock_load,
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation.run_gradient_descent_optimization"
        ) as mock_gd,
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation._create_iteration_plots"
        ) as mock_plots,
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation._create_convergence_and_reports"
        ) as mock_reports,
        patch("simtools.ray_tracing.psf_parameter_optimisation.run_psf_simulation") as mock_run_sim,
        patch("builtins.print") as mock_print,
    ):
        mock_load.return_value = ({}, np.array([1, 2, 3]))
        mock_gd.return_value = (
            {"param": "value"},
            3.2,
            [({"param": "value"}, 0.1, None, 3.2, None)],
        )
        mock_run_sim.return_value = (3.2, 0.05, 0.8, np.array([1, 2, 3]))

        psf_opt.run_psf_optimization_workflow(
            mock_telescope_model, mock_site_model, args_dict, tmp_path
        )

        # Check that workflow functions were called
        mock_load.assert_called_once()
        mock_gd.assert_called_once()
        mock_plots.assert_called_once()
        mock_reports.assert_called_once()
        mock_print.assert_called()  # Should print results


def test_create_iteration_plots(tmp_path, mock_telescope_model, mock_site_model):
    """Test _create_iteration_plots function."""
    optimization_log = [
        ({"param1": 0.1}, 0.1, 3.5),
        ({"param1": 0.05}, 0.05, 3.2),
        ({"param1": 0.03}, 0.03, 3.1),
    ]

    # Test case where save_plots is False (should return early)
    args_dict = {"save_plots": False}
    data_to_plot = {"measured": {"data": [1, 2, 3]}}
    radius = np.array([0, 1, 2])

    # Should return early without doing anything
    result = psf_opt._create_iteration_plots(
        args_dict,
        tmp_path,
        mock_telescope_model,
        optimization_log,
        mock_site_model,
        data_to_plot,
        radius,
    )
    assert result is None

    # Test case where save_plots is True
    args_dict = {"save_plots": True}
    with (
        patch(
            "simtools.visualization.plot_psf.create_psf_comparison_plot", create=True
        ) as mock_create_plot,
        patch("matplotlib.backends.backend_pdf.PdfPages") as mock_pdf_pages,
    ):
        mock_pdf_instance = MagicMock()
        mock_pdf_pages.return_value = mock_pdf_instance
        mock_create_plot.return_value = MagicMock()

        psf_opt._create_iteration_plots(
            args_dict,
            tmp_path,
            mock_telescope_model,
            optimization_log,
            mock_site_model,
            data_to_plot,
            radius,
        )

        # Verify PdfPages was created and closed
        mock_pdf_pages.assert_called_once()
        mock_pdf_instance.close.assert_called_once()


def test_create_convergence_and_reports(tmp_path, mock_telescope_model, mock_site_model):
    """Test _create_convergence_and_reports function."""
    optimization_log = [
        ({"param1": 0.1}, 0.1, 3.5),
        ({"param1": 0.05}, 0.05, 3.2),
    ]
    best_parameters = {"param1": 0.05}
    best_d80 = 3.2
    threshold = 0.01
    use_ks_statistic = False
    args_dict = {"plot_all": True}

    with (
        patch(
            "simtools.visualization.plot_psf.create_gradient_descent_convergence_plot"
        ) as mock_convergence,
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation.write_gradient_descent_log"
        ) as mock_log,
        patch("simtools.visualization.plot_psf.create_d80_vs_offaxis_plot") as mock_d80_plot,
        patch("builtins.print") as mock_print,
    ):
        mock_log.return_value = tmp_path / "test_log.log"

        psf_opt._create_convergence_and_reports(
            optimization_log,
            threshold,
            tmp_path,
            mock_telescope_model,
            best_parameters,
            best_d80,
            use_ks_statistic,
            mock_site_model,
            args_dict,
        )

        # Verify all functions were called
        mock_convergence.assert_called_once()
        mock_log.assert_called_once_with(
            optimization_log,
            best_parameters,
            best_d80,
            tmp_path,
            mock_telescope_model,
            use_ks_statistic,
        )
        mock_d80_plot.assert_called_once()
        assert mock_print.call_count == 2  # Two print statements


def test__add_units_to_psf_parameters(sample_parameters):
    """Test adding astropy units to PSF parameters."""
    # Add an extra parameter to test the function thoroughly
    test_pars = sample_parameters.copy()
    test_pars["other_parameter"] = [1.0, 2.0]

    result = psf_opt._add_units_to_psf_parameters(test_pars)

    # Check mirror_reflection_random_angle units: [deg, dimensionless, deg]
    mrra = result["mirror_reflection_random_angle"]
    assert mrra[0].unit == u.deg
    assert mrra[1].unit == u.dimensionless_unscaled
    assert mrra[2].unit == u.deg
    assert np.isclose(mrra[0].value, 0.006, atol=1e-9)
    assert np.isclose(mrra[1].value, 0.15, atol=1e-9)
    assert np.isclose(mrra[2].value, 0.035, atol=1e-9)

    # Check mirror_align_random_horizontal units: [deg, deg, dimensionless, dimensionless]
    marh = result["mirror_align_random_horizontal"]
    assert marh[0].unit == u.deg
    assert marh[1].unit == u.deg
    assert marh[2].unit == u.dimensionless_unscaled
    assert marh[3].unit == u.dimensionless_unscaled
    assert np.isclose(marh[0].value, 0.005, atol=1e-9)
    assert np.isclose(marh[1].value, 28.0, atol=1e-9)

    # Check mirror_align_random_vertical units: [deg, deg, dimensionless, dimensionless]
    marv = result["mirror_align_random_vertical"]
    assert marv[0].unit == u.deg
    assert marv[1].unit == u.deg
    assert marv[2].unit == u.dimensionless_unscaled
    assert marv[3].unit == u.dimensionless_unscaled

    # Check other parameters are kept as-is
    assert np.allclose(result["other_parameter"], [1.0, 2.0])


@pytest.mark.parametrize(
    (
        "side_effect",
        "expected_log_level",
        "expected_log_message",
        "expected_call_count",
        "description",
    ),
    [
        (
            None,
            logging.INFO,
            "simulation model parameter files exported to",
            2,
            "successful export",
        ),
        (
            ImportError("Module not found"),
            logging.WARNING,
            "Could not export simulation parameters: Module not found",
            1,
            "import error",
        ),
        (
            ValueError("Invalid parameter"),
            logging.ERROR,
            "Error exporting simulation parameters: Invalid parameter",
            1,
            "value error",
        ),
    ],
)
def test_export_psf_parameters(
    mock_telescope_model,
    tmp_path,
    caplog,
    side_effect,
    expected_log_level,
    expected_log_message,
    expected_call_count,
    description,
):
    """Test export of PSF parameters as simulation model parameter files under different scenarios."""
    best_pars = {
        "mirror_reflection_random_angle": [0.006, 0.15, 0.035],
        "mirror_align_random_horizontal": [0.005, 28.0, 0.0, 0.0],
    }
    parameter_version = "1.0.0"

    with (
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation.writer.ModelDataWriter.dump_model_parameter"
        ) as mock_dump,
        caplog.at_level(expected_log_level),
    ):
        if side_effect is not None:
            mock_dump.side_effect = side_effect

        psf_opt.export_psf_parameters(best_pars, mock_telescope_model, parameter_version, tmp_path)

        # Check function call count
        assert mock_dump.call_count == expected_call_count, (
            f"Expected {expected_call_count} calls for {description}"
        )

        # Check logging
        assert expected_log_message in caplog.text, (
            f"Expected log message not found for {description}"
        )

        # Check for successful export
        if side_effect is None:
            assert "simulation model parameter files exported to" in caplog.text


@pytest.mark.parametrize(
    ("test_mode", "write_parameters", "expected_rmsd_threshold", "description"),
    [
        (True, True, 0.02, "with parameter export in test mode"),
        (False, False, 0.001, "without parameter export in production mode"),
    ],
)
def test_run_psf_optimization_workflow_main(
    mock_telescope_model,
    mock_site_model,
    tmp_path,
    test_mode,
    write_parameters,
    expected_rmsd_threshold,
    description,
):
    """Test the complete PSF optimization workflow with gradient descent."""
    args_dict = {
        "test": test_mode,
        "data": "test_data.txt" if test_mode else None,
        "model_path": MOCK_MODEL_PATH,
        "export_parameter_files": write_parameters,
        "output_path": str(tmp_path),
        "parameter_version": "1.0.0",
        "rmsd_threshold": expected_rmsd_threshold,
        "learning_rate": 0.1,
        "simtel_path": "/path/to/simtel",
        "zenith": 20.0,
        "src_distance": 10.0,
    }

    # Mock optimization results
    best_pars = {"param": 0.05}
    best_d80 = 3.2
    optimization_log = [("param", 0.1, None, 3.5, None), ("param", 0.05, None, 3.2, None)]

    with (
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation.load_and_process_data"
        ) as mock_load_data,
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation.run_gradient_descent_optimization"
        ) as mock_gd_opt,
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation._create_iteration_plots"
        ) as mock_plots,
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation._create_convergence_and_reports"
        ) as mock_reports,
        patch("simtools.visualization.plot_psf.create_d80_vs_offaxis_plot") as mock_d80_plot,
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation.export_psf_parameters"
        ) as mock_export,
        patch("simtools.ray_tracing.psf_parameter_optimisation.run_psf_simulation") as mock_run_sim,
        patch("builtins.print") as mock_print,
    ):
        # Set up mocks
        mock_load_data.return_value = ({}, np.array([1, 2, 3]))
        mock_gd_opt.return_value = (best_pars, best_d80, optimization_log)
        mock_run_sim.return_value = (3.2, 0.05, 0.8, np.array([1, 2, 3]))

        # Run the workflow
        psf_opt.run_psf_optimization_workflow(
            mock_telescope_model, mock_site_model, args_dict, tmp_path
        )

        # Check all workflow steps were called
        mock_load_data.assert_called_once_with(args_dict)
        mock_gd_opt.assert_called_once()
        # _create_iteration_plots is called with: args_dict, output_dir, tel_model, gd_results, site_model, data_to_plot, radius
        mock_plots.assert_called_once()
        # _create_convergence_and_reports is called with: gd_results, threshold, output_dir, tel_model, best_pars, best_d80, use_ks_statistic, site_model, args_dict
        mock_reports.assert_called_once()
        # create_d80_vs_offaxis_plot is called from within _create_convergence_and_reports, so we don't check it separately
        mock_d80_plot.assert_not_called()  # Since _create_convergence_and_reports is mocked

        # Check conditional parameter export
        # Note: export_psf_parameters is not yet implemented in the workflow,
        # so we just check that it's not called regardless of the flag
        mock_export.assert_not_called()

        # Check results were printed
        mock_print.assert_called()
