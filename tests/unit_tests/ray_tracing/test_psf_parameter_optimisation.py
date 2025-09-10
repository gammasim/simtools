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
    """Create a mock args dictionary."""
    return {
        "test": False,
        "fixed": False,
        "simtel_path": "/path/to/simtel",
        "zenith": 20.0,
        "src_distance": 10.0,
        "data": "test_data.txt",
        "model_path": MOCK_MODEL_PATH,
        "plot_all": True,
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


def test_load_psf_data(tmp_test_directory):
    """Test loading PSF data from file."""
    test_file = "tests/resources/PSFcurve_data_v2.txt"

    result = psf_opt.load_psf_data(str(test_file))

    # Check data structure
    assert psf_opt.RADIUS_CM in result.dtype.names
    assert psf_opt.CUMULATIVE_PSF in result.dtype.names

    expected_radius = (
        np.array(
            [
                0,
                1.1601550664045037,
                2.3203101328090074,
                3.480465199213511,
                4.640620265618015,
                5.800775332022519,
            ]
        )
        * 0.1
    )

    np.testing.assert_array_almost_equal(result[psf_opt.RADIUS_CM][:6], expected_radius)


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


def test_add_parameters():
    """Test adding parameters to the all_parameters list."""
    all_parameters = []
    mirror_reflection = 0.006
    mirror_align = 0.005
    mirror_reflection_fraction = 0.12
    mirror_reflection_2 = 0.04

    psf_opt.add_parameters(
        all_parameters,
        mirror_reflection,
        mirror_align,
        mirror_reflection_fraction,
        mirror_reflection_2,
    )

    assert len(all_parameters) == 1
    pars = all_parameters[0]

    expected_pars = {
        "mirror_reflection_random_angle": [0.006, 0.12, 0.04],
        "mirror_align_random_horizontal": [0.005, 28.0, 0.0, 0.0],
        "mirror_align_random_vertical": [0.005, 28.0, 0.0, 0.0],
    }

    assert pars == expected_pars


def test_add_parameters_default_values():
    """Test adding parameters with default values."""
    all_parameters = []
    mirror_reflection = 0.007
    mirror_align = 0.004

    psf_opt.add_parameters(all_parameters, mirror_reflection, mirror_align)

    assert len(all_parameters) == 1
    pars = all_parameters[0]

    # Check default values are used
    assert np.isclose(pars["mirror_reflection_random_angle"][1], 0.15, atol=1e-9)
    assert np.isclose(pars["mirror_reflection_random_angle"][2], 0.035, atol=1e-9)


def test_get_previous_values(mock_telescope_model, caplog):
    """Test retrieving previous parameter values."""
    with caplog.at_level(logging.DEBUG):
        mrra_0, mfr_0, mrra2_0, mar_0 = psf_opt.get_previous_values(mock_telescope_model)

    assert np.isclose(mrra_0, 0.005, atol=1e-9)
    assert np.isclose(mfr_0, 0.15, atol=1e-9)
    assert np.isclose(mrra2_0, 0.03, atol=1e-9)
    assert np.isclose(mar_0, 0.004, atol=1e-9)

    # Check debug logging
    assert "Previous parameter values:" in caplog.text
    assert "MRRA = 0.005" in caplog.text


@pytest.mark.parametrize(
    (
        "fixed_mode",
        "telescope_name",
        "n_runs",
        "expected_log_message",
        "check_fixed_value",
        "check_mar_zero",
        "description",
    ),
    [
        (False, "LSTN-01", 3, None, False, False, "with fixed=False and single mirror telescope"),
        (
            True,
            "LSTN-01",
            2,
            "fixed=True - First entry of mirror_reflection_random_angle is kept fixed.",
            True,
            False,
            "with fixed=True",
        ),
        (False, "SSTS-01", 3, None, False, True, "with dual mirror telescope"),
        (
            False,
            "LSTN-01",
            3,
            None,
            False,
            False,
            "with single mirror telescope using random mar values",
        ),
    ],
)
def test_generate_random_parameters(
    mock_args_dict,
    mock_telescope_model,
    caplog,
    fixed_mode,
    telescope_name,
    n_runs,
    expected_log_message,
    check_fixed_value,
    check_mar_zero,
    description,
):
    """Test generating random parameters under different scenarios."""
    all_parameters = []
    mock_args_dict["fixed"] = fixed_mode

    mock_telescope_model.name = telescope_name

    # Capture logging if expected
    if expected_log_message:
        with caplog.at_level(logging.DEBUG):
            psf_opt.generate_random_parameters(
                all_parameters,
                n_runs,
                mock_args_dict,
                0.005,
                0.15,
                0.03,
                0.004,
                mock_telescope_model,
            )
    else:
        psf_opt.generate_random_parameters(
            all_parameters, n_runs, mock_args_dict, 0.005, 0.15, 0.03, 0.004, mock_telescope_model
        )

    assert len(all_parameters) == n_runs

    # Check that all parameter sets have expected structure
    for pars in all_parameters:
        assert "mirror_reflection_random_angle" in pars
        assert "mirror_align_random_horizontal" in pars
        assert "mirror_align_random_vertical" in pars
        assert len(pars["mirror_reflection_random_angle"]) == 3
        assert len(pars["mirror_align_random_horizontal"]) == 4

    # Scenario-specific assertions
    if expected_log_message:
        assert expected_log_message in caplog.text

    if check_fixed_value:
        # When fixed=True, the first mirror reflection parameter should be exactly the original value
        for pars in all_parameters:
            assert np.isclose(pars["mirror_reflection_random_angle"][0], 0.005, atol=1e-9)

    if check_mar_zero:
        # Check that all parameter sets have mar set to 0 for dual mirror telescopes
        for pars in all_parameters:
            assert np.isclose(pars["mirror_align_random_horizontal"][0], 0.0, atol=1e-9)
            assert np.isclose(pars["mirror_align_random_vertical"][0], 0.0, atol=1e-9)
    elif telescope_name == "LSTN-01" and not check_fixed_value:
        # Check that parameter sets have non-zero mar values (within expected range) for single mirror telescopes
        for pars in all_parameters:
            mar_horizontal = pars["mirror_align_random_horizontal"][0]
            mar_vertical = pars["mirror_align_random_vertical"][0]

            assert mar_horizontal >= 0.0
            assert mar_vertical >= 0.0
            assert np.isclose(mar_horizontal, mar_vertical, atol=1e-9)  # They should be the same


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
        "return_simulated_data",
        "should_raise_error",
        "expected_d80",
        "expected_error_message",
        "description",
    ),
    [
        (True, True, False, 3.5, None, "data only with radius"),
        (
            False,
            True,
            True,
            None,
            "Radius data is not available.",
            "data only without radius",
        ),
        (True, False, False, 3.5, None, "without returning simulated data"),
    ],
)
def test_run_psf_simulation(
    mock_telescope_model,
    mock_site_model,
    mock_args_dict,
    sample_psf_data,
    sample_parameters,
    has_radius,
    return_simulated_data,
    should_raise_error,
    expected_d80,
    expected_error_message,
    description,
):
    """Test PSF simulation function under different scenarios."""
    radius = sample_psf_data[psf_opt.RADIUS_CM] if has_radius else None
    data_to_plot = {"measured": sample_psf_data}

    with patch(
        "simtools.ray_tracing.psf_parameter_optimisation._run_ray_tracing_simulation"
    ) as mock_sim:
        mock_image = MagicMock()
        mock_image.get_cumulative_data.return_value = sample_psf_data
        mock_sim.return_value = (3.5, mock_image)

        kwargs = {"return_simulated_data": return_simulated_data} if return_simulated_data else {}

        if should_raise_error:
            with pytest.raises(ValueError, match=expected_error_message):
                psf_opt.run_psf_simulation(
                    mock_telescope_model,
                    mock_site_model,
                    mock_args_dict,
                    sample_parameters,
                    data_to_plot,
                    radius,
                    **kwargs,
                )
        else:
            result = psf_opt.run_psf_simulation(
                mock_telescope_model,
                mock_site_model,
                mock_args_dict,
                sample_parameters,
                data_to_plot,
                radius,
                **kwargs,
            )

            # Common assertions
            d80, rmsd = result[0], result[1]
            assert np.isclose(d80, expected_d80, atol=1e-9)
            assert rmsd >= 0

            if return_simulated_data:
                simulated_data = result[2]
                np.testing.assert_array_equal(simulated_data, sample_psf_data)
                mock_image.get_cumulative_data.assert_called_once()


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
        kwargs = {"pdf_pages": mock_pdf_pages, "is_best": is_best}

        with patch(
            "simtools.ray_tracing.psf_parameter_optimisation._create_psf_simulation_plot"
        ) as mock_plot_func:
            result = psf_opt.run_psf_simulation(
                mock_telescope_model,
                mock_site_model,
                mock_args_dict,
                sample_parameters,
                data_to_plot,
                radius,
                **kwargs,
            )

            d80, rmsd = result[0], result[1]
            assert np.isclose(d80, 3.5, atol=1e-9)
            assert rmsd >= 0
            mock_plot_func.assert_called_once()

            # Check data cleanup behavior - simulated data should be cleaned up after plotting
            assert "simulated" not in data_to_plot


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
            patch("simtools.ray_tracing.psf_parameter_optimisation.load_psf_data") as mock_load,
        ):
            mock_find.return_value = "found_file.txt"
            mock_load.return_value = sample_psf_data

            data_to_plot, radius = psf_opt.load_and_process_data(mock_args_dict)

            mock_find.assert_called_once_with("test_data.txt", MOCK_MODEL_PATH)
            mock_load.assert_called_once_with("found_file.txt")

            if expected_measured:
                assert "measured" in data_to_plot
                np.testing.assert_array_equal(data_to_plot["measured"], sample_psf_data)

            if expected_radius_not_none:
                np.testing.assert_array_equal(radius, sample_psf_data[psf_opt.RADIUS_CM])
    else:
        data_to_plot, radius = psf_opt.load_and_process_data(mock_args_dict)

        assert isinstance(data_to_plot, OrderedDict)
        assert len(data_to_plot) == 0
        assert radius is None


def test__create_psf_simulation_plot(sample_psf_data, sample_parameters):
    """Test the _create_psf_simulation_plot function."""
    data_to_plot = {"measured": sample_psf_data}
    pars = sample_parameters
    d80 = 3.5
    rmsd = 0.123

    with (
        patch("simtools.ray_tracing.psf_parameter_optimisation.visualize.plot_1d") as mock_plot,
        patch("matplotlib.pyplot.clf") as mock_clf,
    ):
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_fig.get_axes.return_value = [mock_ax]
        mock_plot.return_value = mock_fig
        mock_pdf_pages = MagicMock()

        # Test with is_best=True
        psf_opt._create_psf_simulation_plot(
            data_to_plot, pars, d80, rmsd, is_best=True, pdf_pages=mock_pdf_pages
        )

        # Check that all matplotlib functions were called
        mock_plot.assert_called_once_with(
            data_to_plot,
            plot_difference=True,
            no_markers=True,
        )
        mock_ax.set_ylim.assert_called_once_with(0, 1.05)
        mock_ax.set_ylabel.assert_called_once_with(psf_opt.CUMULATIVE_PSF)
        mock_ax.set_title.assert_called_once()
        mock_ax.text.assert_called_once()
        mock_fig.text.assert_called_once()  # Footnote for best parameters
        mock_pdf_pages.savefig.assert_called_once_with(mock_fig, bbox_inches="tight")
        mock_clf.assert_called_once()

        # Reset mocks for second test
        mock_plot.reset_mock()
        mock_ax.reset_mock()
        mock_fig.reset_mock()
        mock_pdf_pages.reset_mock()
        mock_clf.reset_mock()

        # Test with is_best=False
        psf_opt._create_psf_simulation_plot(
            data_to_plot, pars, d80, rmsd, is_best=False, pdf_pages=mock_pdf_pages
        )

        # Check that matplotlib functions were called, but no footnote
        mock_plot.assert_called_once()
        mock_ax.set_ylim.assert_called_once_with(0, 1.05)
        mock_ax.set_ylabel.assert_called_once_with(psf_opt.CUMULATIVE_PSF)
        mock_ax.set_title.assert_called_once()
        mock_ax.text.assert_called_once()
        mock_fig.text.assert_not_called()  # No footnote for non-best parameters
        mock_pdf_pages.savefig.assert_called_once_with(mock_fig, bbox_inches="tight")
        mock_clf.assert_called_once()


@pytest.mark.parametrize(
    ("has_original_data", "is_best", "expected_footnote", "description"),
    [
        (False, True, True, "best parameters without original data"),
        (True, False, False, "non-best parameters with original data restoration"),
    ],
)
def test__create_plot_for_parameters(
    mock_data_to_plot,
    sample_psf_data,
    sample_parameters,
    has_original_data,
    is_best,
    expected_footnote,
    description,
):
    """Test creating a plot for a parameter set under different scenarios."""
    rmsd = 0.123
    d80 = 3.5

    # Set up original simulated data if needed
    original_simulated_data = None
    if has_original_data:
        original_simulated_data = {"original": "data"}
        mock_data_to_plot["simulated"] = original_simulated_data

    with (
        patch("simtools.ray_tracing.psf_parameter_optimisation.visualize.plot_1d") as mock_plot,
        patch("matplotlib.pyplot.clf") as mock_clf,
    ):
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_fig.get_axes.return_value = [mock_ax]
        mock_plot.return_value = mock_fig

        mock_pdf_pages = MagicMock()

        psf_opt._create_plot_for_parameters(
            sample_parameters,
            rmsd,
            d80,
            sample_psf_data,
            mock_data_to_plot,
            is_best=is_best,
            pdf_pages=mock_pdf_pages,
        )

        # Check that original simulated data was restored if it existed
        if has_original_data:
            assert mock_data_to_plot["simulated"] == original_simulated_data

        # Check plotting was called correctly
        mock_plot.assert_called_once()
        mock_pdf_pages.savefig.assert_called_once()
        mock_clf.assert_called_once()

        # Check specific assertions for best parameter case
        if is_best:
            mock_ax.set_ylim.assert_called_once_with(0, 1.05)
            mock_ax.set_ylabel.assert_called_once_with(psf_opt.CUMULATIVE_PSF)
            mock_ax.set_title.assert_called_once()
            mock_ax.text.assert_called_once()
            if expected_footnote:
                mock_fig.text.assert_called_once()  # Footnote for best parameters


@pytest.fixture
def test_config(request):
    """Fixture to provide test configuration parameters."""
    return {
        "mock_side_effect": request.param[0],
        "expected_best_index": request.param[1],
        "expected_results_count": request.param[2],
        "expected_log_level": request.param[3],
        "expected_log_message": request.param[4],
        "plot_all": request.param[5],
        "pdf_pages": request.param[6],
        "expected_plots_called": request.param[7],
        "description": request.param[8],
    }


@pytest.mark.parametrize(
    "test_config",
    [
        (
            [(3.5, 0.2, "sample_psf_data"), (3.2, 0.1, "sample_psf_data")],  # All succeed
            1,  # Second parameter set is better (lower RMSD)
            2,  # Two successful results
            None,  # No logging expected
            None,  # No log message expected
            False,  # No plotting
            None,  # No PDF pages
            False,  # No plots expected
            "all simulations succeed without plotting",
        ),
        (
            [
                ValueError("Simulation failed"),
                (3.2, 0.1, "sample_psf_data"),
            ],  # First fails, second succeeds
            1,  # Only successful parameter set (index 1)
            1,  # Only one successful result
            logging.WARNING,  # Warning level logging
            "Simulation failed for parameters",  # Expected log message
            True,  # With plotting
            "mock_pdf_pages",  # Mock PDF pages
            True,  # Plots expected
            "with simulation failures and plotting",
        ),
    ],
    indirect=True,
)
def test_find_best_parameters(
    mock_telescope_model,
    mock_site_model,
    mock_args_dict,
    mock_data_to_plot,
    sample_psf_data,
    caplog,
    test_config,
):
    """Test finding the best parameters with various scenarios including plotting and failures."""
    all_parameters = [
        {
            "mirror_reflection_random_angle": [0.005, 0.15, 0.035],
            "mirror_align_random_horizontal": [0.004, 28.0, 0.0, 0.0],
            "mirror_align_random_vertical": [0.004, 28.0, 0.0, 0.0],
        },
        {
            "mirror_reflection_random_angle": [0.006, 0.15, 0.035],
            "mirror_align_random_horizontal": [0.005, 28.0, 0.0, 0.0],
            "mirror_align_random_vertical": [0.005, 28.0, 0.0, 0.0],
        },
    ]
    radius = sample_psf_data[psf_opt.RADIUS_CM]

    # Set up args_dict based on test parameters
    mock_args_dict["plot_all"] = test_config["plot_all"]

    # Create mock_pdf_pages if needed
    mock_pdf_pages_obj = MagicMock() if test_config["pdf_pages"] == "mock_pdf_pages" else None

    processed_side_effect = []
    for effect in test_config["mock_side_effect"]:
        if isinstance(effect, tuple):
            # Replace string placeholder with actual sample_psf_data
            processed_effect = tuple(
                sample_psf_data if item == "sample_psf_data" else item for item in effect
            )
            processed_side_effect.append(processed_effect)
        else:
            processed_side_effect.append(effect)

    with (
        patch("simtools.ray_tracing.psf_parameter_optimisation.run_psf_simulation") as mock_sim,
        patch("simtools.ray_tracing.psf_parameter_optimisation._create_all_plots") as mock_plots,
    ):
        mock_sim.side_effect = processed_side_effect

        # Set up logging capture if needed
        if test_config["expected_log_level"]:
            with caplog.at_level(test_config["expected_log_level"]):
                best_pars, best_d80, results = psf_opt.find_best_parameters(
                    all_parameters,
                    mock_telescope_model,
                    mock_site_model,
                    mock_args_dict,
                    mock_data_to_plot,
                    radius,
                    pdf_pages=mock_pdf_pages_obj,
                )
        else:
            best_pars, best_d80, results = psf_opt.find_best_parameters(
                all_parameters,
                mock_telescope_model,
                mock_site_model,
                mock_args_dict,
                mock_data_to_plot,
                radius,
                pdf_pages=mock_pdf_pages_obj,
            )

        # Common assertions
        assert best_pars == all_parameters[test_config["expected_best_index"]]
        assert len(results) == test_config["expected_results_count"]
        assert np.isclose(best_d80, 3.2, atol=1e-9)  # Best result available
        assert mock_sim.call_count == 2  # Both parameter sets are attempted

        # Check logging if expected
        if test_config["expected_log_message"]:
            assert test_config["expected_log_message"] in caplog.text

        # Check plotting behavior based on plot_all setting
        if test_config["expected_plots_called"]:
            mock_plots.assert_called_once()
        else:
            mock_plots.assert_not_called()


def test_create_all_plots(mock_data_to_plot):
    """Test creating plots for all parameter sets."""
    pars1 = {"param1": "value1"}
    pars2 = {"param2": "value2"}
    results = [
        (pars1, 0.2, 3.5, {"data": "sim1"}),
        (pars2, 0.1, 3.2, {"data": "sim2"}),
    ]
    best_pars = pars2

    with patch(
        "simtools.ray_tracing.psf_parameter_optimisation._create_plot_for_parameters"
    ) as mock_plot:
        mock_pdf_pages = MagicMock()

        psf_opt._create_all_plots(results, best_pars, mock_data_to_plot, mock_pdf_pages)

        # Check that _create_plot_for_parameters was called for each result
        assert mock_plot.call_count == 2

        # Check that the second call (best parameters) has is_best=True
        calls = mock_plot.call_args_list
        # _create_plot_for_parameters(pars, rmsd, d80, simulated_data, data_to_plot, is_best, pdf_pages)
        # So is_best is the 6th argument (index 5)
        first_call_args = calls[0][0]  # positional arguments
        second_call_args = calls[1][0]
        assert first_call_args[5] is False  # First call, is_best=False
        assert second_call_args[5] is True  # Second call, is_best=True


def test_create_d80_vs_offaxis_plot(
    mock_telescope_model, mock_site_model, mock_args_dict, tmp_path, sample_parameters
):
    """Test creating D80 vs off-axis angle plot."""
    with (
        patch("simtools.ray_tracing.psf_parameter_optimisation.RayTracing") as mock_ray_class,
        patch("matplotlib.pyplot.title") as mock_title,
        patch("matplotlib.pyplot.xlabel") as mock_xlabel,
        patch("matplotlib.pyplot.ylabel") as mock_ylabel,
        patch("matplotlib.pyplot.xticks") as mock_xticks,
        patch("matplotlib.pyplot.xlim") as mock_xlim,
        patch("matplotlib.pyplot.grid") as mock_grid,
        patch("matplotlib.pyplot.close") as mock_close,
        patch("simtools.ray_tracing.psf_parameter_optimisation.visualize.save_figure") as mock_save,
    ):
        mock_ray = MagicMock()
        mock_ray_class.return_value = mock_ray

        psf_opt.create_d80_vs_offaxis_plot(
            mock_telescope_model, mock_site_model, mock_args_dict, sample_parameters, tmp_path
        )

        # Check that telescope parameters were changed
        mock_telescope_model.change_multiple_parameters.assert_called_once_with(**sample_parameters)

        # Check ray tracing was set up correctly
        mock_ray_class.assert_called_once()
        mock_ray.simulate.assert_called_once_with(test=False, force=True)
        mock_ray.analyze.assert_called_once_with(force=True)

        assert mock_title.call_count == 2
        assert mock_xlabel.call_count == 2
        assert mock_ylabel.call_count == 2
        assert mock_xticks.call_count == 2
        assert mock_xlim.call_count == 2
        assert mock_grid.call_count == 2
        assert mock_save.call_count == 2
        mock_close.assert_called_once_with("all")


def test_write_tested_parameters_to_file(tmp_path, mock_telescope_model):
    """Test writing tested parameters to file."""
    pars1 = {
        "mirror_reflection_random_angle": [0.005, 0.15, 0.035],
        "mirror_align_random_horizontal": [0.004, 28.0, 0.0, 0.0],
    }
    pars2 = {
        "mirror_reflection_random_angle": [0.006, 0.15, 0.035],
        "mirror_align_random_horizontal": [0.005, 28.0, 0.0, 0.0],
    }
    results = [
        (pars1, 0.2, 3.5, {}),
        (pars2, 0.1, 3.2, {}),
    ]
    best_pars = pars2  # Same object as used in results
    best_d80 = 3.2

    param_file = psf_opt.write_tested_parameters_to_file(
        results, best_pars, best_d80, tmp_path, mock_telescope_model
    )

    # Check that file was created
    assert param_file.exists()
    assert param_file == tmp_path / f"psf_optimization_{mock_telescope_model.name}.log"

    # Check file contents
    content = param_file.read_text()
    assert "PARAMETER TESTING RESULTS:" in content
    assert "[TESTED] Set 001: RMSD=0.20000, D80=3.50000 cm" in content
    assert "[BEST] Set 002: RMSD=0.10000, D80=3.20000 cm" in content
    assert "OPTIMIZATION SUMMARY:" in content
    assert "Best D80: 3.20000 cm" in content


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
            assert (
                "Exporting best PSF parameters as simulation model parameter files" in caplog.text
            )


@pytest.mark.parametrize(
    ("test_mode", "data_file", "write_parameters", "expected_n_runs", "description"),
    [
        (True, "test_data.txt", True, 5, "with parameter export in test mode"),
        (False, None, False, 50, "without parameter export in production mode"),
    ],
)
def test_run_psf_optimization_workflow(
    mock_telescope_model,
    mock_site_model,
    tmp_path,
    test_mode,
    data_file,
    write_parameters,
    expected_n_runs,
    description,
):
    """Test the complete PSF optimization workflow."""
    args_dict = {
        "test": test_mode,
        "data": data_file,
        "model_path": MOCK_MODEL_PATH,
        "write_psf_parameters": write_parameters,
        "output_path": str(tmp_path),
        "parameter_version": "1.0.0",
        "n_runs": expected_n_runs,  # Add the expected n_runs to args_dict
    }

    # Mock sample results
    sample_results = [
        (
            {
                "mirror_reflection_random_angle": [0.006, 0.15, 0.035],
                "mirror_align_random_horizontal": [0.005, 28.0, 0.0, 0.0],
            },
            0.1,
            3.2,
            {},
        )
    ]

    best_pars = {
        "mirror_reflection_random_angle": [0.006, 0.15, 0.035],
        "mirror_align_random_horizontal": [0.005, 28.0, 0.0, 0.0],
    }

    with (
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation.get_previous_values"
        ) as mock_get_prev,
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation.generate_random_parameters"
        ) as mock_gen_params,
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation.load_and_process_data"
        ) as mock_load_data,
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation.find_best_parameters"
        ) as mock_find_best,
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation.write_tested_parameters_to_file"
        ) as mock_write_params,
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation.create_d80_vs_offaxis_plot"
        ) as mock_create_plot,
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation.export_psf_parameters"
        ) as mock_export_parameters,
        patch("builtins.print") as mock_print,
    ):
        # Set up mocks
        mock_get_prev.return_value = (0.005, 0.15, 0.03, 0.004)
        mock_load_data.return_value = ({}, None)
        mock_find_best.return_value = (best_pars, 3.2, sample_results)
        mock_write_params.return_value = tmp_path / "test_params.txt"

        # Run the workflow
        psf_opt.run_psf_optimization_workflow(
            mock_telescope_model, mock_site_model, args_dict, tmp_path
        )

        # Common assertions for both scenarios
        mock_get_prev.assert_called_once_with(mock_telescope_model)
        mock_gen_params.assert_called_once()
        mock_load_data.assert_called_once_with(args_dict)
        mock_find_best.assert_called_once()
        mock_write_params.assert_called_once_with(
            sample_results, best_pars, 3.2, tmp_path, mock_telescope_model
        )
        mock_create_plot.assert_called_once_with(
            mock_telescope_model, mock_site_model, args_dict, best_pars, tmp_path
        )

        # Check n_runs parameter
        call_args = mock_gen_params.call_args[0]
        n_runs = call_args[1]
        assert n_runs == expected_n_runs, f"Expected n_runs={expected_n_runs} for {description}"

        if write_parameters:
            mock_export_parameters.assert_called_once_with(
                best_pars, mock_telescope_model, "1.0.0", tmp_path.parent
            )
            # Check output messages for test mode
            mock_print.assert_any_call(
                f"\nParameter results written to {tmp_path / 'test_params.txt'}"
            )
            mock_print.assert_any_call("D80 vs off-axis angle plots created successfully")
            mock_print.assert_any_call("\nBest parameters:")
        else:
            # parameter export should NOT be called
            mock_export_parameters.assert_not_called()
