#!/usr/bin/python3

import logging
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import simtools.ray_tracing.psf_parameter_optimisation as psf_opt


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
    mock_tel.name = "test_telescope"
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
        "model_path": "/path/to/model",
        "plot_all": True,
    }


@pytest.fixture
def mock_data_to_plot(sample_psf_data):
    """Create mock data_to_plot structure."""
    data_to_plot = OrderedDict()
    data_to_plot["measured"] = sample_psf_data
    return data_to_plot


@pytest.fixture
def mock_ray_tracing():
    """Create a mock ray tracing object with image results."""
    mock_ray = MagicMock()
    mock_image = MagicMock()
    mock_image.get_psf.return_value = 3.5
    mock_image.get_cumulative_data.return_value = {
        psf_opt.RADIUS_CM: np.linspace(0, 10, 21),
        psf_opt.CUMULATIVE_PSF: np.linspace(0, 1, 21),
    }
    mock_ray.images.return_value = [mock_image]
    return mock_ray


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

    assert pytest.approx(result) == expected_rmsd


def test_calculate_rmsd_identical_arrays():
    """Test RMSD calculation with identical arrays."""
    data = np.array([1.0, 2.0, 3.0, 4.0])
    sim = np.array([1.0, 2.0, 3.0, 4.0])

    result = psf_opt.calculate_rmsd(data, sim)
    assert result == 0.0


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
    assert pars["mirror_reflection_random_angle"][1] == 0.15
    assert pars["mirror_reflection_random_angle"][2] == 0.035


def test_get_previous_values(mock_telescope_model, caplog):
    """Test retrieving previous parameter values."""
    with caplog.at_level(logging.DEBUG):
        mrra_0, mfr_0, mrra2_0, mar_0 = psf_opt.get_previous_values(mock_telescope_model)

    assert mrra_0 == 0.005
    assert mfr_0 == 0.15
    assert mrra2_0 == 0.03
    assert mar_0 == 0.004

    # Check debug logging
    assert "Previous parameter values:" in caplog.text
    assert "MRRA = 0.005" in caplog.text


def test_generate_random_parameters_fixed_false(mock_args_dict):
    """Test generating random parameters with fixed=False."""
    all_parameters = []
    n_runs = 3
    mock_args_dict["fixed"] = False

    psf_opt.generate_random_parameters(
        all_parameters, n_runs, mock_args_dict, 0.005, 0.15, 0.03, 0.004
    )

    assert len(all_parameters) == n_runs

    # Check that all parameter sets have expected structure
    for pars in all_parameters:
        assert "mirror_reflection_random_angle" in pars
        assert "mirror_align_random_horizontal" in pars
        assert "mirror_align_random_vertical" in pars
        assert len(pars["mirror_reflection_random_angle"]) == 3
        assert len(pars["mirror_align_random_horizontal"]) == 4


def test_generate_random_parameters_fixed_true(mock_args_dict, caplog):
    """Test generating random parameters with fixed=True."""
    all_parameters = []
    n_runs = 2
    mock_args_dict["fixed"] = True

    with caplog.at_level(logging.DEBUG):
        psf_opt.generate_random_parameters(
            all_parameters, n_runs, mock_args_dict, 0.005, 0.15, 0.03, 0.004
        )

    assert len(all_parameters) == n_runs
    assert (
        "fixed=True - First entry of mirror_reflection_random_angle is kept fixed." in caplog.text
    )

    # When fixed=True, the first mirror reflection parameter should be exactly the original value
    for pars in all_parameters:
        assert pars["mirror_reflection_random_angle"][0] == 0.005


def test_run_ray_tracing_simulation(mock_telescope_model, mock_site_model, mock_args_dict):
    """Test the ray tracing simulation function."""
    pars = {
        "mirror_reflection_random_angle": [0.006, 0.15, 0.035],
        "mirror_align_random_horizontal": [0.005, 28.0, 0.0, 0.0],
        "mirror_align_random_vertical": [0.005, 28.0, 0.0, 0.0],
    }

    with patch("simtools.ray_tracing.psf_parameter_optimisation.RayTracing") as mock_ray_class:
        mock_ray = MagicMock()
        mock_image = MagicMock()
        mock_image.get_psf.return_value = 3.2
        mock_ray.images.return_value = [mock_image]
        mock_ray_class.return_value = mock_ray

        d80, im = psf_opt._run_ray_tracing_simulation(
            mock_telescope_model, mock_site_model, mock_args_dict, pars
        )

        # Check that telescope parameters were changed
        mock_telescope_model.change_multiple_parameters.assert_called_once_with(**pars)

        # Check ray tracing was called correctly
        mock_ray_class.assert_called_once()
        mock_ray.simulate.assert_called_once_with(test=False, force=True)
        mock_ray.analyze.assert_called_once_with(force=True, use_rx=False)

        assert d80 == 3.2
        assert im == mock_image


def test_run_ray_tracing_simulation_no_parameters(
    mock_telescope_model, mock_site_model, mock_args_dict
):
    """Test ray tracing simulation with no parameters (should raise ValueError)."""
    with pytest.raises(ValueError, match="No best parameters found"):
        psf_opt._run_ray_tracing_simulation(
            mock_telescope_model, mock_site_model, mock_args_dict, None
        )


def test_run_psf_simulation_data_only(
    mock_telescope_model, mock_site_model, mock_args_dict, mock_data_to_plot, sample_psf_data
):
    """Test running PSF simulation for data only (no plotting)."""
    pars = {
        "mirror_reflection_random_angle": [0.006, 0.15, 0.035],
        "mirror_align_random_horizontal": [0.005, 28.0, 0.0, 0.0],
        "mirror_align_random_vertical": [0.005, 28.0, 0.0, 0.0],
    }
    radius = sample_psf_data[psf_opt.RADIUS_CM]

    with patch(
        "simtools.ray_tracing.psf_parameter_optimisation._run_ray_tracing_simulation"
    ) as mock_sim:
        mock_image = MagicMock()
        mock_image.get_cumulative_data.return_value = sample_psf_data
        mock_sim.return_value = (3.5, mock_image)

        d80, rmsd, simulated_data = psf_opt.run_psf_simulation_data_only(
            mock_telescope_model, mock_site_model, mock_args_dict, pars, mock_data_to_plot, radius
        )

        assert d80 == 3.5
        assert rmsd >= 0  # RMSD should be non-negative
        np.testing.assert_array_equal(simulated_data, sample_psf_data)
        mock_image.get_cumulative_data.assert_called_once()


def test_run_psf_simulation_data_only_no_radius(
    mock_telescope_model, mock_site_model, mock_args_dict, mock_data_to_plot
):
    """Test running PSF simulation with no radius data."""
    pars = {
        "mirror_reflection_random_angle": [0.006, 0.15, 0.035],
        "mirror_align_random_horizontal": [0.005, 28.0, 0.0, 0.0],
        "mirror_align_random_vertical": [0.005, 28.0, 0.0, 0.0],
    }

    with patch(
        "simtools.ray_tracing.psf_parameter_optimisation._run_ray_tracing_simulation"
    ) as mock_sim:
        mock_image = MagicMock()
        mock_sim.return_value = (3.5, mock_image)

        with pytest.raises(ValueError, match="Radius data is not available."):
            psf_opt.run_psf_simulation_data_only(
                mock_telescope_model, mock_site_model, mock_args_dict, pars, mock_data_to_plot, None
            )


def test_run_psf_simulation_with_plotting(
    mock_telescope_model, mock_site_model, mock_args_dict, mock_data_to_plot, sample_psf_data
):
    """Test running PSF simulation with plotting enabled."""
    pars = {
        "mirror_reflection_random_angle": [0.006, 0.15, 0.035],
        "mirror_align_random_horizontal": [0.005, 28.0, 0.0, 0.0],
        "mirror_align_random_vertical": [0.005, 28.0, 0.0, 0.0],
    }
    radius = sample_psf_data[psf_opt.RADIUS_CM]

    with (
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation._run_ray_tracing_simulation"
        ) as mock_sim,
        patch("simtools.ray_tracing.psf_parameter_optimisation.visualize.plot_1d") as mock_plot,
        patch("matplotlib.pyplot.clf") as mock_clf,
    ):
        mock_image = MagicMock()
        mock_image.get_cumulative_data.return_value = sample_psf_data
        mock_sim.return_value = (3.5, mock_image)

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_fig.get_axes.return_value = [mock_ax]
        mock_plot.return_value = mock_fig

        mock_pdf_pages = MagicMock()

        d80, rmsd = psf_opt.run_psf_simulation(
            mock_telescope_model,
            mock_site_model,
            mock_args_dict,
            pars,
            mock_data_to_plot,
            radius,
            pdf_pages=mock_pdf_pages,
            is_best=True,
        )

        assert d80 == 3.5
        assert rmsd >= 0

        # Check plotting was called
        mock_plot.assert_called_once()
        mock_ax.set_ylim.assert_called_once_with(0, 1.05)
        mock_ax.set_ylabel.assert_called_once_with(psf_opt.CUMULATIVE_PSF)
        mock_pdf_pages.savefig.assert_called_once()
        mock_clf.assert_called_once()


def test_run_psf_simulation_no_radius(
    mock_telescope_model, mock_site_model, mock_args_dict, mock_data_to_plot
):
    """Test running PSF simulation with no radius data (covers line 239)."""
    pars = {
        "mirror_reflection_random_angle": [0.006, 0.15, 0.035],
        "mirror_align_random_horizontal": [0.005, 28.0, 0.0, 0.0],
        "mirror_align_random_vertical": [0.005, 28.0, 0.0, 0.0],
    }

    with patch(
        "simtools.ray_tracing.psf_parameter_optimisation._run_ray_tracing_simulation"
    ) as mock_sim:
        mock_image = MagicMock()
        mock_sim.return_value = (3.5, mock_image)

        with pytest.raises(ValueError, match="Radius data is not available."):
            psf_opt.run_psf_simulation(
                mock_telescope_model,
                mock_site_model,
                mock_args_dict,
                pars,
                mock_data_to_plot,
                radius=None,
            )


def test_load_and_process_data_with_data(mock_args_dict, sample_psf_data):
    """Test loading and processing data when data file is provided."""
    with (
        patch("simtools.ray_tracing.psf_parameter_optimisation.gen.find_file") as mock_find,
        patch("simtools.ray_tracing.psf_parameter_optimisation.load_psf_data") as mock_load,
    ):
        mock_find.return_value = "found_file.txt"
        mock_load.return_value = sample_psf_data

        data_to_plot, radius = psf_opt.load_and_process_data(mock_args_dict)

        mock_find.assert_called_once_with("test_data.txt", "/path/to/model")
        mock_load.assert_called_once_with("found_file.txt")

        assert "measured" in data_to_plot
        np.testing.assert_array_equal(data_to_plot["measured"], sample_psf_data)
        np.testing.assert_array_equal(radius, sample_psf_data[psf_opt.RADIUS_CM])


def test_load_and_process_data_no_data(mock_args_dict):
    """Test loading and processing data when no data file is provided."""
    mock_args_dict["data"] = None

    data_to_plot, radius = psf_opt.load_and_process_data(mock_args_dict)

    assert isinstance(data_to_plot, OrderedDict)
    assert len(data_to_plot) == 0
    assert radius is None


def test_create_plot_for_parameters(mock_data_to_plot, sample_psf_data):
    """Test creating a plot for a parameter set."""
    pars = {
        "mirror_reflection_random_angle": [0.006, 0.15, 0.035],
        "mirror_align_random_horizontal": [0.005, 28.0, 0.0, 0.0],
        "mirror_align_random_vertical": [0.005, 28.0, 0.0, 0.0],
    }
    rmsd = 0.123
    d80 = 3.5

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
            pars,
            rmsd,
            d80,
            sample_psf_data,
            mock_data_to_plot,
            is_best=True,
            pdf_pages=mock_pdf_pages,
        )

        # Check plotting was called correctly
        mock_plot.assert_called_once()
        mock_ax.set_ylim.assert_called_once_with(0, 1.05)
        mock_ax.set_ylabel.assert_called_once_with(psf_opt.CUMULATIVE_PSF)
        mock_ax.set_title.assert_called_once()
        mock_ax.text.assert_called_once()
        mock_fig.text.assert_called_once()  # Footnote for best parameters
        mock_pdf_pages.savefig.assert_called_once()
        mock_clf.assert_called_once()


def test_create_plot_for_parameters_with_original_data_restoration(
    mock_data_to_plot, sample_psf_data
):
    """Test creating a plot with original simulated data restoration (covers lines 401-402)."""
    pars = {
        "mirror_reflection_random_angle": [0.006, 0.15, 0.035],
        "mirror_align_random_horizontal": [0.005, 28.0, 0.0, 0.0],
        "mirror_align_random_vertical": [0.005, 28.0, 0.0, 0.0],
    }
    rmsd = 0.123
    d80 = 3.5

    # Set up original simulated data
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
            pars,
            rmsd,
            d80,
            sample_psf_data,
            mock_data_to_plot,
            is_best=False,
            pdf_pages=mock_pdf_pages,
        )

        # Check that original simulated data was restored
        assert mock_data_to_plot["simulated"] == original_simulated_data

        # Check plotting was called
        mock_plot.assert_called_once()
        mock_pdf_pages.savefig.assert_called_once()
        mock_clf.assert_called_once()


def test_run_all_simulations(
    mock_telescope_model, mock_site_model, mock_args_dict, mock_data_to_plot, sample_psf_data
):
    """Test running all simulations and collecting results."""
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

    with patch(
        "simtools.ray_tracing.psf_parameter_optimisation.run_psf_simulation_data_only"
    ) as mock_sim:
        # First simulation has higher RMSD, second has lower RMSD (better)
        mock_sim.side_effect = [
            (3.5, 0.2, sample_psf_data),  # d80, rmsd, simulated_data
            (3.2, 0.1, sample_psf_data),
        ]

        best_pars, best_d80, best_rmsd, results = psf_opt._run_all_simulations(
            all_parameters,
            mock_telescope_model,
            mock_site_model,
            mock_args_dict,
            mock_data_to_plot,
            radius,
        )

        assert best_pars == all_parameters[1]  # Second parameter set is better
        assert best_d80 == 3.2
        assert best_rmsd == 0.1
        assert len(results) == 2

        # Check that run_psf_simulation_data_only was called for each parameter set
        assert mock_sim.call_count == 2


def test_run_all_simulations_with_failures(
    mock_telescope_model,
    mock_site_model,
    mock_args_dict,
    mock_data_to_plot,
    sample_psf_data,
    caplog,
):
    """Test running simulations with some failures."""
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

    with patch(
        "simtools.ray_tracing.psf_parameter_optimisation.run_psf_simulation_data_only"
    ) as mock_sim:
        # First simulation fails, second succeeds
        mock_sim.side_effect = [
            ValueError("Simulation failed"),
            (3.2, 0.1, sample_psf_data),
        ]

        with caplog.at_level(logging.WARNING):
            best_pars, best_d80, best_rmsd, results = psf_opt._run_all_simulations(
                all_parameters,
                mock_telescope_model,
                mock_site_model,
                mock_args_dict,
                mock_data_to_plot,
                radius,
            )

        assert best_pars == all_parameters[1]  # Only successful parameter set
        assert best_d80 == 3.2
        assert best_rmsd == 0.1
        assert len(results) == 1  # Only one successful result

        # Check warning was logged for failed simulation
        assert "Simulation failed for parameters" in caplog.text


def test_create_all_plots(mock_data_to_plot):
    """Test creating plots for all parameter sets."""
    results = [
        ({"param1": "value1"}, 0.2, 3.5, {"data": "sim1"}),
        ({"param2": "value2"}, 0.1, 3.2, {"data": "sim2"}),
    ]
    best_pars = {"param2": "value2"}

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


def test_find_best_parameters(
    mock_telescope_model, mock_site_model, mock_args_dict, mock_data_to_plot, sample_psf_data
):
    """Test finding the best parameters."""
    all_parameters = [
        {
            "mirror_reflection_random_angle": [0.005, 0.15, 0.035],
            "mirror_align_random_horizontal": [0.004, 28.0, 0.0, 0.0],
            "mirror_align_random_vertical": [0.004, 28.0, 0.0, 0.0],
        },
    ]
    radius = sample_psf_data[psf_opt.RADIUS_CM]

    with (
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation._run_all_simulations"
        ) as mock_run_all,
        patch("simtools.ray_tracing.psf_parameter_optimisation._create_all_plots") as mock_plots,
    ):
        # Mock the simulation results
        results = [(all_parameters[0], 0.1, 3.2, sample_psf_data)]
        mock_run_all.return_value = (all_parameters[0], 3.2, 0.1, results)

        mock_pdf_pages = MagicMock()

        best_pars, best_d80, returned_results = psf_opt.find_best_parameters(
            all_parameters,
            mock_telescope_model,
            mock_site_model,
            mock_args_dict,
            mock_data_to_plot,
            radius,
            pdf_pages=mock_pdf_pages,
        )

        assert best_pars == all_parameters[0]
        assert best_d80 == 3.2
        assert returned_results == results

        # Check that _run_all_simulations was called
        mock_run_all.assert_called_once()

        # Check that plots were created (since plot_all=True in mock_args_dict)
        mock_plots.assert_called_once()


def test_find_best_parameters_no_plotting(
    mock_telescope_model, mock_site_model, mock_args_dict, mock_data_to_plot, sample_psf_data
):
    """Test finding best parameters without plotting."""
    all_parameters = [
        {
            "mirror_reflection_random_angle": [0.005, 0.15, 0.035],
            "mirror_align_random_horizontal": [0.004, 28.0, 0.0, 0.0],
            "mirror_align_random_vertical": [0.004, 28.0, 0.0, 0.0],
        },
    ]
    radius = sample_psf_data[psf_opt.RADIUS_CM]
    mock_args_dict["plot_all"] = False

    with (
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation._run_all_simulations"
        ) as mock_run_all,
        patch("simtools.ray_tracing.psf_parameter_optimisation._create_all_plots") as mock_plots,
    ):
        results = [(all_parameters[0], 0.1, 3.2, sample_psf_data)]
        mock_run_all.return_value = (all_parameters[0], 3.2, 0.1, results)

        best_pars, best_d80, returned_results = psf_opt.find_best_parameters(
            all_parameters,
            mock_telescope_model,
            mock_site_model,
            mock_args_dict,
            mock_data_to_plot,
            radius,
            pdf_pages=None,
        )

        assert best_pars == all_parameters[0]
        assert best_d80 == 3.2
        assert returned_results == results

        # Check that plots were NOT created
        mock_plots.assert_not_called()
