#!/usr/bin/python3

import copy
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from simtools.ray_tracing.mirror_panel_psf import MirrorPanelPSF


def _create_sample_psf_data():
    """Create sample PSF data for testing."""
    radius = np.linspace(0, 10, 21)
    cumulative = np.linspace(0, 1, 21)
    dtype = {"names": ("Radius [cm]", "Cumulative PSF"), "formats": ("f8", "f8")}
    data = np.empty(21, dtype=dtype)
    data["Radius [cm]"] = radius
    data["Cumulative PSF"] = cumulative
    return data


@pytest.fixture
def mock_args_dict(tmp_test_directory):
    return {
        "test": False,
        "data": "tests/resources/PSFcurve_data_v2.ecsv",
        "site": "North",
        "telescope": "LSTN-01",
        "model_version": "test_version",
        "mirror_list": None,
        "random_focal_length": None,
        "simtel_path": "path/to/simtel",
        "number_of_mirrors_to_test": 2,
        "use_random_focal_length": False,
        "output_path": tmp_test_directory,
        "threshold": 0.03,
        "learning_rate": 0.00001,
        "model_path": "tests/resources",
        "random_focal_length_seed": None,
        "parameter_version": "1.0.0",
    }


@pytest.fixture
def mock_telescope_model_string():
    return "simtools.ray_tracing.mirror_panel_psf.initialize_simulation_models"


@pytest.fixture
def mock_find_file_string():
    return "simtools.ray_tracing.mirror_panel_psf.gen.find_file"


@pytest.fixture
def mock_run_simulations_and_analysis_string():
    return "simtools.ray_tracing.mirror_panel_psf.MirrorPanelPSF.run_simulations_and_analysis"


@pytest.fixture
def dummy_tel():
    class DummyTel:
        def __init__(self):
            self.overwrite_model_parameter = MagicMock(name="overwrite_model_parameter")
            self.overwrite_model_file = MagicMock(name="overwrite_model_file")
            self.get_parameter_value = MagicMock(
                name="get_parameter_value", return_value=[0.0075, 0.22, 0.022]
            )

    return DummyTel()


@pytest.fixture
def mock_mirror_panel_psf(
    mock_args_dict, mock_telescope_model_string, mock_find_file_string, dummy_tel
):
    with patch(mock_telescope_model_string) as mock_init_models, patch(mock_find_file_string):
        mock_init_models.return_value = (dummy_tel, "dummy_site", None)
        db_config = {"db": "config"}
        label = "test_label"
        mirror_panel_psf = MirrorPanelPSF(label, mock_args_dict, db_config)
        yield mirror_panel_psf


@pytest.fixture
def mock_gradient_descent_setup():
    """Factory fixture for common gradient descent setup patches."""

    def _create_patches(best_params=None, final_metric=0.020, gd_results=None, sample_data=None):
        """Create mock patches for gradient descent optimization tests.

        Parameters
        ----------
        best_params : dict, optional
            Best parameters from optimization
        final_metric : float, optional
            Final RMSD metric
        gd_results : list, optional
            List of gradient descent results
        sample_data : np.ndarray, optional
            Sample PSF data

        Returns
        -------
        tuple
            (best_params, final_psf_diameter, gd_results)
        """
        if sample_data is None:
            sample_data = _create_sample_psf_data()

        if best_params is None:
            best_params = {"mirror_reflection_random_angle": [0.008, 0.18, 0.025]}

        if gd_results is None:
            gd_results = [(best_params, final_metric, None, 3.4, sample_data)]

        return best_params, 3.4, gd_results

    return _create_patches


def _setup_mirror_panel_psf_for_gd_test(
    mock_load, mock_optimizer_class, best_params, final_psf, gd_results, sample_data
):
    """Helper to set up common mocks for gradient descent tests."""
    mock_load.return_value = ({"measured": sample_data}, sample_data["Radius [cm]"])

    mock_optimizer = MagicMock()
    mock_optimizer_class.return_value = mock_optimizer
    mock_optimizer.run_gradient_descent.return_value = (best_params, final_psf, gd_results)

    return mock_optimizer


def test_define_telescope_model(
    mock_args_dict, mock_telescope_model_string, mock_find_file_string, dummy_tel
):
    args_dict = copy.deepcopy(mock_args_dict)
    # no mirror list, no random focal length
    with (
        patch(mock_telescope_model_string) as mock_init_models,
        patch(mock_find_file_string) as mock_find_file,
    ):
        mock_init_models.return_value = (dummy_tel, "dummy_site", None)

        args_dict["mirror_list"] = None
        args_dict["random_focal_length"] = None
        db_config = {"db": "config"}
        label = "test_label"

        mirror_panel_psf = MirrorPanelPSF(label, args_dict, db_config)
        tel = mirror_panel_psf.telescope_model

        mock_init_models.assert_called_once_with(
            label=label,
            db_config=db_config,
            site=args_dict["site"],
            telescope_name=args_dict["telescope"],
            model_version=args_dict["model_version"],
        )
        mock_find_file.assert_not_called()
        tel.overwrite_model_parameter.assert_not_called()
        tel.overwrite_model_file.assert_not_called()

    # mirror list and random focal length
    with (
        patch(mock_telescope_model_string) as mock_init_models,
        patch(mock_find_file_string) as mock_find_file,
    ):
        mock_init_models.return_value = (dummy_tel, "dummy_site", None)

        args_dict["mirror_list"] = "mirror_list_CTA-N-LST1_v2019-03-31_rotated.ecsv"
        args_dict["model_path"] = "tests/resources"
        args_dict["random_focal_length"] = 0.1
        db_config = {"db": "config"}
        label = "test_label"

        mirror_panel_psf = MirrorPanelPSF(label, args_dict, db_config)
        tel = mirror_panel_psf.telescope_model

        mock_init_models.assert_called_once_with(
            label=label,
            db_config=db_config,
            site=args_dict["site"],
            telescope_name=args_dict["telescope"],
            model_version=args_dict["model_version"],
        )
        mock_find_file.assert_called_once()
        assert tel.overwrite_model_parameter.call_count == 2
        tel.overwrite_model_file.assert_called_once()


def test_init_with_test_mode(mock_args_dict, mock_telescope_model_string, dummy_tel):
    """Test initialization in test mode sets number_of_mirrors_to_test to 10."""
    args_dict = copy.deepcopy(mock_args_dict)
    args_dict["test"] = True

    with patch(mock_telescope_model_string) as mock_init_models:
        mock_init_models.return_value = (dummy_tel, "dummy_site", None)
        db_config = {"db": "config"}
        label = "test_label"

        mirror_panel_psf = MirrorPanelPSF(label, args_dict, db_config)

        assert mirror_panel_psf.args_dict["number_of_mirrors_to_test"] == 10
        assert mirror_panel_psf.rnda_start == [0.0075, 0.22, 0.022]
        assert mirror_panel_psf.rnda_opt is None
        assert mirror_panel_psf.gd_optimizer is None
        assert mirror_panel_psf.final_rmsd is None


def test_optimize_with_gradient_descent_success(mock_mirror_panel_psf, mock_gradient_descent_setup):
    """Test successful gradient descent optimization."""
    mirror_psf = copy.deepcopy(mock_mirror_panel_psf)

    sample_data = _create_sample_psf_data()
    mock_best_params, final_psf, mock_gd_results = mock_gradient_descent_setup(
        sample_data=sample_data
    )
    mock_gd_results = [
        (mock_best_params, 0.025, None, 3.5, sample_data),
        (mock_best_params, 0.020, None, 3.4, sample_data),
    ]

    with (
        patch("simtools.ray_tracing.mirror_panel_psf.psf_opt.load_and_process_data") as mock_load,
        patch(
            "simtools.ray_tracing.mirror_panel_psf.psf_opt.PSFParameterOptimizer"
        ) as mock_optimizer_class,
        patch(
            "simtools.ray_tracing.mirror_panel_psf.plot_psf.create_final_psf_comparison_plot"
        ) as mock_plot,
    ):
        mock_optimizer = _setup_mirror_panel_psf_for_gd_test(
            mock_load,
            mock_optimizer_class,
            mock_best_params,
            final_psf,
            mock_gd_results,
            sample_data,
        )

        mirror_psf.optimize_with_gradient_descent()

        # Verify optimizer was created with correct parameters
        mock_optimizer_class.assert_called_once()
        call_args = mock_optimizer_class.call_args
        assert call_args[1]["optimize_only"] == ["mirror_reflection_random_angle"]

        # Verify gradient descent was run
        mock_optimizer.run_gradient_descent.assert_called_once()
        call_args = mock_optimizer.run_gradient_descent.call_args
        assert call_args[1]["rmsd_threshold"] == mirror_psf.args_dict["threshold"]
        assert call_args[1]["learning_rate"] == mirror_psf.args_dict["learning_rate"]
        assert call_args[1]["epsilon"] == pytest.approx(0.00005)

        # Verify results were stored
        assert mirror_psf.rnda_opt == [0.008, 0.18, 0.025]
        assert mirror_psf.final_rmsd == pytest.approx(0.020)

        # Verify plot was created
        mock_plot.assert_called_once()


def test_optimize_with_gradient_descent_failure(mock_mirror_panel_psf):
    """Test gradient descent optimization when it fails to find parameters."""
    mirror_psf = copy.deepcopy(mock_mirror_panel_psf)

    sample_data = _create_sample_psf_data()

    with (
        patch("simtools.ray_tracing.mirror_panel_psf.psf_opt.load_and_process_data") as mock_load,
        patch(
            "simtools.ray_tracing.mirror_panel_psf.psf_opt.PSFParameterOptimizer"
        ) as mock_optimizer_class,
    ):
        mock_load.return_value = ({"measured": sample_data}, sample_data["Radius [cm]"])

        mock_optimizer = MagicMock()
        mock_optimizer_class.return_value = mock_optimizer
        mock_optimizer.run_gradient_descent.return_value = (None, None, [])

        with pytest.raises(ValueError, match="Gradient descent optimization failed"):
            mirror_psf.optimize_with_gradient_descent()


def test_optimize_with_gradient_descent_test_mode(
    mock_mirror_panel_psf, mock_gradient_descent_setup
):
    """Test gradient descent uses correct mirror numbers in test mode."""
    mirror_psf = copy.deepcopy(mock_mirror_panel_psf)
    mirror_psf.args_dict["test"] = True
    mirror_psf.args_dict["number_of_mirrors_to_test"] = 10

    sample_data = _create_sample_psf_data()
    mock_best_params, final_psf, mock_gd_results = mock_gradient_descent_setup(
        sample_data=sample_data
    )

    with (
        patch("simtools.ray_tracing.mirror_panel_psf.psf_opt.load_and_process_data") as mock_load,
        patch(
            "simtools.ray_tracing.mirror_panel_psf.psf_opt.PSFParameterOptimizer"
        ) as mock_optimizer_class,
        patch("simtools.ray_tracing.mirror_panel_psf.plot_psf.create_final_psf_comparison_plot"),
    ):
        _setup_mirror_panel_psf_for_gd_test(
            mock_load,
            mock_optimizer_class,
            mock_best_params,
            final_psf,
            mock_gd_results,
            sample_data,
        )

        mirror_psf.optimize_with_gradient_descent()

        # Check that optimizer_args has correct mirror_numbers
        call_args = mock_optimizer_class.call_args
        optimizer_args = call_args[1]["args_dict"]
        assert optimizer_args["single_mirror_mode"] is True
        assert optimizer_args["mirror_numbers"] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def test_optimize_with_gradient_descent_production_mode(
    mock_mirror_panel_psf, mock_gradient_descent_setup
):
    """Test gradient descent uses 'all' mirrors in production mode."""
    mirror_psf = copy.deepcopy(mock_mirror_panel_psf)
    mirror_psf.args_dict["test"] = False

    sample_data = _create_sample_psf_data()
    mock_best_params, final_psf, mock_gd_results = mock_gradient_descent_setup(
        sample_data=sample_data
    )

    with (
        patch("simtools.ray_tracing.mirror_panel_psf.psf_opt.load_and_process_data") as mock_load,
        patch(
            "simtools.ray_tracing.mirror_panel_psf.psf_opt.PSFParameterOptimizer"
        ) as mock_optimizer_class,
        patch("simtools.ray_tracing.mirror_panel_psf.plot_psf.create_final_psf_comparison_plot"),
    ):
        _setup_mirror_panel_psf_for_gd_test(
            mock_load,
            mock_optimizer_class,
            mock_best_params,
            final_psf,
            mock_gd_results,
            sample_data,
        )

        mirror_psf.optimize_with_gradient_descent()

        # Check that optimizer_args has 'all' mirrors
        call_args = mock_optimizer_class.call_args
        optimizer_args = call_args[1]["args_dict"]
        assert optimizer_args["mirror_numbers"] == "all"


def test_optimize_with_gradient_descent_with_random_focal_length(
    mock_mirror_panel_psf, mock_gradient_descent_setup
):
    """Test gradient descent with random focal length settings."""
    mirror_psf = copy.deepcopy(mock_mirror_panel_psf)
    mirror_psf.args_dict["use_random_focal_length"] = True
    mirror_psf.args_dict["random_focal_length_seed"] = 42

    sample_data = _create_sample_psf_data()
    mock_best_params, final_psf, mock_gd_results = mock_gradient_descent_setup(
        sample_data=sample_data
    )

    with (
        patch("simtools.ray_tracing.mirror_panel_psf.psf_opt.load_and_process_data") as mock_load,
        patch(
            "simtools.ray_tracing.mirror_panel_psf.psf_opt.PSFParameterOptimizer"
        ) as mock_optimizer_class,
        patch("simtools.ray_tracing.mirror_panel_psf.plot_psf.create_final_psf_comparison_plot"),
    ):
        _setup_mirror_panel_psf_for_gd_test(
            mock_load,
            mock_optimizer_class,
            mock_best_params,
            final_psf,
            mock_gd_results,
            sample_data,
        )

        mirror_psf.optimize_with_gradient_descent()

        # Check random focal length settings
        call_args = mock_optimizer_class.call_args
        optimizer_args = call_args[1]["args_dict"]
        assert optimizer_args["use_random_focal_length"] is True
        assert optimizer_args["random_focal_length_seed"] == 42


def test_optimize_with_gradient_descent_no_simulated_data(
    mock_mirror_panel_psf, mock_gradient_descent_setup
):
    """Test gradient descent when final result has no simulated data."""
    mirror_psf = copy.deepcopy(mock_mirror_panel_psf)

    sample_data = _create_sample_psf_data()
    mock_best_params, final_psf, _ = mock_gradient_descent_setup(sample_data=sample_data)
    # Last result has None for simulated data
    mock_gd_results = [(mock_best_params, 0.020, None, 3.4, None)]

    with (
        patch("simtools.ray_tracing.mirror_panel_psf.psf_opt.load_and_process_data") as mock_load,
        patch(
            "simtools.ray_tracing.mirror_panel_psf.psf_opt.PSFParameterOptimizer"
        ) as mock_optimizer_class,
        patch(
            "simtools.ray_tracing.mirror_panel_psf.plot_psf.create_final_psf_comparison_plot"
        ) as mock_plot,
    ):
        _setup_mirror_panel_psf_for_gd_test(
            mock_load,
            mock_optimizer_class,
            mock_best_params,
            final_psf,
            mock_gd_results,
            sample_data,
        )

        mirror_psf.optimize_with_gradient_descent()

        # Verify plot was NOT created when simulated_data is None
        mock_plot.assert_not_called()


def test_write_optimization_data(mock_mirror_panel_psf):
    """Test writing optimization data to file."""
    mirror_psf = copy.deepcopy(mock_mirror_panel_psf)
    mirror_psf.rnda_opt = [0.008, 0.18, 0.025]

    with patch(
        "simtools.ray_tracing.mirror_panel_psf.psf_opt.export_psf_parameters"
    ) as mock_export:
        mirror_psf.write_optimization_data()

        mock_export.assert_called_once()
        call_args = mock_export.call_args
        assert call_args[1]["best_pars"] == {"mirror_reflection_random_angle": [0.008, 0.18, 0.025]}
        assert call_args[1]["telescope"] == "LSTN-01"
        assert call_args[1]["parameter_version"] == "1.0.0"
        assert isinstance(call_args[1]["output_dir"], Path)


def test_write_optimization_data_default_version(mock_mirror_panel_psf):
    """Test writing optimization data with default version when not provided."""
    mirror_psf = copy.deepcopy(mock_mirror_panel_psf)
    mirror_psf.rnda_opt = [0.008, 0.18, 0.025]
    mirror_psf.args_dict["parameter_version"] = None

    with patch(
        "simtools.ray_tracing.mirror_panel_psf.psf_opt.export_psf_parameters"
    ) as mock_export:
        mirror_psf.write_optimization_data()

        call_args = mock_export.call_args
        assert call_args[1]["parameter_version"] == "0.0.0"


def test_print_results(mock_mirror_panel_psf, capsys):
    """Test printing optimization results to stdout."""
    mirror_psf = copy.deepcopy(mock_mirror_panel_psf)
    mirror_psf.rnda_start = [0.0075, 0.22, 0.022]
    mirror_psf.rnda_opt = [0.008, 0.18, 0.025]
    mirror_psf.final_rmsd = 0.0234

    mirror_psf.print_results()
    out = capsys.readouterr().out

    assert "Optimization Results (RMSD-based)" in out
    assert "RMSD (full PSF curve): 0.023400" in out
    assert "mirror_reflection_random_angle [sigma1, fraction2, sigma2]" in out
    assert "Previous values" in out
    assert "0.007500" in out
    assert "Optimized values" in out
    assert "0.008000" in out


def test_print_results_no_rmsd(mock_mirror_panel_psf, capsys):
    """Test printing results when final_rmsd is not set."""
    mirror_psf = copy.deepcopy(mock_mirror_panel_psf)
    mirror_psf.rnda_start = [0.0075, 0.22, 0.022]
    mirror_psf.rnda_opt = [0.008, 0.18, 0.025]
    mirror_psf.final_rmsd = None

    mirror_psf.print_results()
    out = capsys.readouterr().out

    assert "Optimization Results (RMSD-based)" in out
    assert "RMSD (full PSF curve)" not in out
    assert "mirror_reflection_random_angle" in out


def test_run_simulations_and_analysis(
    mock_telescope_model_string, mock_find_file_string, dummy_tel
):
    """Test running ray tracing simulations for single mirror mode."""
    rnda = [0.008, 0.18, 0.025]
    args_dict = {
        "test": False,
        "data": "tests/resources/PSFcurve_data_v2.ecsv",
        "site": "North",
        "telescope": "LSTN-01",
        "model_version": "test_version",
        "mirror_list": None,
        "random_focal_length": None,
        "simtel_path": "path/to/simtel",
        "number_of_mirrors_to_test": 5,
        "use_random_focal_length": False,
        "output_path": "",
        "threshold": 0.03,
        "learning_rate": 0.00001,
        "model_path": "tests/resources",
        "random_focal_length_seed": None,
        "parameter_version": "1.0.0",
    }

    with (
        patch(mock_telescope_model_string) as mock_init_models,
        patch(mock_find_file_string),
        patch("simtools.ray_tracing.mirror_panel_psf.RayTracing") as mock_ray_tracing,
    ):
        mock_init_models.return_value = (dummy_tel, "dummy_site", None)
        db_config = {"db": "config"}
        label = "test_label"
        mirror_panel_psf = MirrorPanelPSF(label, args_dict, db_config)

        mock_ray_instance = mock_ray_tracing.return_value

        mirror_panel_psf.run_simulations_and_analysis(rnda)

        # Verify telescope parameter was updated
        dummy_tel.overwrite_model_parameter.assert_called_with(
            "mirror_reflection_random_angle", rnda
        )

        # Verify RayTracing was initialized with correct parameters
        mock_ray_tracing.assert_called_once()
        call_kwargs = mock_ray_tracing.call_args[1]
        assert call_kwargs["single_mirror_mode"] is True
        assert call_kwargs["mirror_numbers"] == "all"
        assert call_kwargs["use_random_focal_length"] is False

        # Verify simulate and analyze were called
        mock_ray_instance.simulate.assert_called_once_with(test=False, force=True)
        mock_ray_instance.analyze.assert_called_once_with(force=True)


def test_run_simulations_and_analysis_test_mode(
    mock_telescope_model_string, mock_find_file_string, dummy_tel
):
    """Test running simulations in test mode with limited mirrors."""
    rnda = [0.008, 0.18, 0.025]
    args_dict = {
        "test": True,
        "data": "tests/resources/PSFcurve_data_v2.ecsv",
        "site": "North",
        "telescope": "LSTN-01",
        "model_version": "test_version",
        "mirror_list": None,
        "random_focal_length": None,
        "simtel_path": "path/to/simtel",
        "number_of_mirrors_to_test": 10,
        "use_random_focal_length": False,
        "output_path": "",
        "threshold": 0.03,
        "learning_rate": 0.00001,
        "model_path": "tests/resources",
        "random_focal_length_seed": None,
        "parameter_version": "1.0.0",
    }

    with (
        patch(mock_telescope_model_string) as mock_init_models,
        patch(mock_find_file_string),
        patch("simtools.ray_tracing.mirror_panel_psf.RayTracing") as mock_ray_tracing,
    ):
        mock_init_models.return_value = (dummy_tel, "dummy_site", None)
        db_config = {"db": "config"}
        label = "test_label"
        mirror_panel_psf = MirrorPanelPSF(label, args_dict, db_config)

        mock_ray_instance = mock_ray_tracing.return_value

        mirror_panel_psf.run_simulations_and_analysis(rnda)

        # Verify mirror_numbers for test mode
        call_kwargs = mock_ray_tracing.call_args[1]
        assert call_kwargs["mirror_numbers"] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        # Verify simulate was called with test=True
        mock_ray_instance.simulate.assert_called_once_with(test=True, force=True)


def test_run_simulations_and_analysis_with_random_focal_length(
    mock_telescope_model_string, mock_find_file_string, dummy_tel
):
    """Test running simulations with random focal length enabled."""
    rnda = [0.008, 0.18, 0.025]
    args_dict = {
        "test": False,
        "data": "tests/resources/PSFcurve_data_v2.ecsv",
        "site": "North",
        "telescope": "LSTN-01",
        "model_version": "test_version",
        "mirror_list": None,
        "random_focal_length": None,
        "simtel_path": "path/to/simtel",
        "number_of_mirrors_to_test": 5,
        "use_random_focal_length": True,
        "random_focal_length_seed": 123,
        "output_path": "",
        "threshold": 0.03,
        "learning_rate": 0.00001,
        "model_path": "tests/resources",
        "parameter_version": "1.0.0",
    }

    with (
        patch(mock_telescope_model_string) as mock_init_models,
        patch(mock_find_file_string),
        patch("simtools.ray_tracing.mirror_panel_psf.RayTracing") as mock_ray_tracing,
    ):
        mock_init_models.return_value = (dummy_tel, "dummy_site", None)
        db_config = {"db": "config"}
        label = "test_label"
        mirror_panel_psf = MirrorPanelPSF(label, args_dict, db_config)

        mirror_panel_psf.run_simulations_and_analysis(rnda)

        # Verify random focal length settings
        call_kwargs = mock_ray_tracing.call_args[1]
        assert call_kwargs["use_random_focal_length"] is True
        assert call_kwargs["random_focal_length_seed"] == 123
