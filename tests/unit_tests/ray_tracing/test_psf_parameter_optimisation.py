#!/usr/bin/python3

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from astropy import units as u

import simtools.ray_tracing.psf_parameter_optimisation as psf_opt

HORIZONTAL = "horizontal"
VERTICAL = "vertical"
BATCH_METHOD = "_run_parameter_batch"
TEST_OUTPUT_DIR = Path("/dummy_test_path")
MIRROR_ALIGN_RANDOM = [0.004, 28.0, 0.0, 0.0]


def make_mirror_align_random_parameters():
    return {
        "mirror_align_random_horizontal": list(MIRROR_ALIGN_RANDOM),
        "mirror_align_random_vertical": list(MIRROR_ALIGN_RANDOM),
    }


def make_step_result(metric, psf_diameter=3.4, step_accepted=True, learning_rate=0.1, data=None):
    return psf_opt.GradientStepResult(
        params=make_mirror_align_random_parameters(),
        psf_diameter=psf_diameter,
        metric=metric,
        p_value=0.85,
        simulated_data=data,
        step_accepted=step_accepted,
        learning_rate=learning_rate,
    )


@pytest.fixture
def mock_telescope_model():
    mock_tel = MagicMock()
    mock_tel.name = "LSTN-01"
    mock_tel.get_parameter_value.side_effect = lambda param: {
        **make_mirror_align_random_parameters(),
        "focal_length": 28.0,
    }.get(param)
    return mock_tel


@pytest.fixture
def mock_args_dict():
    return {
        "data": "test_data.txt",
        "data_search_path": "/path/to/model",
        "ks_statistic": False,
        "learning_rate": 0.1,
        "test": True,
        "plot_all": False,
        "zenith_angle": 20.0 * u.deg,
        "source_distance": 10.0 * u.km,
        "monte_carlo_analysis": False,
        "rmsd_threshold": 0.01,
        "fraction": 0.8,
    }


@pytest.fixture
def mock_site_model():
    return MagicMock()


@pytest.fixture
def sample_data():
    dtype = {"names": (psf_opt.RADIUS, psf_opt.CUMULATIVE_PSF), "formats": ("f8", "f8")}
    data = np.empty(21, dtype=dtype)
    data[psf_opt.RADIUS] = np.linspace(0, 10, 21)
    data[psf_opt.CUMULATIVE_PSF] = np.linspace(0, 1, 21)
    return data


@pytest.fixture
def optimizer(mock_telescope_model, mock_args_dict, sample_data):
    return psf_opt.PSFParameterOptimizer(
        mock_telescope_model,
        MagicMock(),
        mock_args_dict,
        {"measured": sample_data},
        sample_data[psf_opt.RADIUS],
        TEST_OUTPUT_DIR,
    )


@pytest.mark.parametrize(
    ("data_file", "should_raise_error"), [("test_data.txt", False), (None, True)]
)
def test_load_and_process_data(mock_args_dict, data_file, should_raise_error):
    mock_args_dict["data"] = data_file

    if should_raise_error:
        with pytest.raises(FileNotFoundError, match="No data file specified"):
            psf_opt.load_and_process_data(mock_args_dict)
        return

    with (
        patch("simtools.utils.general.find_file", return_value=Path("test.txt")),
        patch("astropy.table.Table.read") as mock_read,
    ):
        mock_table = MagicMock()
        mock_table.__len__.return_value = 3
        mock_table.colnames = ["radius_mm", "integral_psf"]
        mock_radius_col = MagicMock()
        mock_radius_col.to.return_value.value = np.array([1, 2, 3])
        mock_table.__getitem__.side_effect = lambda key: (
            mock_radius_col if "radius" in key else np.array([0.1, 0.2, 0.3])
        )
        mock_read.return_value = mock_table

        data_to_plot, radius = psf_opt.load_and_process_data(mock_args_dict)
        assert "measured" in data_to_plot
        assert len(radius) > 0


@pytest.mark.parametrize(
    ("use_ks_statistic", "plot_all", "radius_none", "expected_behavior"),
    [
        (False, False, False, "basic"),
        (True, False, False, "ks_statistic"),
        (False, True, False, "plotting"),
        (False, False, True, "radius_error"),
    ],
)
@patch("simtools.ray_tracing.psf_parameter_optimisation.RayTracing")
@patch("simtools.visualization.plot_psf.create_psf_parameter_plot")
def test_run_psf_simulation(
    mock_plot,
    mock_ray_class,
    mock_telescope_model,
    mock_args_dict,
    sample_data,
    use_ks_statistic,
    plot_all,
    radius_none,
    expected_behavior,
):
    radius = None if radius_none else sample_data[psf_opt.RADIUS]
    data_to_plot = {"measured": sample_data}
    mock_args_dict["plot_all"] = plot_all
    mock_image = MagicMock()
    mock_image.get_psf.return_value = 3.5
    mock_image.get_cumulative_data.return_value = sample_data
    mock_ray_class.return_value.images.return_value = [mock_image]
    args = (
        mock_telescope_model,
        MagicMock(),
        mock_args_dict,
        make_mirror_align_random_parameters(),
        data_to_plot,
        radius,
    )

    if expected_behavior == "radius_error":
        with pytest.raises(ValueError, match="Radius data is not available"):
            psf_opt.run_psf_simulation(*args)
        return

    if expected_behavior == "plotting":
        psf_opt.run_psf_simulation(*args, pdf_pages=MagicMock())
        mock_plot.assert_called_once()
        return

    psf_diameter, metric, p_value, simulated = psf_opt.run_psf_simulation(
        *args, use_ks_statistic=use_ks_statistic
    )
    assert psf_diameter == pytest.approx(3.5)
    assert simulated is sample_data
    if expected_behavior == "ks_statistic":
        assert p_value is not None
    else:
        assert p_value is None
        assert metric >= 0
    mock_plot.assert_not_called()


@pytest.mark.parametrize("failed", [False, True])
def test_calculate_gradient(optimizer, mocker, failed):
    parameters = {HORIZONTAL: [2.0, 3.0], VERTICAL: 4.0}
    results = [(1.0, value, None, None) for value in [11.0, 12.0, 13.0]]
    if failed:
        results[1] = None
    batch = mocker.patch.object(optimizer, BATCH_METHOD, return_value=results)

    gradients = optimizer.calculate_gradient(parameters, 10.0, epsilon=0.5)

    batch.assert_called_once_with(
        [
            {HORIZONTAL: [2.5, 3.0], VERTICAL: 4.0},
            {HORIZONTAL: [2.0, 3.5], VERTICAL: 4.0},
            {HORIZONTAL: [2.0, 3.0], VERTICAL: 4.5},
        ],
        use_cache=True,
    )
    if failed:
        assert gradients is None
    else:
        assert gradients == {HORIZONTAL: [2.0, 4.0], VERTICAL: 6.0}


def test_perform_gradient_step_comprehensive(optimizer, sample_data):
    current_params = {"dummy_list_param": [0.005, 0.15, 0.03]}
    gradient = {"dummy_list_param": [-0.001, 0.01, -0.002]}
    stepped = {"dummy_list_param": [0.006, 0.15, 0.031]}

    # Gradient is None: step is not accepted
    with patch.object(optimizer, "calculate_gradient", return_value=None):
        result = optimizer.perform_gradient_step_with_retries(current_params, 0.1, 0.1)
        assert result.step_accepted is False
        assert result.learning_rate == pytest.approx(0.1)

    # Worse metric first, then better: learning rate is reduced and step accepted
    with (
        patch.object(optimizer, "calculate_gradient", return_value=gradient),
        patch.object(
            optimizer,
            "apply_gradient_step",
            return_value={"dummy_list_param": [0.006, 0.15, 0.031]},
        ),
        patch.object(
            optimizer,
            "run_simulation",
            side_effect=[
                (3.6, 0.15, 0.75, sample_data),
                (3.4, 0.08, 0.85, sample_data),
            ],
        ),
    ):
        result = optimizer.perform_gradient_step_with_retries(
            current_params, 0.1, 0.1, max_retries=3
        )
        assert result.step_accepted is True
        assert result.learning_rate == pytest.approx(0.07, rel=1e-2)

    # Parameters out of bounds in all retries
    with (
        patch.object(optimizer, "calculate_gradient", return_value=gradient),
        patch.object(
            optimizer, "apply_gradient_step", return_value={"dummy_list_param": [999, 999, 999]}
        ),
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation._are_all_parameters_within_allowed_range",
            return_value=False,
        ),
    ):
        result = optimizer.perform_gradient_step_with_retries(
            current_params, 0.1, 0.1, max_retries=3
        )
        assert result.step_accepted is False

    # Learning rate below 1e-6 is reset
    with (
        patch.object(optimizer, "calculate_gradient", return_value=gradient),
        patch.object(optimizer, "apply_gradient_step", return_value=stepped),
        patch.object(optimizer, "run_simulation"),
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation._are_all_parameters_within_allowed_range",
            side_effect=[False, False, True],
        ),
    ):
        result = optimizer.perform_gradient_step_with_retries(
            current_params, 0.1, 0.0000015, max_retries=3
        )
        assert result.learning_rate == pytest.approx(0.0001)

    # Two rejected steps (worse metric), then accepted with reset learning rate
    with (
        patch.object(optimizer, "calculate_gradient", return_value=gradient),
        patch.object(optimizer, "apply_gradient_step", return_value=stepped),
        patch.object(
            optimizer,
            "run_simulation",
            side_effect=[
                (3.6, 0.15, 0.8, sample_data),
                (3.6, 0.15, 0.8, sample_data),
                (3.4, 0.08, 0.8, sample_data),
            ],
        ),
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation._are_all_parameters_within_allowed_range",
            return_value=True,
        ),
    ):
        result = optimizer.perform_gradient_step_with_retries(
            current_params, 0.1, 0.0000015, max_retries=3
        )
        assert result.step_accepted is True
        assert result.learning_rate == pytest.approx(0.0001)


def test_run_gradient_descent_convergence_and_tracking(optimizer, sample_data):
    def run(step_results, **kwargs):
        with (
            patch.object(
                optimizer,
                "get_initial_parameters",
                return_value=make_mirror_align_random_parameters(),
            ),
            patch.object(optimizer, "run_simulation", return_value=(3.5, 0.1, 0.8, sample_data)),
            patch.object(optimizer, "perform_gradient_step_with_retries", side_effect=step_results),
        ):
            return optimizer.run_gradient_descent(rmsd_threshold=0.01, learning_rate=0.1, **kwargs)

    # Convergence when threshold is reached
    _, _, results = run([make_step_result(metric=0.008, data=sample_data)], max_iterations=10)
    assert len(results) == 2

    # Max iterations reached
    _, _, results = run([make_step_result(metric=0.095, data=sample_data)] * 2, max_iterations=2)
    assert len(results) == 3

    # Best metric is tracked across iterations, not the last one
    _, best_diameter, _ = run(
        [
            make_step_result(metric=0.08, psf_diameter=3.4, data=sample_data),
            make_step_result(metric=0.05, psf_diameter=3.3, data=sample_data),
            make_step_result(metric=0.12, psf_diameter=3.2, data=sample_data),
        ],
        max_iterations=3,
    )
    assert best_diameter == pytest.approx(3.3)

    # A rejected step does not stop the descent
    best_pars, _, _ = run(
        [
            make_step_result(metric=None, step_accepted=False, data=None),
            make_step_result(metric=0.008, data=sample_data),
        ],
        max_iterations=5,
    )
    assert best_pars is not None


def test_analyze_monte_carlo_error(optimizer, mock_telescope_model, mock_args_dict, sample_data):
    # Normal case: all simulations succeed
    with (
        patch.object(
            optimizer, "get_initial_parameters", return_value=make_mirror_align_random_parameters()
        ),
        patch.object(optimizer, BATCH_METHOD, return_value=[(3.5, 0.1, 0.8, sample_data)] * 2),
    ):
        assert len(optimizer.analyze_monte_carlo_error(n_simulations=2)) == 9

    # No data: empty statistics
    optimizer_no_data = psf_opt.PSFParameterOptimizer(
        mock_telescope_model, MagicMock(), mock_args_dict, None, None, TEST_OUTPUT_DIR
    )
    result = optimizer_no_data.analyze_monte_carlo_error()
    assert result[0] is None
    assert result[1] is None
    assert result[2] == []

    # All simulations fail
    with (
        patch.object(
            optimizer, "get_initial_parameters", return_value=make_mirror_align_random_parameters()
        ),
        patch.object(optimizer, BATCH_METHOD, return_value=[None, None]),
    ):
        result = optimizer.analyze_monte_carlo_error(n_simulations=2)
        assert result[0] is None
        assert result[1] is None
        assert result[2] == []

    # KS statistic requested
    mock_args_dict["ks_statistic"] = True
    optimizer_ks = psf_opt.PSFParameterOptimizer(
        mock_telescope_model,
        MagicMock(),
        mock_args_dict,
        {"measured": sample_data},
        sample_data[psf_opt.RADIUS],
        TEST_OUTPUT_DIR,
    )
    with (
        patch.object(
            optimizer_ks,
            "get_initial_parameters",
            return_value=make_mirror_align_random_parameters(),
        ),
        patch.object(optimizer_ks, BATCH_METHOD, return_value=[(3.5, 0.1, 0.8, sample_data)] * 2),
    ):
        result = optimizer_ks.analyze_monte_carlo_error(n_simulations=2)
        assert len(result) == 9
        assert result[3] is not None


def test_run_simulation_with_caching_and_ks_override(
    optimizer, mock_telescope_model, mock_args_dict, sample_data
):
    params = make_mirror_align_random_parameters()

    with patch(
        "simtools.ray_tracing.psf_parameter_optimisation.run_psf_simulation",
        return_value=(3.5, 0.1, 0.8, sample_data),
    ):
        optimizer.run_simulation(params, use_cache=True)
        assert optimizer.cache_misses == 1
        optimizer.run_simulation(params, use_cache=True)
        assert optimizer.cache_hits == 1
        # cache bypass with pdf_pages, is_best, or use_cache=False
        optimizer.run_simulation(params, pdf_pages=MagicMock(), use_cache=True)
        optimizer.run_simulation(params, is_best=True, use_cache=True)
        optimizer.run_simulation(params, use_cache=False)

    mock_args_dict["ks_statistic"] = False
    optimizer2 = psf_opt.PSFParameterOptimizer(
        mock_telescope_model,
        MagicMock(),
        mock_args_dict,
        {"measured": sample_data},
        sample_data[psf_opt.RADIUS],
        TEST_OUTPUT_DIR,
    )
    with patch(
        "simtools.ray_tracing.psf_parameter_optimisation.run_psf_simulation",
        return_value=(3.5, 0.1, 0.8, sample_data),
    ) as mock_sim:
        optimizer2.run_simulation(params, use_ks_statistic=True)
        assert mock_sim.call_args[0][8] is True


def test_workflow_with_all_features(
    mock_telescope_model, mock_site_model, mock_args_dict, sample_data, tmp_test_directory
):
    mock_args_dict.update(
        {
            "write_psf_parameters": True,
            "parameter_version": "1.0.0",
        }
    )
    best_params = make_mirror_align_random_parameters()
    with (
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation.load_and_process_data",
            return_value=({"measured": sample_data}, sample_data[psf_opt.RADIUS]),
        ),
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation.PSFParameterOptimizer"
        ) as mock_opt_cls,
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation.create_optimization_plots"
        ) as mock_opt_plot,
        patch(
            "simtools.visualization.plot_psf.create_gradient_descent_convergence_plot"
        ) as mock_conv_plot,
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation.write_gradient_descent_log",
            return_value=Path(tmp_test_directory) / "log.txt",
        ) as mock_write_log,
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation.create_psf_vs_offaxis_plot"
        ) as mock_offaxis_plot,
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation.export_psf_parameters"
        ) as mock_export,
    ):
        mock_optimizer = mock_opt_cls.return_value
        mock_optimizer.run_gradient_descent.return_value = (
            best_params,
            3.5,
            [(best_params, 0.05, 0.8, 3.5, sample_data)],
        )
        mock_optimizer.use_ks_statistic = False
        mock_optimizer.fraction = 0.8

        psf_opt.run_psf_optimization_workflow(
            mock_telescope_model, mock_site_model, mock_args_dict, Path(tmp_test_directory)
        )

        mock_opt_plot.assert_called_once()
        mock_conv_plot.assert_called_once()
        mock_write_log.assert_called_once()
        mock_offaxis_plot.assert_called_once()
        mock_export.assert_called_once()


def test_run_psf_optimization_job(mocker, tmp_test_directory, run_context):
    args = {
        "site": "North",
        "telescope": "LSTN-01",
        "model_version": "6.0.0",
        "label": "psf",
        "monte_carlo_analysis": False,
    }
    telescope, site = object(), object()
    initialize = mocker.patch.object(
        psf_opt, "initialize_simulation_models", return_value=(telescope, site, None)
    )
    workflow = mocker.patch.object(psf_opt, "run_psf_optimization_workflow")

    job = psf_opt.PSFOptimizationJob(args, tmp_test_directory, run_context)
    psf_opt.run_psf_optimization_job(job)

    initialize.assert_called_once_with(
        label="psf",
        site="North",
        telescope_name="LSTN-01",
        model_version="6.0.0",
        context=run_context,
    )
    workflow.assert_called_once_with(telescope, site, args, tmp_test_directory, run_context)


def test_calculate_rmsd_and_ks_statistic():
    data = np.array([1, 2, 3, 4])
    sim = np.array([1.1, 2.1, 3.1, 4.1])
    assert psf_opt.calculate_rmsd(data, sim) == pytest.approx(0.1, rel=1e-3)
    ks_stat, p_value = psf_opt.calculate_ks_statistic(data, sim)
    assert isinstance(ks_stat, float)
    assert 0.0 <= p_value <= 1.0


def test_export_psf_parameters(mock_telescope_model, tmp_test_directory):
    parameters = make_mirror_align_random_parameters()
    with (
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation._add_units_to_psf_parameters",
            return_value=parameters,
        ) as mock_units,
        patch("simtools.ray_tracing.psf_parameter_optimisation.writer") as mock_writer,
    ):
        psf_opt.export_psf_parameters(
            parameters, mock_telescope_model.name, "1.0.0", Path(tmp_test_directory)
        )
        mock_units.assert_called_once_with(parameters)
        assert mock_writer.ModelDataWriter.write_model_parameter.call_count == len(parameters)
        for _, kwargs in mock_writer.ModelDataWriter.write_model_parameter.call_args_list:
            assert kwargs["instrument"] == mock_telescope_model.name
            assert kwargs["parameter_version"] == "1.0.0"

    with (
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation._add_units_to_psf_parameters",
            return_value=parameters,
        ),
        patch("simtools.ray_tracing.psf_parameter_optimisation.writer") as mock_writer,
        patch("simtools.ray_tracing.psf_parameter_optimisation.logger") as mock_logger,
    ):
        mock_writer.ModelDataWriter.write_model_parameter.side_effect = ValueError("Test error")
        psf_opt.export_psf_parameters(
            parameters, mock_telescope_model.name, "1.0.0", Path(tmp_test_directory)
        )
        mock_logger.error.assert_called_once_with(
            "Error exporting simulation parameters: Test error"
        )


def test_apply_gradient_step(optimizer):
    current_params = {"dummy_list_param": [0.005, 0.15, 0.03]}
    gradients = {"dummy_list_param": [-0.001, 0.01, -0.002]}
    new_params = optimizer.apply_gradient_step(current_params, gradients, 0.1)
    assert new_params["dummy_list_param"][0] != current_params["dummy_list_param"][0]

    new_params = optimizer.apply_gradient_step(
        {"camera_filter_relative_efficiency": 1.0},
        {"camera_filter_relative_efficiency": 0.01},
        0.1,
    )
    assert new_params["camera_filter_relative_efficiency"] == pytest.approx(0.999)

    # zenith angle (second element) of mirror_align parameters is preserved
    for name in ("mirror_align_random_horizontal", "mirror_align_random_vertical"):
        new_params = optimizer.apply_gradient_step(
            {name: list(MIRROR_ALIGN_RANDOM)},
            {name: [-0.001, -0.02, -0.003, 0.001]},
            0.1,
        )
        assert new_params[name][1] == pytest.approx(28.0)


def test_create_step_plot(optimizer, sample_data):
    current_params = make_mirror_align_random_parameters()
    pages = MagicMock()
    optimizer.args_dict["plot_all"] = True
    with patch("simtools.visualization.plot_psf.create_psf_parameter_plot") as mock_plot:
        optimizer._create_step_plot(pages, current_params, 3.5, 0.1, 0.8, sample_data)
        mock_plot.assert_called_once_with(
            optimizer.data_to_plot,
            current_params,
            3.5,
            0.1,
            False,
            pages,
            fraction=0.8,
            p_value=0.8,
            use_ks_statistic=False,
        )


@pytest.mark.parametrize("case", ["no_pages", "plot_all_off", "no_data"])
def test_create_step_plot_early_return(optimizer, sample_data, case):
    current_params = make_mirror_align_random_parameters()
    pages, data = MagicMock(), sample_data
    if case == "no_pages":
        pages = None
    elif case == "plot_all_off":
        optimizer.args_dict["plot_all"] = False
    else:
        data = None
    with patch("simtools.visualization.plot_psf.create_psf_parameter_plot") as mock_plot:
        assert optimizer._create_step_plot(pages, current_params, 3.5, 0.1, 0.8, data) is None
        mock_plot.assert_not_called()


def test_create_final_plot(optimizer, sample_data):
    best_params = make_mirror_align_random_parameters()
    data_to_plot = optimizer.data_to_plot
    with (
        patch("simtools.visualization.plot_psf.create_psf_parameter_plot") as mock_plot,
        patch.object(optimizer, "run_simulation", return_value=(3.5, 0.08, 0.9, sample_data)),
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation.calculate_rmsd", return_value=0.05
        ) as mock_rmsd,
    ):
        pages = MagicMock()
        optimizer._create_final_plot(pages, best_params, 3.5)

        optimizer.run_simulation.assert_called_once_with(
            best_params, pdf_pages=None, is_best=False, use_cache=False, use_ks_statistic=True
        )
        np.testing.assert_array_equal(
            mock_rmsd.call_args[0][0], data_to_plot["measured"][psf_opt.CUMULATIVE_PSF]
        )
        mock_plot.assert_called_once_with(
            data_to_plot,
            best_params,
            3.5,
            0.05,
            True,
            pages,
            fraction=0.8,
            p_value=0.9,
            use_ks_statistic=False,
            second_metric=0.08,
        )

    assert optimizer._create_final_plot(None, best_params, 3.5) is None
    assert optimizer._create_final_plot(MagicMock(), None, 3.5) is None


def test_run_gradient_descent_with_no_data(optimizer):
    optimizer.data_to_plot = None
    optimizer.radius = None
    best_pars, best_psf_diameter, gd_results = optimizer.run_gradient_descent(
        rmsd_threshold=0.1, learning_rate=0.1
    )
    assert best_pars is None
    assert best_psf_diameter is None
    assert gd_results == []


def test_are_all_parameters_within_allowed_range():
    assert (
        psf_opt._are_all_parameters_within_allowed_range(
            {"mirror_align_random_horizontal": list(MIRROR_ALIGN_RANDOM)}
        )
        is True
    )
    assert (
        isinstance(
            psf_opt._are_all_parameters_within_allowed_range(
                {"camera_filter_relative_efficiency": 100.0}
            ),
            bool,
        )
        is True
    )
    with patch(
        "simtools.ray_tracing.psf_parameter_optimisation._is_parameter_within_allowed_range",
        return_value=False,
    ):
        assert (
            psf_opt._are_all_parameters_within_allowed_range({"dummy_list_param": [0.005, 999.0]})
            is False
        )
        assert (
            psf_opt._are_all_parameters_within_allowed_range(
                {"camera_filter_relative_efficiency": 999.0}
            )
            is False
        )


def test_create_perturbed_params():
    current_params = {
        "mirror_align_random_horizontal": list(MIRROR_ALIGN_RANDOM),
        "camera_filter_relative_efficiency": 1.0,
    }
    perturbed = psf_opt._create_perturbed_params(
        current_params, "mirror_align_random_horizontal", list(MIRROR_ALIGN_RANDOM), 0, 0.004, 0.001
    )
    assert perturbed["mirror_align_random_horizontal"][0] == pytest.approx(0.005)
    assert perturbed["mirror_align_random_horizontal"][1] == pytest.approx(28.0)

    perturbed = psf_opt._create_perturbed_params(
        current_params, "camera_filter_relative_efficiency", 1.0, 0, 1.0, 0.01
    )
    assert perturbed["camera_filter_relative_efficiency"] == pytest.approx(1.01)


def test_workflow_no_optimization_result(
    mock_telescope_model, mock_site_model, mock_args_dict, sample_data, tmp_test_directory
):
    for load_result in ((None, None), ({"measured": sample_data}, sample_data[psf_opt.RADIUS])):
        with (
            patch(
                "simtools.ray_tracing.psf_parameter_optimisation.load_and_process_data",
                return_value=load_result,
            ),
            patch(
                "simtools.ray_tracing.psf_parameter_optimisation.PSFParameterOptimizer"
            ) as mock_opt_cls,
        ):
            mock_opt_cls.return_value.run_gradient_descent.return_value = (None, None, [])
            psf_opt.run_psf_optimization_workflow(
                mock_telescope_model, mock_site_model, mock_args_dict, Path(tmp_test_directory)
            )


@pytest.mark.parametrize(
    ("mc_result", "expect_write"),
    [
        ((0.1, 0.01, [0.09, 0.11], None, None, [], 3.5, 0.05, [3.4, 3.6]), True),
        ((None, None, [], None, None, [], None, None, []), False),
    ],
)
def test_workflow_monte_carlo_analysis(
    mock_telescope_model,
    mock_site_model,
    mock_args_dict,
    sample_data,
    tmp_test_directory,
    mc_result,
    expect_write,
):
    mock_args_dict["monte_carlo_analysis"] = True
    with (
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation.load_and_process_data",
            return_value=({"measured": sample_data}, sample_data[psf_opt.RADIUS]),
        ),
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation.PSFParameterOptimizer"
        ) as mock_opt_cls,
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation.write_monte_carlo_analysis"
        ) as mock_write,
        patch("simtools.visualization.plot_psf.create_monte_carlo_uncertainty_plot"),
    ):
        mock_opt_cls.return_value.analyze_monte_carlo_error.return_value = mc_result
        psf_opt.run_psf_optimization_workflow(
            mock_telescope_model, mock_site_model, mock_args_dict, Path(tmp_test_directory)
        )
        assert mock_write.called is expect_write


def test_workflow_max_iterations_passed(
    mock_telescope_model, mock_site_model, mock_args_dict, tmp_test_directory
):
    mock_args_dict["max_iterations"] = 42
    with (
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation.load_and_process_data",
            return_value=(None, None),
        ),
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation.PSFParameterOptimizer"
        ) as mock_opt_cls,
    ):
        mock_optimizer = mock_opt_cls.return_value
        mock_optimizer.run_gradient_descent.return_value = (None, None, [])
        psf_opt.run_psf_optimization_workflow(
            mock_telescope_model, mock_site_model, mock_args_dict, Path(tmp_test_directory)
        )
        mock_optimizer.run_gradient_descent.assert_called_once_with(
            mock_args_dict["rmsd_threshold"],
            mock_args_dict["learning_rate"],
            max_iterations=42,
        )


def test_cleanup_intermediate_files(
    tmp_test_directory, mock_telescope_model, mock_site_model, mock_args_dict, sample_data
):
    work_dir = Path(tmp_test_directory)
    mock_args_dict["cleanup"] = True
    (work_dir / "test.log").write_text("log content")
    (work_dir / "sim.lis").write_text("lis content")
    (work_dir / "sim.lis.gz").write_bytes(b"lis.gz content")
    (work_dir / "keep_me.png").write_bytes(b"png content")
    (work_dir / "keep_me.pdf").write_bytes(b"pdf content")

    with (
        patch(
            "simtools.ray_tracing.psf_parameter_optimisation.load_and_process_data",
            return_value=({"measured": sample_data}, sample_data[psf_opt.RADIUS]),
        ),
        patch("simtools.ray_tracing.psf_parameter_optimisation.PSFParameterOptimizer") as mock_opt,
        patch("simtools.ray_tracing.psf_parameter_optimisation.create_optimization_plots"),
        patch("simtools.visualization.plot_psf.create_gradient_descent_convergence_plot"),
        patch("simtools.ray_tracing.psf_parameter_optimisation.write_gradient_descent_log"),
        patch("simtools.ray_tracing.psf_parameter_optimisation.create_psf_vs_offaxis_plot"),
    ):
        mock_opt.return_value.run_gradient_descent.return_value = (
            make_mirror_align_random_parameters(),
            3.5,
            [(make_mirror_align_random_parameters(), 0.05, 0.8, 3.5, sample_data)],
        )
        psf_opt.run_psf_optimization_workflow(
            mock_telescope_model, mock_site_model, mock_args_dict, work_dir
        )

    assert not (work_dir / "test.log").exists()
    assert not (work_dir / "sim.lis").exists()
    assert not (work_dir / "sim.lis.gz").exists()
    assert (work_dir / "keep_me.png").exists()
    assert (work_dir / "keep_me.pdf").exists()
