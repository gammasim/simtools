#!/usr/bin/python3

import json
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from astropy.table import Table

import simtools.ray_tracing.mirror_panel_psf as mpp


def _make_dummy_tel(label="orig", config_dir=None):
    config_file_path = None
    config_file_directory = None
    if config_dir is not None:
        config_file_path = str(config_dir / "orig.cfg")
        config_file_directory = str(config_dir)
    return SimpleNamespace(
        label=label,
        _config_file_path=config_file_path,
        _config_file_directory=config_file_directory,
        overwrite_model_parameter=MagicMock(name="overwrite_model_parameter"),
        overwrite_model_file=MagicMock(name="overwrite_model_file"),
        get_parameter_value=MagicMock(
            name="get_parameter_value", return_value=[0.0075, 0.22, 0.022]
        ),
    )


def _make_minimal_instance(
    label="base",
    args_dict=None,
    telescope_model=None,
) -> mpp.MirrorPanelPSF:
    inst = mpp.MirrorPanelPSF.__new__(mpp.MirrorPanelPSF)
    inst._logger = logging.getLogger(__name__)
    inst.label = label
    inst.args_dict = args_dict or {}
    inst.telescope_model = telescope_model or _make_dummy_tel(label="orig")
    inst.site_model = object()
    inst.measured_data = [10.0]
    inst.rnda_start = [0.0075, 0.22, 0.022]
    inst.rnda_opt = None
    inst.per_mirror_results = []
    inst.final_percentage_diff = None
    return inst


def _default_rnda_settings(**overrides):
    params = {
        "threshold": 0.05,
        "learning_rate": 1e-4,
        "grad_clip": mpp.MirrorPanelPSF.DEFAULT_RNDA_GRAD_CLIP,
        "max_log_step": mpp.MirrorPanelPSF.DEFAULT_RNDA_MAX_LOG_STEP,
        "sigma1": mpp.Bounds(min=1e-4, max=0.1),
        "sigma2": mpp.Bounds(min=1e-4, max=0.1),
        "frac2": mpp.Bounds(min=0.0, max=1.0),
        "max_frac_step": mpp.MirrorPanelPSF.DEFAULT_RNDA_MAX_FRAC_STEP,
        "max_iterations": mpp.MirrorPanelPSF.DEFAULT_RNDA_MAX_ITERATIONS,
    }
    params.update(overrides)
    return mpp.RndaGradientDescentSettings(**params)


def test_load_measured_data_reads_ecsv_and_validates_columns(mocker):
    inst = _make_minimal_instance(args_dict={"data": "data.ecsv", "model_path": "."})
    mocker.patch.object(mpp.gen, "find_file", return_value="/abs/data.ecsv")
    mocker.patch.object(
        mpp.Table,
        "read",
        return_value=Table({"d80": [10.0, 11.0]}),
    )

    table = inst._load_measured_data()
    assert len(table) == 2
    assert list(table) == pytest.approx([10.0, 11.0])


def test_load_measured_data_prefers_psf_opt(mocker):
    inst = _make_minimal_instance(args_dict={"data": "data.ecsv", "model_path": "."})
    mocker.patch.object(mpp.gen, "find_file", return_value="/abs/data.ecsv")
    mocker.patch.object(
        mpp.Table,
        "read",
        return_value=Table({"psf_opt": [9.0, 8.0]}),
    )

    col = inst._load_measured_data()
    assert list(col) == pytest.approx([9.0, 8.0])


def test_load_measured_data_raises_when_columns_missing(mocker):
    inst = _make_minimal_instance(args_dict={"data": "data.ecsv", "model_path": "."})
    mocker.patch.object(mpp.gen, "find_file", return_value="/abs/data.ecsv")
    mocker.patch.object(mpp.Table, "read", return_value=Table({"x": [10.0]}))

    with pytest.raises(ValueError, match="either 'psf_opt' or 'd80'"):
        inst._load_measured_data()


def test_calculate_percentage_difference_raises_on_nonpositive_measured_d80():
    inst = _make_minimal_instance()
    with pytest.raises(ValueError, match="Measured d80 must be positive"):
        inst._calculate_percentage_difference(0.0, 1.0)
    with pytest.raises(ValueError, match="Measured d80 must be positive"):
        inst._calculate_percentage_difference(-1.0, 1.0)


def test_optimize_with_gradient_descent_limits_mirrors_in_test_mode(mocker):
    inst = _make_minimal_instance(args_dict={"test": True, "number_of_mirrors_to_test": 2})
    inst.measured_data = [10.0, 11.0, 12.0]
    mocker.patch("os.cpu_count", return_value=4)

    def _fake_parallel(n_mirrors, n_workers):
        assert n_mirrors == 2
        assert n_workers == 4
        return [
            {"optimized_rnda": [0.01, 0.2, 0.03], "percentage_diff": 10.0},
            {"optimized_rnda": [0.03, 0.2, 0.01], "percentage_diff": 30.0},
        ]

    mocker.patch.object(inst, "_optimize_mirrors_parallel", side_effect=_fake_parallel)
    inst.optimize_with_gradient_descent()


def test_worker_optimize_mirror_forked_uses_measured_data_and_calls_optimizer():
    dummy = SimpleNamespace(
        measured_data=[11.5, 12.25],
        optimize_single_mirror=MagicMock(return_value={"ok": True}),
    )
    result = mpp._worker_optimize_mirror_forked((1, dummy))
    dummy.optimize_single_mirror.assert_called_once_with(1, 12.25)
    assert result == {"ok": True}


def test_simulate_single_mirror_d80_success_and_restores_label(mocker, tmp_path):
    tel = _make_dummy_tel(label="orig", config_dir=tmp_path)
    inst = _make_minimal_instance(label="base", telescope_model=tel)

    class OkRay:
        def __init__(self, **kwargs):
            assert kwargs["telescope_model"].label == "base_m1"
            # Ensure MirrorPanelPSF resets cached config path/directory when relabeling.
            assert kwargs["telescope_model"]._config_file_path is None
            assert kwargs["telescope_model"]._config_file_directory is None
            self._d80_cm = 1.5

        def simulate(self, **kwargs):
            return None

        def analyze(self, **kwargs):
            return None

        def get_d80_mm(self):
            return self._d80_cm * 10.0

    mocker.patch("simtools.ray_tracing.mirror_panel_psf.RayTracing", OkRay)
    d80_mm = inst._simulate_single_mirror_d80(0, [0.01, 0.22, 0.022])
    assert d80_mm == pytest.approx(15.0)
    assert tel.label == "orig"
    assert tel._config_file_path == str(tmp_path / "orig.cfg")
    assert tel._config_file_directory == str(tmp_path)


def test_run_rnda_gradient_descent_converges_immediately(mocker):
    inst = _make_minimal_instance()
    mocker.patch.object(inst, "_simulate_single_mirror_d80", return_value=12.0)
    settings = _default_rnda_settings(max_iterations=10)
    best_rnda, best_sim_d80, best_pct_diff = inst._run_rnda_gradient_descent(
        mirror_idx=0,
        measured_d80_mm=12.0,
        current_rnda=[0.01, 0.22, 0.022],
        settings=settings,
    )
    assert best_rnda == [0.01, 0.22, 0.022]
    assert best_sim_d80 == pytest.approx(12.0)
    assert best_pct_diff == pytest.approx(0.0)


def test_run_rnda_gradient_descent_stops_when_learning_rate_too_small(mocker):
    inst = _make_minimal_instance()
    # One iteration evaluates 8 simulations:
    # initial + 2 (sigma1) + 2 (frac2) + 2 (sigma2) + new.
    mock_sim = mocker.patch.object(
        inst,
        "_simulate_single_mirror_d80",
        side_effect=[20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
    )
    settings = _default_rnda_settings(learning_rate=1e-13, threshold=0.0, max_iterations=200)
    best_rnda, best_sim_d80, best_pct_diff = inst._run_rnda_gradient_descent(
        mirror_idx=0,
        measured_d80_mm=10.0,
        current_rnda=[0.01, 0.22, 0.022],
        settings=settings,
    )
    assert best_rnda == [0.01, 0.22, 0.022]
    assert best_sim_d80 == pytest.approx(20.0)
    assert best_pct_diff == pytest.approx(100.0)
    assert mock_sim.call_count == 8


def test_finite_difference_objective_gradient_returns_zero_when_denom_nonpositive(mocker):
    inst = _make_minimal_instance()
    eval_mock = mocker.patch.object(inst, "_evaluate_rnda_candidate")
    grad = inst._finite_difference_objective_gradient(
        mirror_idx=0,
        measured_d80_mm=10.0,
        current_rnda=[0.01, 0.22, 0.022],
        param_index=0,
        plus_value=0.1,
        minus_value=0.1,
    )
    assert grad == pytest.approx(0.0)
    eval_mock.assert_not_called()


def test_init_sets_test_mirror_limit_in_test_mode(mocker):
    dummy_tel = _make_dummy_tel(label="tel")
    dummy_tel.get_parameter_value = MagicMock(return_value=[0.0075, 0.22, 0.022])
    mocker.patch.object(
        mpp, "initialize_simulation_models", return_value=(dummy_tel, object(), None)
    )
    mocker.patch.object(
        mpp.MirrorPanelPSF,
        "_load_measured_data",
        return_value=[10.0],
    )

    args = {
        "test": True,
        "data": "data.ecsv",
        "site": "North",
        "telescope": "LSTN-01",
        "model_version": "1.0",
    }
    inst = mpp.MirrorPanelPSF("lbl", args)
    assert inst.args_dict["number_of_mirrors_to_test"] == 10


def test_optimize_single_mirror_uses_rnda_optimizer(mocker):
    inst = _make_minimal_instance(args_dict={"threshold": 0.05, "learning_rate": 1e-4})
    mock_run = mocker.patch.object(
        inst,
        "_run_rnda_gradient_descent",
        return_value=([0.003, 0.25, 0.010], 12.0, 2.0),
    )

    result = inst.optimize_single_mirror(0, 12.0)
    assert result["optimized_rnda"] == [0.003, 0.25, 0.010]
    assert result["percentage_diff"] == pytest.approx(2.0)
    assert mock_run.call_count == 1


def test_get_allowed_range_from_schema_returns_none_for_invalid_schema(monkeypatch):
    inst = _make_minimal_instance(args_dict={})

    def _fake_model_parameters():
        return {
            "mirror_reflection_random_angle": {
                "data": [
                    {"allowed_range": None},
                    {"allowed_range": "not-a-dict"},
                ]
            }
        }

    monkeypatch.setattr(mpp.names, "model_parameters", _fake_model_parameters)
    assert inst._get_allowed_range_from_schema("mirror_reflection_random_angle", 0) == (None, None)
    assert inst._get_allowed_range_from_schema("mirror_reflection_random_angle", 1) == (None, None)
    assert inst._get_allowed_range_from_schema("mirror_reflection_random_angle", 2) == (None, None)


def test_run_rnda_gradient_descent_accepts_improvement_and_converges(mocker):
    inst = _make_minimal_instance()
    # The first iteration evaluates:
    # initial + (sigma1 +/-) + (frac2 +/-) + (sigma2 +/-) + new = 8 simulations.
    mocker.patch.object(
        inst,
        "_simulate_single_mirror_d80",
        side_effect=[20.0, 19.0, 21.0, 18.0, 22.0, 17.0, 23.0, 10.2],
    )

    settings = _default_rnda_settings()

    best_rnda, best_sim_d80, best_pct_diff = inst._run_rnda_gradient_descent(
        mirror_idx=0,
        measured_d80_mm=10.0,
        current_rnda=[0.01, 0.22, 0.022],
        settings=settings,
    )

    assert best_sim_d80 == pytest.approx(10.2)
    assert best_pct_diff == pytest.approx(2.0)
    assert best_rnda != [0.01, 0.22, 0.022]


def test_optimize_mirrors_parallel_collects_results_and_returns_list(monkeypatch):
    inst = _make_minimal_instance(args_dict={"parallel": True})
    parent_stub = object()
    monkeypatch.setattr(mpp, "MirrorPanelPSF", MagicMock(return_value=parent_stub))
    ppm = MagicMock(
        return_value=[
            {"mirror": 1, "optimized_rnda": [0.0, 0.0, 0.0], "percentage_diff": 0.0},
            {"mirror": 2, "optimized_rnda": [1.0, 0.0, 0.0], "percentage_diff": 0.0},
            {"mirror": 3, "optimized_rnda": [2.0, 0.0, 0.0], "percentage_diff": 0.0},
        ]
    )
    monkeypatch.setattr(mpp, "process_pool_map_ordered", ppm)

    results = inst._optimize_mirrors_parallel(n_mirrors=3, n_workers=2)
    assert [r["mirror"] for r in results] == [1, 2, 3]

    mpp.MirrorPanelPSF.assert_called_once()
    kwargs = mpp.MirrorPanelPSF.call_args.kwargs
    assert kwargs["label"] == inst.label
    assert kwargs["args_dict"]["parallel"] is False

    ppm.assert_called_once()
    ppm_kwargs = ppm.call_args.kwargs
    assert ppm.call_args.args[0] is mpp._worker_optimize_mirror_forked
    assert ppm_kwargs["max_workers"] == 2
    assert ppm_kwargs["mp_start_method"] == "fork"


def test_write_optimization_data_writes_json(tmp_path):
    inst = _make_minimal_instance(
        args_dict={"output_path": str(tmp_path), "telescope": "LSTN-01", "model_version": "6.0.0"}
    )
    inst.rnda_opt = [0.002, 0.22, 0.022]
    inst.final_percentage_diff = 12.34
    inst.per_mirror_results = [
        {
            "mirror": 1,
            "measured_d80_mm": 10.0,
            "optimized_rnda": [0.002, 0.22, 0.022],
            "simulated_d80_mm": 11.0,
            "percentage_diff": 10.0,
        }
    ]
    inst.write_optimization_data()
    out_file = tmp_path / "LSTN-01" / "per_mirror_rnda.json"
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert data["telescope"] == "LSTN-01"
    assert data["model_version"] == "6.0.0"
    assert len(data["per_mirror_results"]) == 1


def test_init_calls_initialize_simulation_models(monkeypatch, mocker):
    dummy_tel = _make_dummy_tel(label="tel")
    dummy_site = object()
    monkeypatch.setattr(
        mpp, "initialize_simulation_models", lambda **kw: (dummy_tel, dummy_site, None)
    )
    mocker.patch.object(mpp.MirrorPanelPSF, "_load_measured_data", return_value=[10.0])

    args = {
        "data": "data.ecsv",
        "site": "North",
        "telescope": "LSTN-01",
        "model_version": "6.0.0",
    }
    inst = mpp.MirrorPanelPSF("lbl", args)
    assert inst.telescope_model is dummy_tel
    assert inst.site_model is dummy_site


def test_write_optimization_data_warns_when_parameter_export_fails(tmp_path, mocker, caplog):
    inst = _make_minimal_instance(
        args_dict={
            "output_path": str(tmp_path),
            "telescope": "LSTN-01",
            "model_version": "6.0.0",
            "parameter_version": "v1",
        }
    )
    inst.rnda_opt = [0.002, 0.22, 0.022]
    inst.per_mirror_results = []

    dump = mocker.patch(
        "simtools.data_model.model_data_writer.ModelDataWriter.dump_model_parameter",
        side_effect=OSError("boom"),
    )

    with caplog.at_level(logging.WARNING):
        inst.write_optimization_data()

    assert dump.call_count == 1
    assert "Failed to export model parameter" in caplog.text


def test_write_optimization_data_also_exports_model_parameter_json(tmp_path, mocker):
    inst = _make_minimal_instance(
        args_dict={
            "output_path": str(tmp_path),
            "telescope": "LSTN-01",
            "model_version": "6.0.0",
            "parameter_version": "v1",
        }
    )
    inst.rnda_opt = [0.002, 0.22, 0.022]
    inst.final_percentage_diff = 12.34
    inst.per_mirror_results = [
        {
            "mirror": 1,
            "measured_d80_mm": 10.0,
            "optimized_rnda": [0.002, 0.22, 0.022],
            "simulated_d80_mm": 11.0,
            "percentage_diff": 10.0,
        }
    ]

    dump = mocker.patch(
        "simtools.data_model.model_data_writer.ModelDataWriter.dump_model_parameter"
    )
    inst.write_optimization_data()

    dump.assert_called_once()
    kwargs = dump.call_args.kwargs
    assert kwargs["parameter_name"] == "mirror_reflection_random_angle"
    assert kwargs["value"] == pytest.approx([0.002, 0.22, 0.022])
    assert kwargs["unit"] == ["deg", "dimensionless", "deg"]
    assert kwargs["instrument"] == "LSTN-01"
    assert kwargs["parameter_version"] == "v1"
    assert kwargs["output_file"] == "mirror_reflection_random_angle-v1.json"


def test_write_d80_histogram_all_branches(tmp_path):
    inst = _make_minimal_instance(args_dict={"output_path": str(tmp_path)})

    assert inst.write_d80_histogram() is None

    inst.args_dict["d80_hist"] = "hist.png"
    inst.per_mirror_results = []
    assert inst.write_d80_histogram() is None

    inst.per_mirror_results = [
        {"measured_d80_mm": 10.0, "simulated_d80_mm": 12.0},
        {"measured_d80_mm": 11.0, "simulated_d80_mm": 13.0},
    ]
    out_path = inst.write_d80_histogram()
    assert out_path is not None
    assert (tmp_path / "hist.png").exists()
