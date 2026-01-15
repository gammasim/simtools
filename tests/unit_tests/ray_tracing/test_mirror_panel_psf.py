#!/usr/bin/python3

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from astropy.table import Table

import simtools.ray_tracing.mirror_panel_psf as mpp


def _make_dummy_tel(*, label: str = "orig"):
    return SimpleNamespace(
        label=label,
        _config_file_path="/tmp/orig.cfg",
        _config_file_directory="/tmp",
        overwrite_model_parameter=MagicMock(name="overwrite_model_parameter"),
        overwrite_model_file=MagicMock(name="overwrite_model_file"),
        get_parameter_value=MagicMock(
            name="get_parameter_value", return_value=[0.0075, 0.22, 0.022]
        ),
    )


def _make_minimal_instance(
    *, label: str = "base", args_dict: dict | None = None
) -> mpp.MirrorPanelPSF:
    inst = mpp.MirrorPanelPSF.__new__(mpp.MirrorPanelPSF)
    inst._logger = logging.getLogger(__name__)
    inst.label = label
    inst.args_dict = args_dict or {}
    inst.telescope_model = _make_dummy_tel(label="orig")
    inst.site_model = object()
    inst.measured_data = Table({"d80": [10.0], "focal_length": [28.0]})
    inst.rnda_start = [0.0075, 0.22, 0.022]
    inst.rnda_opt = None
    inst.per_mirror_results = []
    inst.final_percentage_diff = None
    return inst


def _default_rnda_settings(**overrides):
    params = {
        "threshold": 0.05,
        "learning_rate": 1e-4,
        "grad_clip": 1e4,
        "max_log_step": 0.25,
        "sigma1_min": 1e-4,
        "sigma1_max": 0.1,
        "sigma2_min": 1e-4,
        "sigma2_max": 0.1,
        "frac2_min": 0.0,
        "frac2_max": 1.0,
        "max_frac_step": 0.1,
        "max_iterations": 10,
    }
    params.update(overrides)
    return mpp.RndaGradientDescentSettings(**params)


def test_load_measured_data_reads_ecsv_and_validates_columns(mocker):
    inst = _make_minimal_instance(args_dict={"data": "data.ecsv", "model_path": "."})
    mocker.patch.object(mpp.gen, "find_file", return_value="/abs/data.ecsv")
    mocker.patch.object(
        mpp.Table,
        "read",
        return_value=Table({"d80": [10.0, 11.0], "focal_length": [28.0, 28.0]}),
    )

    table = inst._load_measured_data()
    assert len(table) == 2
    assert "d80" in table.colnames
    assert "focal_length" in table.colnames


def test_load_measured_data_raises_when_columns_missing(mocker):
    inst = _make_minimal_instance(args_dict={"data": "data.ecsv", "model_path": "."})
    mocker.patch.object(mpp.gen, "find_file", return_value="/abs/data.ecsv")
    mocker.patch.object(mpp.Table, "read", return_value=Table({"d80": [10.0]}))

    with pytest.raises(ValueError, match="must contain 'd80' and 'focal_length'"):
        inst._load_measured_data()


def test_optimize_with_gradient_descent_uses_cpu_count_when_n_workers_zero(mocker):
    inst = _make_minimal_instance(args_dict={"n_workers": 0})
    inst.measured_data = Table({"d80": [10.0, 11.0], "focal_length": [28.0, 28.0]})

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

    assert inst.rnda_opt == pytest.approx([0.02, 0.2, 0.02])
    assert inst.final_percentage_diff == pytest.approx(20.0)


def test_optimize_with_gradient_descent_limits_mirrors_in_test_mode(mocker):
    inst = _make_minimal_instance(args_dict={"test": True, "number_of_mirrors_to_test": 2})
    inst.measured_data = Table({"d80": [10.0, 11.0, 12.0], "focal_length": [28.0, 28.0, 28.0]})
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


def test_worker_optimize_mirror_requires_init(monkeypatch):
    monkeypatch.setattr(mpp, "_WORKER_INSTANCE", None)
    with pytest.raises(RuntimeError, match="Worker not initialized"):
        mpp._worker_optimize_mirror(0)


def test_worker_init_loads_config_and_creates_instance(monkeypatch):
    created = object()
    import simtools.settings as settings

    mock_load = MagicMock()
    monkeypatch.setattr(settings.config, "load", mock_load)
    monkeypatch.setattr(mpp, "MirrorPanelPSF", MagicMock(return_value=created))
    monkeypatch.setattr(mpp, "_WORKER_INSTANCE", None)

    args = {"x": 1}
    db_config = {"db": "cfg"}
    mpp._worker_init("lbl", args, db_config)
    mock_load.assert_called_once_with(args=args, db_config=db_config)
    assert mpp._WORKER_INSTANCE is created


def test_worker_optimize_mirror_uses_measured_data_and_calls_optimizer(monkeypatch):
    dummy = SimpleNamespace(
        measured_data=Table({"d80": [11.5, 12.25], "focal_length": [27.0, 28.5]}),
        _optimize_single_mirror=MagicMock(return_value={"ok": True}),
    )
    monkeypatch.setattr(mpp, "_WORKER_INSTANCE", dummy)
    result = mpp._worker_optimize_mirror(1)
    dummy._optimize_single_mirror.assert_called_once_with(1, 12.25, 28.5)
    assert result == {"ok": True}


def test_simulate_single_mirror_d80_success_and_restores_label(mocker):
    inst = _make_minimal_instance(label="base")
    tel = inst.telescope_model

    class OkRay:
        def __init__(self, **kwargs):
            assert kwargs["telescope_model"].label == "base_m1"
            # Ensure MirrorPanelPSF resets cached config path/directory when relabeling.
            assert kwargs["telescope_model"]._config_file_path is None
            assert kwargs["telescope_model"]._config_file_directory is None
            self._results = {"d80_cm": [SimpleNamespace(value=1.5)]}

        def simulate(self, **kwargs):
            return None

        def analyze(self, **kwargs):
            return None

    mocker.patch("simtools.ray_tracing.mirror_panel_psf.RayTracing", OkRay)
    d80_mm = inst._simulate_single_mirror_d80(0, 28.0, [0.01, 0.22, 0.022])
    assert d80_mm == pytest.approx(15.0)
    assert tel.label == "orig"
    assert tel._config_file_path == "/tmp/orig.cfg"
    assert tel._config_file_directory == "/tmp"


def test_run_rnda_gradient_descent_converges_immediately(mocker):
    inst = _make_minimal_instance()
    mocker.patch.object(inst, "_simulate_single_mirror_d80", return_value=12.0)
    settings = _default_rnda_settings(max_iterations=10)
    best_rnda, best_sim_d80, best_pct_diff = inst._run_rnda_gradient_descent(
        mirror_idx=0,
        measured_d80_mm=12.0,
        focal_length_m=28.0,
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
        focal_length_m=28.0,
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
        focal_length_m=28.0,
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
        mpp.MirrorPanelPSF, "_define_telescope_model", return_value=(dummy_tel, object())
    )
    mocker.patch.object(
        mpp.MirrorPanelPSF,
        "_load_measured_data",
        return_value=Table({"d80": [10.0], "focal_length": [28.0]}),
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


def test_optimize_single_mirror_returns_expected_dict(mocker):
    inst = _make_minimal_instance(args_dict={"threshold": 0.05, "learning_rate": 1e-4})
    mocker.patch.object(
        inst,
        "_run_rnda_gradient_descent",
        return_value=([0.002, 0.23, 0.020], 15.0, 5.0),
    )
    result = inst._optimize_single_mirror(0, 14.0, 28.0)
    assert result["mirror"] == 1
    assert result["measured_d80_mm"] == pytest.approx(14.0)
    assert result["focal_length_m"] == pytest.approx(28.0)
    assert result["optimized_rnda"] == [0.002, 0.23, 0.020]
    assert result["simulated_d80_mm"] == pytest.approx(15.0)
    assert result["percentage_diff"] == pytest.approx(5.0)


def test_optimize_single_mirror_uses_rnda_optimizer(mocker):
    inst = _make_minimal_instance(args_dict={"threshold": 0.05, "learning_rate": 1e-4})
    mock_run = mocker.patch.object(
        inst,
        "_run_rnda_gradient_descent",
        return_value=([0.003, 0.25, 0.010], 12.0, 2.0),
    )

    result = inst._optimize_single_mirror(0, 12.0, 28.0)
    assert result["optimized_rnda"] == [0.003, 0.25, 0.010]
    assert result["percentage_diff"] == pytest.approx(2.0)
    assert mock_run.call_count == 1


def test_optimize_single_mirror_schema_fallback_branches(mocker, monkeypatch):
    inst = _make_minimal_instance(args_dict={"threshold": 0.05, "learning_rate": 1e-4})

    # Exercise:
    # - allowed_range not a dict (idx 0 and 1)
    # - idx out of range (idx 2)
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
    mocker.patch.object(
        inst,
        "_run_rnda_gradient_descent",
        return_value=([0.002, 0.23, 0.020], 15.0, 5.0),
    )

    result = inst._optimize_single_mirror(0, 14.0, 28.0)
    assert result["mirror"] == 1


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
        focal_length_m=28.0,
        current_rnda=[0.01, 0.22, 0.022],
        settings=settings,
    )

    assert best_sim_d80 == pytest.approx(10.2)
    assert best_pct_diff == pytest.approx(2.0)
    assert best_rnda != [0.01, 0.22, 0.022]


def test_optimize_mirrors_parallel_collects_results_and_returns_list(monkeypatch):
    inst = _make_minimal_instance(args_dict={"parallel": True})

    import simtools.settings as settings

    # Use the public API to set DB config; db_config is a read-only property.
    settings.config.load(args={}, db_config={"k": "v"})

    ctx_sentinel = object()
    monkeypatch.setattr(mpp, "get_context", MagicMock(return_value=ctx_sentinel))

    created_executors = []

    class FakeFuture:
        def __init__(self, value=None):
            self._value = value

        def result(self):
            return self._value

    class FakeExecutor:
        def __init__(self, *, max_workers, mp_context, initializer, initargs):
            self.max_workers = max_workers
            self.mp_context = mp_context
            self.initializer = initializer
            self.initargs = initargs
            created_executors.append(self)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, func, i):
            assert func is mpp._worker_optimize_mirror
            return FakeFuture(
                value={
                    "mirror": i + 1,
                    "optimized_rnda": [float(i), 0.0, 0.0],
                    "percentage_diff": 0.0,
                }
            )

    monkeypatch.setattr(mpp, "ProcessPoolExecutor", lambda **kw: FakeExecutor(**kw))
    monkeypatch.setattr(mpp, "as_completed", lambda futs: list(reversed(list(futs))))

    results = inst._optimize_mirrors_parallel(n_mirrors=3, n_workers=2)
    assert [r["mirror"] for r in results] == [1, 2, 3]

    assert len(created_executors) == 1
    ex = created_executors[0]
    assert ex.max_workers == 2
    assert ex.mp_context is ctx_sentinel
    assert ex.initializer is mpp._worker_init
    assert ex.initargs[0] == inst.label
    assert ex.initargs[1]["parallel"] is False
    assert ex.initargs[2] == {"k": "v"}


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
            "focal_length_m": 28.0,
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


def test_define_telescope_model_calls_initialize_simulation_models(monkeypatch):
    inst = _make_minimal_instance(
        args_dict={"site": "North", "telescope": "LSTN-01", "model_version": "6.0.0"}
    )
    tel_sentinel = object()
    site_sentinel = object()

    def _fake_init(*, label, site, telescope_name, model_version):
        assert label == "lbl"
        assert site == "North"
        assert telescope_name == "LSTN-01"
        assert model_version == "6.0.0"
        return tel_sentinel, site_sentinel, None

    monkeypatch.setattr(mpp, "initialize_simulation_models", _fake_init)

    tel, site_model = inst._define_telescope_model("lbl")
    assert tel is tel_sentinel
    assert site_model is site_sentinel


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
        side_effect=RuntimeError("boom"),
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
            "focal_length_m": 28.0,
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


def test_write_results_log_writes_file(tmp_path):
    inst = _make_minimal_instance(
        args_dict={"output_path": str(tmp_path), "telescope": "LSTN-01", "model_version": "6.0.0"}
    )
    inst.rnda_opt = [0.002, 0.22, 0.022]
    inst.final_percentage_diff = 12.34
    inst.per_mirror_results = [
        {
            "mirror": 1,
            "measured_d80_mm": 10.0,
            "simulated_d80_mm": 11.0,
            "percentage_diff": 10.0,
            "optimized_rnda": [0.002, 0.22, 0.022],
        }
    ]

    out_path = inst.write_results_log()
    assert out_path.endswith(".log")
    assert (tmp_path / Path(out_path).name).exists()
    text = (tmp_path / Path(out_path).name).read_text(encoding="utf-8")
    assert "Single-Mirror d80 Optimization Results" in text


def test_write_results_log_includes_plot_columns_when_present(tmp_path):
    inst = _make_minimal_instance(
        args_dict={"output_path": str(tmp_path), "telescope": "LSTN-01", "model_version": "6.0.0"}
    )
    inst.rnda_opt = [0.002, 0.22, 0.022]
    inst.final_percentage_diff = 12.34
    inst.per_mirror_results = [
        {
            "mirror": 1,
            "measured_d80_mm": 10.0,
            "simulated_d80_mm": 11.0,
            "percentage_diff": 10.0,
            "simulated_d80_mm_plot": 10.5,
            "percentage_diff_plot": 5.0,
            "optimized_rnda": [0.002, 0.22, 0.022],
        }
    ]

    out_path = inst.write_results_log()
    text = Path(out_path).read_text(encoding="utf-8")
    assert "Plot Sim" in text
    assert "Plot %" in text


def test_write_d80_histogram_all_branches(tmp_path, caplog):
    inst = _make_minimal_instance(args_dict={"output_path": str(tmp_path)})

    assert inst.write_d80_histogram() is None

    inst.args_dict["d80_hist"] = "hist.png"
    inst.per_mirror_results = []
    with caplog.at_level(logging.WARNING):
        assert inst.write_d80_histogram() is None
    assert "No valid d80 values" in caplog.text

    inst.per_mirror_results = [
        {"measured_d80_mm": 10.0, "simulated_d80_mm": 10.0},
        {"measured_d80_mm": 10.0, "simulated_d80_mm": 10.0},
    ]
    with caplog.at_level(logging.WARNING):
        assert inst.write_d80_histogram() is None
    assert "Invalid d80 range" in caplog.text

    inst.per_mirror_results = [
        {"measured_d80_mm": 10.0, "simulated_d80_mm": 12.0},
        {"measured_d80_mm": 11.0, "simulated_d80_mm": 13.0},
    ]
    out_path = inst.write_d80_histogram()
    assert out_path is not None
    assert (tmp_path / "hist.png").exists()
