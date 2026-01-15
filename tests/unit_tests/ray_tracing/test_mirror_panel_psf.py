#!/usr/bin/python3

import json
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import astropy.units as u
import pytest
from astropy.table import Table

import simtools.ray_tracing.mirror_panel_psf as mpp
from simtools.ray_tracing.mirror_panel_psf import MirrorPanelPSF


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


def _make_minimal_instance(*, label: str = "base", args_dict: dict | None = None) -> MirrorPanelPSF:
    inst = MirrorPanelPSF.__new__(MirrorPanelPSF)
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


def _run_sigma1_gradient_descent(inst: MirrorPanelPSF, **overrides):
    params = {
        "mirror_idx": 0,
        "measured_d80_mm": 10.0,
        "focal_length_m": 28.0,
        "current_rnda": [0.01, 0.22, 0.022],
        "threshold": 0.05,
        "learning_rate": 1e-4,
        "n_runs_per_eval": 1,
        "grad_clip": 1e4,
        "max_log_step": 0.25,
        "sigma_min": 1e-4,
        "sigma_max": 0.1,
        "max_iterations": 10,
    }
    params.update(overrides)
    return inst._run_sigma1_gradient_descent(**params)


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


@pytest.mark.parametrize(
    "mirror_list,model_path,expected_site",
    [
        (None, None, "dummy_site"),
        ("mirrors.ecsv", "/models", "site"),
    ],
)
def test_define_telescope_model_mirror_list_handling(mocker, mirror_list, model_path, expected_site):
    args_dict = {
        "site": "North",
        "telescope": "LSTN-01",
        "model_version": "1.2.3",
        "mirror_list": mirror_list,
    }
    if model_path is not None:
        args_dict["model_path"] = model_path
    inst = _make_minimal_instance(args_dict=args_dict)

    dummy_tel = _make_dummy_tel(label="dummy")
    mock_init = mocker.patch.object(
        mpp,
        "initialize_simulation_models",
        return_value=(dummy_tel, expected_site, None),
    )
    mock_find = mocker.patch.object(mpp.gen, "find_file", return_value=f"{model_path}/{mirror_list}")

    tel_model, site_model = inst._define_telescope_model("lbl")
    assert tel_model is dummy_tel
    assert site_model == expected_site
    mock_init.assert_called_once_with(
        label="lbl",
        site="North",
        telescope_name="LSTN-01",
        model_version="1.2.3",
    )

    if mirror_list is None:
        mock_find.assert_not_called()
        dummy_tel.overwrite_model_parameter.assert_not_called()
        dummy_tel.overwrite_model_file.assert_not_called()
    else:
        mock_find.assert_called_once_with(name="mirrors.ecsv", loc="/models")
        dummy_tel.overwrite_model_parameter.assert_called_once_with("mirror_list", "mirrors.ecsv")
        dummy_tel.overwrite_model_file.assert_called_once_with("mirror_list", "/models/mirrors.ecsv")


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
            self._results = {"d80_cm": [SimpleNamespace(value=1.5)]}

        def simulate(self, **kwargs):
            return None

        def analyze(self, **kwargs):
            return None

    mocker.patch("simtools.ray_tracing.mirror_panel_psf.RayTracing", OkRay)
    d80_mm = inst._simulate_single_mirror_d80(0, 28.0, [0.01, 0.22, 0.022])
    assert d80_mm == pytest.approx(15.0)
    assert tel.label == "orig"
 

def test_run_sigma1_gradient_descent_converges_immediately(mocker):
    inst = _make_minimal_instance()
    mocker.patch.object(inst, "_simulate_single_mirror_d80", return_value=12.0)
    best_rnda, best_sim_d80, best_pct_diff = _run_sigma1_gradient_descent(
        inst,
        measured_d80_mm=12.0,
        threshold=0.05,
        n_runs_per_eval=3,
        max_iterations=10,
    )
    assert best_rnda == [0.01, 0.22, 0.022]
    assert best_sim_d80 == pytest.approx(12.0)
    assert best_pct_diff == pytest.approx(0.0)


def test_run_sigma1_gradient_descent_denominator_collapse_breaks(mocker):
    inst = _make_minimal_instance()
    mocker.patch.object(inst, "_simulate_single_mirror_d80", return_value=20.0)
    best_rnda, best_sim_d80, best_pct_diff = _run_sigma1_gradient_descent(
        inst,
        measured_d80_mm=10.0,
        threshold=0.01,
        n_runs_per_eval=1,
        sigma_min=0.01,
        sigma_max=0.01,
        max_iterations=10,
    )
    assert best_rnda == [0.01, 0.22, 0.022]
    assert best_sim_d80 == pytest.approx(20.0)
    assert best_pct_diff == pytest.approx(100.0)


def test_run_sigma1_gradient_descent_stops_when_learning_rate_too_small(mocker):
    inst = _make_minimal_instance()
    mock_sim = mocker.patch.object(inst, "_simulate_single_mirror_d80", return_value=20.0)
    best_rnda, best_sim_d80, best_pct_diff = _run_sigma1_gradient_descent(
        inst,
        measured_d80_mm=10.0,
        threshold=0.0,
        n_runs_per_eval=1,
        sigma_min=1e-4,
        sigma_max=0.1,
        max_iterations=200,
    )
    assert best_rnda == [0.01, 0.22, 0.022]
    assert best_sim_d80 == pytest.approx(20.0)
    assert best_pct_diff == pytest.approx(100.0)
    assert mock_sim.call_count > 50


def test_init_sets_test_mirror_limit_in_test_mode(mocker):
    dummy_tel = _make_dummy_tel(label="tel")
    dummy_tel.get_parameter_value = MagicMock(return_value=[0.0075, 0.22, 0.022])
    mocker.patch.object(MirrorPanelPSF, "_define_telescope_model", return_value=(dummy_tel, object()))
    mocker.patch.object(
        MirrorPanelPSF,
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
    inst = MirrorPanelPSF("lbl", args)
    assert inst.args_dict["number_of_mirrors_to_test"] == 10


def test_run_sigma1_gradient_descent_accepts_improvement_and_converges(mocker):
    inst = _make_minimal_instance()
    # For n_runs_per_eval=1, the first iteration triggers 4 simulations:
    # initial, plus, minus, new.
    mocker.patch.object(
        inst,
        "_simulate_single_mirror_d80",
        side_effect=[20.0, 19.0, 21.0, 10.1],
    )

    best_rnda, best_sim_d80, best_pct_diff = _run_sigma1_gradient_descent(
        inst,
        measured_d80_mm=10.0,
        threshold=0.05,
        n_runs_per_eval=1,
        max_iterations=10,
    )
    assert best_sim_d80 == pytest.approx(10.1)
    assert best_pct_diff == pytest.approx(1.0)
    assert best_rnda[0] != pytest.approx(0.01)


def test_optimize_single_mirror_returns_expected_dict(mocker):
    inst = _make_minimal_instance(args_dict={"threshold": 0.05, "learning_rate": 1e-4})
    mocker.patch.object(
        inst,
        "_run_sigma1_gradient_descent",
        return_value=([0.002, 0.22, 0.022], 15.0, 5.0),
    )
    result = inst._optimize_single_mirror(0, 14.0, 28.0)
    assert result["mirror"] == 1
    assert result["measured_d80_mm"] == pytest.approx(14.0)
    assert result["focal_length_m"] == pytest.approx(28.0)
    assert result["optimized_rnda"] == [0.002, 0.22, 0.022]
    assert result["simulated_d80_mm"] == pytest.approx(15.0)
    assert result["percentage_diff"] == pytest.approx(5.0)



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
    out_file = tmp_path / "per_mirror_optimization_results.json"
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert data["optimization_metric"] == "percentage_difference"
    assert data["optimized_rnda_averaged"] == [0.002, 0.22, 0.022]
    assert len(data["per_mirror_results"]) == 1


def test_print_results_formats_table_with_and_without_plot_columns(capsys):
    inst = _make_minimal_instance()
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
    inst.print_results()
    out = capsys.readouterr().out
    assert "Single-Mirror d80 Optimization Results" in out

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
    inst.print_results()
    out2 = capsys.readouterr().out
    assert "Plot Sim" in out2
    assert "Plot %" in out2


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
