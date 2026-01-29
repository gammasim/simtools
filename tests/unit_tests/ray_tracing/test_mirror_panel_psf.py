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


def test_signed_pct_diff_raises_on_nonpositive_measured_d80():
    with pytest.raises(ValueError, match="Measured d80 must be positive"):
        mpp.MirrorPanelPSF._signed_pct_diff(0.0, 1.0)
    with pytest.raises(ValueError, match="Measured d80 must be positive"):
        mpp.MirrorPanelPSF._signed_pct_diff(-1.0, 1.0)


def test_signed_pct_diff_returns_signed_value():
    assert mpp.MirrorPanelPSF._signed_pct_diff(10.0, 12.0) == pytest.approx(20.0)
    assert mpp.MirrorPanelPSF._signed_pct_diff(10.0, 8.0) == pytest.approx(-20.0)


def test_evaluate_returns_sim_pct_and_squared_pct(mocker):
    inst = _make_minimal_instance()
    simulate = mocker.patch.object(inst, "_simulate_single_mirror_d80", return_value=12.0)

    sim, pct, obj = inst._evaluate(0, 10.0, [0.01, 0.2, 0.02])

    simulate.assert_called_once()
    assert sim == pytest.approx(12.0)
    assert pct == pytest.approx(20.0)
    assert obj == pytest.approx(400.0)


def test_rnda_bounds_clamps_sigma_min_to_positive(monkeypatch):
    def _fake_model_parameters():
        return {
            "mirror_reflection_random_angle": {
                "data": [
                    {"allowed_range": {"min": 0.0, "max": 0.1}},
                    {"allowed_range": {"min": 0.0, "max": 1.0}},
                    {"allowed_range": {"min": 0.0, "max": 0.2}},
                ]
            }
        }

    monkeypatch.setattr(mpp.names, "model_parameters", _fake_model_parameters)
    bounds = mpp.MirrorPanelPSF._rnda_bounds()
    assert bounds[0][0] == pytest.approx(1e-12)
    assert bounds[1][0] == pytest.approx(0.0)
    assert bounds[2][0] == pytest.approx(1e-12)


def test_optimize_with_gradient_descent_limits_mirrors_in_test_mode(mocker):
    inst = _make_minimal_instance(
        args_dict={"test": True, "number_of_mirrors_to_test": 2, "n_workers": 4}
    )
    inst.measured_data = [10.0, 11.0, 12.0]

    parent_stub = SimpleNamespace(measured_data=list(inst.measured_data))
    mocker.patch.object(mpp, "MirrorPanelPSF", return_value=parent_stub)

    def _fake_ppm(func, items, **kwargs):
        assert func is mpp._optimize_single_mirror_worker
        assert len(items) == 2
        assert kwargs["max_workers"] == 4
        assert kwargs["mp_start_method"] == "fork"
        return [
            mpp.MirrorOptimizationResult(
                mirror=1,
                measured_d80_mm=10.0,
                optimized_rnda=[1.0, 0.2, 0.03],
                simulated_d80_mm=11.0,
                percentage_diff=10.0,
            ),
            mpp.MirrorOptimizationResult(
                mirror=2,
                measured_d80_mm=11.0,
                optimized_rnda=[3.0, 0.2, 0.01],
                simulated_d80_mm=12.0,
                percentage_diff=30.0,
            ),
        ]

    mocker.patch.object(mpp, "process_pool_map_ordered", side_effect=_fake_ppm)

    inst.optimize_with_gradient_descent()

    assert inst.rnda_opt == pytest.approx([2.0, 0.2, 0.02])
    assert inst.final_percentage_diff == pytest.approx(20.0)


def test_optimize_single_mirror_worker_calls_optimizer_with_measured_value():
    dummy = SimpleNamespace(optimize_single_mirror=MagicMock(return_value={"ok": True}))
    result = mpp._optimize_single_mirror_worker((dummy, 1, 12.25))
    dummy.optimize_single_mirror.assert_called_once_with(1, 12.25)
    assert result == {"ok": True}


def test_simulate_single_mirror_d80_success_and_restores_label(mocker, tmp_path):
    tel = _make_dummy_tel(label="orig", config_dir=tmp_path)
    inst = _make_minimal_instance(label="base", telescope_model=tel)

    class OkRay:
        def __init__(self, **kwargs):
            assert kwargs["label"] == "base_m1"
            assert kwargs["telescope_model"].label == "orig"
            self._d80_mm = 15.0

        def simulate(self, **kwargs):
            return None

        def analyze(self, **kwargs):
            return None

        def get_d80_mm(self):
            return self._d80_mm

    mocker.patch("simtools.ray_tracing.mirror_panel_psf.RayTracing", OkRay)
    d80_mm = inst._simulate_single_mirror_d80(0, [0.01, 0.22, 0.022])
    assert d80_mm == pytest.approx(15.0)
    assert tel.label == "orig"
    assert tel._config_file_path == str(tmp_path / "orig.cfg")
    assert tel._config_file_directory == str(tmp_path)


def test_optimize_single_mirror_converges_when_simulated_matches_measured(mocker):
    inst = _make_minimal_instance(args_dict={"threshold": 0.05, "learning_rate": 1e-4})
    mocker.patch.object(
        inst,
        "_rnda_bounds",
        return_value=[(1e-12, 1.0), (0.0, 1.0), (1e-12, 1.0)],
    )

    mocker.patch.object(inst, "_evaluate", return_value=(12.0, 0.0, 0.0))

    def _return_param_value(**kwargs):
        return kwargs["param_value"]

    mocker.patch.object(inst, "_update_single_rnda_parameter", side_effect=_return_param_value)

    result = inst.optimize_single_mirror(0, 12.0)

    assert isinstance(result, mpp.MirrorOptimizationResult)
    assert result.mirror == 1
    assert result.measured_d80_mm == pytest.approx(12.0)
    assert result.simulated_d80_mm == pytest.approx(12.0)
    assert result.percentage_diff == pytest.approx(0.0)


def test_optimize_single_mirror_increases_learning_rate_on_improvement(mocker):
    inst = _make_minimal_instance(args_dict={"threshold": 0.0, "learning_rate": 1e-3})
    inst.MAX_ITER = 2

    mocker.patch.object(
        inst,
        "_rnda_bounds",
        return_value=[(1e-12, 1.0), (0.0, 1.0), (1e-12, 1.0)],
    )

    evaluate_calls = [
        (10.0, 10.0, 100.0),  # initial best
        (9.0, 9.0, 81.0),  # improvement => learning_rate *= 1.1
        (9.5, 9.5, 90.25),  # worse
    ]
    mocker.patch.object(inst, "_evaluate", side_effect=evaluate_calls)

    learning_rates = []

    def _record_lr(**kwargs):
        learning_rates.append(kwargs["learning_rate"])
        return kwargs["param_value"]

    mocker.patch.object(inst, "_update_single_rnda_parameter", side_effect=_record_lr)

    inst.optimize_single_mirror(0, 10.0)

    assert len(learning_rates) == 6
    assert learning_rates[:3] == pytest.approx([1e-3, 1e-3, 1e-3])
    assert learning_rates[3:] == pytest.approx([1.1e-3, 1.1e-3, 1.1e-3])


def test_update_single_rnda_parameter_log_param_branch(mocker):
    inst = _make_minimal_instance(args_dict={"learning_rate": 1e-3})
    rnda = [0.01, 0.2, 0.02]

    def _fake_evaluate(_mirror_idx, _measured_d80, rnda_in):
        # Objective depends only on the parameter being updated.
        return 0.0, 0.0, float(rnda_in[0] ** 2)

    mocker.patch.object(inst, "_evaluate", side_effect=_fake_evaluate)

    updated = inst._update_single_rnda_parameter(
        mirror_idx=0,
        measured_d80=10.0,
        rnda=rnda,
        param_idx=0,
        param_value=rnda[0],
        param_bounds=(1e-12, 1.0),
        learning_rate=1e-3,
    )

    assert 1e-12 <= float(updated) <= 1.0
    assert float(updated) < 0.01
    # Function should leave the input vector unchanged; caller assigns the return value.
    assert rnda[0] == pytest.approx(0.01)


def test_update_single_rnda_parameter_fraction_param_branch(mocker):
    inst = _make_minimal_instance(args_dict={"learning_rate": 1e-3})
    rnda = [0.01, 0.2, 0.02]

    def _fake_evaluate(_mirror_idx, _measured_d80, rnda_in):
        # Objective minimized at fraction2=0.5.
        return 0.0, 0.0, float((rnda_in[1] - 0.5) ** 2)

    mocker.patch.object(inst, "_evaluate", side_effect=_fake_evaluate)

    updated = inst._update_single_rnda_parameter(
        mirror_idx=0,
        measured_d80=10.0,
        rnda=rnda,
        param_idx=1,
        param_value=rnda[1],
        param_bounds=(0.0, 1.0),
        learning_rate=0.05,
    )

    assert 0.0 <= float(updated) <= 1.0
    assert float(updated) > 0.2
    assert rnda[1] == pytest.approx(0.2)


def test_write_optimization_data_writes_json(tmp_path):
    inst = _make_minimal_instance(
        args_dict={"output_path": str(tmp_path), "telescope": "LSTN-01", "model_version": "6.0.0"}
    )
    inst.rnda_opt = [0.002, 0.22, 0.022]
    inst.final_percentage_diff = 12.34
    inst.per_mirror_results = [
        mpp.MirrorOptimizationResult(
            mirror=1,
            measured_d80_mm=10.0,
            optimized_rnda=[0.002, 0.22, 0.022],
            simulated_d80_mm=11.0,
            percentage_diff=10.0,
        )
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
        mpp.MirrorOptimizationResult(
            mirror=1,
            measured_d80_mm=10.0,
            optimized_rnda=[0.002, 0.22, 0.022],
            simulated_d80_mm=11.0,
            percentage_diff=10.0,
        )
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


def test_write_d80_histogram_calls_plotter(monkeypatch):
    inst = _make_minimal_instance(args_dict={"output_path": "."})
    inst.per_mirror_results = [
        mpp.MirrorOptimizationResult(
            mirror=1,
            measured_d80_mm=10.0,
            optimized_rnda=[0.0, 0.0, 0.0],
            simulated_d80_mm=12.0,
            percentage_diff=20.0,
        ),
        mpp.MirrorOptimizationResult(
            mirror=2,
            measured_d80_mm=11.0,
            optimized_rnda=[0.0, 0.0, 0.0],
            simulated_d80_mm=13.0,
            percentage_diff=18.1818,
        ),
    ]

    called = {}

    def _fake_plot(measured, simulated, args_dict):
        called["measured"] = measured
        called["simulated"] = simulated
        called["args_dict"] = args_dict
        return "hist.png"

    monkeypatch.setattr(mpp.plot_psf, "plot_d80_histogram", _fake_plot)
    out = inst.write_d80_histogram()
    assert out == "hist.png"
    assert called["measured"] == pytest.approx([10.0, 11.0])
    assert called["simulated"] == pytest.approx([12.0, 13.0])
