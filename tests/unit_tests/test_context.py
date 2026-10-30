"""Tests for the run context."""

import copy
from pathlib import Path
from unittest.mock import Mock

import pytest

from simtools.context import Activity, RunContext, SimulationSoftware
from simtools.io.output_paths import OutputPaths


def test_run_context_from_arguments_reads_arguments(tmp_test_directory):
    args = {
        "application_label": "unit-test",
        "activity_id": "activity-1",
        "output_path": str(tmp_test_directory),
    }

    context = RunContext.from_arguments(args, resolve_sim_software_executables=False)

    assert context.activity.activity_id == "activity-1"
    assert context.activity.activity_name == "unit-test"
    assert context.paths.get_output_directory() == Path(tmp_test_directory).resolve()
    assert context.args["application_label"] == "unit-test"
    with pytest.raises(TypeError, match="does not support item assignment"):
        context.args["new"] = "value"


def test_run_context_args_are_a_view_of_the_given_arguments():
    args = {"model_version": "6"}
    context = RunContext.from_arguments(args, resolve_sim_software_executables=False)

    args["model_version"] = "6.0.0"

    assert context.args["model_version"] == "6.0.0"


def test_activity_generates_an_id_when_none_is_given():
    first = Activity.from_arguments({})
    second = Activity.from_arguments(None)

    assert first.activity_id
    assert first.activity_id != second.activity_id
    assert first.activity_name is None


def test_simulation_software_prefers_arguments_over_environment(monkeypatch, tmp_test_directory):
    monkeypatch.setenv("SIMTOOLS_SIM_TELARRAY_PATH", "/from/environment")
    monkeypatch.setenv("SIMTOOLS_CORSIKA_HE_INTERACTION", "EPOS")
    monkeypatch.setenv("SIMTOOLS_RAY_TRACING_BACKEND", "sim_telarray")

    software = SimulationSoftware.from_arguments(
        {"sim_telarray_path": str(tmp_test_directory), "corsika_path": None},
        resolve_sim_software_executables=False,
    )

    assert software.sim_telarray_path == Path(tmp_test_directory)
    assert software.corsika_interaction_models[0] == "epos"
    assert software.ray_tracing_backend == "sim_telarray"
    assert software.corsika_dummy_file == Path(tmp_test_directory) / "run9991.corsika.gz"


def test_simulation_software_rejects_unknown_ray_tracing_backend():
    with pytest.raises(ValueError, match="ray_tracing_backend must be one of"):
        SimulationSoftware.from_arguments({"ray_tracing_backend": "unknown"})


def test_simulation_software_reports_missing_installations(tmp_test_directory):
    missing = str(Path(tmp_test_directory) / "missing")
    software = SimulationSoftware(
        sim_telarray_root=missing, corsika_root=missing, corsika_interaction_table_root=missing
    )

    with pytest.raises(FileNotFoundError, match="sim_telarray path not found"):
        _ = software.sim_telarray_path
    with pytest.raises(FileNotFoundError, match="CORSIKA path not found"):
        _ = software.corsika_path
    with pytest.raises(FileNotFoundError, match="CORSIKA interaction table path not found"):
        _ = software.corsika_interaction_table_path
    with pytest.raises(FileNotFoundError, match="CORSIKA build manifest has not been loaded"):
        software.corsika_executable("curved")
    assert software.corsika_exe is None
    assert SimulationSoftware().corsika_dummy_file is None


def test_simulation_software_skips_corsika_without_installation(tmp_test_directory):
    software = SimulationSoftware.from_arguments(
        {"corsika_path": str(Path(tmp_test_directory) / "missing")}
    )

    assert software.corsika_build_variants is None
    assert software.corsika_flat_executable is None


def test_run_context_replace_and_deepcopy(tmp_test_directory):
    context = RunContext(paths=OutputPaths(tmp_test_directory))
    reader = Mock()

    replaced = context.replace(model_reader=reader)

    assert replaced.model_reader is reader
    assert context.model_reader is None
    assert copy.deepcopy(replaced) is replaced
    assert replaced.ray_tracing_backend == replaced.software.ray_tracing_backend


def test_from_environment_returns_a_fresh_context():
    first = RunContext.from_environment()
    second = RunContext.from_environment()
    assert first is not second
    assert first.model_reader is None
