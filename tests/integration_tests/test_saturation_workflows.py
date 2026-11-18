"""Run the small saturation workflows, including simulation-to-analysis handoff."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table

from simtools.application.model_reader import create_model_reader_from_configuration
from simtools.model.telescope_model import TelescopeModel
from simtools.runners.simtools_runner import prepare_workflow
from simtools.simtel.pulse_response import (
    ADC_SUM,
    INTENSITY,
    LINEAR,
    NONLINEAR,
    PEAK,
    RESPONSE,
    WIDTH,
)
from simtools.simtel.simtel_config_writer import INCOMPATIBLE_SIMTEL_MARKER
from simtools.testing import options
from simtools.testing.output_validation.runner import validate_application_output

MODEL_PATH = "simulation_models_path"
CONFIG_FILE = "config_file"
OUTPUT_PLACEHOLDER = "__OUTPUT__"
SUMMARY_SUFFIX = "_summary.ecsv"

_WORKFLOWS = Path(__file__).parent / "config"
_WORKFLOW_NAMES = (
    "validate_pulse_response",
    "validate_saturation_direct_injection_lst",
    "validate_saturation_flasher_lst",
)


@pytest.mark.parametrize("workflow", _WORKFLOW_NAMES)
def test_saturation_workflow(
    workflow, tmp_test_directory, request, simtools_root_path, monkeypatch
):
    model_version = request.config.getoption("model_version")
    if model_version and model_version != "8.0.0":
        pytest.skip("Saturation fixtures use model 8.0.0")
    model_path = request.config.getoption(MODEL_PATH, default=None)
    git_path = request.config.getoption("simulation_models_git_path", default=None)
    if not model_path and not git_path:
        model_path = options.get_mirrored_option(request.config, MODEL_PATH)
        git_path = options.get_mirrored_option(request.config, "simulation_models_git_path")
    revision = options.get_mirrored_option(request.config, "simulation_models_git_revision")
    if not model_path and not git_path:
        model_path = Path(simtools_root_path).parent / "simulation-models"
    use_git = bool(git_path and not model_path)
    model_path = Path(model_path or git_path).resolve()
    if not model_path.is_dir():
        pytest.skip("A local simulation-model repository is required")
    monkeypatch.setenv("MAX_PHOTOELECTRONS", "25000000")
    args = {
        CONFIG_FILE: _WORKFLOWS / f"{workflow}.yml",
        "ignore_runtime_environment": True,
    }
    if use_git:
        monkeypatch.setenv("SIMTOOLS_SIMULATION_MODELS_PATH", "None")
        args["simulation_models_git_path"] = model_path
        args["simulation_models_git_revision"] = revision or "HEAD"
    else:
        monkeypatch.setenv("SIMTOOLS_SIMULATION_MODELS_GIT_PATH", "None")
        args[MODEL_PATH] = model_path
    replacements = {OUTPUT_PLACEHOLDER: str(tmp_test_directory / workflow)}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "simtools.applications.run_application",
            "--config_file",
            str(args[CONFIG_FILE]),
            "--ignore_runtime_environment",
            *[
                str(value)
                for key, value in args.items()
                if key.startswith("simulation_models_")
                for value in (f"--{key}", value)
            ],
            "--replace",
            f"__OUTPUT__={replacements[OUTPUT_PLACEHOLDER]}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    _skip_incompatible_simtel(result)
    assert result.returncode == 0, result.stdout + result.stderr
    applications, *_ = prepare_workflow(args, replacements=replacements)
    for application in applications:
        validate_application_output(application)
    model = _pulse_model(args) if workflow != "validate_saturation_flasher_lst" else None
    if workflow == "validate_pulse_response":
        _check_template_rows(Path(replacements[OUTPUT_PLACEHOLDER]), model, applications[0])
    _check_response_curves(
        Path(replacements[OUTPUT_PLACEHOLDER]), workflow == "validate_pulse_response", model
    )


def _skip_incompatible_simtel(result):
    """Skip a workflow when its simulator is incompatible with its model."""
    if result.returncode != 0 and INCOMPATIBLE_SIMTEL_MARKER in result.stderr:
        reason = result.stderr.rsplit(INCOMPATIBLE_SIMTEL_MARKER, 1)[1].splitlines()[0].strip()
        pytest.skip(reason)


def _pulse_model(source):
    """Read the pulse model selected for the workflow."""
    reader = create_model_reader_from_configuration(source)
    return TelescopeModel(
        site="North", telescope_name="LSTN-01", model_version="8.0.0", model_reader=reader
    )


def test_dictionary_pulse_output_validation(tmp_test_directory):
    """Validate linear-only outputs for an inline pulse parameter."""

    class DictionaryModel:
        def get_parameter_value(self, _name):
            return {"columns": ["time", "amplitude"], "rows": [[0, 1]]}

        def get_parameter_version(self, _name):
            return "2.0.0"

    directory = Path(tmp_test_directory)
    table = Table(
        rows=[
            (gain, LINEAR, level, level, level, level, 1)
            for gain in ("HG", "LG")
            for level in (1, 2)
        ],
        names=("gain", RESPONSE, INTENSITY, PEAK, ADC_SUM, WIDTH, "event_count"),
    )
    table.write(directory / "pulse_response.ecsv")
    table.write(directory / "pulse_response_summary.ecsv")
    (directory / "linear").mkdir()
    table.write(directory / "linear" / "pulse_response_summary.ecsv")
    model = DictionaryModel()
    _check_template_rows(directory, model, {"configuration": {"illumination": [1, 2]}})
    _check_response_curves(directory, True, model)


def _check_template_rows(directory, model, application):
    """Check all expected gain, response and illumination combinations."""
    responses = {LINEAR, NONLINEAR} if _has_nonlinear_pulse(model) else {LINEAR}
    levels = application["configuration"]["illumination"]
    expected = {
        (gain, response, level)
        for gain in ("HG", "LG")
        for response in responses
        for level in levels
    }
    for name in ("pulse_response.ecsv", "pulse_response_summary.ecsv"):
        table = Table.read(directory / name)
        actual = list(zip(table["gain"], table[RESPONSE], table[INTENSITY], strict=True))
        assert len(actual) == len(expected)
        assert set(actual) == expected


def _check_response_curves(directory, template, model):
    """Check that both gains respond and broaden across the scan."""
    stem = "pulse_response" if template else "flasher_response"
    summary = Table.read(directory / f"{stem}{SUMMARY_SUFFIX}")
    for group in summary.group_by(["gain", RESPONSE]).groups:
        group.sort(INTENSITY)
        for metric in (PEAK, ADC_SUM, WIDTH):
            assert np.all(np.isfinite(group[metric]))
        assert group[PEAK][-1] > group[PEAK][0]
        # Individual faint events can have negative sums after pedestal subtraction.
        assert group[ADC_SUM][-1] > group[ADC_SUM][0]
        assert group[WIDTH][-1] > group[WIDTH][0]
        assert np.all(group["event_count"] == (1 if template else 2))
    control = directory / "linear" / f"{stem}{SUMMARY_SUFFIX}"
    if control.is_file():
        linear = Table.read(control)
        response_hg = summary[summary["gain"] == "HG"]
        linear_hg = linear[linear["gain"] == "HG"]
        response_hg.sort(INTENSITY)
        linear_hg.sort(INTENSITY)
        assert np.array_equal(response_hg[INTENSITY], linear_hg[INTENSITY])
        if _has_nonlinear_pulse(model):
            assert not np.allclose(response_hg[PEAK], linear_hg[PEAK])
        elif model.get_parameter_version("fadc_pulse_shape") == "2.0.0":
            assert np.allclose(response_hg[PEAK], linear_hg[PEAK])


def _has_nonlinear_pulse(model):
    """Identify pulse tables containing nonlinear templates."""
    value = model.get_parameter_value("fadc_pulse_shape")
    if isinstance(value, dict):
        return "template_kind" in value["columns"]
    return "template_kind" in model.get_parameter_table("fadc_pulse_shape").colnames
