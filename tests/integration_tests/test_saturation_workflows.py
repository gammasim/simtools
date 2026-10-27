"""Run the small saturation workflows, including simulation-to-analysis handoff."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table

from simtools.runners.simtools_runner import prepare_workflow
from simtools.simtel.pulse_response import ADC_SUM, INTENSITY, PEAK, RESPONSE, WIDTH
from simtools.testing.validate_output import validate_application_output

MODEL_PATH = "simulation_models_path"
CONFIG_FILE = "config_file"
OUTPUT_PLACEHOLDER = "__OUTPUT__"
SUMMARY_SUFFIX = "_summary.ecsv"

_WORKFLOWS = Path(__file__).parent / "config" / "saturation"


@pytest.mark.parametrize("workflow", sorted(path.stem for path in _WORKFLOWS.glob("*.yml")))
def test_saturation_workflow(
    workflow, tmp_test_directory, request, simtools_root_path, monkeypatch
):
    model_version = request.config.getoption("model_version")
    if model_version and model_version != "8.0.0":
        pytest.skip("Saturation fixtures use model 8.0.0")
    model_path = request.config.getoption(MODEL_PATH, default=None)
    model_path = (
        Path(model_path) if model_path else Path(simtools_root_path).parent / "simulation-models"
    )
    if not model_path.is_dir():
        pytest.skip("A local simulation-model repository is required")
    monkeypatch.setenv("MAX_PHOTOELECTRONS", "25000000")
    args = {
        CONFIG_FILE: _WORKFLOWS / f"{workflow}.yml",
        "ignore_runtime_environment": True,
        MODEL_PATH: model_path,
    }
    replacements = {OUTPUT_PLACEHOLDER: str(tmp_test_directory / workflow)}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "simtools.applications.run_application",
            "--config_file",
            str(args[CONFIG_FILE]),
            "--ignore_runtime_environment",
            "--simulation_models_path",
            str(model_path),
            "--replace",
            f"__OUTPUT__={replacements[OUTPUT_PLACEHOLDER]}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    applications, *_ = prepare_workflow(args, replacements=replacements)
    for application in applications:
        validate_application_output(application)
    _check_response_curves(
        Path(replacements[OUTPUT_PLACEHOLDER]), workflow == "validate_pulse_response"
    )


def _check_response_curves(directory, template):
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
        nonlinear_hg = summary[summary["gain"] == "HG"]
        linear_hg = linear[linear["gain"] == "HG"]
        assert not np.allclose(nonlinear_hg[PEAK], linear_hg[PEAK])
