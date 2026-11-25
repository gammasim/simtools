"""Exercise the 365 nm flasher simulation and illumination-map workflow."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table
from eventio import SimTelFile


def test_flashcam_dc_illumination(tmp_path, request, simtools_root_path):
    """The map is normalized over configured pixels with HV enabled."""
    model_version = request.config.getoption("model_version")
    if model_version and model_version != "8.0.0":
        pytest.skip("The FlashCam illumination workflow uses model 8.0.0")
    model_path = request.config.getoption("simulation_models_path", default=None)
    git_path = request.config.getoption("simulation_models_git_path", default=None)
    revision = request.config.getoption("simulation_models_git_revision", default=None)
    if not model_path and not git_path:
        model_path = Path(simtools_root_path).parent / "simulation-models"
    if model_path:
        model_args = ["--simulation_models_path", str(Path(model_path).resolve())]
    else:
        model_args = ["--simulation_models_git_path", str(Path(git_path).resolve())]
        if revision:
            model_args.extend(["--simulation_models_git_revision", revision])
    environment = os.environ.copy()
    environment["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{environment.get('PATH', '')}"
    config = Path(__file__).parent / "config/derive_flashcam_dc_illumination.yml"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "simtools.applications.run_application",
            "--config_file",
            str(config),
            "--ignore_runtime_environment",
            *model_args,
            "--replace",
            f"__OUTPUT__={tmp_path}",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=simtools_root_path,
        env=environment,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    table = Table.read(tmp_path / "flashcam_dc_illumination.ecsv", format="ascii.ecsv")
    assert len(table) == 1764
    assert table.meta["number_of_events"] == 10
    rates = np.asarray(table["relative_rate"])
    simtel_path = next((tmp_path / "simulation/sim_telarray").glob("*.simtel.zst"))
    with SimTelFile(simtel_path) as simtel:
        tel_id = simtel.header["tel_id"][0]
        disabled = simtel.telescope_descriptions[tel_id]["disabled_pixels"]["HV_disabled"]
    assert len(disabled) == 6
    enabled = np.ones(len(rates), dtype=bool)
    enabled[disabled] = False
    assert np.all(rates[~enabled] == 0)
    assert np.mean(rates[enabled]) == pytest.approx(1.0)
