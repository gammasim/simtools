#!/usr/bin/python3

"""Tests for the simulate_prod application."""

import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock

import astropy.units as u
import pytest

import simtools.applications.simulate_prod as app
from simtools.production_configuration import job_grid_io

pytestmark = pytest.mark.usefixtures("_mock_settings_env_vars")


@pytest.fixture
def job_grid_file(tmp_test_directory):
    """Return a two-row production job grid file for parser tests."""
    grid_file = tmp_test_directory / "grid.ecsv"
    row = {
        "run_number": 7,
        "primary": "gamma",
        "azimuth_angle": 45 * u.deg,
        "zenith_angle": 20 * u.deg,
        "ha": 123 * u.deg,
        "dec": -45 * u.deg,
        "energy_min": 30 * u.GeV,
        "energy_max": 10 * u.TeV,
        "cores_per_shower": 10,
        "core_scatter_max": 200 * u.m,
        "view_cone_min": 0 * u.deg,
        "view_cone_max": 5 * u.deg,
        "showers_per_run": 1000,
        "nsb_rate": 0.24,
        "model_version": "7.0.0",
        "array_layout_name": "CTAO-North-Alpha",
        "corsika_le_interaction": "urqmd",
        "corsika_he_interaction": "epos",
    }
    job_grid_io.serialize_job_grid(
        [row, {**row, "run_number": 11, "zenith_angle": 40 * u.deg}],
        grid_file,
        metadata={"site": "North", "simulation_software": "corsika_sim_telarray"},
    )
    return grid_file


def _parse_with_args(monkeypatch, args):
    monkeypatch.setattr(sys, "argv", ["simulate_prod.py", *map(str, args)])
    return app.APPLICATION._parse()[0]


def _job_grid_args(job_grid_file, *extra_args):
    return ["--job_grid_file", job_grid_file, *extra_args]


def _parse_with_config_sources(job_grid_file, cli_keys=(), yaml_keys=()):
    args = {"job_grid_file": str(job_grid_file), "job_grid_row": 1}
    app._resolve_job_grid_arguments(
        args,
        {
            "defaults": set(),
            "environment": set(),
            "constructor": set(),
            "yaml": set(yaml_keys),
            "cli": set(cli_keys),
        },
        argparse.ArgumentParser(),
    )
    return args


def _mock_application_context(mock_application_start, label="test"):
    mock_application_start.return_value = MagicMock(
        args={
            "label": label,
            "reduced_event_lists": True,
            "save_file_lists": False,
            "grid_output_path": None,
        }
    )


def test_list_available_corsika_models_exits_with_table(tmp_test_directory, capsys):
    build_options = tmp_test_directory / "build_opts.yml"
    build_options.write_text(
        "variant:\n"
        "  - executable: corsika_qgs3_urqmd_flat\n"
        "    config: config_qgs3_urqmd_flat\n"
        "    atmosphere_geometry: flat\n"
        "    he_hadronic_model: qgs3\n"
        "    le_hadronic_model: urqmd\n",
        encoding="utf-8",
    )
    executable = Path(tmp_test_directory) / "corsika_qgs3_urqmd_flat"
    executable.touch()
    executable.chmod(0o755)

    with pytest.raises(SystemExit) as exc:
        app._list_available_corsika_models(
            {"corsika_path": tmp_test_directory}, argparse.ArgumentParser()
        )

    assert exc.value.code == 0
    assert "qgs3" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("row_args", "expected"),
    [
        ((), {"run_number": 7, "primary": "gamma", "site": "North"}),
        (("--job_grid_row", 2), {"run_number": 11, "zenith_angle": 40 * u.deg}),
    ],
)
def test_parse_job_grid_file_selects_row(
    monkeypatch, job_grid_file, tmp_test_directory, row_args, expected
):
    args = _parse_with_args(
        monkeypatch,
        _job_grid_args(job_grid_file, *row_args, "--output_path", tmp_test_directory),
    )

    for key, value in expected.items():
        assert args[key] == value
    assert args["simulation_software"] == "corsika_sim_telarray"


def test_sim_telarray_only_does_not_require_primary(monkeypatch):
    args = _parse_with_args(
        monkeypatch,
        ["--simulation_software", "sim_telarray", "--array_layout_name", "alpha"],
    )

    assert args["primary"] is None
