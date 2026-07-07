#!/usr/bin/python3

"""Tests for the simulate_prod application."""

import argparse
import sys
from unittest.mock import MagicMock, patch

import astropy.units as u
import pytest
import yaml

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
    return app._parse()[0]


def _job_grid_args(job_grid_file, *extra_args):
    return ["--job_grid_file", job_grid_file, *extra_args]


def _write_simulate_prod_config(config_file, configuration):
    with open(config_file, "w", encoding="utf-8") as output:
        yaml.safe_dump(
            {
                "applications": [
                    {
                        "application": "simtools-simulate-prod",
                        "configuration": configuration,
                    }
                ]
            },
            output,
        )


def test_add_arguments_registers_job_grid_file_and_row():
    parser = argparse.ArgumentParser()

    app._add_arguments(parser)
    args = parser.parse_args(["--job_grid_file", "my_grid.ecsv", "--job_grid_row", "3"])

    assert args.job_grid_file == "my_grid.ecsv"
    assert args.job_grid_row == 3


def test_add_arguments_job_grid_row_defaults_to_one():
    parser = argparse.ArgumentParser()

    app._add_arguments(parser)
    args = parser.parse_args([])

    assert args.job_grid_file is None
    assert args.job_grid_row == 1


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


def test_parse_job_grid_row_without_file_fails(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _parse_with_args(monkeypatch, ["--job_grid_row", 2])

    stderr = capsys.readouterr().err
    assert "job_grid_row" in stderr
    assert "job_grid_file" in stderr


def test_parse_job_grid_file_rejects_explicit_cli_production_parameter(
    monkeypatch, capsys, job_grid_file
):
    with pytest.raises(SystemExit):
        _parse_with_args(
            monkeypatch,
            _job_grid_args(job_grid_file, "--zenith_angle", 20),
        )

    assert "zenith_angle" in capsys.readouterr().err


def test_parse_job_grid_file_rejects_yaml_production_parameter(
    monkeypatch, capsys, job_grid_file, tmp_test_directory
):
    config_file = tmp_test_directory / "simulate_prod.yml"
    _write_simulate_prod_config(
        config_file,
        {
            "job_grid_file": str(job_grid_file),
            "zenith_angle": 20,
        },
    )

    with pytest.raises(SystemExit):
        _parse_with_args(monkeypatch, ["--config", config_file])

    assert "zenith_angle" in capsys.readouterr().err


def test_parse_job_grid_file_allows_operational_yaml_and_cli_parameters(
    monkeypatch, job_grid_file, tmp_test_directory
):
    config_file = tmp_test_directory / "simulate_prod.yml"
    _write_simulate_prod_config(
        config_file,
        {
            "job_grid_file": str(job_grid_file),
            "label": "grid-test",
            "log_level": "DEBUG",
            "output_path": str(tmp_test_directory),
        },
    )

    args = _parse_with_args(monkeypatch, ["--config", config_file, "--save_file_lists"])

    assert args["run_number"] == 7
    assert args["label"] == "grid-test"
    assert args["save_file_lists"] is True


@pytest.mark.parametrize(
    "context_args",
    [
        {
            "label": "test",
            "save_reduced_event_lists": False,
            "save_file_lists": False,
            "grid_output_path": None,
        }
    ],
)
@patch("simtools.applications.simulate_prod.Simulator")
@patch("simtools.applications.simulate_prod.build_application")
def test_main_calls_build_application_with_simulate_prod_parse_function(
    mock_build_app, mock_simulator_class, context_args
):
    mock_build_app.return_value = MagicMock(args=context_args)
    mock_simulator_class.return_value = MagicMock()

    app.main()

    call_kwargs = mock_build_app.call_args.kwargs
    assert call_kwargs["startup_kwargs"]["setup_io_handler"] is False
    assert call_kwargs["parse_function"] == app._parse


@patch("simtools.applications.simulate_prod.Simulator")
@patch("simtools.applications.simulate_prod.build_application")
def test_main_runs_simulator_and_reports(mock_build_app, mock_simulator_class):
    mock_build_app.return_value = MagicMock(
        args={
            "label": "myprod",
            "save_reduced_event_lists": False,
            "save_file_lists": False,
            "grid_output_path": None,
        }
    )
    mock_simulator = MagicMock()
    mock_simulator_class.return_value = mock_simulator

    app.main()

    mock_simulator_class.assert_called_once_with(label="myprod")
    mock_simulator.simulate.assert_called_once()
    mock_simulator.validate_simulations.assert_called_once()
    mock_simulator.report.assert_called_once()
