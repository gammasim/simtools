#!/usr/bin/python3

"""Tests for the simulate_prod application."""

import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import astropy.units as u
import pytest

import simtools.applications.simulate_prod as app
from simtools.configuration.commandline_parser import CommandLineParser
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
            "save_reduced_event_lists": False,
            "save_file_lists": False,
            "grid_output_path": None,
        }
    )


def test_add_arguments_registers_job_grid_file_and_row():
    parser = CommandLineParser()
    parser.add_argument_definitions(app._ARGUMENTS)
    args = parser.parse_args(["--job_grid_file", "my_grid.ecsv", "--job_grid_row", "3"])

    assert args.job_grid_file == "my_grid.ecsv"
    assert args.job_grid_row == 3


def test_add_arguments_job_grid_row_defaults_to_one():
    parser = CommandLineParser()
    parser.add_argument_definitions(app._ARGUMENTS)
    args = parser.parse_args([])

    assert args.job_grid_file is None
    assert args.job_grid_row == 1
    assert args.save_corsika_output is False


def test_add_arguments_save_corsika_output():
    parser = CommandLineParser()
    parser.add_argument_definitions(app._ARGUMENTS)

    args = parser.parse_args(["--save_corsika_output"])

    assert args.save_corsika_output is True


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


def test_validate_single_interaction_models_rejects_lists(capsys):
    with pytest.raises(SystemExit):
        app._validate_single_interaction_models(
            {"corsika_he_interaction": ["epos", "qgs3"]}, argparse.ArgumentParser()
        )

    assert "accepts exactly one value" in capsys.readouterr().err


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


def test_parse_accepts_simulation_models_path(monkeypatch, job_grid_file, tmp_test_directory):
    args = _parse_with_args(
        monkeypatch,
        _job_grid_args(
            job_grid_file,
            "--output_path",
            tmp_test_directory,
            "--simulation_models_path",
            tmp_test_directory,
        ),
    )

    assert args["simulation_models_path"] == Path(tmp_test_directory)


def test_parse_job_grid_row_without_file_fails(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _parse_with_args(monkeypatch, ["--job_grid_row", 2])

    stderr = capsys.readouterr().err
    assert "job_grid_row" in stderr
    assert "job_grid_file" in stderr


def test_sim_telarray_only_does_not_require_primary(monkeypatch):
    args = _parse_with_args(
        monkeypatch,
        ["--simulation_software", "sim_telarray", "--array_layout_name", "alpha"],
    )

    assert args["primary"] is None


def test_corsika_requires_primary(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _parse_with_args(
            monkeypatch,
            ["--simulation_software", "corsika", "--array_layout_name", "alpha"],
        )

    assert "--primary" in capsys.readouterr().err


@pytest.mark.parametrize("source", ["cli", "yaml"])
def test_job_grid_file_rejects_explicit_production_parameter(capsys, job_grid_file, source):
    source_kwargs = {f"{source}_keys": {"zenith_angle"}}

    with pytest.raises(SystemExit):
        _parse_with_config_sources(job_grid_file, **source_kwargs)

    assert "zenith_angle" in capsys.readouterr().err


def test_job_grid_file_allows_operational_parameters(job_grid_file):
    args = _parse_with_config_sources(
        job_grid_file,
        cli_keys={"save_file_lists"},
        yaml_keys={"job_grid_file", "label", "log_level", "output_path"},
    )

    assert args["run_number"] == 7
    assert args["primary"] == "gamma"


@patch("simtools.applications.simulate_prod.Simulator")
@patch("simtools.application.definition.ApplicationDefinition.start")
def test_main_uses_explicit_application_definition(mock_application_start, mock_simulator_class):
    _mock_application_context(mock_application_start)
    mock_simulator_class.return_value = MagicMock()

    app.main()

    mock_application_start.assert_called_once_with()
    assert app.APPLICATION.setup_io_handler is False
    assert app.APPLICATION.post_parse == app._post_parse


@patch("simtools.applications.simulate_prod.Simulator")
@patch("simtools.application.definition.ApplicationDefinition.start")
def test_main_runs_simulator_and_reports(mock_application_start, mock_simulator_class):
    _mock_application_context(mock_application_start, label="myprod")
    mock_simulator = MagicMock()
    mock_simulator_class.return_value = mock_simulator

    app.main()

    mock_simulator_class.assert_called_once_with(label="myprod")
    mock_simulator.simulate.assert_called_once()
    mock_simulator.validate_simulations.assert_called_once()
    mock_simulator.report.assert_called_once()


@patch("simtools.applications.simulate_prod.write_data_to_file")
@patch("simtools.applications.simulate_prod.build_simulation_job_metadata")
@patch("simtools.applications.simulate_prod.Simulator")
@patch("simtools.application.definition.ApplicationDefinition.start")
def test_main_writes_job_metadata_to_grid_output_path(
    mock_application_start,
    mock_simulator_class,
    mock_build_metadata,
    mock_write_data,
):
    _mock_application_context(mock_application_start)
    mock_application_start.return_value.args["grid_output_path"] = Path("grid-output")
    mock_simulator = MagicMock()
    mock_simulator_class.return_value = mock_simulator
    mock_build_metadata.return_value = {"runNumber": 7}

    app.main()

    mock_build_metadata.assert_called_once_with(
        mock_application_start.return_value.args, mock_simulator
    )
    mock_simulator.pack_for_register.assert_called_once_with(Path("grid-output"))
    mock_write_data.assert_called_once_with(
        {"runNumber": 7}, Path("grid-output") / app._JOB_METADATA_FILE
    )


@patch("simtools.applications.simulate_prod.write_data_to_file")
@patch("simtools.applications.simulate_prod.Simulator")
@patch("simtools.application.definition.ApplicationDefinition.start")
def test_main_does_not_write_job_metadata_when_validation_fails(
    mock_application_start, mock_simulator_class, mock_write_data
):
    _mock_application_context(mock_application_start)
    mock_application_start.return_value.args["grid_output_path"] = Path("grid-output")
    mock_simulator_class.return_value.validate_simulations.side_effect = RuntimeError(
        "invalid output"
    )

    with pytest.raises(RuntimeError, match="invalid output"):
        app.main()

    mock_write_data.assert_not_called()
