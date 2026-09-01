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
        [
            row,
            {
                **row,
                "run_number": 11,
                "zenith_angle": 40 * u.deg,
                "array_layout_name": "CTAO-North-Beta",
            },
        ],
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
    assert args.wait is False


def test_application_parser_includes_show_options():
    parser = app.APPLICATION.build_parser()
    actions = {action.dest: action for action in parser._actions}

    assert "show_options" in actions


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


@pytest.mark.parametrize(
    ("row_args", "expected"),
    [
        (
            (),
            {
                "run_number": 7,
                "primary": "gamma",
                "site": "North",
                "ha": 123 * u.deg,
                "dec": -45 * u.deg,
            },
        ),
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


@pytest.mark.parametrize("backend", ["local", "htcondor"])
def test_parse_job_grid_reads_array_layout_from_selected_row(
    monkeypatch, job_grid_file, tmp_test_directory, backend
):
    """Grid execution uses the selected row's layout without a global fallback."""
    args = _parse_with_args(
        monkeypatch,
        _job_grid_args(
            job_grid_file,
            "--backend",
            backend,
            "--job_grid_row",
            2,
            "--output_path",
            tmp_test_directory,
        ),
    )

    if backend == "local":
        assert args["array_layout_name"] == "CTAO-North-Beta"
    else:
        assert args.get("array_layout_name") is None
        assert len(args["_job_grid_rows"]) == 1
        assert args["_job_grid_rows"][0]["array_layout_name"] == "CTAO-North-Beta"


def test_parse_job_grid_preserves_layout_per_row(monkeypatch, job_grid_file, tmp_test_directory):
    args = _parse_with_args(
        monkeypatch,
        _job_grid_args(
            job_grid_file,
            "--backend",
            "htcondor",
            "--output_path",
            tmp_test_directory,
        ),
    )

    assert args.get("array_layout_name") is None
    assert [row["array_layout_name"] for row in args["_job_grid_rows"]] == [
        "CTAO-North-Alpha",
        "CTAO-North-Beta",
    ]
    assert args["_defer_simulation_dependency_validation"] is True


def test_parse_local_job_grid_keeps_dependency_validation_local(
    monkeypatch, job_grid_file, tmp_test_directory
):
    args = _parse_with_args(
        monkeypatch,
        _job_grid_args(job_grid_file, "--output_path", tmp_test_directory),
    )

    assert args["_defer_simulation_dependency_validation"] is False


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


def test_parse_without_job_grid_requires_array_layout(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _parse_with_args(monkeypatch, ["--simulation_software", "sim_telarray"])

    assert "--array_layout_name" in capsys.readouterr().err


def test_parse_rejects_array_element_list(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _parse_with_args(
            monkeypatch,
            [
                "--simulation_software",
                "sim_telarray",
                "--array_element_list",
                "MSTN-01",
            ],
        )

    assert "unrecognized arguments: --array_element_list" in capsys.readouterr().err


def test_parse_rejects_telescope(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _parse_with_args(
            monkeypatch,
            ["--simulation_software", "sim_telarray", "--telescope", "LSTN-01"],
        )

    assert "unrecognized arguments: --telescope" in capsys.readouterr().err


def test_parser_excludes_redundant_telescope_argument():
    actions = {action.dest for action in app.APPLICATION.build_parser()._actions}

    assert "array_layout_name" in actions
    assert "telescope" not in actions


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


@pytest.mark.parametrize("wait", [False, True])
def test_execute_job_grid_submits_or_waits(mocker, tmp_test_directory, wait):
    """HTCondor grid jobs detach unless explicitly asked to wait."""
    job_specs = [MagicMock()]
    mocker.patch(
        "simtools.applications.simulate_prod.build_simulate_prod_job_specs", return_value=job_specs
    )
    mocker.patch("simtools.applications.simulate_prod.options_from_args")
    execute = mocker.patch("simtools.applications.simulate_prod.execute_jobs")
    submit = mocker.patch("simtools.applications.simulate_prod.submit_jobs")

    app._execute_job_grid(
        {
            "_job_grid_rows": [{}],
            "_job_grid_metadata": {},
            "output_path": tmp_test_directory,
            "wait": wait,
        }
    )

    expected = execute if wait else submit
    expected.assert_called_once()
    (submit if wait else execute).assert_not_called()


@patch("simtools.applications.simulate_prod.Simulator")
@patch("simtools.application.definition.ApplicationDefinition.start")
def test_main_uses_explicit_application_definition(mock_application_start, mock_simulator_class):
    _mock_application_context(mock_application_start)
    mock_simulator_class.return_value = MagicMock()

    app.main()

    mock_application_start.assert_called_once_with()
    assert app.APPLICATION.setup_io_handler is False
    assert app.APPLICATION.validate_simulation_dependencies is True
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
@patch("simtools.applications.simulate_prod.check_manifest")
@patch("simtools.applications.simulate_prod.validate_required_production_outputs")
@patch("simtools.applications.simulate_prod.build_production_job_manifest")
@patch("simtools.applications.simulate_prod.Simulator")
@patch("simtools.application.definition.ApplicationDefinition.start")
def test_main_writes_job_metadata_to_grid_output_path(
    mock_application_start,
    mock_simulator_class,
    mock_build_metadata,
    mock_validate_outputs,
    mock_check_manifest,
    mock_write_data,
):
    _mock_application_context(mock_application_start)
    mock_application_start.return_value.args["grid_output_path"] = Path("grid-output")
    mock_simulator = MagicMock()
    mock_simulator_class.return_value = mock_simulator
    manifest = {
        "configuration": {
            "run_number": 7,
            "simulation_software": "corsika_sim_telarray",
        },
        "files": {"sim_telarray": ["gamma.simtel.zst"]},
    }
    mock_build_metadata.return_value = manifest

    app.main()

    mock_build_metadata.assert_called_once_with(
        mock_application_start.return_value.args,
        mock_simulator,
        Path("grid-output"),
    )
    mock_simulator.pack_for_register.assert_called_once_with(Path("grid-output"))
    mock_validate_outputs.assert_called_once_with(
        manifest["files"],
        "corsika_sim_telarray",
        Path("grid-output"),
    )
    assert mock_check_manifest.call_args.args[0].data == manifest
    mock_write_data.assert_called_once_with(manifest, Path("grid-output") / app._JOB_METADATA_FILE)


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
