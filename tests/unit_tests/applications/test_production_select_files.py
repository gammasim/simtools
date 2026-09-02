"""Tests for the production-file selection application."""

import sys
from pathlib import Path

from simtools.applications import production_select_files
from simtools.configuration.commandline_parser import CommandLineParser


def test_application_parser_accepts_selection_options():
    parser = CommandLineParser()
    parser.add_argument_definitions(production_select_files.APPLICATION.all_arguments)

    args = parser.parse_args(
        [
            "--production_path",
            "production",
            "--select",
            "configuration.primary=gamma",
            "--file_type",
            "reduced_event_data",
            "--require_complete_runs",
        ]
    )

    assert args.production_path == "production"
    assert args.select == ["configuration.primary=gamma"]
    assert args.file_type == "reduced_event_data"
    assert args.require_complete_runs is True


def test_main_prints_selection_summary_and_writes_explicit_output(
    mocker, capsys, tmp_test_directory
):
    output_file = Path(tmp_test_directory) / "selected.yml"
    app_context = mocker.MagicMock()
    app_context.args = {
        "production_path": "production",
        "select": [],
        "file_type": "reduced_event_data",
        "require_complete_runs": False,
        "output_file": str(output_file),
        "output_file_from_default": False,
    }
    mocker.patch(
        "simtools.applications.production_select_files.APPLICATION"
    ).start.return_value = app_context
    mocker.patch(
        "simtools.applications.production_select_files.select_file_groups",
        return_value={"matching_jobs": 1},
    )
    mocker.patch(
        "simtools.applications.production_select_files.selection_summary",
        return_value="Matching jobs: 1",
    )
    mock_write = mocker.patch("simtools.applications.production_select_files.write_selection_file")

    production_select_files.main()

    assert capsys.readouterr().out == "Matching jobs: 1\n"
    mock_write.assert_called_once_with({"matching_jobs": 1}, str(output_file))


def test_application_parser_does_not_require_output_file(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["production_select_files.py", "--production_path", "production"],
    )

    args, _ = production_select_files.APPLICATION._parse()

    assert args["output_file"] is None
