"""Tests for the docs_produce_reports application (one entry point per report type)."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import simtools.applications.docs_produce_reports as app


def _context(**args):
    paths = Mock()
    paths.get_output_directory.return_value = Path("reports")
    return SimpleNamespace(args=args, paths=paths, model_reader=Mock(), logger=Mock())


def test_parser_lists_report_types():
    actions = {action.dest: action for action in app.APPLICATION.build_parser()._actions}

    assert actions["report"].choices == [
        "array_element",
        "calibration",
        "model_parameter",
        "simulation_configuration",
    ]


def test_main_dispatches_calibration_report(mocker):
    application = mocker.patch("simtools.applications.docs_produce_reports.APPLICATION")
    application.start.return_value = _context(report="calibration", model_version="6.0.2")
    generator = mocker.patch("simtools.applications.docs_produce_reports.ReportGenerator")

    app.main()

    generator.return_value.auto_generate_calibration_reports.assert_called_once_with()


def test_main_dispatches_single_array_element_report(mocker):
    application = mocker.patch("simtools.applications.docs_produce_reports.APPLICATION")
    application.start.return_value = _context(
        report="array_element", model_version="6.0.2", site="North", telescope="LSTN-01"
    )
    read = mocker.patch("simtools.applications.docs_produce_reports.ReadParameters")
    generator = mocker.patch("simtools.applications.docs_produce_reports.ReportGenerator")

    app.main()

    read.return_value.produce_array_element_report.assert_called_once_with()
    generator.assert_not_called()


def test_main_dispatches_simulation_configuration_report(mocker):
    application = mocker.patch("simtools.applications.docs_produce_reports.APPLICATION")
    application.start.return_value = _context(
        report="simulation_configuration", simulation_software="corsika"
    )
    generator = mocker.patch("simtools.applications.docs_produce_reports.ReportGenerator")

    app.main()

    generator.return_value.auto_generate_simulation_configuration_reports.assert_called_once_with()
