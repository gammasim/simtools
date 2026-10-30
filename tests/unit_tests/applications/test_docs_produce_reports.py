"""Tests for the docs_produce_reports application (one entry point per report type)."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import simtools.applications.docs_produce_reports as app


class _Recorder:
    """Fake report class recording the method calls made by the application."""

    calls = []

    def __init__(self, *args, **kwargs):
        self.args = args

    def __getattr__(self, name):
        def _call(*_, **__):
            _Recorder.calls.append(name)

        return _call


@pytest.fixture(name="recorder")
def recorder_fixture(monkeypatch):
    _Recorder.calls = []
    monkeypatch.setattr(app, "ReportGenerator", _Recorder)
    monkeypatch.setattr(app, "ReadParameters", _Recorder)
    return _Recorder


def _context(**args):
    return SimpleNamespace(args=args, model_reader=Mock(), logger=Mock())


def test_parser_lists_report_types():
    actions = {action.dest: action for action in app.APPLICATION.build_parser()._actions}

    assert actions["report"].choices == [
        "array_element",
        "calibration",
        "model_parameter",
        "simulation_configuration",
    ]


@pytest.mark.parametrize(
    ("producer", "args", "expected"),
    [
        (
            app._produce_array_element_reports,
            {"model_version": "6.0.2", "site": "North", "telescope": "LSTN-01"},
            "produce_array_element_report",
        ),
        (
            app._produce_array_element_reports,
            {"all_telescopes": True},
            "auto_generate_array_element_reports",
        ),
        (
            app._produce_calibration_reports,
            {"model_version": "6.0.2"},
            "auto_generate_calibration_reports",
        ),
        (
            app._produce_calibration_reports,
            {"all_model_versions": True},
            "auto_generate_calibration_reports",
        ),
        (
            app._produce_model_parameter_reports,
            {"site": "North", "telescope": "LSTN-01"},
            "produce_model_parameter_reports",
        ),
        (
            app._produce_model_parameter_reports,
            {"all_sites": True},
            "auto_generate_parameter_reports",
        ),
        (
            app._produce_simulation_configuration_reports,
            {"simulation_software": "corsika"},
            "auto_generate_simulation_configuration_reports",
        ),
    ],
)
def test_report_producers_call_matching_report_method(recorder, producer, args, expected):
    producer(_context(**args), Path("reports"))

    assert recorder.calls == [expected]
