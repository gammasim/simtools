"""Tests for the validate_optics application (optics and cumulative PSF types)."""

from types import SimpleNamespace

import pytest

import simtools.applications.validate_optics as app


def test_parser_selects_validation_type():
    actions = {action.dest: action for action in app.APPLICATION.build_parser()._actions}

    assert actions["type"].choices == ["optics", "cumulative_psf"]
    assert actions["type"].default == "optics"


@pytest.mark.parametrize(
    ("validation_type", "expected"),
    [("optics", "optics"), ("cumulative_psf", "cumulative_psf")],
)
def test_run_validation_selects_workflow_by_type(monkeypatch, validation_type, expected):
    calls = []
    monkeypatch.setattr(app, "validate_optics", lambda context: calls.append("optics"))
    monkeypatch.setattr(
        app, "validate_cumulative_psf", lambda context: calls.append("cumulative_psf")
    )

    app._run_validation(SimpleNamespace(args={"type": validation_type}))

    assert calls == [expected]
