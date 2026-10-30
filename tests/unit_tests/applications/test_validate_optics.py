"""Tests for the validate_optics application (optics and cumulative PSF types)."""

from types import SimpleNamespace

import simtools.applications.validate_optics as app


def test_parser_selects_validation_type():
    actions = {action.dest: action for action in app.APPLICATION.build_parser()._actions}

    assert actions["type"].choices == ["optics", "cumulative_psf"]
    assert actions["type"].default == "optics"


def test_main_runs_optics_validation_by_default(mocker):
    application = mocker.patch("simtools.applications.validate_optics.APPLICATION")
    application.start.return_value = SimpleNamespace(args={"type": "optics"})
    optics = mocker.patch("simtools.applications.validate_optics.validate_optics")
    cumulative = mocker.patch("simtools.applications.validate_optics.validate_cumulative_psf")

    app.main()

    optics.assert_called_once()
    cumulative.assert_not_called()


def test_main_runs_cumulative_psf_validation(mocker):
    application = mocker.patch("simtools.applications.validate_optics.APPLICATION")
    context = SimpleNamespace(args={"type": "cumulative_psf"})
    application.start.return_value = context
    optics = mocker.patch("simtools.applications.validate_optics.validate_optics")
    cumulative = mocker.patch("simtools.applications.validate_optics.validate_cumulative_psf")

    app.main()

    cumulative.assert_called_once_with(context)
    optics.assert_not_called()
