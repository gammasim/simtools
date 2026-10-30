"""Tests for the plot_tabular_data application (data file and model parameter modes)."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import simtools.applications.plot_tabular_data as app


def _context(**args):
    paths = Mock()
    paths.get_output_directory.return_value = Path("out")
    paths.get_output_file.side_effect = lambda name: Path("out") / name
    return SimpleNamespace(args=args, paths=paths, model_reader=Mock(), logger=Mock())


def test_parser_exposes_both_plot_sources():
    actions = {action.dest: action for action in app.APPLICATION.build_parser()._actions}

    assert {"plot_config", "parameter", "plot_type", "output_file"} <= set(actions)


def test_main_uses_model_parameter_mode(mocker):
    application = mocker.patch("simtools.applications.plot_tabular_data.APPLICATION")
    application.start.return_value = _context(parameter="atmospheric_profile")
    model_mode = mocker.patch("simtools.applications.plot_tabular_data._plot_model_parameter")
    data_mode = mocker.patch("simtools.applications.plot_tabular_data._plot_data_file")

    app.main()

    model_mode.assert_called_once()
    data_mode.assert_not_called()


def test_main_uses_data_file_mode(mocker):
    application = mocker.patch("simtools.applications.plot_tabular_data.APPLICATION")
    application.start.return_value = _context(plot_config="config.yml", parameter=None)
    model_mode = mocker.patch("simtools.applications.plot_tabular_data._plot_model_parameter")
    data_mode = mocker.patch("simtools.applications.plot_tabular_data._plot_data_file")

    app.main()

    data_mode.assert_called_once()
    model_mode.assert_not_called()


def test_data_file_mode_requires_output_file():
    with pytest.raises(ValueError, match="output_file"):
        app._plot_data_file(_context(plot_config="config.yml", output_file=None))


def test_model_parameter_mode_requires_plot_type():
    with pytest.raises(ValueError, match="plot_type"):
        app._plot_model_parameter(_context(parameter="fadc_pulse_shape", plot_type=None))
