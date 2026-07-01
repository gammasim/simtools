#!/usr/bin/python3

"""Tests for the simulate_prod application."""

import argparse
from unittest.mock import MagicMock, patch

import simtools.applications.simulate_prod as app


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


@patch("simtools.applications.simulate_prod.Simulator")
@patch("simtools.applications.simulate_prod.build_application")
def test_main_calls_build_application_with_job_grid_override_flag(
    mock_build_app, mock_simulator_class
):
    mock_context = MagicMock()
    mock_context.args = {
        "label": "test",
        "save_reduced_event_lists": False,
        "save_file_lists": False,
        "pack_for_grid_register": None,
    }
    mock_build_app.return_value = mock_context
    mock_simulator_class.return_value = MagicMock()

    app.main()

    call_kwargs = mock_build_app.call_args.kwargs
    assert call_kwargs["startup_kwargs"]["apply_job_grid_override"] is True
    assert call_kwargs["startup_kwargs"]["setup_io_handler"] is False


@patch("simtools.applications.simulate_prod.Simulator")
@patch("simtools.applications.simulate_prod.build_application")
def test_main_runs_simulator_and_reports(mock_build_app, mock_simulator_class):
    mock_context = MagicMock()
    mock_context.args = {
        "label": "myprod",
        "save_reduced_event_lists": False,
        "save_file_lists": False,
        "pack_for_grid_register": None,
    }
    mock_build_app.return_value = mock_context
    mock_simulator = MagicMock()
    mock_simulator_class.return_value = mock_simulator

    app.main()

    mock_simulator_class.assert_called_once_with(label="myprod")
    mock_simulator.simulate.assert_called_once()
    mock_simulator.validate_simulations.assert_called_once()
    mock_simulator.report.assert_called_once()
