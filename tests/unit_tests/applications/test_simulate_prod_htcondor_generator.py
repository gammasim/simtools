#!/usr/bin/python3

import argparse
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import simtools.applications.simulate_prod_htcondor_generator as app


@patch(
    "simtools.applications.simulate_prod_htcondor_generator.htcondor_script_generator.generate_submission_script"
)
@patch("simtools.applications.simulate_prod_htcondor_generator.build_application")
def test_main_uses_standard_build_application(
    mock_build_application, mock_generate_submission_script
):
    mock_build_application.return_value = SimpleNamespace(args={"output_path": "htcondor_submit"})

    app.main()

    assert mock_build_application.call_args.kwargs["initialization_kwargs"] == {
        "db_config": True,
        "preserve_by_version_keys": ["array_layout_name"],
        "simulation_model": ["site", "layout", "telescope", "model_version"],
        "simulation_configuration": {"software": None, "corsika_configuration": ["all"]},
    }
    mock_generate_submission_script.assert_called_once_with({"output_path": "htcondor_submit"})


def test_add_arguments_registers_nshow_scaling_arguments():
    parser = argparse.ArgumentParser()

    app._add_arguments(parser)
    args = parser.parse_args(
        [
            "--number_of_runs",
            "1",
            "--nshow_power_index",
            "-0.5",
            "--nshow_reference_energy",
            "100 GeV",
        ]
    )

    assert args.nshow_power_index == pytest.approx(-0.5)
    assert args.nshow_reference_energy == "100 GeV"


def test_add_arguments_registers_axis_defined_grid_arguments():
    parser = argparse.ArgumentParser()

    app._add_arguments(parser)
    args = parser.parse_args(
        [
            "--number_of_runs",
            "1",
            "--axes",
            "grid.yml",
            "--coordinate_system",
            "ra_dec",
            "--observing_time",
            "2017-09-16 00:00:00",
            "--lookup_table",
            "limits.ecsv",
            "--telescope_ids",
            "MSTN-15",
            "LSTN-01",
            "--simtel_file",
            "events.simtel.zst",
        ]
    )

    assert args.axes == "grid.yml"
    assert args.coordinate_system == "ra_dec"
    assert args.observing_time == "2017-09-16 00:00:00"
    assert args.lookup_table == "limits.ecsv"
    assert args.telescope_ids == ["MSTN-15", "LSTN-01"]
    assert args.simtel_file == "events.simtel.zst"
