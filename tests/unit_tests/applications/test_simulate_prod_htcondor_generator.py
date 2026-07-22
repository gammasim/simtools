#!/usr/bin/python3

from types import SimpleNamespace
from unittest.mock import patch

import simtools.applications.simulate_prod_htcondor_generator as app
from simtools.configuration.commandline_parser import CommandLineParser


@patch(
    "simtools.applications.simulate_prod_htcondor_generator.htcondor_script_generator.generate_submission_script"
)
@patch("simtools.application.definition.ApplicationDefinition.start")
def test_main_uses_standard_build_application(
    mock_build_application, mock_generate_submission_script
):
    mock_build_application.return_value = SimpleNamespace(args={"output_path": "htcondor_submit"})

    app.main()

    assert mock_build_application.call_args.kwargs == {}
    mock_generate_submission_script.assert_called_once_with({"output_path": "htcondor_submit"})


def test_add_arguments_registers_job_grid_argument():
    parser = CommandLineParser()

    parser.add_argument_definitions(app._ARGUMENTS)
    args = parser.parse_args(
        [
            "--job_grid_file",
            "job_grid.ecsv",
        ]
    )

    assert args.job_grid_file == "job_grid.ecsv"
