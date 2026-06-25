#!/usr/bin/python3

import argparse
from types import SimpleNamespace
from unittest.mock import patch

import simtools.applications.simulate_prod_job_generator as app


@patch(
    "simtools.applications.simulate_prod_job_generator.htcondor_script_generator."
    "generate_submission_script"
)
@patch("simtools.applications.simulate_prod_job_generator.build_application")
def test_main_dispatches_to_htcondor_backend(
    mock_build_application, mock_generate_submission_script
):
    mock_build_application.return_value = SimpleNamespace(
        args={"backend": "htcondor", "output_path": "htcondor_submit"}
    )

    app.main()

    mock_build_application.assert_called_once_with()
    mock_generate_submission_script.assert_called_once_with(
        {"backend": "htcondor", "output_path": "htcondor_submit"}
    )


@patch(
    "simtools.applications.simulate_prod_job_generator.script_job_generator.generate_script_jobs"
)
@patch("simtools.applications.simulate_prod_job_generator.build_application")
def test_main_dispatches_to_script_backend(mock_build_application, mock_generate_script_jobs):
    mock_build_application.return_value = SimpleNamespace(
        args={"backend": "script", "output_path": "scripts", "job_grid_line": 5}
    )

    app.main()

    mock_build_application.assert_called_once_with()
    mock_generate_script_jobs.assert_called_once_with(
        {"backend": "script", "output_path": "scripts", "job_grid_line": 5}
    )


def test_add_arguments_registers_generic_job_grid_arguments():
    parser = argparse.ArgumentParser()

    app._add_arguments(parser)
    args = parser.parse_args(
        [
            "--job_grid_file",
            "job_grid.ecsv",
            "--backend",
            "script",
            "--job_grid_line",
            "5",
            "--run_script",
        ]
    )

    assert args.job_grid_file == "job_grid.ecsv"
    assert args.backend == "script"
    assert args.job_grid_line == 5
    assert args.run_script is True
