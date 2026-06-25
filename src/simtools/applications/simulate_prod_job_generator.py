#!/usr/bin/python3

r"""
Generate simulation production jobs from an executable job grid.

This tool reads a pre-generated executable job grid and writes job files for a selected backend.
The ``htcondor`` backend generates HTCondor submit files. The ``script`` backend writes local
shell scripts and can optionally run a selected script.

Command line arguments
----------------------
backend (str, optional)
    Job backend to use: ``htcondor`` or ``script``. Defaults to ``htcondor``.
job_grid_file (str, required)
    Path to the pre-generated executable job grid file.
job_grid_line (int, optional)
    1-based normalized data-row number from the job grid to generate.
run_script (bool, optional)
    Run the generated script. Only valid for the ``script`` backend.

(all other command line arguments are identical to those of :ref:`simulate_prod` or backend
specific).

"""

from simtools.application_control import build_application
from simtools.job_execution import htcondor_script_generator, script_job_generator


def _add_arguments(parser):
    """Register application-specific command line arguments."""
    parser.add_argument(
        "--backend",
        help="Job backend to use.",
        type=str,
        choices=["htcondor", "script"],
        required=False,
        default="htcondor",
    )
    parser.add_argument(
        "--job_grid_file",
        help="Path to a pre-generated executable job grid file.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--job_grid_line",
        help="1-based normalized data-row number from the job grid to generate.",
        type=int,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--run_script",
        help="Run the generated script. Only valid for the script backend.",
        action="store_true",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--priority",
        help="HTCondor job priority.",
        type=int,
        required=False,
        default=1,
    )
    parser.add_argument(
        "--htcondor_log_path",
        help="Directory for HTCondor output files (default: output_path/htcondor_logs).",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--simulation_output",
        help="Output path for simulation data (default: ./simtools-output).",
        type=str,
        required=False,
        default="./simtools-output",
    )


def main():
    """See CLI description."""
    app_context = build_application()

    backend = app_context.args.get("backend", "htcondor")
    if backend == "htcondor":
        htcondor_script_generator.generate_submission_script(app_context.args)
    elif backend == "script":
        script_job_generator.generate_script_jobs(app_context.args)
    else:
        raise ValueError(f"Unsupported production job backend: {backend}")


if __name__ == "__main__":
    main()
