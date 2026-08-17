#!/usr/bin/python3

"""Generate a run script and submit file for HT Condor job submission of a simulation production."""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.job_execution import htcondor_script_generator

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "job_grid_file",
        help="Path to a pre-generated executable job grid file.",
        type=str,
        required=True,
    ),
    cli.ArgumentDefinition("priority", help="Job priority.", type=int, required=False, default=1),
    cli.ArgumentDefinition(
        "htcondor_log_path",
        help="Directory for HTCondor output files (default: output_path/htcondor_logs).",
        type=str,
        required=False,
        default=None,
    ),
    cli.ArgumentDefinition(
        "simulation_output",
        help="Output path for simulation data (default: ./simtools-output).",
        type=str,
        required=False,
        default="./simtools-output",
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        *cli.OUTPUT_PATH_ARGUMENTS,
        *cli.SIM_TELARRAY_PATH_ARGUMENTS,
        *cli.CORSIKA_PATH_ARGUMENTS,
    ),
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()

    htcondor_script_generator.generate_submission_script(app_context.args)


if __name__ == "__main__":
    main()
