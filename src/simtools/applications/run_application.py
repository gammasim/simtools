#!/usr/bin/python3

"""Run several simtools applications using a configuration file."""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.runners.simtools_runner import run_applications

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "config_file", help="Application configuration.", type=str, required=True, default=None
    ),
    cli.ArgumentDefinition(
        "steps",
        type=int,
        nargs="+",
        help="List of steps to be execution (e.g., '--steps 7 8 9'; do not specify to run all).",
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(*_ARGUMENTS,),
    database=True,
    setup_io_handler=False,
    resolve_sim_software_executables=False,
    usage="simtools-run-application --config_file config_file_name",
)


def main():
    """Run several simtools applications using a configuration file."""
    app_context = APPLICATION.start()

    run_applications(app_context.args, run_time=app_context.run_time)


if __name__ == "__main__":
    main()
