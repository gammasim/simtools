#!/usr/bin/python3

"""
    Generate run scripts for HT Condor job submission for simulation production.

    Run scripts are generated

    - to execute simulations using the `simtools-simulate-prod` application
    - use an apptainer image containing the SimPipe simulation software and tools
    - pack data nad histogram files write them in a specified directory

    This tool is intended to be used in a HT Condor environment.

"""

import logging

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.job_execution import htcondor_script_generator


def _parse(description=None):
    """
    Parse command line configuration.

    Parameters
    ----------
    description: str
        Application description.

    Returns
    -------
    CommandLineParser
        Command line parser object.

    """
    config = configurator.Configurator(description=description)
    config.parser.add_argument(
        "--apptainer_image",
        help="Apptainer image to use for the simulation (full path).",
        type=str,
        required=False,
    )
    config.parser.add_argument(
        "--priority",
        help="Job priority.",
        type=int,
        required=False,
        default=1,
    )
    return config.initialize(
        db_config=False,
        job_submission=False,
        simulation_model=["site", "layout", "telescope"],
        simulation_configuration={"software": None, "corsika_configuration": ["all"]},
    )


def main():  # noqa: D103
    args_dict, _ = _parse(description="Prepare simulations production for HT Condor job submission")

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    htcondor_script_generator.generate_submission_script(args_dict)


if __name__ == "__main__":
    main()
