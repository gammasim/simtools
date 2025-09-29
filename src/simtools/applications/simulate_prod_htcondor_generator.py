#!/usr/bin/python3

r"""
Generate a run script and submit file for HT Condor job submission of a simulation production.

This tool facilitates the submission of multiple simulations to the HT Condor batch system,
enabling:

- Execution of simulations using the "simtools-simulate-prod" application.
- 'number_of_runs' jobs are submitted to the HT Condor batch system.
- Utilization of an Apptainer image containing the SimPipe simulation software and tools.
- Packaging of data and histogram files, and writing them to a specified directory.

This tool is intended for use in an HT Condor environment. Jobs run in a container universe
using the Apptainer image specified in the command line ('--apptainer_image'). Output is written
to the 'output_path' directory, with 'simtools-output' and 'logs' subdirectories.

Requirements for the 'simtools-simulate-prod-htcondor-generator' application:

- Availability of an Apptainer image for production (obtainable from the package registry on
  GitHub, e.g., via 'apptainer pull --force docker://ghcr.io/gammasim/simtools-<tag>:latest').
- Environment parameters required to run CORSIKA and sim_telarray, as well as DB access
  credentials.  These should be listed similarly to a '.env' file and copied to
  'output_path/env.txt'.  Ensure that the path to the simulation software is correctly set to
  'SIMTOOLS_SIMTEL_PATH=/workdir/sim_telarray'.

To submit jobs, change to the output directory and run:

.. code-block:: console

    condor_submit simulate_prod.submit

Simulation data products will be stored in the output directory.

Command line arguments
----------------------
output_path (str, required)
    Directory where the output and the simulation data files will be written.
apptainer_image (str, optional)
    Apptainer image to use for the simulation (full path).
priority (int, optional)
    Job priority (default: 1).

(all other command line arguments are identical to those of :ref:`simulate_prod`).

"""

from simtools.application_startup import get_application_label, startup_application
from simtools.configuration import configurator
from simtools.job_execution import htcondor_script_generator


def _parse():
    """Parse command line configuration."""
    config = configurator.Configurator(
        label=get_application_label(__file__),
        description="Prepare simulations production for HT Condor job submission",
    )
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
        simulation_model=["site", "layout", "telescope", "model_version"],
        simulation_configuration={"software": None, "corsika_configuration": ["all"]},
    )


def main():
    """Generate HT Condor submission script and submit file."""
    args_dict, _, _, _ = startup_application(_parse)

    htcondor_script_generator.generate_submission_script(args_dict)


if __name__ == "__main__":
    main()
