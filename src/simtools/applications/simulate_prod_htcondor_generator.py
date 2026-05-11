#!/usr/bin/python3

r"""
Generate a run script and submit file for HT Condor job submission of a simulation production.

This tool generates HTCondor submission files for one or more simulation production grids and
supports either a single Apptainer image or a label-to-image mapping for multi-image submissions.
For each image label, it writes a dedicated ``simulate_prod.submit.<label>.condor`` file and a
matching ``simulate_prod.submit.<label>.params.txt`` file. When only one default image is used,
the unsuffixed ``simulate_prod.submit.condor`` and ``simulate_prod.submit.params.txt`` names are
kept.

HTCondor log files are written below ``htcondor_logs/`` by default, or below the directory given
with ``--htcondor_log_path``. The generator creates the ``log``, ``error``, and ``output``
subdirectories there and points the submit file at those locations.

The ``--simulation_output`` option controls the base directory that is passed to the simulation
production as ``pack_for_grid_register``. Each image label gets its own subdirectory under that
base path, allowing different grids or images to keep their output packages separate.

Requirements for the 'simtools-simulate-prod-htcondor-generator' application:

- Availability of an Apptainer image for production, either as a single path or as a mapping from
  labels to image paths (obtainable from the package registry on GitHub, e.g., via
  'apptainer pull --force docker://ghcr.io/gammasim/simtools-<tag>:latest').
- Environment parameters required to run CORSIKA and sim_telarray, as well as DB access
  credentials.  These should be listed similarly to a '.env' file and copied to
  'output_path/env.txt'.  Ensure that the path to the simulation software is correctly set to
  'SIMTOOLS_SIM_TELARRAY_PATH=/workdir/sim_telarray'.

To submit jobs, change to the output directory and run the generated submit file:

.. code-block:: console

    condor_submit simulate_prod.submit.<label>.condor

For the single-image default case, use ``condor_submit simulate_prod.submit.condor``.

Simulation data products are written to the directory controlled by ``--simulation_output``.

Command line arguments
----------------------
output_path (str, required)
    Directory where the HTCondor submission files are written.
apptainer_image (str or dict, required)
    Apptainer image to use for the simulation. A single string selects one image for all jobs;
    a dictionary maps labels to image paths and generates one submission pair per label.
htcondor_log_path (str, optional)
    Directory for HTCondor log files. Defaults to ``output_path/htcondor_logs``.
simulation_output (str, optional)
    Base directory for simulation output packages passed through as ``pack_for_grid_register``.
priority (int, optional)
    Job priority (default: 1).

(all other command line arguments are identical to those of :ref:`simulate_prod`).

"""

from simtools.application_control import build_application
from simtools.job_execution import htcondor_script_generator


def _add_arguments(parser):
    """Register application-specific command line arguments."""
    parser.add_argument(
        "--number_of_runs",
        help="Number of runs to be simulated.",
        type=int,
        required=True,
        default=1,
    )
    parser.add_argument(
        "--priority",
        help="Job priority.",
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
    parser.add_argument(
        "--corsika_limits",
        help="Path to an ECSV file with CORSIKA limits.",
        type=str,
        required=False,
        default=None,
    )


def main():
    """See CLI description."""
    app_context = build_application(
        initialization_kwargs={
            "db_config": False,
            "preserve_by_version_keys": ["array_layout_name"],
            "simulation_model": ["site", "layout", "telescope", "model_version"],
            "simulation_configuration": {"software": None, "corsika_configuration": ["all"]},
        },
    )

    htcondor_script_generator.generate_submission_script(app_context.args)


if __name__ == "__main__":
    main()
