#!/usr/bin/python3

r"""
Generate HTCondor submission files for parameter scans.

This application generates HTCondor submission files for parameter scans,
where simulate-prod is run with different overwrite YAML files for each
parameter combination.

Supports single and multi-parameter scans with cartesian product combinations.

Command line arguments
----------------------
scan_config (str, required)
    Path to YAML configuration file for parameter scan.

Example configuration
---------------------
.. code-block:: yaml

    simulation:
      site: North
      model_version: "6.0.1"
      # ... other simulate-prod parameters
      number_of_runs: 10

    parameter_scan:
      parameters:
        - path: changes.LSTN-01.asum_threshold
          name: threshold
          values: [220, 230, 240]
      overwrite_template: ./overwrite.yaml

    htcondor:
      apptainer_image: /path/to/simtools.sif
      output_path: ./htcondor_submit
      priority: 5

Example
-------
.. code-block:: console

    simtools-generate-parameter-scan-htcondor --scan_config scan_config.yaml
    cd htcondor_submit
    condor_submit simulate_prod_scan.condor

"""

from simtools.application_control import build_application
from simtools.job_execution import parameter_scan_generator


def _add_arguments(parser):
    """Register application-specific command line arguments."""
    parser.add_argument(
        "--scan_config",
        help="Path to YAML configuration file for parameter scan.",
        type=str,
        required=True,
    )


def main():
    """See CLI description."""
    app_context = build_application(
        initialization_kwargs={"db_config": False, "output": False},
    )

    parameter_scan_generator.generate_parameter_scan_htcondor(app_context.args["scan_config"])


if __name__ == "__main__":
    main()
