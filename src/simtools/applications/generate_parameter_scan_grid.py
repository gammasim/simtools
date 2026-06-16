#!/usr/bin/python3

r"""
Generate a parameter scan job grid by expanding a base production grid.

Reads a base job grid ECSV from ``simtools-production-generate-grid``, generates
one overwrite YAML file per scan parameter combination, and writes an expanded
``scan_grid.ecsv``. The resulting file is passed directly to
``simtools-simulate-prod-htcondor-generator``, which handles submit-file
rendering, image-label grouping, and log-directory setup.

Command line arguments
----------------------
job_grid_file (str, required)
    Base job grid ECSV file from ``simtools-production-generate-grid``.
scan_config (str, required)
    Path to parameter scan YAML configuration.
output_file (str, required)
    Output path for the expanded scan grid ECSV.

Scan configuration format
--------------------------
.. code-block:: yaml

    label: threshold_scan
    parameter_scan:
      overwrite:
        model_version: 7.0.0
        model_update: patch_update
        model_version_history: [7.0.0]
        description: Tune for NSB telescope trigger scan
        changes:
          LSTN-01:
            min_photons:
              version: 2.0.0
              value: 0
            min_photoelectrons:
              version: 2.0.0
              value: 0
          OBS-North:
            nsb_scaling_factor:
              version: 2.0.0
              value: 2
      parameters:
        - name: asum_threshold
          path: changes.LSTN-01.asum_threshold
          version: 2.0.0
          values: [220, 230, 240]

For each value, the scan generator creates an overwrite YAML file by copying the
``overwrite`` dictionary and setting the requested parameter path.

Example
-------
.. code-block:: console

    simtools-production-generate-grid --output_file base_grid.ecsv ...
    simtools-generate-parameter-scan-grid \
        --job_grid_file base_grid.ecsv \
        --scan_config scan.yaml \
        --output_file scan_grid.ecsv
    simtools-simulate-prod-htcondor-generator \
        --job_grid_file scan_grid.ecsv \
        --output_path htcondor_submit \
        --apptainer_image /path/to/simtools.sif

"""

from simtools.application_control import build_application
from simtools.job_execution import parameter_scan_generator


def _add_arguments(parser):
    """Register application-specific command line arguments."""
    parser.add_argument(
        "--job_grid_file",
        help="Base job grid ECSV file from simtools-production-generate-grid.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--scan_config",
        help="Path to parameter scan YAML configuration.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_file",
        help="Output path for the expanded scan grid ECSV.",
        type=str,
        required=True,
    )


def main():
    """See CLI description."""
    app_context = build_application(
        initialization_kwargs={"db_config": False, "output": False},
    )
    parameter_scan_generator.expand_job_grid_with_scan(
        app_context.args["job_grid_file"],
        app_context.args["scan_config"],
        app_context.args["output_file"],
    )


if __name__ == "__main__":
    main()
