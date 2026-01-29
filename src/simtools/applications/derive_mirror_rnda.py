#!/usr/bin/python3

r"""Derive mirror random reflection angle based on per-mirror d80 optimization.

Description
-----------

This application derives the value of the simulation model parameter
*mirror_reflection_random_angle* using measurements of the d80 (spot size)
and focal length of individual mirror panels.

The optimization uses percentage difference as the metric::

    pct_diff = 100 * (simulated_d80 - measured_d80) / measured_d80

Each mirror is optimized individually, and the final RNDA is the average of all
per-mirror optimized values.

Command line arguments
----------------------

site (str, required)
    North or South.
telescope (str, required)
    Telescope name (e.g. LSTN-01, SSTS-25).
model_version (str, optional)
    Model version.
data (str, required)
    ECSV file with d80 (mm) and focal_length (m) columns per mirror.
threshold (float, optional)
    Convergence threshold for percentage difference (e.g. 0.05 for 5%).
    Default: 0.05.
learning_rate (float, optional)
    Learning rate for gradient descent. Default: 0.001.
test (optional)
    Only optimize a small number of mirrors.
n_workers (int, optional)
    Number of parallel worker processes to use. Default: 0 (auto chooses maximum).
d80_hist (str, optional)
    If activated, write a histogram comparing measured vs simulated d80 distributions.
cleanup (optional)
    Remove intermediate files (patterns: ``*.log``, ``*.lis*``, ``*.dat``)
    from output.

Example
-------

.. code-block:: console

    simtools-derive-mirror-rnda \
        --site North \
        --telescope LSTN-01 \
        --model_version 7.0.0 \
        --data tests/resources/MLTdata-preproduction.ecsv \
        --parameter_version 1.0.0 \
        --test --d80_hist --cleanup

"""

from pathlib import Path

from simtools.application_control import get_application_label, startup_application
from simtools.configuration import configurator
from simtools.ray_tracing.mirror_panel_psf import MirrorPanelPSF
from simtools.ray_tracing.psf_parameter_optimisation import cleanup_intermediate_files


def _parse():
    """Parse command line configuration."""
    config = configurator.Configurator(
        description="Derive mirror RNDA using per-mirror d80 optimization (percentage difference).",
        label=get_application_label(__file__),
    )
    config.parser.add_argument(
        "--data",
        help="ECSV file with d80 (mm) and focal_length (m) columns per mirror",
        type=str,
        required=True,
    )
    config.parser.add_argument(
        "--threshold",
        help="Convergence threshold for percentage difference.",
        type=float,
        required=False,
        default=0.05,
    )
    config.parser.add_argument(
        "--learning_rate",
        help="Learning rate for gradient descent.",
        type=float,
        required=False,
        default=0.001,
    )
    config.parser.add_argument(
        "--n_workers",
        help="Number of parallel worker processes to use.",
        type=int,
        required=False,
        default=0,
    )
    config.parser.add_argument(
        "--number_of_mirrors_to_test",
        help="Number of mirrors to optimize when --test is used.",
        type=int,
        required=False,
        default=10,
    )
    config.parser.add_argument(
        "--d80_hist",
        nargs="?",
        const="d80_distributions.png",
        default=None,
        help=(
            "Write a histogram comparing measured vs simulated d80 distributions. "
            "Optionally provide a filename (relative to output dir unless absolute)."
        ),
    )
    config.parser.add_argument(
        "--cleanup",
        action="store_true",
        default=False,
        help=(
            "Remove intermediate files from the output directory (patterns: *.log, *.lis*, *.dat)."
        ),
    )
    return config.initialize(
        db_config=True,
        output=True,
        simulation_model=["telescope", "model_version", "site", "parameter_version"],
    )


def main():
    """Derive mirror random reflection angle using per-mirror d80 optimization."""
    app_context = startup_application(_parse)
    panel_psf = MirrorPanelPSF(app_context.args.get("label"), app_context.args)
    panel_psf.optimize_with_gradient_descent()
    panel_psf.write_optimization_data()
    if app_context.args.get("d80_hist"):
        panel_psf.write_d80_histogram()

    if app_context.args.get("cleanup"):
        output_dir = Path(app_context.args.get("output_path", "."))
        cleanup_intermediate_files(output_dir)


if __name__ == "__main__":
    main()
