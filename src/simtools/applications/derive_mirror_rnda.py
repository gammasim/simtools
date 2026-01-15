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
test (activation mode, optional)
    If activated, only optimize a small number of mirrors.
n_workers (int, optional)
    Number of parallel worker processes to use. Default: 0 (auto chooses maximum).
d80_hist (str, optional)
    If activated, write a histogram comparing measured vs simulated d80 distributions.
cleanup (activation mode, optional)
    If activated, remove intermediate files (patterns: ``*.log``, ``*.lis*``, ``*.dat``)
    from output.

Example
-------

.. code-block:: console

    simtools-derive-mirror-rnda \
        --site North \
        --telescope LSTN-01 \
        --model_version 7.0.0 \
        --data tests/resources/198mir_190925.ecsv \
        --test --d80_hist

Example log output
------------------

.. code-block:: text

    =====================================================================
    Single-Mirror d80 Optimization Results (Percentage Difference Metric)
    =====================================================================

    Number of mirrors optimized: 10
    Mean percentage difference: 1.58%

    Per-mirror results:
    ------------------------------------------------------------------------------------------
    Mirror   Meas d80    Sim d80   Pct Diff Optimized RNDA [sigma1, frac2, sigma2]
                (mm)       (mm)        (%) (deg, -, deg)
    ------------------------------------------------------------------------------------------
        1     14.285     14.350       0.45 [0.003827, 0.004825, 0.010393]
        2     15.275     15.446       1.12 [0.004148, 0.000000, 0.013305]
        3     15.195     15.267       0.48 [0.004201, 0.000000, 0.013502]
        4     14.270     14.041       1.60 [0.003881, 0.000000, 0.017040]
        5     13.695     13.390       2.23 [0.003695, 0.000000, 0.010499]
        6     14.950     15.475       3.51 [0.003980, 0.036984, 0.015252]
        7     13.770     13.463       2.23 [0.003612, 0.003628, 0.028229]
        8     13.375     12.950       3.18 [0.003543, 0.000000, 0.017134]
        9     15.200     15.334       0.88 [0.004123, 0.000000, 0.013361]
        10    15.145     15.120       0.17 [0.004249, 0.000000, 0.013110]
    ------------------------------------------------------------------------------------------

    mirror_reflection_random_angle [sigma1, fraction2, sigma2]
    Previous values = ['0.007500', '0.220000', '0.022000']
    Optimized values (averaged) = ['0.003926', '0.004544', '0.015182']
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
        "--mirror_list",
        help=(
            "Mirror list file to use (overrides the telescope model default). "
            "Useful for testing or for custom mirror layouts."
        ),
        type=str,
        required=False,
        default=None,
    )
    config.parser.add_argument(
        "--threshold",
        help="Convergence threshold for percentage difference (e.g. 0.05 for 5%%).",
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
        "--use_random_focal_length",
        action="store_true",
        default=False,
        help="Enable random variation of mirror-panel focal length (single-mirror mode).",
    )
    config.parser.add_argument(
        "--random_focal_length_seed",
        type=int,
        required=False,
        default=None,
        help="Seed for the random focal length generator.",
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
        hist_path = panel_psf.write_d80_histogram()
        if hist_path:
            print(f"d80 histogram written to: {hist_path}")

    if app_context.args.get("cleanup"):
        output_dir = Path(app_context.args.get("output_path", "."))
        cleanup_intermediate_files(output_dir)


if __name__ == "__main__":
    main()
