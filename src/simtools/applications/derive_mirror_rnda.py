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
        --data tests/resources/198mir_190925.ecsv \
        --test --d80_hist

Example log output
------------------

.. code-block:: text

    ======================================================================
    Single-Mirror d80 Optimization Results (Percentage Difference Metric)
    ======================================================================

    Number of mirrors optimized: 10
    Mean percentage difference: 1.73%

    Per-mirror results:
    ------------------------------------------------------------------------------------------
    Mirror   Meas d80    Sim d80   Pct Diff Optimized RNDA [sigma1, frac2, sigma2]
                (mm)       (mm)        (%) (deg, -, deg)
    ------------------------------------------------------------------------------------------
        1     14.285     14.635       2.45 [0.0038, 0.0266, 0.0172]
        2     15.275     15.180       0.62 [0.0042, 0.0000, 0.0133]
        3     15.195     15.053       0.94 [0.0042, 0.0000, 0.0132]
        4     14.270     14.237       0.23 [0.0036, 0.0687, 0.0104]
        5     13.695     13.882       1.37 [0.0037, 0.0278, 0.0283]
        6     14.950     14.313       4.26 [0.0038, 0.0077, 0.0166]
        7     13.770     14.237       3.39 [0.0037, 0.0597, 0.0104]
        8     13.375     12.881       3.70 [0.0035, 0.0000, 0.0104]
        9     15.200     15.142       0.38 [0.0041, 0.0000, 0.0134]
       10     15.145     15.145       0.00 [0.0042, 0.0000, 0.0222]
    ------------------------------------------------------------------------------------------

    mirror_reflection_random_angle [sigma1, fraction2, sigma2]
    Previous values = ['0.0075', '0.2200', '0.0220']
    Optimized values (averaged) = ['0.0039', '0.0191', '0.0155']
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
