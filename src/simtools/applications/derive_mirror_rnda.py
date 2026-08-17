#!/usr/bin/python3

"""Derive mirror random reflection angle based on per-mirror PSF diameter optimization."""

from pathlib import Path

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.ray_tracing.mirror_panel_psf import MirrorPanelPSF
from simtools.ray_tracing.psf_parameter_optimisation import cleanup_intermediate_files

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "data", help="ECSV file with a PSF diameter column (mm) per mirror", type=str, required=True
    ),
    cli.ArgumentDefinition(
        "threshold",
        help="Convergence threshold for percentage difference.",
        type=float,
        required=False,
        default=0.05,
    ),
    cli.ArgumentDefinition(
        "learning_rate",
        help="Learning rate for gradient descent.",
        type=float,
        required=False,
        default=0.001,
    ),
    cli.ArgumentDefinition(
        "fraction",
        help="PSF containment fraction for diameter calculation (e.g., 0.8 for D80, 0.95 for D95).",
        type=float,
        default=0.8,
    ),
    cli.ArgumentDefinition(
        "max_workers",
        help="Number of parallel worker processes to use.",
        type=int,
        required=False,
        default=0,
    ),
    cli.ArgumentDefinition(
        "number_of_mirrors_to_test",
        help="Number of mirrors to optimize when --test is used.",
        type=int,
        required=False,
        default=10,
    ),
    cli.ArgumentDefinition(
        "profile_serial",
        action="store_true",
        default=False,
        help="Run optimization in a single process (no process pool).",
    ),
    cli.ArgumentDefinition(
        "psf_hist",
        nargs="?",
        const="psf_distributions.png",
        default=None,
        help=(
            "Write a histogram comparing measured vs simulated PSF diameter distributions. "
            "Optionally provide a filename (relative to output dir unless absolute)."
        ),
    ),
    cli.ArgumentDefinition(
        "cleanup",
        action="store_true",
        default=False,
        help=(
            "Remove intermediate files from the output directory (patterns: *.log, *.lis*, *.dat)."
        ),
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        cli.MODEL_VERSION,
        cli.PARAMETER_VERSION,
        cli.OVERWRITE_MODEL_PARAMETERS,
        cli.SITE,
        cli.TELESCOPE,
        *cli.OUTPUT_PATH_ARGUMENTS,
        *cli.OUTPUT_ARGUMENTS,
    ),
    database=True,
    initialize_output=True,
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()
    panel_psf = MirrorPanelPSF(app_context.args.get("label"), app_context.args)
    panel_psf.optimize_with_gradient_descent()
    panel_psf.write_optimization_data()
    if app_context.args.get("psf_hist"):
        panel_psf.write_psf_histogram()

    if app_context.args.get("cleanup"):
        output_dir = Path(app_context.args.get("output_path", "."))
        cleanup_intermediate_files(output_dir)


if __name__ == "__main__":
    main()
