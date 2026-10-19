#!/usr/bin/python3

"""Derives the mirror alignment parameters using cumulative PSF measurement."""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.ray_tracing import psf_parameter_optimisation as psf_opt

_ARGUMENTS = (
    cli.SOURCE_DISTANCE,
    cli.RAY_TRACING_ZENITH_ANGLE,
    cli.DATA,
    cli.ArgumentDefinition(
        "max_workers",
        type=int,
        default=None,
        help=(
            "Concurrent local evaluations (default: half the system CPUs, minimum one; "
            "zero or negative: all CPUs)."
        ),
    ),
    cli.ArgumentDefinition(
        "plot_all",
        help=(
            "On: plot cumulative PSF for all tested combinations, Off: plot it only for "
            "the best set of values"
        ),
        action="store_true",
    ),
    cli.ArgumentDefinition(
        "write_psf_parameters",
        help="Write the optimized PSF parameters as simulation model parameter files",
        action="store_true",
        required=False,
    ),
    cli.ArgumentDefinition(
        "rmsd_threshold",
        help=(
            "RMSD threshold for gradient descent convergence "
            "(not used with --monte_carlo_analysis)."
        ),
        type=float,
        default=0.01,
    ),
    cli.ArgumentDefinition(
        "learning_rate",
        help=(
            "Learning rate for gradient descent optimization "
            "(not used with --monte_carlo_analysis)."
        ),
        type=float,
        default=0.0001,
    ),
    cli.ArgumentDefinition(
        "max_iterations",
        help=(
            "Maximum number of gradient descent iterations (not used with --monte_carlo_analysis)."
        ),
        type=int,
        default=200,
    ),
    cli.ArgumentDefinition(
        "monte_carlo_analysis",
        help="Run analysis to find monte carlo uncertainties.",
        action="store_true",
    ),
    cli.ArgumentDefinition(
        "ks_statistic",
        help="Use KS statistic for monte carlo uncertainty analysis.",
        action="store_true",
    ),
    cli.ArgumentDefinition(
        "fraction",
        help="PSF containment fraction for diameter calculation (e.g., 0.8 for D80, 0.95 for D95).",
        type=float,
        default=0.8,
    ),
    cli.ArgumentDefinition(
        "cleanup",
        help="Remove intermediate *.log and *.lis* files after optimization.",
        action="store_true",
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    model_repository=True,
    arguments=(
        *_ARGUMENTS,
        *cli.BACKEND_ARGUMENTS,
        cli.MODEL_VERSION,
        cli.PARAMETER_VERSION,
        cli.OVERWRITE_MODEL_PARAMETERS,
        cli.SITE,
        cli.TELESCOPE,
        cli.DATA_SEARCH_PATH,
        *cli.OUTPUT_PATH_ARGUMENTS,
        *cli.SIM_TELARRAY_PATH_ARGUMENTS,
    ),
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()

    job = psf_opt.PSFOptimizationJob(
        args=dict(app_context.args),
        output_dir=app_context.io_handler.get_output_directory(),
    )
    psf_opt.run_psf_optimization_job(job)


if __name__ == "__main__":
    main()
