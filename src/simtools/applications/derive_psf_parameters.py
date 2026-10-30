#!/usr/bin/python3

"""Derives the mirror alignment parameters using cumulative PSF measurement."""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.ray_tracing import psf_parameter_optimisation as psf_opt

_ARGUMENTS = (
    cli.SOURCE_DISTANCE,
    cli.RAY_TRACING_ZENITH_ANGLE,
    cli.DATA,
    cli.MAX_WORKERS,
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
    cli.PSF_CONTAINMENT_FRACTION,
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
        cli.IGNORE_EXISTING_PARAMETER_VERSION,
        *cli.OUTPUT_PATH_ARGUMENTS,
        *cli.SIM_TELARRAY_PATH_ARGUMENTS,
        *cli.RAY_TRACING_ARGUMENTS,
    ),
    needs=("model", "corsika"),
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()

    job = psf_opt.PSFOptimizationJob(
        args=dict(app_context.args),
        output_dir=app_context.paths.get_output_directory(),
        context=app_context,
    )
    psf_opt.run_psf_optimization_job(job)


if __name__ == "__main__":
    main()
