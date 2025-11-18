#!/usr/bin/python3

r"""
    Derive mirror random reflection angle (mirror roughness) of a single mirror panel.

    Description
    -----------

    This application derives the value of the simulation model parameter
    *mirror_reflection_random_angle* using measurements of the focal length
    and point-spread function (PSF) of individual mirror panels.
    This parameter is sometimes referred to as the "mirror roughness".

    The derivation is performed through gradient descent optimization that minimizes the
    Root Mean Squared Deviation (RMSD) between measured and simulated cumulative PSF curves.

    PSF measurements should be provided as an ECSV file containing radial distance and
    cumulative (integral) PSF values using the ``--data`` argument.

    Mirror panels are simulated individually, using one of the following options to set the
    mirror panel focal length:

        * file (table) with measured focal lengths per mirror panel
          (provided through ``--mirror_list``)
        * randomly generated focal lengths using an expected spread (value given through
          ``--random_focal_length``) around the mean focal length (provided through the
          Model Parameters DB). This option is switched with ``--use_random_focal_length``.

    The starting values for the random reflection angle parameters are taken from the
    Model Parameters DB.

    Results of the optimization include:

        * Final PSF comparison plot showing measured vs simulated cumulative PSF
        * Optimized mirror_reflection_random_angle parameters
          (3-component: sigma1, fraction2, sigma2)
        * JSON model parameter file (if RMSD converges below threshold)

    Command line arguments
    ----------------------
    site (str, required)
        North or South.
    telescope (str, required)
        Telescope name (e.g. LSTN-01, SSTS-25)
    model_version (str, optional)
        Model version
    parameter_version (str, optional)
        Parameter version for model parameter file export.
    data (str, required)
        Results from PSF measurements for each mirror panel spot size (ECSV file).
    mirror_list (file, optional)
        Table with mirror ID and panel radius to replace the default one.
    use_random_focal_length (activation mode, optional)
        Use random focal lengths, instead of the measured ones. The argument random_focal_length
        can be used to replace the default random_focal_length from the model.
    random_focal_length (float, optional)
        Value of the random focal lengths to replace the default random_focal_length. Only used if
        'use_random_focal_length' is activated.
    random_focal_length_seed (int, optional)
        Seed for the random number generator used for focal length variation.
    threshold (float, optional)
        Convergence threshold for gradient descent (RMSD threshold, dimensionless). Default: 0.03
    learning_rate (float, optional)
        Initial learning rate for gradient descent. Default: 0.00001
    cleanup (activation mode, optional)
        Remove intermediate *.log and *.lis* files after optimization.
    test (activation mode, optional)
        If activated, application will be faster by simulating only few mirrors.

    Example
    -------
    Derive mirror random reflection angle for a large-sized telescope (LSTN-01),
    simulation production 6.0.2

    .. code-block:: console

        simtools-derive-mirror-rnda \\
            --site North \\
            --telescope LSTN-01 \\
            --model_version 6.0.2 \\
            --mirror_list tests/resources/mirror_list_CTA-N-LST1_v2019-03-31_rotated.ecsv \\
            --test --data tests/resources/PSFcurve_data_v2.ecsv

    Expected final print-out message:

    .. code-block:: console

        Optimization Results (RMSD-based):
        RMSD (full PSF curve): 0.022782

        mirror_reflection_random_angle [sigma1, fraction2, sigma2]
        Previous values = ['0.007500', '0.220000', '0.022000']
        Optimized values = ['0.009735', '0.213793', '0.002987']


"""

from simtools.application_control import get_application_label, startup_application
from simtools.configuration import configurator
from simtools.ray_tracing import psf_parameter_optimisation as psf_opt
from simtools.ray_tracing.mirror_panel_psf import MirrorPanelPSF


def _parse():
    """Parse command line configuration."""
    config = configurator.Configurator(
        description="Derive mirror random reflection angle.", label=get_application_label(__file__)
    )
    psf_group = config.parser.add_mutually_exclusive_group()
    psf_group.add_argument(
        "--data",
        help="Results from PSF measurements for each mirror panel spot size",
        type=str,
        required=False,
    )
    config.parser.add_argument(
        "--mirror_list",
        help=("Mirror list file to replace the default one."),
        type=str,
        required=False,
    )
    config.parser.add_argument(
        "--use_random_focal_length",
        help=("Use random focal lengths."),
        action="store_true",
        required=False,
    )
    config.parser.add_argument(
        "--random_focal_length",
        help=(
            "Value of the random focal length. Only used if 'use_random_focal_length' is activated."
        ),
        default=None,
        type=float,
        required=False,
    )
    config.parser.add_argument(
        "--random_focal_length_seed",
        help="Seed for the random number generator used for focal length variation.",
        type=int,
        required=False,
        default=None,
    )
    config.parser.add_argument(
        "--threshold",
        help="Convergence threshold for gradient descent (RMSD threshold, dimensionless).",
        type=float,
        required=False,
        default=0.03,
    )
    config.parser.add_argument(
        "--learning_rate",
        help="Initial learning rate for gradient descent.",
        type=float,
        required=False,
        default=0.00001,
    )
    config.parser.add_argument(
        "--cleanup",
        help="Remove intermediate *.log and *.lis* files after optimization.",
        action="store_true",
    )
    return config.initialize(
        db_config=True,
        output=True,
        simulation_model=["telescope", "model_version", "parameter_version"],
    )


def main():
    """Derive mirror random reflection angle of a single mirror panel."""
    app_context = startup_application(_parse)

    panel_psf = MirrorPanelPSF(
        app_context.args.get("label"), app_context.args, app_context.db_config
    )
    panel_psf.optimize_with_gradient_descent()
    panel_psf.print_results()

    # Only write JSON file if optimization converged below threshold
    threshold = app_context.args.get("threshold")
    tolerance = 0.0001
    if panel_psf.final_rmsd <= threshold + tolerance:
        panel_psf.write_optimization_data()
    else:
        app_context.logger.info(
            f"\nSkipping parameter export: "
            f"RMSD {panel_psf.final_rmsd:.6f} > threshold {threshold:.6f}"
        )

    # Cleanup intermediate files if requested
    if app_context.args.get("cleanup", False):
        psf_opt.cleanup_intermediate_files(app_context.io_handler.get_output_directory().parent)


if __name__ == "__main__":
    main()
