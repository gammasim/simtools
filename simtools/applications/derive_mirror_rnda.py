#!/usr/bin/python3

"""
    Summary
    -------
    Derive the simulation model parameter **mirror_reflection_random_angle**
    (sometimes called mirror roughness) to match the measured containment diameter
    of the optical point-spread function (PSF) of individual mirror panels.

    Description
    -----------

    This application derives the value of the simulation model parameter
    *mirror_reflection_random_angle* using measurements of the focal length
    and PSF of individual mirror panels.

    PSF measurements are provided by one of the following options:

        * mean and sigma value obtained from the measurement of containment diameters of a number of
          mirror panels in cm (``--psf_measurement_containment_mean`` and
          ``--psf_measurement_containment_sigma``)
        * file (table) with measured PSF for each mirror panel spot size (``--psf_measurement``)

    The containment fraction used for the PSF diameter calculation is set through
    the argument ``--containment_fraction`` (typically 0.8 = 80%).

    Mirror panels are simulated individually, using one of the following options to set the
    mirror panel focal length:

        * file (table) with measured focal lengths per mirror panel
          (provided through ``--mirror_list``)
        * randomly generated focal lengths using an expected spread (value given through
          ``--random_flen``) around the mean focal length (provided through the
          :ref:`Model Parameters DB`). This option is switched with ``--use_random_flen``.

    The tuning algorithm requires a starting value for the random reflection angle. This is either
    taken from the :ref:`Model Parameters DB` (default) or can be set using the argument ``--rnda``.

    Ray-tracing simulations are performed for single mirror configurations for each
    mirror given in the mirror_list. The mean simulated containment diameter for all the mirrors
    is compared with the mean measured containment diameter. The algorithm defines a new value for
    the random reflection angle based on the sign of the difference between measured and simulated
    containment diameters and a new set of simulations is performed. This process is repeated
    until the sign of the difference changes, meaning that the two final values of the random
    reflection angle brackets the optimal. These two values are used to find the optimal one by
    a linear interpolation. Finally, simulations are performed by using the interpolated value,
    which is defined as the desired optimal.

    The option ``--no_tuning`` can be used if one only wants to simulate one value for the random
    reflection angle and compare the results with the measured ones.

    Results of the tuning are plotted. See examples of the PSF containment diameter
    D80 vs random reflection angle plot, on the left, and the D80 distributions
    (per mirror panel), on the right.

    .. _derive_rnda_plot:
    .. image:: images/derive_mirror_rnda_North-MST-FlashCam-D.png
      :width: 49 %
    .. image:: images/derive_mirror_rnda_North-MST-FlashCam-D_D80-distributions.png
      :width: 49 %

    This application uses the following :ref:`SimulationSoftware` tools:

        - sim_telarray/bin/sim_telarray
        - sim_telarray/bin/rx (optional)

    Command line arguments
    ----------------------
    telescope (str, required)
        Telescope name (e.g. North-LST-1, South-SST-D, ...)
    model_version (str, optional)
        Model version
    psf_measurement (str, optional)
        Table with results from PSF measurements for each mirror panel spot size
    psf_measurement_containment_mean (float, required)
        Mean of measured containment diameter [cm]
    psf_measurement_containment_sigma (float, optional)
        Std dev of measured containment diameter [cm]
    containment_fraction (float, required)
        Containment fraction for diameter calculation
    rnda (float, optional)
        Starting value of mirror_reflection_random_angle [deg]. If not given, the value from the
        default model is read from the simulation model database.
    mirror_list (file, optional)
        Table with mirror ID and panel radius.
    use_random_flen (activation mode, optional)
        Use random focal lengths, instead of the measured ones. The argument random_flen can be
        used to replace the default random_focal_length from the model.
    random_flen (float, optional)
        Value of the random focal lengths to replace the default random_focal_length. Only used if
         use_random_flen is activated.
    no_tuning (activation mode, optional)
        Turn off the tuning - A single case will be simulated and plotted.
    test (activation mode, optional)
        If activated, application will be faster by simulating only few mirrors.
    verbosity (str, optional)
        Log level to print.

    Example
    -------
    Derive mirror random reflection angle for a mid-sized telescope (MST),
    simulation production Prod5.

    .. code-block:: console

        simtools-derive-mirror-rnda \\
            --site North \\
            --telescope MST-FlashCam-D \\
            --containment_fraction 0.8 \\
            --mirror_list MLTdata-preproduction.ecsv \\
            --psf_measurement MLTdata-preproduction.ecsv \\
            --rnda 0.0063 \\
            --test

    Runtime about 4 min.

    The output is saved in `simtools-output/derive_mirror_rnda`.
    Use the parameter ``--output_path`` to change the output directory.

    Expected final print-out message:

    .. code-block:: console

        Measured D80:
        Mean = 1.403 cm, StdDev = 0.163 cm

        Simulated D80:
        Mean = 1.404 cm, StdDev = 0.608 cm

        mirror_random_reflection_angle
        Previous value = 0.006300
        New value = 0.004975

"""

import logging
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.table import QTable, Table

import simtools.data_model.model_data_writer as writer
import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.data_model.metadata_collector import MetadataCollector
from simtools.model.telescope_model import TelescopeModel
from simtools.ray_tracing import RayTracing


def _parse(label):
    """
    Parse command line configuration
    """

    config = configurator.Configurator(
        description="Derive mirror random reflection angle.", label=label
    )
    psf_group = config.parser.add_mutually_exclusive_group()
    psf_group.add_argument(
        "--psf_measurement_containment_mean",
        help="Mean of measured PSF containment diameter [cm]",
        type=float,
        required=False,
    )
    psf_group.add_argument(
        "--psf_measurement",
        help="Results from PSF measurements for each mirror panel spot size",
        type=str,
        required=False,
    )
    config.parser.add_argument(
        "--psf_measurement_containment_sigma",
        help="Std dev of measured PSF containment diameter [cm]",
        type=float,
        required=False,
    )
    config.parser.add_argument(
        "--containment_fraction",
        help="Containment fraction for diameter calculation (in interval 0,1)",
        type=config.parser.efficiency_interval,
        required=False,
        default=0.8,
    )
    config.parser.add_argument(
        "--rnda",
        help="Starting value of mirror_reflection_random_angle",
        type=float,
        required=False,
        default=0.0,
    )
    config.parser.add_argument(
        "--mirror_list",
        help=("Mirror list file to replace the default one."),
        type=str,
        required=False,
    )
    config.parser.add_argument(
        "--use_random_flen",
        help=("Use random focal lengths."),
        action="store_true",
        required=False,
    )
    config.parser.add_argument(
        "--random_flen",
        help="Value of the random focal length. Only used if use_random_flen is activated.",
        default=None,
        type=float,
        required=False,
    )
    config.parser.add_argument(
        "--no_tuning",
        help="no tuning of random_reflection_angle (a single case will be simulated).",
        action="store_true",
        required=False,
    )
    return config.initialize(db_config=True, output=True, telescope_model=True)


def _define_telescope_model(label, args_dict, db_config):
    """
    Define telescope model and update configuration
    with mirror list and/or random focal length given
    as input

    Attributes
    ----------
    label: str
        Application label.
    args_dict: dict
        Dictionary with configuration parameters.
    db_config:
        Dictionary with database configuration.

    Returns
    -------
    tel TelescopeModel
        telescope model

    """

    tel = TelescopeModel(
        site=args_dict["site"],
        telescope_model_name=args_dict["telescope"],
        model_version=args_dict["model_version"],
        mongo_db_config=db_config,
        label=label,
    )
    if args_dict["mirror_list"] is not None:
        mirror_list_file = gen.find_file(name=args_dict["mirror_list"], loc=args_dict["model_path"])
        tel.change_parameter("mirror_list", args_dict["mirror_list"])
        tel.add_parameter_file("mirror_list", mirror_list_file)
    if args_dict["random_flen"] is not None:
        tel.change_parameter("random_focal_length", str(args_dict["random_flen"]))

    return tel


def _print_and_write_results(
    args_dict, rnda_start, rnda_opt, mean_d80, sig_d80, results_rnda, results_mean, results_sig
):
    """
    Print results to screen and write metadata and data files
    in the requested format

    """

    containment_fraction_percent = int(args_dict["containment_fraction"] * 100)

    # Printing results to stdout
    print(f"\nMeasured D{containment_fraction_percent}:")
    if args_dict["psf_measurement_containment_sigma"] is not None:
        print(
            f"Mean = {args_dict['psf_measurement_containment_mean']:.3f} cm, "
            f"StdDev = {args_dict['psf_measurement_containment_sigma']:.3f} cm"
        )
    else:
        print(f"Mean = {args_dict['psf_measurement_containment_mean']:.3f} cm")
    print(f"\nSimulated D{containment_fraction_percent}:")
    print(f"Mean = {mean_d80:.3f} cm, StdDev = {sig_d80:.3f} cm")
    print("\nmirror_random_reflection_angle")
    print(f"Previous value = {rnda_start:.6f}")
    print(f"New value = {rnda_opt:.6f}\n")

    # Result table written to ecsv file using file_writer
    # First entry is always the best fit result
    result_table = QTable(
        [
            [True] + [False] * len(results_rnda),
            ([rnda_opt] + results_rnda) * u.deg,
            ([0.0] * (len(results_rnda) + 1)),
            ([0.0] * (len(results_rnda) + 1)) * u.deg,
            ([mean_d80] + results_mean) * u.cm,
            ([sig_d80] + results_sig) * u.cm,
        ],
        names=(
            "best_fit",
            "mirror_reflection_random_angle_sigma1",
            "mirror_reflection_random_angle_fraction2",
            "mirror_reflection_random_angle_sigma2",
            f"containment_radius_D{containment_fraction_percent}",
            f"containment_radius_sigma_D{containment_fraction_percent}",
        ),
    )
    writer.ModelDataWriter.dump(
        args_dict=args_dict,
        metadata=MetadataCollector(args_dict=args_dict).top_level_meta,
        product_data=result_table,
    )


def _get_psf_containment(logger, args_dict):
    """
    Read measured single-mirror point-spread function (containment)
    from file and return mean and sigma

    """

    # If this is a test, read just the first few lines since we only simulate those mirrors
    data_end = args_dict["number_of_mirrors_to_test"] + 1 if args_dict["test"] else None
    _psf_list = Table.read(args_dict["psf_measurement"], format="ascii.ecsv", data_end=data_end)
    try:
        args_dict["psf_measurement_containment_mean"] = np.nanmean(
            np.array(_psf_list["psf_opt"].to("cm").value)
        )
        args_dict["psf_measurement_containment_sigma"] = np.nanstd(
            np.array(_psf_list["psf_opt"].to("cm").value)
        )
    except KeyError:
        logger.debug(
            f"Missing column for psf measurement (psf_opt) in {args_dict['psf_measurement']}"
        )
        raise

    logger.info(
        f"Determined PSF containment to {args_dict['psf_measurement_containment_mean']:.4} "
        f"+- {args_dict['psf_measurement_containment_sigma']:.4} cm"
    )


def main():
    label = Path(__file__).stem

    args_dict, db_config = _parse(label)

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    tel = _define_telescope_model(label, args_dict, db_config)

    if args_dict["test"]:
        args_dict["number_of_mirrors_to_test"] = 4

    if args_dict["psf_measurement"]:
        _get_psf_containment(logger, args_dict)
    if not args_dict["psf_measurement_containment_mean"]:
        logger.error("Missing PSF measurement")
        raise ValueError

    def run(rnda):
        """Runs the simulations for one given value of rnda"""
        tel.change_parameter("mirror_reflection_random_angle", str(rnda))
        ray = RayTracing.from_kwargs(
            telescope_model=tel,
            single_mirror_mode=True,
            mirror_numbers=list(range(1, args_dict["number_of_mirrors_to_test"] + 1))
            if args_dict["test"]
            else "all",
            simtel_source_path=args_dict.get("simtel_path", None),
            use_random_focal_length=args_dict["use_random_flen"],
        )
        ray.simulate(test=False, force=True)  # force has to be True, always
        ray.analyze(force=True)

        return (
            ray.get_mean("d80_cm").to(u.cm).value,
            ray.get_std_dev("d80_cm").to(u.cm).value,
        )

    # First - rnda from previous model or from command line
    if args_dict["rnda"] != 0:
        rnda_start = args_dict["rnda"]
    else:
        rnda_start = tel.get_parameter("mirror_reflection_random_angle")["Value"]
        if isinstance(rnda_start, str):
            rnda_start = float(rnda_start.split()[0])

    logger.info(f"Start value for mirror_reflection_random_angle: {rnda_start} deg")

    results_rnda = []
    results_mean = []
    results_sig = []
    if args_dict["no_tuning"]:
        rnda_opt = rnda_start
    else:

        def collect_results(rnda, mean, sig):
            results_rnda.append(rnda)
            results_mean.append(mean)
            results_sig.append(sig)

        stop = False
        mean_d80, sig_d80 = run(rnda_start)
        rnda = rnda_start
        sign_delta = np.sign(mean_d80 - args_dict["psf_measurement_containment_mean"])
        collect_results(rnda, mean_d80, sig_d80)
        while not stop:
            rnda = rnda - (0.1 * rnda_start * sign_delta)
            if rnda < 0:
                rnda = 0
                collect_results(rnda, mean_d80, sig_d80)
                break
            mean_d80, sig_d80 = run(rnda)
            new_sign_delta = np.sign(mean_d80 - args_dict["psf_measurement_containment_mean"])
            stop = new_sign_delta != sign_delta
            sign_delta = new_sign_delta
            collect_results(rnda, mean_d80, sig_d80)

        # Linear interpolation using two last rnda values
        results_rnda, results_mean, results_sig = gen.sort_arrays(
            results_rnda, results_mean, results_sig
        )
        rnda_opt = np.interp(
            x=args_dict["psf_measurement_containment_mean"],
            xp=results_mean,
            fp=results_rnda,
        )

    mean_d80, sig_d80 = run(rnda_opt)

    _print_and_write_results(
        args_dict, rnda_start, rnda_opt, mean_d80, sig_d80, results_rnda, results_mean, results_sig
    )


if __name__ == "__main__":
    main()
