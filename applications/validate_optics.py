#!/usr/bin/python3

"""
    Summary
    -------
    This application validates the optical model parameters through ray tracing simulations \
    of the whole telescope, assuming a point-like light source. The output includes PSF (D80), \
    effective mirror area and effective focal length as a function of the off-axis angle. \

    The telescope zenith angle and the source distance can be set by command line arguments.

    Examples of the plots generated by this application are shown below. On the top, the D80 \
    vs off-axis is shown in cm (left) and deg (right). On the bottom, the effective mirror \
    area (left) and the effective focal length (right) vs off-axis angle are shown.

    .. _validate_optics_plot:
    .. image::  images/validate_optics_North-LST-1_d80_cm.png
      :width: 49 %
    .. image::  images/validate_optics_North-LST-1_d80_deg.png
      :width: 49 %

    .. image::  images/validate_optics_North-LST-1_eff_area.png
      :width: 49 %
    .. image::  images/validate_optics_North-LST-1_eff_flen.png
      :width: 49 %


    Command line arguments
    ----------------------
    site (str, required)
        North or South.
    telescope (str, required)
        Telescope model name (e.g. LST-1, SST-D, ...).
    model_version (str, optional)
        Model version (default='Current').
    src_distance (float, optional)
        Source distance in km (default=10).
    zenith (float, optional)
        Zenith angle in deg (default=20).
    max_offset (float, optional)
        Maximum offset angle in deg (default=4).
    offset_steps (float, optional)
        Offset angle step size (default=0.25 deg)
    plot_images (activation mode, optional)
        Produce a multiple pages pdf file with the image plots.
    test (activation mode, optional)
        If activated, application will be faster by simulating fewer photons.
    verbosity (str, optional)
        Log level to print (default=INFO).

    Example
    -------
    LST-1 Prod5

    .. code-block:: console

        python applications/validate_optics.py --site North --telescope LST-1 --max_offset 1.0 \
        --zenith 20 --src_distance 11 --test

    The output is saved in simtools-output/validate_optics

    Expected final print-out message:

    .. code-block:: console

        INFO::ray_tracing(l434)::plot::Plotting eff_area vs off-axis angle
        INFO::ray_tracing(l434)::plot::Plotting eff_flen vs off-axis angle

"""

import logging
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

import simtools.util.general as gen
from simtools import io_handler
from simtools.configuration import configurator
from simtools.model.telescope_model import TelescopeModel
from simtools.ray_tracing import RayTracing

# from simtools.visualize import set_style

# set_style()


def _parse(label):
    """
    Parse command line configuratio

    """

    config = configurator.Configurator(
        label=label,
        description=(
            "Calculate and plot the PSF and effective mirror area as a function of off-axis angle "
            "of the telescope requested."
        ),
    )

    config.parser.add_argument(
        "--src_distance",
        help="Source distance in km (default=10)",
        type=float,
        default=10,
    )
    config.parser.add_argument(
        "--zenith", help="Zenith angle in deg (default=20)", type=float, default=20
    )
    config.parser.add_argument(
        "--max_offset",
        help="Maximum offset angle in deg (default=4)",
        type=float,
        default=4,
    )
    config.parser.add_argument(
        "--offset_steps",
        help="Offset angle step size (default=0.25 deg)",
        type=float,
        default=0.25,
    )
    config.parser.add_argument(
        "--plot_images",
        help="Produce a multiple pages pdf file with the image plots.",
        action="store_true",
    )
    return config.initialize(db_config=True, telescope_model=True)


def main():

    label = Path(__file__).stem
    args_dict, db_config = _parse(label)

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    _io_handler = io_handler.IOHandler()
    output_dir = _io_handler.get_output_directory(label, dir_type="application-plots")

    tel_model = TelescopeModel(
        site=args_dict["site"],
        telescope_model_name=args_dict["telescope"],
        model_version=args_dict["model_version"],
        label=label,
        mongo_db_config=db_config,
    )

    ######################################################################
    # This is here as an example how to change parameters when necessary.
    ######################################################################
    # pars_to_change = {
    #     'mirror_focal_length': 1608.3,
    #     'mirror_offset': -177.5,
    #     'camera_body_diameter': 289.7,
    #     'telescope_transmission': 1
    # }
    # tel_model.change_multiple_parameters(**pars_to_change)

    print(
        "\nValidating telescope optics with ray tracing simulations"
        " for {}\n".format(tel_model.name)
    )

    ray = RayTracing.from_kwargs(
        telescope_model=tel_model,
        simtel_source_path=args_dict["simtel_path"],
        source_distance=args_dict["src_distance"] * u.km,
        zenith_angle=args_dict["zenith"] * u.deg,
        off_axis_angle=np.linspace(
            0, args_dict["max_offset"], int(args_dict["max_offset"] / args_dict["offset_steps"]) + 1
        )
        * u.deg,
    )
    ray.simulate(test=args_dict["test"], force=False)
    ray.analyze(force=True)

    # Plotting
    for key in ["d80_deg", "d80_cm", "eff_area", "eff_flen"]:
        plt.figure(figsize=(8, 6), tight_layout=True)

        ray.plot(key, marker="o", linestyle=":", color="k")

        plot_file_name = "_".join((label, tel_model.name, key))
        plot_file = output_dir.joinpath(plot_file_name)
        plt.savefig(str(plot_file) + ".pdf", format="pdf", bbox_inches="tight")
        plt.savefig(str(plot_file) + ".png", format="png", bbox_inches="tight")
        plt.clf()

    # Plotting images
    if args_dict["plot_images"]:
        plot_file_name = "_".join((label, tel_model.name, "images.pdf"))
        plot_file = output_dir.joinpath(plot_file_name)
        pdf_pages = PdfPages(plot_file)

        logger.info("Plotting images into {}".format(plot_file))

        for image in ray.images():
            fig = plt.figure(figsize=(8, 6), tight_layout=True)
            image.plot_image()
            pdf_pages.savefig(fig)
            plt.clf()
        plt.close()
        pdf_pages.close()


if __name__ == "__main__":
    main()
