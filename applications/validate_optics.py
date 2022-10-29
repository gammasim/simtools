#!/usr/bin/python3

"""
    Summary
    -------
    This application validates the optical model parameters through ray tracing simulations \
    of the whole telescope, assuming a point-like light source. The output includes PSF (D80), \
    effective mirror area and effective focal length as a function of the off-axis angle. \

    The telescope zenith angle and the source distance can be set by command line arguments.

    Examples of the plots generated by this applications are shown below. On the top, the D80 \
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
        Model version (default=prod4).
    src_distance (float, optional)
        Source distance in km (default=10).
    zenith (float, optional)
        Zenith angle in deg (default=20).
    max_offset (float, optional)
        Maximum offset angle in deg (default=4).
    plot_images (activation mode, optional)
        Produce a multiple pages pdf file with the image plots.
    test (activation mode, optional)
        If activated, application will be faster by simulating fewer photons.
    verbosity (str, optional)
        Log level to print (default=INFO).

    Example
    -------
    LST-1 Prod5

    Runtime about 1-2 min.

    .. code-block:: console

        python applications/validate_optics.py --site North --telescope LST-1 --max_offset 3.0
"""

import logging
import os

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

import simtools.configuration as configurator
import simtools.util.general as gen
from simtools import io_handler
from simtools.model.telescope_model import TelescopeModel
from simtools.ray_tracing import RayTracing

# from simtools.visualize import setStyle

# setStyle()


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

    label = os.path.basename(__file__).split(".")[0]
    args_dict, db_config = _parse(label)

    logger = logging.getLogger()
    logger.setLevel(gen.getLogLevelFromUser(args_dict["log_level"]))

    _io_handler = io_handler.IOHandler()
    outputDir = _io_handler.getOutputDirectory(label, dirType="application-plots")

    telModel = TelescopeModel(
        site=args_dict["site"],
        telescopeModelName=args_dict["telescope"],
        modelVersion=args_dict["model_version"],
        label=label,
        mongoDBConfig=db_config,
    )

    ######################################################################
    # This is here as an example how to change parameters when necessary.
    ######################################################################
    # parsToChange = {
    #     'mirror_focal_length': 1608.3,
    #     'mirror_offset': -177.5,
    #     'camera_body_diameter': 289.7,
    #     'telescope_transmission': 1
    # }
    # telModel.changeMultipleParameters(**parsToChange)

    print(
        "\nValidating telescope optics with ray tracing simulations"
        " for {}\n".format(telModel.name)
    )

    ray = RayTracing.fromKwargs(
        telescopeModel=telModel,
        simtelSourcePath=args_dict["simtelpath"],
        sourceDistance=args_dict["src_distance"] * u.km,
        zenithAngle=args_dict["zenith"] * u.deg,
        offAxisAngle=np.linspace(
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

        plotFileName = "_".join((label, telModel.name, key))
        plotFile = outputDir.joinpath(plotFileName)
        plt.savefig(str(plotFile) + ".pdf", format="pdf", bbox_inches="tight")
        plt.savefig(str(plotFile) + ".png", format="png", bbox_inches="tight")
        plt.clf()

    # Plotting images
    if args_dict["plot_images"]:
        plotFileName = "_".join((label, telModel.name, "images.pdf"))
        plotFile = outputDir.joinpath(plotFileName)
        pdfPages = PdfPages(plotFile)

        logger.info("Plotting images into {}".format(plotFile))

        for image in ray.images():
            fig = plt.figure(figsize=(8, 6), tight_layout=True)
            image.plotImage()
            pdfPages.savefig(fig)
            plt.clf()
        plt.close()
        pdfPages.close()


if __name__ == "__main__":
    main()
