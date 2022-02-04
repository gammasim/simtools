#!/usr/bin/python3

"""
    Summary
    -------
    This application validate the camera efficiency by simulating it using \
    the testeff program provided by sim_telarray.

    The results of camera efficiency for Cherenkov (left) and NSB (right) light as a function\
    of wavelength are plotted. See examples below.

    .. _validate_camera_eff_plot:
    .. image:: images/validate_camera_efficiency_North-MST-NectarCam-D_cherenkov.png
      :width: 49 %
    .. image:: images/validate_camera_efficiency_North-MST-NectarCam-D_nsb.png
      :width: 49 %

    Command line arguments
    ----------------------
    site (str, required)
        North or South.
    telescope (str, required)
        Telescope model name (e.g. LST-1, SST-D, ...)
    model_version (str, optional)
        Model version (default=prod4)
    verbosity (str, optional)
        Log level to print (default=INFO).

    Example
    -------
    MST-NectarCam - Prod4

    Runtime < 1 min.

    .. code-block:: console

        python applications/validate_camera_efficiency.py --site North --telescope MST-NectarCam-D --model_version prod4

    .. todo::

        * Change default model to default (after this feature is implemented in db_handler)
        * Fix the setStyle. For some reason, sphinx cannot built docs with it on.
"""

import logging
import matplotlib.pyplot as plt
import argparse

import simtools.util.general as gen
import simtools.io_handler as io
import simtools.config as cfg
from simtools.model.telescope_model import TelescopeModel
from simtools.camera_efficiency import CameraEfficiency


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Calculate the camera efficiency of the telescope requested. "
            "Plot the camera efficiency vs wavelength for cherenkov and NSB light."
        )
    )
    parser.add_argument("-s", "--site", help="North or South", type=str, required=True)
    parser.add_argument(
        "-t",
        "--telescope",
        help="Telescope model name (e.g. LST-1, SST-D)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-m",
        "--model_version",
        help="Model version (default=prod4)",
        type=str,
        default="prod4",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        dest="logLevel",
        action="store",
        default="info",
        help="Log level to print (default is INFO)",
    )

    args = parser.parse_args()
    label = "validate_camera_efficiency"

    logger = logging.getLogger()
    logger.setLevel(gen.getLogLevelFromUser(args.logLevel))

    # Output directory to save files related directly to this app
    outputDir = io.getApplicationOutputDirectory(cfg.get("outputLocation"), label)

    telModel = TelescopeModel(
        site=args.site,
        telescopeModelName=args.telescope,
        modelVersion=args.model_version,
        label=label,
    )

    # For debugging purposes
    telModel.exportConfigFile()

    logger.info("Validating the camera efficiency of {}".format(telModel.name))

    ce = CameraEfficiency(telescopeModel=telModel)
    ce.simulate(force=False)
    ce.analyze(force=True)

    # Plotting the camera efficiency for Cherenkov light
    plt = ce.plotCherenkovEfficiency()
    cherenkovPlotFileName = label + "_" + telModel.name + "_cherenkov"
    cherenkovPlotFile = outputDir.joinpath(cherenkovPlotFileName)
    for f in ["pdf", "png"]:
        plt.savefig(str(cherenkovPlotFile) + "." + f, format=f, bbox_inches="tight")
    logger.info("Plotted cherenkov efficiency in {}".format(cherenkovPlotFile))
    plt.clf()

    # Plotting the camera efficiency for NSB light
    plt = ce.plotNSBEfficiency()
    nsbPlotFileName = label + "_" + telModel.name + "_nsb"
    nsbPlotFile = outputDir.joinpath(nsbPlotFileName)
    for f in ["pdf", "png"]:
        plt.savefig(str(nsbPlotFile) + "." + f, format=f, bbox_inches="tight")
    logger.info("Plotted NSB efficiency in {}".format(nsbPlotFile))
    plt.clf()
