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

        python applications/validate_camera_efficiency.py --site North \
            --telescope MST-NectarCam-D --model_version prod5

    .. todo::

        * Change default model to default (after this feature is implemented in db_handler)
        * Fix the setStyle. For some reason, sphinx cannot built docs with it on.
"""

import logging

import simtools.configuration as configurator
import simtools.io_handler as io
import simtools.util.general as gen
from simtools.camera_efficiency import CameraEfficiency
from simtools.model.telescope_model import TelescopeModel


def _parse():
    """
    Parse command line configuratio

    """
    config = configurator.Configurator(
        description=(
            "Calculate the camera efficiency of the telescope requested. "
            "Plot the camera efficiency vs wavelength for cherenkov and NSB light."
        )
    )
    config.parser.initialize_telescope_model_arguments()
    config.parser.initialize_default_arguments(add_workflow_config=False)

    return config.initialize(add_workflow_config=False)


def main():

    args_dict = _parse()
    label = "validate_camera_efficiency"

    logger = logging.getLogger()
    logger.setLevel(gen.getLogLevelFromUser(args_dict["log_level"]))

    # Output directory to save files related directly to this app
    outputDir = io.getOutputDirectory(args_dict["output_path"], label, dirType="application-plots")

    telModel = TelescopeModel(
        site=args_dict["site"],
        modelFilesLocations=args_dict["model_path"],
        filesLocation=args_dict["output_path"],
        telescopeModelName=args_dict["telescope"],
        modelVersion=args_dict["model_version"],
        label=label,
    )

    # For debugging purposes
    telModel.exportConfigFile()

    logger.info("Validating the camera efficiency of {}".format(telModel.name))

    ce = CameraEfficiency(
        telescopeModel=telModel,
        simtelSourcePath=args_dict["simtelpath"],
        filesLocation=args_dict["output_path"],
        dataLocation=args_dict["data_path"],
    )
    ce.simulate(force=True)
    ce.analyze(force=True)

    # Plotting the camera efficiency for Cherenkov light
    fig = ce.plotCherenkovEfficiency()
    cherenkovPlotFileName = label + "_" + telModel.name + "_cherenkov"
    cherenkovPlotFile = outputDir.joinpath(cherenkovPlotFileName)
    for f in ["pdf", "png"]:
        fig.savefig(str(cherenkovPlotFile) + "." + f, format=f, bbox_inches="tight")
    logger.info("Plotted cherenkov efficiency in {}".format(cherenkovPlotFile))
    fig.clf()

    # Plotting the camera efficiency for NSB light
    fig = ce.plotNSBEfficiency()
    nsbPlotFileName = label + "_" + telModel.name + "_nsb"
    nsbPlotFile = outputDir.joinpath(nsbPlotFileName)
    for f in ["pdf", "png"]:
        fig.savefig(str(nsbPlotFile) + "." + f, format=f, bbox_inches="tight")
    logger.info("Plotted NSB efficiency in {}".format(nsbPlotFile))
    fig.clf()


if __name__ == "__main__":
    main()
