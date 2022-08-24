#!/usr/bin/python3

"""
    Summary
    -------
    This application calculate the camera FoV of the telescope requested and plot the camera \
    as seen for an observer facing the camera.

    An example of the camera plot can be found below.

    .. _camera_fov_plot:
    .. image:: images/validate_camera_fov_North-LST-1_pixelLayout.png
      :width: 50 %


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
    LST - Prod4

    Runtime 1 min

    .. code-block:: console

        python applications/validate_camera_fov.py --site North \
            --telescope LST-1 --model_version prod4

    .. todo::

        * Change default model to default (after this feature is implemented in db_handler)
        * Fix the setStyle. For some reason, sphinx cannot built docs with it on.
"""

import logging

import simtools.config as cfg
import simtools.io_handler as io
import simtools.util.commandline_parser as argparser
import simtools.util.general as gen
from simtools.model.camera import Camera
from simtools.model.telescope_model import TelescopeModel

if __name__ == "__main__":

    parser = argparser.CommandLineParser(
        description=(
            "Calculate the camera FoV of the telescope requested. "
            "Plot the camera as well, as seen for an observer facing the camera."
        )
    )
    parser.initialize_telescope_model_arguments()
    parser.initialize_default_arguments(add_workflow_config=False)

    args = parser.parse_args()
    label = "validate_camera_fov"
    cfg.setConfigFileName(args.configFile)

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

    print("\nValidating the camera FoV of {}\n".format(telModel.name))

    cameraConfigFile = telModel.getParameterValue("camera_config_file")
    focalLength = float(telModel.getParameterValue("effective_focal_length"))
    camera = Camera(
        telescopeModelName=telModel.name,
        cameraConfigFile=cfg.findFile(cameraConfigFile),
        focalLength=focalLength,
    )

    fov, rEdgeAvg = camera.calcFOV()

    print("\nEffective focal length = " + "{0:.3f} cm".format(focalLength))
    print("{0} FoV = {1:.3f} deg".format(telModel.name, fov))
    print("Avg. edge radius = {0:.3f} cm\n".format(rEdgeAvg))

    # Now plot the camera as well
    fig = camera.plotPixelLayout()
    plotFileName = label + "_" + telModel.name + "_pixelLayout"
    plotFile = outputDir.joinpath(plotFileName)
    for f in ["pdf", "png"]:
        fig.savefig(str(plotFile) + "." + f, format=f, bbox_inches="tight")
    print("\nPlotted camera in {}\n".format(plotFile))
    fig.clf()
