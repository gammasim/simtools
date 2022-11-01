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
            --telescope LST-1 --model_version prod5

    .. todo::

        * Change default model to default (after this feature is implemented in db_handler)
        * Fix the setStyle. For some reason, sphinx cannot built docs with it on.
"""

import logging
from pathlib import Path

import simtools.configuration as configurator
import simtools.util.general as gen
from simtools import io_handler
from simtools.model.telescope_model import TelescopeModel


def main():

    config = configurator.Configurator(
        label=Path(__file__).stem,
        description=(
            "Calculate the camera FoV of the telescope requested. "
            "Plot the camera, as seen for an observer facing the camera."
        ),
    )
    config.parser.add_argument(
        "--cameraInSkyCoor",
        help=(
            "Plot the camera layout in sky coordinates "
            "(akin to looking at it from behind for single mirror telesecopes)"
        ),
        action="store_true",
        default=False,
    )
    config.parser.add_argument(
        "--printPixelsID",
        help=(
            "Up to which pixel ID to print (default: 50). "
            "To suppress printing of pixel IDs, set to zero (--printPixelsID 0). "
            "To print all pixels, set to 'All'."
        ),
        default=50,
    )

    args_dict, db_config = config.initialize(db_config=True, telescope_model=True)
    label = "validate_camera_fov"

    logger = logging.getLogger()
    logger.setLevel(gen.getLogLevelFromUser(args_dict["log_level"]))

    # Output directory to save files related directly to this app
    _io_handler = io_handler.IOHandler()
    outputDir = _io_handler.getOutputDirectory(label, dirType="application-plots")

    telModel = TelescopeModel(
        site=args_dict["site"],
        telescopeModelName=args_dict["telescope"],
        mongoDBConfig=db_config,
        modelVersion=args_dict["model_version"],
        label=label,
    )
    telModel.exportModelFiles()

    print("\nValidating the camera FoV of {}\n".format(telModel.name))

    focalLength = float(telModel.getParameterValue("effective_focal_length"))
    camera = telModel.camera

    fov, rEdgeAvg = camera.calcFOV()

    print("\nEffective focal length = " + "{0:.3f} cm".format(focalLength))
    print("{0} FoV = {1:.3f} deg".format(telModel.name, fov))
    print("Avg. edge radius = {0:.3f} cm\n".format(rEdgeAvg))

    # Now plot the camera as well
    try:
        pixelIDsToPrint = int(args_dict["printPixelsID"])
        if pixelIDsToPrint == 0:
            pixelIDsToPrint = -1  # so not print the zero pixel
    except ValueError:
        if args_dict["printPixelsID"].lower() == "all":
            pixelIDsToPrint = camera.getNumberOfPixels()
        else:
            raise ValueError(
                f"The value provided to --printPixelsID ({args_dict['printPixelsID']}) "
                "should be an integer or All"
            )
    fig = camera.plotPixelLayout(args_dict["cameraInSkyCoor"], pixelIDsToPrint)
    plotFilePrefix = outputDir.joinpath(f"{label}_{telModel.name}_pixelLayout")
    for suffix in ["pdf", "png"]:
        fileName = f"{str(plotFilePrefix)}.{suffix}"
        fig.savefig(fileName, format=suffix, bbox_inches="tight")
        print("\nSaved camera plot in {}\n".format(fileName))
    fig.clf()


if __name__ == "__main__":
    main()
