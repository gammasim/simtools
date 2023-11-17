#!/usr/bin/python3

"""
    Summary
    -------
    This application validate the camera efficiency by simulating it using \
    the testeff program provided by sim_telarray.

    The results of camera efficiency for Cherenkov (left) and NSB light (right) as a function\
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
        Model version (default='Current')
    verbosity (str, optional)
        Log level to print (default=INFO).

    Example
    -------
    MST-NectarCam - Prod5

    Runtime < 1 min.

    .. code-block:: console

        simtools-validate-camera-efficiency --site North \
            --telescope MST-NectarCam-D --model_version prod5

    The output is saved in simtools-output/validate_camera_efficiency.

    Expected final print-out message:

    .. code-block:: console

        INFO::validate_camera_efficiency(l118)::main::Plotted NSB efficiency in /workdir/external/\
        simtools/simtools-output/validate_camera_efficiency/application-plots/validate_camera\
        _efficiency_MST-NectarCam-D_nsb

    .. todo::

        * Change default model to default (after this feature is implemented in db_handler)
        * Fix the set_style. For some reason, sphinx cannot built docs with it on.
"""

import logging
from pathlib import Path

import simtools.utils.general as gen
from simtools.camera_efficiency import CameraEfficiency
from simtools.configuration import configurator
from simtools.io_operations import io_handler
from simtools.model.telescope_model import TelescopeModel


def _parse(label):
    """
    Parse command line configuration

    """
    config = configurator.Configurator(
        label=label,
        description=(
            "Calculate the camera efficiency of the telescope requested. "
            "Plot the camera efficiency vs wavelength for cherenkov and NSB light."
        ),
    )
    return config.initialize(db_config=True, telescope_model=True)


def main():
    label = Path(__file__).stem
    args_dict, _db_config = _parse(label)

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    # Output directory to save files related directly to this app
    _io_handler = io_handler.IOHandler()
    output_dir = _io_handler.get_output_directory(label, sub_dir="application-plots")

    tel_model = TelescopeModel(
        site=args_dict["site"],
        telescope_model_name=args_dict["telescope"],
        mongo_db_config=_db_config,
        model_version=args_dict["model_version"],
        label=label,
    )

    # For debugging purposes
    tel_model.export_config_file()

    logger.info(f"Validating the camera efficiency of {tel_model.name}")

    ce = CameraEfficiency(
        telescope_model=tel_model,
        simtel_source_path=args_dict["simtel_path"],
    )
    ce.simulate(force=True)
    ce.analyze(force=True)

    # Plotting the camera efficiency for Cherenkov light
    fig = ce.plot_cherenkov_efficiency()
    cherenkov_plot_file_name = label + "_" + tel_model.name + "_cherenkov"
    cherenkov_plot_file = output_dir.joinpath(cherenkov_plot_file_name)
    for f in ["pdf", "png"]:
        fig.savefig(str(cherenkov_plot_file) + "." + f, format=f, bbox_inches="tight")
    logger.info(f"Plotted cherenkov efficiency in {cherenkov_plot_file}")
    fig.clf()

    # Plotting the camera efficiency for NSB light
    fig = ce.plot_nsb_efficiency()
    nsb_plot_file_name = label + "_" + tel_model.name + "_nsb"
    nsb_plot_file = output_dir.joinpath(nsb_plot_file_name)
    for f in ["pdf", "png"]:
        fig.savefig(str(nsb_plot_file) + "." + f, format=f, bbox_inches="tight")
    logger.info(f"Plotted NSB efficiency in {nsb_plot_file}")
    fig.clf()


if __name__ == "__main__":
    main()
