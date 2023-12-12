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
        Model version (default='Released')
    zenith_angle (float, optional)
        Zenith angle in degrees (between 0 and 180).
    azimuth_angle (float, optional)
        Telescope pointing direction in azimuth. It can be in degrees between 0 and 360 or
        one of north, south, east or west (case insensitive). Note that North is 0 degrees
        and the azimuth grows clockwise, so East is 90 degrees.
    nsb_spectrum (str, optional)
        File with NSB spectrum to use for the efficiency simulation.
        The expected format is two columns with wavelength in nm and
        NSB flux with the units: [1e9 * ph/m2/s/sr/nm].
        If the file has more than two columns, the first and third are used,
        and the second is ignored (native sim_telarray behaviour).
    verbosity (str, optional)
        Log level to print

    Example
    -------
    MST-NectarCam - Prod5

    Runtime < 1 min.

    .. code-block:: console

        simtools-validate-camera-efficiency --site North \
            --azimuth_angle 0 --zenith_angle 20 \
            --nsb_spectrum average_nsb_spectrum_CTAO-N_ze20_az0.txt \
            --telescope MST-NectarCam-D --model_version prod5

    The output is saved in simtools-output/validate_camera_efficiency.

    Expected final print-out message:

    .. code-block:: console

        INFO::validate_camera_efficiency(l118)::main::Plotted NSB efficiency in /workdir/external/\
        simtools/simtools-output/validate_camera_efficiency/application-plots/validate_camera\
        _efficiency_MST-NectarCam-D_nsb

"""

import logging
from pathlib import Path

import simtools.utils.general as gen
from simtools.camera_efficiency import CameraEfficiency
from simtools.configuration import configurator
from simtools.configuration.commandline_parser import CommandLineParser
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
    config.parser.add_argument(
        "--azimuth_angle",
        help=(
            "Telescope pointing direction in azimuth. "
            "It can be in degrees between 0 and 360 or one of north, south, east or west "
            "(case insensitive). Note that North is 0 degrees and "
            "the azimuth grows clockwise, so East is 90 degrees."
        ),
        type=CommandLineParser.azimuth_angle,
        default=0,
        required=False,
    )
    config.parser.add_argument(
        "--zenith_angle",
        help="Zenith angle in degrees (between 0 and 180).",
        type=CommandLineParser.zenith_angle,
        default=20,
        required=False,
    )
    config.parser.add_argument(
        "--nsb_spectrum",
        help=(
            "File with NSB spectrum to use for the efficiency simulation."
            "The expected format is two columns with wavelength in nm and "
            "NSB flux with the units: [1e9 * ph/m2/s/sr/nm]."
            "If the file has more than two columns, the first and third are used,"
            "and the second is ignored (native sim_telarray behaviour)."
        ),
        type=str,
        default=None,
        required=False,
    )
    _args_dict, _db_config = config.initialize(db_config=True, telescope_model=True)
    if _args_dict["site"] is None or _args_dict["telescope"] is None:
        config.parser.print_help()
        print("\n\nSite and telescope must be provided\n\n")
        raise RuntimeError("Site and telescope must be provided")
    return _args_dict, _db_config


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
        label=label,
        config_data={
            "zenith_angle": args_dict["zenith_angle"],
            "azimuth_angle": args_dict["azimuth_angle"],
            "nsb_spectrum": args_dict["nsb_spectrum"],
        },
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
