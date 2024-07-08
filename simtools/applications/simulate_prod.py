#!/usr/bin/python3

r"""
    Generate simulation configuration and run simulations (if required).

    Multipipe scripts will be produced as part of this application.
    Allows to run array layout simulation including shower and detector simulations

    The entire simulation chain, parts of it, or nothing is executed:

    - shower simulations with CORSIKA only
    - shower simulations with CORSIKA which are piped directly to sim_telarray using
      the sim_telarray multipipe mechanism.

    Command line arguments
    ----------------------
    model_version (str, required)
        The telescope model version to use (e.g., Prod5).
    site (str, required)
        North or South (case insensitive).
    primary (str, required)
        Name of the primary particle to simulate. The available options are
        gamma, gamma_diffuse, electron, proton, muon, helium, nitrogen, silicon, and iron.
    azimuth_angle (str or float, required)
        Telescope pointing direction in azimuth.
        It can be in degrees between 0 and 360 or one of north, south, east or west
        (case insensitive). Note that North is 0 degrees and the azimuth grows clockwise,
        so East is 90 degrees.
    zenith_angle (float, required)
        Zenith angle in degrees.
    nshow (int, optional)
        Number of showers to simulate.
        The Number of simulated events depends on the number of times a shower is re-used in the
        telescope simulation. The number provided here is before any reuse factors.
    start_run (int, required)
        Start run number such that the actual run number will be 'start_run' + 'run'.
        This is useful in case a new transform is submitted for the same production.
        It allows the transformation system to keep using sequential run numbers without repetition.
    run (int, required)
        Run number (actual run number will be 'start_run' + 'run').
    data_directory (str, optional)
        The location of the output directories corsika-data and simtel-data.
        the label is added to the data_directory, such that the output
        will be written to data_directory/label/simtel-data.
    pack_for_grid_register (bool, optional)
        Set whether to prepare a tarball for registering the output files on the grid.
        The files are written to the output_path/directory_for_grid_upload directory.
    log_level (str, optional)
        Log level to print.

    Example
    -------
    Run the application:

    .. code-block:: console

        simtools-simulate-prod \\
        --model_version Prod5 --site north --primary gamma --azimuth_angle north \\
        --zenith_angle 20 --start_run 0 --run 1

    By default the configuration is saved in simtools-output/test-production
    together with the actual simulation output in corsika-data and simtel-data within.
    The location of the latter directories can be set
    to a different location via the option --data_directory,
    but the label is always added to the data_directory, such that the output
    will be written to data_directory/label/simtel-data.
"""

import logging
import shutil
import tarfile
from pathlib import Path

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.simulator import Simulator


def _parse(description=None):
    """
    Parse command line configuration.

    Parameters
    ----------
    description: str
        Application description.

    Returns
    -------
    CommandLineParser
        Command line parser object.

    """
    config = configurator.Configurator(description=description)
    config.parser.add_argument(
        "--data_directory",
        help=(
            "The directory where to save the corsika-data and simtel-data output directories."
            "the label is added to the data_directory, such that the output"
            "will be written to data_directory/label/simtel-data."
        ),
        type=str.lower,
        required=False,
        default="./simtools-output/",
    )
    config.parser.add_argument(
        "--pack_for_grid_register",
        help="Set whether to prepare a tarball for registering the output files on the grid.",
        action="store_true",
        required=False,
        default=False,
    )
    return config.initialize(
        db_config=True,
        simulation_model=["site", "layout", "telescope"],
        simulation_configuration=["software", "corsika_configuration"],
    )


def pack_for_register(logger, simulator, args_dict):
    """
    Pack the output files for registering on the grid.

    Parameters
    ----------
    logger: logging.Logger
        Logger object.
    simulator: Simulator
        Simulator object.

    """
    logger.info("Packing the output files for registering on the grid")
    output_files = simulator.get_file_list(file_type="output")
    log_files = simulator.get_file_list(file_type="log")
    histogram_files = simulator.get_file_list(file_type="hist")
    tar_file_name = Path(log_files[0]).name.replace("log.gz", "log_hist.tar.gz")

    with tarfile.open(tar_file_name, "w:gz") as tar:
        files_to_tar = log_files[:1] + histogram_files[:1]
        for file_to_tar in files_to_tar:
            tar.add(file_to_tar, arcname=Path(file_to_tar).name)
    directory_for_grid_upload = Path(args_dict.get("output_path")).joinpath(
        "directory_for_grid_upload"
    )
    directory_for_grid_upload.mkdir(parents=True, exist_ok=True)
    for file_to_move in [*output_files, tar_file_name]:
        source_file = Path(file_to_move)
        destination_file = directory_for_grid_upload / source_file.name
        # Note that this will overwrite previous files which exist in the directory
        # It should be fine for normal production since each run is on a separate node
        # so no files are expected there.
        shutil.move(source_file, destination_file)
    logger.info(f"Output files for the grid placed in {directory_for_grid_upload!s}")


def main():  # noqa: D103
    args_dict, db_config = _parse(description="Run simulations for productions")

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    simulator = Simulator(
        label=args_dict.get("label"),
        args_dict=args_dict,
        submit_command="local",
        test=args_dict["test"],
        mongo_db_config=db_config,
    )

    simulator.simulate()

    logger.info(
        f"Production run is complete for primary {args_dict['primary']} showers "
        f"coming from {args_dict['azimuth_angle']} azimuth and zenith angle of "
        f"{args_dict['zenith_angle']} at the {args_dict['site']} site, "
        f"using the {args_dict['model_version']} simulation model."
    )

    if args_dict["pack_for_grid_register"]:
        pack_for_register(logger, simulator, args_dict)


if __name__ == "__main__":
    main()
