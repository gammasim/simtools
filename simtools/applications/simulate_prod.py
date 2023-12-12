#!/usr/bin/python3

"""
    Summary
    -------
    This application is used to run simulations for productions (typically on the grid).
    It allows to run a Paranal (CTAO-South) or La Palma (CTAO-North) array layout simulation
    with the provided "prod_tag" simulation configuration (e.g., Prod6)
    for a given primary particle, azimuth, and zenith angle.

    The entire simulation chain is performed, i.e., shower simulations with CORSIKA
    which are piped directly to sim_telarray using the sim_telarray multipipe mechanism.
    This script produces all the necessary configuration files for CORSIKA and
    sim_telarray before running simulation.
    The multipipe scripts will be produced as part of this script.

    This script does not provide a mechanism to submit jobs to a batch system like others
    in simtools since it is meant to be executed on a grid node
    (distributed to it by the workload management system).

    Command line arguments
    ----------------------
    production_config (str, Path, required)
        Simulation configuration file
        (contains the default setup which can be overwritten by the command line options).
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
        will be written to `data_directory/label/simtel-data`.
    pack_for_grid_register (bool, optional)
        Set whether to prepare a tarball for registering the output files on the grid.
        The files are written to the `output_path/directory_for_grid_upload` directory.
    log_level (str, optional)
        Log level to print.

    Example
    -------
    Run the application:

    .. code-block:: console

        simtools-simulate-prod \
        --production_config tests/resources/prod_multi_config_test.yml --model_version Prod5 \
        --site north --primary gamma --azimuth_angle north --zenith_angle 20 \
        --start_run 0 --run 1

    By default the configuration is saved in simtools-output/test-production
    together with the actual simulation output in corsika-data and simtel-data within.
    The location of the latter directories can be set
    to a different location via the option --data_directory,
    but the label is always added to the data_directory, such that the output
    will be written to `data_directory/label/simtel-data`.

    Expected final print-out message:

    .. code-block:: console

        INFO::array_layout(l569)::read_telescope_list_file::Reading array elements from ...
        WARNING::corsika_runner(l127)::_load_corsika_config_data::data_directory not given
        in corsika_config - default output directory will be set.
        INFO::array_layout(l569)::read_telescope_list_file::Reading array elements from ...
        INFO::corsika_config(l493)::_set_output_file_and_directory::Creating directory
        INFO::simulator(l405)::simulate::Submission command: local
        INFO::simulator(l410)::simulate::Starting submission for 1 run
        INFO::array_model(l315)::export_simtel_array_config_file::Writing array config file into
        INFO::job_manager(l95)::submit::Submitting script
        INFO::job_manager(l96)::submit::Job output stream
        INFO::job_manager(l97)::submit::Job error stream
        INFO::job_manager(l98)::submit::Job log stream
        INFO::job_manager(l119)::_submit_local::Running script locally
"""

import logging
import shutil
import tarfile
from pathlib import Path

from astropy.io.misc import yaml

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.configuration.commandline_parser import CommandLineParser
from simtools.simulator import Simulator


def _parse(description=None):
    """
    Parse the command line configuration.

    Parameters
    ----------
    description: str
        Description of the application.

    Returns
    -------
    CommandLineParser
        Command line parser object.

    """

    config = configurator.Configurator(description=description)
    config.parser.add_argument(
        "--production_config",
        help=(
            "Simulation configuration file "
            "(contains the default setup which can be overwritten by the command line options)"
        ),
        type=str,
        required=True,
    )
    config.parser.add_argument(
        "--primary",
        help="Primary particle to simulate.",
        type=str.lower,
        required=True,
        choices=[
            "gamma",
            "gamma_diffuse",
            "electron",
            "proton",
            "muon",
            "helium",
            "nitrogen",
            "silicon",
            "iron",
        ],
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
        required=True,
    )
    config.parser.add_argument(
        "--zenith_angle",
        help="Zenith angle in degrees (between 0 and 180).",
        type=CommandLineParser.zenith_angle,
        required=True,
    )
    config.parser.add_argument(
        "--nshow",
        help="Number of showers to simulate.",
        type=int,
        required=False,
    )
    config.parser.add_argument(
        "--start_run",
        help=(
            "Start run number such that the actual run number will be 'start_run' + 'run'. "
            "This is useful in case a new transform is submitted for the same production. "
            "It allows the transformation system to keep using sequential run numbers without "
            "repetition."
        ),
        type=int,
        required=True,
    )
    config.parser.add_argument(
        "--run",
        help="Run number (actual run number will be 'start_run' + 'run')",
        type=int,
        required=True,
    )
    config.parser.add_argument(
        "--data_directory",
        help=(
            "The directory where to save the corsika-data and simtel-data output directories."
            "the label is added to the data_directory, such that the output"
            "will be written to `data_directory/label/simtel-data`."
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
    return config.initialize(db_config=True, telescope_model=True)


def main():
    args_dict, db_config = _parse(description="Run simulations for productions")

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    try:
        with open(args_dict["production_config"], encoding="utf-8") as file:
            config_data = yaml.load(file)
    except FileNotFoundError:
        logger.error(
            f"Error loading simulation configuration file from {args_dict['production_config']}"
        )
        raise

    # Overwrite default and optional settings
    config_data["showers"]["run_list"] = args_dict["run"] + args_dict["start_run"]
    config_data["showers"]["primary"] = args_dict["primary"]
    config_data["common"]["site"] = args_dict["site"]
    config_data["common"]["zenith"] = args_dict["zenith_angle"]
    config_data["common"]["phi"] = args_dict["azimuth_angle"]
    label = config_data["common"].pop("label", "test-production")
    config_data["common"]["data_directory"] = Path(args_dict["data_directory"]) / label

    if args_dict["nshow"] is not None:
        config_data["showers"]["nshow"] = args_dict["nshow"]
    if args_dict["label"] is not None:
        label = args_dict["label"]

    simulator = Simulator(
        label=label,
        simulator="corsika_simtel",
        simulator_source_path=args_dict["simtel_path"],
        config_data=config_data,
        submit_command="local",
        test=args_dict["test"],
        mongo_db_config=db_config,
    )

    simulator.simulate()

    logger.info(
        f"Production run is complete for primary {config_data['showers']['primary']} showers "
        f"coming from {config_data['common']['phi']} azimuth and zenith angle of "
        f"{config_data['common']['zenith']} at the {args_dict['site']} site, "
        f"using the {config_data['array']['model_version']} telescope model."
    )

    if args_dict["pack_for_grid_register"]:
        logger.info("Packing the output files for registering on the grid")
        output_files = simulator.get_list_of_output_files()
        log_files = simulator.get_list_of_log_files()
        histogram_files = simulator.get_list_of_histogram_files()
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
        logger.info(f"Output files for the grid placed in {str(directory_for_grid_upload)}")


if __name__ == "__main__":
    main()
