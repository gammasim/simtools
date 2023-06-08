#!/usr/bin/python3

"""
    Summary
    -------
    This application is used to run simulations for productions (typically on the grid).
    It allows to run a Paranal (CTA-South) or La Palma (CTA-North) array layout simulation
    with the provided "prod_tag" simulation configuration (e.g., Prod6)
    for a given primary particle, azimuth, and zenith angle.

    The entire simulation chain is performed, i.e., shower simulations with CORSIKA
    which are piped directly to sim_telarray using the sim_telarray multipipe mechanism.
    This script assumes that all the necessary configuration files for CORISKA and
    sim_telarray are available. FIXME - This is not true at the moment, need to fix I guess.
    The multipipe scripts will be produced as part of this script.

    This script does not provide a mechanism to submit jobs to a batch system like others
    in gammasim-tools since it is meant to be executed on a grid node
    (distributed to it by the workload management system).

    Command line arguments
    ----------------------
    production_config (str, Path, required)
        Simulation configuration file
        (contains the default setup which can be overwritten by the command line options)
    prod_tag (str, required)
        The production tag (ID) to use (e.g., Prod5)
    site (str, required)
        Paranal or LaPalma (case insensitive)
    primary (str, required)
        Name of the primary particle to simulate. The available options are
        gamma, gamma_diffuse, electron, proton, muon, helium, nitrogen, silicon, and iron.
    from_direction (str, required)
        Should be one of North, South, East, West (case insensitive)
    zenith_angle (float, required)
        Zenith angle in degrees
    nshow (int, optional)
        Number of showers to simulate
    start_run (int, required)
        Start run number such that the actual run number will be 'start_run' + 'run'.
        This is useful in case a new transform is submitted for the same production.
        It allows the transformation system to keep using sequential run numbers without repetition.
    run (int, required)
        Run number (actual run number will be 'start_run' + 'run')
    log_level (str, optional)
        Log level to print (default=INFO).

    Example
    -------
    Run the application:

    .. code-block:: console

        python applications/simulate_prod.py \
        --production_config tests/resources/prod_multi_config_test.yml --prod_tag Prod5 \
        --site lapalma --primary gamma --from_direction north --zenith_angle 20 \
         --start_run 0 --run 1

    By default the configuration is saved in simtools-output/test-production
    and the output in corsika-data and simtel-data. The location of the latter directories
    can be set to a different location via the option --data_directory)

    Expected final print-out message:

    .. code-block:: console

        INFO::layout_array(l569)::read_telescope_list_file::Reading array elements from ...
        WARNING::corsika_runner(l127)::_load_corsika_config_data::data_directory not given
        in corsika_config - default output directory will be set.
        INFO::layout_array(l569)::read_telescope_list_file::Reading array elements from ...
        INFO::corsika_config(l493)::_set_output_file_and_directory::Creating directory
        INFO::simulator(l405)::simulate::Submission command: local
        INFO::simulator(l410)::simulate::Starting submission for 1 runs
        INFO::array_model(l315)::export_simtel_array_config_file::Writing array config file into
        INFO::job_manager(l95)::submit::Submitting script
        INFO::job_manager(l96)::submit::Job output stream
        INFO::job_manager(l97)::submit::Job error stream
        INFO::job_manager(l98)::submit::Job log stream
        INFO::job_manager(l119)::_submit_local::Running script locally
"""

import logging
from copy import copy

import astropy.units as u
from astropy.io.misc import yaml

import simtools.util.general as gen
from simtools.configuration import configurator
from simtools.configuration.commandline_parser import CommandLineParser
from simtools.simulator import Simulator


def _parse(description=None):
    """
    Parse command line configuration

    Parameters
    ----------
    description: str
        description of application.

    Returns
    -------
    CommandLineParser
        command line parser object

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
        "--prod_tag",
        help="The production tag (ID) to use (e.g., Prod5)",
        type=str,
        required=True,
    )
    config.parser.add_argument(
        "--site",
        help="CTAO site (e.g., Paranal or LaPalma, case insensitive)",
        type=str.lower,
        required=True,
        choices=[
            "paranal",
            "lapalma",
        ],
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
        "--from_direction",
        help="Direction from which the primary reaches the atmosphere",
        type=str.lower,
        required=True,
        choices=[
            "north",
            "south",
            "east",
            "west",
        ],
    )
    config.parser.add_argument(
        "--zenith_angle",
        help="Zenith angle in degrees",
        type=CommandLineParser.zenith_angle,
        required=True,
    )
    config.parser.add_argument(
        "--nshow",
        help="Number of showers to simulate",
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
        help="The directory where to save the corsika-data and simtel-data output directories.",
        type=str.lower,
        required=False,
        default="./",
    )
    return config.initialize(db_config=True)


def _proccess_simulation_config_file(config_file, primary_config, logger):
    """
    Read simulation configuration file with details on shower
    and array simulations

    Attributes
    ----------
    config_file: str
        Name of simulation configuration file
    primary_config: str
        Name of the primary selected from the configuration file.

    Returns
    -------
    str
        label of simulation configuration
    dict
        configuration of shower simulations
    dict
        configuration of array simulations

    """

    try:
        with open(config_file) as file:
            config_data = yaml.load(file)
    except FileNotFoundError:
        logger.error(f"Error loading simulation configuration file from {config_file}")
        raise

    label = config_data.pop("label", "test_run")
    default_data = config_data.pop("default", {})

    for primary, primary_data in config_data.items():

        if primary_config is not None and primary != primary_config:
            continue

        this_default = copy(default_data)

        config_showers = copy(this_default.pop("showers", {}))
        config_arrays = copy(this_default.pop("array", {}))

        # Grabbing common entries for showers and array
        for key, value in primary_data.items():
            if key in ["showers", "array"]:
                continue
            config_showers[key] = value
            config_arrays[key] = value

        # Grabbing showers entries
        for key, value in primary_data.get("showers", {}).items():
            config_showers[key] = value
        config_showers["primary"] = primary

        # Grabbing array entries
        for key, value in primary_data.get("array", {}).items():
            config_arrays[key] = value
        config_arrays["primary"] = primary

        # Filling in the remaining default keys
        for key, value in this_default.items():
            config_showers[key] = value
            config_arrays[key] = value

    config_arrays["data_directory"] = config_showers["data_directory"]
    config_arrays["site"] = config_showers["site"]
    config_arrays["layout_name"] = config_showers["layout_name"]

    return label, config_showers, config_arrays


def _translate_from_direction_to_azimuth(logger, from_direction):
    """
    Translate the direction particles are coming from to an azimuth angle

    Attributes
    ----------
    from_direction: str (north, south, east, west)
        The direction particles are coming from

    Returns
    -------
    float (Astropy.Quantity)
        The phi angle for CORSIKA configuration

    """

    if from_direction == "north":
        return 0 * u.deg
    if from_direction == "south":
        return 180 * u.deg
    if from_direction == "east":
        return 90 * u.deg
    if from_direction == "west":
        return 270 * u.deg

    logger.error(f"The direction {from_direction} to simulate from was not recognised")
    raise ValueError


def main():

    args_dict, db_config = _parse(description=("Run simulations for productions"))

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    label, shower_configs, array_configs = _proccess_simulation_config_file(
        args_dict["production_config"], args_dict["primary"], logger
    )

    # Overwrite default and optional settings
    array_configs["site"] = shower_configs["site"] = args_dict["site"]
    array_configs["zenith"] = shower_configs["zenith"] = args_dict["zenith_angle"] * u.deg
    array_configs["phi"] = shower_configs["phi"] = _translate_from_direction_to_azimuth(
        logger, args_dict["from_direction"]
    )
    if args_dict["nshow"] is not None:
        shower_configs["nshow"] = args_dict["nshow"]
    if "label" in args_dict:
        label = args_dict["label"]
    if "data_directory" in args_dict:
        array_configs["data_directory"] = shower_configs["data_directory"] = args_dict[
            "data_directory"
        ]

    simulator = Simulator(
        label=label,
        simulator="corsika_simtel",
        simulator_source_path=args_dict["simtel_path"],
        config_data=shower_configs | array_configs,
        submit_command="local",
        test=args_dict["test"],
        mongo_db_config=db_config,
    )

    simulator.simulate()


if __name__ == "__main__":
    main()
