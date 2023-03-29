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
    sim_telarray are available.
    The multipipe and/or the generic run scripts are either assumed to be available
    or will be produced as part of this script FIXME - decide and update this docstring.

    This script does not provide a mechanism to submit jobs to a batch system like others
    in gammasim-tools since it is meant to be executed on a grid node
    (distributed to it by the worload management system).

    Command line arguments
    ----------------------
    prod_tag (str, required)
        The production tag (ID) to use (e.g., Prod5)
    site (str, required)
        Paranal or LaPalma (case insensitive)
    primary (str, required)
        Name of the primary particla to simulate. The available options are
        gamma, gamma_diffuse, electron, proton, muon, helium, nitrogen, silicon, and iron.
    from_direction (str, required)
        Should be one of North, South, East, West (case insensitive)
    zenith_angle (float, required)
        Zenith angle in degrees
    verbosity (str, optional)
        Log level to print (default=INFO).

    Example
    -------
    FIXME
    Write an example

    .. code-block:: console

        python applications/get_file_from_db.py --file_name prod_config_test.yml

    Run the application:

    .. code-block:: console

        python applications/production.py --task simulate --productionconfig prod_config_test.yml \
        --test --showers_only --submit_command local

    The output is saved in simtools-output/test-production.

    Expected final print-out message:

    .. code-block:: console

        INFO::job_manager(l124)::_submit_local::Running script locally
        INFO::job_manager(l133)::_submit_local::Testing (local)
"""

import logging
from copy import copy

from astropy.io.misc import yaml

import simtools.util.general as gen
from simtools.configuration import configurator
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
        "--productionconfig",
        help="Simulation configuration file",
        type=str,
        required=True,
    )
    config.parser.add_argument(
        "--task",
        help=(
            "What task to execute. Options: "
            "simulate (perform simulations),"
            "file_list (print list of output files),"
            "inspect (plot sim_telarray histograms for quick inspection),"
            "resources (print report of computing resources)"
        ),
        type=str,
        required=True,
        choices=["simulate", "file_list", "inspect", "resources"],
    )
    config.parser.add_argument(
        "--primary",
        help="Primary to be selected from the simulation configuration file.",
        type=str,
        required=False,
        choices=[
            "gamma",
            "electron",
            "proton",
            "helium",
            "nitrogen",
            "silicon",
            "iron",
        ],
    )
    group = config.parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--showers_only",
        help="Simulates only showers, no array detection",
        action="store_true",
    )
    group.add_argument(
        "--array_only",
        help="Simulates only array detection, no showers",
        action="store_true",
    )
    return config.initialize(db_config=True, job_submission=True)


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

    label = config_data.pop("label", dict())
    default_data = config_data.pop("default", dict())
    config_showers = dict()
    config_arrays = dict()

    for primary, primary_data in config_data.items():

        if primary_config is not None and primary != primary_config:
            continue

        this_default = copy(default_data)

        config_showers[primary] = copy(this_default.pop("showers", dict()))
        config_arrays[primary] = copy(this_default.pop("array", dict()))

        # Grabbing common entries for showers and array
        for key, value in primary_data.items():
            if key in ["showers", "array"]:
                continue
            config_showers[primary][key] = value
            config_arrays[primary][key] = value

        # Grabbing showers entries
        for key, value in primary_data.get("showers", dict()).items():
            config_showers[primary][key] = value
        config_showers[primary]["primary"] = primary

        # Grabbing array entries
        for key, value in primary_data.get("array", dict()).items():
            config_arrays[primary][key] = value
        config_arrays[primary]["primary"] = primary

        # Filling in the remaining default keys
        for key, value in this_default.items():
            config_showers[primary][key] = value
            config_arrays[primary][key] = value

    return label, config_showers, config_arrays


def main():

    args_dict, db_config = _parse(description=("Air shower and array simulations"))

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    label, shower_configs, array_configs = _proccess_simulation_config_file(
        args_dict["productionconfig"], args_dict["primary"], logger
    )
    if args_dict["label"] is None:
        args_dict["label"] = label

    shower_simulators = dict()
    for primary, config_data in shower_configs.items():
        shower_simulators[primary] = Simulator(
            label=label,
            simulator="corsika",
            simulator_source_path=args_dict["simtel_path"],
            config_data=config_data,
            submit_command=args_dict["submit_command"],
            test=args_dict["test"],
            mongo_db_config=db_config,
        )

    if args_dict["showers_only"]:
        for primary, shower in shower_simulators.items():
            _task_function = getattr(shower, args_dict["task"])
            _task_function()

    if args_dict["array_only"]:
        array_simulators = dict()
        for primary, config_data in array_configs.items():
            array_simulators[primary] = Simulator(
                label=label,
                simulator="simtel",
                simulator_source_path=args_dict["simtel_path"],
                config_data=config_data,
                submit_command=args_dict["submit_command"],
                mongo_db_config=db_config,
            )
        for primary, array in array_simulators.items():
            input_list = shower_simulators[primary].get_list_of_output_files()
            _task_function = getattr(array, args_dict["task"])
            _task_function(input_file_list=input_list)


if __name__ == "__main__":
    main()
