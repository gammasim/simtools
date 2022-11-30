#!/usr/bin/python3

"""
    Summary
    -------
    This application perform array simulations.

    The simulations are split into two stages: showers and array.
    Shower simulations are performed with CORSIKA and array simulations \
    with sim_telarray. Note that either shower or array simulations are \
    submitted (so typically you first run shower simulations, and then the \
    array simulations).

    A configuration file is required.

    The workload management system used is given in the configuration file. \
    Allowed systems are qsub (using gridengine), condor_submit \
    (using HTcondor), and local (running the script locally).

    Command line arguments
    ----------------------
    productionconfig (str, required)
        Path to the simulation configuration file.
    primary (str)
        Name of the primary to be selected from the configuration file. In case it \
        is not given, all the primaries listed in the configuration file will be simulated.
        The choices are: gamma, electron, proton, helium, nitrogen, silicon, and iron.
    task (str)
        What task to execute. Options:
            simulate (perform simulations),
            file_list (print list of output files)
            inspect (plot sim_telarray histograms for quick inspection)
            resources (print quicklook into used computational resources)
    array_only (activation mode)
        Simulates only array detector (no showers).
    showers_only (activation mode)
        Simulates only showers (no array detector).
    test (activation mode, optional)
        If activated, no job will be submitted, but all configuration files \
        and run scripts will be created.
    verbosity (str, optional)
        Log level to print (default=INFO).

    Example
    -------
    Testing a mini-prod5 simulation.

    Get the configuration file from the DB

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
