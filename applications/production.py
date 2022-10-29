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

    A configuration file is required. See tests/resources/prodConfigTest.yml \
    for an example.

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
    task (str)
        What task to execute. Options:
            simulate (perform simulations),
            filelist (print list of output files)
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

    .. code-block:: console

        python applications/production.py -t simulate -p tests/resources/prodConfigTest.yml --test

    Running shower simulations.
"""

import logging
from copy import copy

from astropy.io.misc import yaml

import simtools.configuration as configurator
import simtools.util.general as gen
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
            "filelist (print list of output files),"
            "inspect (plot sim_telarray histograms for quick inspection),"
            "resources (print report of computing resources)"
        ),
        type=str,
        required=True,
        choices=["simulate", "filelist", "inspect", "resources"],
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


def _proccessSimulationConfigFile(configFile, primaryConfig, logger):
    """
    Read simulation configuration file with details on shower
    and array simulations

    Attributes
    ----------
    configFile: str
        Name of simulation configuration file
    primaryConfig: str
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
        with open(configFile) as file:
            configData = yaml.load(file)
    except FileNotFoundError:
        logger.error("Error loading simulation configuration file from {}".format(configFile))
        raise

    label = configData.pop("label", dict())
    defaultData = configData.pop("default", dict())
    configShowers = dict()
    configArrays = dict()

    for primary, primaryData in configData.items():

        if primaryConfig is not None and primary != primaryConfig:
            continue

        thisDefault = copy(defaultData)

        configShowers[primary] = copy(thisDefault.pop("showers", dict()))
        configArrays[primary] = copy(thisDefault.pop("array", dict()))

        # Grabbing common entries for showers and array
        for key, value in primaryData.items():
            if key in ["showers", "array"]:
                continue
            configShowers[primary][key] = value
            configArrays[primary][key] = value

        # Grabbing showers entries
        for key, value in primaryData.get("showers", dict()).items():
            configShowers[primary][key] = value
        configShowers[primary]["primary"] = primary

        # Grabbing array entries
        for key, value in primaryData.get("array", dict()).items():
            configArrays[primary][key] = value
        configArrays[primary]["primary"] = primary

        # Filling in the remaining default keys
        for key, value in thisDefault.items():
            configShowers[primary][key] = value
            configArrays[primary][key] = value

    return label, configShowers, configArrays


def main():

    args_dict, db_config = _parse(description=("Air shower and array simulations"))

    logger = logging.getLogger()
    logger.setLevel(gen.getLogLevelFromUser(args_dict["log_level"]))

    label, showerConfigs, arrayConfigs = _proccessSimulationConfigFile(
        args_dict["productionconfig"], args_dict["primary"], logger
    )

    showerSimulators = dict()
    for primary, configData in showerConfigs.items():
        showerSimulators[primary] = Simulator(
            label=label,
            simulator="corsika",
            simulatorSourcePath=args_dict["simtelpath"],
            configData=configData,
            submitCommand=args_dict["submit_command"],
            test=args_dict["test"],
        )

    if args_dict["showers_only"]:
        for primary, shower in showerSimulators.items():
            _taskFunction = getattr(shower, args_dict["task"])
            _taskFunction()

    if args_dict["array_only"]:
        arraySimulators = dict()
        for primary, configData in arrayConfigs.items():
            arraySimulators[primary] = Simulator(
                label=label,
                simulator="simtel",
                simulatorSourcePath=args_dict["simtelpath"],
                configData=configData,
                submitCommand=args_dict["submit_command"],
                mongoDBConfig=db_config,
            )
        for primary, array in arraySimulators.items():
            inputList = showerSimulators[primary].getListOfOutputFiles()
            _taskFunction = getattr(array, args_dict["task"])
            _taskFunction(inputFileList=inputList)


if __name__ == "__main__":
    main()
