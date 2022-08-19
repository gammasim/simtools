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
    (using HTcondor), and seriell_script (running the script locally).

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
            lists (print list of output files) [NOT IMPLEMENTED]
            inspect (plot sim_telarray histograms for quick inspection) [NOT IMPLEMENTED]
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

import simtools.util.commandline_parser as argparser
import simtools.config as cfg
import simtools.util.general as gen
from simtools.shower_simulator import ShowerSimulator
from simtools.array_simulator import ArraySimulator


def parse(description=None):
    """
    Parse command line configuration

    """

    parser = argparser.CommandLineParser(description=description)
    parser.add_argument(
        "-p",
        "--productionconfig",
        help="Simulation configuration file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-t",
        "--task",
        help=(
            'What task to execute. Options: '
            + 'simulate (perform simulations),'
            + 'lists (print list of output files),'
            + 'inspect (plot sim_telarray histograms for quick inspection),'
            + 'resources (print report of computing resources)'
        ),
        type=str,
        required=True,
        choices=['simulate', 'lists', 'inspect', 'resources']
    )
    parser.add_argument(
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
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--array_only",
        help="Simulates only array detection, no showers",
        action="store_true",
    )
    group.add_argument(
        "--showers_only",
        help="Simulates only showers, no array detection",
        action="store_true",
    )
    parser.initialize_default_arguments(
        add_workflow_config=False)
    return parser.parse_args()


def proccessSimulationConfigFile(configFile, primaryConfig, logger):
    """
    Read simulation configuration file with details on shower
    and array simulations

    """

    try:
        with open(configFile) as file:
            configData = yaml.load(file)
    except FileNotFoundError:
        logger.error(
            "Error loading simulation configuration file from {}".format(
                configFile)
        )
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


if __name__ == "__main__":

    args = parse(description=("Air shower and array simulations"))

    cfg.setConfigFileName(args.configFile)

    logger = logging.getLogger()
    logger.setLevel(gen.getLogLevelFromUser(args.logLevel))

    label, showerConfigs, arrayConfigs = proccessSimulationConfigFile(
        args.productionconfig, args.primary, logger)

    submitCommand = "more " if args.test else None

    # ShowerSimulators
    showerSimulators = dict()
    for primary, configData in showerConfigs.items():
        showerSimulators[primary] = ShowerSimulator(label=label, showerConfigData=configData)

    if not args.array_only:
        # Running Showers
        for primary, shower in showerSimulators.items():

            if args.task == "simulate":
                print("Running ShowerSimulator for primary {}".format(primary))
                shower.submit(submitCommand=submitCommand)

            elif args.task == "list":
                print(
                    "Printing ShowerSimulator file lists for primary {}".format(primary)
                )
                raise NotImplementedError()

            elif args.task == 'resources':
                print('Printing computing resources report for primary {}'.format(primary))
                shower.printResourcesReport()

    # ArraySimulators
    arraySimulators = dict()
    for primary, configData in arrayConfigs.items():
        aa = ArraySimulator(label=label, configData=configData)
        arraySimulators[primary] = aa

    if not args.showers_only:
        # Running Arrays
        for primary, array in arraySimulators.items():

            inputList = showerSimulators[primary].getListOfOutputFiles()
            if args.task == "simulate":
                print("Running ArraySimulator for primary {}".format(primary))
                array.submit(inputFileList=inputList, submitCommand=submitCommand)

            elif args.task == "lists":
                print(
                    "Printing ArraySimulator file lists for primary {}".format(primary)
                )
                raise NotImplementedError()

            elif args.task == "inspect":
                print(
                    "Plotting ArraySimulator histograms for primary {}".format(primary)
                )
                file = array.printHistograms(inputList)
                print('Histograms file {}'.format(file))

            elif args.task == 'resources':
                print('Printing computing resources report for primary {}'.format(primary))
                array.printResourcesReport(inputList)
