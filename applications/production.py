#!/usr/bin/python3

'''
    Summary
    -------
    This application perform array simulations.

    The simulations are split into two stages: showers and array.
    Shower simulations are performed with CORSIKA and array simulations \
    with sim_telarray.

    A configuration file is required. See data/test-data/prodConfigTest.yml \
    for an example.

    Command line arguments
    ----------------------
    config (str, required)
        Path to the configuration file.
    primary (str)
        Name of the primary to be selected from the configuration file. In case it \
        is not given, all the primaries listed in the configuration file will be simulated.
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

        python applications/production.py -c data/test-data/prodConfigTest.yml --test
'''

import logging
import argparse
from copy import copy

from astropy.io.misc import yaml

import simtools.util.general as gen
from simtools.shower_simulator import ShowerSimulator
from simtools.array_simulator import ArraySimulator


def proccessConfigFile(configFile, primaryConfig):

    with open(configFile) as file:
        configData = yaml.load(file)

    label = configData.pop('label', dict())
    defaultData = configData.pop('default', dict())
    configShowers = dict()
    configArrays = dict()

    for primary, primaryData in configData.items():

        if primaryConfig is not None and primary != primaryConfig:
            continue

        thisDefault = copy(defaultData)

        configShowers[primary] = copy(thisDefault.pop('showers', dict()))
        configArrays[primary] = copy(thisDefault.pop('array', dict()))

        # Grabbing common entries for showers and array
        for key, value in primaryData.items():
            if key in ['showers', 'array']:
                continue
            configShowers[primary][key] = value
            configArrays[primary][key] = value

        # Grabbing showers entries
        for key, value in primaryData.get('showers', dict()).items():
            configShowers[primary][key] = value
        configShowers[primary]['primary'] = primary

        # Grabbing array entries
        for key, value in primaryData.get('array', dict()).items():
            configArrays[primary][key] = value
        configArrays[primary]['primary'] = primary

        # Filling in the remaining default keys
        for key, value in thisDefault.items():
            configShowers[primary][key] = value
            configArrays[primary][key] = value

    return label, configShowers, configArrays


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=(
            'Simulate showers to be used for trigger rate calculations'
        )
    )
    parser.add_argument(
        '-c',
        '--config',
        help='Name of the array (e.g. 1MST, 4LST ...)',
        type=str,
        required=True
    )
    parser.add_argument(
        '--primary',
        help='Primary to be selected from the config file.',
        type=str,
        required=False,
        choices=[
            'gamma',
            'electron',
            'proton',
            'helium',
            'nitrogen',
            'silicon',
            'iron'
        ]
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--array_only',
        help='Simulates only array detection, no showers',
        action='store_true'
    )
    group.add_argument(
        '--showers_only',
        help='Simulates only showers, no array detection',
        action='store_true'
    )
    parser.add_argument(
        '--test',
        help='Test option will not submit any job.',
        action='store_true'
    )
    parser.add_argument(
        '-v',
        '--verbosity',
        dest='logLevel',
        action='store',
        default='info',
        help='Log level to print (default is INFO)'
    )

    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(gen.getLogLevelFromUser(args.logLevel))

    label, showerConfigs, arrayConfigs = proccessConfigFile(args.config, args.primary)

    submitCommand = 'more ' if args.test else None

    # ShowerSimulators
    showerSimulators = dict()
    for primary, configData in showerConfigs.items():
        ss = ShowerSimulator(label=label, showerConfigData=configData)
        showerSimulators[primary] = ss

    if not args.array_only:
        # Running Showers
        for primary, shower in showerSimulators.items():
            print('Running ShowerSimulator for primary {}'.format(primary))
            shower.submit(submitCommand=submitCommand)

    # ArraySimulators
    arraySimulators = dict()
    for primary, configData in arrayConfigs.items():
        aa = ArraySimulator(label=label, configData=configData)
        arraySimulators[primary] = aa

    if not args.showers_only:
        # Running Arrays
        for primary, array in arraySimulators.items():
            print('Running ArraySimulator for primary {}'.format(primary))

            inputList = showerSimulators[primary].getListOfOutputFiles()
            array.submit(inputFileList=inputList, submitCommand=submitCommand)
