#!/usr/bin/python3

'''
    Summary
    -------
    This application simulates showers to be used in trigger rate calculations.
    Arrays with one (1MST) or four telescopes (4LST) can be used, in case of \
    mono or stereo trigger configurations, respectively.

    Simulations are managed by the shower_simulator module.
    Each run is simulated in a job. Each job is submitted by using the submission \
    command from the global config settings (see config_template.yml). \
    The config entry extraCommands can be used to extra commands to be ran in each job,
    before the actual simulation.

    At the moment, the shower simulations are performed by CORSIKA, which requires \
    the zstd package. Please, make sure that the command to set your zstd path is \
    properly set by the extraCommands in config.yml.

    Command line arguments
    ----------------------
    array (str, required)
        Name of the array (1MST, 4LST ...).
    site (str, required)
        South or North.
    primary (str, required)
        Name of the primary particle (proton, helium ...).
    nruns (int, optional)
        Number of runs to be simulated (default=100).
    nevents (int, optional)
        Number of events simulated per run (default=100000).
    zenith (float, optional)
        Zenith angle in deg (default=20).
    azimuth (float, optional)
        Azimuth angle in deg (default=0).
    output (str, optional)
        Path of the directory to store the output simulations. By default, \
        the standard output directory defined by config will be used.
    test (activation mode, optional)
        If activated, no job will be submitted. Instead, an example of the \
        run script willbe printed.
    verbosity (str, optional)
        Log level to print (default=INFO).

    Example
    -------
    Producing a set of proton showers for trigger rate simulations of LST.

    .. code-block:: console

        python applications/sim_showers_for_trigger_rates.py -a 4LST -s North \
        --primary proton --nruns 100 --nevents 10000 --output {some dir for large files}
'''

import logging
import argparse
from copy import copy

import astropy.units as u
from astropy.io.misc import yaml

import simtools.io_handler as io
import simtools.config as cfg
import simtools.util.general as gen
from simtools.shower_simulator import ShowerSimulator
from simtools.array_simulator import ArraySimulator


def proccessConfigFile(configFile):

    with open(configFile) as file:
        configData = yaml.load(file)

    label = configData.pop('label', dict())
    defaultData = configData.pop('default', dict())
    configShowers = dict()
    configArrays = dict()

    for primary, primaryData in configData.items():
        configShowers[primary] = copy(defaultData.pop('showers', dict()))
        configArrays[primary] = copy(defaultData.pop('array', dict()))

        for key, value in primaryData.get('showers', dict()).items():
            configShowers[primary][key] = value
        configShowers[primary]['primary'] = primary

        for key, value in primaryData.get('array', dict()).items():
            configArrays[primary][key] = value

        # Filling in the remaining default keys
        for key, value in defaultData.items():
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

    configFile = args.config

    label, showerConfigs, arrayConfigs = proccessConfigFile(args.config)

    # ShowerSimulators
    showerSimulators = list()
    for primary, configData in showerConfigs.items():
        print(configData)
        ss = ShowerSimulator(label=label, showerConfigData=configData)
        print(ss)
        showerSimulators.append(ss)

    print(showerSimulators)
