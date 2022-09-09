#!/usr/bin/python3

"""
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
        run script will be printed.
    verbosity (str, optional)
        Log level to print (default=INFO).

    Example
    -------
    Producing a set of proton showers for trigger rate simulations of LST.

    .. code-block:: console

        python applications/sim_showers_for_trigger_rates.py -a 4LST -s North \
        --primary proton --nruns 100 --nevents 10000 --output {some dir for large files}
"""

import logging

import astropy.units as u

import simtools.config as cfg
import simtools.io_handler as io
import simtools.util.commandline_parser as argparser
import simtools.util.general as gen
from simtools.simulator import Simulator


def main():

    parser = argparser.CommandLineParser(
        description=("Simulate showers to be used for trigger rate calculations")
    )
    parser.add_argument(
        "-a",
        "--array",
        help="Name of the array (e.g. 1MST, 4LST ...)",
        type=str,
        required=True,
    )
    parser.initialize_telescope_model_arguments(add_model_version=False, add_telescope=False)
    parser.add_argument(
        "--primary",
        help="Name of the primary particle (e.g. proton, helium ...)",
        type=str,
        required=True,
    )
    parser.add_argument("--nruns", help="Number of runs (default=100)", type=int, default=100)
    parser.add_argument(
        "--nevents", help="Number of events/run (default=100)", type=int, default=100000
    )
    parser.add_argument("--zenith", help="Zenith angle in deg (default=20)", type=float, default=20)
    parser.add_argument("--azimuth", help="Azimuth angle in deg (default=0)", type=float, default=0)
    parser.add_argument(
        "--output",
        help="Path of the output directory where the simulations will be saved.",
        type=str,
        default=None,
    )
    parser.initialize_default_arguments(add_workflow_config=False)

    args = parser.parse_args()
    label = "trigger_rates"
    cfg.setConfigFileName(args.configFile)

    logger = logging.getLogger()
    logger.setLevel(gen.getLogLevelFromUser(args.logLevel))

    # Output directory to save files related directly to this app
    outputDir = io.getApplicationOutputDirectory(cfg.get("outputLocation"), label)

    showerConfigData = {
        "dataDirectory": args.output,
        "site": args.site,
        "layoutName": args.array,
        "runRange": [1, args.nruns + 1],
        "nshow": args.nevents,
        "primary": args.primary,
        "erange": [10 * u.GeV, 300 * u.TeV],
        "eslope": -2,
        "zenith": args.zenith * u.deg,
        "azimuth": args.azimuth * u.deg,
        "viewcone": 10 * u.deg,
        "cscat": [20, 1500 * u.m, 0],
    }

    showerSimulator = Simulator(label=label, simulator="corsika", configData=showerConfigData)

    if not args.test:
        showerSimulator.submit()
    else:
        logger.info("Test flag is on - it will not submit any job.")
        logger.info("This is an example of the run script:")
        showerSimulator.submit(test=args.test)

    # Exporting the list of output/log/input files into the application folder
    outputFileList = outputDir.joinpath("outputFiles_{}.list".format(args.primary))
    logFileList = outputDir.joinpath("logFiles_{}.list".format(args.primary))

    def printListIntoFile(listOfFiles, fileName):
        with open(fileName, "w") as f:
            for line in listOfFiles:
                f.write(line + "\n")

    logger.info("List of output files exported to {}".format(outputFileList))
    printListIntoFile(showerSimulator.getListOfOutputFiles(), outputFileList)
    logger.info("List of log files exported to {}".format(logFileList))
    printListIntoFile(showerSimulator.getListOfLogFiles(), logFileList)


if __name__ == "__main__":
    main()
