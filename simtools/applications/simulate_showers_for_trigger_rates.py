#!/usr/bin/python3
r"""
    Simulates showers to be used in trigger rate calculations.

    Arrays with one or four telescopes can be used, in case of \
    mono or stereo trigger configurations, respectively.

    Simulations are managed by the Simulator module.
    Each run is simulated in a job. Each job is submitted by using the submission \
    command from the global config settings. The config entry extra_commands can be used \
    to extra commands to be ran in each job, before the actual simulation.

    At the moment, the shower simulations are performed by CORSIKA, which requires \
    the zstd package. Please, make sure that the command to set your zstd path is \
    properly set by the extra_commands in the command line configuration.

    Command line arguments
    ----------------------
    array_layout_name (str, required)
        Name of the array (pre-defined in the site model).
    site (str, required)
        South or North.
    primary (str, required)
        Name of the primary particle (proton, helium ...).
    nruns (int, optional)
        Number of runs to be simulated.
    nevents (int, optional)
        Number of events simulated per run.
    zenith (float, optional)
        Zenith angle in deg.
    azimuth (float, optional)
        Azimuth angle in deg.
    view_cone (float, optional)
        View cone in deg.
    scatter_x (float, optional)
        Scatter distance (X axis) in m.
    num_use (int, optional)
        Number of use for each shower.
    energy_min (float, optional)
        Energy threshold (TeV).
    energy_max (float, optional)
        Maximum energy (TeV).
    e_slope (int, optional)
        Energy slope (spectral index).
    data_directory (str, optional)
        The location of the output directories corsika-data.
        the label is added to the data_directory, such that the output
        will be written to data_directory/label/corsika-data.
    verbosity (str, optional)
        Log level to print.

    Example
    -------
    Producing a set of proton showers for trigger rate simulations of LST.

    .. code-block:: console

        simtools-simulate-showers-for-trigger-rates --array 4LST --site North --primary \\
        proton --nruns 2 --nevents 10000 --submit_command local

    The output is saved in simtools-output/simulate_showers_for_trigger_rates.

    Expected final print-out message:

    .. code-block:: console

        INFO::simulate_showers_for_trigger_rates(l174)::main::List of log files exported to \
        /workdir/external/simtools/simtools-output/simulate_showers_for_trigger_rates/application-\
        plots/log_files_proton.list
        INFO::simulator(l646)::get_list_of_log_files::Getting list of log files

"""

import logging
from pathlib import Path

import astropy.units as u

from simtools.configuration import configurator
from simtools.io_operations import io_handler
from simtools.simulator import Simulator
from simtools.utils import general as gen


def _parse(label=None, description=None):
    """
    Parse command line configuration.

    Parameters
    ----------
    label: str
        application label.
    description: str
        description of application.

    Returns
    -------
    CommandLineParser
        command line parser object

    """
    config = configurator.Configurator(label=label, description=description)
    config.parser.add_argument(
        "--array_layout_name",
        help="Name of the (pre-defined) array",
        type=str,
        required=True,
    )
    config.parser.add_argument(
        "--primary",
        help="Name of the primary particle (e.g. proton, helium ...)",
        type=str,
        required=True,
    )
    config.parser.add_argument("--nruns", help="Number of runs", type=int, default=100)
    config.parser.add_argument(
        "--run_number", help="Run number of the starting run", type=int, default=1
    )
    config.parser.add_argument("--nevents", help="Number of events/run", type=int, default=100000)
    config.parser.add_argument("--zenith", help="Zenith angle in deg", type=float, default=20)
    config.parser.add_argument("--azimuth", help="Azimuth angle in deg", type=float, default=0)
    config.parser.add_argument("--view_cone", help="View cone in deg", type=float, default=10)
    config.parser.add_argument(
        "--scatter_x", help="Scatter distance (X axis) in m", type=float, default=1500
    )

    config.parser.add_argument(
        "--num_use", help="Number of use for each shower", type=int, default=10
    )
    config.parser.add_argument(
        "--energy_min", help="Energy threshold (TeV)", type=float, default=0.01
    )
    config.parser.add_argument("--energy_max", help="Maximum energy (TeV)", type=float, default=300)
    config.parser.add_argument(
        "--e_slope", help="Energy slope (spectral index)", type=float, default=-2
    )
    config.parser.add_argument(
        "--data_directory",
        help=(
            "The directory where to save the corsika-data output directories."
            "the label is added to the data_directory, such that the output"
            "will be written to data_directory/label/corsika-data."
        ),
        type=str.lower,
        required=False,
        default="./simtools-output/",
    )
    return config.initialize(simulation_model="telescope", job_submission=True, db_config=True)


def print_list_into_file(list_of_files, file_name):
    """
    Print the list of output files from the simulation into a log file.

    Parameters
    ----------
    list_of_files: list
        list of files to be printed out.
    file_name: str
        name of the output file.
    """
    with open(file_name, "w", encoding="utf-8") as f:
        for line in list_of_files:
            f.write(line + "\n")


def main():  # noqa: D103
    label = Path(__file__).stem
    args_dict, db_config = _parse(
        label=label, description="Simulate showers to be used for trigger rate calculations"
    )

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    # Output directory to save files related directly to this app
    _io_handler = io_handler.IOHandler()
    output_dir = _io_handler.get_output_directory(label, sub_dir="application-plots")
    shower_config_data = {
        "common": {
            "data_directory": Path(args_dict["data_directory"]) / label,
            "site": args_dict["site"],
            "layout_name": args_dict["array_layout_name"],
            "run_range": [
                args_dict["run_number"],
                args_dict["run_number"] + args_dict["nruns"] - 1,
            ],
            "nshow": args_dict["nevents"],
            "primary": args_dict["primary"],
            "zenith": args_dict["zenith"] * u.deg,
            "azimuth": args_dict["azimuth"] * u.deg,
        },
        "showers": {
            "erange": [args_dict["energy_min"] * u.TeV, args_dict["energy_max"] * u.TeV],
            "eslope": args_dict["e_slope"],
            "viewcone": args_dict["view_cone"] * u.deg,
            "cscat": [
                args_dict["num_use"],
                args_dict["scatter_x"] * u.m,
                0 * u.m,
            ],
        },
    }

    shower_simulator = Simulator(
        label=label,
        simulation_software="corsika_simtel",
        simulator_source_path=args_dict.get("simtel_path", None),
        config_data=shower_config_data,
        submit_command=args_dict.get("submit_command", ""),
        mongo_db_config=db_config,
        model_version=args_dict.get("model_version", None),
    )
    if shower_simulator.array_model.number_of_telescopes == 1:
        shower_simulator.array_model.site_model.change_parameter(
            "array_triggers", "array_trigger_1MST_lapalma.dat"
        )

    shower_simulator.simulate()

    # Exporting the list of output/log/input files into the application folder
    output_file_list = output_dir.joinpath(f"output_files_{args_dict['primary']}.list")
    log_file_list = output_dir.joinpath(f"log_files_{args_dict['primary']}.list")

    logger.info(f"List of output files exported to {output_file_list}")
    print_list_into_file(shower_simulator.get_list_of_output_files(), output_file_list)
    logger.info(f"List of log files exported to {log_file_list}")
    print_list_into_file(shower_simulator.get_list_of_log_files(), log_file_list)


if __name__ == "__main__":
    main()
