#!/usr/bin/python3

"""
Summary
-------
This application calculates the trigger rate from a simtel_array output file, a list of
simtel_array output files ou from a file containing a list of simtel_array files.


Command line arguments
----------------------
simtel_file_names (str or list):
    Path to the simtel_array file or a list of simtel_array output files.
save_tables (bool):
    If true, save the tables with the energy-dependent trigger rate to a ecsv file.
area_from_distribution (bool):
    If true, the area thrown (the area in which the simulated events are distributed)
    in the trigger rate calculation is estimated based on the event distribution.
    The expected shape of the distribution of events as function of the core distance is triangular
    up to the maximum distance. The weighted mean radius of the triangular distribution is 2/3 times
    the upper edge. Therefore, when using the ``area_from_distribution`` flag, the mean distance
    times 3/2, returns just the position of the upper edge in the triangle distribution with little
    impact of the binning and little dependence on the scatter area defined in the simulation.
    This is special useful when calculating trigger rate for individual telescopes.
    If false, the area thrown is estimated based on the maximum distance as given in
    the simulation configuration.

Example
-------
.. code-block:: console

    simtools-calculate-trigger-rate --simtel_file_names tests/resources/
    run201_proton_za20deg_azm0deg_North_TestLayout_test-prod.simtel.zst
"""

import logging
from pathlib import Path

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.io_operations import io_handler
from simtools.simtel.simtel_histograms import SimtelHistograms


def _parse(label, description):
    """
    Parse command line configuration

    Parameters
    ----------
    label: str
        Label describing the application.
    description: str
        Description of the application.

    Returns
    -------
    CommandLineParser
        Command line parser object

    """
    config = configurator.Configurator(label=label, description=description)

    config.parser.add_argument(
        "--simtel_file_names",
        help="Name of the simtel_array output files to be calculate the trigger rate from  or the "
        "text file containing the list of simtel_array output files.",
        nargs="+",
        required=True,
        type=str,
    )

    config.parser.add_argument(
        "--save_tables",
        help="If true, saves the trigger rates per energy bin into ECSV files.",
        action="store_true",
    )

    config.parser.add_argument(
        "--area_from_distribution",
        help="If true, calculates the trigger rates using the event distribution.",
        action="store_true",
    )

    config.parser.add_argument(
        "--stack_files",
        help="If true, stacks all the histograms.",
        action="store_true",
    )

    config_parser, _ = config.initialize(db_config=False, paths=True)

    return config_parser


def main():
    label = Path(__file__).stem
    description = (
        "Calculates the simulated and triggered event rate based on simtel_array output files."
    )
    config_parser = _parse(label, description)

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(config_parser["log_level"]))

    # Building list of simtel_array files from the input files
    simtel_array_files = []
    for one_file in config_parser["simtel_file_names"]:
        try:
            if Path(one_file).suffix in [".zst", ".simtel", ".hdata"]:
                simtel_array_files.append(one_file)
            else:
                # Collecting hist files
                with open(one_file, encoding="utf-8") as file:
                    for line in file:
                        # Removing '\n' from filename, in case it is left there.
                        simtel_array_files.append(line.replace("\n", ""))
        except FileNotFoundError as exc:
            msg = f"{one_file} is not a file."
            logger.error(msg)
            raise FileNotFoundError from exc

    histograms = SimtelHistograms(
        simtel_array_files, area_from_distribution=config_parser["area_from_distribution"]
    )

    logger.info("Calculating simulated and triggered event rate")
    (
        sim_event_rates,
        triggered_event_rates,
        triggered_event_rate_uncertainties,
        trigger_rate_in_tables,
    ) = histograms.calculate_trigger_rates(
        print_info=True, stack_files=config_parser["stack_files"]
    )

    # Print out results
    for i_hist, _ in enumerate(sim_event_rates):
        print(f"\nFile {histograms.histogram_files[i_hist]}\n")
        print(
            f"System trigger rate (Hz): {triggered_event_rates[i_hist].value:.4e} \u00B1 "
            f"{triggered_event_rate_uncertainties[i_hist].value:.4e} Hz"
        )
    if config_parser["save_tables"]:
        io_handler_instance = io_handler.IOHandler()
        output_path = io_handler_instance.get_output_directory(label, sub_dir="application-plots")
        for i_table, table in enumerate(trigger_rate_in_tables):
            output_file = (
                str(output_path.joinpath(Path(simtel_array_files[i_table]).stem)) + ".ecsv"
            )
            logger.info(f"Writing table {i_table + 1} to {output_file}")
            table.write(output_file, overwrite=True)


if __name__ == "__main__":
    main()
