#!/usr/bin/python3

r"""
Calculates array or single-telescope trigger rates.

The applications reads from a sim_telarray output file, a list of
sim_telarray output files ou from a file containing a list of sim_telarray files.


Command line arguments
----------------------
simtel_file_names (str or list):
    Path to the sim_telarray file or a list of sim_telarray output files.
    Files can be generated in `simulate_prod` using the ``--save_file_lists`` option.
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
Calculate trigger rate from sim_telarray file

.. code-block:: console

    simtools-calculate-trigger-rate --simtel_file_names tests/resources/ \\
    run201_proton_za20deg_azm0deg_North_test_layout_test-prod.simtel.zst

Expected final print-out message:

.. code-block:: console

    System trigger rate (Hz): 9.0064e+03 pm 9.0087e+03 Hz

"""

import logging
from pathlib import Path

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.io import io_handler
from simtools.simtel.simtel_io_histograms import SimtelIOHistograms


def _parse(label, description):
    """
    Parse command line configuration.

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
        help="Name of the sim_telarray output files to be calculate the trigger rate from or the "
        "text file containing the list of sim_telarray output files.",
        nargs="+",
        required=True,
        type=str,
    )

    config.parser.add_argument(
        "--save_tables",
        help="Save trigger rates per energy bin into ECSV files.",
        action="store_true",
    )

    config.parser.add_argument(
        "--area_from_distribution",
        help="Calculate trigger rates using the event distribution.",
        action="store_true",
    )

    config.parser.add_argument(
        "--stack_files",
        help="Stacks all histograms.",
        action="store_true",
    )

    config_parser, _ = config.initialize(
        db_config=False,
        paths=True,
        simulation_configuration={"corsika_configuration": ["energy_range", "view_cone"]},
    )

    return config_parser


def _get_simulation_parameters(config_parser):
    """
    Get energy range and view cone in the correct form to use in the simtel classes.

    Parameters
    ----------
    CommandLineParser:
        Command line parser object as defined by the _parse function.

    Returns
    -------
    list:
        The energy range used in the simulation.
    list:
        The view cone used in the simulation.

    """

    def convert(param, unit):
        return [param[0].to(unit).value, param[1].to(unit).value] if param else None

    return convert(config_parser.get("energy_range"), "TeV"), convert(
        config_parser.get("view_cone"), "deg"
    )


def main():  # noqa: D103
    label = Path(__file__).stem
    description = (
        "Calculates the simulated and triggered event rate based on sim_telarray output files."
    )
    config_parser = _parse(label, description)

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(config_parser["log_level"]))

    sim_telarray_files = gen.get_list_of_files_from_command_line(
        config_parser["simtel_file_names"], [".zst", ".simtel", ".hdata"]
    )
    energy_range, view_cone = _get_simulation_parameters(config_parser)

    histograms = SimtelIOHistograms(
        sim_telarray_files,
        area_from_distribution=config_parser["area_from_distribution"],
        energy_range=energy_range,
        view_cone=view_cone,
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
            f"System trigger rate (Hz): {triggered_event_rates[i_hist].value:.4e} \u00b1 "
            f"{triggered_event_rate_uncertainties[i_hist].value:.4e} Hz"
        )
    if config_parser["save_tables"]:
        io_handler_instance = io_handler.IOHandler()
        output_path = io_handler_instance.get_output_directory(label, sub_dir="application-plots")
        for i_table, table in enumerate(trigger_rate_in_tables):
            output_file = (
                str(output_path.joinpath(Path(sim_telarray_files[i_table]).stem)) + ".ecsv"
            )
            logger.info(f"Writing table {i_table + 1} to {output_file}")
            table.write(output_file, overwrite=True)


if __name__ == "__main__":
    main()
