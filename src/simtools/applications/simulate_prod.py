#!/usr/bin/python3

r"""
    Generate simulation configuration and run simulations.

    Multipipe scripts will be produced as part of this application.
    Allows to run array layout simulation including shower and detector simulations

    The entire simulation chain, parts of it, or nothing is executed:

    - shower simulations with CORSIKA only
    - shower simulations with CORSIKA which are piped directly to sim_telarray using
      the sim_telarray multipipe mechanism.

    Command line arguments
    ----------------------
    model_version (str, required)
        The telescope model version to use (e.g., 5.0.0).
    site (str, required)
        North or South (case insensitive).
    primary (str, required)
        Name or ID of the primary particle to simulate. Allowed are common names like gamma, proton,
        or IDs for CORSIKA7 (e.g. 14 for proton) and PDG (e.g. 2212 for proton). Use the
        'primary_id_type' option to specify the type of ID.
    azimuth_angle (str or float, required)
        Telescope pointing direction in azimuth.
        It can be in degrees between 0 and 360 or one of north, south, east or west
        (case insensitive). Note that North is 0 degrees and the azimuth grows clockwise,
        so East is 90 degrees.
    zenith_angle (float, required)
        Zenith angle in degrees.
    nshow (int, optional)
        Number of showers to simulate.
        The Number of simulated events depends on the number of times a shower is reused in the
        telescope simulation. The number provided here is before any reuse factors.
    start_run (int, required)
        Start run number such that the actual run number will be 'start_run' + 'run'.
        This is useful in case a new transform is submitted for the same production.
        It allows the transformation system to keep using sequential run numbers without repetition.
    run (int, required)
        Run number (actual run number will be 'start_run' + 'run').
    pack_for_grid_register (str, optional)
        Set whether to prepare a tarball for registering the output files on the grid.
        The files are written to the specified directory.
    log_level (str, optional)
        Log level to print.

    Example
    -------
    Run the application:

    .. code-block:: console

        simtools-simulate-prod \\
        --model_version 5.0.0 --site north --primary gamma --azimuth_angle north \\
        --zenith_angle 20 --start_run 0 --run 1
"""

from simtools.application_control import startup_application
from simtools.configuration import configurator
from simtools.simulator import Simulator


def _parse():
    """Parse command line configuration."""
    config = configurator.Configurator(description="Run simulations for productions")
    config.parser.add_argument(
        "--corsika_file",
        help=(
            "Path to the CORSIKA input file (only relevant for simulation software 'sim_telarray')."
        ),
        type=str,
        required=True,
    )
    config.parser.add_argument(
        "--pack_for_grid_register",
        help="Directory for a tarball for registering the output files on the grid.",
        type=str,
        required=False,
        default=None,
    )
    config.parser.add_argument(
        "--save_file_lists",
        help="Save lists of output and log files.",
        action="store_true",
        required=False,
        default=False,
    )
    config.parser.add_argument(
        "--save_reduced_event_lists",
        help=(
            "Save reduced event lists with event data on simulated and triggered events. "
            "Saved with the same name as the sim_telarray output file (different extension). "
        ),
        action="store_true",
        required=False,
        default=False,
    )
    config.parser.add_argument(
        "--corsika_test_seeds",
        help="Use predefined random seeds for CORSIKA for testing purposes.",
        action="store_true",
        required=False,
        default=False,
    )
    config.parser.add_argument(
        "--sequential",
        help=(
            "Enables single-core mode (as far as possible); "
            "otherwise, CORSIKA and sim_telarray run in parallel."
        ),
        action="store_true",
        default=False,
    )
    return config.initialize(
        db_config=True,
        simulation_model=["site", "layout", "telescope", "model_version"],
        simulation_configuration={
            "software": None,
            "corsika_configuration": ["all"],
            "sim_telarray_configuration": ["all"],
        },
    )


def main():
    """Run simulations for productions."""
    app_context = startup_application(_parse, setup_io_handler=False)

    simulator = Simulator(
        label=app_context.args.get("label"),
        args_dict=app_context.args,
        db_config=app_context.db_config,
    )

    simulator.simulate()
    simulator.validate_metadata()

    if app_context.args["save_reduced_event_lists"]:
        simulator.save_reduced_event_lists()

    simulator.verify_simulations()

    if app_context.args.get("pack_for_grid_register"):
        simulator.pack_for_register(app_context.args["pack_for_grid_register"])
    if app_context.args["save_file_lists"]:
        simulator.save_file_lists()

    simulator.report()


if __name__ == "__main__":
    main()
