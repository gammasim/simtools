#!/usr/bin/python3

"""
    Summary
    -------
    This application produces a set of histograms for the distribution of Cherenkov photons on the
    ground (at observation level) read from the given CORSIKA IACT output file.

    The histograms can be saved both in a png and in a yaml file. By default, it saves in both
    formats.

    The following 2D histograms are produced:
        - Number of Cherenkov photons on the ground;
        - Density of Cherenkov photons on the ground;
        - Incoming direction (directive cosinus) of the Cherenkov photons;
        - Time of arrival (ns) vs altitude of production (km);
        - Number of Cherenkov photons per telescope.

    The following 1D histograms are produced:
        - Wavelength;
        - Counts;
        - Density;
        - Time of arrival;
        - Altitude of production;
        - Number of photons per telescope;
        - Number of photons per event.


    Command line arguments
    ----------------------
    IACT_file (str, required)
        The name of the CORSIKA IACT file resulted from the CORSIKA simulation.

    telescope_indices (list, optional)
        The list with the telescope indices to be considered in the generation of the histograms.
        Telescopes that are not in the list will not contribute with photons to the histograms.
        If the argument is not given, all telescopes are considered.

    individual_telescopes (bool, optional)
        Indicates whether single histograms are generated for the individual telescopes, or if
        a master histogram is generated for all the telescopes together.
        If the argument is not given, the Cherenkov photons from the given telescopes are considered
         together in the same histograms.

    hist_config (yaml or dict, optional)
        The configuration used for generating the histograms.
        It includes information about the bin sizes, the ranges, scale of the plot and units.
        By construction, three major histograms are created to start with:
         - hist_direction (2D): Directive cosinus (x and y) for the incoming photohs;
         - hist_position (3D): position x, position y, and wavelength;
         - hist_time_altitude (2D): time of arrival and altitude of emission;

        If the argument is not given, the default configuration is generated:

        .. code-block:: console
            hist_direction:
              x axis: {bins: 100, scale: linear, start: -1, stop: 1}
              y axis: {bins: 100, scale: linear, start: -1, stop: 1}

            hist_position:
              x axis:
                bins: 100
                scale: linear
                start: !astropy.units.Quantity
                  unit: &id001 !astropy.units.Unit {unit: m}
                  value: -1000.0
                stop: &id002 !astropy.units.Quantity
                  unit: *id001
                  value: 1000.0
              y axis:
                bins: 100
                scale: linear
                start: !astropy.units.Quantity
                  unit: *id001
                  value: -1000.0
                stop: *id002
              z axis:
                bins: 80
                scale: linear
                start: !astropy.units.Quantity
                  unit: &id003 !astropy.units.Unit {unit: nm}
                  value: 200.0
                stop: !astropy.units.Quantity
                  unit: *id003
                  value: 1000.0
            hist_time_altitude:
              x axis:
                bins: 100
                scale: linear
                start: !astropy.units.Quantity
                  unit: &id004 !astropy.units.Unit {unit: ns}
                  value: -2000.0
                stop: !astropy.units.Quantity
                  unit: *id004
                  value: 2000.0
              y axis:
                bins: 100
                scale: linear
                start: !astropy.units.Quantity
                  unit: &id005 !astropy.units.Unit {unit: km}
                  value: 120.0
                stop: !astropy.units.Quantity
                  unit: *id005
                  value: 0.0

    Example
    -------
    Generate the histograms for a test IACT file:

     .. code-block:: console

        simtools-generate-corsika-histograms --IACT_file /workdir/external/gammasim-tools/tests/\
            resources/tel_output_10GeV-2-gamma-20deg-CTAO-South.corsikaio


    Expected final print-out message:

    .. code-block:: console
        INFO::generate_corsika_histograms(l226)::main::Finalizing the application.
        Total time needed: 6s.
"""

import logging
import time
from pathlib import Path

import simtools.util.general as gen
from simtools.configuration import configurator
from simtools.corsika import corsika_output_visualize
from simtools.corsika.corsika_output import CorsikaOutput


def _parse(label, description, usage):
    """
    Parse command line configuration

    Parameters
    ----------
    label: str
        Label describing the application.
    description: str
        Description of the application.
    usage: str
        Example on how to use the application.

    Returns
    -------
    CommandLineParser
        Command line parser object

    """
    config = configurator.Configurator(label=label, description=description, usage=usage)

    config.parser.add_argument(
        "--IACT_file",
        help="Name of the CORSIKA IACT file from which to generate the histograms.",
        type=str,
        required=True,
    )

    config.parser.add_argument(
        "--telescope_indices",
        help="Name of the CORSIKA IACT file from which to generate the histograms.",
        type=str,
        required=False,
    )

    config.parser.add_argument(
        "--individual_telescopes",
        help="if False, the histograms are filled for all given telescopes together, otherwise"
        "one histogram is set for each telescope separately.",
        type=bool,
        required=False,
        default=False,
    )

    config.parser.add_argument(
        "--hist_config",
        help="Yaml file with the configuration parameters to create the histograms.",
        type=str,
        required=False,
        default=None,
    )
    return config.initialize()


def main():

    label = Path(__file__).stem
    description = "Generate histograms for the Cherenkov photons saved in the CORSIKA IACT file."
    usage = ""
    args_dict, _ = _parse(label, description, usage)

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))
    initial_time = time.time()
    logger.info("Starting the application.")

    instance = CorsikaOutput(args_dict["IACT_file"])
    instance.set_histograms(
        telescope_indices=args_dict["telescope_indices"],
        individual_telescopes=args_dict["individual_telescopes"],
        hist_config=args_dict["hist_config"],
    )

    plot_function_names = [
        "plot_wavelength_distr",
        "plot_counts_distr",
        "plot_density_distr",
        "plot_2D_counts",
        "plot_2D_density",
        "plot_2D_direction",
        "plot_2D_time_altitude",
        "plot_2D_num_photons_per_telescope",
        "plot_time_distr",
        "plot_altitude_distr",
    ]

    for function_name in plot_function_names:
        function = getattr(corsika_output_visualize, function_name)
        function(instance)

    corsika_output_visualize.plot_num_photons_distr(
        instance, log_y=True, event_or_telescope="event"
    )
    corsika_output_visualize.plot_num_photons_distr(
        instance, log_y=True, event_or_telescope="telescope"
    )

    instance.event_1D_histogram("first_interaction_height")
    instance.event_2D_histogram("first_interaction_height", "total_energy")

    final_time = time.time()
    logger.info(
        f"Finalizing the application. Total time needed: {round(final_time - initial_time)}s."
    )


if __name__ == "__main__":
    main()
