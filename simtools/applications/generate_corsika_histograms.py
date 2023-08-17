#!/usr/bin/python3

"""
    Summary
    -------
    This application produces a set of histograms of the distribution of Cherenkov photons on the
    ground (at observation level) read from the CORSIKA IACT output file provided as input.

    The histograms can be saved both in a png and in a ecsv file. By default, it saves in both
    formats.

    The following 2D histograms are produced:
        - Number of Cherenkov photons on the ground;
        - Density of Cherenkov photons on the ground;
        - Incoming direction (directive cosinus) of the Cherenkov photons;
        - Time of arrival (ns) vs altitude of production (km);
        - Number of Cherenkov photons per event per telescope.

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

    hist_config (ecsv or dict, optional)
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

    png (bool, optional)
        If true, histograms are saved into png files.

    ecsv (bool, optional)
        If true, histograms are saved into ecsv files.

    event_1D_histograms (str, optional)
        Produce 1D histograms for elements given in `--event_1D_histograms` from the CORSIKA event
        header and save into ecsv/png files.
        It allows more than one argument, separated by simple spaces.
        Usage: `--event_1D_histograms first_interaction_height total_energy`.

    event_2D_histograms (str, optional)
        Produce 2D histograms for elements given in `--event_2D_histograms` from the CORSIKA event
        header and save into ecsv/png files.
        It allows more than one argument, separated by simple spaces.
        The elements are grouped into pairs and the 2D histograms are produced always for two
        subsequent elements.
        For example, `--event_2D_histograms first_interaction_height total_energy zenith azimuth`
        will produce one 2D histogram for `first_interaction_height` `total_energy` and another 2D
        histogram for `zenith` and `azimuth`.

    Example
    -------
    Generate the histograms for a test IACT file:

     .. code-block:: console

        simtools-generate-corsika-histograms --IACT_file /workdir/external/simtools/tests/\
            resources/tel_output_10GeV-2-gamma-20deg-CTAO-South.corsikaio --png --ecsv
            --event_2D_histograms zenith azimuth --event_1D_histograms total_energy


    Expected final print-out message:

    .. code-block:: console
        INFO::generate_corsika_histograms(l358)::main::Finalizing the application.
        Total time needed: 8s.
"""

import logging
import time
from pathlib import Path

import numpy as np

import simtools.util.general as gen
from simtools import io_handler
from simtools.configuration import configurator
from simtools.corsika import corsika_histograms_visualize
from simtools.corsika.corsika_histograms import CorsikaHistograms

logger = logging.getLogger()


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
        nargs="+",
        default=None,
    )

    config.parser.add_argument(
        "--individual_telescopes",
        help="if False, the histograms are filled for all given telescopes together, otherwise"
        "one histogram is set for each telescope separately.",
        action="store_true",
        required=False,
        default=False,
    )

    config.parser.add_argument(
        "--hist_config",
        help="ecsv file with the configuration parameters to create the histograms.",
        type=str,
        required=False,
        default=None,
    )

    config.parser.add_argument(
        "--png", help="Save histograms into png files.", action="store_true", required=False
    )

    config.parser.add_argument(
        "--ecsv", help="Save histograms into ecsv files.", action="store_true", required=False
    )

    config.parser.add_argument(
        "--event_1D_histograms",
        help="The keys from the CORSIKA event header to be used for the generation of 1D "
             "histograms.",
        required=False,
        default=None,
        nargs="*",
    )

    config.parser.add_argument(
        "--event_2D_histograms",
        help="The keys from the CORSIKA event header to be used for the generation of 2D "
             "histograms.",
        required=False,
        default=None,
        nargs="*",
    )

    config_parser, _ = config.initialize(db_config=False, paths=True)

    if not config_parser["png"] and not config_parser["ecsv"]:
        config.parser.error("At least one argument between `--png` and `--ecsv` is required.")

    return config_parser, _


def _plot_figures(corsika_histograms_instance):
    """
    Auxiliary function to centralize the plotting functions.

    Parameters
    ----------
    corsika_histograms_instance: `CorsikaHistograms` instance.
        The CorsikaHistograms instance created in main.
    """

    plot_function_names = [
        plotting_method
        for plotting_method in dir(corsika_histograms_visualize)
        if plotting_method.startswith("plot_")
        and "event_header_distribution" not in plotting_method
    ]

    for function_name in plot_function_names:
        function = getattr(corsika_histograms_visualize, function_name)
        figures, figure_names = function(corsika_histograms_instance)
        for figure, figure_name in zip(figures, figure_names):
            output_file_name = Path(corsika_histograms_instance.output_path).joinpath(figure_name)
            logger.info(f"Saving histogram to {output_file_name}")
            figure.savefig(output_file_name, bbox_inches="tight")


def _derive_event_1D_histograms(corsika_histograms_instance, event_1D_header_keys, png, ecsv):
    """
    Auxiliary function to derive the histograms for the arguments given by event_1D_histograms.

    Parameters
    ----------
    corsika_histograms_instance: `CorsikaHistograms` instance.
        The CorsikaHistograms instance created in main.
    event_1D_header_keys: str
        Produce 1D histograms for elements given in `event_1D_header_keys` from the CORSIKA event
        header and save into ecsv/png files.
    png: bool
        If true, histograms are saved into png files.
    ecsv: bool
        If true, histograms are saved into ecsv files.
    """
    for event_header_element in event_1D_header_keys:
        if png:
            figure, figure_name = corsika_histograms_visualize.plot_1D_event_header_distribution(
                corsika_histograms_instance, event_header_element
            )
            output_file_name = Path(corsika_histograms_instance.output_path).joinpath(figure_name)
            logger.info(f"Saving histogram to {output_file_name}")
            figure.savefig(output_file_name, bbox_inches="tight")
        if ecsv:
            corsika_histograms_instance.export_event_header_1D_histogram(
                event_header_element, bins=50, hist_range=None
            )


def _derive_event_2D_histograms(corsika_histograms_instance, event_2D_header_keys, png, ecsv):
    """
    Auxiliary function to derive the histograms for the arguments given by event_1D_histograms.
    If an odd number of event header keys are given, the last one is discarded.

    Parameters
    ----------
    corsika_histograms_instance: `CorsikaHistograms` instance.
        The CorsikaHistograms instance created in main.
    event_2D_header_keys: str
        Produce 1D histograms for elements given in `event_1D_header_keys` from the CORSIKA event
        header and save into ecsv/png files.
    png: bool
        If true, histograms are saved into png files.
    ecsv: bool
        If true, histograms are saved into ecsv files.
    """
    for i_event_header_element, _ in enumerate(event_2D_header_keys[::2]):
        # [::2] to discard the last one in case an odd number of keys are passed

        if len(event_2D_header_keys) % 2 == 1:  # if odd number of keys
            msg = "An odd number of keys was passed to produce 2D histograms." \
                  "The last key is being ignored."
            logger.warning(msg)

        if png:
            figure, figure_name = corsika_histograms_visualize.plot_2D_event_header_distribution(
                corsika_histograms_instance,
                event_2D_header_keys[i_event_header_element],
                event_2D_header_keys[i_event_header_element + 1],
            )
            output_file_name = Path(corsika_histograms_instance.output_path).joinpath(figure_name)
            logger.info(f"Saving histogram to {output_file_name}")
            figure.savefig(output_file_name, bbox_inches="tight")
        if ecsv:
            corsika_histograms_instance.export_event_header_2D_histogram(
                event_2D_header_keys[i_event_header_element],
                event_2D_header_keys[i_event_header_element + 1],
                bins=50,
                hist_range=None,
            )


def main():
    label = Path(__file__).stem
    description = "Generate histograms for the Cherenkov photons saved in the CORSIKA IACT file."
    usage = None
    io_handler_instance = io_handler.IOHandler()
    args_dict, _ = _parse(label, description, usage)

    output_path = io_handler_instance.get_output_directory(label, dir_type="application-plots")

    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))
    initial_time = time.time()
    logger.info("Starting the application.")

    corsika_histograms_instance = CorsikaHistograms(args_dict["IACT_file"], output_path=output_path)
    if args_dict["telescope_indices"] is not None:
        try:
            indices = np.array(args_dict["telescope_indices"]).astype(int)
        except ValueError:
            msg = (
                f"{args_dict['telescope_indices']} not a valid input. "
                f"Please use integer numbers for `telescope_indices`"
            )
            logger.error(msg)
            raise
    else:
        indices = None
    corsika_histograms_instance.set_histograms(
        telescope_indices=indices,
        individual_telescopes=args_dict["individual_telescopes"],
        hist_config=args_dict["hist_config"],
    )

    # Cherenkov photons
    if args_dict["png"]:
        _plot_figures(corsika_histograms_instance=corsika_histograms_instance)
    if args_dict["ecsv"]:
        corsika_histograms_instance.export_histograms()

    # Event information
    if args_dict["event_1D_histograms"] is not None:
        _derive_event_1D_histograms(
            corsika_histograms_instance,
            args_dict["event_1D_histograms"],
            args_dict["png"],
            args_dict["ecsv"],
        )
    if args_dict["event_2D_histograms"] is not None:
        _derive_event_2D_histograms(
            corsika_histograms_instance,
            args_dict["event_2D_histograms"],
            args_dict["png"],
            args_dict["ecsv"],
        )

    final_time = time.time()
    logger.info(
        f"Finalizing the application. Total time needed: {round(final_time - initial_time)}s."
    )


if __name__ == "__main__":
    main()
