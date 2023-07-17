#!/usr/bin/python3

"""
    Summary
    -------
    This application produces a set of histograms for the distribution of Cherenkov photons on the
    ground (at observation level) read from the given CORSIKA IACT output file.

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

    output_directory (str, optional)
        Output directory where to save the histograms.
        If the argument is not given, the histograms are saved in
        `simtools-output/generate_corsika_histograms/application-plots`.

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

        simtools-generate-corsika-histograms --IACT_file /workdir/external/gammasim-tools/tests/\
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

import simtools.util.general as gen
from simtools import io_handler
from simtools.configuration import configurator
from simtools.corsika import corsika_output_visualize
from simtools.corsika.corsika_output import CorsikaOutput

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
        help="ecsv file with the configuration parameters to create the histograms.",
        type=str,
        required=False,
        default=None,
    )

    config.parser.add_argument(
        "--output_directory",
        help="Output directory where to save the histograms.",
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
        help="Arguments from the CORSIKA event header to extract 1D histograms.",
        required=False,
        default=None,
        nargs="*",
    )

    config.parser.add_argument(
        "--event_2D_histograms",
        help="Arguments from the CORSIKA event header to extract 2D histograms.",
        required=False,
        default=None,
        nargs="*",
    )

    config_parser, _ = config.initialize()

    if not config_parser["png"] and not config_parser["ecsv"]:
        config.parser.error("At least one argument between `--png` and `--ecsv` is required.")

    return config_parser, _


def _plot_figures(instance, output_dir):
    """
    Auxiliary function to centralize the plotting functions.

    Parameters
    ----------
    instance: `CorsikaOutput` instance.
        The CorsikaOutput instance created in main.
    output_dir: str
        The output directory where to save the histograms.
    """

    plot_function_names = [
        "plot_wavelength_distr",
        "plot_counts_distr",
        "plot_density_distr",
        "plot_time_distr",
        "plot_altitude_distr",
        "plot_photon_per_event_distr",
        "plot_photon_per_telescope_distr",
        "plot_2D_counts",
        "plot_2D_density",
        "plot_2D_direction",
        "plot_2D_time_altitude",
        "plot_2D_num_photons_per_telescope",
    ]

    for function_name in plot_function_names:
        function = getattr(corsika_output_visualize, function_name)
        figures, figure_names = function(instance)
        for figure, figure_name in zip(figures, figure_names):
            output_file_name = Path(output_dir).joinpath(figure_name)
            logger.info(f"Saving histogram to {output_file_name}")
            figure.savefig(output_file_name, bbox_inches="tight")


def _run_event_1D_histograms(instance, output_dir, event_1D_header_keys, png, ecsv):
    """
    Auxiliary function to run the histograms for the arguments given by event_1D_histograms.

    Parameters
    ----------
    instance: `CorsikaOutput` instance.
        The CorsikaOutput instance created in main.
    output_dir: str
        The output directory where to save the histograms.
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
            figure, figure_name = corsika_output_visualize.plot_1D_event_header_distribution(
                instance, event_header_element
            )
            output_file_name = Path(output_dir).joinpath(figure_name)
            logger.info(f"Saving histogram to {output_file_name}")
            figure.savefig(output_file_name, bbox_inches="tight")
        if ecsv:
            instance.export_event_header_1D_histogram(
                event_header_element, bins=50, hist_range=None
            )


def _run_event_2D_histograms(instance, output_dir, event_2D_header_keys, png, ecsv):
    """
    Auxiliary function to run the histograms for the arguments given by event_1D_histograms.

    Parameters
    ----------
    instance: `CorsikaOutput` instance.
        The CorsikaOutput instance created in main.
    output_dir: str
        The output directory where to save the histograms.
    event_2D_header_keys: str
        Produce 1D histograms for elements given in `event_1D_header_keys` from the CORSIKA event
        header and save into ecsv/png files.
    png: bool
        If true, histograms are saved into png files.
    ecsv: bool
        If true, histograms are saved into ecsv files.
    """
    for i_event_header_element, _ in enumerate(event_2D_header_keys[::2]):
        if png:
            figure, figure_name = corsika_output_visualize.plot_2D_event_header_distribution(
                instance,
                event_2D_header_keys[i_event_header_element],
                event_2D_header_keys[i_event_header_element + 1],
            )
            output_file_name = Path(output_dir).joinpath(figure_name)
            logger.info(f"Saving histogram to {output_file_name}")
            figure.savefig(output_file_name, bbox_inches="tight")
        if ecsv:
            instance.export_event_header_2D_histogram(
                event_2D_header_keys[i_event_header_element],
                event_2D_header_keys[i_event_header_element + 1],
                bins=50,
                hist_range=None,
            )


def main():

    label = Path(__file__).stem
    description = "Generate histograms for the Cherenkov photons saved in the CORSIKA IACT file."
    usage = ""
    args_dict, _ = _parse(label, description, usage)

    io_handler_instance = io_handler.IOHandler()

    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))
    initial_time = time.time()
    logger.info("Starting the application.")

    instance = CorsikaOutput(args_dict["IACT_file"])
    instance.set_histograms(
        telescope_indices=args_dict["telescope_indices"],
        individual_telescopes=args_dict["individual_telescopes"],
        hist_config=args_dict["hist_config"],
    )

    if args_dict["output_directory"] is None:
        output_dir = io_handler_instance.get_output_directory(label, dir_type="application-plots")
    else:
        output_dir = args_dict["output_directory"]

    # Cherenkov photons
    if args_dict["png"]:
        _plot_figures(instance=instance, output_dir=output_dir)
    if args_dict["ecsv"]:
        instance.export_histograms(output_dir=output_dir)

    # Event information
    if args_dict["event_1D_histograms"] is not None:
        _run_event_1D_histograms(
            instance,
            output_dir,
            args_dict["event_1D_histograms"],
            args_dict["png"],
            args_dict["ecsv"],
        )
    if args_dict["event_2D_histograms"] is not None:
        _run_event_2D_histograms(
            instance,
            output_dir,
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
