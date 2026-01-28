#!/usr/bin/python3

r"""
    Generates a set of histograms with Cherenkov photon distributions from CORSIKA output.

    The Cherenkov photons (from observation level) are read from a CORSIKA IACT
    output file(s) provided as input.

    The following 2D histograms are generated:

        - Density of Cherenkov photons on the ground
        - Incoming direction (directive cosines) of the Cherenkov photons
        - Time of arrival (ns) vs altitude of production (km)

    The following 1D histograms are generated:

        - Wavelength distribution of Cherenkov photons
        - Time of arrival (ns) distribution of Cherenkov photons
        - Altitude of production (km) distribution of Cherenkov photons
        - Lateral distribution of Cherenkov photons (distance from shower core in m)

    Command line arguments
    ----------------------
    input_files (str, required)
        The name(s) of the CORSIKA IACT file(s) resulted from the CORSIKA simulation.

    pdf_file_name (str, optional)
        The name of the output pdf file to save the histograms. If not provided,
        the histograms are only shown on screen.

    file_labels (str, optional)
        Labels for the input files (in the same order as input_files). If not provided,
        the file names are used as labels.

    Example
    -------
    Fill and plot histograms for a test IACT file:

     .. code-block:: console

        simtools-generate-corsika-histograms --input_files /workdir/external/simtools/\\
        tests/resources/tel_output_10GeV-2-gamma-20deg-CTAO-South.corsikaio \\
            --pdf_file_name test.pdf

    Fill and plot histograms for several files:

     .. code-block:: console

        simtools-generate-corsika-histograms --input_files file1 file 2 \\
            --file_lablels label1 label2 \\
            --pdf_file_name test.pdf

"""

from astropy import units as u

from simtools.application_control import get_application_label, startup_application
from simtools.configuration import configurator
from simtools.corsika.corsika_histograms import CorsikaHistograms
from simtools.visualization import plot_corsika_histograms


def _parse():
    """Parse command line configuration."""
    config = configurator.Configurator(
        label=get_application_label(__file__),
        description="Generate histograms for the Cherenkov photons saved in the CORSIKA IACT file.",
    )
    config.parser.add_argument(
        "--input_files",
        help="Name(s) of the CORSIKA IACT file(s) to process",
        type=str,
        nargs="+",
        required=True,
    )
    config.parser.add_argument(
        "--file_labels",
        help="Labels for the input files (in the same order as input_files)",
        type=str,
        nargs="+",
        required=None,
    )
    config.parser.add_argument(
        "--normalization",
        help="Normalization method for histograms. Options: 'per-telescope', 'per-bin'",
        type=str,
        choices=["per-telescope", "per-bin"],
        default="per-telescope",
    )
    config.parser.add_argument(
        "--axis_distance",
        help=(
            "Distance from x/y axes to consider when calculating "
            "the lateral density profiles (in meters)."
        ),
        type=float,
        default=1000.0,
    )
    config.parser.add_argument(
        "--pdf_file_name",
        help="Save histograms into a pdf file.",
        type=str,
        required=None,
    )
    return config.initialize(db_config=False, paths=True)


def main():
    """Generate a set of histograms for the Cherenkov photons from CORSIKA IACT file(s)."""
    app_context = startup_application(_parse)

    all_histograms = []
    for input_file in app_context.args["input_files"]:
        corsika_histograms = CorsikaHistograms(
            input_file,
            normalization_method=app_context.args["normalization"],
            axis_distance=app_context.args["axis_distance"] * u.m,
        )
        corsika_histograms.fill()
        all_histograms.append(corsika_histograms)

    plot_corsika_histograms.export_all_photon_figures_pdf(
        all_histograms,
        app_context.io_handler.get_output_file(app_context.args.get("pdf_file_name")),
        app_context.args.get("file_labels"),
    )


if __name__ == "__main__":
    main()
