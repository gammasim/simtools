#!/usr/bin/python3

r"""
    Generates a set of histograms with Cherenkov photon distributions from CORSIKA output.

    The Cherenkov photons (from observation level) are read from a CORSIKA IACT
    output file provided as input.

    The following 2D histograms are generated:

        - Density of Cherenkov photons on the ground
        - Incoming direction (directive cosines) of the Cherenkov photons
        - Time of arrival (ns) vs altitude of production (km)

    Command line arguments
    ----------------------
    input_file (str, required)
        The name of the CORSIKA IACT file resulted from the CORSIKA simulation.

    pdf_file_name (str, optional)
        The name of the output pdf file to save the histograms. If not provided,
        the histograms are only shown on screen.

    Example
    -------
    Generate the histograms for a test IACT file:

     .. code-block:: console

        simtools-generate-corsika-histograms --input_file /workdir/external/simtools/\\
        tests/resources/tel_output_10GeV-2-gamma-20deg-CTAO-South.corsikaio \\
            --pdf_file_name test.pdf

"""

from simtools.application_control import get_application_label, startup_application
from simtools.configuration import configurator
from simtools.corsika.corsika_histograms import CorsikaHistograms


def _parse():
    """Parse command line configuration."""
    config = configurator.Configurator(
        label=get_application_label(__file__),
        description="Generate histograms for the Cherenkov photons saved in the CORSIKA IACT file.",
    )
    config.parser.add_argument(
        "--input_file",
        help="Name of the CORSIKA IACT file from which to generate the histograms.",
        type=str,
        required=True,
    )
    config.parser.add_argument(
        "--pdf_file_name",
        help="Save histograms into a pdf file.",
        type=str,
        required=None,
    )
    return config.initialize(db_config=False, paths=True)


def main():
    """Generate a set of histograms for the Cherenkov photons saved in the CORSIKA IACT file."""
    app_context = startup_application(_parse)

    corsika_histograms = CorsikaHistograms(app_context.args["input_file"])
    corsika_histograms.fill()
    corsika_histograms.plot(pdf_file=app_context.args.get("pdf_file_name"))


if __name__ == "__main__":
    main()
