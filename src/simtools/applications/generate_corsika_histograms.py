#!/usr/bin/python3

"""Generates a set of histograms with Cherenkov photon distributions from CORSIKA output."""

from astropy import units as u

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.corsika.corsika_histograms import CorsikaHistograms
from simtools.visualization import plot_corsika_histograms

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "input_files",
        help="Name(s) of the CORSIKA IACT file(s) to process",
        type=str,
        nargs="+",
        required=True,
    ),
    cli.ArgumentDefinition(
        "file_labels",
        help="Labels for the input files (in the same order as input_files)",
        type=str,
        nargs="+",
        required=None,
    ),
    cli.ArgumentDefinition(
        "normalization",
        help="Normalization method for histograms. Options: 'per-telescope', 'per-bin'",
        type=str,
        choices=["per-telescope", "per-bin"],
        default="per-telescope",
    ),
    cli.ArgumentDefinition(
        "axis_distance",
        help=(
            "Distance from x/y axes to consider when calculating lateral density profiles "
            "(in meters)."
        ),
        type=float,
        default=1000.0,
    ),
    cli.ArgumentDefinition(
        "pdf_file_name", help="Save histograms into a pdf file.", type=str, required=None
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        *cli.OUTPUT_PATH_ARGUMENTS,
    ),
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()

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
