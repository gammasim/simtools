#!/usr/bin/python3

"""Plot simulated event distributions for shower and/or triggered event data."""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.visualization import plot_simtel_event_histograms

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "trigger_histogram_file",
        help="Precomputed trigger-histogram HDF5 file from simtools-write-trigger-histograms.",
        type=str,
        required=True,
    ),
    cli.ArgumentDefinition(
        "array_layout_name",
        help=(
            "Optional array layout name to select from a precomputed trigger-histogram "
            "file. If omitted, plot all layouts available in the file."
        ),
        type=str,
        required=False,
        default=None,
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        *cli.OUTPUT_PATH_ARGUMENTS,
        *cli.OUTPUT_ARGUMENTS,
    ),
    initialize_output=True,
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()
    output_dir = app_context.io_handler.get_output_directory()

    app_context.logger.info(
        f"Loading trigger histogram file from: {app_context.args['trigger_histogram_file']}"
    )
    plot_simtel_event_histograms.plot_trigger_histogram_file(
        app_context.args["trigger_histogram_file"],
        output_dir,
        app_context.args.get("array_layout_name"),
    )


if __name__ == "__main__":
    main()
