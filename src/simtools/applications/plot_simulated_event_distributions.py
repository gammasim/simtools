#!/usr/bin/python3

r"""
Plot simulated event distributions for shower and/or triggered event data.

Reads reduced event data files and generate histogram plots e.g. for energy or
core distance distributions.

Command line arguments
----------------------
trigger_histogram_file (str, required)
    Precomputed trigger-histogram HDF5 file from ``simtools-write-trigger-histograms``.
output_path (str, required)
    Output directory for the generated plots.

Examples
--------
Generate plots from a precomputed trigger-histogram file:

.. code-block:: console

    simtools-plot-simulated-event-distributions \
        --trigger_histogram_file trigger_histograms.hdf5 \
        --output_path simtools_output


"""

from simtools.application_control import build_application
from simtools.production_configuration.trigger_histograms import load_event_data_histograms
from simtools.visualization import plot_simtel_event_histograms


def _add_arguments(parser):
    """Register application-specific command line arguments."""
    parser.add_argument(
        "--trigger_histogram_file",
        help="Precomputed trigger-histogram HDF5 file from simtools-write-trigger-histograms.",
        type=str,
        required=True,
    )


def _plot_histogram_file(trigger_histogram_file, output_dir):
    """Plot all histogram references from a trigger-histogram HDF5 file."""
    loaded_histograms = load_event_data_histograms(trigger_histogram_file)
    for _, histograms in loaded_histograms:
        output_path = output_dir
        if len(loaded_histograms) > 1:
            output_path = output_dir / histograms.array_name
            output_path.mkdir(parents=True, exist_ok=True)
        plot_simtel_event_histograms.plot(histograms.histograms, output_path=output_path)


def main():
    """See CLI description."""
    app_context = build_application(
        initialization_kwargs={"db_config": False, "output": True},
    )
    output_dir = app_context.io_handler.get_output_directory()

    app_context.logger.info(
        f"Loading trigger histogram file from: {app_context.args['trigger_histogram_file']}"
    )
    _plot_histogram_file(app_context.args["trigger_histogram_file"], output_dir)


if __name__ == "__main__":
    main()
