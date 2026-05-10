#!/usr/bin/python3

r"""
Plot simulated event distributions for shower and/or triggered event data.

Reads reduced event data files and generate histogram plots e.g. for energy or
core distance distributions.

Command line arguments
----------------------
input_file (str, required)
    Input file path.
output_path (str, required)
    Output directory for the generated plots.

Examples
--------
Generate plots from a given input file:

.. code-block:: console

    simtools-plot-simulated-event-distributions --event_data_file path/to/simtel_file.hdf5 \
                                                --output_path simtools_output/


"""

from simtools.application_control import build_application
from simtools.sim_events.histograms import EventDataHistograms
from simtools.visualization import plot_simtel_event_histograms


def _add_arguments(parser):
    """Register application-specific command line arguments."""
    parser.initialize_application_arguments(["event_data_file"])


def main():
    """See CLI description."""
    app_context = build_application(
        initialization_kwargs={"db_config": False, "output": True},
    )
    app_context.logger.info(f"Loading event data file from: {app_context.args['event_data_file']}")

    histograms = EventDataHistograms(app_context.args["event_data_file"])
    histograms.fill()
    plot_simtel_event_histograms.plot(
        histograms.histograms, output_path=app_context.io_handler.get_output_directory()
    )


if __name__ == "__main__":
    main()
