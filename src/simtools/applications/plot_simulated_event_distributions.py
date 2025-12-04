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

    simtools-plot-simulated-event-distributions --input_file path/to/simtel_file.hdf5 \
                                                --output_path simtools_output/


"""

from simtools.application_control import get_application_label, startup_application
from simtools.configuration import configurator
from simtools.sim_events.histograms import EventDataHistograms
from simtools.visualization import plot_simtel_event_histograms


def _parse():
    """Parse command line configuration."""
    config = configurator.Configurator(
        label=get_application_label(__file__),
        description="Plot simulated event distributions for shower and/or triggered event data.",
    )
    config.parser.add_argument("--input_file", type=str, required=True, help="Input file path")
    return config.initialize(db_config=False, output=True)


def main():
    """Plot simulated event distributions."""
    app_context = startup_application(_parse)
    app_context.logger.info(f"Loading input file from: {app_context.args['input_file']}")

    histograms = EventDataHistograms(app_context.args["input_file"])
    histograms.fill()
    plot_simtel_event_histograms.plot(
        histograms.histograms, output_path=app_context.io_handler.get_output_directory()
    )


if __name__ == "__main__":
    main()
