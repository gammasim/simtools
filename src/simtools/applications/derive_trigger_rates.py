r"""
Derive cosmic-ray trigger rates for a single telescope or an array of telescopes.

Uses simulated background events (e.g. from proton primaries) to calculate the trigger rates.
Input is reduced event data generated from simulations for the given configuration.


Command line arguments
----------------------
event_data_file (str, required)
    Event data file containing reduced event data.
array_layout_name (list, optional)
    Name of the array layout to use for the simulation.
telescope_ids (str, optional)
    Path to a file containing telescope configurations.
plot_histograms (bool, optional)
    Plot histograms of the event data.
model_version (str, optional)
    Version of the simulation model to use.
site (str, optional)
    Name of the site where the simulation is being run.


Example
-------

Derive trigger rates for the South Alpha layout:

.. code-block:: console

    simtools-derive-trigger-rates \\
        --site South \\
        --model_version 6.0.0 \\
        --event_data_file /path/to/event_data_file.h5 \\
        --array_layout_name alpha\\
        --plot_histograms

"""

from simtools.application_startup import get_application_label, startup_application
from simtools.configuration import configurator
from simtools.telescope_trigger_rates import telescope_trigger_rates


def _parse():
    """Parse command line configuration."""
    config = configurator.Configurator(
        label=get_application_label(__file__),
        description="Derive trigger rates for a single telescope or an array of telescopes.",
    )
    config.parser.add_argument(
        "--event_data_file",
        type=str,
        required=True,
        help="Event data file containing reduced event data.",
    )
    config.parser.add_argument(
        "--telescope_ids",
        type=str,
        required=False,
        help="Path to a file containing telescope configurations.",
    )
    config.parser.add_argument(
        "--plot_histograms",
        help="Plot histograms of the event data.",
        action="store_true",
        default=False,
    )
    return config.initialize(
        db_config=True,
        output=True,
        simulation_model=[
            "site",
            "model_version",
            "layout",
        ],
    )


def main():
    """Derive trigger rates for a single telescope or an array of telescopes."""
    args_dict, db_config, _, _ = startup_application(_parse)

    telescope_trigger_rates(args_dict, db_config)


if __name__ == "__main__":
    main()
