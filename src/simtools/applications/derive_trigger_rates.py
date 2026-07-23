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
cr_spectrum (str, optional)
    Path to a YAML file defining a user-provided cosmic-ray spectrum.
    Supported spectrum types: PowerLaw, LogParabola, PowerLawWithExponentialGaussian.
    If not given, the spectrum is selected from the CTAO spectrum library.


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

Derive trigger rates with a user-defined spectrum:

.. code-block:: console

    simtools-derive-trigger-rates \\
        --site South \\
        --model_version 6.0.0 \\
        --event_data_file /path/to/event_data_file.h5 \\
        --array_layout_name alpha \\
        --cr_spectrum /path/to/spectrum.yml

"""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.telescope_trigger_rates import telescope_trigger_rates

_ARGUMENTS = (
    cli.TELESCOPE_IDS,
    cli.EVENT_DATA_FILE,
    cli.ArgumentDefinition(
        "plot_histograms",
        help="Plot histograms of the event data.",
        action="store_true",
        default=False,
    ),
    cli.ArgumentDefinition(
        "cr_spectrum",
        type=str,
        default=None,
        help=(
            "Path to a YAML file defining the cosmic-ray spectrum. Supported types: "
            "PowerLaw, LogParabola, PowerLawWithExponentialGaussian. If not given, "
            "the spectrum is selected from the CTAO spectrum library."
        ),
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        cli.MODEL_VERSION,
        cli.OVERWRITE_MODEL_PARAMETERS,
        cli.IGNORE_MISSING_DESIGN_MODEL,
        cli.SITE,
        *cli.layout_selection_arguments(),
        *cli.PATH_ARGUMENTS,
        *cli.OUTPUT_ARGUMENTS,
    ),
    database=True,
    initialize_output=True,
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()

    telescope_trigger_rates(app_context.args)


if __name__ == "__main__":
    main()
