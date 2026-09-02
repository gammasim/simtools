#!/usr/bin/python3

"""Simulate pedestal events."""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.simulator import Simulator

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "run_mode",
        help="Calibration run mode",
        type=str,
        required=True,
        choices=["pedestals", "pedestals_dark", "pedestals_nsb_only"],
    ),
    cli.ArgumentDefinition(
        "number_of_events", help="Number of pedestal events to simulate", type=int, required=True
    ),
    cli.ArgumentDefinition(
        "nsb_scaling_factor",
        help=(
            "Scaling factor for the NSB rate. Default is 1.0, corresponding to the nominal "
            "(dark sky) NSB rate."
        ),
        type=float,
        required=False,
        default=1.0,
    ),
    cli.ArgumentDefinition(
        "stars", help="List of stars (azimuth, zenith, weighting factor).", type=str, default=None
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        cli.MODEL_VERSION,
        cli.OVERWRITE_MODEL_PARAMETERS,
        cli.SITE,
        *cli.layout_selection_arguments(),
        cli.RUN_NUMBER,
        cli.AZIMUTH_ANGLE,
        cli.ZENITH_ANGLE,
        *cli.SIM_TELARRAY_ARGUMENTS,
        *cli.OUTPUT_PATH_ARGUMENTS,
        *cli.SIM_TELARRAY_PATH_ARGUMENTS,
    ),
    database=True,
    validate_simulation_dependencies=True,
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()

    simulator = Simulator(
        label=app_context.args.get("label"), model_reader=app_context.model_reader
    )
    simulator.simulate()
    simulator.validate_simulations()


if __name__ == "__main__":
    main()
