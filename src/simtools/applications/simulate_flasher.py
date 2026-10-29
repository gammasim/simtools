#!/usr/bin/python3

"""Simulate flasher devices used e.g. for camera flat fielding."""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.configuration.argument_helpers import scientific_int
from simtools.model.model_utils import get_array_elements_for_layout
from simtools.simtel.simulator_light_emission import SimulatorLightEmission
from simtools.simulator import Simulator
from simtools.utils import value_conversion

_LIGHT_SOURCE_GROUP = "light source"
_TARGET_GROUP = "target"

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "run_mode",
        help="Flasher simulation run mode",
        type=str,
        choices=["direct_injection", "full_simulation"],
        required=True,
    ),
    cli.ArgumentDefinition(
        "light_source",
        exclusive_group=_LIGHT_SOURCE_GROUP,
        exclusive_group_required=True,
        help="Flasher device associated with a specific telescope, i.e. MSFx-FlashCam",
        type=str,
    ),
    cli.ArgumentDefinition(
        "light_source_type",
        exclusive_group=_LIGHT_SOURCE_GROUP,
        exclusive_group_required=True,
        help="Type of the light source (e.g. flat_fielding)",
        type=str,
    ),
    cli.TELESCOPES(exclusive_group=_TARGET_GROUP, exclusive_group_required=True),
    cli.ARRAY_LAYOUT_NAME(exclusive_group=_TARGET_GROUP, exclusive_group_required=True),
    cli.ArgumentDefinition(
        "number_of_events",
        help="Number of flasher events to simulate",
        type=int,
        default=1,
        nargs="+",
    ),
    cli.ArgumentDefinition(
        "flasher_photons",
        help=(
            "Override flasher photon yield (one value for all telescopes). Accepts integers "
            "including scientific notation, e.g. 1e6."
        ),
        type=scientific_int,
        nargs="+",
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    model_repository=True,
    arguments=(
        *_ARGUMENTS,
        cli.MODEL_VERSION,
        cli.OVERWRITE_MODEL_PARAMETERS,
        cli.SITE,
        cli.RUN_NUMBER,
        *cli.SIM_TELARRAY_ARGUMENTS,
        *cli.OUTPUT_PATH_ARGUMENTS,
        *cli.SIM_TELARRAY_PATH_ARGUMENTS,
        cli.RAY_TRACING_BACKEND,
    ),
    validate_simulation_dependencies=True,
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()

    tel_string = (
        f"telescope(s) {app_context.args['telescopes']}"
        if app_context.args.get("telescopes")
        else f"array layout {app_context.args['array_layout_name']}"
    )

    app_context.logger.info(
        f"Flasher simulation for {tel_string}"
        f" with light source {app_context.args['light_source']} "
        f" ({app_context.args['number_of_events']} events, "
        f"run mode: {app_context.args['run_mode']})"
    )

    if app_context.args["run_mode"] == "full_simulation":
        telescopes = (
            get_array_elements_for_layout(
                app_context.args["array_layout_name"], model_reader=app_context.model_reader
            )
            if app_context.args.get("array_layout_name") is not None
            else value_conversion.ensure_list(app_context.args["telescopes"])
        )
        for tel in telescopes:
            light_source = SimulatorLightEmission(
                light_emission_config=app_context.args,
                telescope=tel,
                label=app_context.args.get("label"),
                model_reader=app_context.model_reader,
            )
            light_source.simulate()
            light_source.validate_simulations()
    elif app_context.args["run_mode"] == "direct_injection":
        Simulator.simulate_direct_injection_sequence(label=app_context.args.get("label"))
    else:
        raise ValueError(f"Unsupported run_mode: {app_context.args['run_mode']}")

    app_context.logger.info("Flasher simulation completed.")


if __name__ == "__main__":
    main()
