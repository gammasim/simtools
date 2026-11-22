#!/usr/bin/python3

"""Simulate flasher devices used e.g. for camera flat fielding."""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.configuration.argument_helpers import scientific_int
from simtools.model.model_utils import get_array_elements_for_layout
from simtools.runners import light_emission_runner
from simtools.runners.simulator import Simulator
from simtools.runtime import software_catalog
from simtools.simtel.simulator_light_emission import SimulatorLightEmission
from simtools.utils import value_conversion

_LIGHT_SOURCE_GROUP = "light source"
_TARGET_GROUP = "target"
_MAX_PHOTOELECTRONS_MIN = 1_000
_MAX_PHOTOELECTRONS_MAX = 500_000_000


def _validate_max_photoelectrons(args_dict, _config_sources, parser):
    """Validate the sim_telarray photoelectron buffer limit."""
    max_photoelectrons = args_dict.get("max_photoelectrons")
    if max_photoelectrons is not None and not (
        _MAX_PHOTOELECTRONS_MIN <= max_photoelectrons < _MAX_PHOTOELECTRONS_MAX
    ):
        parser.error("--max_photoelectrons must be at least 1000 and less than 500000000.")
    if args_dict.get("run_mode") in ("direct_injection", "full_simulation") and not (
        args_dict.get("light_source") or args_dict.get("light_source_type")
    ):
        parser.error("A light source is required for pulsed flasher runs.")
    if (
        args_dict.get("run_mode") == "dc"
        and len(value_conversion.ensure_list(args_dict.get("number_of_events"))) != 1
    ):
        parser.error("DC runs require one number of events.")
    if args_dict.get("run_mode") == "dc":
        for option in (
            "light_source",
            "light_source_type",
            "flasher_photons",
            "max_photoelectrons",
        ):
            if args_dict.get(option) is not None:
                parser.error(
                    f"--{option} is not used for DC runs; "
                    "the telescope model selects the DC device."
                )
    elif args_dict.get("flasher_dc_pixel_rate") is not None:
        parser.error("--flasher_dc_pixel_rate requires --run_mode dc.")


def _validate_simulation_software(args_dict, _config_sources, parser):
    """Allow a simulation chain other than the default only in full-simulation mode."""
    chain = args_dict.get("simulation_software")
    if (
        args_dict.get("run_mode") == "direct_injection"
        and chain
        and tuple(chain) != software_catalog.default_chain(software_catalog.LIGHT_SOURCE)
    ):
        parser.error(
            "--simulation_software is available in run mode full_simulation only; "
            "direct_injection runs its own chain."
        )


def _validate(args_dict, config_sources, parser):
    """Validate the arguments of the application."""
    _validate_max_photoelectrons(args_dict, config_sources, parser)
    _validate_simulation_software(args_dict, config_sources, parser)


_ARGUMENTS = (
    cli.ArgumentDefinition(
        "run_mode",
        help="Flasher simulation run mode",
        type=str,
        choices=["direct_injection", "full_simulation", "dc"],
        required=True,
    ),
    cli.ArgumentDefinition(
        "light_source",
        exclusive_group=_LIGHT_SOURCE_GROUP,
        exclusive_group_required=False,
        help="Flasher device associated with a specific telescope, i.e. MSFx-FlashCam",
        type=str,
    ),
    cli.ArgumentDefinition(
        "light_source_type",
        exclusive_group=_LIGHT_SOURCE_GROUP,
        exclusive_group_required=False,
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
    cli.ArgumentDefinition(
        "flasher_dc_pixel_rate",
        help="Mean DC rate in GHz; defaults to the model value.",
        type=float,
        default=None,
    ),
    cli.ArgumentDefinition(
        "nsb_scaling_factor", type=float, default=1.0, help="NSB scaling factor."
    ),
    cli.ArgumentDefinition("stars", type=str, default=None, help="Stars contributing to NSB."),
    cli.ArgumentDefinition(
        "max_workers",
        group="execution",
        help="Maximum parallel direct-injection runs (default: 1, sequential).",
        type=int,
        default=1,
    ),
    cli.ArgumentDefinition(
        "max_photoelectrons",
        group="sim_telarray configuration",
        help=(
            "Override sim_telarray's maximum photoelectrons per telescope event (1000-499999999)."
        ),
        type=int,
        default=None,
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
        cli.AZIMUTH_ANGLE,
        cli.ZENITH_ANGLE,
        *cli.SIM_TELARRAY_ARGUMENTS,
        *cli.OUTPUT_PATH_ARGUMENTS,
        *cli.SIM_TELARRAY_PATH_ARGUMENTS,
        cli.SIMULATION_SOFTWARE(
            default=list(software_catalog.default_chain(software_catalog.LIGHT_SOURCE)),
            help=(
                "Stages of the full simulation in order, each as STAGE or STAGE:KEY=VALUE,... "
                "(e.g. obdeect_flasher sim_telarray:bypass_optics=camera_entrance); converters "
                "between photon formats are inserted."
            ),
        ),
    ),
    post_parse=_validate,
    needs=("model", "sim_telarray", "corsika"),
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()

    if app_context.args["run_mode"] == "dc":
        args = dict(app_context.args)
        args["run_mode"] = "pedestals_flasher_dc"
        args["number_of_events"] = value_conversion.ensure_list(args["number_of_events"])[0]
        simulator = Simulator(
            label=args.get("label"),
            model_reader=app_context.model_reader,
            context=app_context.replace(args=args),
        )
        simulator.simulate()
        simulator.validate_simulations()
        return

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
                app_context.args["array_layout_name"],
                model_reader=app_context.model_reader,
                context=app_context,
            )
            if app_context.args.get("array_layout_name") is not None
            else value_conversion.ensure_list(app_context.args["telescopes"])
        )
        script_writer = light_emission_runner.script_writer_for_chain(
            app_context.args.get("simulation_software")
        )
        for tel in telescopes:
            light_source = SimulatorLightEmission(
                light_emission_config=app_context.args,
                telescope=tel,
                label=app_context.args.get("label"),
                model_reader=app_context.model_reader,
                context=app_context,
            )
            light_source.simulate(script_writer=script_writer)
            light_source.validate_simulations()
    elif app_context.args["run_mode"] == "direct_injection":
        Simulator.simulate_direct_injection_sequence(
            label=app_context.args.get("label"), context=app_context
        )
    else:
        raise ValueError(f"Unsupported run_mode: {app_context.args['run_mode']}")

    app_context.logger.info("Flasher simulation completed.")


if __name__ == "__main__":
    main()
