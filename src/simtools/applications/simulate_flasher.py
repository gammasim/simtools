#!/usr/bin/python3

r"""
Simulate flasher devices used e.g. for camera flat fielding.

The flasher simulation allows for two different run modes:

1. Direct injection of light into the camera (bypassing the telescope optics).
2. Simulation of the full light path (using the light-emission package from sim_telarray).

The direct injection mode uses a simplified model for the flasher light source. Both run modes
provide events in sim_telarray format that can be processed by standard analysis steps or
visualized using e.g. the 'simtools-plot-simtel-events' application.

Example Usage
-------------

1. Simulate a single telescope:

    .. code-block:: console

        simtools-simulate-flasher --run_mode full_simulation \
        --light_source_type flat_fielding --model_version 7.0.0 \
        --site North --telescopes MSTN-04 --run_number 10

2. Simulate several telescopes:

    .. code-block:: console

        simtools-simulate-flasher --run_mode full_simulation \
        --light_source_type flat_fielding --model_version 7.0.0 \
        --site North --telescopes MSTN-04 MSTN-05 --run_number 10

3. Simulate all telescopes from an array layout:

    .. code-block:: console

        simtools-simulate-flasher --run_mode full_simulation \
        --light_source_type flat_fielding --model_version 7.0.0 \
        --site North --array_layout_name alpha --run_number 10

4. Simulate flashers for direct injection:

    .. code-block:: console

        simtools-simulate-flasher --run_mode direct_injection \
        --light_source MSFx-FlashCam --model_version 7.0.0 \
        --array_layout_name subsystem_msts --site South \
        --run_number 3

Command Line Arguments
----------------------
run_mode (str, required)
    Run mode, either "direct_injection" or "full_simulation".
telescopes (str, optional)
    One or more telescope names (e.g. LSTN-01, MSTN-04, SSTS-04, ...).
    Use for single-telescope or multi-telescope simulations.
array_layout_name (str, optional)
    Name of the array layout. In full-simulation mode, all telescopes from this layout
    are simulated (one run per telescope).
site (str, required)
    Site name (North or South).
light_source (str, optional)
    Explicit calibration light source model, e.g. MSFx-FlashCam.
light_source_type (str, optional)
    Light source type, e.g. flat_fielding. Recommended for array-style simulations
    because the corresponding flasher model is read from the model-parameter database
    for each telescope.
number_of_events (int, optional):
    Number of events to simulate (default: 1).
flasher_photons (int, optional)
    Overwrite the model parameter flasher_photons. Applies to both run modes.
model_version (str, optional)
    Version of the simulation model.
run_number (int, optional)
    Run number to use (default: 1, required for direct injection mode).
"""

from simtools.application_control import get_application_label, startup_application
from simtools.configuration import configurator
from simtools.model.model_utils import get_array_elements_for_layout
from simtools.simtel.simulator_light_emission import SimulatorLightEmission
from simtools.simulator import Simulator
from simtools.utils import general


def _parse():
    """Parse command line configuration."""
    config = configurator.Configurator(
        label=get_application_label(__file__), description="Simulate flasher devices."
    )
    config.parser.add_argument(
        "--run_mode",
        help="Flasher simulation run mode",
        type=str,
        choices=["direct_injection", "full_simulation"],
        required=True,
        default="direct_injection",
    )
    group = config.parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--light_source",
        help="Flasher device associated with a specific telescope, i.e. MSFx-FlashCam",
        type=str,
    )
    group.add_argument(
        "--light_source_type",
        help="Type of the light source (e.g. flat_fielding)",
        type=str,
    )
    target_group = config.parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--telescopes",
        help="One or more telescopes (e.g. LSTN-01, MSTN-04, SSTS-04)",
        type=config.parser.telescope,
        nargs="+",
    )
    target_group.add_argument(
        "--array_layout_name",
        help="Array layout name(s) (e.g. alpha, subsystem_msts)",
        nargs="+",
        type=str,
    )
    config.parser.add_argument(
        "--number_of_events",
        help="Number of flasher events to simulate",
        type=int,
        default=1,
        required=False,
    )
    config.parser.add_argument(
        "--flasher_photons",
        help=(
            "Override flasher photon yield (single value for all telescopes). "
            "Accepts integers including scientific notation, e.g. 1e6."
        ),
        type=config.parser.scientific_int,
        required=False,
    )
    return config.initialize(
        db_config=True,
        simulation_model=["site", "model_version"],
        simulation_configuration={
            "corsika_configuration": ["run_number"],
            "sim_telarray_configuration": ["all"],
        },
    )


def main():
    """Simulate flasher devices."""
    app_context = startup_application(_parse)

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
            get_array_elements_for_layout(app_context.args["array_layout_name"])
            if app_context.args.get("array_layout_name") is not None
            else general.ensure_iterable(app_context.args["telescopes"])
        )
        for telescope in telescopes:
            light_source = SimulatorLightEmission(
                light_emission_config=app_context.args,
                telescope=telescope,
                label=app_context.args.get("label"),
            )
            light_source.simulate()
            light_source.validate_simulations()
    elif app_context.args["run_mode"] == "direct_injection":
        light_source = Simulator(label=app_context.args.get("label"))
        light_source.simulate()
        light_source.validate_simulations()
    else:
        raise ValueError(f"Unsupported run_mode: {app_context.args['run_mode']}")

    app_context.logger.info("Flasher simulation completed.")


if __name__ == "__main__":
    main()
