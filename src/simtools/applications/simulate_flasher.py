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

1. Simulate flashers for a telescope (direct injection):

    .. code-block:: console

        simtools-simulate-flasher --run_mode direct_injection \
        --light_source MSFx-FlashCam --model_version 6.0.0 \
        --array_layout_name subsystem_msts --site South \
        --run_number 3

2. Simulate flashers for a telescope (detailed simulation):

    .. code-block:: console

        simtools-simulate-flasher --run_mode full_simulation  \
        --light_source MSFx-NectarCam --model_version 6.0    \
        --telescope MSTS-04 --site South --run_number 1 \
        --array_layout_name 1mst

Command Line Arguments
----------------------
run_mode (str, required)
    Run mode, either "direct_injection" or "full_simulation".
telescope (str, required)
    Telescope model name (e.g. LSTN-01, MSTN-04, SSTS-04, ...)
site (str, required)
    Site name (North or South).
light_source (str, required)
    Calibration light source, e.g., MSFx-FlashCam
number_of_events (int, optional):
    Number of events to simulate (default: 1).
output_prefix (str, optional):
    Prefix for output files (default: empty).
model_version (str, optional)
    Version of the simulation model.
array_layout_name (str, optional)
    Name of the array layout to use (required for direct injection mode).
run_number (int, optional)
    Run number to use (default: 1, required for direct injection mode).
telescope (str, optional)
    Telescope name (required for full simulation mode).
"""

from simtools.application_control import get_application_label, startup_application
from simtools.configuration import configurator
from simtools.simtel.simulator_light_emission import SimulatorLightEmission
from simtools.simulator import Simulator


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
    config.parser.add_argument(
        "--light_source",
        help="Flasher device associated with a specific telescope, i.e. MSFx-FlashCam",
        type=str,
        required=True,
    )
    config.parser.add_argument(
        "--number_of_events",
        help="Number of flasher events to simulate",
        type=int,
        default=1,
        required=False,
    )
    config.parser.add_argument(
        "--output_prefix",
        help="Prefix for output files",
        type=str,
        default=None,
        required=False,
    )
    return config.initialize(
        db_config=True,
        simulation_model=["site", "layout", "telescope", "model_version"],
        simulation_configuration={
            "corsika_configuration": ["run_number"],
            "sim_telarray_configuration": ["all"],
        },
    )


def main():
    """Simulate flasher devices."""
    app_context = startup_application(_parse)

    app_context.logger.info(
        f"Flasher simulation for telescope {app_context.args['telescope']} "
        f" with light source {app_context.args['light_source']} "
        f" ({app_context.args['number_of_events']} events, "
        f"run mode: {app_context.args['run_mode']})"
    )

    if app_context.args["run_mode"] == "full_simulation":
        light_source = SimulatorLightEmission(
            light_emission_config=app_context.args,
            db_config=app_context.db_config,
            label=app_context.args.get("label"),
        )
    elif app_context.args["run_mode"] == "direct_injection":
        light_source = Simulator(
            args_dict=app_context.args,
            db_config=app_context.db_config,
            label=app_context.args.get("label"),
        )
    else:
        raise ValueError(f"Unsupported run_mode: {app_context.args['run_mode']}")

    light_source.simulate()
    app_context.logger.info("Flasher simulation completed.")


if __name__ == "__main__":
    main()
