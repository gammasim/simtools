#!/usr/bin/python3

r"""
Simulate calibration events like pedestal or flasher events.

Example
-------

Simulate pedestal events for MSTN-04 at the North site:

    .. code-block:: console

        simtools-simulate-calibration-events --run_mode="pedestals \\
            --telescope MSTN-04 --site North \\
            --number_of_events 1000 --model_version 6.0.0

Command Line Arguments
----------------------
run_mode (str, required)
    Run mode, e.g. "pedestals" or "flasher".
number_of_events (int, required)
    Number of calibration events to simulate.
telescope (str, required)
    Telescope model name (e.g. LSTN-01, SSTS-design, SSTS-25, ...)
site (str, required)
    Site name.
model_version (str, optional)
    Version of the simulation model.

"""

import logging
from pathlib import Path

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.simulator import Simulator


def _parse(label):
    """Parse command line configuration."""
    config = configurator.Configurator(label=label, description="Simulate calibration events.")
    config.parser.add_argument(
        "--run_mode",
        help="Calibration run mode",
        type=str,
        required=True,
        choices=["pedestals", "dark_pedestals", "flasher"],
    )
    config.parser.add_argument(
        "--number_of_events",
        help="Number of calibration events to simulate",
        type=int,
        required=True,
    )
    config.parser.add_argument(
        "--nsb_scaling_factor",
        help=(
            "Scaling factor for the NSB rate. "
            "Default is 1.0, which corresponds to the nominal NSB rate."
        ),
        type=float,
        required=False,
        default=1.0,
    )
    config.parser.add_argument(
        "--stars",
        help="List of stars (azimuth, zenith, weighting factor).",
        action="store_true",
        default=False,
    )
    # TODO - some parameters should be model parameters and not command line arguments
    flasher_args = config.parser.add_argument_group("Flasher configuration")
    flasher_args.add_argument(
        "--flasher_photons",
        help="Number of photons in the flasher pulse at each photodetector.",
        type=float,
        default=500.0,
    )
    flasher_args.add_argument(
        "--flasher_var_photons",
        help="Relative variance of the number of photons in the flasher pulse.",
        type=float,
        default=0.05,
    )
    flasher_args.add_argument(
        "--flasher_exp_time",
        help="Exponential decay time of the flasher pulse in nano-seconds.",
        type=float,
        default=0.0,
    )
    flasher_args.add_argument(
        "--flasher_sig_time",
        help="Sigma of Gaussian-shaped flasher pulse in nano-seconds.",
        type=float,
        default=0.0,
    )

    return config.initialize(
        db_config=True,
        simulation_model=["site", "layout", "telescope", "model_version"],
        simulation_configuration={
            "corsika_configuration": ["run_number", "azimuth_angle", "zenith_angle"],
            "sim_telarray_configuration": ["all"],
        },
    )


def main():  # noqa: D103
    label = Path(__file__).stem
    args_dict, db_config = _parse(label)

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    simulator = Simulator(label=args_dict.get("label"), args_dict=args_dict, db_config=db_config)
    simulator.simulate()


if __name__ == "__main__":
    main()
