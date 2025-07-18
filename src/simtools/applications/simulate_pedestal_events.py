#!/usr/bin/python3

r"""
Simulate pedestal events for calibration purposes.

Example
-------

Simulate pedestal events for MSTN-04 at the North site:

    .. code-block:: console

        simtools-simulate-pedestal-events --telescope MSTN-04 --site North \
        --number_of_events 1000 --model_version 6.0.0

Command Line Arguments
----------------------
telescope (str, required)
    Telescope model name (e.g. LSTN-01, SSTS-design, SSTS-25, ...)
site (str, required)
    Site name (North or South).
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
    config = configurator.Configurator(
        label=label,
        description=("Simulate pedestal events for calibration purposes."),
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
    # TODO duplication with simulate_prod.py, refactor to a common function
    sim_telarray_seed_group = config.parser.add_argument_group(
        title="Random seeds for sim_telarray instrument setup",
    )
    sim_telarray_seed_group.add_argument(
        "--sim_telarray_instrument_seeds",
        help=(
            "Random seed used for sim_telarray instrument setup. "
            "If '--sim_telarray_random_instrument_instances is not set: use as sim_telarray seed "
            " ('random_seed' parameter). "
            "Otherwise: use as base seed for the generation of random instrument instance seeds."
        ),
        type=str,
        required=False,
    )
    sim_telarray_seed_group.add_argument(
        "--sim_telarray_random_instrument_instances",
        help="Number of random instrument instances initialized in sim_telarray.",
        type=int,
        required=False,
    )
    return config.initialize(
        db_config=True,
        simulation_model=["site", "layout", "telescope", "model_version"],
        simulation_configuration={
            "software": None,
            "corsika_configuration": ["run_number", "nshow", "azimuth_angle", "zenith_angle"],
        },
    )


def main():  # noqa: D103
    label = Path(__file__).stem
    args_dict, db_config = _parse(label)

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    simulator = Simulator(
        label=args_dict.get("label"),
        args_dict=args_dict,
        db_config=db_config,
        run_mode="pedestals",
    )
    simulator.simulate()


if __name__ == "__main__":
    main()
