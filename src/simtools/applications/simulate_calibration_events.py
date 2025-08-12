#!/usr/bin/python3

r"""
Simulate calibration events like pedestal or flasher events.

Use sim_telarray to simulate calibration events for an array of telescopes.
The following types of calibration events are supported:

* Pedestal events (includes night-sky background and camera noise)
* Dark pedestal events (closed camera lid, camera noise only)
* Flasher events (simulated flasher light source)

Example
-------

Simulate pedestal events for alpha North. The assumed level night-sky background is 2.0 times the
nominal value. A list of stars is provided to simulate to simulate additional contributions.

.. code-block:: console

    simtools-simulate-calibration-events --run_mode=pedestals \\
        --array_layout_name alpha --site North \\
        --number_of_events 1000 --model_version 6.0.0 \\
        --zenith_angle 20 --azimuth_angle 0 \\
        --nsb_scaling_factor 2.0 --stars stars.txt

Simulate flasher events for alpha South. Note that the same flasher configuration is used
for all telescopes.

.. code-block:: console

    simtools-simulate-calibration-events --run_mode=flasher \\
        --number_of_events 1000 \\
        --array_layout_name subsystem_msts --site South \\
        --model_version 6.0.0 \\
        --zenith_angle 20 --azimuth_angle 0 \\
        --flasher_photons 500 --flasher_var_photons 0.05 \\
        --flasher_exp_time 1.59 --flasher_sig_time 0.4

Command Line Arguments
----------------------
run_mode (str, required)
    Run mode, e.g. "pedestals" or "flasher".
number_of_events (int, required)
    Number of calibration events to simulate.
array_layout_name (str, required)
    Array layout name, e.g. "alpha".
site (str, required)
    Site name.
model_version (str, optional)
    Version of the simulation model.
nsb_scaling_factor (float, optional)
    Scaling factor for the night-sky background rate. Default is 1.0, which
    corresponds to the nominal (dark sky) NSB rate.
stars (str, optional)
    Path to a file containing a list of stars (azimuth, zenith, weighting factor
    separated by whitespace). If provided, the stars will be used to simulate
    additional contributions to the night-sky background.
zenith_angle (float, optional)
    Zenith angle in degrees.
azimuth_angle (float, optional)
    Azimuth angle in degrees.
flasher_photons (float, optional)
    Number of photons in the flasher pulse at each photodetector.
flasher_var_photons (float, optional)
    Relative variance of the number of photons in the flasher pulse.
flasher_exp_time (float, optional)
    Exponential decay time of the flasher pulse in nano-seconds.
flasher_sig_time (float, optional)
    Sigma of Gaussian-shaped flasher pulse in nano-seconds.

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
        choices=["pedestals", "dark_pedestals", "nsb_only_pedestals", "flasher"],
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
            "Default is 1.0, which corresponds to the nominal (dark sky) NSB rate."
        ),
        type=float,
        required=False,
        default=1.0,
    )
    config.parser.add_argument(
        "--stars",
        help="List of stars (azimuth, zenith, weighting factor).",
        type=str,
        default=None,
    )
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
        help="Exponential decay time of the flasher pulse in nanoseconds.",
        type=float,
        default=0.0,
    )
    flasher_args.add_argument(
        "--flasher_sig_time",
        help="Sigma of Gaussian-shaped flasher pulse in nanoseconds.",
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
