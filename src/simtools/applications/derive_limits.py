#!/usr/bin/python3

r"""
Derives the limits for energy, radial distance, and viewcone using the LimitCalculator.

The limits are derived based on the event loss fraction specified by the user.

Command line arguments
----------------------
hdf5_file (str, required)
    Path to the HDF5 file containing the event data.
loss_fraction (float, required)
    Fraction of events to be lost.


Example
-------
Derive limits for a given HDF5 file with a specified loss fraction.

.. code-block:: console

    simtools-derive-limits\\
        --hdf5_file path/to/hdf5_file.hdf5 \\
        --loss_fraction 0.1
"""

import logging

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.io_operations.hdf5_handler import read_hdf5
from simtools.production_configuration.limits_calculation import LimitCalculator

_logger = logging.getLogger(__name__)


def _parse():
    config = configurator.Configurator(
        description="Derive limits for energy, radial distance, and viewcone."
    )
    config.parser.add_argument(
        "--hdf5_file",
        type=str,
        required=True,
        help="Path to the HDF5 file containing the event data.",
    )
    config.parser.add_argument(
        "--loss_fraction", type=float, required=True, help="Fraction of events to be lost."
    )
    return config.initialize(db_config=False)


def main():
    """Derive limits for energy, radial distance, and viewcone."""
    args_dict, _ = _parse()

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    hdf5_file_path = args_dict["hdf5_file"]
    loss_fraction = args_dict["loss_fraction"]

    _logger.info(f"Loading HDF5 file: {hdf5_file_path}")
    tables = read_hdf5(hdf5_file_path)

    _logger.info("Initializing LimitCalculator")
    calculator = LimitCalculator(tables)

    _logger.info("Computing lower energy limit")
    lower_energy_limit = calculator.compute_lower_energy_limit(loss_fraction)
    _logger.info(f"Lower energy threshold: {lower_energy_limit} TeV")

    _logger.info("Computing upper radial distance")
    upper_radial_distance = calculator.compute_upper_radial_distance(loss_fraction)
    _logger.info(f"Upper radius threshold: {upper_radial_distance} m")

    _logger.info("Computing viewcone")
    viewcone = calculator.compute_viewcone(loss_fraction)
    _logger.info(f"Viewcone threshold: {viewcone} degrees")


if __name__ == "__main__":
    main()
