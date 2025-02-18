#!/usr/bin/python3

r"""
Derive simulation configuration parameters for a simulation production.

Derived simulation configuration parameters include:

* energy range
* shower core scatter radius
* view cone radius
* total number of events to be simulated

Configuration parameters depend on characteristics of the observations, especially elevation,
azimuth, and night sky background.

The configuration parameters are derived according to the required precision. The metrics are:

* statistical uncertainty on the determination of the effective area as function of primary energy
* fraction of lost events to the selected core scatter and view cone radius (to be implemented)
* statistical uncertainty of the energy migration matrix as function of primary energy
    (to be implemented)

Command line arguments
----------------------
azimuth (float, required)
        Azimuth angle in degrees.
elevation (float, required)
        Elevation angle in degrees.
nsb (float, required)
        Night sky background value.
file_path (str, required)
        Path to file with MC events at CTAO DL2 data level.
        Used for statistical uncertainty evaluation.
file_type (str, required)
        Type of the DL2 MC event file ('point-like' or 'cone').
metrics (str, required)
        Path to a YAML file containing metrics for evaluation.
site (str, required)
        The observatory site (North or South).

Example
-------
To run the simulation configuration, execute the script as follows:

.. code-block:: console

        simtools-production-generate-simulation-config --azimuth 60.0 --elevation 45.0 \
                --nsb 0.3 --file_path tests/resources/production_dl2_fits/dl2_mc_events_file.fits \
                --file_type "point-like"    \
                --metrics_file tests/resources/production_simulation_config_metrics.yml --site North

The output will show the derived simulation parameters.
"""

import json
import logging
from pathlib import Path

import astropy.units as u

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.data_model import schema
from simtools.io_operations import io_handler
from simtools.production_configuration.generate_simulation_config import (
    SimulationConfig,
)


def _parse(label):
    """Parse command-line arguments."""
    config = configurator.Configurator(
        label=label,
        description="Derive simulation configuration parameters for a simulation production.",
    )
    config.parser.add_argument(
        "--azimuth", type=float, required=True, help="Azimuth angle in degrees."
    )
    config.parser.add_argument(
        "--elevation", type=float, required=True, help="Elevation angle in degrees."
    )
    config.parser.add_argument(
        "--nsb", type=float, required=True, help="Night sky background in units of 1/(sr*ns*cm**2)."
    )
    config.parser.add_argument(
        "--file_path", type=str, required=True, help="Path to MC event file in DL2 format."
    )
    config.parser.add_argument(
        "--file_type",
        type=str,
        required=True,
        help="Type of the DL2 MC event file ('point-like' or 'cone').",
    )
    config.parser.add_argument(
        "--metrics_file",
        required=True,
        type=str,
        help="Path to YAML file containing metrics and required precision as values.",
    )
    config.parser.add_argument(
        "--site",
        type=str,
        required=True,
        help="Site location for the simulation (e.g., 'North', 'South').",
    )
    config.parser.add_argument(
        "--output_file",
        help="Name of the file to save the configured simulation parameters.",
        type=str,
        required=False,
        default="configured_simulation_params.json",
    )

    return config.initialize(db_config=False)


def main():
    """Run the Simulation Config application."""
    label = Path(__file__).stem
    args_dict, _ = _parse(label)

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))
    output_path = io_handler.IOHandler().get_output_directory(label)
    output_filepath = Path(output_path).joinpath(f"{args_dict['output_file']}")

    grid_point_config = {
        "azimuth": args_dict["azimuth"],
        "elevation": args_dict["elevation"],
        "night_sky_background": args_dict["nsb"],
    }

    metrics = gen.collect_data_from_file(args_dict["metrics_file"])
    schema.validate_dict_using_schema(
        data=metrics, schema_file="production_configuration_metrics.schema.yml"
    )

    simulation_config = SimulationConfig(
        grid_point=grid_point_config,
        file_path=args_dict["file_path"],
        file_type=args_dict["file_type"],
        metrics=metrics,
    )

    simulation_params = simulation_config.configure_simulation()

    serializable_config = {}

    for key, value in simulation_params.items():
        if isinstance(value, u.Quantity):
            serializable_config[key] = f"{value.value} {value.unit}"
        else:
            serializable_config[key] = value

    logger.info(f"Simulation configuration: {serializable_config}")
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(serializable_config, f, indent=4)
    logger.info(f"Simulation configuration saved to: {output_filepath}")


if __name__ == "__main__":
    main()
