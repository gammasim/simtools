#!/usr/bin/env python
"""Module containing the calculate_incident_angles application."""

import logging
from pathlib import Path

import astropy.units as u

from simtools.configuration import configurator
from simtools.ray_tracing.incident_angles import IncidentAnglesCalculator


def _parse(label):
    """Parse command line configuration."""
    config = configurator.Configurator(
        label=label,
        description="Calculate incident angles using sim_telarray ray tracing.",
    )

    # Application-specific arguments
    config.parser.add_argument(
        "--zenith",
        help="Zenith angle in degrees",
        type=float,
        default=20.0,
        required=False,
    )
    config.parser.add_argument(
        "--off_axis_angle",
        help="Off axis angle in degrees",
        type=float,
        default=0.0,
        required=False,
    )
    config.parser.add_argument(
        "--source_distance",
        help="Source distance in kilometers",
        type=float,
        default=10.0,
        required=False,
    )
    config.parser.add_argument(
        "--number_of_rays",
        help="Number of rays to simulate per pixel",
        type=int,
        default=10000,
        required=False,
    )
    config.parser.add_argument(
        "--ray_tracing_config",
        help="Path to ray tracing configuration file",
        type=str,
        required=False,
    )

    # Default configuration
    return config.initialize(db_config=True)


def main():
    """Application to calculate incident angles using ray tracing."""
    label = Path(__file__).stem
    args_dict, db_config = _parse(label)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logger.info("Starting calculation of incident angles")

    output_base = Path(args_dict.get("output_path", "./"))
    output_dir = (
        output_base / label if not args_dict.get("use_plain_output_path", False) else output_base
    )

    # Create the calculator
    calculator = IncidentAnglesCalculator(
        simtel_path=args_dict["simtel_path"],
        db_config=db_config,
        config_data={
            "site": args_dict["site"],
            "telescope": args_dict["telescope"],
            "model_version": args_dict["model_version"],
            "number_of_rays": args_dict["number_of_rays"],
            "zenith_angle": args_dict["zenith"] * u.deg,
            "off_axis_angle": args_dict["off_axis_angle"] * u.deg,
            "source_distance": args_dict["source_distance"] * u.km,
        },
        output_dir=output_dir,
        label=args_dict.get("label", label),
        ray_tracing_config=args_dict.get("ray_tracing_config"),
        test=args_dict.get("test", False),
    )

    # Run the calculation
    results = calculator.run()
    logger.info(f"Calculated incident angles for {len(results)} points")


if __name__ == "__main__":
    main()
