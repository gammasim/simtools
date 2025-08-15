#!/usr/bin/python3

r"""
Simulate flasher devices (flat fielding) using the light emission package.

Run the application in the command line.

Example Usage
-------------

1. Simulate flashers for an MST telescope:

    .. code-block:: console

        simtools-simulate-flasher --telescope MSTN-04 --site North \
        --flasher FLSN-01 --model_version 6.0.0

2. Simulate flashers for an SST telescope:

    .. code-block:: console

        simtools-simulate-flasher --telescope SSTS-04 --site South \
        --flasher FLSS-01 --model_version 6.0.0

Command Line Arguments
----------------------
telescope (str, required)
    Telescope model name (e.g. LSTN-01, MSTN-04, SSTS-04, ...)
site (str, required)
    Site name (North or South).
flasher (str, required)
    Flasher device in array, e.g., FLSN-01.
model_version (str, optional)
    Version of the simulation model.
plot (flag, optional)
    Produce a multiple pages pdf file with the image plots.
integration_window (list, optional)
    Integration window for the signal.
boundary_thresh, picture_thresh, min_neighbors (int, optional)
    Parameters for the image cleaning.
return_cleaned (bool, optional)
    Whether to return the cleaned image.

Example
-------

Simulate flashers for MSTN-04:

.. code-block:: console

    simtools-simulate-flasher --telescope MSTN-04 --site North \
    --flasher FLSN-01 --model_version 6.0.0

Expected Output:
    The simulation will run the light emission package for the flasher
    devices and produce the output files.
"""

import logging
from pathlib import Path

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.model.flasher_model import FlasherModel
from simtools.model.model_utils import initialize_simulation_models
from simtools.simtel.simulator_light_emission import SimulatorLightEmission


def _parse(label):
    """Parse command line configuration."""
    config = configurator.Configurator(
        label=label,
        description=("Simulate flasher devices (flat fielding) using the light emission package."),
    )
    config.parser.add_argument(
        "--plot",
        help="Produce a multiple pages pdf file with the image plots.",
        action="store_true",
    )
    config.parser.add_argument(
        "--flasher",
        help="Flasher device in array, i.e. FLSN-01",
        type=str,
        required=True,
    )
    config.parser.add_argument(
        "--return_cleaned",
        help="If set, perform image cleaning and return cleaned image.",
        action="store_true",
        required=False,
    )
    config.parser.add_argument(
        "--picture_thresh",
        help="Threshold above which all pixels are retained.",
        type=int,
        required=False,
    )
    config.parser.add_argument(
        "--boundary_thresh",
        help=(
            "Threshold above which pixels are retained if they have a neighbor "
            "already above the picture threshold."
        ),
        type=int,
        required=False,
    )
    config.parser.add_argument(
        "--min_neighbors",
        help=(
            "Minimum number of picture neighbors a picture pixel must have "
            "(ignored if keep_isolated_pixels)."
        ),
        type=int,
        required=False,
    )
    config.parser.add_argument(
        "--level",
        help="read_cta plotting level (default 5)",
        type=int,
        default=5,
        required=False,
    )
    config.parser.add_argument(
        "--integration_window",
        help="Integration window width,offset (default 7 3)",
        nargs="*",
        default=["7", "3"],
        required=False,
    )
    return config.initialize(
        db_config=True,
        simulation_model=["telescope", "model_version"],
        require_command_line=True,
    )


def flasher_configs(telescope_model_name):
    """
    Get flasher configurations based on telescope type.

    Parameters
    ----------
    telescope_model_name
        Telescope model name (LST-01, LST-02, MST-04, ...)

    Returns
    -------
    tuple
        Application name and mode

    """
    if "SST" in telescope_model_name:
        return ("ff-gct", "flasher")
    return ("ff-1m", "flasher")


def main():
    """Run the application."""
    label = Path(__file__).stem
    logger = logging.getLogger(__name__)

    args_dict, db_config = _parse(label)
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    telescope_model, site_model = initialize_simulation_models(
        label=label,
        db_config=db_config,
        site=args_dict["site"],
        telescope_name=args_dict["telescope"],
        model_version=args_dict["model_version"],
    )

    flasher_model = FlasherModel(
        site=args_dict["site"],
        flasher_device_model_name=args_dict["flasher"],
        mongo_db_config=db_config,
        model_version=args_dict["model_version"],
        label=label,
    )

    le_application = flasher_configs(telescope_model.name)

    picture_thresh = int(args_dict["picture_thresh"]) if args_dict.get("picture_thresh") else 50
    boundary_thresh = int(args_dict["boundary_thresh"]) if args_dict.get("boundary_thresh") else 20
    min_neighbors = int(args_dict["min_neighbors"]) if args_dict.get("min_neighbors") else 2

    sim_runner = SimulatorLightEmission(
        telescope_model=telescope_model,
        flasher_model=flasher_model,
        site_model=site_model,
        light_emission_config=args_dict,
        le_application=le_application,
        simtel_path=args_dict["simtel_path"],
        light_source_type="flasher",
        label=args_dict["label"],
        test=args_dict.get("test", False),
    )

    figures = []
    simulation_args = {
        "boundary_thresh": boundary_thresh,
        "picture_thresh": picture_thresh,
        "min_neighbors": min_neighbors,
        "return_cleaned": args_dict.get("return_cleaned", False),
        "level": args_dict.get("level", 5),
        "integration_window": args_dict.get("integration_window", ["7", "3"]),
    }

    sim_runner.run_simulation(simulation_args, figures)

    if args_dict.get("plot", False) and figures:
        sim_runner.save_figures_to_pdf(figures, args_dict["telescope"])

    logger.info("Flasher simulation completed.")


if __name__ == "__main__":
    main()
