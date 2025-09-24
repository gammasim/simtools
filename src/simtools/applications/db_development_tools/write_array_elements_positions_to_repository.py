#!/usr/bin/python3
"""
    Read array element positions from file and add them to model repository.

    This is an application for experts and should not be used by the general user.
    Reading of input is fine-tuned to the array element files as provided by CTAO.

    Command line arguments

    input : str
        File containing a table of array element positions.
    repository_path : str
        Path of local copy of model parameter repository.
    parameter_version : str
        Parameter version.
    site : str
        Observatory site.
    coordinate_system : str
        Coordinate system of array element positions (ground or utm).

    Examples
    --------
    Add array element positions to repository:

    .. code-block:: console

        simtools-write-array-element-positions-to-repository \
            --input /path/to/positions.txt \
            --repository_path /path/to/repository \
            --parameter_version 0.1.0 \
            --coordinate_system ground \
            --site North

"""

import logging
from pathlib import Path

import astropy.table

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.data_model.model_data_writer import ModelDataWriter
from simtools.io import ascii_handler
from simtools.model.array_model import ArrayModel


def _parse(label=None, description=None):
    """
    Parse command line configuration.

    Parameters
    ----------
    label : str
        Label describing application.
    description : str
        Description of application.

    Returns
    -------
    CommandLineParser
        Command line parser object.
    """
    config = configurator.Configurator(label=label, description=description)
    config.parser.add_argument(
        "--input",
        help="File containing a table of array element positions.",
        required=False,
    )
    config.parser.add_argument(
        "--repository_path",
        help="Output path to model parameter repository.",
        type=Path,
        required=False,
    )
    config.parser.add_argument(
        "--coordinate_system",
        help="Coordinate system of array element positions (utm or ground).",
        default="ground",
        required=False,
        type=str,
        choices=["ground", "utm"],
    )

    return config.initialize(db_config=True, simulation_model=["site", "parameter_version"])


def write_utm_array_elements_to_repository(args_dict, logger):
    """
    Write UTM position of array elements to model repository.

    Read array element positions from file. The ecsv row definition might
    include telescope_name or asset_code and sequence_number.

    Parameters
    ----------
    args_dict : dict
        Command line arguments.
    logger : Logger
        Logger object.

    """
    array_elements = astropy.table.Table.read(args_dict["input"])
    for row in array_elements:
        instrument = (
            row["telescope_name"]
            if "telescope_name" in array_elements.colnames
            else f"{row['asset_code']}-{row['sequence_number']}"
        )
        output_path = Path(args_dict["repository_path"]) / f"{instrument}"
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Writing array element positions (utm) to {output_path}")

        ModelDataWriter.dump_model_parameter(
            parameter_name="array_element_position_utm",
            instrument=instrument,
            value=f"{row['utm_east']} {row['utm_north']} {row['altitude']}",
            parameter_version=args_dict["parameter_version"],
            output_path=output_path,
            output_file="array_element_position_utm.json",
        )


def write_ground_array_elements_to_repository(args_dict, db_config, logger):
    """
    Write ground position of array elements to model repository.

    Parameters
    ----------
    args_dict : dict
        Command line arguments.
    db_config : dict
        Database configuration.
    logger : Logger
        Logger object.

    """
    array_model = ArrayModel(
        mongo_db_config=db_config,
        model_version=None,
        site=args_dict["site"],
        array_elements=args_dict["input"],
    )
    for element_name, data in array_model.array_elements.items():
        output_path = Path(args_dict["repository_path"]) / f"{element_name}"
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Writing array element positions (ground) to {output_path}")
        ascii_handler.write_data_to_file(
            data=data,
            output_file=output_path / "array_element_position_ground.json",
            sort_keys=False,
            numpy_types=True,
        )


def main():
    """Application main."""
    label = Path(__file__).stem
    args_dict, db_config = _parse(
        label, description="Add array element positions to model parameter repository"
    )
    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    if args_dict["coordinate_system"] == "utm":
        write_utm_array_elements_to_repository(args_dict, logger)
    elif args_dict["coordinate_system"] == "ground":
        write_ground_array_elements_to_repository(args_dict, db_config, logger)
    else:
        logger.error("Invalid coordinate system. Allowed are 'utm' and 'ground'.")
        raise ValueError


if __name__ == "__main__":
    main()
