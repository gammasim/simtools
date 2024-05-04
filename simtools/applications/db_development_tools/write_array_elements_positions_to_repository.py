#!/usr/bin/python3
"""
    Read array element positions from file and add them to model repository.

    This is an application for experts and should not be used by the general user.

    Command line arguments

    input (str, required)
        List of array element positions.

    output_path (str, required)
        Path of local copy of model parameter repository.

    model_version (str, required)
        Model version.

    Examples
    --------

    Add array element positions to repository:

    .. code-block:: console

        simtools-write-array-element-positions-to-repository \
            --input /path/to/positions.txt \
            --repository_path /path/to/repository \
            --model_version 1.0.0 \
            --site North

"""

import json
import logging
from pathlib import Path

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.model.array_model import ArrayModel
from simtools.simtel.simtel_config_reader import JsonNumpyEncoder


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
        help="list of array element positions",
        required=False,
    )
    config.parser.add_argument(
        "--repository_path",
        help="Output path to model parameter repository.",
        type=Path,
        required=False,
    )

    return config.initialize(db_config=True, simulation_model="site")


def main():
    """Application main."""
    label = Path(__file__).stem
    args_dict, db_config = _parse(
        label, description="Add array element positions to model parameter repository"
    )
    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    array_model = ArrayModel(
        mongo_db_config=db_config,
        model_version=args_dict["model_version"],
        site=args_dict["site"],
        array_elements_file=args_dict["input"],
    )
    for element_name, data in array_model.array_elements.items():
        output_path = Path(args_dict["repository_path"]) / f"{element_name}"
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Writing array element positions to {output_path}")
        with open(
            output_path / "array_element_position_ground.json", "w", encoding="utf-8"
        ) as file:
            json.dump(
                data,
                file,
                indent=4,
                sort_keys=False,
                cls=JsonNumpyEncoder,
            )
            file.write("\n")


if __name__ == "__main__":
    main()
