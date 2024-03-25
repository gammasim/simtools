#!/usr/bin/python3
"""
    Summary
    -------
    Convert all simulation model parameters exported from sim_telarray format using
    schema files. Check value, type, and range, convert units, and write json files
    ready to be submitted to the model database. Prints out parameters which are not found
    in simtel configuration file and parameters which are not found in simtools schema files.

    Command line arguments
    ----------------------
    simtel_cfg_file (str)
        File name of sim_telarray configuration file containing all simulation model parameters.

    simtel_telescope_name (str)
        Name of the telescope in the sim_telarray configuration file.

    telescope (str, optional)
        Telescope model name (e.g. LST-1, SST-D, ...)

    Example
    -------

    Extract model parameters with schema files from simtel configuration file
    (requires access to the model parameter repository)

    .. code-block:: console

       simtools-convert-all-model-parameters-from-simtel \
          --schema_directory ../model_parameters/schema\
          --simtel_cfg_file all_telescope_config_la_palma.cfg\
          --simtel_telescope_name CT1\
          --telescope LSTN-01\
          --model_version "2024-03-06"

"""

import logging
import re
from pathlib import Path

import numpy as np

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.io_operations.io_handler import IOHandler
from simtools.simtel.simtel_config_reader import SimtelConfigReader


def _parse(label=None, description=None):
    """
    Parse command line configuration

    Parameters
    ----------
    label: str
        Label describing application.
    description: str
        Description of application.

    Returns
    -------
    CommandLineParser
        Command line parser object

    """

    config = configurator.Configurator(label=label, description=description)

    config.parser.add_argument(
        "--schema_directory",
        help="Directory with schema files for model parameter validation",
        required=True,
    )
    config.parser.add_argument(
        "--simtel_cfg_file",
        help="File name for simtel_array configuration",
        type=str,
        required=True,
    )
    config.parser.add_argument(
        "--simtel_telescope_name",
        help="Name of the telescope in the sim_telarray configuration file",
        type=str,
        required=True,
    )
    return config.initialize(telescope_model=True)


def get_list_of_parameters_and_schema_files(schema_directory):
    """
    Return list of parameters and schema files located in schema file directory.

    Parameters
    ----------
    schema_directory: str
        Directory with schema files for model parameter validation

    Returns
    -------
    list
        List of parameters found in schema file directory.
    list
        List of schema files found in schema file directory.

    """

    schema_files = sorted(list(Path(schema_directory).rglob("*.schema.yml")))
    parameters = []
    for schema_file in schema_files:
        schema_dict = gen.collect_data_from_file_or_dict(file_name=schema_file, in_dict=None)
        parameters.append(schema_dict.get("name"))
    return parameters, schema_files


def get_list_of_simtel_parameters(simtel_config_file, logger):
    """
    Return list of simtel parameters found in simtel configuration file.

    Parameters
    ----------
    simtel_config_file: str
        File name for sim_telarray configuration
    logger: logging.Logger
        Logger object

    Returns
    -------
    list
        List of parameters found in simtel configuration file.

    """

    simtel_parameter_set = set()
    with open(simtel_config_file, "r", encoding="utf-8") as file:
        for line in file:
            parts_of_lines = re.split(r",\s*|\s+", line.strip())
            simtel_parameter_set.add(parts_of_lines[1].lower())
    logger.info(f"Found {len(simtel_parameter_set)} parameters in simtel configuration file.")
    return list(simtel_parameter_set)


def read_and_export_parameters(args_dict, logger):
    """
    Read and export parameters from simtel configuration file to json files.

    Parameters
    ----------
    args_dict: dict
        Dictionary with command line arguments.
    logger: logging.Logger
        Logger object

    Returns
    -------
    list
        List of simtel parameters not found in schema files.
    list
        List of simtools parameter not found in simtel configuration file.

    """

    _parameters, _schema_files = get_list_of_parameters_and_schema_files(
        args_dict["schema_directory"]
    )
    _simtel_parameters = get_list_of_simtel_parameters(args_dict["simtel_cfg_file"], logger)

    io_handler = IOHandler()
    io_handler.set_paths(output_path=args_dict["output_path"], use_plain_output_path=True)

    _parameters_not_in_simtel = []

    for _parameter, _schema_file in zip(_parameters, _schema_files):
        logger.info(f"Parameter: {_parameter} Schema file: {_schema_file}")

        simtel_config_reader = SimtelConfigReader(
            schema_file=_schema_file,
            simtel_config_file=args_dict["simtel_cfg_file"],
            simtel_telescope_name=args_dict["simtel_telescope_name"],
        )
        logger.info(f"Simtel parameter: {simtel_config_reader.parameter_dict}")
        if (
            simtel_config_reader.parameter_dict is None
            or len(simtel_config_reader.parameter_dict) == 0
        ):
            _parameters_not_in_simtel.append(_parameter)
            logger.info("Parameter not found in sim_telarray configuration file.")
            continue
        _json_dict = simtel_config_reader.get_validated_parameter_dict(
            telescope_name=args_dict["telescope"], model_version=args_dict["model_version"]
        )
        logger.info(f"Validated parameter {_json_dict}")

        simtel_config_reader.compare_simtel_config_with_schema()

        simtel_config_reader.export_parameter_dict_to_json(
            io_handler.get_output_file(f"{_parameter}.json"), _json_dict
        )

        if simtel_config_reader.simtel_parameter_name.lower() in _simtel_parameters:
            _simtel_parameters.remove(simtel_config_reader.simtel_parameter_name.lower())

    return _parameters_not_in_simtel, _simtel_parameters


def print_parameters_not_found(_parameters_not_in_simtel, _simtel_parameters, args_dict, logger):
    """
    Print simtel parameters not found in schema files and simtools parameters not found in simtel
    configuration file. For parameters not found, check if the setting for the chose telescope
    is different from the default values.

    Parameters
    ----------
    _parameters_not_in_simtel: list
        List of simtel parameters not found in schema files.
    _simtel_parameters: list
        List of simtel parameters not found in simtools schema files.
    args_dict: dict
        Dictionary with command line arguments.
    logger: logging.Logger
        Logger object

    """

    if len(_parameters_not_in_simtel) > 0:
        logger.info(
            f"Parameters not found in simtools schema files ({len(_parameters_not_in_simtel)}):"
        )
        for para in sorted(_parameters_not_in_simtel):
            logger.info(f"  {para}")

    if len(_simtel_parameters) > 0:
        logger.info(f"Simtel parameters not found in schema files ({len(_simtel_parameters)}):")
        for para in sorted(_simtel_parameters):
            logger.info(f"Simtel parameter: {para}")
            simtel_config_reader = SimtelConfigReader(
                schema_file=None,
                simtel_config_file=args_dict["simtel_cfg_file"],
                simtel_telescope_name=args_dict["simtel_telescope_name"],
                parameter_name=para,
            )
            _default = simtel_config_reader.parameter_dict.get("default")
            _tel_value = simtel_config_reader.parameter_dict.get(args_dict["simtel_telescope_name"])
            # simple comparison of default value and telescope values, does not work for lists
            try:
                if _default == _tel_value or np.isclose(_default, _tel_value):
                    logger.info(f"    Default and telescope values are equal: {_default}")
                    continue
            except (ValueError, TypeError):
                pass
            if isinstance(_default, np.ndarray):
                logger.warning(f"    Default value: {_default} (length: {_default.size})")
            else:
                logger.warning(f"    Default value: {_default}")
            if isinstance(_tel_value, np.ndarray):
                logger.warning(f"    Telescope value: {_tel_value} (length: {_tel_value.size})")
            else:
                logger.warning(f"    Telescope value: {_tel_value}")


def main():

    args_dict, _ = _parse(
        label=Path(__file__).stem,
        description="Convert simulation model parameters from sim_telarray to simtools format.",
    )

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    _parameters_not_in_simtel, _simtel_parameters = read_and_export_parameters(args_dict, logger)
    print_parameters_not_found(_parameters_not_in_simtel, _simtel_parameters, args_dict, logger)


if __name__ == "__main__":
    main()
