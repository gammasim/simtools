#!/usr/bin/python3
"""
    Summary
    -------
    Convert all simulation model parameters from sim_telarray format using the corresponding
    schema files. Check value, type, and range and output (if successful) json files
    ready to be submitted to the model database.

    Command line arguments
    ----------------------
    simtel_cfg_file (str)
        File name of sim_telarray configuration file with all simulation model parameters.

    simtel_telescope_name (str)
        Name of the telescope in the sim_telarray configuration file.

    telescope (str, optional)
        Telescope model name (e.g. LST-1, SST-D, ...)

    Example
    -------

    Extract the num_gains parameter from the sim_telarray configuration file for LSTN-01.

    .. code-block:: console

       simtools-convert-all-model-parameters-from-simtel \
          --simtel_telescope_name CT1\
          --telescope LSTN-01\
          --simtel_cfg_file all_telescope_config_la_palma.cfg

"""

import logging
from pathlib import Path

import simtools.utils.general as gen
from simtools.configuration import configurator
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
        help="Directory with json schema files for model parameter validation",
        required=False,
    )
    config.parser.add_argument(
        "--simtel_cfg_file",
        help="File name for simtel_array configuration",
        type=str,
        required=False,
    )
    config.parser.add_argument(
        "--simtel_telescope_name",
        help="Name of the telescope in the sim_telarray configuration file",
        type=str,
        required=False,
    )
    return config.initialize(telescope_model=True)


def get_list_of_parameters_and_schema_files(schema_directory):
    """
    Return list of parameters and schema file found in schema file directory.

    Parameters
    ----------
    schema_directory: str
        Directory with json schema files for model parameter validation

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
        parameters.append(schema_dict["name"])
    return parameters, schema_files


def main():

    args_dict, _ = _parse(
        label=Path(__file__).stem,
        description="Convert simulation model parameter from sim_telarray to simtools format.",
    )

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    _parameters, _schema_files = get_list_of_parameters_and_schema_files(
        args_dict["schema_directory"]
    )

    # TODO - temporary list of parameters to ignore
    _ignore_parameters_list = [
        "altitude",
        "array_coordinates",
        "array_coordinates_UTM",
        "mirror_panel_2f_measurements",
        "disc_ac_coupled",  # unclear why validation fails
        "dsum_pedsub",  # unclear why validation fails
        "dsum_shaping_renormalize",  # unclear why validation fails
        "fadc_ac_coupled",  # unclear why validation fails
        "flatfielding",  # unclear why validation fails
        "only_triggered_telescopes",  # unclear why validation fails
        "parabolic_dish",  # unclear why validation fails
    ]

    for _parameter, _schema_file in zip(_parameters, _schema_files):
        if _parameter in _ignore_parameters_list:
            continue
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
            logger.error("Parameter not found in sim_telarray configuration file.")
            continue
        _json_dict = simtel_config_reader.get_validated_parameter_dict(
            telescope_name=args_dict["telescope"], model_version=args_dict["model_version"]
        )
        logger.info(f"Validated parameter {_json_dict}")

        simtel_config_reader.compare_simtel_config_with_schema()

        simtel_config_reader.export_parameter_dict_to_json(
            Path(args_dict["output_path"]) / f"{_parameter}.json", _json_dict
        )


if __name__ == "__main__":
    main()
