#!/usr/bin/python3
"""
    Summary
    -------
    Convert simulation model parameter from sim_telarray format using the corresponding
    schema file. Check value, type, and range and output (if successful) a json file
    ready to be submitted to the model database.

    Command line arguments
    ----------------------
    parameter (str, required)
        Parameter name (as used in simtools)

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

       simtools-convert-model-parameter-from-simtel -simtel_telescope_name CT1\
          --telescope LSTN-01\
          --parameter num_gains\
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
        "--schema", help="json schema file for model parameter validation", required=False
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
    config.parser.add_argument(
        "--output_file",
        help="Output file (json format)",
        type=str,
        required=False,
    )
    return config.initialize(telescope_model=True)


def main():

    args_dict, _ = _parse(
        label=Path(__file__).stem,
        description="Convert simulation model parameter from sim_telarray to simtools format.",
    )

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    simtel_config_reader = SimtelConfigReader(
        schema_file=args_dict["schema"],
        simtel_config_file=args_dict["simtel_cfg_file"],
        simtel_telescope_name=args_dict["simtel_telescope_name"],
    )
    logger.info(f"PARAMETER: {simtel_config_reader.parameter_dict}")
    if len(simtel_config_reader.parameter_dict) == 0:
        logger.error("Parameter not found in sim_telarray configuration file.")
        return
    _json_dict = simtel_config_reader.get_validated_parameter_dict(
        telescope_name=args_dict["telescope"], model_version=args_dict["model_version"]
    )
    logger.info("fDB JSON {_json_dict}")

    simtel_config_reader.compare_simtel_config_with_schema()

    if args_dict["output_file"]:
        simtel_config_reader.export_parameter_dict_to_json(args_dict["output_file"], _json_dict)


if __name__ == "__main__":
    main()
