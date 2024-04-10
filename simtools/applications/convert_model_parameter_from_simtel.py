#!/usr/bin/python3
"""
    Summary
    -------
    Convert simulation model parameter from sim_telarray format using the corresponding
    schema file. Check value, type, and range and write a json file
    ready to be submitted to the model database.

    Command line arguments
    ----------------------
    parameter (str, required)
        Parameter name (as used in simtools)

    simtel_cfg_file (str)
        File name of sim_telarray configuration file containing all simulation model parameters.

    simtel_telescope_name (str)
        Name of the telescope in the sim_telarray configuration file.

    telescope (str, optional)
        Telescope model name (e.g. LST-1, SST-D, ...)

    Example
    -------

    Extract the num_gains parameter from a sim_telarray configuration file for LSTN-01
    and write a json file in the same format as the model parameter database:

    .. code-block:: console

       simtools-convert-model-parameter-from-simtel \
          --simtel_telescope_name CT1\
          --telescope LSTN-01\
          --schema tests/resources/num_gains.schema.yml\
          --simtel_cfg_file tests/resources/simtel_config_test_la_palma.cfg\
          --output_file num_gains.json

"""

import logging
from pathlib import Path

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
        "--schema", help="Schema file for model parameter validation", required=True
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
    config.parser.add_argument(
        "--output_file",
        help="Output file (json format)",
        type=str,
        required=False,
    )
    return config.initialize(simulation_model="telescope")


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
    logger.info(f"Simtel parameter: {simtel_config_reader.parameter_dict}")
    if simtel_config_reader.parameter_dict is None or len(simtel_config_reader.parameter_dict) == 0:
        logger.error("Parameter not found in sim_telarray configuration file.")
        return
    _json_dict = simtel_config_reader.get_validated_parameter_dict(
        telescope_name=args_dict["telescope"], model_version=args_dict["model_version"]
    )
    logger.info(f"Validated parameter: {_json_dict}")

    simtel_config_reader.compare_simtel_config_with_schema()

    if args_dict["output_file"]:
        io_handler = IOHandler()
        io_handler.set_paths(output_path=args_dict["output_path"], use_plain_output_path=True)
        simtel_config_reader.export_parameter_dict_to_json(
            io_handler.get_output_file(args_dict["output_file"]), _json_dict
        )


if __name__ == "__main__":
    main()
