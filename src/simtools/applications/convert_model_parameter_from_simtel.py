#!/usr/bin/python3
r"""
    Convert simulation model parameter from sim_telarray format using the corresponding schema file.

    Check value, type, and range and write a json file ready to be submitted to the model database.

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

       simtools-convert-model-parameter-from-simtel \\
          --simtel_telescope_name CT1 \\
          --telescope LSTN-01 \\
          --schema tests/resources/num_gains.schema.yml \\
          --simtel_cfg_file tests/resources/simtel_config_test_la_palma.cfg \\
          --output_file num_gains.json

"""

import logging
from pathlib import Path

import simtools.data_model.model_data_writer as writer
import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.simtel.simtel_config_reader import SimtelConfigReader


def _parse(label=None, description=None):
    """
    Parse command line configuration.

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
    return config.initialize(simulation_model=["telescope", "parameter_version"], output=True)


def main():  # noqa: D103

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

    simtel_config_reader.compare_simtel_config_with_schema()

    _json_dict = writer.ModelDataWriter.dump_model_parameter(
        parameter_name=simtel_config_reader.parameter_name,
        value=simtel_config_reader.parameter_dict.get(args_dict["simtel_telescope_name"]),
        instrument=args_dict["telescope"],
        parameter_version=args_dict["parameter_version"],
        output_file=args_dict["output_file"],
        output_path=args_dict.get("output_path"),
        use_plain_output_path=args_dict.get("use_plain_output_path"),
    )
    logger.info(f"Validated parameter: {_json_dict}")


if __name__ == "__main__":
    main()
