#!/usr/bin/python3
"""
    Summary
    -------
    Convert simulation model parameter from sim_telarray format using the corresponding
    schema files. Check value, type, and range and output (if successful) a json file
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

       simtools-convert-model-parameters-from-simtel -simtel_telescope_name CT1\
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
    config.parser.add_argument("--parameter", help="Parameter name", type=str, required=True)
    return config.initialize(db_config=True, telescope_model=True)


def main():

    args_dict, _ = _parse(
        label=Path(__file__).stem,
        description="Convert simulation model parameter from sim_telarray to simtools format",
    )

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    simtel_config_reader = SimtelConfigReader(
        simtel_config_file=args_dict["simtel_cfg_file"],
        simtel_telescope_name=args_dict["simtel_telescope_name"],
        parameter_name=args_dict["parameter"],
        schema_url=f"{args_dict['db_simulation_model_url']}/schema",
    )
    _parameter_dict, _json_dict = simtel_config_reader.get_validated_parameter_dict(
        args_dict["telescope"]
    )
    print("PARAMETER", _parameter_dict)
    print("DB JSON", _json_dict)


if __name__ == "__main__":
    main()
