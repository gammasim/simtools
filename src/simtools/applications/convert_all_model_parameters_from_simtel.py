#!/usr/bin/python3
r"""
    Convert all simulation model parameters exported from sim_telarray format.

    Check value, type, and range, convert units using schema files. Write json files
    ready to be submitted to the model database. Prints out parameters which are not found
    in sim_telarray configuration file and parameters which are not found in simtools schema files.

    Note that all parameters are assigned the same parameter version.

    Command line arguments
    ----------------------
    simtel_cfg_file (str)
        File name of sim_telarray configuration file containing all simulation model parameters.

    simtel_telescope_name (str)
        Name of the telescope in the sim_telarray configuration file.

    telescope (str, optional)
        Telescope model name (e.g. LST-1, SST-D, ...)

    skip_parameter (str, optional)
        List of parameters to be skipped (use sim_telarray names).

    Example
    -------

    To export the model parameters from sim_telarray, first copy and unpack the configuration
    tar ball from sim_telarray (usually called 'sim_telarray_config.tar.gz') to the sim_telarray
    working directory. Extract the configuration using the following command:

    .. code-block:: console

        ./sim_telarray/bin/sim_telarray -c sim_telarray/cfg/CTA/CTA-PROD6-LaPalma.cfg \\
            -C limits=no-internal -C initlist=no-internal -C list=no-internal \\
            -C typelist=no-internal -C maximum_telescopes=30 -DNSB_AUTOSCALE \\
            -DNECTARCAM -DHYPER_LAYOUT -DNUM_TELESCOPES=30 /dev/null \\
            2>|/dev/null | grep '(@cfg)' | sed 's/^(@cfg) //' >| all_telescope_config_la_palma.cfg

        ./sim_telarray/bin/sim_telarray -c sim_telarray/cfg/CTA/CTA-PROD6-Paranal.cfg \\
            -C limits=no-internal -C initlist=no-internal -C list=no-internal \\
            -C typelist=no-internal -C maximum_telescopes=87 -DNSB_AUTOSCALE \\
            -DFLASHCAM -DHYPER_LAYOUT -DNUM_TELESCOPES=87 /dev/null \\
            2>|/dev/null | grep '(@cfg)' | sed 's/^(@cfg) //' >| all_telescope_config_paranal.cfg


    Extract then model parameters from the sim_telarray configuration file for LSTN-01
    (telescopes are named CT1, CT2, ..., in the sim_telarray configuration file and must be
    provided in the "simtel_telescope_name" command line argument)
    and write json files in the same format as the model parameter database:

    .. code-block:: console

       simtools-convert-all-model-parameters-from-simtel \\
          --simtel_cfg_file all_telescope_config_la_palma.cfg\\
          --simtel_telescope_name CT1\\
          --telescope LSTN-01\\
          --parameter_version "1.0.0"\\
          --output_path /path/to/output

"""

import logging
from pathlib import Path

import numpy as np

import simtools.data_model.model_data_writer as writer
import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.data_model import schema
from simtools.io import ascii_handler
from simtools.io.io_handler import IOHandler
from simtools.simtel import simtel_config_reader


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
        "--simtel_cfg_file",
        help="File name for sim_telarray configuration",
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
        "--skip_parameter",
        help="List of parameters to be skipped.",
        type=str,
        nargs="*",
        default=[],
    )
    return config.initialize(simulation_model=["telescope", "parameter_version"])


def read_simtel_config_file(args_dict, schema_file, camera_pixels=None):
    """
    Read the sim_telarray configuration file.

    Parameters
    ----------
    args_dict: dict
        Dictionary with command line arguments.
    schema_file: str
        Schema path name.
    camera_pixels: int
        Number of camera pixels.

    Returns
    -------
    SimtelConfigReader, None
        SimtelConfigReader object (None if parameter not found)

    """
    config_reader = simtel_config_reader.SimtelConfigReader(
        schema_file=schema_file,
        simtel_config_file=args_dict["simtel_cfg_file"],
        simtel_telescope_name=args_dict["simtel_telescope_name"],
        camera_pixels=camera_pixels,
    )
    if config_reader.parameter_dict is None or len(config_reader.parameter_dict) == 0:
        return None
    return config_reader


def get_number_of_camera_pixel(args_dict, logger):
    """
    Get the number of camera pixels from the sim_telarray configuration file.

    Required to set the dimension some of the parameter correctly, as simtel
    in some cases does not provide the dimension ('all:' in the parameter files).

    Parameters
    ----------
    args_dict: dict
        Dictionary with command line arguments.
    logger: logging.Logger
        Logger object

    Returns
    -------
    int, None
        Number of camera pixels (None if file is not found)

    """
    try:
        config_reader = read_simtel_config_file(
            args_dict, schema.get_model_parameter_schema_file("camera_pixels")
        )
        _camera_pixel = config_reader.parameter_dict.get(args_dict["simtel_telescope_name"])
    except (FileNotFoundError, AttributeError):
        logger.warning("Failed to read camera pixel parameter.")
        _camera_pixel = None
    logger.info(f"Number of camera pixels: {_camera_pixel}")
    return _camera_pixel


def read_and_export_parameters(args_dict, logger):
    """
    Read and export parameters from sim_telarray configuration file to json files.

    Provide extensive logging information on the parameters found in the simtel
    configuration file.

    Parameters
    ----------
    args_dict: dict
        Dictionary with command line arguments.
    logger: logging.Logger
        Logger object

    Returns
    -------
    list
        List of sim_telarray parameters not found in schema files.
    list
        List of simtools parameter not found in sim_telarray configuration file.

    """
    _parameters, _schema_files = schema.get_model_parameter_schema_files()
    _simtel_parameters = simtel_config_reader.get_list_of_simtel_parameters(
        args_dict["simtel_cfg_file"]
    )
    logger.info(f"Found {len(_simtel_parameters)} parameters in sim_telarray configuration file.")

    io_handler = IOHandler()
    io_handler.set_paths(output_path=args_dict["output_path"])

    _camera_pixel = get_number_of_camera_pixel(args_dict, logger)

    _parameters_not_in_simtel = []

    for _parameter, _schema_file in zip(_parameters, _schema_files):
        logger.info(f"Parameter: {_parameter} Schema file: {_schema_file}")
        if _parameter in args_dict["skip_parameter"]:
            logger.info(f"Skipping {_parameter}")
            continue
        config_reader = read_simtel_config_file(args_dict, _schema_file, _camera_pixel)

        if config_reader is None:
            logger.info("Parameter not found in sim_telarray configuration file.")
            _parameters_not_in_simtel.append(_parameter)
            continue

        logger.info(f"sim_telarray parameter: {config_reader.parameter_dict}")

        _json_dict = writer.ModelDataWriter.dump_model_parameter(
            parameter_name=_parameter,
            value=config_reader.parameter_dict.get(args_dict["simtel_telescope_name"]),
            instrument=args_dict["telescope"],
            parameter_version=args_dict["parameter_version"],
            output_file=io_handler.get_output_file(
                f"{_parameter}-{args_dict['parameter_version']}.json",
                label=f"{args_dict['telescope']}",
                sub_dir=f"{_parameter}",
            ),
        )

        config_reader.compare_simtel_config_with_schema()

        if config_reader.simtel_parameter_name.lower() in _simtel_parameters:
            _simtel_parameters.remove(config_reader.simtel_parameter_name.lower())

        if _json_dict["file"]:
            logger.info(f"File name for {_parameter} is {_json_dict['value']}")

    return _parameters_not_in_simtel, _simtel_parameters


def print_parameters_not_found(_parameters_not_in_simtel, _simtel_parameters, args_dict, logger):
    """
    Print simtel/simtools parameter not found in schema and configuration files.

    For sim_telarray parameters not found, check if the setting for the chose
    telescope is different from the default values.

    Parameters
    ----------
    _parameters_not_in_simtel: list
        List of sim_telarray parameters not found in schema files.
    _simtel_parameters: list
        List of sim_telarray parameters not found in simtools schema files.
    args_dict: dict
        Dictionary with command line arguments.
    logger: logging.Logger
        Logger object

    """
    logger.info(
        f"Parameters not found in simtools schema files ({len(_parameters_not_in_simtel)}):"
    )
    for para in sorted(_parameters_not_in_simtel):
        logger.info(f"  {para}")

    logger.info(f"sim_telarray parameters not found in schema files ({len(_simtel_parameters)}):")
    for para in sorted(_simtel_parameters):
        logger.info(f"sim_telarray parameter: {para}")
        config_reader = simtel_config_reader.SimtelConfigReader(
            schema_file=None,
            simtel_config_file=args_dict["simtel_cfg_file"],
            simtel_telescope_name=args_dict["simtel_telescope_name"],
            parameter_name=para,
        )
        _default = config_reader.parameter_dict.get("default")
        _tel_value = config_reader.parameter_dict.get(args_dict["simtel_telescope_name"])
        # simple comparison of default value and telescope values, does not work for lists
        try:
            if _default == _tel_value or np.isclose(_default, _tel_value):
                logger.info(f"    Default and telescope values for {para} are equal: {_default}")
                continue
        except (ValueError, TypeError):
            pass
        if isinstance(_default, np.ndarray):
            logger.warning(f"    Default value ({para}): {_default} (length: {_default.size})")
        else:
            logger.warning(f"    Default value ({para}): {_default}")
        if isinstance(_tel_value, np.ndarray):
            logger.warning(
                f"    Telescope value ({para}): {_tel_value} (length: {_tel_value.size})"
            )
        else:
            logger.warning(f"    Telescope value ({para}): {_tel_value}")


def print_list_of_files(args_dict, logger):
    """
    Print model parameters which describe a file name.

    This is useful to find files which are part of the model.

    Parameters
    ----------
    args_dict: dict
        Dictionary with command line arguments.
    logger: logging.Logger
        Logger object

    """
    model_files = sorted(Path(args_dict["output_path"]).rglob("*.json"))
    for file in model_files:
        model_dict = ascii_handler.collect_data_from_file(file_name=file)
        if model_dict.get("file"):
            logger.info(f"{file.name}: {model_dict['value']}")


def main():  # noqa: D103
    args_dict, _ = _parse(
        label=Path(__file__).stem,
        description="Convert simulation model parameters from sim_telarray to simtools format.",
    )
    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    _parameters_not_in_simtel, _simtel_parameters = read_and_export_parameters(args_dict, logger)
    print_parameters_not_found(_parameters_not_in_simtel, _simtel_parameters, args_dict, logger)
    print_list_of_files(args_dict, logger)


if __name__ == "__main__":
    main()
