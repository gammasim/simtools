#!/usr/bin/python3
"""Utilities to convert simulation model parameters from sim_telarray files."""

import logging
from pathlib import Path

import numpy as np

import simtools.data_model.model_data_writer as writer
from simtools.data_model import schema
from simtools.io import ascii_handler
from simtools.simtel import simtel_config_reader

_logger = logging.getLogger(__name__)


def _read_simtel_config_file(args_dict, schema_file, camera_pixels=None):
    """Read the sim_telarray configuration file."""
    config_reader = simtel_config_reader.SimtelConfigReader(
        schema_file=schema_file,
        simtel_config_file=args_dict["simtel_cfg_file"],
        simtel_telescope_name=args_dict["simtel_telescope_name"],
        camera_pixels=camera_pixels,
    )
    if config_reader.parameter_dict is None or len(config_reader.parameter_dict) == 0:
        return None
    return config_reader


def _get_number_of_camera_pixel(args_dict):
    """
    Get the number of camera pixels from the sim_telarray configuration file.

    Required to set the dimension some of the parameter correctly, as simtel
    in some cases does not provide the dimension ('all:' in the parameter files).
    """
    try:
        config_reader = _read_simtel_config_file(
            args_dict, schema.get_model_parameter_schema_file("camera_pixels")
        )
        _camera_pixel = config_reader.parameter_dict.get(args_dict["simtel_telescope_name"])
    except (FileNotFoundError, AttributeError):
        _logger.warning("Unable to retrieve camera pixel parameter.")
        _camera_pixel = None
    _logger.info(f"Number of camera pixels: {_camera_pixel}")
    return _camera_pixel


def read_and_export_parameters(args_dict, io_handler):
    """
    Read and export parameters from sim_telarray configuration file to json files.

    Parameters
    ----------
    args_dict: dict
        Dictionary with command line arguments.
    io_handler: IOHandler
        IOHandler object

    Returns
    -------
    list
        List of simtools parameters not found in sim_telarray configuration file.
    list
        List of sim_telarray parameters not found in schema files.

    """
    _parameters, _schema_files = schema.get_model_parameter_schema_files()
    simtel_parameters = simtel_config_reader.get_list_of_simtel_parameters(
        args_dict["simtel_cfg_file"]
    )
    _logger.info(f"Found {len(simtel_parameters)} parameters in sim_telarray configuration file.")

    _camera_pixel = _get_number_of_camera_pixel(args_dict)

    parameters_not_in_simtel = []

    for _parameter, _schema_file in zip(_parameters, _schema_files):
        _logger.info(f"Parameter: {_parameter} Schema file: {_schema_file}")
        if _parameter in args_dict["skip_parameter"]:
            _logger.info(f"Skipping {_parameter}")
            continue
        config_reader = _read_simtel_config_file(args_dict, _schema_file, _camera_pixel)

        if config_reader is None:
            _logger.info("Parameter not found in sim_telarray configuration file.")
            parameters_not_in_simtel.append(_parameter)
            continue

        _logger.info(f"sim_telarray parameter: {config_reader.parameter_dict}")

        _json_dict = writer.ModelDataWriter.write_model_parameter(
            parameter_name=_parameter,
            value=config_reader.parameter_dict.get(args_dict["simtel_telescope_name"]),
            instrument=args_dict["telescope"],
            parameter_version=args_dict["parameter_version"],
            output_file=io_handler.get_output_file(
                f"{_parameter}-{args_dict['parameter_version']}.json",
                sub_dir=[f"{args_dict['telescope']}", f"{_parameter}"],
            ),
        )

        config_reader.compare_simtel_config_with_schema()

        if config_reader.simtel_parameter_name.lower() in simtel_parameters:
            simtel_parameters.remove(config_reader.simtel_parameter_name.lower())

        if _json_dict["file"]:
            _logger.info(f"File name for {_parameter} is {_json_dict['value']}")

    return parameters_not_in_simtel, simtel_parameters


def print_parameters_not_found(parameters_not_in_simtel, simtel_parameters, args_dict):
    """
    Print simtel/simtools parameter not found in schema and configuration files.

    For sim_telarray parameters not found, check if the setting for the chose
    telescope is different from the default values.

    Parameters
    ----------
    parameters_not_in_simtel: list
        List of sim_telarray parameters not found in schema files.
    simtel_parameters: list
        List of sim_telarray parameters not found in simtools schema files.
    args_dict: dict
        Dictionary with command line arguments.

    """
    _logger.info(
        f"Parameters not found in simtools schema files ({len(parameters_not_in_simtel)}):"
    )
    for para in sorted(parameters_not_in_simtel):
        _logger.info(f"  {para}")

    _logger.info(f"sim_telarray parameters not found in schema files ({len(simtel_parameters)}):")
    for para in sorted(simtel_parameters):
        _logger.info(f"sim_telarray parameter: {para}")
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
                _logger.info(f"    Default and telescope values for {para} are equal: {_default}")
                continue
        except (ValueError, TypeError) as exc:
            _logger.debug(
                "    Could not directly compare default and telescope values for %s (%s); "
                "printing both values for inspection.",
                para,
                exc,
            )
        if isinstance(_default, np.ndarray):
            _logger.warning(f"    Default value ({para}): {_default} (length: {_default.size})")
        else:
            _logger.warning(f"    Default value ({para}): {_default}")
        if isinstance(_tel_value, np.ndarray):
            _logger.warning(
                f"    Telescope value ({para}): {_tel_value} (length: {_tel_value.size})"
            )
        else:
            _logger.warning(f"    Telescope value ({para}): {_tel_value}")


def print_list_of_files(args_dict):
    """
    Print model parameters which describe a file name.

    This is useful to find files which are part of the model.

    Parameters
    ----------
    args_dict: dict
        Dictionary with command line arguments.
    """
    model_files = sorted(Path(args_dict["output_path"]).rglob("*.json"))
    for file in model_files:
        model_dict = ascii_handler.collect_data_from_file(file_name=file)
        if model_dict.get("file"):
            _logger.info(f"{file.name}: {model_dict['value']}")


def run_conversion_workflow(app_context):
    """
    Run the full model-parameter conversion workflow.

    Parameters
    ----------
    app_context: object
        Application context with ``args`` and ``io_handler`` attributes.
    """
    args_dict = app_context.args
    io_handler = app_context.io_handler

    parameters_not_in_simtel, simtel_parameters = read_and_export_parameters(args_dict, io_handler)
    print_parameters_not_found(parameters_not_in_simtel, simtel_parameters, args_dict)
    print_list_of_files(args_dict)
