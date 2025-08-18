#!/usr/bin/python3
r"""
    Submit a model parameter value and corresponding metadata through the command line.

    Input and metadata is validated, and if necessary enriched and converted following
    the model parameter schemas. Model parameter data is written in the simtools-style
    json format, metadata as a yaml file.

    Command line arguments
    ----------------------
    parameter (str)
        model parameter name
    value (str, value)
        input value (number, string, string-type lists)
    instrument (str)
        instrument name.
    site (str)
        site location.
    parameter_version (str)
        Parameter version.
    input_meta (str, optional)
        input meta data file (yml format)

    Example
    -------

    Submit the number of gains for the LSTN-design readout chain:

    .. code-block:: console

        simtools-submit-model-parameter-from-external \
            --parameter num_gains \\
            --value 2 \\
            --instrument LSTN-design \\
            --site North \\
            --parameter_version 0.1.0 \\
            --input_meta num_gains.metadata.yml

"""

import logging
from pathlib import Path

import simtools.data_model.model_data_writer as writer
import simtools.utils.general as gen
from simtools.configuration import configurator


def _parse(label, description):
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
        "--parameter", type=str, required=True, help="Parameter for simulation model"
    )
    config.parser.add_argument("--instrument", type=str, required=True, help="Instrument name")
    config.parser.add_argument("--site", type=str, required=True, help="Site location")
    config.parser.add_argument(
        "--parameter_version", type=str, required=True, help="Parameter version"
    )
    config.parser.add_argument(
        "--value",
        type=str,
        required=True,
        help=(
            "Model parameter value. "
            "Can be a single number, a number with a unit, or a list of values with units. "
            'Examples: "--value=5", "--value=\'5 km\'", "--value=\'5 cm, 0.5 deg\'"'
        ),
    )
    config.parser.add_argument(
        "--input_meta",
        help="meta data file(s) associated to input data (wildcards or list of files allowed)",
        type=str,
        nargs="+",
        required=False,
    )
    config.parser.add_argument(
        "--check_parameter_version",
        help="Check if the parameter version exists in the database",
        action="store_true",
    )
    return config.initialize(output=True, db_config=True)


def main():  # noqa: D103
    args_dict, db_config = _parse(
        label=Path(__file__).stem,
        description="Submit and validate a model parameters).",
    )

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    output_path = (
        Path(args_dict["output_path"]) / args_dict["parameter"]
        if args_dict.get("output_path")
        else None
    )

    writer.ModelDataWriter.dump_model_parameter(
        parameter_name=args_dict["parameter"],
        value=args_dict["value"],
        instrument=args_dict["instrument"],
        parameter_version=args_dict["parameter_version"],
        output_file=Path(args_dict["parameter"] + "-" + args_dict["parameter_version"] + ".json"),
        output_path=output_path,
        use_plain_output_path=args_dict.get("use_plain_output_path"),
        metadata_input_dict=args_dict,
        db_config=db_config if args_dict.get("check_parameter_version") else None,
    )


if __name__ == "__main__":
    main()
