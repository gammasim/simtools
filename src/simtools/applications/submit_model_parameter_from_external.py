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

from pathlib import Path

import simtools.data_model.model_data_writer as writer
from simtools.application_control import build_application


def _add_arguments(parser):
    """Register application-specific command line arguments."""
    parser.add_argument(
        "--parameter", type=str, required=True, help="Parameter for simulation model"
    )
    parser.add_argument("--instrument", type=str, required=True, help="Instrument name")
    parser.add_argument("--site", type=str, required=True, help="Site location")
    parser.add_argument("--parameter_version", type=str, required=True, help="Parameter version")
    parser.add_argument(
        "--value",
        type=str,
        required=True,
        help=(
            "Model parameter value. "
            "Can be a single number, a number with a unit, or a list of values with units. "
            'Examples: "--value=5", "--value=\'5 km\'", "--value=\'5 cm, 0.5 deg\'"'
        ),
    )
    parser.add_argument(
        "--input_meta",
        help="meta data file(s) associated to input data (wildcards or list of files allowed)",
        type=str,
        nargs="+",
        required=False,
    )
    parser.add_argument(
        "--check_parameter_version",
        help="Check if the parameter version exists in the database",
        action="store_true",
    )


def main():
    """See CLI description."""
    app_context = build_application(
        __file__,
        description=__doc__,
        add_arguments_function=_add_arguments,
        initialization_kwargs={"output": True, "db_config": True},
    )

    if app_context.args.get("output_path"):
        output_path = app_context.io_handler.get_output_directory(
            sub_dir=app_context.args.get("parameter")
        )
    else:
        output_path = None

    writer.ModelDataWriter.dump_model_parameter(
        parameter_name=app_context.args["parameter"],
        value=app_context.args["value"],
        instrument=app_context.args["instrument"],
        parameter_version=app_context.args["parameter_version"],
        output_file=Path(
            app_context.args["parameter"] + "-" + app_context.args["parameter_version"] + ".json"
        ),
        output_path=output_path,
        metadata_input_dict=app_context.args,
        check_db_for_existing_parameter=app_context.args.get("check_parameter_version", False),
    )


if __name__ == "__main__":
    main()
