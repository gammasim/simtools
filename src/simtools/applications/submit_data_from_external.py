#!/usr/bin/python3
r"""
    Submit data file through the command line.

    Input data and metadata is validated, and if necessary enriched
    and converted following a pre-described schema.

    Command line arguments
    ----------------------
    input_meta (str, optional)
        input meta data file (yml format)
    input (str, optional)
        input data file
    schema (str, optional)
        Schema describing the input data

    Example
    -------

    Submit mirror measurements with associated metadata:

    .. code-block:: console

        simtools-submit-data-from-external \\
            --input_meta ./tests/resources/MLTdata-preproduction.meta.yml \\
            --input ./tests/resources/MLTdata-preproduction.ecsv \\
            --schema src/simtools/schemas/input/MST_mirror_2f_measurements.schema.yml \\
            --output_file TEST-submit_data_from_external.ecsv

    Expected final print-out message:

    .. code-block:: console

        INFO::model_data_writer(l70)::write_data::Writing data to \\
            /simtools/simtools-output/d-2023-07-31/TEST-submit_data_from_external.ecsv

"""

import simtools.data_model.model_data_writer as writer
from simtools.application_control import get_application_label, startup_application
from simtools.configuration import configurator
from simtools.data_model import validate_data
from simtools.data_model.metadata_collector import MetadataCollector


def _parse():
    """Parse command line configuration."""
    config = configurator.Configurator(
        label=get_application_label(__file__),
        description="Submit and validate data (e.g., input data to tools, model parameters).",
    )

    config.parser.add_argument(
        "--input_meta",
        help="meta data file associated to input data",
        type=str,
        required=False,
    )
    config.parser.add_argument(
        "--input",
        help="input data file",
        type=str,
        required=True,
    )
    config.parser.add_argument(
        "--schema",
        help="schema file describing input data",
        type=str,
        required=False,
    )
    config.parser.add_argument(
        "--ignore_metadata",
        help="Ignore metadata",
        action="store_true",
        required=False,
    )
    return config.initialize(output=True)


def main():
    """Submit and validate data (e.g., input data to tools, model parameters)."""
    app_context = startup_application(_parse)

    _metadata = (
        None if app_context.args.get("ignore_metadata") else MetadataCollector(app_context.args)
    )

    data_validator = validate_data.DataValidator(
        schema_file=(
            _metadata.get_data_model_schema_file_name()
            if _metadata
            else app_context.args.get("schema")
        ),
        data_file=app_context.args["input"],
    )

    writer.ModelDataWriter.dump(
        args_dict=app_context.args,
        metadata=_metadata,
        product_data=data_validator.validate_and_transform(),
    )


if __name__ == "__main__":
    main()
