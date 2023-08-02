#!/usr/bin/python3
"""
    Summary
    -------
    Submit model parameter (value, table) through the command line.

    Input data and metadata is validated, and if necessary enriched
    and converted following predescribed schema.

    Command line arguments
    ----------------------
    input_meta (str, optional)
        input meta data file (yml format)
    input_data (str, optional)
        input data file

    Example
    -------

    Submit mirror measurements with associated metadata:

    .. code-block:: console

        simtools-submit-data-from-external \
            --input_meta ./tests/resources/MLTdata-preproduction.meta.yml \
            --input_data ./tests/resources/MLTdata-preproduction.ecsv \
            --input_data_schema ./tests/resources/schema_MST_mirror_2f_measurements.yml \
            --output_file TEST-submit_data_from_external.ecsv

    Expected final print-out message:

    .. code-block:: console

        INFO::model_data_writer(l70)::write_data::Writing data to\
            /simtools/simtools-output/d-2023-07-31/TEST-submit_data_from_external.ecsv

"""

import logging
from pathlib import Path

import simtools.data_model.model_data_writer as writer
import simtools.util.general as gen
from simtools.configuration import configurator
from simtools.data_model import validate_data
from simtools.data_model.metadata_collector import MetadataCollector


def _parse(label, description):
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
        "--input_meta",
        help="meta data file associated to input data",
        type=str,
        required=False,
    )
    config.parser.add_argument(
        "--input",
        help="input data file",
        type=str,
        required=False,
    )
    config.parser.add_argument(
        "--schema",
        help="schema file describing input data",
        type=str,
        required=False,
    )
    return config.initialize(outputs=True)


def main():

    args_dict, _ = _parse(
        label=Path(__file__).stem,
        description="Submit and validate data (e.g., input data to tools, model parameters).",
    )

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    _metadata = MetadataCollector(args_dict=args_dict)

    data_validator = validate_data.DataValidator(
        schema_file=args_dict.get("schema", None),
        data_file=args_dict["input"],
    )

    file_writer = writer.ModelDataWriter(
        product_data_file=args_dict["output_file"],
        product_data_format=args_dict["output_file_format"],
    )
    file_writer.write(
        metadata=_metadata.top_level_meta, product_data=data_validator.validate_and_transform()
    )


if __name__ == "__main__":
    main()
