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
            --input_data ./tests/resources/MLTdata-preproduction.ecsv

    The output is saved in simtools-output/submit_data_from_external.

    Expected final print-out message:

    .. code-block:: console

        INFO::model_data_writer(l57)::write_data::Writing data to /workdir/external/simtools/\
        simtools-output/submit_data_from_external/product-data/TEST-submit_data_from_external.ecsv

"""

import logging
from pathlib import Path

import simtools.data_model.model_data_writer as writer
import simtools.util.general as gen
from simtools.configuration import configurator
from simtools.data_model import validate_data
from simtools.data_model.metadata_collector import MetadataCollector


def _parse(label, description, usage):
    """
    Parse command line configuration

    Parameters
    ----------
    label: str
        Label describing application.
    description: str
        Description of application.
    usage: str
        Example on how to use the application.

    Returns
    -------
    CommandLineParser
        Command line parser object

    """

    config = configurator.Configurator(label=label, description=description, usage=usage)

    config.parser.add_argument(
        "--input_meta",
        help="meta data file associated to input data",
        type=str,
        required=False,
    )
    config.parser.add_argument(
        "--input_data",
        help="input data file",
        type=str,
        required=False,
    )
    config.parser.add_argument(
        "--input_data_schema",
        help="schema file describing input data",
        type=str,
        required=False,
    )
    return config.initialize(outputs=True)


def main():

    args_dict, _ = _parse(
        label=Path(__file__).stem,
        description="Submit model parameter (value, table) through an external interface.",
        usage=" python applications/submit_data_from_external.py "
        "--workflow_config <workflow configuration file>",
    )

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    _metadata = MetadataCollector(args_dict=args_dict)
    try:
        _input_data_model = _metadata.top_level_meta["cta"]["product"]["data"]["model"]
    except KeyError:
        logger.error("No data model given to describe input data")
        raise

    data_validator = validate_data.DataValidator(
        schema_file=args_dict.get("input_data_schema", None),
        data_model=_input_data_model,
        data_file=args_dict["input_data"],
    )

    file_writer = writer.ModelDataWriter(
        product_data_file=args_dict["output_file"],
        product_data_format=args_dict["output_file_format"]
    )
    file_writer.write(
        metadata=_metadata.top_level_meta,
        product_data=data_validator.validate_and_transform()
    )


if __name__ == "__main__":
    main()
