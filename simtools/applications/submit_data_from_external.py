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
        help="meta data file describing input data",
        type=str,
        required=False,
    )
    config.parser.add_argument(
        "--input_data",
        help="input data file",
        type=str,
        required=False,
    )
    return config.initialize(workflow_config=True)


def main():

    label = Path(__file__).stem
    args_dict, _ = _parse(
        label,
        description="Submit model parameter (value, table) through an external interface.",
        usage=" python applications/submit_data_from_external.py "
        "--workflow_config <workflow configuration file>",
    )

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    workflow = MetadataCollector(args_dict=args_dict)

    data_validator = validate_data.DataValidator(workflow)
    data_validator.validate()

    file_writer = writer.ModelDataWriter(workflow)
    file_writer.write_metadata()
    file_writer.write_data(data_validator.transform())


if __name__ == "__main__":
    main()
