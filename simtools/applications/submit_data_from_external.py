#!/usr/bin/python3
"""
    Summary
    -------
    Submit model parameter (value, table) through an external interface.

    Prototype implementation allowing to submit metadata and
    data through the command line.

    Command line arguments
    ----------------------
    input_meta (str, optional)
        input meta data file (yml format)
    input_data (str, optional)
        input data file

    Example
    -------

    Get workflow configuration from the DB:

    .. code-block:: console

        simtools-get-file-from_DB --file_name \
        set_MST_mirror_2f_measurements_from_external.config.yml

    Run the application:

    .. code-block:: console

        simtools-submit-data-from-external --workflow_config \
        set_MST_mirror_2f_measurements_from_external.config.yml

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
from simtools.data_model.workflow_description import WorkflowDescription


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
        help="Meta data file describing input data",
        type=str,
        required=False,
    )
    config.parser.add_argument(
        "--input_data",
        help="Input data file",
        type=str,
        required=False,
    )
    return config.initialize(workflow_config=True)


def main():
    label = Path(__file__).stem
    args_dict, _ = _parse(
        label,
        description="Submit model parameter (value, table) through an external interface.",
    )

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    workflow = WorkflowDescription(args_dict=args_dict)

    data_validator = validate_data.DataValidator(workflow)
    data_validator.validate()

    file_writer = writer.ModelDataWriter(workflow)
    file_writer.write_metadata()
    file_writer.write_data(data_validator.transform())


if __name__ == "__main__":
    main()
