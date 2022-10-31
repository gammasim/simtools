#!/usr/bin/python3
"""
    Summary
    -------
    Submit model parameter (value, table) through an external interface.

    Prototype implementation allowing to submit metadata and
    data through the command line.


    Command line arguments
    ----------------------
    workflow_description (str, required)
        Workflow description (yml format)
    input_meta (str, required)
        input meta data file (yml format)
    input_data (str, required)
        input data file

    Example
    -------

    Runtime 1 min

    .. code-block:: console

        python ./submit_data_from_external.py \
            --workflow_description set_quantum_efficiency_from_external.yml \
            --input_meta qe_R12992-100-05b.meta.yml \
            --input_data qe_R12992-100-05b.data.ecsv \


"""

import logging
import os

import simtools.util.general as gen
import simtools.util.model_data_writer as writer
import simtools.util.validate_data as ds
from simtools.configuration import Configurator
from simtools.util.workflow_description import WorkflowDescription


def _parse(label, description, usage):
    """
    Parse command line configuration

    Returns
    -------
    CommandLineParser
        command line parser object

    """

    config = Configurator(label=label, description=description, usage=usage)

    config.parser.add_argument(
        "--input_meta",
        help="Meta data file describing input data",
        type=str,
        required=True,
    )
    config.parser.add_argument(
        "--input_data",
        help="Input data file",
        type=str,
        required=True,
    )
    return config.initialize(workflow_config=True)


def main():

    label = os.path.basename(__file__).split(".")[0]
    args_dict, _ = _parse(
        label,
        description="Submit model parameter (value, table) through an external interface.",
        usage=" python applications/submit_data_from_external.py "
        "--workflow_config <workflow configuration file>"
        "--input_meta <metadata of input data> "
        "--input_data <input data>",
    )

    logger = logging.getLogger()
    logger.setLevel(gen.getLogLevelFromUser(args_dict["log_level"]))

    workflow = WorkflowDescription(args_dict=args_dict)

    data_validator = ds.DataValidator(workflow)
    data_validator.validate()

    file_writer = writer.ModelDataWriter(workflow)
    file_writer.write_metadata()
    file_writer.write_data(data_validator.transform())


if __name__ == "__main__":
    main()
