#!/usr/bin/python3

"""
    Summary
    -------
    Setting workflow for a model parameter (value, table)
    through an external interface.

    Prototype implementation allowing to submit metadata and
    data through the command line.


    Command line arguments
    ----------------------
    workflow_config_file (str, required)
        Workflow configuration (yml format)
    input_meta (str, required)
        User-provided meta data file (yml format)
    input_data (str, required)
        User-provided data file
    verbosity (str, optional)
        Log level to print (default=INFO).

    Example
    -------

    Runtime 1 min

    .. code-block:: console

        python ./set_modelparameter_from_external.py \
            --workflow_config_file set_quantum_efficiency_from_external.yml \
            --input_meta_file qe_R12992-100-05b.usermeta.yml \
            --input_data_file qe_R12992-100-05b.data.ecsv \


"""

import logging
import os

import simtools.util.general as gen
import simtools.util.validate_data as ds
import simtools.util.commandline_parser as argparser
import simtools.util.workflow_description as workflow_config
import simtools.util.model_data_writer as writer


def parse(label):
    """
    Parse command line configuration

    """

    parser = argparser.CommandLineParser(label)
    parser.add_argument(
        "-m",
        "--input_meta_file",
        help="User-provided meta data file (yml)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-d",
        "--input_data_file",
        help="User-provided data file (ecsv)",
        type=str,
        required=True,
    )
    parser.initialize_default_arguments()
    return parser.parse_args()


if __name__ == "__main__":

    label = os.path.basename(__file__).split('.')[0]
    args = parse(label)

    logger = logging.getLogger()
    logger.setLevel(gen.getLogLevelFromUser(args.logLevel))

    workflow = workflow_config.WorkflowDescription(label=label, args=args)

    data_validator = ds.DataValidator(workflow)
    data_validator.validate()

    file_writer = writer.ModelDataWriter(workflow)
    file_writer.write_metadata()
    file_writer.write_data(data_validator.transform())
