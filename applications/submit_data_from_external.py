#!/usr/bin/python3
"""
    Summary
    -------
    Submit model parameter (value, table) through an external interface.

    Prototype implementation allowing to submit metadata and
    data through the command line.


    Command line arguments
    ----------------------
    workflow_config (str, required)
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
            --workflow_config set_quantum_efficiency_from_external.yml \
            --input_meta qe_R12992-100-05b.usermeta.yml \
            --input_data qe_R12992-100-05b.data.ecsv \


"""

import logging
import os

import simtools.configuration as configurator
import simtools.util.general as gen
import simtools.util.model_data_writer as writer
import simtools.util.validate_data as ds
import simtools.util.workflow_description as workflow_config


def _parse(label, description):
    """
    Parse command line configuration

    Returns
    -------
    CommandLineParser
        command line parser object

    """

    config = configurator.Configurator(label=label, description=description)

    config.parser.add_argument(
        "--input_meta",
        help="Meta data file describing input data (yml)",
        type=str,
        required=True,
    )
    config.parser.add_argument(
        "--input_data",
        help="Input data file (ecsv)",
        type=str,
        required=True,
    )
    return config.initialize(workflow_config=True)


def main():

    label = os.path.basename(__file__).split(".")[0]
    args_dict, _ = _parse(
        label, description="Submit model parameter (value, table) through an external interface."
    )

    logger = logging.getLogger()
    logger.setLevel(gen.getLogLevelFromUser(args_dict["log_level"]))

    workflow = workflow_config.WorkflowDescription(args_dict=args_dict)

    data_validator = ds.DataValidator(workflow)
    data_validator.validate()

    file_writer = writer.ModelDataWriter(workflow)
    file_writer.write_metadata()
    file_writer.write_data(data_validator.transform())


if __name__ == "__main__":
    main()
