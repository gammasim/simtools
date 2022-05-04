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
    reference_schema_directory (str, required)
        directory for reference schema
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
            --reference_schema_directory ./REFERENCE_DIR


"""

import argparse
import logging

import simtools.util.general as gen
import simtools.util.validate_schema as vs
import simtools.util.validate_data as ds
import simtools.util.workflow_configuration as workflow_config
import simtools.util.write_model_data as writer


def transform_input(_args, _workflow_config):
    """
    data transformation for simulation model data

    includes:
    -
    - schema validation
    - data validation
    - data cleaning
    - data conversion to standard units
    - metadata writer

    Parameters:
    -----------
    _args
        command line parameters
    _workflow_config
        workflow configuration

    Returns:
    -------
    output_meta: dict
        transformed user meta data
    output_data: astropy Table
        transformed data table

    """

    _schema_validator = vs.SchemaValidator(_workflow_config)
    _output_meta = _schema_validator.validate_and_transform(
        _args.input_meta_file)

    _data_validator = ds.DataValidator(
        _workflow_config["CTASIMPIPE"]["DATA_COLUMNS"],
        _args.input_data_file)
    _data_validator.validate()
    _output_data = _data_validator.transform()

    return _output_meta, _output_data


def parse():
    """
    Parse command line configuration

    """

    parser = argparse.ArgumentParser(
        description=("Setting workflow model parameter data")
    )
    parser.add_argument(
        "-c",
        "--workflow_config_file",
        help="workflow configuration",
        type=str,
        required=True,
    )
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
    parser.add_argument(
        "-p",
        "--product_directory",
        help="Directory for data products (output)",
        type=str,
        default='',
        required=False,
    )
    parser.add_argument(
        "-r",
        "--reference_schema_directory",
        help="Directory with reference schema",
        type=str,
        default=None,
        required=False
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        dest="logLevel",
        action="store",
        default="info",
        help="Log level to print (default is INFO)",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse()

    logger = logging.getLogger()
    logger.setLevel(gen.getLogLevelFromUser(args.logLevel))

    workflow_config = workflow_config.WorkflowConfiguration()
    workflow_config.collect_configuration(
        args.workflow_config_file,
        args.reference_schema_directory)

    output_meta, output_data = transform_input(
        args,
        workflow_config.configuration)

    file_writer = writer.ModelData(workflow_config.configuration)
    file_writer.write_model_file(output_meta,
                                 output_data,
                                 args.product_directory)
