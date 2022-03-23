#!/usr/bin/python3

"""
    Summary
    -------
    Setting workflow for photodetector / quantum efficiency


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

        python ./set_quantum_efficiency_from_external.py \
            --workflow_config_file set_quantum_efficiency_from_external.yml \
            --input_meta_file qe_R12992-100-05b.usermeta.yml \
            --input_data_file qe_R12992-100-05b.data.ecsv \
            --reference_schema_directory ./REFERENCE_DIR


"""

import argparse
import logging

import simtools.config as cfg
import simtools.io_handler as io
import simtools.util.general as gen
import simtools.util.validate_schema as validator

def transformInput(
        workflow_config,
        reference_schema_dir,
        input_meta_file):
    """
    data transformation for simulation model data

    includes:
    -
    - schema validation
    - data validation
    - data cleaning
    - data conversion to standard units
    - metadata enrichment
    """

    input_meta = gen.collectDataFromYamlOrDict(
        input_meta_file,
        None)

    _schema_validator = validator.SchemaValidator(
        reference_schema_dir + '/' +
        workflow_config["CTASIMPIPE"]["SCHEMA"]["USERINPUT"],
        input_meta)
    _schema_validator.validate()

    return None, None


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Setting workflow for photodetector / quantum efficiency. "
        )
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
        "-r",
        "--reference_schema_directory",
        help="Directory with reference schema",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        dest="logLevel",
        action="store",
        default="info",
        help="Log level to print (default is INFO)",
    )

    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(gen.getLogLevelFromUser(args.logLevel))

    workflow_config = gen.collectDataFromYamlOrDict(
        args.workflow_config_file,
        None)

    outputDir = io.getApplicationOutputDirectory(
        cfg.get("outputLocation"),
        workflow_config["CTASIMPIPE"]["ACTIVITY"]["NAME"])
    logger.info("Outputdirectory %s", outputDir)

    # validate, transform, clean, enrich user metadata and data
    output_meta, output_data = transformInput(
        workflow_config,
        args.reference_schema_directory,
        args.input_meta_file)

    # write model data in the format expected
#    writeModelData(
#        output_meta,
#        output_data)
