#!/usr/bin/python3

"""
    Summary
    -------
    Setting workflow for photodetector / quantum efficiency


    Command line arguments
    ----------------------
    workflow_schema (str, required)
        Workflow description (yml format)
    input_meta (str, required)
        User-provided meta data file (yml format)
    verbosity (str, optional)
        Log level to print (default=INFO).

    Example
    -------

    Runtime 1 min

    .. code-block:: console

        python ./set_quantum_efficiency_from_external.py \
            --workflow_schema_file set_quantum_efficiency_from_external.yml \
            --input_meta_file qe_R12992-100-05b.usermeta.yml \


"""

import argparse
import logging

import simtools.config as cfg
import simtools.io_handler as io
import simtools.util.general as gen
import simtools.util.validate_schema as validator

def transformInput(
    workflow_schema_file,
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
        workflow_schema_file,
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
        "-w",
        "--workflow_schema_file",
        help="workflow description",
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
        "-v",
        "--verbosity",
        dest="logLevel",
        action="store",
        default="info",
        help="Log level to print (default is INFO)",
    )

    args = parser.parse_args()
    label = "set_quantum_efficiency_from_external"

    logger = logging.getLogger()
    logger.setLevel(gen.getLogLevelFromUser(args.logLevel))

    outputDir = io.getApplicationOutputDirectory(cfg.get("outputLocation"), label)
    logger.info("Outputdirectory {}".format(outputDir))

    # validate, transform, clean, enrich user metadata and data
    output_meta, output_data = transformInput(
        args.workflow_schema_file,
        args.input_meta_file)

    # write model data in the format expected
#    writeModelData(
#        output_meta,
#        output_data)
