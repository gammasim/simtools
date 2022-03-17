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
    input_data (str, required)
        User-provided data file (ecsv format)
    verbosity (str, optional)
        Log level to print (default=INFO).

    Example
    -------

    Runtime 1 min

    .. code-block:: console

        python ./set_quantum_efficiency_from_external.py \
            --workflow_schema set_quantum_efficiency_from_external.yml \
            --input_meta qe_R12992-100-05b.meta.yml \
            --input_data qe_R12992-100-05b.meta.ecsv


"""

import argparse
import logging

import simtools.config as cfg
import simtools.io_handler as io
import simtools.util.general as gen

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Setting workflow for photodetector / quantum efficiency. "
        )
    )
    parser.add_argument(
        "-w",
        "--workflow_schema",
        help="workflow description",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-m",
        "--input_meta",
        help="User-provided meta data file (yml)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-d",
        "--input_data",
        help="User-provided data file (ecsv)",
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
#    output_meta, output_data = transformInput(
#        args.workflow_schema,
#        args.input_meta,
#        args.input_data)

    # write model data in the format expected 
#    writeModelData(
#        output_meta,
#        output_data)
