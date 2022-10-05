#!/usr/bin/python3

"""
    Summary
    -------
    Winifu - a sandbox for configuration testing

"""

import logging
import os

import simtools.configuration as configurator
import simtools.util.general as gen


def parse(label):
    """
    Parse command line configuration

    """

    config = configurator.Configurator(label=label)
    config.parser.add_argument(
        "-m",
        "--input_meta_file",
        help="user-provided meta data file (yml)",
        type=str,
        required=True,
    )
    config.parser.add_argument(
        "--input_float",
        help="a float as input",
        type=float,
        required=True,
    )

    return config.initialize(add_workflow_config=False)

    return config.initialize(add_workflow_config=False)


def main():

    label = os.path.basename(__file__).split(".")[0]
    args_dict = parse(label)

    logger = logging.getLogger()
    logger.setLevel(gen.getLogLevelFromUser(args_dict["log_level"]))

    print("command line arguments:", args_dict)


if __name__ == "__main__":
    main()
