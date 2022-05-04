import argparse
import logging


class CommandLineParser(argparse.ArgumentParser):
    """
    Command line parser for application and workflows

    """

    def initialize_default_arguments(self):
        """
        Initialize default arguments used for all applications

        """

        self.add_argument(
            "-v",
            "--verbosity",
            dest="logLevel",
            action="store",
            default="info",
            help="Log level to print (default is INFO)",
        )

    def initialize_workflow_arguments(self):
        """
        Initialize default arguments for workflow configuration parameters

        """

        self.add_argument(
            "-c",
            "--workflow_config_file",
            help="Workflow configuration file",
            type=str,
            required=True,
        )
        self.add_argument(
            "-p",
            "--product_directory",
            help="Directory for data products (output)",
            type=str,
            default='',
            required=False,
        )
        self.add_argument(
            "-r",
            "--reference_schema_directory",
            help="Directory with reference schema",
            type=str,
            default=None,
            required=False
        )

    def initialize_telescope_model_arguments(self):
        """
        Initialize default arguments for telescope model definition

        """
