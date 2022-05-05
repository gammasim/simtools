import argparse


class CommandLineParser(argparse.ArgumentParser):
    """
    Command line parser for application and workflows

    """

    def initialize_default_arguments(self):
        """
        Initialize default arguments used for all applications

        """

        self.add_argument(
            "--test",
            help="Test option for faster execution during development",
            action="store_true",
            required=False,
        )
        self.add_argument(
            "-v",
            "--verbosity",
            dest="logLevel",
            action="store",
            default="info",
            help="Log level to print (default is INFO)",
            required=False,
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

        self.add_argument(
            "-s",
            "--site",
            help="North or South",
            type=str,
            required=True
        )
        self.add_argument(
            "-t",
            "--telescope",
            help="Telescope model name (e.g. LST-1, SST-D, ...)",
            type=str,
            required=True,
        )
        self.add_argument(
            "-m",
            "--model_version",
            help="Model version (default=Current)",
            type=str,
            default="Current",
        )

    @staticmethod
    def efficiency_interval(value):
        """
        Argument parser check that value is an efficiency in the interval [0,1]

        """
        fvalue = float(value)
        if fvalue < 0. or fvalue > 1.:
            raise argparse.ArgumentTypeError(
                "{} outside of allowed [0,1] interval".format(value))

        return fvalue
