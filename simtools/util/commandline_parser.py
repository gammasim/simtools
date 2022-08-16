import argparse

import simtools.version
import simtools.util.names as names


class CommandLineParser(argparse.ArgumentParser):
    """
    Command line parser for application and workflows
    Wrapper around standard python argparse.ArgumentParser

    Methods
    -------
    initialize_default_arguments:
        Initialize default arguments used by all applications
    initialize_telescope_model_arguments:
        nitialize default arguments for telescope model definitions

    """

    def initialize_default_arguments(self, add_workflow_config=True):
        """
        Initialize default arguments used by all applications
        (e.g., verbosity or test flag)


        """

        self.add_argument(
            "--configFile",
            help="gammasim-tools configuration file",
            required=False,
        )
        if add_workflow_config:
            self.add_argument(
                "-c",
                "--workflow_config_file",
                help="Workflow configuration file",
                type=str,
                required=False,
            )
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
        self.add_argument(
            '-V',
            '--version',
            action='version',
            version=f'%(prog)s {simtools.version.__version__}'
        )

    def initialize_telescope_model_arguments(self,
                                             add_model_version=True,
                                             add_telescope=True):
        """
        Initialize default arguments for site and telescope model
        definition

        """

        self.add_argument(
            "-s",
            "--site",
            help="CTAO site (e.g. North, South)",
            type=self.site,
            required=True
        )
        if add_telescope:
            self.add_argument(
                "-t",
                "--telescope",
                help="Telescope model name (e.g. LST-1, SST-D, ...)",
                type=str,
                required=True,
            )
        if add_model_version:
            self.add_argument(
                "-m",
                "--model_version",
                help="Model version (default=Current)",
                type=str,
                default="Current",
            )

    @staticmethod
    def site(value):
        """
        Argument parser type to check that a valid site name is given

        Parameters
        ----------
        value: str
            site name

        Raises
        ------
        argparse.ArgumentTypeError
            for invalid sites


        """

        fsite = str(value)

        if not names.validateSiteName(fsite):
            raise argparse.ArgumentTypeError(
                "{} is an invalid site".format(fsite))

        return fsite

    @staticmethod
    def efficiency_interval(value):
        """
        Argument parser type to check that value is an efficiency
        in the interval [0,1]

        Parameters
        ----------
        value: float
            value provided through the command line

        Raises
        ------
        argparse.ArgumentTypeError
            When value is outside of the interval [0,1]


        """
        fvalue = float(value)
        if fvalue < 0. or fvalue > 1.:
            raise argparse.ArgumentTypeError(
                "{} outside of allowed [0,1] interval".format(value))

        return fvalue
