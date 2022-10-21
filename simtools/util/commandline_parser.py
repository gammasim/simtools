import argparse
from pathlib import Path

import simtools.util.names as names
import simtools.version


class CommandLineParser(argparse.ArgumentParser):
    """
    Command line parser for application and workflows
    Wrapper around standard python argparse.ArgumentParser

    Command line arguments should be given in snake_case, e.g. `config_file`.

    Methods
    -------
    initialize_default_arguments:
        Initialize default arguments used by all applications
    initialize_telescope_model_arguments:
        nitialize default arguments for telescope model definitions

    """

    def initialize_default_arguments(self, add_workflow_config=True):
        """
        Initialize default arguments used by all applications (e.g., verbosity or test flag).

        Parameters
        ----------
        add_workflow_config: bool
           Add workflow configuration file to list of args.

        """

        self.add_argument(
            "--config_file",
            help="gammasim-tools configuration file",
            default=None,
            type=str,
            required=False,
        )
        if add_workflow_config:
            self.add_argument(
                "--workflow_config_file",
                help="workflow configuration file",
                type=str,
                required=False,
            )
        self.add_argument(
            "--data_path",
            help="path pointing towards data directory",
            type=Path,
            default="./data/",
            required=False,
        )
        self.add_argument(
            "--output_path",
            help="path pointing towards output directory",
            type=Path,
            default="./",
            required=False,
        )
        self.add_argument(
            "--model_path",
            help="path pointing towards model file directory (temporary - will go in future)",
            type=Path,
            default="./",
            required=False,
        )
        self.add_argument(
            "--mongodb_config_file",
            help="configuration file for Mongo DB",
            type=str,
            default="dbDetails.yml",
            required=False,
        )
        self.add_argument(
            "--simtelpath",
            help="path pointing to sim_telarray installation",
            type=Path,
            required=False,
        )
        self.add_argument(
            "--test",
            help="test option for faster execution during development",
            action="store_true",
            required=False,
        )
        self.add_argument(
            "-v",
            "--log_level",
            action="store",
            default="info",
            help="log level to print (default is INFO)",
            required=False,
        )
        self.add_argument(
            "-V", "--version", action="version", version=f"%(prog)s {simtools.version.__version__}"
        )

    def initialize_job_submission_arguments(self):
        """
        Initialize job submission arguments for simulator.

        """

        self.add_argument(
            "--submit_command",
            help="Job submission command",
            type=str,
            required=True,
            choices=[
                "qsub",
                "condor_submit",
                "local",
            ],
        )
        self.add_argument(
            "--extra_submit_options",
            help="Additional options for submission command",
            type=str,
            required=False,
        )

    def initialize_telescope_model_arguments(self, add_model_version=True, add_telescope=True):
        """
        Initialize default arguments for site and telescope model
        definition

        """

        self.add_argument(
            "--site", help="CTAO site (e.g. North, South)", type=self.site, required=True
        )
        if add_telescope:
            self.add_argument(
                "--telescope",
                help="telescope model name (e.g. LST-1, SST-D, ...)",
                type=str,
                required=True,
            )
        if add_model_version:
            self.add_argument(
                "--model_version",
                help="model version (default=Current)",
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
            raise argparse.ArgumentTypeError("{} is an invalid site".format(fsite))

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
        if fvalue < 0.0 or fvalue > 1.0:
            raise argparse.ArgumentTypeError("{} outside of allowed [0,1] interval".format(value))

        return fvalue
