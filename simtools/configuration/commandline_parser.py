import argparse
from pathlib import Path

import simtools.version
from simtools.util import names

__all__ = [
    "CommandLineParser",
]


class CommandLineParser(argparse.ArgumentParser):
    """
    Command line parser for application and workflows.

    Wrapper around standard python argparse.ArgumentParser.

    Command line arguments should be given in snake_case, e.g.  `input_meta`.

    Parameters
    ----------
    argparse.ArgumentParser class
        Object for parsing command line strings into Python objects. For a list of keywords, please\
        refer to argparse.ArgumentParser documentation.
    """

    def initialize_default_arguments(
        self,
        paths=True,
        telescope_model=False,
        workflow_config=False,
        db_config=False,
        job_submission=False,
    ):
        """
        Initialize default arguments used by all applications (e.g., verbosity or test flag).

        Parameters
        ----------
        paths: bool
            Add path configuration to list of args.
        telescope_model: bool
            Add telescope model configuration to list of args.
        workflow_config: bool
            Add workflow configuration to list of args.
        db_config: bool
            Add database configuration parameters to list of args.
        job_submission: bool
            Add job submission configuration parameters to list of args.
        """

        if telescope_model:
            self.initialize_telescope_model_arguments()
        if job_submission:
            self.initialize_job_submission_arguments()
        if db_config:
            self.initialize_db_config_arguments()
        if paths:
            self.initialize_path_arguments()
        self.initialize_config_files(workflow_config)
        self.initialize_application_execution_arguments()

    def initialize_config_files(self, workflow_config=False):
        """
        Initialize configuration and workflow files.

        Parameters
        ----------
        workflow_config: str
            workflow configuration file.
        """

        _job_group = self.add_argument_group("configuration")
        _job_group.add_argument(
            "--config",
            help="simtools configuration file",
            default=None,
            type=str,
            required=False,
        )
        if workflow_config:
            _job_group.add_argument(
                "--workflow_config",
                help="workflow configuration file",
                type=str,
                required=False,
            )

    def initialize_path_arguments(self):
        """
        Initialize paths.
        """
        _job_group = self.add_argument_group("paths")
        _job_group.add_argument(
            "--data_path",
            help="path pointing towards data directory",
            type=Path,
            default="./data/",
            required=False,
        )
        _job_group.add_argument(
            "--output_path",
            help="path pointing towards output directory",
            type=Path,
            default="./",
            required=False,
        )
        _job_group.add_argument(
            "--use_plain_output_path",
            help="use plain output path (without adding the tool name and dates)",
            action="store_true",
            required=False,
        )
        _job_group.add_argument(
            "--model_path",
            help="path pointing towards simulation model file directory",
            type=Path,
            default="./",
            required=False,
        )
        _job_group.add_argument(
            "--simtel_path",
            help="path pointing to sim_telarray installation",
            type=Path,
            required=False,
        )

    def initialize_application_execution_arguments(self):
        """
        Initialize application execution arguments.
        """

        _job_group = self.add_argument_group("execution")
        _job_group.add_argument(
            "--test",
            help="test option for faster execution during development",
            action="store_true",
            required=False,
        )
        _job_group.add_argument(
            "--label",
            help="Job label",
            required=False,
        )
        _job_group.add_argument(
            "--log_level",
            action="store",
            default="info",
            help="log level to print (default is INFO)",
            required=False,
        )
        _job_group.add_argument(
            "--version", action="version", version=f"%(prog)s {simtools.version.__version__}"
        )

    def initialize_db_config_arguments(self):
        """
        Initialize DB configuration parameters.
        """

        _job_group = self.add_argument_group("MongoDB configuration")
        _job_group.add_argument("--db_api_user", help="database user", type=str, required=False)
        _job_group.add_argument("--db_api_pw", help="database password", type=str, required=False)
        _job_group.add_argument("--db_api_port", help="database port", type=int, required=False)
        _job_group.add_argument(
            "--db_server", help="database server address", type=str, required=False
        )
        _job_group.add_argument(
            "--db_api_authentication_database",
            help="database  with user info (optional, default is 'admin')",
            type=str,
            required=False,
            default="admin",
        )

    def initialize_job_submission_arguments(self):
        """
        Initialize job submission arguments for simulator.

        """
        _job_group = self.add_argument_group("job submission")
        _job_group.add_argument(
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
        _job_group.add_argument(
            "--extra_submit_options",
            help="Additional options for submission command",
            type=str,
            required=False,
        )

    def initialize_telescope_model_arguments(self, add_model_version=True, add_telescope=True):
        """
        Initialize default arguments for site and telescope model definition

        Parameters
        ----------
        add_model_version: bool
            Set to allow a simulation model argument.
        add_telescope: bool
            Set to allow a telescope name argument.
        """

        _job_group = self.add_argument_group("telescope model")
        _job_group.add_argument(
            "--site", help="CTAO site (e.g., North, South)", type=self.site, required=False
        )
        if add_telescope:
            _job_group.add_argument(
                "--telescope",
                help="telescope model name (e.g., LST-1, SST-D, ...)",
                type=self.telescope,
            )
        if add_model_version:
            _job_group.add_argument(
                "--model_version",
                help="model version",
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

        Returns
        -------
        str
            Validated site name

        Raises
        ------
        argparse.ArgumentTypeError
            for invalid sites

        """

        fsite = str(value)
        if not names.validate_site_name(fsite):
            raise argparse.ArgumentTypeError(f"{fsite} is an invalid site")
        return fsite

    @staticmethod
    def telescope(value):
        """
        Argument parser type to check that a valid telescope name is given

        Parameters
        ----------
        value: str
            telescope name

        Returns
        -------
        str
            Validated telescope name

        Raises
        ------
        argparse.ArgumentTypeError
            for invalid telescope

        """

        ftelescope = str(value)
        if not names.validate_telescope_model_name(ftelescope):
            raise argparse.ArgumentTypeError(f"{ftelescope} is an invalid telescope name")
        return ftelescope

    @staticmethod
    def efficiency_interval(value):
        """
        Argument parser type to check that value is an efficiency
        in the interval [0,1]

        Parameters
        ----------
        value: float
            value provided through the command line

        Returns
        -------
        float
            Validated efficiency interval

        Raises
        ------
        argparse.ArgumentTypeError
            When value is outside of the interval [0,1]


        """
        fvalue = float(value)
        if fvalue < 0.0 or fvalue > 1.0:
            raise argparse.ArgumentTypeError(f"{value} outside of allowed [0,1] interval")

        return fvalue

    @staticmethod
    def zenith_angle(angle):
        """
        Argument parser type to check that the zenith angle provided is in the interval [0, 180].
        We allow here zenith angles larger than 90 degrees in the improbable case
        such simulations are requested. It is not guaranteed that the actual simulation software
        supports such angles!

        Parameters
        ----------
        angle: float
            zenith angle to verify

        Returns
        -------
        float
            Validated zenith angle

        Raises
        ------
        argparse.ArgumentTypeError
            When angle is outside of the interval [0, 180]


        """
        fangle = float(angle)
        if fangle < 0.0 or fangle > 180.0:
            raise argparse.ArgumentTypeError(
                f"The provided zenith angle, {angle:.1f}, "
                "is outside of the allowed [0, 180] interval"
            )

        return fangle
