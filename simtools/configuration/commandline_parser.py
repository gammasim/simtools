"""Command line parser for applications."""

import argparse
import logging
from pathlib import Path

import astropy.units as u

import simtools.version
from simtools.utils import names

__all__ = [
    "CommandLineParser",
]


class CommandLineParser(argparse.ArgumentParser):
    """
    Command line parser for applications.

    Wrapper around standard python argparse.ArgumentParser.

    Command line arguments should be given in snake_case, e.g.  input_meta.

    Parameters
    ----------
    argparse.ArgumentParser class
        Object for parsing command line strings into Python objects. For a list of keywords, please\
        refer to argparse.ArgumentParser documentation.
    """

    def initialize_default_arguments(
        self,
        paths=True,
        output=False,
        simulation_model=None,
        db_config=False,
        job_submission=False,
    ):
        """
        Initialize default arguments used by all applications (e.g., verbosity or test flag).

        Parameters
        ----------
        paths: bool
            Add path configuration to list of args.
        output: bool
            Add output file configuration to list of args.
        simulation_model: list
            List of simulation model configuration parameters to add to list of args
            (use: 'version', 'telescope', 'site')
        db_config: bool
            Add database configuration parameters to list of args.
        job_submission: bool
            Add job submission configuration parameters to list of args.
        """
        self.initialize_simulation_model_arguments(simulation_model)
        if job_submission:
            self.initialize_job_submission_arguments()
        if db_config:
            self.initialize_db_config_arguments()
        if paths:
            self.initialize_path_arguments()
        if output:
            self.initialize_output_arguments()
        self.initialize_config_files()
        self.initialize_application_execution_arguments()

    def initialize_config_files(self):
        """Initialize configuration files."""
        _job_group = self.add_argument_group("configuration")
        _job_group.add_argument(
            "--config",
            help="simtools configuration file",
            default=None,
            type=str,
            required=False,
        )
        _job_group.add_argument(
            "--env_file",
            help="file with environment variables",
            default=".env",
            type=str,
            required=False,
        )

    def initialize_path_arguments(self):
        """Initialize paths."""
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
            default="./simtools-output/",
            required=False,
        )
        _job_group.add_argument(
            "--use_plain_output_path",
            help="use plain output path (without the tool name and dates)",
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

    def initialize_output_arguments(self):
        """Initialize application output files(s)."""
        _job_group = self.add_argument_group("output")
        _job_group.add_argument(
            "--output_file",
            help="output data file",
            type=str,
            required=False,
        )
        _job_group.add_argument(
            "--output_file_format",
            help="file format of output data",
            type=str,
            default="ecsv",
            required=False,
        )
        _job_group.add_argument(
            "--skip_output_validation",
            help="skip output data validation against schema",
            default=False,
            required=False,
            action="store_true",
        )

    def initialize_application_execution_arguments(self):
        """Initialize application execution arguments."""
        _job_group = self.add_argument_group("execution")
        _job_group.add_argument(
            "--test",
            help="test option for faster execution during development",
            action="store_true",
            required=False,
        )
        _job_group.add_argument(
            "--label",
            help="job label",
            required=False,
        )
        _job_group.add_argument(
            "--log_level",
            action="store",
            default="info",
            help="log level to print",
            required=False,
        )
        _job_group.add_argument(
            "--version", action="version", version=f"%(prog)s {simtools.version.__version__}"
        )

    def initialize_db_config_arguments(self):
        """Initialize DB configuration parameters."""
        _job_group = self.add_argument_group("MongoDB configuration")
        _job_group.add_argument("--db_api_user", help="database user", type=str, required=False)
        _job_group.add_argument("--db_api_pw", help="database password", type=str, required=False)
        _job_group.add_argument("--db_api_port", help="database port", type=int, required=False)
        _job_group.add_argument(
            "--db_server", help="database server address", type=str, required=False
        )
        _job_group.add_argument(
            "--db_api_authentication_database",
            help="database  with user info (optional)",
            type=str,
            required=False,
            default="admin",
        )
        _job_group.add_argument(
            "--db_simulation_model",
            help="name of simulation model database",
            type=str,
            required=False,
            default=None,
        )
        _job_group.add_argument(
            "--db_simulation_model_url",
            help="simulation model repository URL",
            type=str,
            required=False,
            default=None,
        )

    def initialize_job_submission_arguments(self):
        """Initialize job submission arguments for simulator."""
        _job_group = self.add_argument_group("job submission")
        _job_group.add_argument(
            "--submit_command",
            help="job submission command",
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
            help="additional options for submission command",
            type=str,
            required=False,
        )

    def initialize_simulation_model_arguments(self, model_options):
        """
        Initialize default arguments for simulation model definition.

        Note that the model version is always required.

        Parameters
        ----------
        model_options: list
            Options to be set: "telescope", "site", "layout", "layout_file"
        """
        if model_options is None:
            return

        _job_group = self.add_argument_group("simulation model")
        _job_group.add_argument(
            "--model_version",
            help="model version",
            type=str,
            default="Released",
        )
        if any(
            option in model_options for option in ["site", "telescope", "layout", "layout_file"]
        ):
            self._add_model_option_site(_job_group)

        if "telescope" in model_options:
            _job_group.add_argument(
                "--telescope",
                help="telescope model name (e.g., LSTN-01, SSTS-design, ...)",
                type=self.telescope,
            )

        if "layout" in model_options or "layout_file" in model_options:
            _job_group = self._add_model_option_layout(_job_group, "layout_file" in model_options)

    @staticmethod
    def _add_model_option_layout(job_group, add_layout_file):
        """
        Add layout option to the job group.

        Parameters
        ----------
        job_group: argparse.ArgumentParser
            Job group
        add_layout_file: bool
            Add layout file option

        Returns
        -------
        argparse.ArgumentParser
        """
        _layout_group = job_group.add_mutually_exclusive_group(required=False)
        _layout_group.add_argument(
            "--array_layout_name",
            help="array layout name (e.g., alpha, subsystem_msts)",
            type=str,
            required=False,
        )
        _layout_group.add_argument(
            "--array_element_list",
            help="list of array elements (e.g., LSTN-01, LSTN-02, MSTN).",
            nargs="+",
            type=str,
            required=False,
            default=None,
        )
        if add_layout_file:
            _layout_group.add_argument(
                "--array_layout_file",
                help="file(s) with the list of array elements (astropy table format).",
                nargs="+",
                type=str,
                required=False,
                default=None,
            )
        return job_group

    def _add_model_option_site(self, job_group):
        """
        Add site option to the job group.

        Parameters
        ----------
        job_group: argparse.ArgumentParser
            Job group

        Returns
        -------
        argparse.ArgumentParser
        """
        job_group.add_argument(
            "--site", help="site (e.g., North, South)", type=self.site, required=False
        )
        return job_group

    @staticmethod
    def site(value):
        """
        Argument parser type to check that a valid site name is given.

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
        names.validate_site_name(str(value))
        return str(value)

    @staticmethod
    def telescope(value):
        """
        Argument parser type to check that a valid telescope name is given.

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
        names.validate_telescope_name(str(value))
        return str(value)

    @staticmethod
    def efficiency_interval(value):
        """
        Argument parser type to check that value is an efficiency in the interval [0,1].

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
        angle: float, str, astropy.Quantity
            zenith angle to verify

        Returns
        -------
        Astropy.Quantity
            Validated zenith angle in degrees

        Raises
        ------
        argparse.ArgumentTypeError
            When angle is outside of the interval [0, 180]


        """
        logger = logging.getLogger(__name__)

        try:
            try:
                fangle = float(angle) * u.deg
            except ValueError:
                fangle = u.Quantity(angle).to("deg")
        except TypeError as exc:
            logger.error(
                "The zenith angle provided is not a valid numerical or astropy.Quantity value."
            )
            raise exc
        if fangle < 0.0 * u.deg or fangle > 180.0 * u.deg:
            raise argparse.ArgumentTypeError(
                f"The provided zenith angle, {angle:.1f}, "
                "is outside of the allowed [0, 180] interval"
            )
        return fangle

    @staticmethod
    def azimuth_angle(angle):
        """
        Argument parser type to check that the azimuth angle provided is in the interval [0, 360].

        Other allowed options are north, south, east or west which will be translated to an angle
        where north corresponds to zero.

        Parameters
        ----------
        angle: float or str
            azimuth angle to verify or convert

        Returns
        -------
        Astropy.Quantity
            Validated/Converted azimuth angle in degrees

        Raises
        ------
        argparse.ArgumentTypeError
            When angle is outside of the interval [0, 360] or not in (north, south, east, west)


        """
        logger = logging.getLogger(__name__)
        try:
            fangle = float(angle)
            if fangle < 0.0 or fangle > 360.0:
                raise argparse.ArgumentTypeError(
                    f"The provided azimuth angle, {angle:.1f}, "
                    "is outside of the allowed [0, 360] interval"
                )

            return fangle * u.deg
        except ValueError:
            logger.debug(
                "The azimuth angle provided is not a valid numeric value. "
                "Will check if it is an astropy.Quantity instead"
            )
        try:
            return u.Quantity(angle).to("deg")
        except TypeError:
            logger.debug(
                "The azimuth angle provided is not a valid astropy.Quantity. "
                "Will check if it is (north, south, east, west) instead"
            )
        if isinstance(angle, str):
            azimuth_angle = angle.lower()
            if azimuth_angle == "north":
                return 0 * u.deg
            if azimuth_angle == "south":
                return 180 * u.deg
            if azimuth_angle == "east":
                return 90 * u.deg
            if azimuth_angle == "west":
                return 270 * u.deg
            raise argparse.ArgumentTypeError(
                "The azimuth angle can only be a number or one of "
                f"(north, south, east, west), not {angle}"
            )
        logger.error(
            f"The azimuth value provided, {angle}, is not a valid number "
            "nor one of (north, south, east, west)."
        )
        raise TypeError

    @staticmethod
    def energy_range(energy_range, energy_unit="GeV"):
        """
        Argument parser type for energy ranges.

        Energy ranges are given as string in following the example "10 GeV 200 PeV".

        Parameters
        ----------
        energy_range : str
            energy range string
        energy_unit : str
            default energy unit to use

        Returns
        -------
        string
            energy range  (min, max)
        """
        logger = logging.getLogger(__name__)

        parts = energy_range.split()
        if len(parts) != 4:
            logger.error(
                "Energy range must be given in the form 'E1 unit E1 unit' (e.g., '10 GeV 100 TeV')"
            )
            raise TypeError
        value1, unit1, value2, unit2 = parts
        try:
            energy1 = float(value1) * u.Unit(unit1)
            energy2 = float(value2) * u.Unit(unit2)
        except ValueError as exc:
            logger.error(f"Invalid energy values: {value1} {unit1} {value2} {unit2}")
            raise exc
        return f"{energy1.to(energy_unit)} {energy2.to(energy_unit)}"

    @staticmethod
    def viewcone(viewcone):
        """
        Argument parser type for viewcone argument.

        Viewcone is given as string in the form "min max" where min and max are in degrees.

        Parameters
        ----------
        viewcone: str
            viewcone string

        Returns
        -------
        string
            viewcone (min, max)
        """
        logger = logging.getLogger(__name__)

        parts = viewcone.split()
        if len(parts) != 4:
            logger.error(
                "Viewcone must be given in the form 'min deg max deg ' (e.g., '0 deg 5 deg')"
            )
            raise TypeError
        value1, unit1, value2, unit2 = parts
        try:
            viewcone_min = float(value1) * u.Unit(unit1)
            viewcone_max = float(value2) * u.Unit(unit2)
        except ValueError as exc:
            logger.error(f"Invalid viewcone  values: {value1} {unit1} {value2} {unit2}")
            raise exc
        return f"{viewcone_min.to('deg')} {viewcone_max.to('deg')}"

    @staticmethod
    def core_scatter(core_scatter):
        """
        Argument parser type for core scatter argument for multiple use of events.

        Arguments are given as string with two values separated by a space:

        - Number of uses of each event
        - Radius of scatter area

        Parameters
        ----------
        core_scatter: str
            core scatter string

        Returns
        -------
        string
            Core scatter string.
        """
        logger = logging.getLogger(__name__)

        parts = core_scatter.split()
        if len(parts) != 3:
            logger.error(
                "Core scatter argument must be given in the form "
                "'n_scatter radius (m)' (e.g., '0 1500 m')"
            )
            raise TypeError
        value1, value2, unit2 = parts
        try:
            n_scatter = int(value1)
            core_radius = float(value2) * u.Unit(unit2)
        except ValueError as exc:
            logger.error(f"Invalid core scatter argument: {value1} {value2} {unit2}")
            raise exc
        return f"{n_scatter} {core_radius.to('m')}"
