"""Command line parser for applications."""

import argparse
import logging
import re
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
        simulation_configuration=None,
        db_config=False,
    ):
        """
        Initialize default arguments used by all applications (e.g., log level or test flag).

        Parameters
        ----------
        paths: bool
            Add path configuration to list of args.
        output: bool
            Add output file configuration to list of args.
        simulation_model: list
            List of simulation model configuration parameters to add to list of args
            (use: 'model_version', 'telescope', 'site')
        simulation_configuration: dict
            Dict of simulation software configuration parameters to add to list of args.
        db_config: bool
            Add database configuration parameters to list of args.
        """
        self.initialize_simulation_model_arguments(simulation_model)
        self.initialize_simulation_configuration_arguments(simulation_configuration)
        if db_config:
            self.initialize_db_config_arguments()
        if paths:
            self.initialize_path_arguments()
        if output:
            self.initialize_output_arguments()
        self.initialize_config_files()
        self.initialize_application_execution_arguments()
        self.initialize_user_arguments()

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

    def initialize_user_arguments(self):
        """Initialize user arguments."""
        _job_group = self.add_argument_group("user")
        _job_group.add_argument(
            "--user_name",
            help="user name",
            type=str,
            required=False,
        )
        _job_group.add_argument(
            "--user_organization",
            help="user organization",
            type=str,
            required=False,
        )
        _job_group.add_argument(
            "--user_email",
            help="user email",
            type=str,
            required=False,
        )
        _job_group.add_argument(
            "--user_orcid",
            help="user ORCID",
            type=str,
            required=False,
        )

    def initialize_db_config_arguments(self):
        """Initialize DB configuration parameters."""
        _job_group = self.add_argument_group("database configuration")
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
            default=None,
        )
        _job_group.add_argument(
            "--db_simulation_model",
            help="name of simulation model database",
            type=str.strip,
            required=False,
            default=None,
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
        if "model_version" in model_options:
            _job_group.add_argument(
                "--model_version",
                help="production model version",
                type=str,
                default=None,
                nargs="+",
            )
        if "parameter_version" in model_options:
            _job_group.add_argument(
                "--parameter_version",
                help="model parameter version",
                type=str,
                default=None,
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
            _job_group.add_argument(
                "--telescope_model_file",
                help=(
                    "Path to a YAML file containing modifications to the telescope model. "
                    "This feature is intended for developers and lacks validation."
                ),
                type=Path,
                required=False,
            )
        if "layout" in model_options or "layout_file" in model_options:
            _job_group = self._add_model_option_layout(
                job_group=_job_group,
                model_options=model_options,
                # layout info is always required for layout related tasks with the exception
                # of listing the available layouts in the DB
                required="--list_available_layouts" not in self._option_string_actions,
            )

    def initialize_simulation_configuration_arguments(self, simulation_configuration):
        """
        Initialize default arguments for simulation configuration and simulation software.

        Parameters
        ----------
        simulation_configuration: dict
            Dict of simulation software configuration parameters.
        """
        if simulation_configuration is None:
            return

        if "software" in simulation_configuration:
            self._initialize_simulation_software()
        if "corsika_configuration" in simulation_configuration:
            self._initialize_simulation_configuration(
                group_name="simulation configuration",
                selected_parameters=simulation_configuration["corsika_configuration"],
                available_parameters=self._get_dictionary_with_corsika_configuration(),
            )
            self._initialize_simulation_configuration(
                group_name="shower parameters",
                selected_parameters=simulation_configuration["corsika_configuration"],
                available_parameters=self._get_dictionary_with_shower_configuration(),
            )
        if "sim_telarray_configuration" in simulation_configuration:
            self._initialize_simulation_configuration(
                group_name="sim_telarray configuration",
                selected_parameters=simulation_configuration["sim_telarray_configuration"],
                available_parameters=self._get_dictionary_with_sim_telarray_configuration(),
            )

    def _initialize_simulation_software(self):
        """Initialize simulation software arguments."""
        _software_group = self.add_argument_group("simulation software")
        _software_group.add_argument(
            "--simulation_software",
            help="Simulation software steps.",
            type=str,
            choices=["corsika", "sim_telarray", "corsika_sim_telarray"],
            required=True,
            default="corsika_sim_telarray",
        )

    @staticmethod
    def _get_dictionary_with_corsika_configuration():
        """Return dictionary with CORSIKA configuration parameters."""
        from simtools.corsika.primary_particle import PrimaryParticle  # pylint: disable=C0415

        return {
            "primary": {
                "help": (
                    "Primary particle to simulate. "
                    "(choices for common names: "
                    f"{', '.join(PrimaryParticle.particle_names().keys())}; "
                    "use '--primary_ID_type' to use other particle ID types)."
                ),
                "type": str.lower,
                "required": True,
            },
            "primary_id_type": {
                "help": "Primary particle ID type",
                "type": str,
                "required": False,
                "choices": ["common_name", "corsika7_id", "pdg_id"],
                "default": "common_name",
            },
            "azimuth_angle": {
                "help": (
                    "Telescope pointing direction in azimuth. "
                    "It can be in degrees between 0 and 360 or one of north, south, east or west. "
                    "North is 0 degrees and the azimuth grows clockwise (East is 90 degrees)."
                ),
                "type": CommandLineParser.azimuth_angle,
                "required": True,
                "default": 0 * u.deg,
            },
            "zenith_angle": {
                "help": "Zenith angle in degrees (between 0 and 180).",
                "type": CommandLineParser.zenith_angle,
                "required": True,
                "default": 20 * u.deg,
            },
            "nshow": {
                "help": "Number of showers per run to simulate.",
                "type": int,
                "required": False,
            },
            "run_number_offset": {
                "help": "An offset for the run number to be simulated.",
                "type": int,
                "required": False,
                "default": 0,
            },
            "run_number": {
                "help": "Run number to be simulated.",
                "type": int,
                "required": True,
                "default": 1,
            },
            "number_of_runs": {
                "help": "Number of runs to be simulated.",
                "type": int,
                "required": True,
                "default": 1,
            },
            "event_number_first_shower": {
                "help": "Event number of first shower",
                "type": int,
                "required": False,
                "default": 1,
            },
            "correct_for_b_field_alignment": {
                "help": "Correct for B-field alignment",
                "action": "store_true",
                "required": False,
                "default": True,
            },
        }

    @staticmethod
    def _get_dictionary_with_shower_configuration():
        """Return dictionary with shower configuration parameters."""
        return {
            "eslope": {
                "help": "Slope of the energy spectrum.",
                "type": float,
                "required": False,
                "default": -2.0,
            },
            "energy_range": {
                "help": (
                    "Energy range of the primary particle (min/max value, e'g', '10 GeV 5 TeV')."
                ),
                "type": CommandLineParser.parse_quantity_pair,
                "required": False,
                "default": ["3 GeV 330 TeV"],
            },
            "view_cone": {
                "help": (
                    "View cone radius for primary arrival directions "
                    "(min/max value, e.g. '0 deg 10 deg')."
                ),
                "type": CommandLineParser.parse_quantity_pair,
                "required": False,
                "default": ["0 deg 0 deg"],
            },
            "core_scatter": {
                "help": "Scatter radius for shower cores (number of use; scatter radius).",
                "type": CommandLineParser.parse_integer_and_quantity,
                "required": False,
                "default": ["10 1400 m"],
            },
        }

    @staticmethod
    def _get_dictionary_with_sim_telarray_configuration():
        """Return dictionary with sim_telarray configuration parameters."""
        return {
            "sim_telarray_instrument_seeds": {
                "help": (
                    "Random seed used for sim_telarray instrument setup. "
                    "If '--sim_telarray_random_instrument_instances' is not set: "
                    "use as sim_telarray seed ('random_seed' parameter). Otherwise: "
                    "use as base seed to generate the random instrument instance seeds."
                ),
                "type": str,
                "required": False,
            },
            "sim_telarray_random_instrument_instances": {
                "help": "Number of random instrument instances initialized in sim_telarray.",
                "type": int,
                "required": False,
            },
        }

    def _initialize_simulation_configuration(
        self, group_name, selected_parameters, available_parameters
    ):
        """
        Initialize simulation configuration arguments.

        Parameters
        ----------
        group_name : str
            Name of the group of arguments.
        selected_parameters : list
            List of selected parameters to be added to the group.
        available_parameters : dict
            Dictionary with available parameters and their configuration.
        """
        configuration_group = self.add_argument_group(group_name)

        if "all" in selected_parameters:
            selected_parameters = available_parameters.keys()

        for param in selected_parameters:
            try:
                configuration_group.add_argument(f"--{param}", **available_parameters[param])
            except KeyError:
                pass

    @staticmethod
    def _add_model_option_layout(job_group, model_options, required=True):
        """
        Add layout option to the job group.

        Parameters
        ----------
        job_group: argparse.ArgumentParser
            Job group
        model_options: list
            List of model options.

        Returns
        -------
        argparse.ArgumentParser
        """
        _layout_group = job_group.add_mutually_exclusive_group(required=required)
        _layout_group.add_argument(
            "--array_layout_name",
            help="array layout name(s) (e.g., alpha, subsystem_msts)",
            nargs="+",
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
        if "layout_file" in model_options:
            _layout_group.add_argument(
                "--array_layout_file",
                help="file(s) with the list of array elements (astropy table format).",
                nargs="+",
                type=str,
                required=False,
                default=None,
            )
        if "layout_parameter_file" in model_options:
            _layout_group.add_argument(
                "--array_layout_parameter_file",
                help="Array layout model parameter file (typically in JSON format).",
                type=str,
                required=False,
                default=None,
            )
        if "plot_all_layouts" in model_options:
            _layout_group.add_argument(
                "--plot_all_layouts",
                help="plot all available layouts",
                action="store_true",
                required=False,
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
        names.validate_array_element_name(str(value))
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
        supports such angles!.

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
        except TypeError as exc:
            logger.error("The azimuth angle provided is not a valid numerical or string value.")
            raise exc
        try:
            return u.Quantity(angle).to("deg")
        except TypeError:
            logger.debug(
                "The azimuth angle provided is not a valid astropy.Quantity. "
                "Will check if it is (north, south, east, west) instead"
            )
        azimuth_map = {
            "north": 0 * u.deg,
            "south": 180 * u.deg,
            "east": 90 * u.deg,
            "west": 270 * u.deg,
        }
        azimuth_angle = angle.lower()
        if azimuth_angle in azimuth_map:
            return azimuth_map[azimuth_angle]
        raise argparse.ArgumentTypeError(
            "The azimuth angle given as string can only be one of "
            f"(north, south, east, west), not {angle}. Otherwise use numerical values."
        )

    @staticmethod
    def parse_quantity_pair(string):
        """
        Parse a string representing a pair of astropy quantities.

        Returns
        -------
        tuple
            A tuple of two astropy.units.Quantity objects.

        Raises
        ------
        ValueError
            If the string cannot be parsed into exactly two quantities.
        """
        matches = re.findall(r"[\d\.eE+-]+\s*[A-Za-z]+", string)
        if len(matches) != 2:
            tokens = string.strip().split()
            if len(tokens) == 2:
                matches = tokens
            elif len(tokens) == 3:
                matches = [tokens[0], " ".join(tokens[1:])]
            elif len(tokens) == 4:
                matches = [" ".join(tokens[:2]), " ".join(tokens[2:])]
            else:
                raise ValueError("Input string does not contain exactly two quantities.")

        try:
            return tuple(u.Quantity(m) for m in matches)
        except Exception as exc:
            raise ValueError(f"Could not parse quantities: {exc}") from exc

    @staticmethod
    def parse_integer_and_quantity(input_string):
        """
        Parse a string representing an integer and a quantity with units.

        This is e.g., used for the 'core_scatter' argument.

        Parameters
        ----------
        input_string: str
            The input string (e.g., "5 1500 m") or
            a tuple converted to string (e.g., "(5, <Quantity 1500 m>)").

        Returns
        -------
        tuple: A tuple containing an integer and an astropy.units.Quantity object.

        Raises
        ------
        ValueError: If the input string does not match the required format.
        """
        # tuple converted to string: "(5, <Quantity 1500 m>)"
        if all(char in input_string for char in ["(", ")", ","]):
            pattern = r"\((\d+), <Quantity ([\d.]+) (.+)>\)"
            match = re.match(pattern, input_string)
        # string with integer and quantity: "5 1500 m"
        else:
            pattern = r"(\d+)\s+(\d+\.?\d*)\s*([a-zA-Z]+)"
            match = re.match(pattern, input_string.strip())

        if not match:
            raise ValueError("Input string does not contain an integer and a astropy quantity.")

        return (int(match.group(1)), u.Quantity(float(match.group(2)), match.group(3)))
