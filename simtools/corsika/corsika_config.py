"""CORSIKA configuration."""

import logging
from pathlib import Path

import astropy.units as u
import numpy as np

import simtools.utils.general as gen
from simtools.io_operations import io_handler

__all__ = [
    "CorsikaConfig",
    "InvalidCorsikaInputError",
]


class InvalidCorsikaInputError(Exception):
    """Exception for invalid corsika input."""


class CorsikaConfig:
    """
    Configuration for the CORSIKA air shower simulation software.

    Follows closely the CORSIKA definitions and output format (see CORSIKA manual).

    The configuration is set as a dict corresponding to the command line configuration groups
    (especially simulation_software, simulation_model, simulation_parameters).

    Parameters
    ----------
    array_model : ArrayModel
        Array model.
    label : str
        Instance label.
    args_dict : dict
        Configuration dictionary
        includes simulation_software, simulation_model, simulation_parameters groups)
    """

    def __init__(self, array_model, args_dict, label=None):
        """Initialize CorsikaConfig."""
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init CorsikaConfig")

        self.label = label
        self.primary = None
        self.zenith_angle = None
        self.azimuth_angle = None
        self._run_number = None
        self.args_dict = args_dict
        self.config_file_path = None

        self.io_handler = io_handler.IOHandler()
        self.array_model = array_model
        self._corsika_default_parameters = self._load_corsika_default_parameters_file()
        self.config = self.setup_configuration()
        self._is_file_updated = False

    def __repr__(self):
        """CorsikaConfig representation."""
        text = (
            f"<class {self.__class__.__name__}> "
            f"(site={self.array_model.site}, "
            f"layout={self.array_model.layout_name}, label={self.label})"
        )
        return text

    def _load_corsika_default_parameters_file(self):
        """
        Load CORSIKA parameters.

        TODO - will be replaced by a call to the CORSIKA configuration collection
        in the simtools database.

        Returns
        -------
        corsika_parameters: dict
            Dictionary with CORSIKA parameters.
        """
        corsika_parameters_file = self.io_handler.get_input_data_file(
            "parameters", "corsika_parameters.yml"
        )
        self._logger.debug(f"Reading CORSIKA default parameters from {corsika_parameters_file}")
        return gen.collect_data_from_file_or_dict(file_name=corsika_parameters_file, in_dict=None)

    def setup_configuration(self):
        """
        Set configuration parameters for CORSIKA and CorsikaConfig.

        Converted values to CORSIKA-consistent units.

        Returns
        -------
        dict
            Dictionary with CORSIKA parameters.
        """
        if self.args_dict is None:
            return None
        self._logger.debug("Setting CORSIKA parameters ")
        self._is_file_updated = False
        self.azimuth_angle = int(self.args_dict["azimuth_angle"].to("deg").value)
        self.zenith_angle = self.args_dict["zenith_angle"].to("deg").value

        # lists provides as strings
        e_range = self.args_dict["erange"].split(" ")
        view_cone = self.args_dict["viewcone"].split(" ")
        core_scatter = self.args_dict["core_scatter"].split(" ")

        return {
            "EVTNR": [self.args_dict["event_number_first_shower"]],
            "NSHOW": [self.args_dict["nshow"]],
            "PRMPAR": [
                self._convert_primary_input_and_store_primary_name(self.args_dict["primary"])
            ],
            "ESLOPE": [self.args_dict["eslope"]],
            "ERANGE": [
                float(e_range[0]) * u.Unit(e_range[1]).to("GeV"),
                float(e_range[2]) * u.Unit(e_range[3]).to("GeV"),
            ],
            "THETAP": [
                float(self.args_dict["zenith_angle"].to("deg").value),
                float(self.args_dict["zenith_angle"].to("deg").value),
            ],
            "PHIP": [
                self._rotate_azimuth_by_180deg(self.args_dict["azimuth_angle"].to("deg").value),
                self._rotate_azimuth_by_180deg(self.args_dict["azimuth_angle"].to("deg").value),
            ],
            "VIEWCONE": [
                float(view_cone[0]) * u.Unit(view_cone[1]).to("deg"),
                float(view_cone[2]) * u.Unit(view_cone[3]).to("deg"),
            ],
            "CSCAT": [
                int(core_scatter[0]),
                float(core_scatter[1]) * u.Unit(core_scatter[2]).to("cm"),
                0.0,
            ],
        }

    def _rotate_azimuth_by_180deg(self, az):
        """
        Convert azimuth angle to the CORSIKA coordinate system.

        Parameters
        ----------
        az: float
            Azimuth angle in degrees.

        Returns
        -------
        float
            Azimuth angle in degrees in the CORSIKA coordinate system.
        """
        phi = 180.0 - az
        phi = phi + 360.0 if phi < 0.0 else phi
        phi = phi - 360.0 if phi >= 360.0 else phi
        return phi

    def _convert_primary_input_and_store_primary_name(self, value):
        """
        Convert a primary name into the CORSIKA particle ID.

        Parameters
        ----------
        value: str
            Input primary name (e.g gamma, proton ...)

        Raises
        ------
        InvalidPrimary
            If the input name is not found.

        Returns
        -------
        int
            Respective number of the given primary.

        Notes
        -----
        TODO - this will be replaced using the 'particle' PDG package.
        """
        for prim_name, prim_info in self._corsika_default_parameters["PRIMARIES"].items():
            if value.upper() == prim_name or value.upper() in prim_info["names"]:
                self.primary = prim_name.lower()
                return prim_info["number"]
        msg = f"Primary not valid: {value}"
        self._logger.error(msg)
        raise InvalidCorsikaInputError(msg)

    def get_config_parameter(self, par_name):
        """
        Get value of CORSIKA configuration parameter.

        Parameters
        ----------
        par_name: str
            Name of the parameter as used in the CORSIKA input file (e.g. PRMPAR, THETAP ...)

        Raises
        ------
        KeyError
            When par_name is not a valid parameter name.

        Returns
        -------
        list
            Value(s) of the parameter.
        """
        try:
            par_value = self.config[par_name]
        except KeyError as exc:
            self._logger.error(f"Parameter {par_name} is not a CORSIKA config parameter")
            raise exc
        return par_value if len(par_value) > 1 else par_value[0]

    def print_config_parameter(self):
        """Print CORSIKA config parameters for inspection."""
        for par, value in self.config.items():
            print(f"{par} = {value}")

    @staticmethod
    def _get_text_single_line(pars, line_begin=""):
        """
        Return one parameter per line for each input parameter.

        Parameters
        ----------
        pars: dict
            Dictionary with the parameters to be written in the file.

        Returns
        -------
        str
            Text with the parameters.
        """
        text = ""
        for par, values in pars.items():
            line = line_begin + par + " "
            for v in values:
                line += str(v) + " "
            line += "\n"
            text += line
        return text

    def generate_corsika_input_file(self, use_multipipe=False):
        """
        Generate a CORSIKA input file.

        Parameters
        ----------
        use_multipipe: bool
            Whether to set the CORSIKA Inputs file to pipe
            the output directly to sim_telarray.

        """
        if self._is_file_updated:
            self._logger.debug(f"CORSIKA input file already updated: {self.config_file_path}")
            return self.config_file_path
        _output_generic_file_name = self.set_output_file_and_directory(use_multipipe=use_multipipe)
        self._logger.info(f"Exporting CORSIKA input file to {self.config_file_path}")

        with open(self.config_file_path, "w", encoding="utf-8") as file:
            file.write("\n* [ RUN PARAMETERS ]\n")
            text_parameters = self._get_text_single_line(self.config)
            file.write(text_parameters)

            file.write("\n* [ SITE PARAMETERS ]\n")
            text_site_parameters = self._get_text_single_line(
                self.array_model.site_model.get_corsika_site_parameters(config_file_style=True)
            )
            file.write(text_site_parameters)

            file.write("\n* [ IACT ENV PARAMETERS ]\n")
            file.write(f"IACT setenv PRMNAME {self.primary}\n")
            file.write(f"IACT setenv ZA {int(self.config['THETAP'][0])}\n")
            file.write(f"IACT setenv AZM {self.azimuth_angle}\n")

            file.write("\n* [ SEEDS ]\n")
            self._write_seeds(file)

            file.write("\n* [ TELESCOPES ]\n")
            telescope_list_text = self.get_corsika_telescope_list()
            file.write(telescope_list_text)

            file.write("\n* [ INTERACTION FLAGS ]\n")
            text_interaction_flags = self._get_text_single_line(
                self._corsika_default_parameters["INTERACTION_FLAGS"]
            )
            file.write(text_interaction_flags)

            file.write("\n* [ CHERENKOV EMISSION PARAMETERS ]\n")
            text_cherenkov = self._get_text_single_line(
                self._corsika_default_parameters["CHERENKOV_EMISSION_PARAMETERS"]
            )
            file.write(text_cherenkov)

            file.write("\n* [ DEBUGGING OUTPUT PARAMETERS ]\n")
            text_debugging = self._get_text_single_line(
                self._corsika_default_parameters["DEBUGGING_OUTPUT_PARAMETERS"]
            )
            file.write(text_debugging)

            file.write("\n* [ OUTPUT FILE ]\n")
            if use_multipipe:
                run_cta_script = Path(self.config_file_path.parent).joinpath("run_cta_multipipe")
                file.write(f"TELFIL |{str(run_cta_script)}\n")
            else:
                file.write(f"TELFIL {_output_generic_file_name}\n")

            file.write("\n* [ IACT TUNING PARAMETERS ]\n")
            text_iact = self._get_text_single_line(
                self._corsika_default_parameters["IACT_TUNING_PARAMETERS"],
                "IACT ",
            )
            file.write(text_iact)
            file.write("\nEXIT")

        self._is_file_updated = True
        return self.config_file_path

    def get_corsika_config_file_name(self, file_type, run_number=None):
        """
        Get a CORSIKA config style file name for various file types.

        TODO - overlap with runner_services.get_file_name

        Parameters
        ----------
        file_type: str
            The type of file (determines the file suffix).
            Choices are config_tmp, config or output_generic.
        run_number: int
            Run number.

        Returns
        -------
        str
            for file_type="config_tmp":
                Get the CORSIKA input file for one specific run.
                This is the input file after being pre-processed by sim_telarray (pfp).
            for file_type="config":
                Get a general CORSIKA config inputs file.
            for file_type="output_generic"
                Get a generic file name for the TELFIL option in the CORSIKA inputs file.
            for file_type="multipipe"
                Get a multipipe "file name" for the TELFIL option in the CORSIKA inputs file.

        Raises
        ------
        ValueError
            If file_type is unknown or if the run number is not given for file_type==config_tmp.
        """
        file_label = f"_{self.label}" if self.label is not None else ""
        view_cone = ""
        if self.config["VIEWCONE"][0] != 0 or self.config["VIEWCONE"][1] != 0:
            view_cone = (
                f"_cone{int(self.config['VIEWCONE'][0]):d}-" f"{int(self.config['VIEWCONE'][1]):d}"
            )
        file_name = (
            f"{self.primary}_{self.array_model.site}_{self.array_model.layout_name}_"
            f"za{int(self.config['THETAP'][0]):03}-"
            f"azm{self.azimuth_angle:03}deg"
            f"{view_cone}{file_label}"
        )
        if file_type == "config_tmp":
            if run_number is not None:
                return f"corsika_config_run{run_number:06}_{file_name}.txt"
            raise ValueError("Must provide a run number for a temporary CORSIKA config file")
        if file_type == "config":
            return f"corsika_config_{file_name}.input"
        if file_type == "output_generic":
            # The XXXXXX will be replaced by the run number after the pfp step with sed
            file_name = (
                f"runXXXXXX_"
                f"{self.primary}_za{int(self.config['THETAP'][0]):03}deg_"
                f"azm{self.azimuth_angle:03}deg"
                f"_{self.array_model.site}_{self.array_model.layout_name}{file_label}.zst"
            )
            return file_name
        if file_type == "multipipe":
            return f"multi_cta-{self.array_model.site}-{self.array_model.layout_name}.cfg"

        raise ValueError(f"The requested file type ({file_type}) is unknown")

    def set_output_file_and_directory(self, use_multipipe=False):
        """
        Set output file names and directories.

        Parameters
        ----------
        use_multipipe: bool
            Whether to set the CORSIKA Inputs file to pipe
            the output directly to sim_telarray. Defines directory names.

        Returns
        -------
        str
            Output file name.
        """
        sub_dir = "corsika_simtel" if use_multipipe else "corsika"
        config_file_name = self.get_corsika_config_file_name(file_type="config")
        file_directory = self.io_handler.get_output_directory(label=self.label, sub_dir=sub_dir)
        self._logger.debug(f"Creating directory {file_directory}")
        file_directory.mkdir(parents=True, exist_ok=True)
        self.config_file_path = file_directory.joinpath(config_file_name)

        return self.get_corsika_config_file_name(file_type="output_generic")

    def _write_seeds(self, file):
        """
        Generate and write seeds in the CORSIKA input file.

        Parameters
        ----------
        file: stream
            File where the telescope positions will be written.
        """
        random_seed = self.config["PRMPAR"][0] + self._run_number
        rng = np.random.default_rng(random_seed)
        corsika_seeds = [int(rng.uniform(0, 1e7)) for _ in range(4)]
        for s in corsika_seeds:
            file.write(f"SEED {s} 0 0\n")

    def get_corsika_telescope_list(self):
        """
        List of telescope positions in the format required for the CORSIKA input file.

        Returns
        -------
        str
            Piece of text to be added to the CORSIKA input file.
        """
        corsika_input_list = ""
        for telescope_name, telescope in self.array_model.telescope_model.items():
            positions = telescope.get_parameter_value_with_unit("array_element_position_ground")
            corsika_input_list += "TELESCOPE"
            for pos in positions:
                corsika_input_list += f"\t {pos.to('cm').value:.3f}"
            sphere_radius = telescope.get_parameter_value_with_unit("telescope_sphere_radius").to(
                "cm"
            )
            corsika_input_list += f"\t {sphere_radius:.3f}"
            corsika_input_list += f"\t # {telescope_name}\n"

        return corsika_input_list

    @property
    def run_number(self):
        """Set run number."""
        return self._run_number

    @run_number.setter
    def run_number(self, run_number):
        """
        Set run number and validate it.

        Parameters
        ----------
        run_number: int
            Run number.
        """
        self._run_number = self.validate_run_number(run_number)

    def validate_run_number(self, run_number):
        """
        Validate run number and return it.

        Return run number from configuration if None.

        Parameters
        ----------
        run_number: int
            Run number.

        Returns
        -------
        int
            Run number.

        Raises
        ------
        ValueError
            If run_number is not a valid value (e.g., < 1).
        """
        if run_number is None:
            return self.run_number
        if not float(run_number).is_integer() or run_number < 1 or run_number > 999999:
            msg = f"Invalid type of run number ({run_number}) - it must be an uint < 1000000."
            self._logger.error(msg)
            raise ValueError(msg)
        return run_number
