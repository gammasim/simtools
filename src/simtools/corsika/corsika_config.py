"""CORSIKA configuration."""

import logging
from collections.abc import Mapping

import numpy as np
from astropy import units as u

from simtools import settings
from simtools.corsika.primary_particle import PrimaryParticle
from simtools.io import io_handler
from simtools.model.model_parameter import ModelParameter
from simtools.sim_events import file_info
from simtools.utils import general as gen


class CorsikaConfig:
    """
    Configuration for the CORSIKA air shower simulation software.

    Follows closely the CORSIKA definitions and output format (see CORSIKA manual).

    The configuration is set as a dict corresponding to the command line configuration groups
    (especially simulation_software, simulation configuration, simulation parameters).

    Parameters
    ----------
    array_model : ArrayModel
        Array model.
    run_number : int
        Run number.
    label : str
        Instance label.
    """

    def __init__(self, array_model, run_number, label=None):
        """Initialize CorsikaConfig."""
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init CorsikaConfig")

        self.label = label
        self.shower_events = self.mc_events = None
        self.zenith_angle = self.azimuth_angle = None
        self.curved_atmosphere_min_zenith_angle = None
        self.run_number = run_number
        self.primary_particle = settings.config.args  # see setter for primary_particle
        self.use_curved_atmosphere = settings.config.args  # see setter for use_curved_atmosphere
        self.run_mode = settings.config.args.get("run_mode")

        self.io_handler = io_handler.IOHandler()
        self.array_model = array_model
        self.config = self._fill_corsika_configuration(settings.config.args)
        self._initialize_from_config(settings.config.args)

    @property
    def primary_particle(self):
        """Primary particle."""
        return self._primary_particle

    @primary_particle.setter
    def primary_particle(self, args):
        """
        Set primary particle from input dictionary or CORSIKA 7 particle ID.

        This is to make sure that when setting the primary particle,
        we get the full PrimaryParticle object expected.

        Parameters
        ----------
        args: dict, corsika particle ID, or None
            Configuration dictionary
        """
        if (
            isinstance(args, Mapping)  # dict-like (includes mappingproxy)
            and args.get("primary_id_type") is not None
            and args.get("primary") is not None
        ):
            self._primary_particle = PrimaryParticle(
                particle_id_type=args.get("primary_id_type"), particle_id=args.get("primary")
            )
        elif isinstance(args, int):
            self._primary_particle = PrimaryParticle(
                particle_id_type="corsika7_id", particle_id=args
            )
        else:
            self._primary_particle = PrimaryParticle()

    @property
    def use_curved_atmosphere(self):
        """Check if zenith angle condition for curved atmosphere usage for CORSIKA is met."""
        return self._use_curved_atmosphere

    @use_curved_atmosphere.setter
    def use_curved_atmosphere(self, args):
        """Check if zenith angle condition for curved atmosphere usage for CORSIKA is met."""
        self._use_curved_atmosphere = False
        if isinstance(args, bool):
            self._use_curved_atmosphere = args
        elif isinstance(args, Mapping):  # dict-like (includes mappingproxy)
            try:
                self._use_curved_atmosphere = (
                    args.get("zenith_angle", 0.0 * u.deg).to("deg").value
                    > args["curved_atmosphere_min_zenith_angle"].to("deg").value
                )
            except KeyError:
                self._use_curved_atmosphere = False

    def _fill_corsika_configuration(self, args):
        """
        Fill CORSIKA configuration.

        Dictionary keys are CORSIKA parameter names. Values are converted to
        CORSIKA-consistent units.

        Parameters
        ----------
        args: dict
            Configuration dictionary.

        Returns
        -------
        dict
            Dictionary with CORSIKA parameters.
        """
        if args is None:
            return {}

        config = {}
        config["RUNNR"] = [self.run_number]
        config["USER"] = [settings.config.user]
        config["HOST"] = [settings.config.hostname]
        if self.is_calibration_run():
            config["USER_INPUT"] = self._corsika_configuration_for_dummy_simulations(args)
        elif args.get("corsika_file", None) is not None:
            config["USER_INPUT"] = self._corsika_configuration_from_corsika_file(
                args["corsika_file"]
            )
        else:
            config["USER_INPUT"] = self._corsika_configuration_from_user_input(args)

        config.update(
            self._fill_corsika_configuration_from_db(gen.ensure_iterable(args.get("model_version")))
        )
        return config

    def _fill_corsika_configuration_from_db(self, model_versions):
        """Fill CORSIKA configuration from database."""
        config = {}
        # all following parameters require DB
        if settings.config.db_config is None or not model_versions:
            return config

        # For multiple model versions, check that CORSIKA parameters are identical
        self.assert_corsika_configurations_match(model_versions)
        model_version = model_versions[0]

        self._logger.debug(f"Using model version {model_version} for CORSIKA parameters from DB")
        db_model_parameters = ModelParameter(model_version=model_version)
        parameters_from_db = db_model_parameters.get_simulation_software_parameters("corsika")

        config["INTERACTION_FLAGS"] = self._corsika_configuration_interaction_flags(
            parameters_from_db
        )
        config["CHERENKOV_EMISSION_PARAMETERS"] = self._corsika_configuration_cherenkov_parameters(
            parameters_from_db
        )
        config["DEBUGGING_OUTPUT_PARAMETERS"] = self._corsika_configuration_debugging_parameters()
        config["IACT_PARAMETERS"] = self._corsika_configuration_iact_parameters(parameters_from_db)
        return config

    def _initialize_from_config(self, args):
        """
        Initialize additional parameters either from command line args or from derived config.

        Takes into account that in the case of a given CORSIKA input file, some parameters are read
        from the file instead of the command line args.

        Parameters
        ----------
        args: dict
            Command line arguments.
        """
        self.primary_particle = int(self.config.get("USER_INPUT", {}).get("PRMPAR", [1])[0])
        self.shower_events = int(self.config.get("USER_INPUT", {}).get("NSHOW", [0])[0])
        self.mc_events = int(
            self.shower_events * self.config.get("USER_INPUT", {}).get("CSCAT", [1])[0]
        )

        if args.get("corsika_file", None) is not None:
            azimuth = self._rotate_azimuth_by_180deg(
                0.5 * (self.config["USER_INPUT"]["PHIP"][0] + self.config["USER_INPUT"]["PHIP"][1]),
                invert_operation=True,
            )
            zenith = 0.5 * (
                self.config["USER_INPUT"]["THETAP"][0] + self.config["USER_INPUT"]["THETAP"][1]
            )
        else:
            azimuth = args.get("azimuth_angle", 0.0 * u.deg).to("deg").value
            zenith = args.get("zenith_angle", 20.0 * u.deg).to("deg").value

        self.azimuth_angle = round(azimuth)
        self.zenith_angle = round(zenith)

        self.curved_atmosphere_min_zenith_angle = (
            args.get("curved_atmosphere_min_zenith_angle", 90.0 * u.deg).to("deg").value
        )

    def assert_corsika_configurations_match(self, model_versions):
        """
        Assert that CORSIKA configurations match across all model versions.

        Parameters
        ----------
        model_versions : list
            List of model versions to check.

        Raises
        ------
        InvalidCorsikaInputError
            If CORSIKA parameters are not identical across all model versions.
        """
        if len(model_versions) < 2:
            return

        parameters_from_db_list = []

        # Get parameters for all model versions
        for model_version in model_versions:
            db_model_parameters = ModelParameter(model_version=model_version)
            parameters_from_db_list.append(
                db_model_parameters.get_simulation_software_parameters("corsika")
            )

        # Parameters that can differ between model versions (e.g., i/o buffer size)
        skip_parameters = ["corsika_iact_io_buffer", "corsika_iact_split_auto"]

        # Check if all parameters match
        for i in range(len(parameters_from_db_list) - 1):
            for key in parameters_from_db_list[i]:
                if key in skip_parameters:
                    continue

                current_value = parameters_from_db_list[i][key]["value"]
                next_value = parameters_from_db_list[i + 1][key]["value"]

                if current_value != next_value:
                    self._logger.warning(
                        f"Parameter '{key}' mismatch between model versions:\n"
                        f"  {model_versions[i]}: {current_value}\n"
                        f"  {model_versions[i + 1]}: {next_value}"
                    )
                    raise ValueError(
                        f"CORSIKA parameter '{key}' differs between model versions "
                        f"{model_versions[i]} and {model_versions[i + 1]}. "
                        f"Values are {current_value} and {next_value} respectively."
                    )

    def _corsika_configuration_for_dummy_simulations(self, args_dict):
        """
        Return CORSIKA configuration for dummy simulations.

        Settings are such that that the simulations are fast
        and none (or not many) Cherenkov photons are generated.
        This is e.g. used for some calibration run modes in sim_telarray.

        Returns
        -------
        dict
            Dictionary with CORSIKA parameters for dummy simulations.
        """
        theta, phi = self._get_corsika_theta_phi(args_dict)
        self._logger.info("Using CORSIKA configuration for dummy simulations.")
        return {
            "EVTNR": [1],
            "NSHOW": [1],
            "PRMPAR": [1],  # CORSIKA ID 1 for primary gamma
            "ESLOPE": [-2.0],
            "ERANGE": [0.1, 0.1],
            "THETAP": [theta, theta],
            "PHIP": [phi, phi],
            "VIEWCONE": [0.0, 0.0],
            "CSCAT": [1, 0.0, 10.0],
        }

    def _corsika_configuration_from_corsika_file(self, corsika_input_file):
        """
        Get CORSIKA configuration run header of provided input files.

        Reads configuration from the run and event headers from the CORSIKA input file
        (unfortunately quite fine tuned to the pycorsikaio run and event
        header implementation).

        Parameters
        ----------
        corsika_input_file : str, path
            Path to the CORSIKA input file.

        Returns
        -------
        dict
            Dictionary with CORSIKA parameters from input file.
        """
        run_header, event_header = file_info.get_corsika_run_and_event_headers(corsika_input_file)
        self._logger.debug(f"CORSIKA run header from {corsika_input_file}")

        def to_float32(value):
            """Convert value to numpy float32."""
            return np.float32(value) if value is not None else 0.0

        def to_int32(value):
            """Convert value to numpy int32."""
            return np.int32(value) if value is not None else 0

        if run_header["n_observation_levels"] > 0:
            self._check_altitude_and_site(run_header["observation_height"][0])

        return {
            "EVTNR": [to_int32(event_header["event_number"])],
            "NSHOW": [to_int32(run_header["n_showers"])],
            "PRMPAR": [to_int32(event_header["particle_id"])],
            "ESLOPE": [to_float32(run_header["energy_spectrum_slope"])],
            "ERANGE": [to_float32(run_header["energy_min"]), to_float32(run_header["energy_max"])],
            "THETAP": [
                to_float32(event_header["theta_min"]),
                to_float32(event_header["theta_max"]),
            ],
            "PHIP": [to_float32(event_header["phi_min"]), to_float32(event_header["phi_max"])],
            "VIEWCONE": [
                to_float32(event_header["viewcone_inner_angle"]),
                to_float32(event_header["viewcone_outer_angle"]),
            ],
            "CSCAT": [
                to_int32(event_header["n_reuse"]),
                to_float32(event_header["reuse_x"]),
                to_float32(event_header["reuse_y"]),
            ],
        }

    def _corsika_configuration_from_user_input(self, args_dict):
        """
        Get CORSIKA configuration from user input.

        Parameters
        ----------
        args_dict : dict
            Configuration dictionary.

        Returns
        -------
        dict
            Dictionary with CORSIKA parameters.
        """
        theta, phi = self._get_corsika_theta_phi(args_dict)
        return {
            "EVTNR": [args_dict["event_number_first_shower"]],
            "NSHOW": [args_dict["nshow"]],
            "PRMPAR": [self.primary_particle.corsika7_id],
            "ESLOPE": [args_dict["eslope"]],
            "ERANGE": [
                args_dict["energy_range"][0].to("GeV").value,
                args_dict["energy_range"][1].to("GeV").value,
            ],
            "THETAP": [theta, theta],
            "PHIP": [phi, phi],
            "VIEWCONE": [
                args_dict["view_cone"][0].to("deg").value,
                args_dict["view_cone"][1].to("deg").value,
            ],
            "CSCAT": [
                args_dict["core_scatter"][0],
                args_dict["core_scatter"][1].to("cm").value,
                0.0,
            ],
        }

    def _check_altitude_and_site(self, observation_height):
        """Check that observation height from CORSIKA file matches site model."""
        site_altitude = self.array_model.site_model.get_parameter_value("corsika_observation_level")
        if not np.isclose(observation_height / 1.0e2, site_altitude, atol=1.0):
            raise ValueError(
                "Observatory altitude does not match CORSIKA file observation height: "
                f"{site_altitude} m (site model) != {observation_height / 1.0e2} m (CORSIKA file)"
            )

    def _get_corsika_theta_phi(self, args_dict):
        """Get CORSIKA theta and phi angles from args_dict."""
        theta = args_dict.get("zenith_angle", 20.0 * u.deg).to("deg").value
        phi = self._rotate_azimuth_by_180deg(
            args_dict.get("azimuth_angle", 0.0 * u.deg).to("deg").value,
            correct_for_geomagnetic_field_alignment=args_dict.get(
                "correct_for_b_field_alignment", True
            ),
        )
        return theta, phi

    def _corsika_configuration_interaction_flags(self, parameters_from_db):
        """
        Return CORSIKA interaction flags / parameters.

        Parameters
        ----------
        parameters_from_db : dict
            CORSIKA parameters from the database.

        Returns
        -------
        interaction_parameters : dict
            Dictionary with CORSIKA interaction parameters.
        """
        parameters = {}
        parameters["FIXHEI"] = self._input_config_first_interaction_height(
            parameters_from_db["corsika_first_interaction_height"]
        )
        parameters["FIXCHI"] = [
            self._input_config_corsika_starting_grammage(
                parameters_from_db["corsika_starting_grammage"]
            )
        ]
        if not self.use_curved_atmosphere:
            parameters["TSTART"] = ["T"]
        parameters["ECUTS"] = self._input_config_corsika_particle_kinetic_energy_cutoff(
            parameters_from_db["corsika_particle_kinetic_energy_cutoff"]
        )
        parameters["MUADDI"] = ["F"]
        parameters["MUMULT"] = ["T"]
        parameters["LONGI"] = self._input_config_corsika_longitudinal_parameters(
            parameters_from_db["corsika_longitudinal_shower_development"]
        )
        parameters["MAXPRT"] = ["10"]
        parameters["ECTMAP"] = ["1.e6"]

        self._logger.debug(f"Interaction parameters: {parameters}")
        return parameters

    def _input_config_first_interaction_height(self, entry):
        """Return FIXHEI parameter CORSIKA format."""
        return [f"{entry['value'] * u.Unit(entry['unit']).to('cm'):.2f}", "0"]

    def _input_config_corsika_starting_grammage(self, entry):
        """Return FIXCHI parameter CORSIKA format."""
        value = self._get_starting_grammage_value(entry["value"])
        return f"{value * u.Unit(entry['unit']).to('g/cm2')}"

    def _get_starting_grammage_value(self, value_entry):
        """
        Get appropriate starting grammage value from entry values.

        Parameters
        ----------
        value_entry : float or list
            Value or list of grammage configurations

        Returns
        -------
        float
            Selected grammage value
        """
        if not isinstance(value_entry, list):
            return value_entry

        tel_types = {tel.design_model for tel in self.array_model.telescope_model.values()}
        particle = self.primary_particle.name
        matched_values = self._get_matching_grammage_values(value_entry, tel_types, particle)

        return min(matched_values) if matched_values else 0

    def _get_matching_grammage_values(self, configs, tel_types, particle):
        """Get list of matching grammage values for particle and telescope types."""
        matched = []
        defaults = []

        for config in configs:
            if config.get("instrument") is None or config.get("instrument") in tel_types:
                if config["primary_particle"] == particle:
                    matched.append(config["value"])
                elif config["primary_particle"] == "default":
                    defaults.append(config["value"])

        return matched if matched else defaults

    def _input_config_corsika_particle_kinetic_energy_cutoff(self, entry):
        """Return ECUTS parameter CORSIKA format."""
        e_cuts = entry["value"]
        return [
            f"{e_cuts[0] * u.Unit(entry['unit']).to('GeV')} "
            f"{e_cuts[1] * u.Unit(entry['unit']).to('GeV')} "
            f"{e_cuts[2] * u.Unit(entry['unit']).to('GeV')} "
            f"{e_cuts[3] * u.Unit(entry['unit']).to('GeV')}"
        ]

    def _input_config_corsika_longitudinal_parameters(self, entry):
        """Return LONGI parameter CORSIKA format."""
        return ["T", f"{entry['value'] * u.Unit(entry['unit']).to('g/cm2')}", "F", "F"]

    def _corsika_configuration_cherenkov_parameters(self, parameters_from_db):
        """
        Return CORSIKA Cherenkov emission parameters.

        Parameters
        ----------
        parameters_from_db : dict
            CORSIKA parameters from the database.

        Returns
        -------
        dict
            Dictionary with CORSIKA Cherenkov emission parameters.
        """
        parameters = {}
        parameters["CERSIZ"] = [parameters_from_db["corsika_cherenkov_photon_bunch_size"]["value"]]
        parameters["CERFIL"] = "0"
        parameters["CWAVLG"] = self._input_config_corsika_cherenkov_wavelength(
            parameters_from_db["corsika_cherenkov_photon_wavelength_range"]
        )
        self._logger.debug(f"Cherenkov parameters: {parameters}")
        return parameters

    def _input_config_corsika_cherenkov_wavelength(self, entry):
        """Return CWAVLG parameter CORSIKA format."""
        wavelength_range = entry["value"]
        return [
            f"{wavelength_range[0] * u.Unit(entry['unit']).to('nm')}",
            f"{wavelength_range[1] * u.Unit(entry['unit']).to('nm')}",
        ]

    def _corsika_configuration_iact_parameters(self, parameters_from_db):
        """
        Return CORSIKA IACT parameters.

        Parameters
        ----------
        parameters_from_db : dict
            CORSIKA parameters from the database.

        Returns
        -------
        dict
            Dictionary with CORSIKA IACT parameters.
        """
        parameters = {}
        parameters["SPLIT_AUTO"] = [parameters_from_db["corsika_iact_split_auto"]["value"]]
        parameters["IO_BUFFER"] = [
            self._input_config_io_buff(parameters_from_db["corsika_iact_io_buffer"])
        ]
        parameters["MAX_BUNCHES"] = [parameters_from_db["corsika_iact_max_bunches"]["value"]]
        self._logger.debug(f"IACT parameters: {parameters}")
        return parameters

    def _corsika_configuration_debugging_parameters(self):
        """Return CORSIKA debugging output parameters."""
        return {
            "DEBUG": ["F", 6, "F", 1000000],
            "DATBAS": ["yes"],
            "DIRECT": ["./"],
            "PAROUT": ["F", "F"],
        }

    def _input_config_io_buff(self, entry):
        """Return IO_BUFFER parameter CORSIKA format (Byte or MB required)."""
        value = entry["value"] * u.Unit(entry["unit"]).to("Mbyte")
        # check if value is integer-like
        if value.is_integer():
            return f"{int(value)}MB"
        return f"{int(entry['value'] * u.Unit(entry['unit']).to('byte'))}"

    def _rotate_azimuth_by_180deg(
        self, az, correct_for_geomagnetic_field_alignment=True, invert_operation=False
    ):
        """
        Convert azimuth angle to the CORSIKA coordinate system.

        Corresponds to a rotation by 180 degrees, and optionally a correction for the
        for the differences between the geographic and geomagnetic north pole.

        Parameters
        ----------
        az: float
            Azimuth angle in degrees.
        correct_for_geomagnetic_field_alignment: bool
            Whether to correct for the geomagnetic field alignment.
        invert_operation: bool
            Whether to invert the operation (i.e., convert from CORSIKA to geographic system).

        Returns
        -------
        float
            Azimuth angle in degrees in the CORSIKA coordinate system.
        """
        b_field_declination = 0
        if correct_for_geomagnetic_field_alignment:
            b_field_declination = self.array_model.site_model.get_parameter_value("geomag_rotation")
        if invert_operation:
            return (az - 180 - b_field_declination) % 360
        return (az + 180 + b_field_declination) % 360

    def get_config_parameter(self, par_name):
        """
        Get value of CORSIKA configuration parameter.

        Parameters
        ----------
        par_name: str
            Name of the parameter as used in the CORSIKA input file (e.g. PRMPAR, THETAP ...).

        Raises
        ------
        KeyError
            When par_name is not a valid parameter name.

        Returns
        -------
        list
            Value(s) of the parameter.
        """
        par_value = []
        for values in self.config.values():
            if par_name in values:
                par_value = values[par_name]
        if len(par_value) == 0:
            raise KeyError(f"Parameter {par_name} is not a CORSIKA config parameter")
        return par_value if len(par_value) > 1 else par_value[0]

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

    def generate_corsika_input_file(
        self, use_multipipe, corsika_seeds, input_file, output_file, corsika_path=None
    ):
        """
        Generate a CORSIKA input file.

        Parameters
        ----------
        use_multipipe: bool
            Whether to set the CORSIKA Inputs file to pipe
            the output directly to sim_telarray.
        corsika_seeds: list
            List of fixed seeds used for CORSIKA random number generators.
        input_file: Path
            Path to the input file to be generated.
        output_file: Path
            Path to the output file to be generated.
        """
        self._logger.info(f"Exporting CORSIKA input file to {input_file}")

        with open(input_file, "w", encoding="utf-8") as file:
            file.write("\n* [ RUN PARAMETERS ]\n")
            text_parameters = self._get_text_single_line(self.config["USER_INPUT"])
            file.write(text_parameters)

            file.write("\n* [ SITE PARAMETERS ]\n")
            text_site_parameters = self._get_text_single_line(
                self.array_model.site_model.get_corsika_site_parameters(
                    config_file_style=True, model_directory=self.array_model.get_config_directory()
                )
            )
            file.write(text_site_parameters)

            file.write("\n* [ IACT ENV PARAMETERS ]\n")
            file.write(f"IACT setenv PRMNAME {self.primary_particle.name}\n")
            file.write(f"IACT setenv ZA {int(self.get_config_parameter('THETAP')[0])}\n")
            file.write(f"IACT setenv AZM {self.azimuth_angle}\n")

            file.write("\n* [ SEEDS ]\n")
            self._write_seeds(file, corsika_seeds)

            file.write("\n* [ TELESCOPES ]\n")
            telescope_list_text = self.get_corsika_telescope_list()
            file.write(telescope_list_text)

            file.write("\n* [ INTERACTION FLAGS ]\n")
            text_interaction_flags = self._get_text_single_line(self.config["INTERACTION_FLAGS"])
            file.write(text_interaction_flags)
            if corsika_path is not None:
                file.write(f"DATDIR {corsika_path!s}\n")

            file.write("\n* [ CHERENKOV EMISSION PARAMETERS ]\n")
            text_cherenkov = self._get_text_single_line(
                self.config["CHERENKOV_EMISSION_PARAMETERS"]
            )
            file.write(text_cherenkov)

            file.write("\n* [ DEBUGGING OUTPUT PARAMETERS ]\n")
            text_debugging = self._get_text_single_line(self.config["DEBUGGING_OUTPUT_PARAMETERS"])
            file.write(text_debugging)

            file.write("\n* [ OUTPUT FILE ]\n")
            if use_multipipe:
                file.write(f"TELFIL |{output_file!s}\n")
            else:
                file.write(f"TELFIL {output_file.name}\n")

            file.write("\n* [ IACT TUNING PARAMETERS ]\n")
            text_iact = self._get_text_single_line(
                self.config["IACT_PARAMETERS"],
                "IACT ",
            )
            file.write(text_iact)
            file.write("\nEXIT")

        # Write out the atmospheric transmission file to the model directory.
        # This is done explicitly because it is not done "automatically" when CORSIKA is not piped
        # to sim_telarray.
        self.array_model.site_model.export_atmospheric_transmission_file(
            model_directory=self.array_model.get_config_directory()
        )

    def _write_seeds(self, file, corsika_seeds=None):
        """
        Generate and write seeds in the CORSIKA input file.

        Parameters
        ----------
        file: stream
            File where the telescope positions will be written.
        """
        if not corsika_seeds:
            random_seed = self.get_config_parameter("PRMPAR") + self.run_number
            rng = np.random.default_rng(random_seed)
            corsika_seeds = [int(rng.uniform(0, 1e7)) for _ in range(4)]
        if len(corsika_seeds) != 4:
            raise ValueError("Exactly 4 CORSIKA seeds must be provided.")
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
        for telescope_name, telescope in self.array_model.telescope_models.items():
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

    def is_calibration_run(self):
        """
        Check if this simulation is a calibration run.

        Parameters
        ----------
        run_mode: str
            Run mode of the simulation.

        Returns
        -------
        bool
            True if it is a calibration run, False otherwise.
        """
        return self.run_mode in [
            "pedestals",
            "pedestals_dark",
            "pedestals_nsb_only",
            "direct_injection",
        ]
