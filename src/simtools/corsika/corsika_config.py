"""CORSIKA configuration."""

import logging
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from astropy import units as u

from simtools.constants import CORSIKA_MAX_SEED
from simtools.corsika.input_writer import format_parameter
from simtools.corsika.primary_particle import PrimaryParticle
from simtools.model.model_parameter import ModelParameter
from simtools.sim_events import file_info
from simtools.simulation.configuration import get_model_writer
from simtools.simulation.parameters import SimulationParameters
from simtools.utils import value_conversion
from simtools.utils.geometry import geographic_to_corsika_azimuth
from simtools.utils.random import seeds

CALIBRATION_RUN_MODES = ["pedestals", "pedestals_dark", "pedestals_nsb_only", "direct_injection"]


def _format_repository_entry(parameters_from_repository, name):
    """Format a model repository CORSIKA parameter using its schema-declared format."""
    entry = parameters_from_repository[name]
    return format_parameter(name, entry["value"], entry["unit"])


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

    def __init__(self, array_model, run_number, label=None, context=None):
        """Initialize CorsikaConfig."""
        self._logger = logging.getLogger(__name__)

        self.label = label
        self.shower_events = self.mc_events = None
        self.zenith_angle = self.azimuth_angle = None
        self.curved_atmosphere_min_zenith_angle = None
        self.run_number = run_number
        self.context = context or array_model.context
        self.primary_particle = self.context.args  # see setter for primary_particle
        self.use_curved_atmosphere = self.context.args  # see setter for use_curved_atmosphere
        self.run_mode = self.context.args.get("run_mode")

        self.io_handler = self.context.paths
        self.array_model = array_model
        self.interaction_table_path = self.context.software.corsika_interaction_table_path
        self.config = self._fill_corsika_configuration(self.context.args)
        self._initialize_from_config(self.context.args)

    @property
    def simulation_parameters(self):
        """Return physical run settings without CORSIKA steering cards."""
        return SimulationParameters(
            run_number=self.run_number,
            primary_particle=self.primary_particle,
            zenith_angle=self.zenith_angle,
            azimuth_angle=self.azimuth_angle,
            shower_events=self.shower_events,
            mc_events=self.mc_events,
            viewcone_max=self.get_config_parameter("VIEWCONE")[1],
            zenith_min=self.get_config_parameter("THETAP")[0],
            run_mode=self.run_mode,
            use_curved_atmosphere=self.use_curved_atmosphere,
        )

    @property
    def primary_particle(self):
        """Primary particle."""
        return self._primary_particle

    @primary_particle.setter
    def primary_particle(self, args):
        """
        Set primary particle from input dictionary or CORSIKA 7 particle ID.

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
                    >= args["curved_atmosphere_min_zenith_angle"].to("deg").value
                )
            except KeyError:
                self._use_curved_atmosphere = False

    def _fill_corsika_configuration(self, args):
        """Fill CORSIKA configuration (dictionary keys are CORSIKA parameter names)."""
        if args is None:
            return {}

        config = {
            "RUNNR": [self.run_number],
            "USER": [self.context.activity.user],
            "HOST": [self.context.activity.hostname],
        }
        if self.is_calibration_run():
            config["USER_INPUT"] = self._corsika_configuration_for_dummy_simulations(args)
        elif args.get("corsika_file", None) is not None:
            config["USER_INPUT"] = self._corsika_configuration_from_corsika_file(
                args["corsika_file"]
            )
        else:
            config["USER_INPUT"] = self._corsika_configuration_from_user_input(args)

        config.update(
            self._fill_corsika_configuration_from_repository(
                value_conversion.ensure_list(args.get("model_version"))
            )
        )
        return config

    def _fill_corsika_configuration_from_repository(self, model_versions):
        """Fill CORSIKA configuration from the model repository."""
        # all following parameters require model repository
        if not model_versions:
            return {}

        # For multiple model versions, check that CORSIKA parameters are identical
        self.assert_corsika_configurations_match(model_versions)
        self._logger.debug(
            f"Using model version {model_versions[0]} for CORSIKA parameters "
            "from the model repository"
        )
        parameters_from_repository = self.array_model.site_model.get_simulation_software_parameters(
            "corsika"
        )
        return {
            "INTERACTION_FLAGS": self._corsika_configuration_interaction_flags(
                parameters_from_repository
            ),
            "CHERENKOV_EMISSION_PARAMETERS": self._corsika_configuration_cherenkov_parameters(
                parameters_from_repository
            ),
            "DEBUGGING_OUTPUT_PARAMETERS": self._corsika_configuration_debugging_parameters(),
            "IACT_PARAMETERS": self._corsika_configuration_iact_parameters(
                parameters_from_repository
            ),
        }

    def _initialize_from_config(self, args):
        """Initialize azimuth, zenith and event numbers from command line args or the config."""
        user_input = self.config.get("USER_INPUT", {})
        self.primary_particle = int(user_input.get("PRMPAR", [1])[0])
        self.shower_events = int(user_input.get("NSHOW", [0])[0])
        self.mc_events = int(self.shower_events * user_input.get("CSCAT", [1])[0])

        if args.get("corsika_file", None) is not None:
            phip = self.config["USER_INPUT"]["PHIP"]
            thetap = self.config["USER_INPUT"]["THETAP"]
            azimuth = self._to_corsika_phi(0.5 * (phip[0] + phip[1]))
            zenith = 0.5 * (thetap[0] + thetap[1])
        else:
            azimuth = args.get("azimuth_angle", 0.0 * u.deg).to("deg").value
            zenith = args.get("zenith_angle", 20.0 * u.deg).to("deg").value

        self.azimuth_angle = round(azimuth, 2)
        self.zenith_angle = round(zenith, 2)

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

        parameters_list = [
            ModelParameter(
                model_version=version, context=self.context
            ).get_simulation_software_parameters("corsika")
            for version in model_versions
        ]

        # Parameters that can differ between model versions (e.g., i/o buffer size)
        skip_parameters = ["corsika_iact_io_buffer", "corsika_iact_split_auto"]

        for i in range(len(parameters_list) - 1):
            for key in parameters_list[i]:
                if key in skip_parameters:
                    continue

                current_value = parameters_list[i][key]["value"]
                next_value = parameters_list[i + 1][key]["value"]

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
        """Return fast CORSIKA settings with no or few Cherenkov photons (calibration runs)."""
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
        """Return CORSIKA run and event header settings from an existing CORSIKA input file."""
        run_header, event_header = file_info.get_corsika_run_and_event_headers(corsika_input_file)
        self._logger.debug(f"CORSIKA run header from {corsika_input_file}")

        def to_float32(value):
            return np.float32(value) if value is not None else 0.0

        def to_int32(value):
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
        """Return CORSIKA run settings from user input."""
        theta, phi = self._get_corsika_theta_phi(args_dict)
        return {
            "EVTNR": [args_dict["event_number_first_shower"]],
            "NSHOW": [args_dict["showers_per_run"]],
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
        phi = self._to_corsika_phi(
            args_dict.get("azimuth_angle", 0.0 * u.deg).to("deg").value,
            correct_for_geomagnetic_field_alignment=args_dict.get(
                "correct_for_b_field_alignment", True
            ),
        )
        return theta, phi

    def _corsika_configuration_interaction_flags(self, parameters_from_repository):
        """Return CORSIKA interaction flags and parameters."""
        parameters = {}
        if "corsika_first_interaction_height" in parameters_from_repository:
            parameters["FIXHEI"] = _format_repository_entry(
                parameters_from_repository, "corsika_first_interaction_height"
            )
        grammage = parameters_from_repository["corsika_starting_grammage"]
        parameters["FIXCHI"] = [
            format_parameter(
                "corsika_starting_grammage",
                self._get_starting_grammage_value(grammage["value"]),
                grammage["unit"],
            )
        ]
        if not self.use_curved_atmosphere:
            parameters["TSTART"] = ["T"]
        parameters["ECUTS"] = _format_repository_entry(
            parameters_from_repository, "corsika_particle_kinetic_energy_cutoff"
        )
        parameters["MUADDI"] = ["F"]
        parameters["MUMULT"] = ["T"]
        parameters["LONGI"] = _format_repository_entry(
            parameters_from_repository, "corsika_longitudinal_shower_development"
        )
        parameters["MAXPRT"] = ["10"]
        parameters["ECTMAP"] = ["1.e6"]
        transition_energy = self.context.args.get("corsika_hadronic_transition_energy")
        if transition_energy is not None:
            parameters["HILOW"] = [transition_energy.to_value("GeV")]

        if self.context.software.corsika_interaction_models[0] == "epos":
            parameters.update(self._epos_flags())

        self._logger.debug(f"Interaction parameters: {parameters}")
        return parameters

    def _epos_flags(self):
        """Return EPOS interaction model flags."""
        epos_path = Path(self.interaction_table_path)
        flags = {"EPOPAR fname pathnx": [f"{epos_path}/"]}
        for epos_file in ["inics", "iniev", "inirj", "initl"]:
            flags[f"EPOPAR fname {epos_file}"] = [str(epos_path / f"epos.{epos_file}")]
        flags["EPOPAR fname hpf"] = [str(epos_path / "tables.dat")]
        for dummy_output in ["check", "histo", "data", "copy"]:
            flags[f"EPOPAR fname {dummy_output}"] = ["none"]
        return flags

    def _get_starting_grammage_value(self, value_entry):
        """Return the starting grammage value selected for the primary particle and telescopes."""
        if not isinstance(value_entry, list):
            return value_entry

        tel_types = {tel.design_model for tel in self.array_model.telescope_models.values()}
        matched_values = self._get_matching_grammage_values(
            value_entry, tel_types, self.primary_particle.name
        )

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

    def _corsika_configuration_cherenkov_parameters(self, parameters_from_repository):
        """Return CORSIKA Cherenkov emission parameters."""
        parameters = {
            "CERSIZ": [parameters_from_repository["corsika_cherenkov_photon_bunch_size"]["value"]],
            "CERFIL": "0",
            "CWAVLG": _format_repository_entry(
                parameters_from_repository, "corsika_cherenkov_photon_wavelength_range"
            ),
        }
        self._logger.debug(f"Cherenkov parameters: {parameters}")
        return parameters

    def _corsika_configuration_iact_parameters(self, parameters_from_repository):
        """Return CORSIKA IACT parameters."""
        parameters = {
            "SPLIT_AUTO": [parameters_from_repository["corsika_iact_split_auto"]["value"]],
            "IO_BUFFER": [
                _format_repository_entry(parameters_from_repository, "corsika_iact_io_buffer")
            ],
            "MAX_BUNCHES": [parameters_from_repository["corsika_iact_max_bunches"]["value"]],
        }
        self._logger.debug(f"IACT parameters: {parameters}")
        return parameters

    def _corsika_configuration_debugging_parameters(self):
        """Return CORSIKA debugging output parameters."""
        return {
            "DEBUG": ["F", 6, "F", 1000000],
            "DATBAS": ["F"],
            "DIRECT": ["./"],
            "PAROUT": ["F", "F"],
        }

    def _to_corsika_phi(self, az, correct_for_geomagnetic_field_alignment=True):
        """Convert geographic arrival azimuth to CORSIKA phi (shower frame, plus site rotation)."""
        b_field_declination = 0
        if correct_for_geomagnetic_field_alignment:
            b_field_declination = self.array_model.site_model.get_parameter_value("geomag_rotation")
        return (geographic_to_corsika_azimuth(az) + b_field_declination) % 360.0 % 360.0

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
        par_value = self.config.get(par_name, [])
        if isinstance(par_value, dict):
            par_value = []
        for values in self.config.values():
            if isinstance(values, dict) and par_name in values:
                par_value = values[par_name]
        if len(par_value) == 0:
            raise KeyError(f"Parameter {par_name} is not a CORSIKA config parameter")
        return par_value if len(par_value) > 1 else par_value[0]

    @staticmethod
    def _get_text_single_line(pars, line_begin=""):
        """Return one line of CORSIKA card text per parameter."""
        text = ""
        for par, values in pars.items():
            text += line_begin + par + " " + "".join(str(v) + " " for v in values) + "\n"
        return text

    def _get_top_level_run_parameters(self):
        """Return top-level configuration parameters for the CORSIKA input file."""
        return {key: values for key, values in self.config.items() if not isinstance(values, dict)}

    def generate_corsika_input_file(self, use_multipipe, input_file, output_file):
        """
        Generate a CORSIKA input file.

        Parameters
        ----------
        use_multipipe: bool
            Whether to set the CORSIKA Inputs file to pipe
            the output directly to sim_telarray.
        input_file: Path
            Path to the input file to be generated.
        output_file: Path
            Path to the output file to be generated.
        """
        self._logger.info(f"Exporting CORSIKA input file to {input_file}")

        with open(input_file, "w", encoding="utf-8") as file:
            file.write("\n* [ RUN PARAMETERS ]\n")
            file.write(self._get_text_single_line(self._get_top_level_run_parameters()))
            file.write(self._get_text_single_line(self.config["USER_INPUT"]))

            file.write("\n* [ SITE PARAMETERS ]\n")
            file.write(
                self._get_text_single_line(
                    self.array_model.site_model.get_corsika_site_parameters(
                        config_file_style=True,
                        model_directory=self.array_model.get_config_directory(),
                    )
                )
            )

            file.write("\n* [ IACT ENV PARAMETERS ]\n")
            file.write(f"IACT setenv PRMNAME {self.primary_particle.name}\n")
            file.write(f"IACT setenv ZA {self.zenith_angle}\n")
            file.write(f"IACT setenv AZM {self.azimuth_angle}\n")

            file.write("\n* [ SEEDS ]\n")
            self._write_seeds(file)

            file.write("\n* [ TELESCOPES ]\n")
            file.write(self.get_corsika_telescope_list())

            file.write("\n* [ INTERACTION FLAGS ]\n")
            file.write(self._get_text_single_line(self.config["INTERACTION_FLAGS"]))
            file.write(f"DATDIR {self.interaction_table_path}\n")

            file.write("\n* [ CHERENKOV EMISSION PARAMETERS ]\n")
            file.write(self._get_text_single_line(self.config["CHERENKOV_EMISSION_PARAMETERS"]))

            file.write("\n* [ DEBUGGING OUTPUT PARAMETERS ]\n")
            file.write(self._get_text_single_line(self.config["DEBUGGING_OUTPUT_PARAMETERS"]))

            file.write("\n* [ OUTPUT FILE ]\n")
            if use_multipipe:
                file.write(f"TELFIL |{output_file!s}\n")
            else:
                file.write(f"TELFIL {output_file.name}\n")

            file.write("\n* [ IACT TUNING PARAMETERS ]\n")
            file.write(self._get_text_single_line(self.config["IACT_PARAMETERS"], "IACT "))
            file.write("\nEXIT")

        # Write out the atmospheric transmission file to the model directory.
        # This is done explicitly because it is not done "automatically" when CORSIKA is not piped
        # to sim_telarray.
        self.array_model.site_model.export_atmospheric_transmission_file(
            model_directory=self.array_model.get_config_directory()
        )

    def _write_seeds(self, file):
        """Generate and write the four CORSIKA seeds to the input file."""
        corsika_seeds = self.context.args.get("corsika_seeds", False)
        if not corsika_seeds:
            corsika_seeds = seeds(n_seeds=4, max_seed=CORSIKA_MAX_SEED)
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
            sphere_radius = telescope.get_parameter_value_with_unit("telescope_sphere_radius")
            corsika_input_list += "TELESCOPE"
            for pos in positions:
                corsika_input_list += f"\t {pos.to('cm').value:.3f}"
            corsika_input_list += f"\t {sphere_radius.to('cm'):.3f}"
            corsika_input_list += f"\t # {telescope_name}\n"

        return corsika_input_list

    def is_calibration_run(self):
        """
        Check if this simulation is a calibration run.

        Returns
        -------
        bool
            True if it is a calibration run, False otherwise.
        """
        return self.run_mode in CALIBRATION_RUN_MODES


def get_site_config_parameters(site_model, model_directory=None):
    """Translate site quantities into CORSIKA7 configuration parameters.

    Parameters
    ----------
    site_model : SiteModel
        Resolved site model with atmosphere and geomagnetic field.
    model_directory : pathlib.Path, optional
        Directory containing the atmospheric profile.

    Returns
    -------
    dict
        Native CORSIKA7 site configuration.
    """
    model_directory = model_directory or Path()
    return {
        "OBSLEV": [
            site_model.get_parameter_value_with_unit("corsika_observation_level").to_value("cm")
        ],
        # We always use a custom profile by filename, so this has to be set to 99
        "ATMOSPHERE": [99, "Y"],
        "IACT ATMOFILE": [
            model_directory
            / get_model_writer(
                site_model, context=site_model.context
            ).get_atmospheric_profile_file_name()
        ],
        "MAGNET": [
            site_model.get_parameter_value("geomag_horizontal"),
            site_model.get_parameter_value("geomag_vertical"),
        ],
        "ARRANG": [site_model.get_parameter_value("geomag_rotation")],
    }
