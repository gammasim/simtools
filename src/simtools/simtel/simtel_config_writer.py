#!/usr/bin/python3
"""Configuration file writer for sim_telarray."""

import logging
from copy import deepcopy
from pathlib import Path

import astropy.units as u
import numpy as np

import simtools.utils.general as gen
import simtools.version
from simtools.io import ascii_handler
from simtools.simtel.pulse_shapes import generate_pulse_from_rise_fall_times
from simtools.utils import names

# Module-level logger for class/static utilities
logger = logging.getLogger(__name__)


def sim_telarray_random_seeds(seed, number):
    """
    Generate random seeds to be used in sim_telarray.

    Parameters
    ----------
    seed: int
        Seed for the random number generator.
    number: int
        Number of random seeds to generate.
        Number of random seeds to generate.

    Returns
    -------
    list
        List of random seeds.
    """
    rng = np.random.default_rng(seed)
    max_int32 = np.iinfo(np.int32).max  # sim_telarray requires 32 bit integers
    return list(rng.integers(low=1, high=max_int32, size=number, dtype=np.int32))


class SimtelConfigWriter:
    """
    SimtelConfigWriter writes sim_telarray configuration files.

    It is designed to be used by model classes (TelescopeModel and ArrayModel) only.

    Parameters
    ----------
    site: str
        South or North.
    model_version: str
        Model version.
    telescope_model_name: str
        Telescope model name.
    layout_name: str
        Layout name.
    label: str
        Instance label. Important for output file naming.
    simtel_path: str or Path
        Path to the sim_telarray installation directory.
    """

    TAB = " " * 3

    def __init__(
        self,
        site,
        model_version,
        layout_name=None,
        telescope_model_name=None,
        telescope_design_model=None,
        label=None,
        simtel_path=None,
    ):
        """Initialize SimtelConfigWriter."""
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init SimtelConfigWriter")

        self._site = site
        self._model_version = model_version
        self._label = label
        self._layout_name = layout_name
        self._telescope_model_name = telescope_model_name
        self._telescope_design_model = telescope_design_model
        self._simtel_path = simtel_path

    def write_telescope_config_file(
        self, config_file_path, parameters, telescope_name=None, telescope_design_model=None
    ):
        """
        Write the sim_telarray config file for a single telescope.

        Parameters
        ----------
        config_file_path: str or Path
            Path of the file to write on.
        parameters: dict
            Model parameters
        telescope_name: str
            Name of the telescope (use self._telescope_model_name if None)
        telescope_design_model: str
            Telescope design model.
        """
        self._logger.debug(f"Writing telescope config file {config_file_path}")

        simtel_par = self._get_parameters_for_sim_telarray(parameters, config_file_path)
        telescope_name = telescope_name or self._telescope_model_name
        _telescope_design_model = telescope_design_model or self._telescope_design_model

        with open(config_file_path, "w", encoding="utf-8") as file:
            self._write_header(file, "TELESCOPE CONFIGURATION FILE")

            file.write("#ifdef TELESCOPE\n")
            file.write(f"   echo Configuration for {telescope_name} - TELESCOPE $(TELESCOPE)\n")
            file.write("#endif\n\n")

            for simtel_name, simtel_value in simtel_par.items():
                file.write(f"{simtel_name} = {self._get_value_string_for_simtel(simtel_value)}\n")
            for meta in self._get_sim_telarray_metadata(
                "telescope",
                parameters,
                telescope_name,
                telescope_design_model=_telescope_design_model,
            ):
                file.write(f"{meta}\n")

    @staticmethod
    def write_lightpulse_table_gauss_expconv(
        file_path,
        width_ns=None,
        exp_decay_ns=None,
        dt_ns=0.1,
        rise_range=(0.1, 0.9),
        fall_range=(0.9, 0.1),
        fadc_sum_bins=None,
        time_margin_ns=10.0,
    ):
        """Write a pulse table for a Gaussian convolved with a causal exponential.

        Parameters
        ----------
        file_path : str or pathlib.Path
            Destination path of the ASCII pulse table to write. Parent directory must exist.
        width_ns : float
            Target rise time in ns between the fractional levels defined by ``rise_range``.
            Defaults correspond to 10-90% rise time.
        exp_decay_ns : float
            Target fall time in ns between the fractional levels defined by ``fall_range``.
            Defaults correspond to 90-10% fall time.
        dt_ns : float, optional
            Time sampling step in ns for the generated pulse table. Default is 0.1.
        rise_range : tuple[float, float], optional
            Fractional amplitude bounds (low, high) for rise-time definition. Default (0.1, 0.9).
        fall_range : tuple[float, float], optional
            Fractional amplitude bounds (high, low) for fall-time definition. Default (0.9, 0.1).
        fadc_sum_bins : int
            Length of the FADC integration window (treated as ns here) used to derive
            the internal time sampling window of the solver as [-(margin), bins + margin].
        time_margin_ns : float, optional
            Margin in ns to add to both ends of the FADC window when ``fadc_sum_bins`` is given.
            Default is 5.0 ns.

        Returns
        -------
        pathlib.Path
            The path to the created pulse table file.

        Notes
        -----
        The underlying model is a Gaussian convolved with a causal exponential. The model
        parameters (sigma, tau) are solved such that the normalized pulse matches the requested
        rise and fall times. The pulse is normalized to a peak amplitude of 1.
        """
        if width_ns is None or exp_decay_ns is None:
            raise ValueError("width_ns (rise 10-90) and exp_decay_ns (fall 90-10) are required")
        logger.info(
            f"Generating lightpulse table with rise10-90={width_ns} ns, "
            f"fall90-10={exp_decay_ns} ns, dt={dt_ns} ns"
        )

        width = float(fadc_sum_bins)
        t_start_ns = -abs(time_margin_ns + width)
        t_stop_ns = +abs(time_margin_ns + width)
        t, y = generate_pulse_from_rise_fall_times(
            width_ns,
            exp_decay_ns,
            dt_ns=dt_ns,
            rise_range=rise_range,
            fall_range=fall_range,
            t_start_ns=t_start_ns,
            t_stop_ns=t_stop_ns,
        )

        with open(file_path, "w", encoding="utf-8") as fh:
            fh.write("# time[ns] amplitude\n")
            for ti, yi in zip(t, y):
                fh.write(f"{ti:.6f} {yi:.8f}\n")
        return Path(file_path)

    def _get_parameters_for_sim_telarray(self, parameters, config_file_path):
        """
        Convert parameter dictionary to sim_telarray configuration file format.

        Accounts for differences between the data models for sim_telarray configuration
        and the simulation models.

        Parameters
        ----------
        parameters: dict
            Model parameters.
        config_file_path: str or Path
            Path of the file to write on.

        Returns
        -------
        dict
            Model parameters in sim_telarray format.
        """
        simtel_par = {}
        for par, value in parameters.items():
            simtel_name, simtel_value = self._convert_model_parameters_to_simtel_format(
                names.get_simulation_software_name_from_parameter_name(
                    par, software_name="sim_telarray"
                ),
                value["value"],
                config_file_path,
                None,
            )
            if simtel_name:
                simtel_par[simtel_name] = simtel_value
        if "stars" not in parameters:  # sim_telarray requires 'stars' to be set
            simtel_par["stars"] = None

        return self._get_flasher_parameters_for_sim_telarray(parameters, simtel_par)

    def _get_flasher_parameters_for_sim_telarray(self, parameters, simtel_par):
        """
        Combine flasher pulse time parameters into a single parameter.

        Takes into account that sim_telarray expects a single parameter with a specific name.

        Parameters
        ----------
        parameters: dict
            Model parameters.
        simtel_par: dict
            Model parameters in sim_telarray format.

        Returns
        -------
        dict
            Model parameters in sim_telarray format including flasher parameters.

        """
        if "flasher_pulse_shape" not in parameters and "flasher_pulse_width" not in parameters:
            return simtel_par

        mapping = {
            "gauss": "laser_pulse_sigtime",
            "tophat": "laser_pulse_twidth",
        }

        shape = parameters.get("flasher_pulse_shape", {}).get("value", "").lower()
        if "exponential" in shape:
            simtel_par["laser_pulse_exptime"] = parameters.get("flasher_pulse_exp_decay", {}).get(
                "value", 0.0
            )
        else:
            simtel_par["laser_pulse_exptime"] = 0.0

        width = parameters.get("flasher_pulse_width", {}).get("value", 0.0)

        simtel_par.update(dict.fromkeys(mapping.values(), 0.0))
        if shape == "gauss-exponential":
            simtel_par["laser_pulse_sigtime"] = width
        elif shape in mapping:
            simtel_par[mapping[shape]] = width
        else:
            self._logger.warning(f"Flasher pulse shape '{shape}' without width definition")

        return simtel_par

    def _get_value_string_for_simtel(self, value):
        """
        Return a value string for simtel.

        Parameters
        ----------
        value: any
            Value to convert to string.

        Returns
        -------
        str
            Value string for simtel.
        """
        value = "none" if value is None else value  # simtel requires 'none'
        if isinstance(value, bool):
            value = 1 if value else 0
        elif isinstance(value, list | np.ndarray):
            value = gen.convert_list_to_string(value, shorten_list=True)
        return value

    def _get_sim_telarray_metadata(
        self,
        config_type,
        model_parameters,
        telescope_model_name,
        additional_metadata=None,
        telescope_design_model=None,
    ):
        """
        Return sim_telarray metadata.

        Parameters
        ----------
        type: str
            Type of the configuration file (telescope, site)
        model_parameters: dict
            Model parameters dictionary.
        telescope_model_name: str
            Name of the telescope model
        additional_metadata: dict
            Dictionary with additional metadata to include using 'set'.
        telescope_design_model: str
            Name of the telescope design model.

        Returns
        -------
        list
            List with sim_telarray metadata.
        """
        meta_parameters = [
            f"config_release = {self._model_version} with simtools v{simtools.version.__version__}",
            f"config_version = {self._model_version}",
        ]
        telescope_design_model = telescope_design_model or "design_model_not_set"
        if config_type == "telescope":
            meta_parameters.extend(
                [
                    f"camera_config_name = {telescope_design_model}",
                    f"camera_config_variant = {telescope_model_name}",
                    f"camera_config_version = {self._model_version}",
                    f"optics_config_name = {telescope_design_model}",
                    f"optics_config_variant = {telescope_model_name}",
                    f"optics_config_version = {self._model_version}",
                ]
            )
            prefix = "metaparam telescope"
        elif config_type == "site":
            meta_parameters.extend(
                [
                    f"site_config_name = {self._site}",
                    "site_config_variant = ",
                    f"site_config_version = {self._model_version}",
                    f"array_config_name = {self._layout_name}",
                    "array_config_variant = ",
                    f"array_config_version = {self._model_version}",
                ]
            )
            prefix = "metaparam global"
            meta_parameters.append("metaparam global add random_seed")
        else:
            raise ValueError(f"Unknown metadata type {config_type}")

        self._add_model_parameters_to_metadata(model_parameters, meta_parameters, prefix)

        if additional_metadata:
            for key, value in additional_metadata.items():
                if value:
                    meta_parameters.append(f"{prefix} set {key}={value}")

        return meta_parameters

    def _add_model_parameters_to_metadata(self, model_parameters, meta_parameters, prefix):
        """Add model parameters to metadata."""
        if not model_parameters:
            return

        for key, value in model_parameters.items():
            simtel_name = names.get_simulation_software_name_from_parameter_name(
                key, software_name="sim_telarray", set_meta_parameter=False
            )
            if simtel_name and value.get("meta_parameter"):
                meta_parameters.append(f"{prefix} add {simtel_name}")
            simtel_name = names.get_simulation_software_name_from_parameter_name(
                key, software_name="sim_telarray", set_meta_parameter=True
            )
            if simtel_name and value.get("meta_parameter"):
                meta_parameters.append(f"{prefix} set {simtel_name}={value['value']}")

    def write_array_config_file(
        self, config_file_path, telescope_model, site_model, additional_metadata=None
    ):
        """
        Write the sim_telarray config file for an array of telescopes.

        Parameters
        ----------
        config_file_path: str or Path
            Path of the file to write on.
        telescope_model: dict of TelescopeModel
            Dictionary of TelescopeModel's instances as used by the ArrayModel instance.
        site_model: Site model
            Site model.
        additional_metadata: dict
            Dictionary with additional metadata to include.
        """
        config_file_directory = Path(config_file_path).parent
        with open(config_file_path, "w", encoding="utf-8") as file:
            self._write_header(file, "ARRAY CONFIGURATION FILE")

            file.write("#ifndef TELESCOPE\n")
            file.write("# define TELESCOPE 0\n")
            file.write("#endif\n\n")

            # TELESCOPE 0 - global parameters
            file.write("#if TELESCOPE == 0\n")
            file.write(self.TAB + "echo *****************************\n")
            file.write(self.TAB + f"echo Site: {self._site}\n")
            file.write(self.TAB + f"echo LayoutName: {self._layout_name}\n")
            file.write(self.TAB + f"echo ModelVersion: {self._model_version}\n")
            file.write(self.TAB + "echo *****************************\n\n")

            self._write_simtools_parameters(file)

            self._write_site_parameters(
                file,
                site_model.parameters,
                config_file_directory,
                telescope_model,
                additional_metadata,
            )

            file.write(self.TAB + f"maximum_telescopes = {len(telescope_model)}\n\n")

            # Default telescope in sim_telarray - 0th tel in telescope list
            _, first_telescope = next(iter(telescope_model.items()))
            invalid_telescope_name = "InvalidTelescope"
            file.write(f"# include <{invalid_telescope_name}.cfg>\n\n")
            self.write_dummy_telescope_configuration_file(
                deepcopy(first_telescope.parameters),
                config_file_directory / f"{invalid_telescope_name}.cfg",
                invalid_telescope_name,
            )

            for count, (tel_name, tel_model) in enumerate(telescope_model.items()):
                tel_config_file = tel_model.config_file_path.name
                file.write(f"% {tel_name}\n")
                file.write(f"#elif TELESCOPE == {count + 1}\n\n")
                file.write(f"# include <{tel_config_file}>\n\n")
            file.write("#endif \n\n")  # configuration files need to end with \n\n

        if additional_metadata and additional_metadata.get("random_instrument_instances"):
            self._write_random_seeds_file(additional_metadata, config_file_directory)

    def _write_random_seeds_file(self, sim_telarray_seeds, config_file_directory):
        """
        Write list of random number used to generate random instances of instrument.

        Parameters
        ----------
        random_instrument_instances: int
            Number of random instances of the instrument.
        """
        self._logger.info(
            "Writing random seed file "
            f"{config_file_directory}/{sim_telarray_seeds['seed_file_name']}"
            f" (global seed {sim_telarray_seeds['seed']})"
        )
        if sim_telarray_seeds["random_instrument_instances"] > 1024:
            raise ValueError("Number of random instances of instrument must be less than 1024")
        random_integers = sim_telarray_random_seeds(
            sim_telarray_seeds["seed"], sim_telarray_seeds["random_instrument_instances"]
        )
        with open(
            config_file_directory / sim_telarray_seeds["seed_file_name"], "w", encoding="utf-8"
        ) as file:
            file.write(
                "# Random seeds for instrument configuration generated with seed "
                f"{sim_telarray_seeds['seed']}"
                f" (model version {self._model_version}, site {self._site})\n"
            )
            for number in random_integers:
                file.write(f"{number}\n")

    def write_single_mirror_list_file(
        self, mirror_number, mirrors, single_mirror_list_file, set_focal_length_to_zero=False
    ):
        """
        Write the sim_telarray mirror list file for a single mirror.

        Parameters
        ----------
        mirror_number: int
            Mirror number.
        mirrors: Mirrors
            Instance of Mirrors.
        single_mirror_list_file: str or Path
            Path of the file to write on.
        set_focal_length_to_zero: bool
            Flag to set the focal length to zero.
        """
        (
            __,
            __,
            mirror_panel_diameter,
            focal_length,
            shape_type,
        ) = mirrors.get_single_mirror_parameters(mirror_number)
        with open(single_mirror_list_file, "w", encoding="utf-8") as file:
            self._write_header(file, "MIRROR LIST FILE", "#")

            file.write("# Column 1: X pos. [cm] (North/Down)\n")
            file.write("# Column 2: Y pos. [cm] (West/Right from camera)\n")
            file.write("# Column 3: flat-to-flat diameter [cm]\n")
            file.write(
                "# Column 4: focal length [cm], typically zero = adapting in sim_telarray.\n"
            )
            file.write(
                "# Column 5: shape type: 0=circular, 1=hex. with flat side parallel to y, "
                "2=square, 3=other hex. (default: 0)\n"
            )
            file.write(
                "# Column 6: Z pos (height above dish backplane) [cm], typ. omitted (or zero)"
                " to adapt to dish shape settings.\n"
            )
            file.write("#\n")
            file.write(
                f"0. 0. {mirror_panel_diameter.to('cm').value} "
                f"{focal_length.to('cm').value if not set_focal_length_to_zero else 0} "
                f"{shape_type} 0.\n"
            )

    def _write_header(self, file, title, comment_char="%"):
        """
        Write a generic header.

        Parameters
        ----------
        file: file
            File to write.
        title: str
            Title of the header.
        comment_char: str
            Character to be used for comments, which differs among types of config files.
        """
        header = f"{comment_char}{50 * '='}\n"
        header += f"{comment_char} {title}\n"
        header += f"{comment_char} Site: {self._site}\n"
        header += f"{comment_char} ModelVersion: {self._model_version}\n"
        header += (
            f"{comment_char} TelescopeModelName: {self._telescope_model_name}\n"
            if self._telescope_model_name is not None
            else ""
        )
        header += (
            f"{comment_char} LayoutName: {self._layout_name}\n"
            if self._layout_name is not None
            else ""
        )
        header += f"{comment_char} Label: {self._label}\n" if self._label is not None else ""
        header += f"{comment_char}{50 * '='}\n"
        header += f"{comment_char}\n"
        file.write(header)

    def _write_simtools_parameters(self, file):
        """Write simtools-specific parameters."""
        meta_items = {
            "simtools_version": simtools.version.__version__,
            "simtools_model_production_version": self._model_version,
        }
        try:
            build_opts = ascii_handler.collect_data_from_file(
                Path(self._simtel_path) / "build_opts.yml"
            )
            for key, value in build_opts.items():
                meta_items[f"simtools_{key}"] = value
        except (FileNotFoundError, TypeError):
            pass  # don't expect build_opts.yml to be present on all systems

        file.write(f"{self.TAB}% Simtools parameters\n")
        for key, value in meta_items.items():
            file.write(f"{self.TAB}metaparam global set {key} = {value}\n")

    def _write_site_parameters(
        self, file, site_parameters, model_path, telescope_model, additional_metadata=None
    ):
        """
        Write site parameters.

        Parameters
        ----------
        file: file
            File to write on.
        site_parameters: site parameters
            Site parameters.
        model_path: Path
            Path to the model for writing of additional files.
        telescope_model: dict of TelescopeModel
            Telescope models.
        additional_metadata: dict
            Dictionary with additional metadata to include.
        """
        file.write(self.TAB + "% Site parameters\n")
        for par, value in site_parameters.items():
            simtel_name, simtel_value = self._convert_model_parameters_to_simtel_format(
                names.get_simulation_software_name_from_parameter_name(
                    par, software_name="sim_telarray"
                ),
                value["value"],
                model_path,
                telescope_model,
            )
            if simtel_name is not None:
                file.write(f"{self.TAB}{simtel_name} = {simtel_value}\n")
        for meta in self._get_sim_telarray_metadata(
            "site", site_parameters, None, additional_metadata
        ):
            file.write(f"{self.TAB}{meta}\n")
        file.write("\n")

    def _convert_model_parameters_to_simtel_format(
        self, simtel_name, value, model_path, telescope_model
    ):
        """
        Convert model parameter value to simtel format.

        This might involve format or unit conversion and writing to a parameter file.

        Parameters
        ----------
        simtel_name: str
            Parameter name.
        value: any
            Value to convert.
        model_path: Path
            Path to the model for writing of additional files.
        telescope_model: dict of TelescopeModel
            Telescope models.

        Returns
        -------
        str, any
            Converted parameter name and value.
        """
        conversion_dict = {
            "array_triggers": self._write_array_triggers_file,
        }
        try:
            value = conversion_dict[simtel_name](value, model_path, telescope_model)
        except KeyError:
            pass
        except AttributeError:  # covers cases where telescope_model is None
            return None, None
        return simtel_name, value

    def _write_array_triggers_file(self, array_triggers, model_path, telescope_model):
        """
        Write array trigger definition file in simtel format.

        Parameters
        ----------
        array_triggers: dict
            Array trigger definitions.
        model_path: Path
            Path to the model for writing of additional files.
        telescope_model: dict of TelescopeModel
            Telescope models.
        """
        trigger_per_telescope_type = self._group_telescopes_by_type(telescope_model)
        hardstereo_lines, non_hardstereo_groups, all_non_hardstereo_tels, multiplicity = (
            self._process_telescope_triggers(array_triggers, trigger_per_telescope_type)
        )

        array_triggers_file = "array_triggers.dat"
        with open(model_path / array_triggers_file, "w", encoding="utf-8") as file:
            file.write("# Array trigger definition\n")
            self._write_trigger_lines(
                file, hardstereo_lines, non_hardstereo_groups, all_non_hardstereo_tels, multiplicity
            )

        return array_triggers_file

    def _group_telescopes_by_type(self, telescope_model):
        """Group telescopes by their type."""
        trigger_per_telescope_type = {}
        for count, tel_name in enumerate(telescope_model.keys()):
            telescope_type = names.get_array_element_type_from_name(tel_name)
            trigger_per_telescope_type.setdefault(telescope_type, []).append(count + 1)
        return trigger_per_telescope_type

    def _process_telescope_triggers(self, array_triggers, trigger_per_telescope_type):
        """Process telescope triggers and group them by hardstereo and parameters."""
        hardstereo_lines = []
        non_hardstereo_groups = {}
        all_non_hardstereo_tels = []
        multiplicity = None

        for tel_type, tel_list in trigger_per_telescope_type.items():
            trigger_dict = self._get_array_triggers_for_telescope_type(
                array_triggers, tel_type, len(tel_list)
            )
            width, minsep = self._extract_trigger_parameters(trigger_dict)
            multiplicity = trigger_dict["multiplicity"]["value"]  # Store for later use

            if trigger_dict.get("hard_stereo", {}).get("value"):
                line = self._build_trigger_line(
                    trigger_dict, tel_list, width, minsep, hardstereo=True
                )
                hardstereo_lines.append(line)
            else:
                key = (width, minsep)
                non_hardstereo_groups.setdefault(key, []).extend(tel_list)
                all_non_hardstereo_tels.extend(tel_list)

        return hardstereo_lines, non_hardstereo_groups, all_non_hardstereo_tels, multiplicity

    def _extract_trigger_parameters(self, trigger_dict):
        """Extract width and min_separation parameters from trigger dictionary."""
        width = trigger_dict["width"]["value"] * u.Unit(trigger_dict["width"]["unit"]).to("ns")
        minsep = None
        if all(trigger_dict["min_separation"][key] is not None for key in ["value", "unit"]):
            minsep = trigger_dict["min_separation"]["value"] * u.Unit(
                trigger_dict["min_separation"]["unit"]
            ).to("m")
        return width, minsep

    def _build_trigger_line(self, trigger_dict, tel_list, width, minsep, hardstereo=False):
        """Build a trigger line string."""
        line = f"Trigger {trigger_dict['multiplicity']['value']} of "
        line += ", ".join(map(str, tel_list))
        line += f" width {width}"
        if hardstereo:
            line += " hardstereo"
        if minsep is not None:
            line += f" minsep {minsep}"
        return line

    def _write_trigger_lines(
        self, file, hardstereo_lines, non_hardstereo_groups, all_non_hardstereo_tels, multiplicity
    ):
        """Write all trigger lines to file."""
        # Write hardstereo lines first
        for line in hardstereo_lines:
            file.write(f"{line}\n")

        # Write individual non-hardstereo groups if they have different parameters
        if len(non_hardstereo_groups) > 1:
            for (width, minsep), tel_list in non_hardstereo_groups.items():
                line = f"Trigger {multiplicity} of "
                line += ", ".join(map(str, tel_list))
                line += f" width {width}"
                if minsep is not None:
                    line += f" minsep {minsep}"
                file.write(f"{line}\n")

        # Write combined line with all non-hardstereo telescopes using shortest values
        if all_non_hardstereo_tels:
            min_width = min(width for width, minsep in non_hardstereo_groups.keys())
            min_minsep = self._get_minimum_minsep(non_hardstereo_groups)

            combined_line = f"Trigger {multiplicity} of "
            combined_line += ", ".join(map(str, sorted(all_non_hardstereo_tels)))
            combined_line += f" width {min_width}"
            if min_minsep is not None:
                combined_line += f" minsep {min_minsep}"
            file.write(f"{combined_line}\n")

    def _get_minimum_minsep(self, non_hardstereo_groups):
        """Get minimum min_separation value from groups."""
        minsep_values = [
            minsep for width, minsep in non_hardstereo_groups.keys() if minsep is not None
        ]
        return min(minsep_values) if minsep_values else None

    def _get_array_triggers_for_telescope_type(
        self, array_triggers, telescope_type, num_telescopes_of_type
    ):
        """
        Get array trigger for a specific telescope type.

        Parameters
        ----------
        array_triggers: dict
            Array trigger definitions.
        telescope_type: str
            Telescope type.
        num_telescopes_of_type: int
            Number of telescopes of the specified type.

        Returns
        -------
        dict
            Array trigger for the telescope type.
        """
        suffix = "_array"
        if num_telescopes_of_type == 1:
            suffix = "_single_telescope"
        for trigger_dict in array_triggers:
            if trigger_dict["name"] == telescope_type + suffix:
                return trigger_dict
        return None

    def write_dummy_telescope_configuration_file(
        self, parameters, config_file_path, telescope_name
    ):
        """
        Write 'dummy' telescope configuration file used as zeroth telescope in sim_telarray.

        Replaces key telescope configuration values with dummy values.

        Parameters
        ----------
        parameters: dict
            Telescope parameters used as template.
        config_file_path: str or Path
            Path of the dummy configuration file to write on.
        telescope_name: str
            Name of the telescope.
        """
        self._logger.debug(f"Writing {telescope_name} telescope config file {config_file_path}")
        dummy_defaults = {
            "camera_config_file": f"{telescope_name}_single_pixel_camera.dat",
            "discriminator_pulse_shape": f"{telescope_name}_pulse.dat",
            "fadc_pulse_shape": f"{telescope_name}_pulse.dat",
            "mirror_list": f"{telescope_name}_single_12m_mirror.dat",
            "mirror_reflectivity": f"{telescope_name}_reflectivity.dat",
            "camera_pixels": 1,
            "trigger_pixels": 1,
            "num_gains": 1,
            "fadc_bins": 10,
            "disc_bins": 10,
            "fadc_sum_bins": 10,
            "fadc_sum_offset": 0,
            "asum_threshold": 9999,
            "dsum_threshold": 9999,
            "discriminator_threshold": 9999,
            "fadc_amplitude": 1.0,
            "discriminator_amplitude": 1.0,
        }

        for key, val in dummy_defaults.items():
            if key in parameters:
                parameters[key]["value"] = val

        self.write_telescope_config_file(config_file_path, parameters, telescope_name)

        config_file_directory = Path(config_file_path).parent
        self._write_dummy_mirror_list_files(config_file_directory, telescope_name)
        self._write_dummy_camera_files(config_file_directory, telescope_name)

    def _write_dummy_mirror_list_files(self, config_directory, telescope_name):
        """Write dummy mirror list with single mirror and reflectivity file."""
        with open(
            config_directory / f"{telescope_name}_single_12m_mirror.dat", "w", encoding="utf-8"
        ) as file:
            file.write("0 0 1200 0.0 0\n")
        with open(
            config_directory / f"{telescope_name}_reflectivity.dat", "w", encoding="utf-8"
        ) as file:
            file.writelines(f"{w} 0.8\n" for w in range(200, 801, 50))

    def _write_dummy_camera_files(self, config_directory, telescope_name):
        """Write dummy camera, pulse shape, and funnels file with a single pixel."""
        with open(
            config_directory / f"{telescope_name}_single_pixel_camera.dat", "w", encoding="utf-8"
        ) as file:
            file.write(f'PixType 1   0  0 300   1 300 0.00   "{telescope_name}_funnels.dat"\n')
            file.write("Pixel 0 1 0. 0.  0  0  0 0x00 1\n")
            file.write("Trigger 1 of 0\n")

        with open(config_directory / f"{telescope_name}_funnel.dat", "w", encoding="utf-8") as file:
            file.writelines(f"{a} 0.78 1.5\n" for a in range(36))

        with open(config_directory / f"{telescope_name}_pulse.dat", "w", encoding="utf-8") as file:
            file.write("0 0 0\n")
            file.write("1 1 1\n")
            file.write("2 2 2\n")
            file.write("3 3 3\n")
            file.write("4 0 0\n")
