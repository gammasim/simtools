#!/usr/bin/python3
"""Configuration file writer for sim_telarray."""

import logging
import random
from pathlib import Path

import astropy.units as u
import numpy as np

import simtools.utils.general as gen
import simtools.version
from simtools.utils import names

__all__ = ["SimtelConfigWriter"]


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
    """

    TAB = " " * 3

    def __init__(
        self, site, model_version, layout_name=None, telescope_model_name=None, label=None
    ):
        """Initialize SimtelConfigWriter."""
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init SimtelConfigWriter")

        self._site = site
        self._model_version = model_version
        self._label = label
        self._layout_name = layout_name
        self._telescope_model_name = telescope_model_name

    def write_telescope_config_file(self, config_file_path, parameters):
        """
        Write the sim_telarray config file for a single telescope.

        Parameters
        ----------
        config_file_path: str or Path
            Path of the file to write on.
        parameters: dict
            Model parameters
        """
        self._logger.debug(f"Writing telescope config file {config_file_path}")

        with open(config_file_path, "w", encoding="utf-8") as file:
            self._write_header(file, "TELESCOPE CONFIGURATION FILE")

            file.write("#ifdef TELESCOPE\n")
            file.write(
                f"   echo Configuration for {self._telescope_model_name} - TELESCOPE $(TELESCOPE)\n"
            )
            file.write("#endif\n\n")

            for par, value in parameters.items():
                simtel_name, value = self._convert_model_parameters_to_simtel_format(
                    names.get_simulation_software_name_from_parameter_name(
                        par, software_name="sim_telarray"
                    ),
                    value["value"],
                    config_file_path,
                    None,
                )
                if simtel_name:
                    file.write(f"{simtel_name} = {self._get_value_string_for_simtel(value)}\n")
            if "stars" not in parameters:  # sim_telarray requires 'stars' to be set
                file.write("stars = none\n")
            for meta in self._get_sim_telarray_metadata("telescope", parameters):
                file.write(f"{meta}\n")

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
        elif isinstance(value, (list, np.ndarray)):  # noqa: UP038
            value = gen.convert_list_to_string(value, shorten_list=True)
        return value

    def _get_sim_telarray_metadata(self, config_type, model_parameters):
        """
        Return sim_telarray metadata.

        Parameters
        ----------
        type: str
            Type of the configuration file (telescope, site)
        model_parameters: dict
            Model parameters dictionary.

        Returns
        -------
        list
            List with sim_telarray metadata.
        """
        meta_parameters = [
            f"config_release = {self._model_version} with simtools v{simtools.version.__version__}",
            f"config_version = {self._model_version}",
        ]
        if config_type == "telescope":
            meta_parameters.extend(
                [
                    f"camera_config_name = {self._telescope_model_name}",
                    "camera_config_variant = ",
                    f"camera_config_version = {self._model_version}",
                    f"optics_config_name = {self._telescope_model_name}",
                    "optics_config_variant = ",
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
        else:
            raise ValueError(f"Unknown metadata type {config_type}")

        if model_parameters:
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

        return meta_parameters

    def write_array_config_file(
        self, config_file_path, telescope_model, site_model, sim_telarray_seeds=None
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
        sim_telarray_seeds: dict
            Dictionary with configuration for sim_telarray random instrument setup.
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

            self._write_site_parameters(
                file, site_model.parameters, config_file_directory, telescope_model
            )

            file.write(self.TAB + f"maximum_telescopes = {len(telescope_model)}\n\n")

            # Default telescope in sim_telarray - 0th tel in telescope list
            _, first_telescope = next(iter(telescope_model.items()))
            tel_config_file = first_telescope.config_file_path.name
            file.write(f"# include <{tel_config_file}>\n\n")

            for count, (tel_name, tel_model) in enumerate(telescope_model.items()):
                tel_config_file = tel_model.config_file_path.name
                file.write(f"%{tel_name}\n")
                file.write(f"#elif TELESCOPE == {count + 1}\n\n")
                file.write(f"# include <{tel_config_file}>\n\n")
            file.write("#endif \n\n")  # configuration files need to end with \n\n

        if sim_telarray_seeds and sim_telarray_seeds.get("random_instances"):
            self._write_random_seeds_file(sim_telarray_seeds, config_file_directory)

    def _write_random_seeds_file(self, sim_telarray_seeds, config_file_directory):
        """
        Write list of random number used to generate random instances of instrument.

        Parameters
        ----------
        random_instances_of_instrument: int
            Number of random instances of the instrument.
        """
        random.seed(sim_telarray_seeds["seed"])
        self._logger.info(
            "Writing random seed file "
            f"{config_file_directory}/{sim_telarray_seeds['seed_file_name']}"
            f" (global seed {sim_telarray_seeds['seed']})"
        )
        random_integers = [
            random.randint(0, 2**32 - 1) for _ in range(sim_telarray_seeds["random_instances"])
        ]
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
            Character to be used for comments, which differs among ctypes of config files.
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

    def _write_site_parameters(self, file, site_parameters, model_path, telescope_model):
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
        """
        file.write(self.TAB + "% Site parameters\n")
        for par, value in site_parameters.items():
            simtel_name, value = self._convert_model_parameters_to_simtel_format(
                names.get_simulation_software_name_from_parameter_name(
                    par, software_name="sim_telarray"
                ),
                value["value"],
                model_path,
                telescope_model,
            )
            if simtel_name is not None:
                file.write(f"{self.TAB}{simtel_name} = {value}\n")
        for meta in self._get_sim_telarray_metadata("site", site_parameters):
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
        trigger_per_telescope_type = {}
        for count, tel_name in enumerate(telescope_model.keys()):
            telescope_type = names.get_array_element_type_from_name(tel_name)
            trigger_per_telescope_type.setdefault(telescope_type, []).append(count + 1)

        trigger_lines = {}
        for tel_type, tel_list in trigger_per_telescope_type.items():
            trigger_dict = self._get_array_triggers_for_telescope_type(array_triggers, tel_type)
            trigger_lines[tel_type] = f"Trigger {trigger_dict['multiplicity']['value']} of "
            trigger_lines[tel_type] += ", ".join(map(str, tel_list))
            width = trigger_dict["width"]["value"] * u.Unit(trigger_dict["width"]["unit"]).to("ns")
            trigger_lines[tel_type] += f" width {width}"
            if trigger_dict.get("hard_stereo"):
                trigger_lines[tel_type] += " hard_stereo"
            if all(trigger_dict["min_separation"][key] is not None for key in ["value", "unit"]):
                min_sep = trigger_dict["min_separation"]["value"] * u.Unit(
                    trigger_dict["min_separation"]["unit"]
                ).to("m")
                trigger_lines[tel_type] += f" minsep {min_sep}"

        array_triggers_file = "array_triggers.dat"
        with open(model_path / array_triggers_file, "w", encoding="utf-8") as file:
            file.write("# Array trigger definition\n")
            file.writelines(f"{line}\n" for line in trigger_lines.values())

        return array_triggers_file

    def _get_array_triggers_for_telescope_type(self, array_triggers, telescope_type):
        """
        Get array trigger for a specific telescope type.

        Parameters
        ----------
        array_triggers: dict
            Array trigger definitions.
        telescope_type: str
            Telescope type.

        Returns
        -------
        dict
            Array trigger for the telescope type.
        """
        for trigger_dict in array_triggers:
            if trigger_dict["name"] == telescope_type + "_array":
                return trigger_dict
        return None
