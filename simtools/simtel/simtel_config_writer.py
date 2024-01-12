#!/usr/bin/python3

import logging

__all__ = ["SimtelConfigWriter"]


class SimtelConfigWriter:
    """
    SimtelConfigWriter writes sim_telarray configuration files. It is designed to be used by model
    classes (TelescopeModel and ArrayModel) only.

    Parameters
    ----------
    site: str
        South or North.
    model_version: str
        Version of the model (ex. prod5).
    telescope_model_name: str
        Telescope model name.
    layout_name: str
        Layout name.
    label: str
        Instance label. Important for output file naming.
    """

    TAB = " " * 3
    SITE_PARS = [
        "altitude",
        "atmospheric_transmission",
        "ref_lat",
        "ref_long",
        "array_coordinates",
        "atmospheric_profile",
        "magnetic_field",
    ]
    PARS_NOT_TO_WRITE = [
        "pixel_shape",
        "pixel_diameter",
        "lightguide_efficiency_angle_file",
        "lightguide_efficiency_wavelength_file",
        "ref_lat",
        "ref_long",
        "array_coordinates",
        "atmospheric_profile",
        "magnetic_field",
        "EPSG",
        "mirror_panel_shape",
        "mirror_panel_diameter",
    ]

    def __init__(
        self, site, model_version, layout_name=None, telescope_model_name=None, label=None
    ):
        """
        Initialize SimtelConfigWriter.
        """
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init SimtelConfigWriter")

        self._site = site
        self._model_version = model_version
        self._label = label
        self._layout_name = layout_name
        self._telescope_model_name = telescope_model_name

    def write_telescope_config_file(self, config_file_path, parameters):
        """
        Writes the sim_telarray config file for a single telescope.

        Parameters
        ----------
        config_file_path: str or Path
            Path of the file to write on.
        parameters: dict
            Model parameters in the same structure as used by the TelescopeModel class.
        """
        with open(config_file_path, "w", encoding="utf-8") as file:
            self._write_header(file, "TELESCOPE CONFIGURATION FILE")

            file.write("#ifdef TELESCOPE\n")
            file.write(
                f"   echo Configuration for {self._telescope_model_name}"
                + " - TELESCOPE $(TELESCOPE)\n"
            )
            file.write("#endif\n\n")

            for par in parameters.keys():
                if par in self.PARS_NOT_TO_WRITE:
                    continue
                value = parameters[par]["Value"]
                file.write(f"{par} = {value}\n")

    def write_array_config_file(self, config_file_path, layout, telescope_model, site_parameters):
        """
        Writes the sim_telarray config file for an array of telescopes.

        Parameters
        ----------
        config_file_path: str or Path
            Path of the file to write on.
        layout: ArrayLayout
            Instance of ArrayLayout referent to the array model.
        telescope_model: list of TelescopeModel
            List of TelescopeModel's instances as used by the ArrayModel instance.
        site_parameters: dict
            Site parameters.
        """
        with open(config_file_path, "w", encoding="utf-8") as file:
            self._write_header(file, "ARRAY CONFIGURATION FILE")

            # Be careful with the formatting - simtel is sensitive
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

            # Writing site parameters
            self._write_site_parameters(file, site_parameters)

            # Maximum telescopes
            file.write(self.TAB + f"maximum_telescopes = {len(telescope_model)}\n\n")

            # Default telescope - 0th tel in telescope list
            tel_config_file = telescope_model[0].get_config_file(no_export=True).name
            file.write(f"# include <{tel_config_file}>\n\n")

            # Looping over telescopes - from 1 to ...
            for count, tel_model in enumerate(telescope_model):
                tel_config_file = tel_model.get_config_file(no_export=True).name
                file.write(f"%{layout[count].name}\n")
                file.write(f"#elif TELESCOPE == {count + 1}\n\n")
                file.write(f"# include <{tel_config_file}>\n\n")
            file.write("#endif \n\n")

    def write_single_mirror_list_file(
        self, mirror_number, mirrors, single_mirror_list_file, set_focal_length_to_zero=False
    ):
        """
        Writes the sim_telarray mirror list file for a single mirror.

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
        Writes a generic header. commen_char is the character to be used for comments, which \
        differs among ctypes of config files.
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

    def _write_site_parameters(self, file, site_parameters):
        """Writes site parameters."""
        file.write(self.TAB + "% Site parameters\n")
        for par in site_parameters:
            if par in self.PARS_NOT_TO_WRITE:
                continue
            value = site_parameters[par]["Value"]
            file.write(self.TAB + f"{par} = {value}\n")
        file.write("\n")
