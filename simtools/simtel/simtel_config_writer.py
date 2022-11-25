#!/usr/bin/python3

import logging

__all__ = ["SimtelConfigWriter"]


class SimtelConfigWriter:
    """
    SimtelConfigWriter writes sim_telarray configuration files. It is designed to be used by model\
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
        with open(config_file_path, "w") as file:
            self._write_header(file, "TELESCOPE CONFIGURATION FILE")

            file.write("#ifdef TELESCOPE\n")
            file.write(
                "   echo Configuration for {}".format(self._telescope_model_name)
                + " - TELESCOPE $(TELESCOPE)\n"
            )
            file.write("#endif\n\n")

            for par in parameters.keys():
                if par in self.PARS_NOT_TO_WRITE:
                    continue
                value = parameters[par]["Value"]
                file.write("{} = {}\n".format(par, value))

    def write_array_config_file(self, config_file_path, layout, telescope_model, site_parameters):
        """
        Writes the sim_telarray config file for an array of telescopes.

        Parameters
        ----------
        config_file_path: str or Path
            Path of the file to write on.
        layout: LayoutArray
            Instance of LayoutArray referent to the array model.
        telescope_model: list of TelescopeModel
            List of TelescopeModel's instances as used by the ArrayModel instance.
        site_parameters: dict
            Site parameters.
        """
        with open(config_file_path, "w") as file:
            self._write_header(file, "ARRAY CONFIGURATION FILE")

            # Be careful with the formatting - simtel is sensitive
            file.write("#ifndef TELESCOPE\n")
            file.write("# define TELESCOPE 0\n")
            file.write("#endif\n\n")

            # TELESCOPE 0 - global parameters
            file.write("#if TELESCOPE == 0\n")
            file.write(self.TAB + "echo *****************************\n")
            file.write(self.TAB + "echo Site: {}\n".format(self._site))
            file.write(self.TAB + "echo LayoutName: {}\n".format(self._layout_name))
            file.write(self.TAB + "echo ModelVersion: {}\n".format(self._model_version))
            file.write(self.TAB + "echo *****************************\n\n")

            # Writing site parameters
            self._write_site_parameters(file, site_parameters)

            # Maximum telescopes
            file.write(self.TAB + "maximum_telescopes = {}\n\n".format(len(telescope_model)))

            # Default telescope - 0th tel in telescope list
            tel_config_file = telescope_model[0].get_config_file(no_export=True).name
            file.write("# include <{}>\n\n".format(tel_config_file))

            # Looping over telescopes - from 1 to ...
            for count, tel_model in enumerate(telescope_model):
                tel_config_file = tel_model.get_config_file(no_export=True).name
                file.write("%{}\n".format(layout[count].name))
                file.write("#elif TELESCOPE == {}\n\n".format(count + 1))
                file.write("# include <{}>\n\n".format(tel_config_file))
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
        __, __, diameter, flen, shape = mirrors.get_single_mirror_parameters(mirror_number)

        with open(single_mirror_list_file, "w") as file:
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
                "0. 0. {} {} {} 0.\n".format(
                    diameter, flen if not set_focal_length_to_zero else 0, shape
                )
            )

    def _write_header(self, file, title, comment_char="%"):
        """
        Writes a generic header. commen_char is the character to be used for comments, which \
        differs among ctypes of config files.
        """
        header = "{}{}\n".format(comment_char, 50 * "=")
        header += "{} {}\n".format(comment_char, title)
        header += "{} Site: {}\n".format(comment_char, self._site)
        header += "{} ModelVersion: {}\n".format(comment_char, self._model_version)
        header += (
            "{} TelescopeModelName: {}\n".format(comment_char, self._telescope_model_name)
            if self._telescope_model_name is not None
            else ""
        )
        header += (
            "{} LayoutName: {}\n".format(comment_char, self._layout_name)
            if self._layout_name is not None
            else ""
        )
        header += (
            "{} Label: {}\n".format(comment_char, self._label) if self._label is not None else ""
        )
        header += "{}{}\n".format(comment_char, 50 * "=")
        header += "{}\n".format(comment_char)
        file.write(header)

    def _write_site_parameters(self, file, site_parameters):
        """Writes site parameters."""
        file.write(self.TAB + "% Site parameters\n")
        for par in site_parameters:
            if par in self.PARS_NOT_TO_WRITE:
                continue
            value = site_parameters[par]["Value"]
            file.write(self.TAB + "{} = {}\n".format(par, value))
        file.write("\n")
