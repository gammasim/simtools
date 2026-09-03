"""MC model of a telescope."""

import logging
from copy import deepcopy

import numpy as np
from astropy.table import Table

import simtools.utils.general as gen
from simtools.data_model import schema
from simtools.model.camera import Camera
from simtools.model.mirrors import Mirrors
from simtools.model.model_parameter import InvalidModelParameterError, ModelParameter
from simtools.utils import names


class TelescopeModel(ModelParameter):
    """
    TelescopeModel represents the MC model of an individual telescope.

    TelescopeModel contains parameter names and values for a specific telescope model.

    Parameters
    ----------
    site: str
        Site name (e.g., South or North).
    telescope_name: str
        Telescope name (ex. LSTN-01, LSTN-design, ...).
    model_version: str
        Model version.
    label: str, optional
        Instance label.
    overwrite_model_parameter_dict: dict, optional
        Dictionary to overwrite model parameters from DB with provided values.
    ignore_software_version: bool, optional
        If True, ignore software version checks for deprecated parameters.
    """

    def __init__(
        self,
        site,
        telescope_name,
        model_version,
        label=None,
        overwrite_model_parameter_dict=None,
        ignore_software_version=False,
        model_reader=None,
    ):
        """Initialize TelescopeModel."""
        super().__init__(
            site=site,
            array_element_name=telescope_name,
            model_version=model_version,
            label=label,
            overwrite_model_parameter_dict=overwrite_model_parameter_dict,
            ignore_software_version=ignore_software_version,
            model_reader=model_reader,
        )

        self._logger = logging.getLogger(__name__)

        self._single_mirror_list_file_paths = None
        self._mirrors = None
        self._camera = None

    @property
    def mirrors(self):
        """Load the mirror information if the class instance hasn't done it yet."""
        if self._mirrors is None:
            self._load_mirrors()
        return self._mirrors

    @property
    def camera(self):
        """Load the camera information if the class instance hasn't done it yet."""
        if self._camera is None:
            self._load_camera()
        return self._camera

    def export_single_mirror_list_file(self, mirror_number: int, set_focal_length_to_zero: bool):
        """
        Export a mirror list file with a single mirror in it.

        Parameters
        ----------
        mirror_number: int
            Number index of the mirror.
        set_focal_length_to_zero: bool
            Set the focal length to zero if True.
        """
        if mirror_number > self.mirrors.number_of_mirrors:
            logging.error("mirror_number > number_of_mirrors")
            return

        file_name = names.simtel_single_mirror_list_file_name(
            self.site, self.name, self.model_version, mirror_number, self.label
        )
        if self._single_mirror_list_file_paths is None:
            self._single_mirror_list_file_paths = {}
        self._single_mirror_list_file_paths[mirror_number] = self.config_file_directory.joinpath(
            file_name
        )

        # Using SimtelConfigWriter
        self._load_simtel_config_writer()
        self.simtel_config_writer.write_single_mirror_list_file(
            mirror_number,
            self.mirrors,
            self._single_mirror_list_file_paths[mirror_number],
            set_focal_length_to_zero,
        )

    def get_single_mirror_list_file(
        self, mirror_number: int, set_focal_length_to_zero: bool = False
    ):
        """
        Get the path to the single mirror list file.

        Parameters
        ----------
        mirror_number: int
            Mirror number.
        set_focal_length_to_zero: bool
            Flag to set the focal length to zero.

        Returns
        -------
        Path
            Path of the single mirror list file.
        """
        self.export_single_mirror_list_file(mirror_number, set_focal_length_to_zero)
        return self._single_mirror_list_file_paths[mirror_number]

    def _load_mirrors(self):
        """Load the attribute mirrors by creating a Mirrors object with the mirror list file."""
        mirror_list_file_name = self.get_parameter_value("mirror_list")
        self._logger.debug(f"Reading mirror list from {mirror_list_file_name}")
        try:
            mirror_list_file = gen.find_file(mirror_list_file_name, self.config_file_directory)
        except FileNotFoundError:
            mirror_list_file = gen.find_file(mirror_list_file_name, self.io_handler.model_path)
            self._logger.warning(
                "Mirror_list_file was not found in the config directory - "
                "Using the one found in the model_path"
            )
        except TypeError as exc:
            raise TypeError("Undefined mirror list") from exc

        self._mirrors = Mirrors(mirror_list_file, parameters=self.parameters)

    def get_telescope_effective_focal_length(
        self, unit: str = "m", return_focal_length_if_zero: bool = False
    ) -> float:
        """
        Return effective focal length.

        The function ensures backwards compatibility with older sim-telarray versions.

        Parameters
        ----------
        unit: str
            Unit of the effective focal length. Default is 'm'.
        return_focal_length_if_zero: bool
            If True, return the focal length if the effective focal length is 0.

        Returns
        -------
        float:
            Effective focal length.
        """
        try:
            eff_focal_length = self.get_parameter_value_with_unit("effective_focal_length")[0]
        except TypeError:
            eff_focal_length = self.get_parameter_value_with_unit("effective_focal_length")
        try:
            eff_focal_length = eff_focal_length.to(unit).value
        except AttributeError:
            eff_focal_length = 0.0
        if return_focal_length_if_zero and (
            eff_focal_length is None or np.isclose(eff_focal_length, 0.0)
        ):
            self._logger.warning("Using focal_length because effective_focal_length is 0")
            return self.get_parameter_value_with_unit("focal_length").to(unit).value
        return eff_focal_length

    def _load_camera(self):
        """Load the camera from its validated component parameters."""
        configuration = self._resolve_camera_components()
        focal_length = self.get_telescope_effective_focal_length("cm", True)
        self._camera = Camera.from_configuration(self.name, configuration, focal_length)

    def _resolve_camera_components(self):
        """Resolve the independent camera component parameters."""
        pixel_types = deepcopy(self.get_parameter_value("camera_pixel_types"))
        for pixel_type in pixel_types:
            wavelength_parameter = pixel_type.get("lightguide_wavelength_parameter")
            if wavelength_parameter:
                pixel_type["lightguide_wavelength_file"] = f"{wavelength_parameter}-{self.name}.dat"

        return {
            "rotate": self.get_parameter_value("camera_rotate"),
            "pixel_types": pixel_types,
            "pixels": self._parameter_table_records("camera_pixel_layout"),
            "triggers": self._parameter_table_records("camera_trigger_groups"),
            "trigger_members": self._parameter_table_records("camera_trigger_members"),
        }

    def _parameter_table_records(self, parameter_name):
        """Return scalar records from a validated model-parameter table."""
        table = self.get_parameter_table(parameter_name)
        return [
            {column: getattr(row[column], "value", row[column]) for column in table.colnames}
            for row in table
        ]

    def is_file_2d(self, par: str) -> bool:
        """
        Check if the file referenced by par is a 2D table.

        Parameters
        ----------
        par: str
            Name of the parameter.

        Returns
        -------
        bool:
            True if the file is a 2D map type.
        """
        try:
            self.get_parameter_table(par)
        except InvalidModelParameterError:
            logging.warning(f"Parameter {par} does not exist")
            return False

        parameter_schema = schema.get_model_parameter_schema(par)
        return any(
            software.get("name") == "sim_telarray" and software.get("table_format") == "rpol_matrix"
            for software in parameter_schema.get("simulation_software", [])
        )

    def read_two_dim_wavelength_angle(self, parameter_name: str) -> dict:
        """
        Read a two dimensional distribution of wavelength and angle (z-axis can be anything).

        Return a dictionary with three arrays, wavelength, angles, z (can be transmission,
        reflectivity, etc.)

        Parameters
        ----------
        parameter_name: str
            Name of the ECSV model parameter.

        Returns
        -------
        dict:
            dict of three arrays, wavelength, degrees, z.
        """
        table = self.get_parameter_table(parameter_name)
        angle_name = "angle" if "angle" in table.colnames else "incidence_angle"
        value_columns = [name for name in table.colnames if name not in {"wavelength", angle_name}]
        if len(value_columns) != 1:
            raise ValueError(
                f"Two-dimensional table '{parameter_name}' must have one value column, "
                f"found {value_columns}."
            )
        angles = np.unique(np.asarray(table[angle_name]))
        wavelengths = np.unique(np.asarray(table["wavelength"]))
        values = np.empty((len(angles), len(wavelengths)))
        for angle_index, angle in enumerate(angles):
            for wavelength_index, wavelength in enumerate(wavelengths):
                selection = table[
                    (np.asarray(table[angle_name]) == angle)
                    & (np.asarray(table["wavelength"]) == wavelength)
                ]
                if len(selection) != 1:
                    raise ValueError(
                        f"Two-dimensional table '{parameter_name}' is not a complete grid."
                    )
                values[angle_index, wavelength_index] = selection[value_columns[0]][0]
        return {"Wavelength": wavelengths, "Angle": angles, "z": values}

    def get_on_axis_eff_optical_area(self) -> float:
        """Return the on-axis effective optical area (derived previously for this telescope)."""
        ray_tracing_data = self.get_parameter_table("optics_properties")
        if not np.isclose(ray_tracing_data["Off-axis_angle"][0], 0):
            msg = (
                f"No value for the on-axis effective optical area exists."
                f" The minimum off-axis angle is {ray_tracing_data['Off-axis_angle'][0]}"
            )
            raise ValueError(msg)
        return ray_tracing_data["eff_area"][0]

    def read_incidence_angle_distribution(self, parameter_name: str) -> Table:
        """
        Read the incidence angle distribution from a file.

        Parameters
        ----------
        parameter_name: str
            Name of the incidence angle distribution parameter.

        Returns
        -------
        incidence_angle_dist: astropy.table.Table
            Instance of astropy.table.Table with the incidence angle distribution.
        """
        self._logger.debug(
            "Reading incidence angle distribution from %s",
            parameter_name,
        )
        return self.get_parameter_table(parameter_name)

    @staticmethod
    def calc_average_curve(curves: dict, incidence_angle_dist: Table) -> Table:
        """
        Calculate an average curve from a set of curves.

        The calculation uses weights the distribution of incidence angles provided in
        incidence_angle_dist.

        Parameters
        ----------
        curves: dict
            dict of with 3 "columns", Wavelength, Angle and z. The dictionary represents a two \
            dimensional distribution of wavelengths and angles with the z value being e.g., \
            reflectivity, transmission, etc.
        incidence_angle_dist: astropy.table.Table
            Instance of astropy.table.Table with the incidence angle distribution. The assumed \
            columns are "Incidence angle" and "Fraction".

        Returns
        -------
        average_curve: astropy.table.Table
            Instance of astropy.table.Table with the averaged curve.
        """
        weights = [
            incidence_angle_dist["Fraction"][
                np.nanargmin(np.abs(angle_now - incidence_angle_dist["Incidence angle"].value))
            ]
            for angle_now in curves["Angle"]
        ]

        return Table(
            [curves["Wavelength"], np.average(curves["z"], weights=weights, axis=0)],
            names=("Wavelength", "z"),
        )

    def export_table_to_model_directory(self, file_name: str, table: Table) -> str:
        """
        Write out a file with the provided table to the model directory.

        Parameters
        ----------
        file_name: str
            File name to write to.
        table: astropy.table.Table
            Instance of astropy.table.Table with the values to write to the file.

        Returns
        -------
        Path:
            Path to the file exported.
        """
        file_to_write_to = self.config_file_directory.joinpath(file_name)
        table.write(file_to_write_to, format="ascii.commented_header", overwrite=True)
        return file_to_write_to.absolute()

    def position(self, coordinate_system: str = "ground") -> list:
        """
        Get coordinates in the given system.

        Parameters
        ----------
        coordinate_system: str
            Coordinates system. Default is 'ground'.

        Returns
        -------
        list :
            List of telescope position in the requested coordinate system.

        Raises
        ------
        KeyError
            If the coordinate system is not found.
        """
        try:
            return self.get_parameter_value_with_unit(f"array_element_position_{coordinate_system}")
        except InvalidModelParameterError as exc:
            raise InvalidModelParameterError(
                f"Coordinate system {coordinate_system} not found."
            ) from exc

    def get_calibration_device_name(self, device_type):
        """
        Get the calibration device name for this telescope.

        Parameters
        ----------
        device_type: str
            Type of the calibration device (e.g., 'flasher', 'illuminator')

        Returns
        -------
        str or None
            Calibration device name or None if not defined.
        """
        try:
            devices = self.get_parameter_value("calibration_devices") or {}
        except InvalidModelParameterError:
            return None
        return devices.get(device_type)
