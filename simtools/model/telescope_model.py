import logging

import astropy.io.ascii
import numpy as np
from astropy.table import Table

import simtools.utils.general as gen
from simtools.model.camera import Camera
from simtools.model.mirrors import Mirrors
from simtools.model.model_parameter import ModelParameter
from simtools.utils import names

__all__ = ["TelescopeModel"]


class TelescopeModel(ModelParameter):
    """
    TelescopeModel represents the MC model of an individual telescope. It contains the list of \
    parameters that can be read from the DB.

    Parameters
    ----------
    site: str
        Site name (e.g., South or North).
    telescope_model_name: str
        Telescope model name (ex. LSTN-01, LSTN-design, ...).
    mongo_db_config: dict
        MongoDB configuration.
    model_version: str
        Version of the model (ex. prod5).
    label: str
        Instance label. Important for output file naming.
    """

    def __init__(
        self,
        site,
        telescope_model_name,
        mongo_db_config,
        model_version,
        label=None,
    ):
        """
        Initialize TelescopeModel.
        """
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init TelescopeModel %s %s", site, telescope_model_name)
        ModelParameter.__init__(
            self,
            site=site,
            telescope_model_name=telescope_model_name,
            mongo_db_config=mongo_db_config,
            model_version=model_version,
            db=None,
            label=label,
        )

        self._single_mirror_list_file_paths = None
        self._mirrors = None
        self._camera = None

    @property
    def mirrors(self):
        """
        Load the mirror information if the class instance hasn't done it yet.
        """
        if self._mirrors is None:
            self._load_mirrors()
        return self._mirrors

    @property
    def camera(self):
        """
        Load the camera information if the class instance hasn't done it yet.
        """
        if self._camera is None:
            self._load_camera()
        return self._camera

    def export_single_mirror_list_file(self, mirror_number, set_focal_length_to_zero):
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
        self._single_mirror_list_file_paths[mirror_number] = self._config_file_directory.joinpath(
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

    def get_single_mirror_list_file(self, mirror_number, set_focal_length_to_zero=False):
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
        try:
            mirror_list_file = gen.find_file(mirror_list_file_name, self._config_file_directory)
        except FileNotFoundError:
            mirror_list_file = gen.find_file(mirror_list_file_name, self.io_handler.model_path)
            self._logger.warning(
                "Mirror_list_file was not found in the config directory - "
                "Using the one found in the model_path"
            )
        self._mirrors = Mirrors(mirror_list_file, parameters=self._parameters)

    def get_telescope_effective_focal_length(self):
        """
        Return effective focal length. Ensure backwards compatibility with older
        sim-telarray versions.

        Returns
        -------
        float:
            Effective focal length in 'm'.

        """
        try:
            return self.get_parameter_value_with_unit("effective_focal_length")[0].to("m").value
        except TypeError:
            return self.get_parameter_value_with_unit("effective_focal_length").to("m").value

    def _load_camera(self):
        """Loading camera attribute by creating a Camera object with the camera config file."""
        camera_config_file = self.get_parameter_value("camera_config_file")
        focal_length = 0.0
        try:
            focal_length = self.get_telescope_effective_focal_length()
        except IndexError:
            pass
        if focal_length == 0.0:
            self._logger.warning("Using focal_length because effective_focal_length is 0.")
            focal_length = self.get_parameter_value("focal_length")
        try:
            camera_config_file_path = gen.find_file(camera_config_file, self._config_file_directory)
        except FileNotFoundError:
            self._logger.warning(
                "The camera_config_file was not found in the config directory - "
                "Using the one found in the model_path"
            )
            camera_config_file_path = gen.find_file(camera_config_file, self.io_handler.model_path)

        self._camera = Camera(
            telescope_model_name=self.name,
            camera_config_file=camera_config_file_path,
            focal_length=focal_length,
        )

    def is_file_2d(self, par):
        """
        Check if the file referenced by par is a 2D table.

        Parameters
        ----------
        par: str
            Name of the parameter.

        Returns
        -------
        bool:
            True if the file is a 2D map type, False otherwise.
        """
        try:
            file_name = self.get_parameter_value(par)
        except KeyError:
            logging.error(f"Parameter {par} does not exist")
            return False

        file = self.config_file_directory.joinpath(file_name)
        with open(file, "r", encoding="utf-8") as f:
            is_2d = "@RPOL@" in f.read()
        return is_2d

    def read_two_dim_wavelength_angle(self, file_name):
        """
        Read a two dimensional distribution of wavelength and angle (z-axis can be anything).
        Return a dictionary with three arrays, wavelength, angles, z (can be transmission,
        reflectivity, etc.)

        Parameters
        ----------
        file_name: str or Path
            File assumed to be in the model directory.

        Returns
        -------
        dict:
            dict of three arrays, wavelength, degrees, z.
        """

        _file = self.config_file_directory.joinpath(file_name)
        self._logger.debug("Reading two dimensional distribution from %s", _file)
        line_to_start_from = 0
        with open(_file, "r", encoding="utf-8") as f:
            for i_line, line in enumerate(f):
                if line.startswith("ANGLE"):
                    degrees = np.array(line.strip().split("=")[1].split(), dtype=np.float16)
                    line_to_start_from = i_line + 1
                    break  # The rest can be read with np.loadtxt

        _data = np.loadtxt(_file, skiprows=line_to_start_from)

        return {
            "Wavelength": _data[:, 0],
            "Angle": degrees,
            "z": np.array(_data[:, 1:]).T,
        }

    def get_on_axis_eff_optical_area(self):
        """Return the on-axis effective optical area (derived previously for this telescope)."""

        ray_tracing_data = astropy.io.ascii.read(
            self.config_file_directory.joinpath(self.get_parameter_value("optics_properties"))
        )
        if not np.isclose(ray_tracing_data["Off-axis angle"][0], 0):
            self._logger.error(
                f"No value for the on-axis effective optical area exists."
                f" The minimum off-axis angle is {ray_tracing_data['Off-axis angle'][0]}"
            )
            raise ValueError
        return ray_tracing_data["eff_area"][0]

    def read_incidence_angle_distribution(self, incidence_angle_dist_file):
        """
        Read the incidence angle distribution from a file.

        Parameters
        ----------
        incidence_angle_dist_file: str
            File name of the incidence angle distribution

        Returns
        -------
        incidence_angle_dist: astropy.table.Table
            Instance of astropy.table.Table with the incidence angle distribution.
        """

        self._logger.debug(
            "Reading incidence angle distribution from %s",
            self.config_file_directory.joinpath(incidence_angle_dist_file),
        )
        incidence_angle_dist = astropy.io.ascii.read(
            self.config_file_directory.joinpath(incidence_angle_dist_file)
        )
        return incidence_angle_dist

    @staticmethod
    def calc_average_curve(curves, incidence_angle_dist):
        """
        Calculate an average curve from a set of curves, using as weights the distribution of \
        incidence angles provided in incidence_angle_dist.

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

        weights = []
        for angle_now in curves["Angle"]:
            weights.append(
                incidence_angle_dist["Fraction"][
                    np.nanargmin(np.abs(angle_now - incidence_angle_dist["Incidence angle"].value))
                ]
            )

        average_curve = Table(
            [curves["Wavelength"], np.average(curves["z"], weights=weights, axis=0)],
            names=("Wavelength", "z"),
        )

        return average_curve

    def export_table_to_model_directory(self, file_name, table):
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

        file_to_write_to = self._config_file_directory.joinpath(file_name)
        table.write(file_to_write_to, format="ascii.commented_header", overwrite=True)
        return file_to_write_to.absolute()
