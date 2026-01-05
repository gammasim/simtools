"""Array model represents an observatory consisting of site, telescopes, and further devices."""

import logging
from pathlib import Path

import astropy.units as u
from astropy.table import QTable

from simtools.data_model import data_reader, schema
from simtools.io import io_handler
from simtools.model.calibration_model import CalibrationModel
from simtools.model.site_model import SiteModel
from simtools.model.telescope_model import TelescopeModel
from simtools.simtel.simtel_config_writer import SimtelConfigWriter
from simtools.utils import general, names


class ArrayModel:
    """
    Representation of an observatory consisting of site, telescopes, and further devices.

    Parameters
    ----------
    model_version: str
        Model version.
    label: str, optional
        Instance label. Used for output file naming.
    site: str, optional
        Site name.
    layout_name: str, optional
        Layout name.
    array_elements: Union[str, Path, List[str]], optional
        Array element definitions (list of array element or path to file with
        the array element positions).
    calibration_device_types: List[str], optional
        List of calibration device types (e.g., 'flat_fielding') attached to each telescope.
    overwrite_model_parameters: str, optional
        File name to overwrite model parameters from DB with provided values.
    """

    def __init__(
        self,
        model_version,
        label=None,
        site=None,
        layout_name=None,
        array_elements=None,
        calibration_device_types=None,
        overwrite_model_parameters=None,
    ):
        """Initialize ArrayModel."""
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init ArrayModel")
        self.model_version = model_version
        self.label = label
        self.layout_name = (
            layout_name[0]
            if isinstance(layout_name, list) and len(layout_name) == 1
            else layout_name
        )
        self._config_file_path = None
        self._config_file_directory = None
        self.io_handler = io_handler.IOHandler()

        self.overwrite_model_parameters = overwrite_model_parameters

        self.array_elements, self.site_model, self.telescope_models, self.calibration_models = (
            self._initialize(site, array_elements, calibration_device_types)
        )

        self._telescope_model_files_exported = False
        self._array_model_file_exported = False
        self._sim_telarray_seeds = None

    def _initialize(self, site, array_elements_config, calibration_device_types):
        """
        Initialize ArrayModel taking different configuration options into account.

        Parameters
        ----------
        site: str
            Site name.
        array_elements_config: Union[str, Path, List[str]]
            Array element definitions.
        calibration_device_types: str
            Calibration device types.

        Returns
        -------
        dict
            Dict with telescope positions.
        SiteModel
            Site model.
        dict
            Dict with telescope models.
        """
        self._logger.debug(f"Getting site parameters from DB ({site})")
        site_model = SiteModel(
            site=names.validate_site_name(site),
            model_version=self.model_version,
            label=self.label,
            overwrite_model_parameters=self.overwrite_model_parameters,
        )

        # Case 1: array_elements is a file name
        if isinstance(array_elements_config, str | Path):
            array_elements = self._load_array_element_positions_from_file(
                array_elements_config, site
            )
        # Case 2: array elements is a list of elements
        elif isinstance(array_elements_config, list):
            array_elements = self._get_array_elements_from_list(array_elements_config, site_model)
        # Case 3: array elements defined in DB by array layout name
        elif self.layout_name is not None:
            array_elements = self._get_array_elements_from_list(
                site_model.get_array_elements_for_layout(self.layout_name)
            )
        else:
            raise ValueError(
                "No array elements found. "
                "Possibly missing valid layout name or missing telescope list."
            )

        telescope_models, calibration_models = self._build_telescope_models(
            site_model, array_elements, calibration_device_types
        )

        return array_elements, site_model, telescope_models, calibration_models

    @property
    def sim_telarray_seeds(self):
        """
        Return sim_telarray seeds.

        Returns
        -------
        dict
            Dictionary with sim_telarray seeds.
        """
        return self._sim_telarray_seeds

    @sim_telarray_seeds.setter
    def sim_telarray_seeds(self, value):
        """
        Set sim_telarray seeds.

        Parameters
        ----------
        value: dict
            Dictionary with sim_telarray seeds.
        """
        if isinstance(value, dict):
            required_keys = {
                "seed",
                "random_instrument_instances",
                "seed_file_name",
            }
            if not required_keys.issubset(value):
                raise ValueError(
                    "sim_telarray_seeds dictionary must contain the following keys: "
                    f"{required_keys}"
                )
        self._sim_telarray_seeds = value

    @property
    def config_file_path(self):
        """
        Return the path of the array config file for sim_telarray.

        Returns
        -------
        Path
            Path of the exported config file for sim_telarray.
        """
        if self._config_file_path is None:
            config_file_name = names.simtel_config_file_name(
                array_name=self.layout_name,
                site=self.site_model.site,
                label=self.label,
            )
            self._config_file_path = self.get_config_directory().joinpath(config_file_name)
        return self._config_file_path

    @property
    def number_of_telescopes(self):
        """
        Return the number of telescopes.

        Returns
        -------
        int
            Number of telescopes.
        """
        return len(self.telescope_models)

    @property
    def site(self):
        """
        Return site.

        Returns
        -------
        str
            Site name.
        """
        return self.site_model.site

    def _build_telescope_models(self, site_model, array_elements, calibration_device_types):
        """
        Build telescope models for all telescopes of this array.

        Adds calibration device models, if requested through the calibration_device_types argument.
        Calibration device models are stored in a dictionary with the telescope name as key (to
        identify the calibration device model on a given telescope).

        Includes reading of telescope model parameters from the database.
        The array is defined in the array_elements dictionary. Array element positions
        are read from the database if no values are given in this dictionary.

        Parameters
        ----------
        site_model: SiteModel
            Site model.
        array_elements: dict
            Dict with array elements.
        calibration_device_types: List[str]
            List of calibration device types (e.g., 'flat_fielding')

        Returns
        -------
        dict, dict
            Dictionaries with telescope and calibration models.
        """
        telescope_models, calibration_models = {}, {}

        for element_name in array_elements:
            if names.get_collection_name_from_array_element_name(element_name) != "telescopes":
                continue

            telescope_models[element_name] = TelescopeModel(
                site=site_model.site,
                telescope_name=element_name,
                model_version=self.model_version,
                label=self.label,
                overwrite_model_parameters=self.overwrite_model_parameters,
            )
            calibration_models[element_name] = self._build_calibration_models(
                telescope_models[element_name],
                site_model,
                calibration_device_types,
            )

        return telescope_models, calibration_models

    def _build_calibration_models(self, telescope_model, site_model, calibration_device_types):
        """
        Build calibration device models for all telescopes in the array.

        A telescope can have multiple calibration devices of different types.

        Returns
        -------
        dict
            Dict with calibration device models.
        """
        calibration_models = {}
        for calibration_device_type in calibration_device_types or []:
            device_name = telescope_model.get_calibration_device_name(calibration_device_type)
            if device_name is None:
                continue

            calibration_models[device_name] = CalibrationModel(
                site=site_model.site,
                calibration_device_model_name=device_name,
                model_version=self.model_version,
                label=self.label,
                overwrite_model_parameters=self.overwrite_model_parameters,
            )
        return calibration_models

    def print_telescope_list(self):
        """Print list of telescopes."""
        for tel_name, data in self.telescope_models.items():
            print(f"Name: {tel_name}\t Model: {data.name}")

    def export_simtel_telescope_config_files(self):
        """Export sim_telarray configuration files for all telescopes into the model directory."""
        exported_models = []
        for tel_model in self.telescope_models.values():
            name = tel_model.name
            if name not in exported_models:
                self._logger.debug(f"Exporting configuration file for telescope {name}")
                tel_model.write_sim_telarray_config_file(
                    additional_models=self.calibration_models.get(tel_model.name)
                )
                exported_models.append(name)
            else:
                self._logger.debug(
                    f"Configuration file for telescope {name} already exists - skipping"
                )

        self._telescope_model_files_exported = True

    def export_sim_telarray_config_file(self):
        """Export sim_telarray configuration file for the array into the model directory."""
        self.site_model.export_model_files()

        self._logger.info(f"Writing array configuration file into {self.config_file_path}")
        simtel_writer = SimtelConfigWriter(
            site=self.site_model.site,
            layout_name=self.layout_name,
            model_version=self.model_version,
            label=self.label,
        )
        simtel_writer.write_array_config_file(
            config_file_path=self.config_file_path,
            telescope_model=self.telescope_models,
            site_model=self.site_model,
            additional_metadata=self._get_additional_simtel_metadata(),
        )
        self._array_model_file_exported = True

    def export_all_simtel_config_files(self):
        """
        Export sim_telarray config file for the array and for each individual telescope.

        Config files are exported into the output model directory.
        """
        if not self._telescope_model_files_exported:
            self.export_simtel_telescope_config_files()
        if not self._array_model_file_exported:
            self.export_sim_telarray_config_file()

    def get_config_directory(self):
        """
        Get the path of the array config directory for sim_telarray.

        Returns
        -------
        Path
            Path of the config directory path for sim_telarray.
        """
        if self._config_file_directory is None:
            self._config_file_directory = self.io_handler.get_model_configuration_directory(
                model_version=self.model_version
            )
        return self._config_file_directory

    def pack_model_files(self):
        """
        Pack all model files into a tar.gz archive.

        Returns
        -------
        Path
            Path of the packed model files archive.
        """
        model_files = list(Path(self.get_config_directory()).rglob("*"))
        if not model_files:
            self._logger.warning("No model files found to pack.")
            return None

        archive_name = self.get_config_directory() / f"model_files_{self.model_version}.tar.gz"
        general.pack_tar_file(archive_name, model_files, sub_dir=f"model/{self.model_version}")
        self._logger.info(f"Packed model files into {archive_name}")
        return archive_name

    def _load_array_element_positions_from_file(self, array_elements_file, site):
        """
        Load array element (e.g. telescope) positions from a file into a dict.

        Dictionary format: {telescope_name: {position_x: x, position_y: y, position_z: z}}

        Parameters
        ----------
        array_elements_file: Union[str, Path]
            Path to the file with the array element positions.
        site: str
            Site name.

        Returns
        -------
        dict
            Dict with telescope positions.
        """
        table = data_reader.read_table_from_file(file_name=array_elements_file)

        return {
            row["telescope_name"]: self._get_telescope_position_parameter(
                row["telescope_name"], site, row["position_x"], row["position_y"], row["position_z"]
            )
            for row in table
        }

    def _get_telescope_position_parameter(
        self, telescope_name, site, x, y, z, parameter_version=None
    ):
        """
        Return dictionary with telescope position parameters (following DB model database format).

        Parameters
        ----------
        telescope_name: str
            Name of the telescope.
        site: str
            Site name.
        x: astropy.Quantity
            X ground position.
        y: astropy.Quantity
            Y ground position.
        z: astropy.Quantity
            Z ground position.

        Returns
        -------
        dict
            Dict with telescope position parameters.
        """
        return {
            "schema_version": schema.get_model_parameter_schema_version(),
            "parameter": "array_element_position_ground",
            "instrument": telescope_name,
            "site": site,
            "parameter_version": parameter_version,
            "unique_id": None,
            "value": [x.to("m").value, y.to("m").value, z.to("m").value],
            "unit": "m",
            "type": "float64",
            "file": False,
            "meta_parameter": False,
            "model_parameter_schema_version": "0.1.0",
        }

    def _get_array_elements_from_list(self, array_elements_list, site_model=None):
        """
        Return dictionary with array elements from a list of telescope names.

        Input list can contain telescope names (e.g, LSTN-01) or a telescope
        type (e.g., MSTN). In the latter case, all telescopes of this specific
        type are added.

        Parameters
        ----------
        array_elements_list: list
            List of telescope names.
        site_model: SiteModel
            Site model.

        Returns
        -------
        dict
            Dict with array elements.
        """
        array_elements_dict = {}
        for name in array_elements_list:
            try:
                array_elements_dict[names.validate_array_element_name(name)] = None
            except ValueError:
                array_elements_dict.update(self._get_all_array_elements_of_type(name, site_model))
        return array_elements_dict

    def _get_all_array_elements_of_type(self, array_element_type, site_model):
        """
        Return all array elements of a specific type.

        Parameters
        ----------
        array_element_type : str
            Type of the array element (e.g. LSTN, MSTS)
        site_model: SiteModel
            Site model.

        Returns
        -------
        dict
            Dict with array elements.
        """
        return self._get_array_elements_from_list(
            site_model.get_array_elements_of_type(array_element_type)
        )

    def export_array_elements_as_table(self, coordinate_system="ground"):
        """
        Export array elements positions to astropy table.

        Parameters
        ----------
        coordinate_system: str
            Positions are exported in this coordinate system.

        Returns
        -------
        astropy.table.QTable
            Astropy table with the telescope layout information.
        """
        table = QTable(meta={"array_name": self.layout_name, "site": self.site_model.site})

        name, pos_x, pos_y, pos_z, tel_r = [], [], [], [], []
        for tel_name, data in self.telescope_models.items():
            name.append(tel_name)
            xyz = data.position(coordinate_system=coordinate_system)
            pos_x.append(xyz[0])
            pos_y.append(xyz[1])
            pos_z.append(xyz[2])
            try:
                # add tests of KeyError after positions calibration_elements are added to DB
                tel_r.append(data.get_parameter_value_with_unit("telescope_sphere_radius"))
            except KeyError:  # not all array elements have a sphere radius
                tel_r.append(0.0 * u.m)

        table["telescope_name"] = name
        if coordinate_system == "ground":
            table["position_x"] = pos_x
            table["position_y"] = pos_y
            table["position_z"] = pos_z
        elif coordinate_system == "utm":
            table["utm_east"] = pos_x
            table["utm_north"] = pos_y
            table["altitude"] = pos_z
        table["sphere_radius"] = tel_r

        table.sort("telescope_name")
        return table

    def _get_additional_simtel_metadata(self):
        """
        Collect additional metadata to be included in sim_telarray output.

        Returns
        -------
        dict
            Dictionary with additional metadata.
        """
        metadata = {}
        if self.sim_telarray_seeds is not None:
            metadata.update(self.sim_telarray_seeds)

        metadata["nsb_integrated_flux"] = self.site_model.get_nsb_integrated_flux()

        return metadata
