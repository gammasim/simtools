import logging

from simtools.data_model import data_reader
from simtools.io_operations import io_handler
from simtools.model.site_model import SiteModel
from simtools.model.telescope_model import TelescopeModel
from simtools.simtel.simtel_config_writer import SimtelConfigWriter
from simtools.utils import general, names

__all__ = ["ArrayModel", "InvalidArrayConfigData"]


class InvalidArrayConfigData(Exception):
    """Exception for invalid array configuration data."""


class ArrayModel:
    """
    Representation of an observatory consisting of site, telescopes, and further devices.

    Parameters
    ----------
    mongo_db_config: dict
        MongoDB configuration.
    model_version: str
        Version of the model (e.g., prod5).
    label: str
        Instance label. Used for output file naming.
    site: str
        Site name.
    layout_name: str
        Layout name.
    array_elements_file: str
        Path to the file with the array element positions.
    parameters_to_change: dict
        Dict with the parameters to be changed with respect to the DB model.
    """

    def __init__(
        self,
        mongo_db_config,
        model_version,
        label=None,
        site=None,
        layout_name=None,
        array_elements_file=None,
        parameters_to_change=None,
    ):
        """
        Initialize ArrayModel.
        """
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init ArrayModel")
        self.mongo_db_config = mongo_db_config
        self.model_version = model_version
        self.label = label
        self.layout_name = names.validate_array_layout_name(layout_name) if layout_name else None
        self._config_file_path = None
        self._config_file_directory = None
        self.io_handler = io_handler.IOHandler()

        self.array_elements, self.site_model, self.telescope_model = self._initialize(
            site, array_elements_file, parameters_to_change
        )

        self._telescope_model_files_exported = False
        self._array_model_file_exported = False

    def _initialize(self, site, array_elements_file, parameters_to_change):
        """
        Initialize ArrayModel taking different configuration options into account.

        Parameters
        ----------
        site: str
            Site name.
        array_elements_file: str
            Path to the file with the array element positions.
        parameters_to_change: dict
            Dict with the parameters to be changed with respect to the DB model.

        Returns
        -------
        dict
            Dict with telescope positions.
        SiteModel
            Site model.
        dict
            Dict with telescope models.

        """

        if self.layout_name is not None and array_elements_file is None:
            array_elements_file = io_handler.IOHandler().get_input_data_file(
                "layout",
                "telescope_positions-"
                f"{names.validate_site_name(site)}-"
                f"{names.validate_array_layout_name(self.layout_name)}"
                ".ecsv",
            )

        self.array_elements = (
            None
            if array_elements_file is None
            else self._load_array_element_positions_from_file(array_elements_file, site)
        )
        self._set_config_file_directory()
        site_model, telescope_model = self._build_array_model(
            names.validate_site_name(site), parameters_to_change
        )

        return self.array_elements, site_model, telescope_model

    @property
    def number_of_telescopes(self):
        """
        Return the number of telescopes.

        Returns
        -------
        int
            Number of telescopes.
        """
        return len(self.telescope_model)

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

    def _set_config_file_directory(self):
        """
        Define and create config file directory.

        """
        self._config_file_directory = self.io_handler.get_output_directory(self.label, "model")
        if not self._config_file_directory.exists():
            self._config_file_directory.mkdir(parents=True, exist_ok=True)
            self._logger.info(f"Creating directory {self._config_file_directory}")

    def _build_array_model(self, site, parameters_to_change=None):
        """
        Build the constituents of the array model (site, telescopes, etc).
        Includes reading of all model parameters from the DB.
        The array is define in the telescopes dictionary. Positions
        are read from the database if no values are given in this dictionary.

        Parameters
        ----------
        site: str
            Site name.
        parameters_to_change: dict
            Dict with the parameters to be changed with respect to the DB model.

        Returns
        -------
        SiteModel
            Site model.

        """

        self._logger.debug(f"Getting site parameters from DB ({site})")
        site_model = SiteModel(
            site=site,
            mongo_db_config=self.mongo_db_config,
            model_version=self.model_version,
            label=self.label,
        )

        telescope_model = {}
        # TODO - check if this is the correct way to define an array
        for element_name, _ in self.array_elements.items():
            collection = names.get_collection_name_from_array_element_name(element_name)
            if collection == "telescopes":
                telescope_model[element_name] = TelescopeModel(
                    site=site_model.site,
                    telescope_name=element_name,
                    model_version=self.model_version,
                    mongo_db_config=self.mongo_db_config,
                    label=self.label,
                )
            # Collecting parameters to change from array_config_data
            pars_to_change = self._get_single_telescope_info_from_array_config(
                element_name, parameters_to_change
            )
            if len(pars_to_change) > 0:
                self._logger.debug(
                    f"Changing {len(pars_to_change)} parameters of "
                    f"{element_name}: "
                    f"{*pars_to_change, }, ..."
                )
                if element_name in telescope_model:
                    telescope_model[element_name].change_multiple_parameters(**pars_to_change)
                    telescope_model[element_name].set_extra_label(element_name)

        return site_model, telescope_model

    def _get_single_telescope_info_from_array_config(self, tel_name, array_config_data):
        """
        array_config_data contains the default telescope models for each telescope type and the \
        list of specific telescopes. For each case, the data can be given only as a name or as a \
        dict with 'name' and parameters to change. This function has to identify these two cases\
        and collect the telescope name and the dict with the parameters to change.

        Parameters
        ----------
        tel_name: str
            Name of the telescope at the layout level (LSTN-01, MSTN-05, ...).
        array_config_data: dict
            Dict with the array config data.
        """

        def _process_single_telescope(data):
            """
            Parameters
            ----------
            data: dict or str
                Piece of the array_config_data for one specific telescope.
            """

            if isinstance(data, dict):
                # Case 0: data is dict
                if "name" not in data.keys():
                    msg = "ArrayConfig has no name for a telescope"
                    self._logger.error(msg)
                    raise InvalidArrayConfigData(msg)
                pars_to_change = {k: v for (k, v) in data.items() if k != "name"}
                self._logger.debug(
                    "Grabbing tel data as dict - " f"{len(pars_to_change)} pars to change"
                )
                return pars_to_change
            if isinstance(data, str):
                # Case 1: data is string (only name)
                return {}

            # Case 2: data has a wrong type
            msg = "ArrayConfig has wrong input for a telescope"
            self._logger.error(msg)
            raise InvalidArrayConfigData(msg)

        if array_config_data and tel_name in array_config_data.keys():
            return _process_single_telescope(array_config_data[tel_name])
        return {}

    def print_telescope_list(self):
        """
        Print list of telescopes

        """

        for tel_name, data in self.telescope_model.items():
            print(f"Name: {tel_name}\t Model: {data.name}")

    def export_simtel_telescope_config_files(self):
        """
        Export sim_telarray configuration files for all telescopes into the model directory.

        """

        exported_models = []
        for _, tel_model in self.telescope_model.items():
            name = tel_model.name + (
                "_" + tel_model.extra_label if tel_model.extra_label != "" else ""
            )
            if name not in exported_models:
                self._logger.debug(f"Exporting configuration file for telescope {name}")
                tel_model.export_config_file()
                exported_models.append(name)
            else:
                self._logger.debug(
                    f"Configuration file for telescope {name} already exists - skipping"
                )

        self._telescope_model_files_exported = True

    def export_simtel_array_config_file(self):
        """
        Export sim_telarray configuration file for the array into the model directory.

        """

        # Setting file name and the location
        config_file_name = names.simtel_config_file_name(
            array_name=self.layout_name,
            site=self.site_model.site,
            model_version=self.model_version,
            label=self.label,
        )
        self._config_file_path = self._config_file_directory.joinpath(config_file_name)

        # Writing parameters to the file
        self._logger.info(f"Writing array configuration file into {self._config_file_path}")
        simtel_writer = SimtelConfigWriter(
            site=self.site_model.site,
            layout_name=self.layout_name,
            model_version=self.model_version,
            label=self.label,
        )
        simtel_writer.write_array_config_file(
            config_file_path=self._config_file_path,
            telescope_model=self.telescope_model,
            site_model=self.site_model,
        )
        self._array_model_file_exported = True

    def export_all_simtel_config_files(self):
        """
        Export sim_telarray config file for the array and for each individual telescope into the \
        output model directory.
        """

        if not self._telescope_model_files_exported:
            self.export_simtel_telescope_config_files()
        if not self._array_model_file_exported:
            self.export_simtel_array_config_file()

    def get_config_file(self):
        """
        Get the path of the array config file for sim_telarray. The config file is produced if the \
        file is not updated.

        Returns
        -------
        Path
            Path of the exported config file for sim_telarray.
        """

        self.export_all_simtel_config_files()
        return self._config_file_path

    def get_config_directory(self):
        """
        Get the path of the array config directory for sim_telarray.

        Returns
        -------
        Path
            Path of the config directory path for sim_telarray.
        """
        return self._config_file_directory

    def _load_array_element_positions_from_file(self, array_elements_file, site):
        """
        Load telescope positions from a file.

        Parameters
        ----------
        array_elements_file: str
            Path to the file with the telescope positions.
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

    def _get_telescope_position_parameter(self, telescope_name, site, x, y, z):
        """
        Return dictionary with telescope position parameters (following DB model database format)

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
            "parameter": "array_element_position_ground",
            "instrument": telescope_name,
            "site": site,
            "version": self.model_version,
            "value": general.convert_list_to_string(
                [x.to("m").value, y.to("m").value, z.to("m").value]
            ),
            "unit": "m",
            "type": "float64",
            "applicable": True,
            "file": False,
        }
