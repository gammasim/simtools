import logging

from simtools.data_model import data_reader
from simtools.io_operations import io_handler
from simtools.layout.array_layout import ArrayLayout
from simtools.model.site_model import SiteModel
from simtools.model.telescope_model import TelescopeModel
from simtools.simtel.simtel_config_writer import SimtelConfigWriter
from simtools.utils import general, names

__all__ = ["ArrayModel", "InvalidArrayConfigData"]


class InvalidArrayConfigData(Exception):
    """Exception for invalid array configuration data."""


class ArrayModel:
    """
    Representation of an observatory consisting of site and telescopes.

    Parameters
    ----------
    mongo_db_config: dict
        MongoDB configuration.
    model_version: str
        Version of the model (e.g., prod5).
    array_config_file: str
        Path to a yaml file with the array config data.
    array_config_data: dict
        Dict with the array config data.
    label: str
        Instance label. Used for output file naming.
    """

    def __init__(
        self,
        mongo_db_config,
        model_version,
        label=None,
        array_config_file=None,
        array_config_data=None,
    ):
        """
        Initialize ArrayModel.
        """
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init ArrayModel")
        self.mongo_db_config = mongo_db_config
        self.model_version = model_version
        self.label = label
        self.layout = None
        self.layout_name = None
        self._config_file_path = None
        self.io_handler = io_handler.IOHandler()

        self._array_config_data, site, telescopes = self._load_array_data(
            general.collect_data_from_file_or_dict(array_config_file, array_config_data)
        )
        self._set_config_file_directory()

        self.site_model, self.telescope_model = self._build_array_model(
            site=site,
            telescopes=telescopes,
        )

        self._telescope_model_files_exported = False
        self._array_model_file_exported = False

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

    def _load_array_data(self, array_config_data):
        """
        Load parameters from array configuration file.

        Parameters
        ----------
        array_config_data: dict
            Dict with the array config data.

        Returns
        -------
        dict
            Dict with updated array configuration data.
        str
            Site name.
        dict
            Dict with telescope positions from file
            (if configured in the array config data).
        """

        self._validate_array_data(array_config_data)
        site = names.validate_site_name(array_config_data["site"])

        if array_config_data.get("layout_name") is not None:
            telescope_positions = self._load_telescope_positions_from_file(
                array_config_data["layout_name"], site
            )

        self.layout_name = names.validate_array_layout_name(array_config_data["layout_name"])
        self.layout = ArrayLayout.from_array_layout_name(
            mongo_db_config=self.mongo_db_config,
            array_layout_name=site + "-" + self.layout_name,
            model_version=self.model_version,
            label=self.label,
        )

        # Removing keys that were stored in attributes and keeping the remaining as a dict
        return (
            {
                k: v
                for (k, v) in array_config_data.items()
                if k not in ["site", "layout_name", "model_version"]
            },
            site,
            telescope_positions,
        )

    def _validate_array_data(self, array_config_data):
        """
        Validate array_data by checking the existence of the relevant keys.

        Parameters
        ----------
        array_config_data: dict
            Dict with the array config data.

        Raises
        ------
        InvalidArrayConfigData
            If the array configuration data is invalid.

        """

        def run_over_pars(pars, data, parent=None):
            """Run over pars and validate it."""
            all_keys = data.keys() if parent is None else data[parent].keys()
            for pp in pars:
                if pp not in all_keys:
                    key = pp if parent is None else parent + "." + pp
                    msg = (
                        f"Key {key} was not found in array_config_data "
                        + "- impossible to build array model"
                    )
                    self._logger.error(msg)
                    raise InvalidArrayConfigData(msg)

        run_over_pars(["site", "layout_name"], array_config_data)

    def _set_config_file_directory(self):
        """
        Define and create config file directory.

        """
        self._config_file_directory = self.io_handler.get_output_directory(self.label, "model")
        if not self._config_file_directory.exists():
            self._config_file_directory.mkdir(parents=True, exist_ok=True)
            self._logger.info(f"Creating directory {self._config_file_directory}")

    def _build_array_model(self, site, telescopes):
        """
        Build the constituents of the array model (site, telescopes, etc).
        Includes reading of all model parameters from the DB.
        The array is define in the telescopes dictionary. Positions
        are read from the database if no values are given in this dictionary.

        Parameters
        ----------
        site: str
            Site name.
        telescopes: dict
            Dict with telescopes forming this array (values optional).

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
        for tel_name, position in telescopes.items():
            telescope_model[tel_name] = TelescopeModel(
                site=site_model.site,
                telescope_model_name=tel_name,
                model_version=self.model_version,
                mongo_db_config=self.mongo_db_config,
                label=self.label,
            )
            # Collecting parameters to change from array_config_data
            pars_to_change = self._get_single_telescope_info_from_array_config(tel_name)
            if len(position) > 0:
                pars_to_change[position["parameter"]] = position
            if len(pars_to_change) > 0:
                self._logger.debug(
                    f"Changing {len(pars_to_change)} parameters of "
                    f"{tel_name}: "
                    f"{*pars_to_change, }, ..."
                )
                telescope_model[tel_name].change_multiple_parameters(**pars_to_change)
                telescope_model[tel_name].set_extra_label(tel_name)
                # TODO TMP add position manually (not in DB yet)
                # pylint: disable=protected-access
                telescope_model[tel_name]._parameters[position["parameter"]] = position

        return site_model, telescope_model

    def _get_single_telescope_info_from_array_config(self, tel_name):
        """
        array_config_data contains the default telescope models for each telescope type and the \
        list of specific telescopes. For each case, the data can be given only as a name or as a \
        dict with 'name' and parameters to change. This function has to identify these two cases\
        and collect the telescope name and the dict with the parameters to change.

        Parameters
        ----------
        tel_name: str
            Name of the telescope at the layout level (LSTN-01, MSTN-05, ...).
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

        if tel_name in self._array_config_data.keys():
            return _process_single_telescope(self._array_config_data[tel_name])
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
            layout=self.layout,
            telescope_model=list(self.telescope_model.values()),
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

    def _load_telescope_positions_from_file(self, layout_name, site):
        """
        Load telescope positions from a file.

        Parameters
        ----------
        layout_name: str
            Name of the layout.
        site: str
            Site name.

        Returns
        -------
        dict
            Dict with telescope positions.

        """

        array_layout_name = (
            names.validate_site_name(site) + "-" + names.validate_array_layout_name(layout_name)
        )
        telescope_list_file = io_handler.IOHandler().get_input_data_file(
            "layout", f"telescope_positions-{array_layout_name}.ecsv"
        )
        table = data_reader.read_table_from_file(file_name=telescope_list_file)

        telescope_positions = {}
        for row in table:
            telescope_name = row["telescope_name"]
            telescope_positions[telescope_name] = self._get_telescope_position_parameter(
                telescope_name, site, row["position_x"], row["position_y"], row["position_z"]
            )
        return telescope_positions

    def _get_telescope_position_parameter(self, telescope_name, site, x, y, z):
        """
        Return dictionary with telescope position parameters.


        Parameters
        ----------
        telescope_name: str
            Name of the telescope.
        site: str
            Site name.
        x: astropy.Quantity
            X position.
        y: astropy.Quantity
            Y position.
        z: astropy.Quantity
            Z position.

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
            "application": True,
            "file:": False,
        }
