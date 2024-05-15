import logging
from copy import copy

from simtools.io_operations import io_handler
from simtools.layout.array_layout import ArrayLayout
from simtools.model.site_model import SiteModel
from simtools.model.telescope_model import TelescopeModel
from simtools.simtel.simtel_config_writer import SimtelConfigWriter
from simtools.utils import names
from simtools.utils.general import collect_data_from_file_or_dict

__all__ = ["ArrayModel", "InvalidArrayConfigData"]


class InvalidArrayConfigData(Exception):
    """Exception for invalid array configuration data."""


class ArrayModel:
    """
    ArrayModel is an abstract representation of the MC model at the array level. It contains the\
    list of TelescopeModels, SiteModel, and a ArrayLayout.

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
        Instance label. Important for output file naming.
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
        self.label = label
        self.site = None
        self.layout = None
        self.layout_name = None
        self.model_version = model_version
        self._config_file_path = None
        self.io_handler = io_handler.IOHandler()
        self._load_array_data(collect_data_from_file_or_dict(array_config_file, array_config_data))
        self._set_config_file_directory()
        self._build_array_model()
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
        return self.layout.get_number_of_telescopes()

    def _load_array_data(self, array_config_data):
        """Load parameters from array_data.

        Parameters
        ----------
        array_config_data: dict
        """

        # Validating array_config_data
        # Keys 'site', 'layout_name' and 'default' are mandatory.
        # 'default' must have 'LST', 'MST' and 'SST' (for South site) keys.
        self._validate_array_data(array_config_data)

        self.site = names.validate_site_name(array_config_data["site"])

        self.layout_name = names.validate_array_layout_name(array_config_data["layout_name"])
        self.layout = ArrayLayout.from_array_layout_name(
            mongo_db_config=self.mongo_db_config,
            array_layout_name=self.site + "-" + self.layout_name,
            model_version=self.model_version,
            label=self.label,
        )

        # Removing keys that were stored in attributes and keeping the remaining as a dict
        self._array_config_data = {
            k: v
            for (k, v) in array_config_data.items()
            if k not in ["site", "layout_name", "model_version"]
        }

    def _validate_array_data(self, array_config_data):
        """
        Validate array_data by checking the existence of the relevant keys.

         Searching for the keys: 'site', 'array'
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
        """Define the variable _config_file_directory and create directories, if needed"""
        self._config_file_directory = self.io_handler.get_output_directory(self.label, "model")
        if not self._config_file_directory.exists():
            self._config_file_directory.mkdir(parents=True, exist_ok=True)
            self._logger.info(f"Creating directory {self._config_file_directory}")

    def _build_array_model(self):
        """
        Build the site parameters and the list of telescope models,
        including reading the parameters from the DB.

        """

        # Getting site parameters from DB
        self._logger.debug("Getting site parameters from DB")
        self._site_model = SiteModel(
            site=self.site,
            mongo_db_config=self.mongo_db_config,
            model_version=self.model_version,
            label=self.label,
        )

        # Building telescope models
        self._telescope_model = []  # List of telescope models
        _all_telescope_names = []  # List of telescope names without repetition
        _all_pars_to_change = {}
        for tel in self.layout:
            # Collecting pars to change from array_config_data
            tel_name = tel.name
            pars_to_change = self._get_single_telescope_info_from_array_config(tel_name)
            if len(pars_to_change) > 0:
                _all_pars_to_change[tel.name] = pars_to_change

            # Building the basic models - no pars to change yet
            if tel_name not in _all_telescope_names:
                # First time a telescope name is built
                _all_telescope_names.append(tel_name)
                tel_model = TelescopeModel(
                    site=self.site,
                    telescope_model_name=tel_name,
                    model_version=self.model_version,
                    mongo_db_config=self.mongo_db_config,
                    label=self.label,
                )
            else:
                # Telescope name already exists.
                # Finding the TelescopeModel and copying it.
                for tel_now in self._telescope_model:
                    if tel_now.name != tel_name:
                        continue
                    tel_model = copy(tel_now)
                    break

            self._telescope_model.append(tel_model)

        # Checking whether the size of the telescope list and the layout match
        if len(self._telescope_model) != len(self.layout):
            self._logger.warning(
                "Number of telescopes in the list of telescope models does "
                "not match the number of telescopes in the ArrayLayout - something is wrong!"
            )

        # Changing parameters, if there are any in all_pars_to_change
        if len(_all_pars_to_change) > 0:
            for tel_data, tel_model in zip(self.layout, self._telescope_model):
                if tel_data.name not in _all_pars_to_change:
                    continue
                self._logger.debug(
                    f"Changing {len(_all_pars_to_change[tel_data.name])} pars of a "
                    f"{tel_data.name}: {*_all_pars_to_change[tel_data.name], }, ..."
                )
                tel_model.change_multiple_parameters(**_all_pars_to_change[tel_data.name])
                tel_model.set_extra_label(tel_data.name)

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
        """Print out the list of telescopes for quick inspection."""

        for tel_data, tel_model in zip(self.layout, self._telescope_model):
            print(f"Name: {tel_data.name}\t Model: {tel_model.name}")

    def export_simtel_telescope_config_files(self):
        """
        Export sim_telarray config files for all the telescopes into the output model directory.
        """

        exported_models = []
        for tel_model in self._telescope_model:
            name = tel_model.name + (
                "_" + tel_model.extra_label if tel_model.extra_label != "" else ""
            )
            if name not in exported_models:
                self._logger.debug(f"Exporting config file for tel {name}")
                tel_model.export_config_file()
                exported_models.append(name)
            else:
                self._logger.debug(f"Config file for tel {name} already exists - skipping")

        self._telescope_model_files_exported = True

    def export_simtel_array_config_file(self):
        """
        Export sim_telarray config file for the array into the output model directory.
        """

        # Setting file name and the location
        config_file_name = names.simtel_config_file_name(
            array_name=self.layout_name,
            site=self.site,
            model_version=self.model_version,
            label=self.label,
        )
        self._config_file_path = self._config_file_directory.joinpath(config_file_name)

        # Writing parameters to the file
        self._logger.info(f"Writing array config file into {self._config_file_path}")
        simtel_writer = SimtelConfigWriter(
            site=self.site,
            layout_name=self.layout_name,
            model_version=self.model_version,
            label=self.label,
        )
        simtel_writer.write_array_config_file(
            config_file_path=self._config_file_path,
            layout=self.layout,
            telescope_model=self._telescope_model,
            site_model=self._site_model,
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
