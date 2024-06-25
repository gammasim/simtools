#!/usr/bin/python3

import logging
import shutil
from copy import copy

import astropy.units as u

import simtools.utils.general as gen
from simtools.db import db_handler
from simtools.io_operations import io_handler
from simtools.simtel.simtel_config_writer import SimtelConfigWriter
from simtools.utils import names

__all__ = ["InvalidModelParameterError", "ModelParameter"]


class InvalidModelParameterError(Exception):
    """Exception for invalid model parameter."""


class ModelParameter:
    """
    Base class for model parameters.

    Provides methods to read and manipulate parameters from DB.

    Parameters
    ----------
    db: DatabaseHandler
        Database handler.
    model_version: str
        Version of the model (ex. prod5).
    site: str
        Site name (e.g., South or North).
    telescope_name: str
        Telescope name (e.g., LSTN-01, LSTN-design).
    collection: str
        instrument class (e.g. telescopes, calibration_devices)
        as stored under collection in the DB.
    mongo_db_config: dict
        MongoDB configuration.
    label: str
        Instance label. Important for output file naming.

    """

    def __init__(
        self,
        mongo_db_config,
        model_version,
        site=None,
        telescope_name=None,
        collection="telescopes",
        db=None,
        label=None,
    ):
        self._logger = logging.getLogger(__name__)
        self._extra_label = None
        self.io_handler = io_handler.IOHandler()
        self.db = (
            db if db is not None else db_handler.DatabaseHandler(mongo_db_config=mongo_db_config)
        )

        self._parameters = {}
        self._config_parameters = {}
        self._derived = None
        self.collection = collection
        self.label = label
        self.model_version = self.db.model_version(model_version)
        self.site = names.validate_site_name(site) if site is not None else None
        self.name = (
            names.validate_telescope_name(
                self.db.get_telescope_db_name(
                    telescope_name=telescope_name,
                    model_version=self.model_version,
                    collection=self.collection,
                )
            )
            if telescope_name is not None
            else None
        )
        self._config_file_directory = None
        self._config_file_path = None
        self._load_parameters_from_db()

        self.simtel_config_writer = None
        self._added_parameter_files = None
        self._is_config_file_up_to_date = False
        self._is_exported_model_files_up_to_date = False

    def _get_parameter_dict(self, par_name):
        """
        Get model parameter dictionary as stored in the DB.

        No conversion to values are applied for the use in simtools
        (e.g., no conversion from the string representation of lists
        to lists).

        Parameters
        ----------
        par_name: str
            Name of the parameter.

        Returns
        -------
        dict
            Dictionary with complete DB entry for the given parameter.

        Raises
        ------
        InvalidModelParameterError
            If par_name does not match any parameter in this model.
        """
        try:
            return self._parameters[par_name]
        except KeyError:
            pass
        try:
            return self.derived[par_name]
        except (KeyError, ValueError) as e:
            msg = f"Parameter {par_name} was not found in the model"
            self._logger.error(msg)
            raise InvalidModelParameterError(msg) from e

    def get_parameter_value(self, par_name, parameter_dict=None):
        """
        Get the value of a model parameter.

        List of values stored in strings are returns as lists, so that no knowledge
        of the database structure is needed when accessing the model parameters.

        Parameters
        ----------
        par_name: str
            Name of the parameter.
        parameter_dict: dict
            Dictionary with complete DB entry for the given parameter
            (including the 'value', 'units' fields).

        Returns
        -------
        Value of the parameter.

        Raises
        ------
        KeyError
            If par_name does not match any parameter in this model.
        """
        parameter_dict = parameter_dict if parameter_dict else self._get_parameter_dict(par_name)
        try:
            _parameter = parameter_dict["value"]
        except KeyError as exc:
            self._logger.error(f"Parameter {par_name} does not have a value")
            raise exc
        if isinstance(_parameter, str):
            _is_float = False
            try:
                _is_float = self.get_parameter_type(par_name).startswith("float")
            except (InvalidModelParameterError, TypeError):  # float - in case we don't know
                _is_float = True
            _parameter = gen.convert_string_to_list(_parameter, is_float=_is_float)
            _parameter = _parameter if len(_parameter) > 1 else _parameter[0]

        return _parameter

    def get_parameter_value_with_unit(self, par_name):
        """
        Get the value of an existing parameter of the model as an Astropy Quantity with its unit.

        If no unit is provided in the model, the value is returned without a unit.

        Parameters
        ----------
        par_name: str
            Name of the parameter.

        Returns
        -------
        Astropy quantity with the value of the parameter multiplied by its unit. If no unit is \
        provided in the model, the value is returned without a unit.

        """
        _parameter = self._get_parameter_dict(par_name)
        _value = self.get_parameter_value(par_name, _parameter)
        try:
            return _value * u.Unit(_parameter.get("unit"))
        except (KeyError, TypeError):
            return _value

    def get_parameter_type(self, par_name):
        """
        Get the type of existing parameter of the model (value of 'type' field of DB entry).

        Parameters
        ----------
        par_name: str
            Name of the parameter.

        Returns
        -------
        str or None
            type of the parameter (None if no type is defined)

        """
        parameter_dict = self._get_parameter_dict(par_name)
        try:
            return parameter_dict.get("type")
        except KeyError:
            self._logger.debug(f"Parameter {par_name} does not have a type")
        return None

    def get_parameter_file_flag(self, par_name):
        """
        Get value of parameter file flag of this database entry (boolean 'file' field of DB entry).

        Parameters
        ----------
        par_name: str
            Name of the parameter.

        Returns
        -------
        bool
            True if file flag is set.

        """
        parameter_dict = self._get_parameter_dict(par_name)
        try:
            if parameter_dict.get("file"):
                return True
        except KeyError:
            self._logger.debug(f"Parameter {par_name} does not have a file associated with it.")
        return False

    @property
    def derived(self):
        """Load the derived values and export them if the class instance hasn't done it yet."""
        if self._derived is None:
            self._load_derived_values()
            self._export_derived_files()
        return self._derived

    def _load_derived_values(self):
        """Load derived values from the DB."""
        self._logger.debug("Reading derived values from DB")
        self._derived = self.db.get_derived_values(
            self.site,
            self.name,
            self.model_version,
        )

    def _export_derived_files(self):
        """Write to disk a file from the derived values DB."""
        for par_now in self.derived.values():
            if par_now.get("File") or par_now.get("file"):
                self.db.export_file_db(
                    db_name=self.db.DB_DERIVED_VALUES,
                    dest=self.config_file_directory,
                    file_name=(par_now.get("value") or par_now.get("Value")),
                )

    def print_parameters(self):
        """Print parameters and their values for debugging purposes."""
        for par in self._parameters:
            print(f"{par} = {self.get_parameter_value(par)}")

    def _set_config_file_directory_and_name(self):
        """Set and create the directory and the name of the config file."""
        if self.name is None:
            return

        self._config_file_directory = self.io_handler.get_output_directory(
            label=(self.label or self.name), sub_dir="model"
        )

        # Setting file name and the location
        if self.site is not None and self.name is not None:
            config_file_name = names.simtel_config_file_name(
                self.site,
                self.model_version,
                telescope_model_name=self.name,
                label=self.label,
                extra_label=self._extra_label,
            )
            self._config_file_path = self.config_file_directory.joinpath(config_file_name)

        self._logger.debug(f"Config file path: {self._config_file_path}")

    def _load_parameters_from_db(self):
        """Read parameters from DB and store them in _parameters."""
        if self.db is None:
            return

        if self.name is not None:
            self._logger.debug(
                f"Reading array element parameters from DB "
                f"({self.name}, {self.model_version}, {self.site}, {self.collection})"
            )
            self._parameters = self.db.get_model_parameters(
                self.site, self.name, self.model_version, self.collection, only_applicable=True
            )
            try:
                self._config_parameters = self.db.get_sim_telarray_configuration_parameters(
                    self.site,
                    self.name,
                    self.model_version,
                )
            except ValueError:
                self._logger.warning(
                    f"No sim_telarray configuration parameters found for {self.site}, {self.name}"
                    f" (model version {self.model_version})."
                )

        if self.site is not None:
            self._logger.debug(f"Reading site parameters from DB ({self.site} site)")
            _site_pars = self.db.get_site_parameters(
                self.site, self.model_version, only_applicable=True
            )
            self._parameters.update(_site_pars)

    def set_extra_label(self, extra_label):
        """
        Set an extra label for the name of the config file.

        Notes
        -----
        The config file directory name is not affected by the extra label. Only the file name is \
        changed. This is important for the ArrayModel class to export multiple config files in the\
        same directory.

        Parameters
        ----------
        extra_label: str
            Extra label to be appended to the original label.
        """
        self._extra_label = extra_label
        self._set_config_file_directory_and_name()

    @property
    def extra_label(self):
        """Return the extra label if defined, if not return ''."""
        return self._extra_label if self._extra_label is not None else ""

    def get_simtel_parameters(self, parameters=None, telescope_model=True, site_model=True):
        """
        Get simtel parameters as name and value pairs.

        Parameters
        ----------
        parameters: dict
            Parameters (simtools) to be renamed (if necessary)
        telescope_model: bool
            If True, telescope model parameters are included.
        site_model: bool
            If True, site model parameters are included.

        Returns
        -------
        dict
            simtel parameters as dict (sorted by parameter names)

        """
        if parameters is None:
            parameters = self._parameters

        _simtel_parameter_value = {}
        for key in parameters:
            _par_name = names.get_simulation_software_name_from_parameter_name(
                key,
                simulation_software="sim_telarray",
                search_telescope_parameters=telescope_model,
                search_site_parameters=site_model,
            )
            if _par_name is not None:
                _simtel_parameter_value[_par_name] = parameters[key].get("value")
        return dict(sorted(_simtel_parameter_value.items()))

    def change_parameter(self, par_name, value):
        """
        Change the value of an existing parameter.

        This function does not modify the  DB, it affects only the current instance.

        Parameters
        ----------
        par_name: str
            Name of the parameter.
        value:
            Value of the parameter.

        Raises
        ------
        InvalidModelParameterError
            If the parameter to be changed does not exist in this model.
        """
        if par_name not in self._parameters:
            msg = f"Parameter {par_name} not in the model"
            self._logger.error(msg)
            raise InvalidModelParameterError(msg)

        if isinstance(value, str):
            value = gen.convert_string_to_list(value)

        if not gen.validate_data_type(
            reference_dtype=self.get_parameter_type(par_name),
            value=value,
            dtype=None,
            allow_subtypes=True,
        ):
            self._logger.error(
                f"Could not cast {value} of type {type(value)} "
                f"to {self.get_parameter_type(par_name)}."
            )
            raise ValueError

        self._logger.debug(
            f"Changing parameter {par_name} "
            f"from {self.get_parameter_value(par_name)} to {value}"
        )
        self._parameters[par_name]["value"] = value

        # In case parameter is a file, the model files will be outdated
        if self.get_parameter_file_flag(par_name):
            self._is_exported_model_files_up_to_date = False

        self._is_config_file_up_to_date = False

    def change_multiple_parameters(self, **kwargs):
        """
        Change the value of multiple existing parameters in the model.

        This function does not modify the DB, it affects only the current instance.

        Parameters
        ----------
        **kwargs
            Parameters should be passed as parameter_name=value.

        """
        for par, value in kwargs.items():
            if par in self._parameters:
                self.change_parameter(par, value)

        self._is_config_file_up_to_date = False

    def export_parameter_file(self, par_name, file_path):
        """
        Export a file to the config file directory.

        Parameters
        ----------
        par_name: str
            Name of the parameter.
        file_path: str
            Path of the file to be added to the config file directory.
        """
        if self._added_parameter_files is None:
            self._added_parameter_files = []
        self._added_parameter_files.append(par_name)
        shutil.copy(file_path, self.config_file_directory)

    def export_model_files(self):
        """Export the model files into the config file directory."""
        # Removing parameter files added manually (which are not in DB)
        pars_from_db = copy(self._parameters)
        if self._added_parameter_files is not None:
            for par in self._added_parameter_files:
                pars_from_db.pop(par)

        self.db.export_model_files(pars_from_db, self.config_file_directory)
        self._is_exported_model_files_up_to_date = True

    def export_config_file(self):
        """Export the config file used by sim_telarray."""
        # Exporting model file
        if not self._is_exported_model_files_up_to_date:
            self.export_model_files()

        # Using SimtelConfigWriter to write the config file.
        self._load_simtel_config_writer()
        self.simtel_config_writer.write_telescope_config_file(
            config_file_path=self.config_file_path,
            parameters=self.get_simtel_parameters(parameters=self._parameters),
            config_parameters=self.get_simtel_parameters(parameters=self._config_parameters),
        )

    @property
    def config_file_directory(self):
        """Directory for configure files. Configure, if necessary."""
        if self._config_file_directory is None:
            self._set_config_file_directory_and_name()
        return self._config_file_directory

    @property
    def config_file_path(self):
        """Path of the config file. Configure, if necessary."""
        if self._config_file_path is None:
            self._set_config_file_directory_and_name()
        return self._config_file_path

    def get_config_file(self, no_export=False):
        """
        Get the path of the config file for sim_telarray.

        The config file is produced if the file is not up to date.

        Parameters
        ----------
        no_export: bool
            Turn it on if you do not want the file to be exported.

        Returns
        -------
        Path
            Path of the exported config file for sim_telarray.
        """
        if not self._is_config_file_up_to_date and not no_export:
            self.export_config_file()
        return self.config_file_path

    def _load_simtel_config_writer(self):
        """Load the SimtelConfigWriter object."""
        if self.simtel_config_writer is None:
            self.simtel_config_writer = SimtelConfigWriter(
                site=self.site,
                telescope_model_name=self.name,
                model_version=self.model_version,
                label=self.label,
            )
