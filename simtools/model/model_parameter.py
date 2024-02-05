#!/usr/bin/python3

import logging
import shutil
from copy import copy
from pydoc import locate

import astropy.units as u

from simtools.db import db_handler
from simtools.io_operations import io_handler
from simtools.simtel.simtel_config_writer import SimtelConfigWriter
from simtools.utils import names

__all__ = ["InvalidModelParameter", "ModelParameter"]


class InvalidModelParameter(Exception):
    """Exception for invalid model parameter."""


class ModelParameter:
    """
    Base class for model parameters.
    Provides methods to read and manipulate parameters from DB.

    Parameters
    ----------
    site: str
        Site name (e.g., South or North).
    telescope_name: str
        Telescope name (e.g., LSTN-01).
    mongo_db_config: dict
        MongoDB configuration.
    model_version: str
        Version of the model (ex. prod5).
    db: DatabaseHandler
        Database handler.
    label: str
        Instance label. Important for output file naming.

    """

    def __init__(
        self,
        site=None,
        telescope_name=None,
        mongo_db_config=None,
        model_version="Released",
        db=None,
        label=None,
    ):
        self._logger = logging.getLogger(__name__)
        self._extra_label = None
        self.io_handler = io_handler.IOHandler()
        self.db = db
        if mongo_db_config is not None:
            self.db = db_handler.DatabaseHandler(mongo_db_config=mongo_db_config)

        self._parameters = {}
        self._derived = None
        self.site = names.validate_site_name(site) if site is not None else None
        self.name = (
            names.validate_telescope_name(telescope_name) if telescope_name is not None else None
        )
        self.label = label
        self.model_version = names.validate_model_version_name(model_version)
        self._config_file_directory = None
        self._config_file_path = None

        self._load_parameters_from_db()

        self.simtel_config_writer = None
        self._added_parameter_files = None
        self._is_config_file_up_to_date = False
        self._is_exported_model_files_up_to_date = False

    def get_parameter(self, par_name):
        """
        Get an existing parameter of the model.

        Parameters
        ----------
        par_name: str
            Name of the parameter.

        Returns
        -------
        Value of the parameter

        Raises
        ------
        InvalidModelParameter
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
            raise InvalidModelParameter(msg) from e

    def get_parameter_value(self, par_name, parameter_dict=None):
        """
        Get the value of an existing parameter of the model.

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
        InvalidModelParameter
            If par_name does not match any parameter in this model.
        """
        parameter_dict = parameter_dict if parameter_dict else self.get_parameter(par_name)
        try:
            return parameter_dict.get("value") or parameter_dict.get("Value")
        except KeyError as exc:
            self._logger.error(f"Parameter {par_name} does not have a value")
            raise exc

    def get_parameter_value_with_unit(self, par_name):
        """
        Get the value of an existing parameter of the model as an Astropy Quantity with its unit.\
        If no unit is provided in the model, the value is returned without a unit.

        Parameters
        ----------
        par_name: str
            Name of the parameter.

        Returns
        -------
        Astropy quantity with the value of the parameter multiplied by its unit. If no unit is \
        provided in the model, the value is returned without a unit.

        Raises
        ------
        InvalidModelParameter
            If par_name does not match any parameter in this model.
        """
        _parameter = self.get_parameter(par_name)
        _value = self.get_parameter_value(None, _parameter)
        try:
            _units = _parameter.get("unit") or _parameter.get("units")
            return float(_value) * u.Unit(_units)
        except (KeyError, TypeError):
            return _value

    def get_parameter_type(self, par_name):
        """
        Get the type of an existing parameter of the model.

        Parameters
        ----------
        par_name: str
            Name of the parameter.

        Returns
        -------
        str or None
            type of the parameter (None if no type is defined)

        """
        parameter_dict = self.get_parameter(par_name)
        try:
            return parameter_dict.get("type") or parameter_dict.get("Type")
        except KeyError:
            self._logger.debug(f"Parameter {par_name} does not have a type")
        return None

    def get_parameter_file_flag(self, par_name):
        """
        Get value of parameter file flag.

        Parameters
        ----------
        par_name: str
            Name of the parameter.

        Returns
        -------
        bool
            True if file flag is set.

        """
        parameter_dict = self.get_parameter(par_name)
        try:
            if parameter_dict.get("file") or parameter_dict.get("File"):
                return True
        except KeyError:
            self._logger.debug(f"Parameter {par_name} does not have a file associated with it.")
        return False

    @property
    def derived(self):
        """
        Load the derived values and export them if the class instance hasn't done it yet.
        """
        if self._derived is None:
            self._load_derived_values()
            self.export_derived_files()
        return self._derived

    def _load_derived_values(self):
        """
        Load derived values from the DB

        """
        self._logger.debug("Reading derived data from DB")
        self._derived = self.db.get_derived_values(
            self.site,
            self.name,
            self.model_version,
        )

    def export_derived_files(self):
        """Write to disk a file from the derived values DB."""

        for par_now in self.derived.values():
            if par_now.get("File") or par_now.get("file"):
                self.db.export_file_db(
                    db_name=self.db.DB_DERIVED_VALUES,
                    dest=self.io_handler.get_output_directory(label=self.label, sub_dir="derived"),
                    file_name=(par_now.get("value") or par_now.get("Value")),
                )

    def print_parameters(self):
        """Print parameters and their values for debugging purposes."""
        for par in self._parameters:
            print(f"{par} = {self.get_parameter_value(par)}")

    def get_config_directory(self):
        """
        Get the path where all the configuration files for sim_telarray are written to.

        Returns
        -------
        Path
            Path where all the configuration files for sim_telarray are written to.
        """
        return self._config_file_directory

    def _set_config_file_directory_and_name(self):
        """
        Set and create the directory model parameter files are written to.

        """

        if self.name is None:
            return

        self._config_file_directory = self.io_handler.get_output_directory(
            label=(self.label or self.name), sub_dir="model"
        )

        # Setting file name and the location
        if self.site is not None and self.name is not None:
            config_file_name = names.simtel_telescope_config_file_name(
                self.site, self.name, self.model_version, self.label, self._extra_label
            )
            self._config_file_path = self._config_file_directory.joinpath(config_file_name)

        self._logger.debug(f"Config file path: {self._config_file_path}")

    def _load_parameters_from_db(self):
        """

        Read parameters from DB and store them in _parameters.

        """

        if self.db is None:
            return

        if self.name is not None:
            self._logger.debug(f"Reading telescope parameters from DB ({self.name} telescope)")
            self._set_config_file_directory_and_name()
            self._parameters = self.db.get_model_parameters(
                self.site, self.name, self.model_version, only_applicable=True
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
        """
        Return the extra label if defined, if not return ''.
        """
        return self._extra_label if self._extra_label is not None else ""

    def get_simtel_parameters(self, telescope_model=True, site_model=True):
        """
        Get simtel parameters as name and value pairs. Do not include parameters
        labels with 'simtel': False in names.site_parameters or names.telescope_parameters.

        TODO - not sure if this is ok

        Parameters
        ----------
        telescope_model: bool
            If True, telescope model parameters are included.
        site_model: bool
            If True, site model parameters are included.

        Returns
        -------
        dict
            simtel parameters as dict

        """

        _parameter_names = {}
        if telescope_model:
            _parameter_names.update(names.telescope_parameters)
        if site_model:
            _parameter_names.update(names.site_parameters)

        _simtel_parameter_value = {}
        for key in self._parameters:
            # not all parameters are listed in names.site_parameters
            # or site.telescope_parameters; use it as a filter list
            try:
                _par_name = (
                    _parameter_names[key]["db_name"] if _parameter_names[key]["simtel"] else None
                )
            except KeyError:
                _par_name = key
            # check for new and old parameter names
            for _, _simtools_name_config in _parameter_names.items():
                if (
                    key == _simtools_name_config["db_name"]
                    and _simtools_name_config["simtel"] is False
                ):
                    _par_name = None
            if _par_name is not None:
                _simtel_parameter_value[_par_name] = self._parameters[key].get(
                    "value"
                ) or self._parameters[key].get("Value")
        return _simtel_parameter_value

    def get_parameter_name_from_simtel_name(self, simtel_name):
        """
        Get the model parameter name from the simtel parameter name.
        Assumes that both names are equal if not defined otherwise in names.py.

        Parameters
        ----------
        simtel_name: str
            Simtel parameter name.

        Returns
        -------
        str
            Model parameter name.
        """

        _parameter_names = {}
        _parameter_names.update(names.telescope_parameters)
        _parameter_names.update(names.site_parameters)

        for par_name, par_info in _parameter_names.items():
            if par_info.get("db_name") == simtel_name:
                return par_name
        return simtel_name

    def add_parameter(self, par_name, value, is_file=False, is_applicable=True):
        """
        Add a new parameters to the model. This function does not modify the DB, it affects only \
        the current instance.

        TODO - check why no units are set (plus site, etc)

        Parameters
        ----------
        par_name: str
            Name of the parameter.
        value:
            Value of the parameter.
        is_file: bool
            Indicates whether the new parameter is a file or not.
        is_applicable: bool
            Indicates whether the new parameter is applicable or not.

        Raises
        ------
        InvalidModelParameter
            If an existing parameter is tried to be added.
        """
        if par_name in self._parameters:
            msg = f"Parameter {par_name} already in the model, use change_parameter instead"
            self._logger.error(msg)
            raise InvalidModelParameter(msg)

        self._logger.info(f"Adding {par_name}={value} to the model")
        self._parameters[par_name] = {}
        self._parameters[par_name]["value"] = value
        self._parameters[par_name]["type"] = type(value)
        self._parameters[par_name]["applicable"] = is_applicable
        self._parameters[par_name]["file"] = is_file

        self._is_config_file_up_to_date = False
        if is_file:
            self._is_exported_model_files_up_to_date = False

    def change_parameter(self, par_name, value):
        """
        Change the value of an existing parameter to the model. This function does not modify the \
        DB, it affects only the current instance.

        Parameters
        ----------
        par_name: str
            Name of the parameter.
        value:
            Value of the parameter.

        Raises
        ------
        InvalidModelParameter
            If the parameter to be changed does not exist in this model.
        """
        if par_name not in self._parameters:
            msg = f"Parameter {par_name} not in the model, use add_parameters instead"
            self._logger.error(msg)
            raise InvalidModelParameter(msg)

        type_of_par_name = locate(self.get_parameter_type(par_name))
        if not isinstance(value, type_of_par_name):
            self._logger.warning(
                f"The type of the provided value ({value}, {type(value)}) "
                f"is different from the type of {par_name} "
                f"({self.get_parameter_type(par_name)}). "
                f"Attempting to cast to the correct type."
            )
            try:
                value = type_of_par_name(value)
            except ValueError:
                self._logger.error(
                    f"Could not cast {value} to {self.get_parameter_type(par_name)}."
                )
                raise

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
        Change the value of multiple existing parameters in the model. This function does not \
        modify the DB, it affects only the current instance.

        Parameters
        ----------
        **kwargs
            Parameters should be passed as parameter_name=value.

        Raises
        ------
        InvalidModelParameter
            If at least one of the parameters to be changed does not exist in this model.
        """
        for par, value in kwargs.items():
            if par in self._parameters:
                self.change_parameter(par, value)
            else:
                self.add_parameter(par, value)

        self._is_config_file_up_to_date = False

    def add_parameter_file(self, par_name, file_path):
        """
        Add a file to the config file directory.

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
        shutil.copy(file_path, self._config_file_directory)

    def export_model_files(self):
        """Exports the model files into the config file directory."""

        # Removing parameter files added manually (which are not in DB)
        pars_from_db = copy(self._parameters)
        if self._added_parameter_files is not None:
            for par in self._added_parameter_files:
                pars_from_db.pop(par)

        self.db.export_model_files(pars_from_db, self._config_file_directory)
        self._is_exported_model_files_up_to_date = True

    def export_config_file(self):
        """Export the config file used by sim_telarray."""

        # Exporting model file
        if not self._is_exported_model_files_up_to_date:
            self.export_model_files()

        # Using SimtelConfigWriter to write the config file.
        self._load_simtel_config_writer()
        self.simtel_config_writer.write_telescope_config_file(
            config_file_path=self._config_file_path, parameters=self.get_simtel_parameters()
        )

    def get_config_file(self, no_export=False):
        """
        Get the path of the config file for sim_telarray. The config file is produced if the file\
        is not updated.

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
        return self._config_file_path

    def get_derived_directory(self):
        """
        Get the path where all the files with derived values for are written to.

        Returns
        -------
        Path
            Path where all the files with derived values are written to.
        """
        return self._config_file_directory.parents[0].joinpath("derived")

    def _load_simtel_config_writer(self):
        """
        Load the SimtelConfigWriter object.

        """
        if self.simtel_config_writer is None:
            self.simtel_config_writer = SimtelConfigWriter(
                site=self.site,
                telescope_name=self.name,
                model_version=self.model_version,
                label=self.label,
            )
