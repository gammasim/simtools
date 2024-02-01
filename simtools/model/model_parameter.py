#!/usr/bin/python3

import logging

import astropy.units as u

from simtools import db_handler
from simtools.io_operations import io_handler
from simtools.utils import names

__all__ = ["InvalidModelParameter", "ModelParameter"]


class InvalidModelParameter(Exception):
    """Exception for invalid model parameter."""


class ModelParameter:
    """
    Base class for model parameters.
    Provides methods to read parameters from DB and manipulate parameters
    (changing, adding, removing, etc).

    Parameters
    ----------
    site: str
        Site name (e.g., South or North).
    telescope_model_name: str
        Telescope model name (e.g., LST-1).
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
        telescope_model_name=None,
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
            self._logger.debug("Connecting to DB")
            self.db = db_handler.DatabaseHandler(mongo_db_config=mongo_db_config)

        self._parameters = {}
        self.site = names.validate_site_name(site) if site is not None else None
        self.name = self._get_telescope_name(telescope_model_name)
        self.label = label
        self.model_version = names.validate_model_version_name(model_version)
        self._config_file_directory = None
        self._config_file_path = None

        self._load_parameters_from_db()

    def has_parameter(self, par_name):
        """
        Verify if the parameter is in the model.

        Parameters
        ----------
        par_name: str
            Name of the parameter.

        Returns
        -------
        bool
            True if parameter is in the model.
        """
        return par_name in self._parameters

    def get_parameter(self, par_name):
        """
        Get an existing parameter of the model. Allow parameter name to be
        names used in the model database or simulation model names.

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
        if par_name in names.telescope_parameters:
            par_name = names.telescope_parameters[par_name]["db_name"]
        elif par_name in names.site_parameters:
            par_name = names.site_parameters[par_name]["db_name"]
        try:
            return self._parameters[par_name]
        except KeyError as e:
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
        # TODO check for None in parameter_dict and par_name
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
            self._logger.debug("Reading telescope parameters from DB")
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
                    _parameter_names[key]["name"] if _parameter_names[key]["simtel"] else None
                )
            except KeyError:
                _par_name = key
            # check for new and old parameter names
            for _, _simtools_name_config in _parameter_names.items():
                if (
                    key == _simtools_name_config["name"]
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
            if par_info.get("name") == simtel_name:
                return par_name
        return simtel_name

    def _get_telescope_name(self, telescope_model_name):
        """
        Telescope model name in simtools style.
        Allow input name in simtools style (e.g., MST-FlashCam-D)
        or array-element style (e.g., MSTN-01)

        """

        _name = None
        try:
            _name = names.validate_telescope_model_name(telescope_model_name)
        except AttributeError:
            return None
        except ValueError:
            pass
        if self.db:
            _name = names.telescope_model_name_from_array_element_id(
                array_element_id=telescope_model_name,
                sub_system_name="structure",
                available_telescopes=self.db.get_all_available_telescopes(),
            )
        return _name
