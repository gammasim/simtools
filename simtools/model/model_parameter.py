#!/usr/bin/python3

import logging

import astropy.units as u

from simtools import db_handler
from simtools.io_operations import io_handler
from simtools.utils import names

__all__ = ["InvalidParameter", "ModelParameter"]


class InvalidParameter(Exception):
    """Exception for invalid parameter."""


class ModelParameter:
    """
    Base class for model parameters.
    Provides methods to read parameters from DB
    and methods are available to manipulate parameters
    (changing, adding, removing etc).

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

        self.site = names.validate_site_name(site) if site is not None else None
        self.name = (
            names.validate_telescope_model_name(telescope_model_name)
            if telescope_model_name is not None
            else None
        )
        self.label = label
        self.model_version = names.validate_model_version_name(model_version)

        self.io_handler = io_handler.IOHandler()
        self.db = None
        if db is not None:
            self.db = db
        elif mongo_db_config is not None:
            self._logger.debug("Connecting to DB")
            self.db = db_handler.DatabaseHandler(mongo_db_config=mongo_db_config)

        self._parameters = {}
        self._load_parameters_from_db()

        self._config_file_directory = None
        self._config_file_path = None
        self._set_config_file_directory_and_name()

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
        Get an existing parameter of the model.
        TODO: does not including derived parameters.

        Parameters
        ----------
        par_name: str
            Name of the parameter.

        Returns
        -------
        Value of the parameter

        Raises
        ------
        InvalidParameter
            If par_name does not match any parameter in this model.
        """
        try:
            return self._parameters[par_name]
        except KeyError as e:
            msg = f"Parameter {par_name} was not found in the model"
            self._logger.error(msg)
            raise InvalidParameter(msg) from e

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
        InvalidParameter
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
        InvalidParameter
            If par_name does not match any parameter in this model.
        """
        _parameter = self.get_parameter(par_name)
        _value = self.get_parameter_value(None, _parameter)
        try:
            _units = _parameter.get("unit") or _parameter.get("units")
            self._logger.debug(f"Parameter {par_name} has units {_units}")
            return float(_value) * u.Unit(_units)
        except (KeyError, TypeError):
            self._logger.debug(f"Parameter {par_name} has value {_value} without units")
            return _value

    def print_parameters(self):
        """Print parameters and their values for debugging purposes."""
        for par, info in self._parameters.items():
            print(f"{par} = {info['Value']}")

    def _set_config_file_directory_and_name(self):
        """
        Define the variable _config_file_directory and create directories, if needed.

        """

        if self.name is None:
            return

        if self.label is not None:
            self._config_file_directory = self.io_handler.get_output_directory(
                label=self.label, sub_dir="model"
            )

        # Setting file name and the location
        if self.site is not None and self.name is not None:
            config_file_name = names.simtel_telescope_config_file_name(
                self.site, self.name, self.model_version, self.label, self._extra_label
            )
            self._config_file_path = self._config_file_directory.joinpath(config_file_name)

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
            self._logger.debug("Reading site parameters from DB")
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
