#!/usr/bin/python3
"""Base class for simulation model parameters."""

import logging
import shutil
from copy import copy

import astropy.units as u

import simtools.utils.general as gen
from simtools.db import db_handler
from simtools.io import ascii_handler, io_handler
from simtools.simtel.simtel_config_writer import SimtelConfigWriter
from simtools.utils import names

__all__ = ["InvalidModelParameterError", "ModelParameter"]


class InvalidModelParameterError(Exception):
    """Exception for invalid model parameter."""


class ModelParameter:
    """
    Base class for simulation model parameters.

    Provides methods to read and manipulate parameters from DB and to write
    sim_telarray configuration files.

    Parameters
    ----------
    db: DatabaseHandler
        Database handler.
    model_version: str
        Version of the model (ex. 5.0.0).
    site: str
        Site name (e.g., South or North).
    array_element_name: str
        Array element name (e.g., LSTN-01, LSTN-design, ILLN-01).
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
        array_element_name=None,
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
        self._simulation_config_parameters = {sw: {} for sw in names.simulation_software()}
        self.collection = collection
        self.label = label
        self.model_version = model_version
        self.site = names.validate_site_name(site) if site is not None else None
        self.name = (
            names.validate_array_element_name(array_element_name)
            if array_element_name is not None
            else None
        )
        self.design_model = self.db.get_design_model(
            self.model_version, self.name, collection="telescopes"
        )
        self._config_file_directory = None
        self._config_file_path = None
        self._load_parameters_from_db()

        self.simtel_config_writer = None
        self._added_parameter_files = None
        self._is_config_file_up_to_date = False
        self._is_exported_model_files_up_to_date = False

    @property
    def model_version(self):
        """Model version."""
        return self._model_version

    @model_version.setter
    def model_version(self, model_version):
        """
        Set model version.

        Parameters
        ----------
        model_version: str or list
            Model version (e.g., "6.0.0").
            If a list is passed, it must contain exactly one element,
            and only that element will be used.

        Raises
        ------
        ValueError
            If more than one model version is passed.
        """
        if isinstance(model_version, list):
            raise ValueError(
                f"Only one model version can be passed to {self.__class__.__name__}, not a list."
            )
        self._model_version = model_version

    @property
    def parameters(self):
        """
        Model parameters dictionary.

        Returns
        -------
        dict
            Dictionary containing all model parameters
        """
        return self._parameters

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
            return self.parameters[par_name]
        except (KeyError, ValueError) as e:
            raise InvalidModelParameterError(
                f"Parameter {par_name} was not found in the model {self.name}, {self.site}."
            ) from e

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

        Parameters
        ----------
        par_name: str
            Name of the parameter.

        Returns
        -------
        Astropy quantity with the value of the parameter multiplied by its unit.
        If no unit is provided in the model, the value is returned without a unit.

        """
        _parameter = self._get_parameter_dict(par_name)
        _value = self.get_parameter_value(par_name, _parameter)

        try:
            if isinstance(_parameter.get("unit"), str):
                _unit = [item.strip() for item in _parameter.get("unit").split(",")]
            else:
                _unit = _parameter.get("unit")

            # if there is only one value or the values share one unit
            if (isinstance(_value, (int | float))) or (len(_value) > len(_unit)):
                return _value * u.Unit(_unit[0])

            # entries with 'null' units should be returned as dimensionless
            _astropy_units = [
                u.Unit(item) if item != "null" else u.dimensionless_unscaled for item in _unit
            ]

            return [_value[i] * _astropy_units[i] for i in range(len(_value))]

        except (KeyError, TypeError, AttributeError) as exc:
            self._logger.debug(
                f"{exc} encountered for parameter {par_name}, returning only value without units."
            )
            return _value  # if unit is NoneType

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
            return parameter_dict["type"]
        except KeyError:
            self._logger.debug(f"Parameter {par_name} does not have a type.")
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
            return parameter_dict["file"]
        except KeyError:
            self._logger.debug(f"Parameter {par_name} does not have a file associated with it.")
        return False

    def get_parameter_version(self, par_name):
        """
        Get version for a given parameter used in the model.

        Parameters
        ----------
        par_name: str
            Name of the parameter.

        Returns
        -------
        str
            parameter version used in the model (eg. '1.0.0')
        """
        return self._get_parameter_dict(par_name)["parameter_version"]

    def print_parameters(self):
        """Print parameters and their values for debugging purposes."""
        for par in self.parameters:
            print(f"{par} = {self.get_parameter_value(par)}")

    def _set_config_file_directory_and_name(self):
        """Set and create the directory and the name of the config file."""
        if self.name is None and self.site is None:
            return

        self._config_file_directory = self.io_handler.get_model_configuration_directory(
            label=self.label, model_version=self.model_version
        )

        # Setting file name and the location
        config_file_name = names.simtel_config_file_name(
            self.site,
            self.model_version,
            telescope_model_name=self.name,
            label=self.label,
            extra_label=self._extra_label,
        )
        self._config_file_path = self.config_file_directory.joinpath(config_file_name)

        self._logger.debug(f"Config file path: {self._config_file_path}")

    def get_simulation_software_parameters(self, simulation_software):
        """
        Get simulation software parameters.

        Parameters
        ----------
        simulation_software: str
            Simulation software name.

        Returns
        -------
        dict
            Simulation software parameters.
        """
        return self._simulation_config_parameters.get(simulation_software)

    def _load_simulation_software_parameter(self):
        """Read simulation software parameters from DB."""
        for simulation_software in self._simulation_config_parameters:
            try:
                self._simulation_config_parameters[simulation_software] = (
                    self.db.get_simulation_configuration_parameters(
                        site=self.site,
                        array_element_name=self.name,
                        model_version=self.model_version,
                        simulation_software=simulation_software,
                    )
                )
            except ValueError:
                pass

    def _load_parameters_from_db(self):
        """Read parameters from DB and store them in _parameters."""
        if self.db is None:
            return

        if self.name or self.site:
            self._parameters = self.db.get_model_parameters(
                self.site, self.name, self.collection, self.model_version
            )

        self._load_simulation_software_parameter()

    @property
    def extra_label(self):
        """Return the extra label if defined, if not return ''."""
        return self._extra_label if self._extra_label is not None else ""

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
        if par_name not in self.parameters:
            raise InvalidModelParameterError(f"Parameter {par_name} not in the model")

        value = gen.convert_string_to_list(value) if isinstance(value, str) else value

        par_type = self.get_parameter_type(par_name)
        if not gen.validate_data_type(
            reference_dtype=par_type,
            value=value,
            dtype=None,
            allow_subtypes=True,
        ):
            raise ValueError(f"Could not cast {value} of type {type(value)} to {par_type}.")

        self._logger.debug(
            f"Changing parameter {par_name} from {self.get_parameter_value(par_name)} to {value}"
        )
        self.parameters[par_name]["value"] = value

        # In case parameter is a file, the model files will be outdated
        if self.get_parameter_file_flag(par_name):
            self._is_exported_model_files_up_to_date = False

        self._is_config_file_up_to_date = False

    def change_multiple_parameters_from_file(self, file_name):
        """
        Change values of multiple existing parameters in the model from a file.

        This function does not modify the DB, it affects only the current instance.
        This feature is intended for developers and lacks validation.

        Parameters
        ----------
        file_name: str
            File containing the parameters to be changed.
        """
        self._logger.warning(
            "Changing multiple parameters from file is a feature for developers."
            "Insufficient validation of parameters."
        )
        self._logger.debug(f"Changing parameters from file {file_name}")
        self.change_multiple_parameters(**ascii_handler.collect_data_from_file(file_name=file_name))

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
            if par in self.parameters:
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
        self._added_parameter_files = self._added_parameter_files or []
        self._added_parameter_files.append(par_name)
        shutil.copy(file_path, self.config_file_directory)

    def export_model_files(self, destination_path=None, update_if_necessary=False):
        """
        Export the model files into the config file directory.

        Parameters
        ----------
        destination_path: str
            Path to the directory where the model files should be exported.
            If None, the config file directory is used.
        update_if_necessary: bool
            If True, the model files are only exported if they are not up to date.
        """
        if self._is_exported_model_files_up_to_date and update_if_necessary:
            self._logger.debug(
                f"Model files for {self.name} are already exported to {self.config_file_directory}"
            )
            return
        # Removing parameter files added manually (which are not in DB)
        pars_from_db = copy(self.parameters)
        if self._added_parameter_files is not None:
            for par in self._added_parameter_files:
                pars_from_db.pop(par)

        self.db.export_model_files(
            parameters=pars_from_db,
            dest=destination_path or self.config_file_directory,
        )
        self._is_exported_model_files_up_to_date = True

    def write_sim_telarray_config_file(self, additional_models=None):
        """
        Write the sim_telarray configuration file.

        Parameters
        ----------
        additional_models: TelescopeModel or SiteModel
            Model object for additional parameter to be written to the config file.
        """
        self.parameters.update(self._simulation_config_parameters.get("sim_telarray", {}))
        self.export_model_files(update_if_necessary=True)

        self._add_additional_models(additional_models)

        self._load_simtel_config_writer()
        self.simtel_config_writer.write_telescope_config_file(
            config_file_path=self.config_file_path,
            parameters=self.parameters,
        )

    def _add_additional_models(self, additional_models):
        """Add additional models to the current model parameters."""
        if additional_models is None:
            return

        if isinstance(additional_models, dict):
            for additional_model in additional_models.values():
                self._add_additional_models(additional_model)
            return

        self.parameters.update(additional_models.parameters)
        additional_models.export_model_files(self.config_file_directory, update_if_necessary=True)

    @property
    def config_file_directory(self):
        """Directory for configuration files. Configure if not yet set."""
        if self._config_file_directory is None:
            self._set_config_file_directory_and_name()
        return self._config_file_directory

    @property
    def config_file_path(self):
        """Path of the config file. Configure, if necessary."""
        if self._config_file_path is None:
            self._set_config_file_directory_and_name()
        return self._config_file_path

    def _load_simtel_config_writer(self):
        """Load the SimtelConfigWriter object."""
        if self.simtel_config_writer is None:
            self.simtel_config_writer = SimtelConfigWriter(
                site=self.site,
                telescope_model_name=self.name,
                model_version=self.model_version,
                label=self.label,
            )

    def export_nsb_spectrum_to_telescope_altitude_correction_file(self, model_directory):
        """
        Export the NSB spectrum to the telescope altitude correction file.

        This method is needed because testeff corrects the NSB spectrum from the original altitude
        used in the Benn & Ellison model to the telescope altitude.
        This is done internally in testeff, but the NSB spectrum is not written out to the model
        directory. This method allows to export it explicitly.

        Parameters
        ----------
        model_directory: Path
            Model directory to export the file to.
        """
        self.db.export_model_files(
            parameters={
                "nsb_spectrum_at_2200m": {
                    "value": self._simulation_config_parameters["sim_telarray"][
                        "correct_nsb_spectrum_to_telescope_altitude"
                    ]["value"],
                    "file": True,
                }
            },
            dest=model_directory,
        )
