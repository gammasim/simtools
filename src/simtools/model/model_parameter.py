#!/usr/bin/python3
"""Base class for simulation model parameters (e.g., for SiteModel or TelescopeModel)."""

import logging
import shutil
from copy import copy, deepcopy
from pathlib import Path

import astropy.units as u

import simtools.utils.general as gen
from simtools.data_model import schema
from simtools.db import db_handler
from simtools.io import io_handler
from simtools.model import legacy_model_parameter
from simtools.simtel.simtel_config_writer import SimtelConfigWriter
from simtools.utils import names, value_conversion


class InvalidModelParameterError(Exception):
    """Exception for invalid model parameter."""


class ModelParameter:
    """
    Base class for simulation model parameters.

    Provides methods to read and manipulate parameters from DB and to write
    sim_telarray configuration files.

    Parameters
    ----------
    model_version: str
        Version of the model (ex. 5.0.0).
    site: str
        Site name (e.g., South or North).
    array_element_name: str
        Array element name (e.g., LSTN-01, LSTN-design, ILLN-01).
    collection: str
        instrument class (e.g. telescopes, calibration_devices)
        as stored under collection in the DB.
    label: str
        Instance label. Used for output file naming.
    overwrite_model_parameter_dict: dict, optional
        Dictionary to overwrite model parameters from DB with provided values.
        Instance label. Important for output file naming.
    ignore_software_version: bool
        If True, ignore software version checks for deprecated parameters.
        Useful for documentation generation.
    """

    def __init__(
        self,
        model_version,
        site=None,
        array_element_name=None,
        collection="telescopes",
        label=None,
        overwrite_model_parameter_dict=None,
        ignore_software_version=False,
    ):
        self._logger = logging.getLogger(__name__)
        self.io_handler = io_handler.IOHandler()
        self.db = db_handler.DatabaseHandler()
        if not self.db.is_configured():
            raise RuntimeError("Database is not configured.")

        self.parameters = {}
        self._simulation_config_parameters = {sw: {} for sw in names.simulation_software()}
        self.collection = collection
        self.label = label
        self.model_version = model_version
        self.ignore_software_version = ignore_software_version
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
        self.overwrite_model_parameter_dict = overwrite_model_parameter_dict
        self._added_parameter_files = None
        self._is_exported_model_files_up_to_date = False

        self._load_parameters_from_db()

        self.simtel_config_writer = None

    def _get_parameter_dict(self, par_name):
        """
        Get model parameter dictionary for a specific parameter as stored in the DB.

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

    def get_parameter_value(self, par_name):
        """
        Get the value of a model parameter.

        List of values stored in strings are returns as lists, so that no knowledge
        of the database structure is needed when accessing the model parameters.

        Parameters
        ----------
        par_name: str
            Name of the parameter.

        Returns
        -------
        Value of the parameter.

        Raises
        ------
        InvalidModelParameterError
            If par_name does not match any parameter in this model.
        """
        try:
            value = self._get_parameter_dict(par_name)["value"]
        except KeyError as exc:
            raise InvalidModelParameterError(f"Parameter {par_name} does not have a value") from exc

        if isinstance(value, str):
            try:
                _is_float = self.get_parameter_type(par_name).startswith("float")
            except (
                InvalidModelParameterError,
                TypeError,
                AttributeError,
            ):  # float - in case we don't know
                _is_float = True
            value = gen.convert_string_to_list(value, is_float=_is_float)
            if len(value) == 1:
                value = value[0]

        return value

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
        _value = self.get_parameter_value(par_name)

        try:
            if isinstance(_parameter.get("unit"), str):
                _unit = [item.strip() for item in _parameter.get("unit").split(",")]
            else:
                _unit = _parameter.get("unit")

            # if there is only one value or the values share one unit
            if (isinstance(_value, (int | float))) or (len(_value) > len(_unit)):
                return _value * u.Unit(_unit[0])

            # Create list of quantities for multiple values with different units
            return [
                value_conversion.get_value_as_quantity(
                    _value[i], _unit[i] if i < len(_unit) else None
                )
                for i in range(len(_value))
            ]

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
        str
            type of the parameter
        """
        try:
            return self._get_parameter_dict(par_name)["type"]
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
        try:
            return self._get_parameter_dict(par_name)["file"]
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

    def _set_config_file_directory_and_name(self):
        """Set and create the directory and the name of the config file."""
        if self.name is None and self.site is None:
            return

        self._config_file_directory = self.io_handler.get_model_configuration_directory(
            model_version=self.model_version
        )

        # Setting file name and the location
        config_file_name = names.simtel_config_file_name(
            self.site,
            telescope_model_name=self.name,
            label=self.label,
        )
        self._config_file_path = self.config_file_directory.joinpath(config_file_name)

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
        """
        Read parameters from Database.

        This is the main function to load the model parameters from the DB.
        """
        if self.db is None:
            return

        if self.name or self.site:
            # copy parameters dict, is it may be modified later on
            self.parameters = deepcopy(
                self.db.get_model_parameters(
                    self.site, self.name, self.collection, self.model_version
                )
            )
            self.overwrite_parameters(self.overwrite_model_parameter_dict)
            self._check_model_parameter_versions(self.parameters)

        self._load_simulation_software_parameter()
        for software_name, parameters in self._simulation_config_parameters.items():
            self._check_model_parameter_versions(parameters, software_name=software_name)

    def _check_model_parameter_versions(self, parameters, software_name=None):
        """
        Ensure parameters follow the latest schema and are compatible with installed software.

        Compares software versions listed in schema files with the installed software versions
        (e.g., sim_telarray, CORSIKA).

        For outdated model parameter schemas, legacy update functions are called to update
        the parameters to the latest schema version.

        Parameters
        ----------
        parameters: dict
            Dictionary containing model parameters.
        software_name: str
            Name of the software for which the parameters are checked.
        """
        _legacy_updates = {}
        for par_name, par_data in parameters.items():
            if par_name in (parameter_schema := names.model_parameters()):
                schema.validate_deprecation_and_version(
                    data=parameter_schema[par_name],
                    software_name=software_name,
                    ignore_software_version=self.ignore_software_version,
                )
                _latest_schema_version = parameter_schema[par_name]["schema_version"]
                if par_data["model_parameter_schema_version"] != _latest_schema_version:
                    _legacy_updates.update(
                        legacy_model_parameter.update_parameter(
                            par_name, parameters, _latest_schema_version
                        )
                    )

        legacy_model_parameter.apply_legacy_updates_to_parameters(parameters, _legacy_updates)

    def overwrite_model_parameter(self, par_name, value, parameter_version=None):
        """
        Overwrite the parameter dictionary for a specific parameter in the model.

        This function does not modify the DB, it affects only the current instance of
        the model parameter dictionary.

        If the parameter version is given only, the parameter dictionary is updated
        from the database for the given version.

        Parameters
        ----------
        par_name: str
            Name of the parameter.
        value:
            New value for the parameter.
        parameter_version: str, optional
            New version for the parameter.

        Raises
        ------
        InvalidModelParameterError
            If the parameter to be changed does not exist in this model.
        """
        if par_name not in self.parameters:
            raise InvalidModelParameterError(f"Parameter {par_name} not in the model")

        if value is None and parameter_version:
            self._overwrite_model_parameter_from_db(par_name, parameter_version)
        else:
            self._overwrite_model_parameter_from_value(par_name, value, parameter_version)

        # In case parameter is a file, the model files will be outdated
        if self.get_parameter_file_flag(par_name):
            self._is_exported_model_files_up_to_date = False

    def _overwrite_model_parameter_from_value(self, par_name, value, parameter_version=None):
        """Overwrite model parameter from provided value only."""
        value = gen.convert_string_to_list(value) if isinstance(value, str) else value
        par_type = self.get_parameter_type(par_name)

        if par_type in ("list", "dict"):
            if not gen.validate_data_type(
                reference_dtype=par_type,
                value=value,
                dtype=None,
                allow_subtypes=True,
            ):
                raise ValueError(f"Could not cast {value} of type {type(value)} to {par_type}.")
        else:
            for value_element in gen.ensure_iterable(value):
                if not gen.validate_data_type(
                    reference_dtype=par_type,
                    value=value_element,
                    dtype=None,
                    allow_subtypes=True,
                ):
                    raise ValueError(
                        f"Could not cast {value_element} of type "
                        f"{type(value_element)} to {par_type}."
                    )

        self._logger.debug(
            f"Changing parameter {par_name} from {self.get_parameter_value(par_name)} to {value}"
        )
        self.parameters[par_name]["value"] = value
        if parameter_version:
            self.parameters[par_name]["parameter_version"] = parameter_version

    def _overwrite_model_parameter_from_db(self, par_name, parameter_version):
        """Overwrite model parameter from DB for a specific version."""
        _para_dict = self.db.get_model_parameter(
            parameter=par_name,
            site=self.site,
            array_element_name=self.name,
            parameter_version=parameter_version,
        )
        if _para_dict:
            self.parameters[par_name] = _para_dict.get(par_name)
        self._logger.debug(
            f"Changing parameter {par_name} to version {parameter_version} with value "
            f"{self.parameters[par_name]['value']}"
        )

    def _get_key_for_parameter_changes(self, site, array_element_name, changes_data):
        """
        Get the key for parameter changes based on site and array element name.

        For array elements, the following cases are taken into account:

        - array element name in changes_data: specific array element is returned
        - design type in changes_data: specific design type is returned if array
          element matches this design

        Parameters
        ----------
        site: str
            Site name.
        array_element_name: str
            Array element name.
        changes_data: dict
            Dictionary containing the changes data.

        Returns
        -------
        str
            Key for parameter changes.
        """
        if site and not array_element_name:
            return f"OBS-{site}"

        if array_element_name in changes_data:
            return array_element_name

        design_type = self.db.get_design_model(
            model_version=self.model_version,
            array_element_name=array_element_name,
            collection=self.collection,
        )
        if design_type in changes_data:
            return design_type

        return None

    def overwrite_parameters(self, changes, flat_dict=False):
        """
        Change the value of multiple existing parameters in the model.

        This function does not modify the DB, it affects only the current instance.

        Allows for two types of 'changes' dictionary:

        - simple (flat_dict=True): '{parameter_name: new_value, ...}'
        - model repository style (flat_dict=False):
          '{array_element: {parameter_name: {"value": new_value, "version": new_version}, ...}}'

        Parameters
        ----------
        changes: dict
            Parameters to be changed.
        """
        if not changes:
            return
        if not flat_dict:
            key_for_changes = self._get_key_for_parameter_changes(self.site, self.name, changes)
            changes = changes.get(key_for_changes, {})
        if not changes:
            return

        if flat_dict:
            self._logger.debug(f"Overwriting parameters with changes: {changes}")
        else:
            self._logger.debug(
                f"Overwriting parameters for {key_for_changes} with changes: {changes}"
            )

        for par_name, par_value in changes.items():
            if par_name not in self.parameters:
                self._logger.warning(
                    f"Parameter {par_name} not found in model {self.name}, cannot overwrite it."
                )
                continue

            if isinstance(par_value, dict) and ("value" in par_value or "version" in par_value):
                self.overwrite_model_parameter(
                    par_name, par_value.get("value"), par_value.get("version")
                )
            else:
                self.overwrite_model_parameter(par_name, par_value)

    def overwrite_model_file(self, par_name, file_path):
        """
        Overwrite the existing model file in the config file directory.

        Keeps track of updated model file with '_added_parameter_files' attribute.

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
        Export model files from the database into the config file directory.

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

    def get_config_file_path(self, label=None):
        """Return config file path for a given label.

        Parameters
        ----------
        label : str or None
            Label used for output file naming. If None, use this model's label.

        Returns
        -------
        pathlib.Path
            Path to the sim_telarray configuration file.
        """
        config_file_name = names.simtel_config_file_name(
            self.site,
            telescope_model_name=self.name,
            label=self.label if label is None else label,
        )
        return self.config_file_directory.joinpath(config_file_name)

    def write_sim_telarray_config_file(
        self, additional_models=None, label=None, config_file_path=None
    ):
        """
        Write the sim_telarray configuration file.

        Parameters
        ----------
        additional_models: TelescopeModel or SiteModel
            Model object for additional parameter to be written to the config file.
        label: str or None
            Optional label override used for output file naming.
        config_file_path: pathlib.Path or str or None
            Optional explicit path of the config file. If not given, it is derived from ``label``.
        """
        self.parameters.update(self._simulation_config_parameters.get("sim_telarray", {}))
        self.export_model_files(update_if_necessary=True)

        self._add_additional_models(additional_models)

        config_file_path = (
            Path(config_file_path)
            if config_file_path is not None
            else self.get_config_file_path(label=label)
        )

        # Ensure the writer label matches the config file naming label.
        self._load_simtel_config_writer(label=label)
        self.simtel_config_writer.write_telescope_config_file(
            config_file_path=config_file_path,
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

    def _load_simtel_config_writer(self, label=None):
        """Load the SimtelConfigWriter object."""
        desired_label = self.label if label is None else label
        if label is not None or self.simtel_config_writer is None:
            self.simtel_config_writer = SimtelConfigWriter(
                site=self.site,
                telescope_model_name=self.name,
                telescope_design_model=self.design_model,
                model_version=self.model_version,
                label=desired_label,
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
