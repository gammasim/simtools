#!/usr/bin/python3
"""Base class for simulation model parameters (e.g., for SiteModel or TelescopeModel)."""

import logging
from copy import deepcopy
from pathlib import Path

import astropy.units as u

from simtools.context import RunContext
from simtools.data_model import parameter_catalog as catalog
from simtools.data_model import schema
from simtools.data_model.json_validation import validate_finite_json_values
from simtools.data_model.validate_data import DataValidator
from simtools.utils import names, value_conversion

SIMULATION_SOFTWARE_COLLECTIONS = {
    "sim_telarray": "configuration_sim_telarray",
    "corsika": "configuration_corsika",
}
_SIM_TELARRAY_COLLECTION = "configuration_sim_telarray"
_METADATA_KEYS = ("unit", "model_parameter_schema_version")


class InvalidModelParameterError(Exception):
    """Exception for invalid model parameter."""


def _normalise_overrides(changes):
    """
    Normalise model overrides to ``{(collection, element, parameter): entry}``.

    Entry is a dict with ``value``, ``version`` and ``metadata`` (unit, schema version or None).
    Element is the top-level key of the override mapping (site, array element, design type
    or ``global``). Unknown parameter names have collection ``None``.
    """
    normalised = {}
    for element, parameters in (changes or {}).items():
        if not isinstance(parameters, dict):
            continue
        for par_name, par_value in parameters.items():
            normalised[(_collection_or_none(par_name), element, par_name)] = _normalise_entry(
                par_value
            )
    return normalised


def _collection_or_none(par_name):
    """Return the collection name of a parameter, or None for unknown parameter names."""
    try:
        return catalog.get_collection_name_from_parameter_name(par_name)
    except KeyError:
        return None


def _normalise_entry(par_value):
    """Return the ``value``, ``version`` and ``metadata`` entry of a single override."""
    if isinstance(par_value, dict) and ("value" in par_value or "version" in par_value):
        return {
            "value": par_value.get("value"),
            "version": par_value.get("version"),
            "metadata": {k: par_value[k] for k in _METADATA_KEYS if k in par_value} or None,
        }
    return {"value": par_value, "version": None, "metadata": None}


class ModelParameter:
    """
    Base class for simulation model parameters.

    Provides methods to read and manipulate parameters from the model repository.
    Writing of sim_telarray configuration files and model exports is done by the model
    writers (see ``simulation.configuration.get_model_writer``).

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
        as stored under collection in the model repository.
    label: str
        Instance label. Used for output file naming.
    overwrite_model_parameter_dict: dict, optional
        Dictionary to overwrite model parameters from the model repository with provided values.
    ignore_software_version: bool
        If True, ignore software version checks for deprecated parameters.
        Useful for documentation generation.
    model_directory: pathlib.Path or str, optional
        Directory for generated model assets and sim_telarray configuration files.
    parameter_names: iterable of str, optional
        If supplied, load only these model parameters.
    load_simulation_software_parameters: bool
        If False, do not load CORSIKA and sim_telarray configuration parameters.
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
        model_reader=None,
        model_directory=None,
        parameter_names=None,
        load_simulation_software_parameters=True,
        context=None,
    ):
        self._logger = logging.getLogger(__name__)
        self.context = context or RunContext.from_environment()
        self.io_handler = self.context.paths
        self.model_reader = model_reader or self.context.model_reader
        if self.model_reader is None:
            raise RuntimeError("No simulation model reader is configured.")

        self.parameters = {}
        self._simulation_config_parameters = {sw: {} for sw in catalog.simulation_software()}
        self.collection = collection
        self.label = label
        self.model_version = model_version
        self.ignore_software_version = ignore_software_version
        self._model_loading_options = {
            "parameter_names": None if parameter_names is None else frozenset(parameter_names),
            "load_simulation_software_parameters": load_simulation_software_parameters,
        }
        self.site = names.validate_site_name(site) if site is not None else None
        self.name = (
            names.validate_array_element_name(array_element_name)
            if array_element_name is not None
            else None
        )
        self.design_model = (
            self.model_reader.get_design_model(
                self.model_version, self.name, collection="telescopes"
            )
            if self.name is not None
            else None
        )
        self._config_file_directory = Path(model_directory) if model_directory is not None else None
        self._config_file_path = None
        self.overwrite_model_parameter_dict = overwrite_model_parameter_dict
        self._overrides = _normalise_overrides(overwrite_model_parameter_dict)
        self.model_files_exported = False

        self._load_parameters_from_repository()

        self.configuration_writers = {}

    def _get_parameter_dict(self, par_name):
        """
        Get model parameter dictionary for a specific parameter as stored in the model repository.

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
            Dictionary with complete model repository entry for the given parameter.

        Raises
        ------
        InvalidModelParameterError
            If par_name does not match any parameter in this model.
        """
        try:
            return self.parameters[par_name]
        except (KeyError, ValueError) as exc:
            raise InvalidModelParameterError(
                f"Parameter {par_name} was not found in the model {self.name}, {self.site}."
            ) from exc

    def get_parameter_value(self, par_name):
        """
        Get the value of a model parameter.

        List of values stored in strings are returns as lists, so that no knowledge
        of the model repository structure is needed when accessing the model parameters.

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
            value = value_conversion.convert_string_to_list(value, is_float=_is_float)
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

    def get_parameter_table(self, par_name):
        """Return the validated ECSV table associated with a model parameter."""
        parameter_data = self._get_parameter_dict(par_name)
        try:
            return self.model_reader.get_parameter_table(parameter_data)
        except ValueError as exc:
            raise InvalidModelParameterError(
                f"Parameter {par_name} does not reference a valid ECSV table: {exc}"
            ) from exc

    def get_parameter_type(self, par_name):
        """
        Get the type of an existing model parameter (the ``type`` field).

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

    def _set_config_file_directory_and_name(self):
        """Set and create the directory and the name of the config file."""
        if self.name is None and self.site is None:
            return

        if self._config_file_directory is None:
            self._config_file_directory = self.io_handler.get_model_configuration_directory(
                model_version=self.model_version
            )

        self._config_file_path = self.config_file_directory.joinpath(
            names.sim_telarray_config_file_name(site=self.site, telescope_model_name=self.name)
        )

    def _load_parameters_from_repository(self):
        """Read model parameters, apply overrides, and load simulation software parameters."""
        if not (self.name or self.site):
            return

        self.parameters = deepcopy(
            self.model_reader.get_model_parameters(
                self.site,
                self.name,
                self.collection,
                self.model_version,
                parameter_names=self._model_loading_options["parameter_names"],
            )
        )
        selected_element = self._selected_override_element()
        for (collection, element, par_name), entry in self._overrides.items():
            if element != selected_element:
                continue
            if collection == _SIM_TELARRAY_COLLECTION:
                self._route_to_sim_telarray(par_name, entry)
            elif collection not in SIMULATION_SOFTWARE_COLLECTIONS.values():
                if par_name not in self.parameters:
                    raise ValueError(
                        f"Parameter {par_name} not found in model {self.name}, cannot overwrite it."
                    )
                self._set_parameter(self.parameters, par_name, **_entry_kwargs(entry))

        self._check_model_parameter_versions(self.parameters, self.ignore_software_version)
        if self._model_loading_options["load_simulation_software_parameters"]:
            self._load_simulation_software_parameters()

    def _selected_override_element(self):
        """Return the override element (site, array element or design type) for this model."""
        if self.site and not self.name:
            return f"OBS-{self.site}"
        elements = {element for (_, element, _) in self._overrides}
        if self.name in elements:
            return self.name
        design_model = self.model_reader.get_design_model(
            self.model_version, self.name, collection=self.collection
        )
        return design_model if design_model in elements else None

    def _load_simulation_software_parameters(self):
        """Load simulation software parameters and apply the matching overrides."""
        for software in self._simulation_config_parameters:
            try:
                loaded = self.model_reader.get_simulation_configuration_parameters(
                    site=self.site,
                    array_element_name=self.name,
                    model_version=self.model_version,
                    simulation_software=software,
                )
            except ValueError:
                continue
            # overrides routed to sim_telarray before loading take precedence over the repository
            if existing := self._simulation_config_parameters[software]:
                loaded.update(existing)
            self._simulation_config_parameters[software] = loaded
            self._apply_software_overrides(software)

    def _apply_software_overrides(self, software):
        """Apply overrides for one simulation software (global and matching elements)."""
        collection = SIMULATION_SOFTWARE_COLLECTIONS.get(software)
        if collection is None:
            return
        store = self._simulation_config_parameters[software]
        for (par_collection, element, par_name), entry in self._overrides.items():
            if par_collection != collection:
                continue
            if software == "sim_telarray" and not (
                element == "global"
                or names.matches_array_element_name_or_design_type(element, self.name)
            ):
                continue
            if software == "sim_telarray":
                self._route_to_sim_telarray(par_name, entry)
            elif par_name in store:
                self._set_parameter(store, par_name, **_entry_kwargs(entry))
            else:
                self._logger.warning(
                    f"Skipping overwrite of '{par_name}': not in {software} configuration."
                )

    def _route_to_sim_telarray(self, par_name, entry):
        """Store an override as a sim_telarray configuration entry (no schema validation)."""
        sim_telarray_params = self._simulation_config_parameters.setdefault("sim_telarray", {})
        sim_telarray_params[par_name] = {"value": entry["value"], **(entry["metadata"] or {})}

    @staticmethod
    def _check_model_parameter_versions(parameters, ignore_software_version, software_name=None):
        """
        Ensure parameters follow the latest schema and are compatible with installed software.

        Compares software versions listed in schema files with the installed software versions
        (e.g., sim_telarray, CORSIKA).

        Parameters
        ----------
        parameters: dict
            Dictionary containing model parameters.
        ignore_software_version: bool
            If True, ignore software version checks for deprecated parameters.
        software_name: str
            Name of the software for which the parameters are checked.
        """
        for par_name, par_data in parameters.items():
            validate_finite_json_values(par_data.get("value"))
            if par_name in (parameter_schema := catalog.model_parameters()):
                schema.validate_deprecation_and_version(
                    data=parameter_schema[par_name],
                    software_name=software_name,
                    ignore_software_version=ignore_software_version,
                )
                schema_version = par_data.get("model_parameter_schema_version")
                if schema_version is None:
                    raise ValueError(f"Parameter '{par_name}' has no schema version")

    def overwrite_model_parameter(self, par_name, value, parameter_version=None, metadata=None):
        """
        Overwrite a parameter of this model in memory (the model repository is not modified).

        If only the parameter version is given (value None), the parameter is taken from the
        model repository for that version.

        Parameters
        ----------
        par_name: str
            Name of the parameter.
        value:
            New value for the parameter.
        parameter_version: str, optional
            New version for the parameter.
        metadata: dict, optional
            Additional metadata (unit, model_parameter_schema_version).

        Raises
        ------
        InvalidModelParameterError
            If the parameter to be changed does not exist in this model.
        """
        if par_name not in self.parameters:
            raise InvalidModelParameterError(f"Parameter {par_name} not in the model")
        self._set_parameter(self.parameters, par_name, value, parameter_version, metadata)
        if self.parameters[par_name].get("file", False):
            self.model_files_exported = False

    def _set_parameter(self, store, par_name, value, parameter_version=None, metadata=None):
        """Set one parameter in a store, validated against its schema (or taken from repository)."""
        if value is None and parameter_version:
            parameter_data = self.model_reader.get_model_parameter(
                parameter=par_name,
                site=self.site,
                array_element_name=self.name,
                parameter_version=parameter_version,
            )
            if parameter_data:
                store[par_name] = parameter_data.get(par_name)
            return

        if isinstance(value, str):
            value = value_conversion.convert_string_to_list(value)
        try:
            DataValidator.validate_model_parameter(
                self._update_parameter_dict(store, par_name, value, parameter_version, metadata),
                par_name=par_name,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Schema file for parameter {par_name} not found.") from exc

    def _update_parameter_dict(self, store, par_name, value, parameter_version, metadata):
        """
        Update a parameter dictionary with value and metadata, filling schema-derived defaults.

        This keeps overwrite behavior robust when metadata dictionaries are incomplete
        (e.g. missing type or unit).
        """
        par_dict = store[par_name]
        par_dict["value"] = value
        if parameter_version:
            par_dict["parameter_version"] = parameter_version

        schema_version = (
            metadata.get("model_parameter_schema_version")
            if metadata and "model_parameter_schema_version" in metadata
            else par_dict.get("model_parameter_schema_version")
        ) or "latest"
        try:
            schema_version = schema.get_model_parameter_schema(par_name, schema_version).get(
                "schema_version"
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Schema file for parameter {par_name} not found.") from exc
        except ValueError as exc:
            raise ValueError(
                f"Schema version '{schema_version}' not available for parameter '{par_name}'"
            ) from exc

        par_type = schema.get_parameter_attribute_from_schema(par_name, schema_version, "type")
        schema_unit = schema.get_parameter_attribute_from_schema(par_name, schema_version, "unit")

        if metadata:
            for key in _METADATA_KEYS:
                if key in metadata:
                    par_dict[key] = metadata[key]
            if "model_parameter_schema_version" in metadata and isinstance(par_type, list):
                par_dict["type"] = par_type

        par_dict["model_parameter_schema_version"] = schema_version
        if par_type is not None:
            par_dict.setdefault("type", par_type)
        par_dict.setdefault("unit", schema_unit)
        return par_dict


def _entry_kwargs(entry):
    """Return keyword arguments for ``ModelParameter._set_parameter`` from an override entry."""
    return {
        "value": entry["value"],
        "parameter_version": entry["version"],
        "metadata": entry["metadata"],
    }
