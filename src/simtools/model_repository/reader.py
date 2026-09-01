"""Read simulation models through a source-neutral interface."""

import shutil
from copy import deepcopy
from pathlib import Path

from astropy.table import Table
from packaging.version import Version

from simtools.io import ascii_handler
from simtools.model_repository import files
from simtools.utils import names, value_conversion
from simtools.version import resolve_version_to_latest_patch


class FileSystemModelSource:
    """Read-only source backed by a checked-out simulation-model repository."""

    def __init__(self, simulation_models_path):
        """Initialize and validate a repository path."""
        self.simulation_models_path = Path(simulation_models_path).expanduser().resolve()
        self.productions_path = self.simulation_models_path / "simulation-models/productions"
        self.model_parameters_path = (
            self.simulation_models_path / "simulation-models/model_parameters"
        )
        self.files_path = self.model_parameters_path / "Files"
        self._production_tables = {}
        self._production_files = {}
        self._parameters = {}
        self._model_versions = None
        self._validate_model_path()

    @property
    def source_name(self):
        """Return a user-facing description of the model source."""
        return str(self.simulation_models_path)

    def _validate_model_path(self):
        """Validate the required simulation-model directories."""
        if not self.simulation_models_path.exists():
            raise FileNotFoundError(
                f"Simulation models path does not exist: {self.simulation_models_path}"
            )
        for required_path in (self.productions_path, self.model_parameters_path):
            if not required_path.is_dir():
                raise FileNotFoundError(
                    f"Expected simulation models directory not found: {required_path}"
                )

    def get_model_versions(self, collection_name="telescopes"):
        """Return semantically sorted production versions."""
        del collection_name
        if self._model_versions is None:
            versions = [path.name for path in self.productions_path.iterdir() if path.is_dir()]
            self._model_versions = sorted(versions, key=Version)
        return list(self._model_versions)

    def read_production_table(self, collection_name, model_version):
        """Return an aggregated production table for a collection and version."""
        key = (str(model_version), collection_name)
        if key not in self._production_tables:
            model_path = self.productions_path / str(model_version)
            if not model_path.is_dir():
                raise ValueError(f"Model version {model_version} not found in {self.source_name}")
            version_key = str(model_version)
            if version_key not in self._production_files:
                self._production_files[version_key] = files.get_production_table_files(model_path)
            tables = files.read_production_tables(
                model_path,
                collection_name=collection_name,
                production_files=self._production_files[version_key],
            )
            try:
                self._production_tables[key] = tables[collection_name]
            except KeyError as exc:
                raise ValueError(
                    f"No production table for {collection_name} in model version {model_version}"
                ) from exc
        return deepcopy(self._production_tables[key])

    def query_model_parameters(self, query, collection_name):
        """Read parameter files matching an internal source query."""
        parameter_queries = query.get("$or", [query])
        instrument = query.get("instrument")
        if not instrument and collection_name == "sites" and query.get("site"):
            instrument = f"OBS-{query['site']}"
        if not instrument and collection_name == "configuration_corsika":
            instrument = "xSTx-design"
        if not instrument:
            raise ValueError(
                f"Filesystem lookup for collection {collection_name} requires an array element name"
            )

        parameters = []
        for parameter_query in parameter_queries:
            parameter = parameter_query.get("parameter")
            parameter_version = parameter_query.get("parameter_version")
            if not parameter or not parameter_version:
                continue
            parameter_path = self._parameter_path(
                collection_name, instrument, parameter, parameter_version
            )
            if not parameter_path.is_file():
                continue
            parameter_data = self._read_parameter_file(parameter_path)
            if self._matches_query(parameter_data, query):
                parameters.append(parameter_data)
        if not parameters:
            raise ValueError(f"No parameters found for {collection_name}: {query}")
        return parameters

    def read_parameters(self, parameter_versions, collection_name, instrument=None, site=None):
        """Read parameter files by name and version."""
        if not instrument and collection_name == "sites" and site:
            instrument = f"OBS-{site}"
        if not instrument and collection_name == "configuration_corsika":
            instrument = "xSTx-design"
        if not instrument:
            raise ValueError(
                f"Filesystem lookup for collection {collection_name} requires an array element name"
            )

        parameters = []
        for parameter, parameter_version in parameter_versions.items():
            parameter_path = self._parameter_path(
                collection_name, instrument, parameter, parameter_version
            )
            if not parameter_path.is_file():
                continue
            parameter_data = self._read_parameter_file(parameter_path)
            if self._matches_filters(parameter_data, instrument, site):
                parameters.append(parameter_data)
        if not parameters:
            raise ValueError(f"No parameters found for {collection_name}: {parameter_versions}")
        return parameters

    def _parameter_path(self, collection_name, instrument, parameter, parameter_version):
        """Return the path for one parameter version."""
        path = self.model_parameters_path
        if collection_name in ("configuration_sim_telarray", "configuration_corsika"):
            path /= collection_name
        if collection_name != "configuration_corsika":
            path /= instrument
        return path / parameter / f"{parameter}-{parameter_version}.json"

    def _read_parameter_file(self, parameter_path):
        """Read and cache one parameter JSON file."""
        key = str(parameter_path)
        if key not in self._parameters:
            data = ascii_handler.collect_data_from_file(file_name=parameter_path)
            data["value"], _ = value_conversion.split_value_and_unit(
                data["value"], "int" in data.get("type", "float")
            )
            data["value"], base_unit, _ = value_conversion.get_value_unit_type(
                value=data["value"], unit_str=data.get("unit")
            )
            data["unit"] = value_conversion.normalize_model_parameter_unit(data["value"], base_unit)
            self._parameters[key] = data
        return deepcopy(self._parameters[key])

    @staticmethod
    def _matches_query(parameter_data, query):
        """Return whether parameter metadata matches source query filters."""
        return FileSystemModelSource._matches_filters(
            parameter_data, query.get("instrument"), query.get("site")
        )

    @staticmethod
    def _matches_filters(parameter_data, instrument, site):
        """Return whether parameter metadata matches source filters."""
        if instrument and parameter_data.get("instrument") != instrument:
            return False
        parameter_sites = parameter_data.get("site")
        if site and isinstance(parameter_sites, list):
            return site in parameter_sites
        return not site or parameter_sites == site

    def export_model_files(self, parameters=None, file_names=None, dest=None):
        """Copy referenced model files to a destination directory."""
        if dest is None:
            raise ValueError("Destination path is required to export model files.")
        names_to_export = file_names
        if names_to_export is None:
            names_to_export = [
                parameter["value"]
                for parameter in (parameters or {}).values()
                if isinstance(parameter, dict) and parameter.get("file") and parameter.get("value")
            ]
        if isinstance(names_to_export, str):
            names_to_export = [names_to_export]
        destination = Path(dest)
        destination.mkdir(parents=True, exist_ok=True)
        exported = {}
        for file_name in names_to_export:
            source = self._safe_file_path(file_name)
            target = destination / file_name
            if target.exists():
                exported[file_name] = "file exists"
                continue
            if not source.is_file():
                raise FileNotFoundError(f"Model file not found: {source}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            exported[file_name] = "copied from filesystem"
        return exported

    def _safe_file_path(self, file_name):
        """Resolve a model file without allowing path traversal."""
        files_path = self.files_path.resolve()
        source = (files_path / file_name).resolve()
        if not source.is_relative_to(files_path):
            raise ValueError(f"Model file path escapes model Files directory: {file_name}")
        return source

    def get_ecsv_file_as_astropy_table(self, file_name):
        """Read an ECSV model file."""
        source = self._safe_file_path(file_name)
        if not source.is_file():
            raise FileNotFoundError(f"Model file not found: {source}")
        return Table.read(source, format="ascii.ecsv")


class SimulationModelReader:
    """Source-neutral read-only interface for simulation-model data."""

    def __init__(self, source):
        """Initialize the reader with a source implementation."""
        self._source = source

    @classmethod
    def from_files(cls, simulation_models_path):
        """Create a reader for a checked-out model repository."""
        return cls(FileSystemModelSource(simulation_models_path))

    @property
    def source_name(self):
        """Return a user-facing description of the selected source."""
        return self._source.source_name

    def get_model_versions(self, collection_name="telescopes"):
        """Return available model versions."""
        return self._source.get_model_versions(collection_name)

    def read_production_table(self, collection_name, model_version):
        """Read a production table."""
        model_version = resolve_version_to_latest_patch(
            model_version, self.get_model_versions(collection_name)
        )
        return self._source.read_production_table(collection_name, model_version)

    def get_model_parameter(
        self, parameter, site, array_element_name, parameter_version=None, model_version=None
    ):
        """Read one model parameter by parameter or model version."""
        collection = names.get_collection_name_from_parameter_name(parameter)
        if model_version:
            if isinstance(model_version, list):
                raise ValueError(
                    "Only one model version can be passed to get_model_parameter, not a list."
                )
            model_version = resolve_version_to_latest_patch(
                model_version, self.get_model_versions(collection)
            )
            production = self.read_production_table(collection, model_version)
            for element in reversed(
                self._get_array_element_list(array_element_name, site, production, collection)
            ):
                parameter_version = production["parameters"].get(element, {}).get(parameter)
                if parameter_version:
                    array_element_name = element
                    break
        return self._read_parameters(
            {parameter: parameter_version}, collection, array_element_name, site
        )

    def get_model_parameters(self, site, array_element_name, collection, model_version):
        """Read resolved parameters for an array element and model version."""
        model_version = resolve_version_to_latest_patch(
            model_version, self.get_model_versions(collection)
        )
        production = self.read_production_table(collection, model_version)
        parameters = {}
        for element in self._get_array_element_list(
            array_element_name, site, production, collection
        ):
            versions = production["parameters"].get(element, {})
            parameters.update(self._read_parameters(versions, collection, element, site))
        return {key: parameters[key] for key in sorted(parameters)}

    def get_array_elements(self, model_version, collection="telescopes"):
        """Return array elements in a model version."""
        table = self.read_production_table(collection, model_version)
        return sorted(entry for entry in table["parameters"] if "-design" not in entry)

    def get_design_model(self, model_version, array_element_name, collection="telescopes"):
        """Return the design model for an array element."""
        table = self.read_production_table(collection, model_version)
        return table["design_model"].get(array_element_name, array_element_name)

    def get_array_elements_of_type(self, array_element_type, model_version, collection):
        """Return non-design elements of one type."""
        return sorted(
            element
            for element in self.get_array_elements(model_version, collection)
            if element.startswith(array_element_type)
        )

    def get_simulation_configuration_parameters(
        self, simulation_software, site, array_element_name, model_version
    ):
        """Read CORSIKA or sim_telarray configuration parameters."""
        if simulation_software == "corsika":
            return self.get_model_parameters(None, None, "configuration_corsika", model_version)
        if simulation_software == "sim_telarray":
            if not site or not array_element_name:
                return {}
            return self.get_model_parameters(
                site, array_element_name, "configuration_sim_telarray", model_version
            )
        raise ValueError(f"Unknown simulation software: {simulation_software}")

    def export_model_files(self, parameters=None, file_names=None, dest=None):
        """Export model files through the selected source."""
        return self._source.export_model_files(parameters, file_names, dest)

    def get_ecsv_file_as_astropy_table(self, file_name):
        """Read an ECSV model file through the selected source."""
        return self._source.get_ecsv_file_as_astropy_table(file_name)

    def _read_parameters(self, parameter_versions, collection, instrument=None, site=None):
        """Read parameters and return them keyed by parameter name."""
        parameters = {
            post["parameter"]: post
            for post in self._source.read_parameters(
                parameter_versions, collection, instrument, site
            )
        }
        return {key: parameters[key] for key in sorted(parameters)}

    def _get_array_element_list(self, array_element_name, site, production, collection):
        """Return the design and concrete elements represented by a request."""
        if collection == "configuration_corsika":
            return ["xSTx-design"]
        if collection == "sites":
            return [f"OBS-{site}"]
        if names.is_design_type(array_element_name):
            return [array_element_name]
        if collection == "configuration_sim_telarray":
            source_collection = names.get_collection_name_from_array_element_name(
                array_element_name
            )
            production = self.read_production_table(source_collection, production["model_version"])
        design = production["design_model"].get(array_element_name)
        return [element for element in (design, array_element_name) if element]
