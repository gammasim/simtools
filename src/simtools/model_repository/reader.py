"""Read simulation models through a source-neutral interface."""

import filecmp
import shutil
from copy import deepcopy
from pathlib import Path

from astropy.table import Table
from packaging.version import Version

from simtools import settings
from simtools.data_model import schema
from simtools.data_model.table_asset import read_ecsv_asset, resolve_asset_path
from simtools.io import ascii_handler
from simtools.model_repository import files
from simtools.model_repository.git_model import GitModelSource
from simtools.model_repository.parsing import normalize_model_parameter
from simtools.simtel import simtel_table_reader
from simtools.utils import names
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

    def is_configured(self):
        """Return whether this source is ready to read."""
        return True

    @property
    def source_name(self):
        """Return a user-facing description of the model source."""
        return str(self.simulation_models_path)

    @property
    def source_config(self):
        """Return the serializable source selection for worker processes."""
        return {"type": "filesystem", "path": str(self.simulation_models_path)}

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
        instrument = self._get_parameter_instrument(query, collection_name)
        parameters = [
            parameter
            for parameter_query in parameter_queries
            if (
                parameter := self._read_query_parameter(
                    parameter_query, query, collection_name, instrument
                )
            )
        ]
        if not parameters:
            raise ValueError(f"No parameters found for {collection_name}: {query}")
        return parameters

    def read_parameters(self, parameter_versions, collection_name, instrument=None, site=None):
        """Read parameter files by name and version."""
        instrument = self._get_parameter_instrument(
            {"instrument": instrument, "site": site}, collection_name
        )

        parameters = []
        for parameter, parameter_version in parameter_versions.items():
            parameter_scope = names.get_model_parameter_scope(
                collection_name, instrument, parameter
            )
            parameter_path = self._parameter_path(
                collection_name, parameter_scope, parameter, parameter_version
            )
            if not parameter_path.is_file():
                continue
            parameter_data = self._read_parameter_file(parameter_path)
            value = parameter_data.get("value")
            if (
                parameter_data.get("file")
                and isinstance(value, str)
                and value.lower().endswith(".ecsv")
                and parameter_data.get("model_parameter_schema_version") == "0.3.0"
            ):
                self.get_parameter_table(parameter_data)
            if self._matches_filters(parameter_data, parameter_scope, site):
                parameters.append(parameter_data)
        if not parameters:
            raise ValueError(f"No parameters found for {collection_name}: {parameter_versions}")
        return parameters

    @staticmethod
    def _get_parameter_instrument(query, collection_name):
        """Resolve the instrument scope used for a parameter lookup."""
        instrument = query.get("instrument")
        if instrument:
            return instrument
        if collection_name == "sites" and query.get("site"):
            return f"OBS-{query['site']}"
        if collection_name in ("configuration_corsika", "configuration_sim_telarray"):
            return "global"
        raise ValueError(
            f"Filesystem lookup for collection {collection_name} requires an array element name"
        )

    def _read_query_parameter(self, parameter_query, query, collection_name, instrument):
        """Read one parameter selected by a source query."""
        parameter = parameter_query.get("parameter")
        parameter_version = parameter_query.get("parameter_version")
        if not parameter or not parameter_version:
            return None
        parameter_scope = names.get_model_parameter_scope(collection_name, instrument, parameter)
        parameter_path = self._parameter_path(
            collection_name, parameter_scope, parameter, parameter_version
        )
        if not parameter_path.is_file():
            return None
        parameter_data = self._read_parameter_file(parameter_path)
        return (
            parameter_data
            if self._matches_filters(parameter_data, parameter_scope, query.get("site"))
            else None
        )

    def _parameter_path(self, collection_name, instrument, parameter, parameter_version):
        """Return the path for one parameter version."""
        scope = names.get_model_parameter_scope(collection_name, instrument, parameter)
        return (
            self.model_parameters_path / scope / parameter / f"{parameter}-{parameter_version}.json"
        )

    def _read_parameter_file(self, parameter_path):
        """Read and cache one parameter JSON file."""
        key = str(parameter_path)
        if key not in self._parameters:
            data = ascii_handler.collect_data_from_file(file_name=parameter_path)
            data = normalize_model_parameter(data)
            self._parameters[key] = data
        return deepcopy(self._parameters[key])

    @staticmethod
    def _matches_filters(parameter_data, instrument, site):
        """Return whether parameter metadata matches source filters."""
        if instrument == "global":
            instrument = None
        if instrument and parameter_data.get("instrument") != instrument:
            return False
        parameter_sites = parameter_data.get("site")
        if site and isinstance(parameter_sites, list):
            return site in parameter_sites
        if not site or parameter_sites == site:
            return True
        return instrument is None and parameter_sites is None

    def export_model_files(self, parameters=None, file_names=None, dest=None):
        """Copy referenced model files to a destination directory."""
        if dest is None:
            raise ValueError("Destination path is required to export model files.")
        destination = Path(dest)
        destination.mkdir(parents=True, exist_ok=True)
        return {
            source.name: self._copy_model_file(parameter, source, destination)
            for parameter in self._files_to_export(parameters, file_names)
            for source in [self.resolve_parameter_asset(parameter)]
        }

    @staticmethod
    def _files_to_export(parameters, file_names):
        """Return parameter records selected for export."""
        if parameters is not None and file_names is None:
            return [
                parameter
                for parameter in parameters.values()
                if isinstance(parameter, dict) and parameter.get("file") and parameter.get("value")
            ]
        names_to_export = [file_names] if isinstance(file_names, str) else file_names or []
        return [
            {"value": file_name, "asset_location": "shared_files"} for file_name in names_to_export
        ]

    def _copy_model_file(self, parameter, source, destination):
        """Copy one resolved model asset and return its export status."""
        target = destination / source.name
        if target.exists():
            if filecmp.cmp(source, target, shallow=False):
                return "file exists"
            raise FileExistsError(
                f"Refusing to overwrite colliding model asset '{target.name}' in {destination}"
            )
        if not source.is_file():
            raise FileNotFoundError(f"Model file not found: {source}")
        if source.suffix.lower() == ".ecsv" and parameter.get("parameter"):
            self.get_parameter_table(parameter)
        shutil.copy2(source, target)
        return "copied from filesystem"

    def resolve_parameter_asset(self, parameter_data):
        """Resolve a parameter asset from its declared location."""
        value = parameter_data.get("value") if isinstance(parameter_data, dict) else parameter_data
        if not isinstance(value, str):
            raise ValueError(f"Model asset value must be a relative filename, got {value!r}")
        parameter = parameter_data.get("parameter") if isinstance(parameter_data, dict) else None
        version = (
            parameter_data.get("parameter_version") if isinstance(parameter_data, dict) else None
        )
        instrument = parameter_data.get("instrument") if isinstance(parameter_data, dict) else None
        if parameter and version:
            scope = instrument or "global"
            parameter_file = (
                self.model_parameters_path / scope / parameter / f"{parameter}-{version}.json"
            )
        else:
            parameter_file = self.files_path / value
        asset_location = (
            parameter_data.get("asset_location", "parameter_directory")
            if isinstance(parameter_data, dict)
            else "shared_files"
        )
        return resolve_asset_path(value, parameter_file, self.files_path, asset_location)

    def get_parameter_table(self, parameter_data):
        """Resolve and validate an ECSV table referenced by a parameter record."""
        parameter = parameter_data.get("parameter")
        schema_version = parameter_data.get("model_parameter_schema_version")
        schema_dict = schema.get_model_parameter_schema(parameter, schema_version)
        data_entries = [
            entry for entry in schema_dict.get("data", []) if entry.get("type") == "file"
        ]
        schema_entry = data_entries[0] if data_entries else None
        return read_ecsv_asset(
            self.resolve_parameter_asset(parameter_data),
            schema_entry=schema_entry,
            parameter_data=parameter_data,
        )

    def _safe_file_path(self, file_name):
        """Resolve a model file without allowing path traversal."""
        files_path = self.files_path.resolve()
        source = (files_path / file_name).resolve()
        if not source.is_relative_to(files_path):
            raise ValueError(f"Model file path escapes model Files directory: {file_name}")
        return source

    def get_ecsv_file_as_astropy_table(self, file_name, parameter_data=None):
        """Read an ECSV model file."""
        source = (
            self.resolve_parameter_asset(parameter_data)
            if parameter_data is not None
            else self._safe_file_path(file_name)
        )
        if not source.is_file():
            raise FileNotFoundError(f"Model file not found: {source}")
        if parameter_data is None:
            return Table.read(source, format="ascii.ecsv")
        return read_ecsv_asset(source, parameter_data=parameter_data)


class SimulationModelReader:
    """Source-neutral read-only interface for simulation-model data."""

    def __init__(self, source):
        """Initialize the reader with a source implementation."""
        self._source = source

    @classmethod
    def from_files(cls, simulation_models_path):
        """Create a reader for a checked-out model repository."""
        return cls(FileSystemModelSource(simulation_models_path))

    @classmethod
    def from_git(cls, repository_path, revision, object_store=None):
        """Create a reader for one immutable revision of a Git repository."""
        return cls(GitModelSource(repository_path, revision, object_store=object_store))

    @property
    def source_name(self):
        """Return a user-facing description of the selected source."""
        return self._source.source_name

    @property
    def source_config(self):
        """Return the serializable source selection for worker processes."""
        return getattr(self._source, "source_config", None)

    def is_configured(self):
        """Return whether the selected source is configured for reads."""
        return self._source.is_configured()

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
            if versions:
                parameters.update(self._read_parameters(versions, collection, element, site))
        return {key: parameters[key] for key in sorted(parameters)}

    def get_model_parameters_for_all_model_versions(self, site, array_element_name, collection):
        """Read resolved parameters for an element across all model versions."""
        parameters = {}
        for model_version in self.get_model_versions(collection):
            try:
                parameters[model_version] = self.get_model_parameters(
                    site, array_element_name, collection, model_version
                )
            except KeyError, ValueError:
                continue
        return parameters

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
            if not site:
                return {}
            return self.get_model_parameters(
                site, array_element_name, "configuration_sim_telarray", model_version
            )
        raise ValueError(f"Unknown simulation software: {simulation_software}")

    def export_model_files(self, parameters=None, file_names=None, dest=None):
        """Export model files through the selected source."""
        return self._source.export_model_files(parameters, file_names, dest)

    def get_parameter_table(self, parameter_data):
        """Return the validated Astropy table referenced by a model parameter."""
        return self._source.get_parameter_table(parameter_data)

    def export_model_file(
        self,
        parameter,
        site,
        array_element_name,
        model_version=None,
        parameter_version=None,
        export_file_as_table=False,
        dest=None,
    ):
        """Export one model file or return a file-backed value as a table."""
        parameters = self.get_model_parameter(
            parameter,
            site,
            array_element_name,
            parameter_version=parameter_version,
            model_version=model_version,
        )
        parameter_data = parameters[parameter]
        if parameter_data.get("type") == "dict" and isinstance(parameter_data.get("value"), dict):
            return (
                simtel_table_reader.row_data_to_astropy_table(parameter_data["value"])
                if export_file_as_table
                else None
            )
        if dest is None:
            raise ValueError("Destination path is required to export a model file.")
        self.export_model_files(parameters=parameters, dest=dest)
        if export_file_as_table:
            value = parameter_data.get("value")
            if (
                isinstance(value, str)
                and value.lower().endswith(".ecsv")
                and hasattr(self._source, "get_parameter_table")
            ):
                return self._source.get_parameter_table(parameter_data)
            return simtel_table_reader.read_simtel_table(
                parameter, Path(dest) / parameter_data["value"]
            )
        return None

    def get_ecsv_file_as_astropy_table(self, file_name, parameter_data=None):
        """Read an ECSV model file through the selected source."""
        if parameter_data is not None and hasattr(self._source, "get_parameter_table"):
            return self._source.get_parameter_table(parameter_data)
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
            return ["global"]
        if collection == "sites":
            return [f"OBS-{site}"]
        if collection == "configuration_sim_telarray":
            return self._get_sim_telarray_array_element_list(array_element_name, production)
        if names.is_design_type(array_element_name):
            return [array_element_name]
        design = production["design_model"].get(array_element_name)
        return [element for element in (design, array_element_name) if element]

    def _get_sim_telarray_array_element_list(self, array_element_name, production):
        """Return global, design, and concrete sim_telarray scopes."""
        design_model = None
        if array_element_name not in (None, "global") and not names.is_design_type(
            array_element_name
        ):
            source_collection = names.get_collection_name_from_array_element_name(
                array_element_name
            )
            telescope_production = self.read_production_table(
                source_collection, production["model_version"]
            )
            design_model = telescope_production["design_model"].get(array_element_name)
        try:
            return names.get_sim_telarray_parameter_scopes(
                array_element_name,
                design_model,
                settings.config.args.get("ignore_missing_design_model", False),
            )
        except KeyError as exc:
            raise KeyError(
                f"Failed to generate array element list for model query for {array_element_name}"
            ) from exc
