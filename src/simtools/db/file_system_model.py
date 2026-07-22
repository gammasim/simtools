"""Read simulation model parameters from files."""

import shutil
from copy import deepcopy
from pathlib import Path

from astropy.table import Table
from packaging.version import Version

from simtools.data_model.validate_data import DataValidator
from simtools.db import db_model_upload
from simtools.io import ascii_handler
from simtools.utils import names, value_conversion


class FileSystemModelHandler:
    """Read-only access to simulation model files.

    Parameters
    ----------
    simulation_models_path : str or Path
        Root directory containing ``simulation-models/productions`` and
        ``simulation-models/model_parameters``.
    """

    production_tables_cached = {}
    model_parameters_cached = {}
    model_versions_cached = {}

    def __init__(self, simulation_models_path):
        """Initialize and validate the simulation model path."""
        self.simulation_models_path = Path(simulation_models_path).expanduser().resolve()
        self.productions_path = self.simulation_models_path / "simulation-models/productions"
        self.model_parameters_path = (
            self.simulation_models_path / "simulation-models/model_parameters"
        )
        self.files_path = self.model_parameters_path / "Files"
        self._validate_model_path()

    @property
    def source_name(self):
        """Return a user-facing description of the model source."""
        return str(self.simulation_models_path)

    def _validate_model_path(self):
        """Validate the required simulation model directories."""
        if not self.simulation_models_path.exists():
            raise FileNotFoundError(
                f"Simulation models path does not exist: {self.simulation_models_path}"
            )
        for required_path in (self.productions_path, self.model_parameters_path):
            if not required_path.is_dir():
                raise FileNotFoundError(
                    f"Expected simulation models directory not found: {required_path}"
                )

    def get_model_versions(self):
        """Return semantically sorted production versions available in the model files."""
        cache_key = str(self.simulation_models_path)
        if cache_key not in self.model_versions_cached:
            versions = [path.name for path in self.productions_path.iterdir() if path.is_dir()]
            self.model_versions_cached[cache_key] = sorted(versions, key=Version)
        return list(self.model_versions_cached[cache_key])

    def read_production_table(self, collection_name, model_version):
        """Return an aggregated production table for a collection and model version."""
        cache_key = (str(self.simulation_models_path), str(model_version))
        if cache_key not in self.production_tables_cached:
            model_path = self.productions_path / str(model_version)
            if not model_path.is_dir():
                raise ValueError(f"Model version {model_version} not found in {self.source_name}")
            self.production_tables_cached[cache_key] = db_model_upload.read_production_tables(
                model_path
            )
        try:
            return deepcopy(self.production_tables_cached[cache_key][collection_name])
        except KeyError as exc:
            raise ValueError(
                f"No production table for collection {collection_name} and model version "
                f"{model_version} in {self.source_name}"
            ) from exc

    def query_model_parameters(self, query, collection_name):
        """Read parameter JSON files matching a DatabaseHandler query."""
        parameter_queries = query.get("$or", [query])
        instrument = self._resolve_instrument(query, collection_name)
        parameters = []
        for parameter_query in parameter_queries:
            parameter = parameter_query.get("parameter")
            parameter_version = parameter_query.get("parameter_version")
            if not parameter or not parameter_version:
                continue
            parameter_path = self._get_parameter_file_path(instrument, parameter, parameter_version)
            parameter_data = self._read_parameter_file(parameter_path)
            if self._matches_query(parameter_data, query):
                parameters.append(parameter_data)
        return parameters

    def _get_parameter_file_path(self, instrument, parameter, parameter_version):
        """Return the path for one model parameter version."""
        collection = names.get_collection_name_from_parameter_name(parameter)
        path = self.model_parameters_path
        if collection in ("configuration_sim_telarray", "configuration_corsika"):
            path /= collection
        if collection != "configuration_corsika":
            path /= instrument
        return path / parameter / f"{parameter}-{parameter_version}.json"

    @staticmethod
    def _resolve_instrument(query, collection_name):
        """Resolve the model parameter directory name represented by a query."""
        instrument = query.get("instrument")
        if instrument:
            return instrument
        if collection_name == "sites" and query.get("site"):
            return f"OBS-{query['site']}"
        if collection_name == "configuration_corsika":
            return "xSTx-design"
        raise ValueError(
            f"Filesystem lookup for collection {collection_name} requires an array element name"
        )

    def _read_parameter_file(self, parameter_path):
        """Read and cache one model parameter JSON file."""
        cache_key = str(parameter_path)
        if cache_key not in self.model_parameters_cached:
            if not parameter_path.is_file():
                raise FileNotFoundError(f"Model parameter file not found: {parameter_path}")
            parameter_data = ascii_handler.collect_data_from_file(file_name=parameter_path)
            parameter_data = DataValidator.validate_model_parameter(parameter_data)
            parameter_data["value"], base_unit, _ = value_conversion.get_value_unit_type(
                value=parameter_data["value"], unit_str=parameter_data.get("unit")
            )
            parameter_data["unit"] = value_conversion.normalize_dimensionless_unit(base_unit)
            self.model_parameters_cached[cache_key] = parameter_data
        return deepcopy(self.model_parameters_cached[cache_key])

    @staticmethod
    def _matches_query(parameter_data, query):
        """Return whether parameter metadata matches instrument and site filters."""
        instrument = query.get("instrument")
        if instrument and parameter_data.get("instrument") != instrument:
            return False
        site = query.get("site")
        parameter_sites = parameter_data.get("site")
        if site and isinstance(parameter_sites, list):
            return site in parameter_sites
        return not site or parameter_sites == site

    def export_model_files(self, parameters=None, file_names=None, dest=None):
        """Copy model files to a destination directory."""
        if dest is None:
            raise ValueError("Destination path is required to export model files.")
        names_to_export = self._normalize_file_names(file_names, parameters)
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

    @staticmethod
    def _normalize_file_names(file_names, parameters):
        """Normalize explicit file names or derive them from parameter metadata."""
        if file_names:
            return [file_names] if isinstance(file_names, str) else list(file_names)
        return [
            parameter["value"]
            for parameter in (parameters or {}).values()
            if isinstance(parameter, dict)
            and parameter.get("file")
            and parameter.get("value") is not None
        ]

    def _safe_file_path(self, file_name):
        """Resolve a model file while preventing paths outside the Files directory."""
        files_path = self.files_path.resolve()
        source = (files_path / file_name).resolve()
        if not source.is_relative_to(files_path):
            raise ValueError(f"Model file path escapes model Files directory: {file_name}")
        return source

    def get_ecsv_file_as_astropy_table(self, file_name):
        """Read an ECSV model file from the filesystem."""
        source = self._safe_file_path(file_name)
        if not source.is_file():
            raise FileNotFoundError(f"Model file not found: {source}")
        return Table.read(source, format="ascii.ecsv")

    @classmethod
    def clear_caches(cls):
        """Clear all filesystem model caches."""
        cls.production_tables_cached.clear()
        cls.model_parameters_cached.clear()
        cls.model_versions_cached.clear()
