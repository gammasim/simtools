"""Read simulation models through a storage-neutral source."""

import io
import logging
import shutil
import threading
from copy import deepcopy
from pathlib import Path, PurePosixPath
from time import perf_counter

from astropy.table import Table
from packaging.version import Version

from simtools.data_model import parameter_catalog as catalog
from simtools.data_model import schema
from simtools.data_model.table_asset import validate_table_asset
from simtools.io import ascii_handler
from simtools.model_repository import files
from simtools.model_repository.asset_names import (
    SOURCE_VALUE_KEY,
    qualify_parameter_file_name,
)
from simtools.model_repository.parsing import normalize_model_parameter

logger = logging.getLogger(__name__)
_PRODUCTIONS_PATH = PurePosixPath("simulation-models/productions")
_PARAMETERS_PATH = PurePosixPath("simulation-models/model_parameters")


class ModelSource:
    """Read-only access to model data through a storage implementation."""

    def __init__(self, storage):
        """Initialize the source with model storage."""
        self.storage = storage
        self._model_versions = None
        self._snapshots = {}
        self._production_tables = {}
        self._parameters = {}
        self._lock = threading.RLock()

    @property
    def source_name(self):
        """Return the storage's user-facing identity."""
        return self.storage.name

    @property
    def source_config(self):
        """Return the serializable storage configuration."""
        return self.storage.config

    def is_configured(self):
        """Return whether the source is ready for reads."""
        return True

    def get_model_versions(self, collection_name="telescopes"):
        """Return semantically sorted production versions."""
        del collection_name
        if self._model_versions is None:
            prefix = _PRODUCTIONS_PATH.as_posix()
            versions = {
                PurePosixPath(path).relative_to(_PRODUCTIONS_PATH).parts[0]
                for path in self.storage.iter_files(prefix)
                if len(PurePosixPath(path).relative_to(_PRODUCTIONS_PATH).parts) > 1
            }
            self._model_versions = sorted(versions, key=Version)
        return list(self._model_versions)

    def get_production_descriptions(self):
        """Return model-version descriptions from production info files."""
        prefix = _PRODUCTIONS_PATH.as_posix()
        info_files = [
            PurePosixPath(path)
            for path in self.storage.iter_files(prefix)
            if PurePosixPath(path).parent.parent == _PRODUCTIONS_PATH
            and PurePosixPath(path).name in {"info.yaml", "info.yml"}
        ]

        def read_info(path):
            return ascii_handler.collect_data_from_bytes(
                self.storage.read_bytes(path.as_posix()), path
            )

        return files.collect_production_descriptions(info_files, read_info)

    def _warm_model_version(self, model_version):
        """Load production and referenced parameter data for one version."""
        model_version = str(model_version)
        with self._lock:
            if model_version in self._snapshots:
                return
            started = perf_counter()
            documents = self._production_documents(model_version)
            tables = files.read_production_tables_from_documents(model_version, documents)
            parameter_paths = self._parameter_paths(tables)
            self._read_parameter_paths(parameter_paths)
            for collection, table in tables.items():
                self._production_tables[(model_version, collection)] = table
            self._snapshots[model_version] = True
            logger.debug(
                "Model warm-up source=%s model_version=%s production_files=%d "
                "parameter_files=%d duration=%.3fs",
                self.source_name,
                model_version,
                len(documents),
                len(parameter_paths),
                perf_counter() - started,
            )

    def _production_documents(self, model_version):
        """Read production documents for a version and its patch history."""
        documents = []
        for model_name in self._model_history(model_version):
            for path in self._iter_files_or_empty(_PRODUCTIONS_PATH / model_name):
                if path.endswith(".json"):
                    documents.append((model_name, path, self._read_document(path)))
        if not documents:
            raise ValueError(f"Model version {model_version} not found in {self.source_name}")
        return documents

    def _model_history(self, model_version):
        """Return the version and, for patch updates, its history in version order."""
        model_names = [model_version]
        model_prefix = _PRODUCTIONS_PATH / model_version
        paths = self._iter_files_or_empty(model_prefix)
        for info_name in ("info.yml", "info.yaml"):
            info_path = (model_prefix / info_name).as_posix()
            if info_path in paths:
                info = self._read_document(info_path)
                if info.get("model_update") == "patch_update":
                    model_names.extend(info.get("model_version_history", []))
                break
        return sorted(set(model_names), key=Version)

    def _iter_files_or_empty(self, prefix):
        """List files below a prefix; an absent prefix has no files."""
        try:
            return self.storage.iter_files(prefix.as_posix())
        except FileNotFoundError:
            return []

    def _read_document(self, path):
        """Read and parse one JSON or YAML repository document."""
        return ascii_handler.collect_data_from_bytes(self.storage.read_bytes(path), path)

    @staticmethod
    def _parameter_paths(tables):
        """Return unique parameter paths referenced by production tables."""
        paths = {}
        for collection, table in tables.items():
            parameters = table.get("parameters", {})
            parameter_sets = (
                {"global": parameters}
                if collection in ("configuration_corsika", "configuration_sim_telarray")
                else parameters
            )
            for instrument, values in parameter_sets.items():
                if not isinstance(values, dict):
                    continue
                for parameter, version in values.items():
                    if isinstance(version, str):
                        path = ModelSource._parameter_path(
                            collection, instrument, parameter, version
                        )
                        paths[path] = (parameter, version)
        return paths

    @staticmethod
    def _parameter_path(collection_name, instrument, parameter, parameter_version):
        """Return a repository-relative parameter document path."""
        scope = catalog.get_model_parameter_scope(collection_name, instrument, parameter)
        return (
            _PARAMETERS_PATH / scope / parameter / f"{parameter}-{parameter_version}.json"
        ).as_posix()

    def _read_parameter_paths(self, paths):
        """Read and cache parameter documents by repository path."""
        for path in paths:
            if path in self._parameters:
                continue
            try:
                data = ascii_handler.collect_data_from_bytes(self.storage.read_bytes(path), path)
            except FileNotFoundError:
                continue
            self._parameters[path] = normalize_model_parameter(data)

    def read_production_table(self, collection_name, model_version):
        """Return an aggregated production table for a collection and version."""
        model_version = str(model_version)
        self._warm_model_version(model_version)
        try:
            return deepcopy(self._production_tables[(model_version, collection_name)])
        except KeyError as exc:
            raise ValueError(
                f"No production table for {collection_name} in model version {model_version}"
            ) from exc

    def read_parameters(self, parameter_versions, collection_name, instrument=None, site=None):
        """Read and filter normalized parameter documents."""
        instrument = self._get_parameter_instrument(
            {"instrument": instrument, "site": site}, collection_name
        )
        paths = {
            self._parameter_path(collection_name, instrument, parameter, version): parameter
            for parameter, version in parameter_versions.items()
        }
        self._read_parameter_paths(paths)
        result = []
        for path, parameter in paths.items():
            if path not in self._parameters:
                continue
            scope = catalog.get_model_parameter_scope(collection_name, instrument, parameter)
            data = self._parameters[path]
            if self._matches_filters(data, scope, site):
                result.append(deepcopy(data))
        if not result:
            raise ValueError(f"No parameters found for {collection_name}: {parameter_versions}")
        return result

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
            f"Model lookup for collection {collection_name} requires an array element name"
        )

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

    @staticmethod
    def resolve_parameter_asset(parameter_data):
        """Resolve a file-valued parameter relative to its parameter directory."""
        value = parameter_data.get(SOURCE_VALUE_KEY, parameter_data.get("value"))
        parameter = parameter_data.get("parameter")
        version = parameter_data.get("parameter_version")
        instrument = parameter_data.get("instrument") or "global"
        if not all(isinstance(part, str) for part in (value, parameter, version)):
            raise ValueError(
                "Model file access requires parameter, parameter_version, and value metadata."
            )
        value_path = PurePosixPath(value)
        if value_path.is_absolute() or ".." in value_path.parts:
            raise ValueError(f"Model asset path escapes parameter directory: {value!r}")
        return (_PARAMETERS_PATH / instrument / parameter / value_path).as_posix()

    def get_parameter_table(self, parameter_data):
        """Read and validate an ECSV table referenced by a parameter record."""
        value = parameter_data.get("value")
        if not isinstance(value, str) or not value.lower().endswith(".ecsv"):
            raise ValueError("Parameter does not reference an ECSV table")
        path = self.resolve_parameter_asset(parameter_data)
        model_schema = schema.get_model_parameter_schema(
            parameter_data.get("parameter"), parameter_data.get("model_parameter_schema_version")
        )
        entries = [entry for entry in model_schema.get("data", []) if entry.get("type") == "file"]
        try:
            raw = self.storage.read_bytes(path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Model file not found in {self.source_name}: {path}") from exc
        return validate_table_asset(
            Table.read(io.BytesIO(raw), format="ascii.ecsv"),
            schema_entry=entries[0] if entries else None,
            parameter_data=parameter_data,
        )

    def export_model_files(self, parameters=None, file_names=None, dest=None):
        """Copy referenced model files to a destination directory."""
        if dest is None:
            raise ValueError("Destination path is required to export model files.")
        if parameters is None and file_names:
            raise ValueError("Model file export requires parameter metadata")
        selected = [
            p
            for p in (parameters or {}).values()
            if isinstance(p, dict) and p.get("file") and p.get("value")
        ]
        destination = Path(dest)
        destination.mkdir(parents=True, exist_ok=True)
        exported = {}
        for parameter in selected:
            target_name, status = self._export_parameter_file(parameter, destination)
            exported[parameter.get("value", target_name)] = status
        return exported

    def _export_parameter_file(self, parameter, destination):
        """Export one parameter asset; return the target name and export status.

        Identical files are skipped. A different ECSV file with the same name raises;
        other collisions are written with a scope suffix and the parameter value updated.
        """
        source = self.resolve_parameter_asset(parameter)
        scope = PurePosixPath(source).parent.parent.name
        target_name = qualify_parameter_file_name(parameter, fallback_instrument=scope)
        target = destination / target_name
        if not target.exists():
            self._copy(source, target)
            return target_name, "copied from model source"
        if self._same_content(source, target):
            return target_name, "file exists"
        if Path(source).suffix.lower() == ".ecsv":
            raise FileExistsError(f"Refusing to overwrite colliding model asset '{target.name}'")
        original_name = parameter.get(SOURCE_VALUE_KEY, parameter.get("value", target_name))
        target_name = self._get_collision_file_name(original_name, source)
        parameter.setdefault(SOURCE_VALUE_KEY, original_name)
        parameter["value"] = target_name
        target = destination / target_name
        if not target.exists():
            self._copy(source, target)
            return target_name, "copied from model source"
        if not self._same_content(source, target):
            raise FileExistsError(f"Refusing to overwrite colliding model asset '{target.name}'")
        return target_name, "file exists"

    def _same_content(self, source, target):
        """Return whether a stored asset and a local file have equal content."""
        with self.storage.open(source) as stream:
            return self._matches_file(stream, target)

    @staticmethod
    def _get_collision_file_name(file_name, source):
        """Return a deterministic filename qualified by the model scope."""
        path = Path(file_name)
        scope = PurePosixPath(source).parent.parent.name
        return f"{path.stem}-{scope}{path.suffix}"

    @staticmethod
    def _matches_file(stream, target):
        """Compare a streamed storage asset with a local file in chunks."""
        with target.open("rb") as existing:
            while True:
                chunk = stream.read(1024 * 1024)
                if chunk != existing.read(1024 * 1024):
                    return False
                if not chunk:
                    return True

    def _copy(self, source, target):
        """Copy a stored asset to a local destination."""
        target.parent.mkdir(parents=True, exist_ok=True)
        with self.storage.open(source) as input_file, target.open("wb") as output_file:
            shutil.copyfileobj(input_file, output_file)
