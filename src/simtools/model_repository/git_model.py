"""Read simulation models directly from Git blobs."""

import logging
import threading
from copy import deepcopy
from io import BytesIO
from pathlib import Path, PurePosixPath
from time import perf_counter

from astropy.table import Table
from packaging.version import Version

from simtools.io import ascii_handler
from simtools.model_repository import files
from simtools.model_repository.git_backend import Pygit2ObjectStore
from simtools.model_repository.parsing import normalize_model_parameter
from simtools.utils import names

logger = logging.getLogger(__name__)
_PRODUCTIONS_PATH = PurePosixPath("simulation-models/productions")


class GitModelSource:
    """Read a fixed simulation-model revision from a local Git object store."""

    def __init__(self, repository_path, revision, object_store=None):
        """Open ``repository_path`` and resolve ``revision`` once."""
        self.repository_path = Path(repository_path).expanduser().resolve()
        self._object_store = object_store or Pygit2ObjectStore(self.repository_path)
        self.commit = self._object_store.resolve_revision(revision)
        self._model_versions = None
        self._snapshots = {}
        self._production_tables = {}
        self._parameters = {}
        self._lock = threading.RLock()

    @property
    def source_name(self):
        """Return the repository and immutable revision used for reads."""
        return f"{self.repository_path}@{self.commit}"

    @property
    def source_config(self):
        """Return the serializable source selection for worker processes."""
        return {
            "type": "git",
            "repository": str(self.repository_path),
            "commit": self.commit,
        }

    def is_configured(self):
        """Return whether the Git source was opened successfully."""
        return True

    def get_model_versions(self, collection_name="telescopes"):
        """Return semantically sorted production versions."""
        del collection_name
        if self._model_versions is None:
            prefix = _PRODUCTIONS_PATH
            versions = {
                PurePosixPath(path).relative_to(prefix).parts[0]
                for path in self._object_store.iter_files(self.commit, prefix.as_posix())
                if len(PurePosixPath(path).relative_to(prefix).parts) > 1
            }
            self._model_versions = sorted(versions, key=Version)
        return list(self._model_versions)

    def _warm_model_version(self, model_version):
        """Warm all production and referenced parameter data for a model version."""
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
                "Git model warm-up commit=%s model_version=%s production_blobs=%d "
                "parameter_blobs=%d duration=%.3fs",
                self.commit,
                model_version,
                len(documents),
                len(parameter_paths),
                perf_counter() - started,
            )

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
        """Read and normalize parameter blobs by name and version."""
        instrument = self._get_parameter_instrument(
            {"instrument": instrument, "site": site}, collection_name
        )
        paths = {
            self._parameter_path(collection_name, instrument, parameter, version): (
                parameter,
                version,
            )
            for parameter, version in parameter_versions.items()
        }
        self._read_parameter_paths(paths)
        parameters = []
        for path, (parameter, _version) in paths.items():
            if path not in self._parameters:
                continue
            parameter_data = self._parameters[path]
            scope = names.get_model_parameter_scope(collection_name, instrument, parameter)
            if self._matches_filters(parameter_data, scope, site):
                parameters.append(deepcopy(parameter_data))
        if not parameters:
            raise ValueError(f"No parameters found for {collection_name}: {parameter_versions}")
        return parameters

    def _production_documents(self, model_version):
        """Read all production JSON documents for a version and its patch history."""
        model_prefix = _PRODUCTIONS_PATH / model_version
        files_in_version = self._object_store.iter_files(self.commit, model_prefix.as_posix())
        model_names = [model_version]
        info_path = model_prefix / "info.yml"
        if info_path.as_posix() in files_in_version:
            info = ascii_handler.collect_data_from_bytes(
                self._object_store.read_blob(self.commit, info_path.as_posix()), info_path
            )
            if info.get("model_update") == "patch_update":
                model_names.extend(info.get("model_version_history", []))
        model_names = sorted(set(model_names), key=Version)
        documents = []
        for model_name in model_names:
            prefix = _PRODUCTIONS_PATH / model_name
            for path in self._object_store.iter_files(self.commit, prefix.as_posix()):
                if path.endswith(".json"):
                    documents.append(
                        (
                            model_name,
                            path,
                            ascii_handler.collect_data_from_bytes(
                                self._object_store.read_blob(self.commit, path), path
                            ),
                        )
                    )
        if not documents:
            raise ValueError(f"Model version {model_version} not found in {self.source_name}")
        return documents

    @staticmethod
    def _parameter_paths(tables):
        """Return unique parameter paths referenced by production tables."""
        paths = {}
        for collection, table in tables.items():
            for instrument, parameters in GitModelSource._parameter_sets(collection, table).items():
                if not isinstance(parameters, dict):
                    continue
                for parameter, version in parameters.items():
                    if not isinstance(version, str):
                        continue
                    path = GitModelSource._parameter_path(
                        collection, instrument, parameter, version
                    )
                    paths[path] = (parameter, version)
        return paths

    @staticmethod
    def _parameter_sets(collection, table):
        """Return parameter mappings grouped by their repository scope."""
        parameters = table.get("parameters", {})
        if collection in ("configuration_corsika", "configuration_sim_telarray"):
            return {"global": parameters}
        return parameters

    def _read_parameter_paths(self, paths):
        """Read all missing parameter paths in one source batch."""
        for path in paths:
            if path in self._parameters:
                continue
            try:
                data = ascii_handler.collect_data_from_bytes(
                    self._object_store.read_blob(self.commit, path), path
                )
            except FileNotFoundError:
                continue
            self._parameters[path] = normalize_model_parameter(data)

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
            f"Git lookup for collection {collection_name} requires an array element name"
        )

    @staticmethod
    def _parameter_path(collection_name, instrument, parameter, parameter_version):
        """Return a repository-relative parameter path."""
        scope = names.get_model_parameter_scope(collection_name, instrument, parameter)
        return (
            PurePosixPath("simulation-models/model_parameters")
            / scope
            / parameter
            / f"{parameter}-{parameter_version}.json"
        ).as_posix()

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
        """Stream referenced model files from Git into ``dest``."""
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
            source_path = self._safe_file_path(file_name)
            target = destination / file_name
            if target.exists():
                exported[file_name] = "file exists"
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                with self._object_store.open_blob(self.commit, source_path) as source:
                    with target.open("wb") as output:
                        while chunk := source.read(1024 * 1024):
                            output.write(chunk)
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"Model file not found at commit {self.commit}: {source_path}"
                ) from exc
            exported[file_name] = "copied from Git"
        return exported

    @staticmethod
    def _safe_file_path(file_name):
        """Resolve a model file path below the repository Files directory."""
        path = PurePosixPath(str(file_name))
        files_root = PurePosixPath("simulation-models/model_parameters/Files")
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Model file path escapes model Files directory: {file_name}")
        return (files_root / path).as_posix()

    def get_ecsv_file_as_astropy_table(self, file_name):
        """Read an ECSV model file from a Git blob."""
        source_path = self._safe_file_path(file_name)
        try:
            return Table.read(
                BytesIO(self._object_store.read_blob(self.commit, source_path)), format="ascii.ecsv"
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Model file not found at commit {self.commit}: {source_path}"
            ) from exc
