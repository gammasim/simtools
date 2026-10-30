"""Read simulation models through a source-neutral interface."""

from copy import deepcopy
from pathlib import Path

from simtools import context as run_context
from simtools.data_model import parameter_catalog as catalog
from simtools.model_repository import source as model_repository_source
from simtools.model_repository.asset_names import ECSV_SUFFIX, SOURCE_VALUE_KEY
from simtools.model_repository.storage import DirectoryStorage, GitStorage, MemoryStorage
from simtools.utils import names
from simtools.version import resolve_version_to_latest_patch


class SimulationModelReader:
    """Source-neutral read-only interface for simulation-model data."""

    def __init__(self, source):
        """Initialize the reader with a source implementation."""
        self._source = source
        self._parameter_tables = {}
        self._parameter_table_records = {}

    @classmethod
    def from_files(cls, simulation_models_path):
        """Create a reader for a checked-out model repository."""
        storage = DirectoryStorage(simulation_models_path)
        if not storage.root.exists():
            raise FileNotFoundError(f"Simulation models path does not exist: {storage.root}")
        for required in ("simulation-models/productions", "simulation-models/model_parameters"):
            if not (storage.root / required).is_dir():
                raise FileNotFoundError(
                    f"Expected simulation models directory not found: {storage.root / required}"
                )
        return cls(model_repository_source.ModelSource(storage))

    @classmethod
    def from_git(cls, repository_path, revision, object_store=None):
        """Create a reader for one immutable revision of a Git repository."""
        return cls(
            model_repository_source.ModelSource(
                GitStorage(repository_path, revision, object_store=object_store)
            )
        )

    @classmethod
    def from_memory(cls, files):
        """Create a reader from in-memory model files."""
        return cls(model_repository_source.ModelSource(MemoryStorage(files)))

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

    def get_production_descriptions(self):
        """Return model-version descriptions from the selected source."""
        return self._source.get_production_descriptions()

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
        collection = catalog.get_collection_name_from_parameter_name(parameter)
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
                self._get_array_element_list(
                    array_element_name,
                    site,
                    production,
                    collection,
                    run_context.RunContext.from_environment(),
                )
            ):
                parameter_version = production["parameters"].get(element, {}).get(parameter)
                if parameter_version:
                    array_element_name = element
                    break
        return self._read_parameters(
            {parameter: parameter_version}, collection, array_element_name, site
        )

    def get_model_parameters(
        self,
        site,
        array_element_name,
        collection,
        model_version,
        parameter_names=None,
        context=None,
    ):
        """Read resolved parameters for an array element and model version.

        Parameters
        ----------
        parameter_names : iterable of str, optional
            If supplied, read only these parameters. This is useful for callers
            that need a small subset of a model, such as coordinate transforms.
        """
        if parameter_names is not None:
            parameter_names = frozenset(parameter_names)
        context = context or run_context.RunContext.from_environment()
        model_version = resolve_version_to_latest_patch(
            model_version, self.get_model_versions(collection)
        )
        production = self.read_production_table(collection, model_version)
        parameters = {}
        for element in self._get_array_element_list(
            array_element_name, site, production, collection, context
        ):
            versions = production["parameters"].get(element, {})
            if parameter_names is not None:
                versions = {
                    parameter: version
                    for parameter, version in versions.items()
                    if parameter in parameter_names
                }
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

    def _export_file_parameter_data(
        self,
        parameter,
        parameter_data,
        parameters,
        output_file,
        export_model_file,
        export_model_file_as_table,
        dest,
    ):
        """Export a file-backed parameter and optionally its ECSV table."""
        exported = self.export_model_files(parameters=parameters, dest=dest)
        if not exported:
            raise ValueError(f"Parameter {parameter} does not reference an exportable model file.")

        source_file = Path(dest) / next(iter(exported))
        model_output_file = Path(dest) / output_file if output_file else source_file
        output_files = []
        table = (
            self._read_exported_parameter_table(parameter, parameter_data, source_file)
            if export_model_file_as_table
            else None
        )

        if export_model_file:
            model_output_file.parent.mkdir(parents=True, exist_ok=True)
            if source_file != model_output_file:
                source_file.rename(model_output_file)
            output_files.append(model_output_file)

        if export_model_file_as_table:
            table_output_file = model_output_file.with_suffix(ECSV_SUFFIX)
            table_output_file.parent.mkdir(parents=True, exist_ok=True)
            table.write(table_output_file, format="ascii.ecsv", overwrite=True)
            if table_output_file not in output_files:
                output_files.append(table_output_file)
            if not export_model_file and source_file != table_output_file and source_file.exists():
                source_file.unlink()

        return output_files

    def _read_exported_parameter_table(self, parameter, parameter_data, _source_file):
        """Read an exported file-backed parameter as an Astropy table."""
        value = parameter_data.get("value")
        if isinstance(value, str) and value.lower().endswith(ECSV_SUFFIX):
            return self.get_parameter_table(parameter_data)
        raise ValueError(
            f"Parameter '{parameter}' is not an ECSV model table and cannot be exported as a table"
        )

    def get_parameter_table(self, parameter_data):
        """Return the validated Astropy table referenced by a model parameter."""
        cache_key = self._parameter_table_cache_key(parameter_data)
        if cache_key not in self._parameter_tables:
            self._parameter_tables[cache_key] = self._source.get_parameter_table(parameter_data)
        return deepcopy(self._parameter_tables[cache_key])

    @staticmethod
    def _parameter_table_cache_key(parameter_data):
        """Return a stable cache key for one model-parameter table."""
        return repr(
            (
                parameter_data.get("parameter"),
                parameter_data.get("parameter_version"),
                parameter_data.get("instrument"),
                parameter_data.get("site"),
                parameter_data.get(SOURCE_VALUE_KEY, parameter_data.get("value")),
            )
        )

    def get_parameter_table_records(self, parameter_data):
        """Return cached scalar records for a validated model-parameter table."""
        cache_key = self._parameter_table_cache_key(parameter_data)
        if cache_key not in self._parameter_table_records:
            table = self.get_parameter_table(parameter_data)
            column_values = [
                [getattr(value, "value", value) for value in table[column]]
                for column in table.colnames
            ]
            self._parameter_table_records[cache_key] = [
                dict(zip(table.colnames, row)) for row in zip(*column_values)
            ]
        return deepcopy(self._parameter_table_records[cache_key])

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
            if export_file_as_table:
                raise ValueError("Structured JSON model parameters are not ECSV tables")
            return None
        if dest is None:
            raise ValueError("Destination path is required to export a model file.")
        self.export_model_files(parameters=parameters, dest=dest)
        if export_file_as_table:
            value = parameter_data.get("value")
            if isinstance(value, str) and value.lower().endswith(ECSV_SUFFIX):
                return self.get_parameter_table(parameter_data)
            raise ValueError(
                f"Parameter '{parameter}' is not an ECSV model table and cannot be "
                "exported as a table"
            )
        return None

    def _read_parameters(self, parameter_versions, collection, instrument=None, site=None):
        """Read parameters and return them keyed by parameter name."""
        parameters = {
            post["parameter"]: post
            for post in self._source.read_parameters(
                parameter_versions, collection, instrument, site
            )
        }
        return {key: parameters[key] for key in sorted(parameters)}

    def _get_array_element_list(
        self, array_element_name, site, production, collection, context=None
    ):
        """Return the design and concrete elements represented by a request."""
        context = context or run_context.RunContext.from_environment()
        if collection == "configuration_corsika":
            return ["global"]
        if collection == "sites":
            return [f"OBS-{site}"]
        if collection == "configuration_sim_telarray":
            return self._get_sim_telarray_array_element_list(
                array_element_name, production, context
            )
        if names.is_design_type(array_element_name):
            return [array_element_name]
        design = production["design_model"].get(array_element_name)
        return [element for element in (design, array_element_name) if element]

    def _get_sim_telarray_array_element_list(self, array_element_name, production, context=None):
        """Return global, design, and concrete sim_telarray scopes."""
        context = context or run_context.RunContext.from_environment()
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
            return catalog.get_sim_telarray_parameter_scopes(
                array_element_name,
                design_model,
                context.args.get("ignore_missing_design_model", False),
            )
        except KeyError as exc:
            raise KeyError(
                f"Failed to generate array element list for model query for {array_element_name}"
            ) from exc
