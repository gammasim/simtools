"""Adapt the MongoDB model handler to the model-reader source protocol."""

from copy import deepcopy
from pathlib import Path

from simtools.data_model import schema
from simtools.data_model.table_asset import validate_table_asset


class MongoDBModelSource:
    """Read-only model source backed by a configured ``DatabaseHandler``."""

    def __init__(self, database_handler):
        """Initialize the source with a database handler."""
        self.database_handler = database_handler
        self._model_versions = {}
        self._production_tables = {}
        self._parameters = {}

    @property
    def source_name(self):
        """Return the configured database name."""
        return self.database_handler.model_source_name

    @property
    def source_config(self):
        """Return the serializable source selection for worker processes."""
        return {"type": "mongodb", "name": self.source_name}

    def is_configured(self):
        """Return whether the wrapped database handler is configured."""
        return self.database_handler.is_configured()

    def get_model_versions(self, collection_name="telescopes"):
        """Return available model versions."""
        if collection_name not in self._model_versions:
            self._model_versions[collection_name] = list(
                self.database_handler.get_model_versions(collection_name)
            )
        return list(self._model_versions[collection_name])

    def read_production_table(self, collection_name, model_version):
        """Read a production table."""
        key = (collection_name, str(model_version))
        if key not in self._production_tables:
            self._production_tables[key] = deepcopy(
                self.database_handler.read_production_table_from_db(collection_name, model_version)
            )
        return deepcopy(self._production_tables[key])

    def read_parameters(self, parameter_versions, collection_name, instrument=None, site=None):
        """Read parameter documents by name and version."""
        key = repr((parameter_versions, collection_name, instrument, site))
        if key in self._parameters:
            return deepcopy(self._parameters[key])
        query = {
            "$or": [
                {"parameter": parameter, "parameter_version": version}
                for parameter, version in parameter_versions.items()
            ]
        }
        if instrument and instrument != "global":
            query["instrument"] = instrument
        if site and instrument != "global":
            query["site"] = site
        parameters = self.database_handler.read_parameter_documents(query, collection_name)
        self._parameters[key] = deepcopy(list(parameters.values()))
        for parameter_data in self._parameters[key]:
            value = parameter_data.get("value")
            if (
                parameter_data.get("file")
                and isinstance(value, str)
                and value.endswith(".ecsv")
                and parameter_data.get("model_parameter_schema_version") == "0.3.0"
            ):
                self.get_parameter_table(parameter_data)
        return deepcopy(self._parameters[key])

    def export_model_files(self, parameters=None, file_names=None, dest=None):
        """Export model files."""
        return self.database_handler.export_model_files(
            parameters=parameters, file_names=file_names, dest=dest
        )

    def get_ecsv_file_as_astropy_table(self, file_name, parameter_data=None):
        """Read an ECSV model file."""
        if parameter_data is not None:
            return self.get_parameter_table(parameter_data)
        return self.database_handler.get_ecsv_file_as_astropy_table(file_name)

    def get_parameter_table(self, parameter_data):
        """Read and validate an ECSV table referenced by a parameter record."""
        value = parameter_data.get("value")
        if not isinstance(value, str) or not value.lower().endswith(".ecsv"):
            raise ValueError("Parameter does not reference an ECSV table")
        table = self.get_ecsv_file_as_astropy_table(Path(value).name)
        schema_dict = schema.get_model_parameter_schema(
            parameter_data.get("parameter"), parameter_data.get("model_parameter_schema_version")
        )
        entries = [entry for entry in schema_dict.get("data", []) if entry.get("type") == "file"]
        return validate_table_asset(
            table,
            schema_entry=entries[0] if entries else None,
            parameter_data=parameter_data,
        )
