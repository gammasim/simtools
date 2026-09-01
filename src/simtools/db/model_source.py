"""Adapt the MongoDB model handler to the model-reader source protocol."""


class MongoDBModelSource:
    """Read-only model source backed by a configured ``DatabaseHandler``."""

    def __init__(self, database_handler):
        """Initialize the source with a database handler."""
        self.database_handler = database_handler

    @property
    def source_name(self):
        """Return the configured database name."""
        return self.database_handler.model_source_name

    def get_model_versions(self, collection_name="telescopes"):
        """Return available model versions."""
        return self.database_handler.get_model_versions(collection_name)

    def read_production_table(self, collection_name, model_version):
        """Read a production table."""
        return self.database_handler.read_production_table_from_db(collection_name, model_version)

    def read_parameters(self, parameter_versions, collection_name, instrument=None, site=None):
        """Read parameter documents by name and version."""
        query = {
            "$or": [
                {"parameter": parameter, "parameter_version": version}
                for parameter, version in parameter_versions.items()
            ]
        }
        if instrument and instrument != "global":
            query["instrument"] = instrument
        if site:
            query["site"] = site
        parameters = self.database_handler._read_db(  # pylint: disable=protected-access
            query, collection_name
        )
        return list(parameters.values())

    def export_model_files(self, parameters=None, file_names=None, dest=None):
        """Export model files."""
        return self.database_handler.export_model_files(
            parameters=parameters, file_names=file_names, dest=dest
        )

    def get_ecsv_file_as_astropy_table(self, file_name):
        """Read an ECSV model file."""
        return self.database_handler.get_ecsv_file_as_astropy_table(file_name)
