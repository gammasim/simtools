import logging

from astropy.table import QTable

from simtools.data_model import validate_data
from simtools.data_model.metadata_collector import MetadataCollector

__all__ = ["DataReader"]


class DataReader:
    """
    Data reader for input data to simulation tools. Reads data and metadata from file
    and allows to validate data against a schema.

    """

    def __init__(self):
        """
        Initialize data reader.

        """
        self.logger = logging.getLogger(__name__)
        self.data_table = None
        self.metadata = None

    @staticmethod
    def read_table_from_file(file_name, schema_file=None, validate=True, metadata_file=None):
        """
        Read astropy table from file and validate against schema.
        Metadata is read from metadata file or from the metadata section of the data file.
        Schema for validation can be given as argument, or is determined
        from the metadata associated to the file.

        Parameters:
        -----------
        file_name: str or Path
            Name of file to be read.
        schema_file: str or Path
            Name of schema file to be used for validation.
        validate: bool
            Validate data against schema.
        metadata_file: str or Path
            Name of metadata file to be read.

        Returns:
        --------
        astropy Table
            Table read from file.

        Raises
        ------
        FileNotFoundError
            If file does not exist.

        """

        reader = DataReader()
        try:
            reader.data_table = QTable.read(file_name)
        except FileNotFoundError as exc:
            reader.logger.error("Error reading tabled data from %s", file_name)
            raise exc
        reader.logger.info("Reading table data from %s", file_name)

        validate = False
        if validate:
            reader.read_metadata(
                metadata_file=metadata_file if metadata_file is not None else file_name
            )

            schema_file = (
                schema_file
                if schema_file is not None
                else reader.metadata.get_data_model_schema_file_name()
            )

        return (
            reader.validate_and_transform(schema_file=schema_file)
            if validate
            else reader.data_table
        )

    def read_metadata(self, metadata_file):
        """
        Read metadata from file (either a metadata file in yaml format or from
        the metadata section of a data file)

        Parameters:
        -----------
        metadata_file: str or Path
            Name of metadata file to be read.

        """

        self.metadata = MetadataCollector(
            args_dict=None, metadata_file_name=metadata_file, data_model_name=None
        )

    def validate_and_transform(self, schema_file):
        """
        Validate and transform data using the DataValidator module.

        Parameters:
        -----------
        schema_file: str or Path
            Name of schema file to be used for validation.

        Returns:
        --------
        astropy Table
            Table read from file.

        """

        _validator = validate_data.DataValidator(
            schema_file=schema_file,
            data_table=self.data_table,
        )
        return _validator.validate_and_transform()
