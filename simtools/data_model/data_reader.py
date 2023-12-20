import logging

from astropy.table import QTable

from simtools.data_model import validate_data
from simtools.data_model.metadata_collector import MetadataCollector

__all__ = ["DataReader"]


class DataReader:
    """
    Reader for simulation data and metadata. Data includes input to simulation tools
    and reading of simulation model parameters. Allows to validate data against a schema
    at the time of reading.

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
            reader.data_table = QTable.read(file_name, format="ascii.ecsv")
        except FileNotFoundError as exc:
            reader.logger.error("Error reading list of array elements from %s", file_name)
            raise exc
        reader.logger.info("Reading table data from %s", file_name)

        reader.read_metadata(file_name, metadata_file)

        # TMPTMP
        validate = False
        return reader.validate_and_transform(schema_file) if validate else reader.data_table

    def read_metadata(self, file_name, metadata_file=None):
        """
        Read metadata from file.

        Parameters:
        -----------
        file_name: str or Path
            Name of file to be read.
        metadata_file: str or Path
            Name of metadata file to be read.

        """
        print(file_name, metadata_file)

        if self.data_table:
            self.metadata = MetadataCollector(
                args_dict=None, metadata_file_name=file_name, data_model_name=None
            )

    def validate_and_transform(self, schema_file=None):
        """
        Validate data using jsonschema. If necessary, transform data to match schema requirements.

        Parameters:
        -----------
        schema_file: str or Path
            Name of schema file to be used for validation.

        Returns:
        --------
        astropy Table
            Table read from file.

        Raises
        ------
        FileNotFoundError
            If file does not exist.

        """

        schema_file = self._get_schema_file(self.data_table) if schema_file is None else schema_file

        _validator = validate_data.DataValidator(
            schema_file=schema_file,
            data_table=self.data_table,
        )
        return _validator.validate_and_transform()

    def _get_schema_file(self, table):
        """
        Get schema file name.

        Parameters:
        -----------
        table: astropy Table
            Table to be validated.

        Returns:
        --------
        str
            Schema file name.

        Raises
        ------
        FileNotFoundError
            If file does not exist_.

        """

        try:
            _schema_file = table.meta[""]
        except KeyError:
            self.logger.debug("No metadata found in table")

        _schema_file = (
            "https://raw.githubusercontent.com/gammasim/"
            "workflows/main/schemas/array_coordinates_UTM.schema.yml"
        )

        return _schema_file

    def _get_metadata_from_table(self, table):
        """
        Get metadata from astropy table.
        Return empty dict in case no metadata is found.

        Parameters:
        -----------
        table: astropy Table
            Table with metadata

        Returns:
        --------
        dict
            Metadata.

        """

        try:
            return dict(table.meta)
        except TypeError:
            self.logger.info("No metadata found in table")

        return {}
