import logging

from astropy.io.registry.base import IORegistryError
from astropy.table import QTable

from simtools.data_model import validate_data
from simtools.data_model.metadata_collector import MetadataCollector

__all__ = ["read_table_from_file"]

_logger = logging.getLogger(__name__)


def read_table_from_file(file_name, schema_file=None, validate=False, metadata_file=None):
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

    try:
        data_table = QTable.read(file_name)
    except (FileNotFoundError, IORegistryError) as exc:
        _logger.error("Error reading tabled data from %s", file_name)
        raise exc
    _logger.info("Reading table data from %s", file_name)

    if validate:
        metadata = MetadataCollector(
            args_dict=None,
            metadata_file_name=(metadata_file if metadata_file is not None else file_name),
            data_model_name=None,
        )

        _validator = validate_data.DataValidator(
            schema_file=(
                schema_file
                if schema_file is not None
                else metadata.get_data_model_schema_file_name()
            ),
            data_table=data_table,
        )
        return _validator.validate_and_transform()

    return data_table
