import logging
from pathlib import Path

import astropy
import yaml

import simtools.utils.general as gen
from simtools.data_model import validate_data
from simtools.io_operations import io_handler

__all__ = ["ModelDataWriter"]


class ModelDataWriter:
    """
    Writer for simulation model data and metadata.

    Parameters
    ----------
    args_dict: Dictionary
        Dictionary with configuration parameters.
    """

    def __init__(self, product_data_file=None, product_data_format=None):
        """
        Initialize model data writer.
        """

        self._logger = logging.getLogger(__name__)
        self.io_handler = io_handler.IOHandler()
        try:
            self.product_data_file = self.io_handler.get_output_file(
                file_name=product_data_file, dir_type="simtools-result"
            )
        except TypeError:
            self.product_data_file = None
        self.product_data_format = self._astropy_data_format(product_data_format)

    @staticmethod
    def dump(args_dict, metadata=None, product_data=None, validate_schema_file=False):
        """
        Write model data and metadata (as static method).

        Parameters
        ----------
        args_dict: dict
            Dictionary with configuration parameters (including output file name and path).
        metadata: dict
            Metadata to be written.
        product_data: astropy Table
            Model data to be written
        validate_schema_file: str
            Schema file used in validation of output data.

        """

        writer = ModelDataWriter(
            product_data_file=args_dict.get("output_file", None),
            product_data_format=args_dict.get("output_file_format", "ascii.ecsv"),
        )
        if validate_schema_file and not args_dict.get("skip_output_validation", True):
            product_data = writer.validate_and_transform(
                product_data=product_data,
                validate_schema_file=validate_schema_file,
            )
        writer.write(metadata=metadata, product_data=product_data)

    def write(self, metadata=None, product_data=None):
        """
        Write model data and metadata

        Parameters
        ----------
        metadata: dict
            Metadata to be written.
        product_data: astropy Table
            Model data to be written

        """

        if metadata is not None:
            self.write_metadata(metadata=metadata)
        if product_data is not None:
            self.write_data(product_data=product_data)

    def validate_and_transform(self, product_data=None, validate_schema_file=None):
        """
        Validate product data using jsonschema given in metadata.
        If necessary, transform product data to match schema.

        Parameters
        ----------
        product_data: astropy Table
            Model data to be validated
        validate_schema_file: str
            Schema file used in validation of output data.

        """

        _validator = validate_data.DataValidator(
            schema_file=validate_schema_file,
            data_table=product_data,
        )
        return _validator.validate_and_transform()

    def write_data(self, product_data):
        """
        Write model data.

        Parameters
        ----------
        product_data: astropy Table
            Model data to be written.

        Raises
        ------
        FileNotFoundError
            if data writing was not sucessfull.

        """

        try:
            if product_data is not None:
                self._logger.info(f"Writing data to {self.product_data_file}")
                product_data.write(
                    self.product_data_file, format=self.product_data_format, overwrite=True
                )
        except astropy.io.registry.base.IORegistryError:
            self._logger.error(f"Error writing model data to {self.product_data_file}.")
            raise

    def write_metadata(self, metadata, yml_file=None, keys_lower_case=False):
        """
        Write model metadata file (yaml file format).

        Parameters
        ----------
        metadata: dict
            Metadata to be stored
        yml_file: str
            Name of output file.
        keys_lower_case: bool
            Write yaml keys in lower case.

        Returns
        -------
        str
            Name of output file

        Raises
        ------
        FileNotFoundError
            If yml_file not found.
        TypeError
            If yml_file is not defined.
        """

        try:
            yml_file = Path(yml_file or self.product_data_file).with_suffix(".metadata.yml")
            with open(yml_file, "w", encoding="UTF-8") as file:
                yaml.safe_dump(
                    gen.change_dict_keys_case(metadata, keys_lower_case),
                    file,
                    sort_keys=False,
                )
            self._logger.info(f"Writing metadata to {yml_file}")
            return yml_file
        except FileNotFoundError:
            self._logger.error(f"Error writing model data to {yml_file}")
            raise
        except AttributeError:
            self._logger.error("No metadata defined for writing")
            raise
        except TypeError:
            self._logger.error("No output file for metadata defined")
            raise

    @staticmethod
    def _astropy_data_format(product_data_format):
        """
        Ensure conformance with astropy data format naming.

        Parameters
        ----------
        product_data_format: string
            format identifier

        """

        if product_data_format == "ecsv":
            product_data_format = "ascii.ecsv"
        return product_data_format
