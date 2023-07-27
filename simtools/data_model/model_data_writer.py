import logging
from pathlib import Path

import astropy
import yaml

import simtools.util.general as gen
from simtools import io_handler

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
        self.product_data_file = self.io_handler.get_output_file(file_name=product_data_file)
        self.product_data_format = self._astropy_data_format(product_data_format)

    def write(self, metadata=None, product_data=None):
        """
        Write model data and metadata

        Parameters:
        -----------
        metadata: dict
            Metadata to be written.
        product_data: astropy Table
            Model data to be written

        """

        self.write_metadata(metadata=metadata)
        self.write_data(product_data=product_data)

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
        TODO - check if this error makes sense
        """

        try:
            if product_data is not None:
                self._logger.info(f"Writing data to {self.product_data_file}")
                product_data.write(
                    self.product_data_file,
                    format=self.product_data_format,
                    overwrite=True
                )
        except astropy.io.registry.base.IORegistryError:
            self._logger.error("Error writing model data to {self.product_data_file}.")
            raise

    def write_metadata(self, metadata, ymlfile=None, keys_lower_case=False):
        """
        Write model metadata file (yaml file format).

        Parameters
        ----------
        metadata: dict
            Metadata to be stored
        ymlfile: str
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
            If ymlfile not found.
        AttributeError
            If no metadata defined for writing.
        TypeError
            If ymlfile is not defined.
        """

        try:
            ymlfile = Path(ymlfile or self.product_data_file).with_suffix('.metadata.yml')
            self._logger.info(f"Writing metadata to {ymlfile}")
            with open(ymlfile, "w", encoding="UTF-8") as file:
                yaml.safe_dump(
                    gen.change_dict_keys_case(metadata, keys_lower_case),
                    file,
                    sort_keys=False,
                )
            return ymlfile
        except FileNotFoundError:
            self._logger.error(f"Error writing model data to {ymlfile}")
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
