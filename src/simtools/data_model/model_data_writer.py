"""Model data writer module."""

import logging
from pathlib import Path

import packaging.version
from astropy.io.registry.base import IORegistryError

import simtools.utils.general as gen
from simtools.data_model import schema, validate_data
from simtools.data_model.metadata_collector import MetadataCollector
from simtools.db import db_handler
from simtools.io import ascii_handler, io_handler
from simtools.utils import names, value_conversion

__all__ = ["ModelDataWriter"]


class ModelDataWriter:
    """
    Writer for simulation model data and metadata.

    Parameters
    ----------
    product_data_file: str
        Name of output file.
    product_data_format: str
        Format of output file.
    args_dict: Dictionary
        Dictionary with configuration parameters.
    output_path: str or Path
        Path to output file.
    args_dict: dict
        Dictionary with configuration parameters.

    """

    def __init__(
        self,
        product_data_file=None,
        product_data_format=None,
        output_path=None,
        args_dict=None,
    ):
        """Initialize model data writer."""
        self._logger = logging.getLogger(__name__)
        self.io_handler = io_handler.IOHandler()
        self.schema_dict = {}
        if args_dict is not None:
            output_path = args_dict.get("output_path", output_path)
        if output_path is not None:
            self.io_handler.set_paths(output_path=output_path)
        try:
            self.product_data_file = self.io_handler.get_output_file(file_name=product_data_file)
        except TypeError:
            self.product_data_file = None
        self.product_data_format = self._astropy_data_format(product_data_format)

    @staticmethod
    def dump(
        args_dict, output_file=None, metadata=None, product_data=None, validate_schema_file=None
    ):
        """
        Write model data and metadata (as static method).

        Parameters
        ----------
        args_dict: dict
            Dictionary with configuration parameters (including output file name and path).
        output_file: string or Path
            Name of output file (args["output_file"] is used if this parameter is not set).
        metadata: MetadataCollector object
            Metadata to be written.
        product_data: astropy Table
            Model data to be written
        validate_schema_file: str
            Schema file used in validation of output data.

        """
        writer = ModelDataWriter(
            product_data_file=(
                args_dict.get("output_file", None) if output_file is None else output_file
            ),
            product_data_format=args_dict.get("output_file_format", "ascii.ecsv"),
            args_dict=args_dict,
        )
        if validate_schema_file is not None and not args_dict.get("skip_output_validation", True):
            product_data = writer.validate_and_transform(
                product_data_table=product_data,
                validate_schema_file=validate_schema_file,
            )
        writer.write(metadata=metadata, product_data=product_data)

    @staticmethod
    def dump_model_parameter(
        parameter_name,
        value,
        instrument,
        parameter_version,
        output_file,
        output_path=None,
        metadata_input_dict=None,
        db_config=None,
        unit=None,
        meta_parameter=False,
    ):
        """
        Generate DB-style model parameter dict and write it to json file.

        Parameters
        ----------
        parameter_name: str
            Name of the parameter.
        value: any
            Value of the parameter.
        instrument: str
            Name of the instrument.
        parameter_version: str
            Version of the parameter.
        output_file: str
            Name of output file.
        output_path: str or Path
            Path to output file.
        metadata_input_dict: dict
            Input to metadata collector.
        db_config: dict
            Database configuration. If not None, check if parameter with the same version exists.
        unit: str
            Unit of the parameter value (if applicable and value is not of type astropy Quantity).

        Returns
        -------
        dict
            Validated parameter dictionary.
        """
        writer = ModelDataWriter(
            product_data_file=output_file,
            product_data_format="json",
            args_dict=None,
            output_path=output_path,
        )
        if db_config is not None:
            writer.check_db_for_existing_parameter(
                parameter_name, instrument, parameter_version, db_config
            )

        unique_id = None
        if metadata_input_dict is not None:
            metadata_input_dict["output_file"] = output_file
            metadata_input_dict["output_file_format"] = Path(output_file).suffix.lstrip(".")
            metadata = MetadataCollector(args_dict=metadata_input_dict)
            metadata.write(output_path / Path(output_file))
            unique_id = (
                metadata.get_top_level_metadata().get("cta", {}).get("product", {}).get("id")
            )

        _json_dict = writer.get_validated_parameter_dict(
            parameter_name,
            value,
            instrument,
            parameter_version,
            unique_id,
            unit=unit,
            meta_parameter=meta_parameter,
        )
        writer.write_dict_to_model_parameter_json(output_file, _json_dict)
        return _json_dict

    def check_db_for_existing_parameter(
        self, parameter_name, instrument, parameter_version, db_config
    ):
        """
        Check if a parameter with the same version exists in the simulation model database.

        Parameters
        ----------
        parameter_name: str
            Name of the parameter.
        instrument: str
            Name of the instrument.
        parameter_version: str
            Version of the parameter.
        db_config: dict
            Database configuration.

        Raises
        ------
        ValueError
            If parameter with the same version exists in the database.
        """
        db = db_handler.DatabaseHandler(mongo_db_config=db_config)
        try:
            db.get_model_parameter(
                parameter=parameter_name,
                parameter_version=parameter_version,
                site=names.get_site_from_array_element_name(instrument),
                array_element_name=instrument,
            )
        except ValueError:
            pass  # parameter does not exist - expected behavior
        else:
            raise ValueError(
                f"Parameter {parameter_name} with version {parameter_version} already exists."
            )

    def get_validated_parameter_dict(
        self,
        parameter_name,
        value,
        instrument,
        parameter_version,
        unique_id=None,
        schema_version=None,
        unit=None,
        meta_parameter=False,
    ):
        """
        Get validated parameter dictionary.

        Parameters
        ----------
        parameter_name: str
            Name of the parameter.
        value: any
            Value of the parameter.
        instrument: str
            Name of the instrument.
        parameter_version: str
            Version of the parameter.
        schema_version: str
            Version of the schema.
        unique_id: str
            Unique ID of the parameter set (from metadata).
        unit: str
            Unit of the parameter value (if applicable and value is not an astropy Quantity).
        meta_parameter: bool
            Setting for meta parameter flag.

        Returns
        -------
        dict
            Validated parameter dictionary.
        """
        self._logger.debug(f"Getting validated parameter dictionary for {instrument}")
        self.schema_dict, schema_file = self._read_schema_dict(parameter_name, schema_version)

        if unit is None:
            value, unit = value_conversion.split_value_and_unit(value)

        data_dict = {
            "schema_version": schema.get_model_parameter_schema_version(schema_version),
            "parameter": parameter_name,
            "instrument": instrument,
            "site": names.get_site_from_array_element_name(instrument),
            "parameter_version": parameter_version,
            "unique_id": unique_id,
            "value": value,
            "unit": unit,
            "type": self._get_parameter_type(),
            "file": self._parameter_is_a_file(),
            "meta_parameter": meta_parameter,
            "model_parameter_schema_version": self.schema_dict.get("schema_version", "0.1.0"),
        }
        return self.validate_and_transform(
            product_data_dict=data_dict,
            validate_schema_file=schema_file,
            is_model_parameter=True,
        )

    def _read_schema_dict(self, parameter_name, schema_version):
        """
        Read schema dict for given parameter name and version.

        Use newest schema version if schema_version is None.

        Parameters
        ----------
        parameter_name: str
            Name of the parameter.
        schema_version: str
            Schema version.

        Returns
        -------
        dict
            Schema dictionary.
        """
        schema_file = schema.get_model_parameter_schema_file(parameter_name)
        schemas = ascii_handler.collect_data_from_file(schema_file)
        if isinstance(schemas, list):
            if schema_version is None:
                return self._find_highest_schema_version(schemas), schema_file
            for entry in schemas:
                if entry.get("schema_version") == schema_version:
                    return entry, schema_file
        else:
            return schemas, schema_file

        raise ValueError(f"Schema version {schema_version} not found for {parameter_name}")

    def _find_highest_schema_version(self, schema_list):
        """
        Find entry with highest schema_version in a list of schema dicts.

        Parameters
        ----------
        schema_list: list
            List of schema dictionaries.

        Returns
        -------
        dict
            Schema dictionary with highest schema_version.
        """
        try:
            valid_entries = [entry for entry in schema_list if "schema_version" in entry]
        except TypeError as exc:
            raise TypeError("No valid schema versions found in the list.") from exc
        return max(valid_entries, key=lambda e: packaging.version.Version(e["schema_version"]))

    def _get_parameter_type(self):
        """
        Return parameter type from schema.

        Reduce list of types to single type if all types are the same.

        Returns
        -------
        str or list[str]
            Parameter type
        """
        _parameter_type = [data["type"] for data in self.schema_dict["data"]]
        return (
            _parameter_type[0]
            if all(t == _parameter_type[0] for t in _parameter_type)
            else _parameter_type
        )

    def _parameter_is_a_file(self):
        """
        Check if parameter is a file.

        Returns
        -------
        bool
            True if parameter is a file.

        """
        try:
            return self.schema_dict["data"][0]["type"] == "file"
        except (KeyError, IndexError):
            pass
        return False

    def _get_unit_from_schema(self):
        """
        Return unit(s) from schema dict.

        Returns
        -------
        str or list
            Parameter unit(s)
        """
        try:
            unit_list = []
            for data in self.schema_dict["data"]:
                unit_list.append(data["unit"] if data["unit"] != "dimensionless" else None)
            return unit_list if len(unit_list) > 1 else unit_list[0]
        except (KeyError, IndexError):
            pass
        return None

    def validate_and_transform(
        self,
        product_data_table=None,
        product_data_dict=None,
        validate_schema_file=None,
        is_model_parameter=False,
    ):
        """
        Validate product data using jsonschema given in metadata.

        If necessary, transform product data to match schema.

        Parameters
        ----------
        product_data_table: astropy Table
            Model data to be validated.
        product_data_dict: dict
            Model data to be validated.
        validate_schema_file: str
            Schema file used in validation of output data.
        is_model_parameter: bool
            True if data describes a model parameter.
        """
        _validator = validate_data.DataValidator(
            schema_file=validate_schema_file,
            data_table=product_data_table,
            data_dict=product_data_dict,
            check_exact_data_type=False,
        )
        return _validator.validate_and_transform(is_model_parameter)

    def write(self, product_data=None, metadata=None):
        """
        Write model data and metadata.

        Parameters
        ----------
        product_data: astropy Table
            Model data to be written
        metadata: MetadataCollector object
            Metadata to be written.

        Raises
        ------
        FileNotFoundError
            if data writing was not successful.

        """
        if product_data is None:
            return

        if metadata is not None:
            product_data.meta.update(
                gen.change_dict_keys_case(metadata.get_top_level_metadata(), True)
            )

        self._logger.info(f"Writing data to {self.product_data_file}")
        if isinstance(product_data, dict) and Path(self.product_data_file).suffix == ".json":
            self.write_dict_to_model_parameter_json(self.product_data_file, product_data)
            return
        try:
            product_data.write(
                self.product_data_file, format=self.product_data_format, overwrite=True
            )
        except IORegistryError:
            self._logger.error(f"Error writing model data to {self.product_data_file}.")
            raise
        if metadata is not None:
            metadata.write(self.product_data_file, add_activity_name=True)

    def write_dict_to_model_parameter_json(self, file_name, data_dict):
        """
        Write dictionary to model-parameter-style json file.

        Parameters
        ----------
        file_name : str
            Name of output file.
        data_dict : dict
            Data dictionary.

        Raises
        ------
        FileNotFoundError
            if data writing was not successful.
        """
        data_dict = ModelDataWriter.prepare_data_dict_for_writing(data_dict)
        self._logger.info(f"Writing data to {self.io_handler.get_output_file(file_name)}")
        ascii_handler.write_data_to_file(
            data=data_dict,
            output_file=self.io_handler.get_output_file(file_name),
            sort_keys=True,
            numpy_types=True,
        )

    @staticmethod
    def prepare_data_dict_for_writing(data_dict):
        """
        Prepare data dictionary for writing to json file.

        Ensure sim_telarray style lists as strings 'type' and 'unit' entries.
        Replace "None" with "null" for unit field.
        Replace list of equal units with single unit string.

        Parameters
        ----------
        data_dict: dict
            Dictionary with lists.

        Returns
        -------
        dict
            Dictionary with lists converted to strings.

        """
        try:
            if isinstance(data_dict["unit"], str):
                data_dict["unit"] = data_dict["unit"].replace("None", "null")
            elif isinstance(data_dict["unit"], list):
                data_dict["unit"] = [
                    unit.replace("None", "null") if isinstance(unit, str) else unit
                    for unit in data_dict["unit"]
                ]
                if all(u == data_dict["unit"][0] for u in data_dict["unit"]):
                    data_dict["unit"] = data_dict["unit"][0]
        except KeyError:
            pass
        return data_dict

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
