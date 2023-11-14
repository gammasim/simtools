import datetime
import logging
import re

import simtools.utils.general as gen
from simtools.data_model import metadata_model


class SchemaValidator:
    """
    Validate a dictionary against the simpipe reference schema.
    Used e.g., to validate metadata provided as input.

    Parameters
    ----------
    data_dict: dict
        Metadata dict to be validated against reference schema.

    """

    def __init__(self, data_dict=None):
        """
        Initialize validation class and load reference schema.
        """

        self._logger = logging.getLogger(__name__)

        self._reference_schema = gen.change_dict_keys_case(
            metadata_model.metadata_input_reference_schema(), lower_case=True
        )
        self.data_dict = data_dict

    def validate_and_transform(self, meta_file_name=None):
        """
        Schema validation and processing.

        Parameters
        ----------
        meta_file_name
            file name for file with meta data to
            be validated (might also be given as
            dictionary during initialization of the class).

        Returns
        -------
        dict
            Complete set of metadata following the CTA top-level metadata defintion
            (None if meta_file_name is undefined)

        """
        try:
            self._logger.debug(f"Reading meta data from {meta_file_name}")
            self.data_dict = gen.collect_data_from_yaml_or_dict(meta_file_name, self.data_dict)
        except gen.InvalidConfigData:
            self._logger.debug("Failed reading metadata from file.")
            return None

        self.data_dict = gen.change_dict_keys_case(self.data_dict, True)
        self._validate_schema(self._reference_schema, self.data_dict)
        self._process_schema()
        return self.data_dict

    def _validate_schema(self, ref_schema, data_dict):
        """
        Validate schema for data types and required fields.

        Parameters
        ----------
        ref_schema: dict
            Reference metadata schema
        data_dict: dict
            input metadata dict to be validated against
            reference schema.

        Raises
        ------
        UnboundLocalError
            If no data is available for metadata key from the
            reference schema.

        """

        for key, value in ref_schema.items():
            if data_dict and key in data_dict:
                _this_data = data_dict[key]
            else:
                if self._field_is_optional(value):
                    self._logger.debug(f"Optional field {key}")
                    continue
                msg = f"Missing required field '{key}'"
                raise ValueError(msg)

            if isinstance(value, dict):
                # "type" is used for data types (str) and for telescope types (dict)
                if "type" in value and isinstance(value["type"], str):
                    try:
                        self._validate_data_type(value, key, _this_data)
                    except UnboundLocalError:
                        self._logger.error(f"No data for {key} key")
                        raise
                else:
                    self._validate_schema(value, _this_data)

    def _process_schema(self):
        """
        Process schema entries for inconsistencies
        (quite fine tuned)
        - remove linefeeds from description string

        Raises
        ------
        KeyError
            if data_dict["product"]["description"] is not available.

        """

        try:
            self.data_dict["product"]["description"] = self._remove_line_feed(
                self.data_dict["product"]["description"]
            )
        except KeyError:
            pass

    def _validate_data_type(self, schema, key, data_field):
        """
        Validate data type against the expected data type
        from schema.

        Parameters
        ----------
        schema: dict
            metadata description from reference schema.
        key: str
            data field name to be validated.
        data_field: dict
            data field to be validated.

        Raises
        ------
        ValueError
            if data types are inconsistent.

        """

        self._logger.debug(f"checking data field {key} for {schema['type']}")

        convert = {"str": type("str"), "float": type(1.0), "int": type(0), "bool": type(True)}

        if schema["type"] == "datetime":
            self._validate_datetime(data_field, self._field_is_optional(schema))
        elif schema["type"] == "email":
            self._validate_email(data_field, key)
        elif schema["type"].endswith("list"):
            self._validate_list(schema["type"], data_field)
        elif type(data_field).__name__ != schema["type"]:
            try:
                if isinstance(data_field, (int, str)):
                    convert[schema["type"]](data_field)
                elif data_field is not None:
                    raise ValueError
            except ValueError as error:
                raise ValueError(
                    f"invalid type for key {key}. Expected: {schema['type']}, "
                    f"Found: {type(data_field).__name__}"
                ) from error

    @staticmethod
    def _validate_datetime(data_field, optional_field=False):
        """
        Validate entry to be of type datetime and of
        format %Y-%m-%d %H:%M:%S.

        Parameters
        ----------
        data_field: dict
            data field to be validated
        optional_field: boolean
            data field is optional

        Raises
        ------
        ValueError
            if data field is of invalid format

        """
        format_date = "%Y-%m-%d %H:%M:%S"
        try:
            datetime.datetime.strptime(data_field, format_date)
        except (ValueError, TypeError) as error:
            if not optional_field:
                raise ValueError(
                    f"invalid date format. Expected {format_date}; Found {data_field}"
                ) from error

    @staticmethod
    def _validate_email(data_field, key):
        """
        Validate entry to be a email address

        Parameters
        ----------
        data_field: dict
            data field to be validated.
        key: str
            data field name to be validated.

        Raises
        ------
        ValueError
            if data field is of invalid format.

        """
        regex = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        if not re.fullmatch(regex, data_field):
            raise ValueError(f"invalid email format in field {key}: {data_field}")

    def _validate_list(self, schema_type, data_list):
        """
        Validate schmema for list type entry

        Parameters
        ----------
        schema_type
            reference schema type (e.g., instrument_list, document_list).
        data_list: list
            list of dictionaries to be validated.
        """

        _ref_schema = gen.change_dict_keys_case(
            metadata_model.metadata_input_reference_document_list(schema_type), lower_case=True
        )
        for entry in data_list:
            self._validate_schema(_ref_schema, entry)

    def _field_is_optional(self, field_dict):
        """
        Check if data field is labeled as optional in the reference metadata schema.
        Dictionaries as datafields are tested for any optional fields.

        Parameters
        ----------
        field_dict: dict
            required field from reference metadata schema

        Returns
        -------
        boolean
            True if data field is required

        Raises
        ------
        KeyError
            if 'required' field is not defined in reference
            metadata schema

        """
        try:
            if field_dict["required"]:
                return False
        except KeyError:
            if isinstance(field_dict, dict):
                for value in field_dict.values():
                    if isinstance(value, dict) and not self._field_is_optional(value):
                        return False
                return True
            return False
        return True

    @staticmethod
    def _remove_line_feed(string):
        """
        Remove all line feeds from a string

        Parameters
        ----------
        str
            input string

        Returns
        -------
        str
            with line feeds removed
        """

        return string.replace("\n", " ").replace("\r", "")
