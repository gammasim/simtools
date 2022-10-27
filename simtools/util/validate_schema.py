import datetime
import logging
import re

import simtools.util.data_model as data_model
import simtools.util.general as gen


class SchemaValidator:
    """
    Validate a dictionary against a reference schema.

    Attributes
    ----------
    schema_file: str
        file nome for user input schema
    data_dict: dict
        User-provided metadata dict to be validated against
        reference schema

    Methods
    -------
    validate_and_transform(user_meta_file_name=None)
        validate user meta data

    """

    def __init__(self, data_dict=None):
        """
        Initalize validation class and read required
        reference schema

        Parameters
        ----------
        data_dict: dict
            User-provided metadata dict to be validated against
            reference schema

        """

        self._logger = logging.getLogger(__name__)

        self._reference_schema = gen.change_dict_keys_case(
            data_model.user_input_reference_schema(), True
        )
        self.data_dict = data_dict

    def validate_and_transform(self, user_meta_file_name=None, lower_case=False):
        """
        Schema validation and processing

        Parameters
        ----------
        user_meta_file_name
            file name for file with user meta data to
            be validated (might also be given as
            dictionary during initialization of the class)
        lower_case: bool
            compare schema keys in lower case only (gammasim-tools convention).

        Returns
        -------
        dict
            Complete set of metadata following the CTA top-level metadata defintion

        """
        if user_meta_file_name:
            self._logger.debug("Reading user meta data from {}".format(user_meta_file_name))
            self.data_dict = gen.collectDataFromYamlOrDict(user_meta_file_name, None)

        if lower_case:
            self._validate_schema(
                self._reference_schema, gen.change_dict_keys_case(self.data_dict, True)
            )
        else:
            self._validate_schema(self._reference_schema, self.data_dict)

        self._process_schema()

        return self.data_dict

    def _validate_schema(self, ref_schema, data_dict):
        """
        Validate schema for data types and required fields

        Parameters
        ----------
        ref_schema: dict
            Reference metadata schema
        data_dict: dict
            User-provided metadata dict to be validated against
            reference schema

        Raises
        ------
        UnboundLocalError
            If no data is available for metadata key from the
            reference schema

        """

        for key, value in ref_schema.items():
            if data_dict and key in data_dict:
                _this_data = data_dict[key]
            else:
                if self._field_is_optional(value):
                    self._logger.debug(f"Optional field {key}")
                    continue
                else:
                    msg = f"Missing required field {key}"
                    raise ValueError(msg)

            if isinstance(value, dict):
                if "type" in value:
                    try:
                        self._validate_data_type(value, key, _this_data)
                    except UnboundLocalError:
                        self._logger.error(f"No data for `{key}` key")
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
            if data_dict['product']['description'] is not available

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
        from schema

        Parameters
        ----------
        schema: dict
            metadata description from reference schema
        key: str
            data field name to be validated
        data_field: dict
            data field to be validated

        Raises
        ------
        ValueError
            if data types are inconsistent

        """

        self._logger.debug("checking data field {} for {}".format(key, schema["type"]))

        convert = {"str": type("str"), "float": type(1.0), "int": type(0), "bool": type(True)}

        if schema["type"] == "datetime":
            self._validate_datetime(data_field, self._field_is_optional(schema))

        elif schema["type"] == "email":
            self._validate_email(data_field, key)

        elif schema["type"] == "instrumentlist":
            self._validate_instrument_list(data_field)

        elif type(data_field).__name__ != schema["type"]:
            try:
                if isinstance(data_field, (int, str)):
                    convert[schema["type"]](data_field)
                else:
                    raise ValueError
            except ValueError as error:
                raise ValueError(
                    "invalid type for key {}. Expected: {}, Found: {}".format(
                        key, schema["type"], type(data_field).__name__
                    )
                ) from error

    @staticmethod
    def _validate_datetime(data_field, optional_field=False):
        """
        Validate entry to be of type datetime and of
        format %Y-%m-%d %H:%M:%S

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
                    "invalid date format. Expected {}; Found {}".format(format_date, data_field)
                ) from error

    @staticmethod
    def _validate_email(data_field, key):
        """
        Validate entry to be a email address

        Parameters
        ----------
        data_field: dict
            data field to be validated
        key: str
            data field name to be validated

        Raises
        ------
        ValueError
            if data field is of invalid format

        """
        regex = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        if not re.fullmatch(regex, data_field):
            raise ValueError("invalid email format in field {}: {}".format(key, data_field))

    def _validate_instrument_list(self, instrument_list):
        """
        Validate entry to be of type INSTRUMENT

        Parameters
        ----------
        instrument list: list
            list of dictionaries of type INSTRUMENT
            to be validated

        """

        for instrument in instrument_list:
            self._validate_schema(self._reference_schema["instrument"], instrument)

    @staticmethod
    def _field_is_optional(value):
        """
        Check if data field is labeled as not required in
        the reference metadata schema

        Parameters
        ----------
        value: dict
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
            if value["required"]:
                return False
            else:
                return True
        except KeyError:
            return False

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
