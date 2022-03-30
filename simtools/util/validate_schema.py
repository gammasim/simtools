import datetime
import logging
import re

import simtools.util.general as gen


class SchemaValidator:
    """
    Validate a dictionary against a reference schema

    Attributes
    ----------
    workflow_config: dict
        workflow configuration
    data_dict: dict
        data dictionary to be validated agains reference schema

    Methods
    -------
    validate(user_meta_file_name=None)
        validate user meta data

    """

    def __init__(self, workflow_config=None, data_dict=None):
        """
        Initalize validation class and read required
        reference schema

        """

        self._logger = logging.getLogger(__name__)

        reference_schema_file = None
        if workflow_config:
            reference_schema_file = self._read_reference_schema_file(
                workflow_config)
        if reference_schema_file:
            self._logger.debug(
                "Reading reference schema from {}".format(
                    reference_schema_file))
            self._reference_schema = gen.collectDataFromYamlOrDict(
                reference_schema_file, None)

        self.data_dict = data_dict

    def validate_and_transform(self, user_meta_file_name=None):
        """
        Schema validation and processing

        Parameters
        ----------
        user_meta_file_name
            file name with meta data to be validated
            (might also be given as dictionary during
            initialization of the class)

        """
        if user_meta_file_name:
            self._logger.debug(
                 "Reading user meta data from {}".format(
                     user_meta_file_name))
            self.data_dict = gen.collectDataFromYamlOrDict(
                    user_meta_file_name, None)

        self._validate_schema(
            self._reference_schema,
            self.data_dict)

        self._process_schema()

        return self.data_dict

    def _validate_schema(self, ref_schema, data_dict):
        """
        Validate schema for data types and required fields

        """

        for key, value in ref_schema.items():
            if data_dict and key in data_dict:
                _this_data = data_dict[key]
            else:
                if self._field_is_optional(value):
                    self._logger.debug(f"Optional field {key}")
                    continue

            if isinstance(value, dict):
                try:
                    if 'type' in value:
                        self._validate_data_type(value, key, _this_data)
                    else:
                        self._validate_schema(value, _this_data)
                except UnboundLocalError:
                    self._logger.error(
                        f"No data for `{key}` key")
                    raise

    def _process_schema(self):
        """
        Process schema entries for inconsistencies
        (quite fine tuned)

        """

        try:
            self.data_dict['PRODUCT']['DESCRIPTION'] = \
                self._remove_line_feed(
                    self.data_dict['PRODUCT']['DESCRIPTION'])
        except KeyError:
            pass

    def _remove_line_feed(self, string):
        """
        Remove all line feed from a string
        """

        return string.replace('\n', ' ').replace('\r', '')

    def _validate_data_type(self, schema, key, data_field):
        """
        Validate data type against the expected data type
        from schema

        Raises value error if data types are inconsistent
        """
        self._logger.debug("checking data field {} for {}".format(
                           key, schema['type']))

        convert = {'str': type('str'), 'float': type(1.0),
                   'int': type(0), 'bool': type(True)}

        if schema['type'] == 'datetime':
            format_date = "%Y-%m-%d %H:%M:%S"
            try:
                datetime.datetime.strptime(data_field, format_date)
            except (ValueError, TypeError) as error:
                if not self._field_is_optional(schema):
                    raise ValueError(
                        'invalid date format. Expected {}; Found {}'.format(
                            format_date, data_field)) from error

        elif schema['type'] == 'email':
            regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            if not re.fullmatch(regex, data_field):
                raise ValueError(
                    'invalid email format in field {}: {}'.format(
                        key, data_field))

        elif schema['type'] == 'instrumentlist':
            self._validate_instrument_list(data_field)

        elif type(data_field).__name__ != schema['type']:
            try:
                if isinstance(data_field, (int, str)):
                    convert[schema['type']](data_field)
                else:
                    raise ValueError
            except ValueError as error:
                raise ValueError(
                    'invalid type for key {}. Expected: {}, Found: {}'.format(
                        key, schema['type'],
                        type(data_field).__name__)) from error

    def _validate_instrument_list(self, instrument_list):
        """
        Validate entry to be of type INSTRUMENT

        """

        for instrument in instrument_list:
            self._validate_schema(
                self._reference_schema['INSTRUMENT'],
                instrument)

    @staticmethod
    def _field_is_optional(value):
        """
        Check if data field is labeled as not required in
        the reference schema

        """

        try:
            if value['required']:
                return False
            return True
        except KeyError:
            return False

    def _read_reference_schema_file(self, workflow_config):
        """
        Return full path and name of reference schema file

        """

        try:
            return str(
                workflow_config['CTASIMPIPE']['DATAMODEL']['SCHEMADIRECTORY'] +
                '/' +
                workflow_config["CTASIMPIPE"]["DATAMODEL"]["USERINPUTSCHEMA"])
        except KeyError:
            self._logger.error(
                "Missing description of DATAMODEL schema file")
            raise
