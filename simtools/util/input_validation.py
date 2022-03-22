import datetime
import logging
import re

import simtools.util.general as gen

class SchemaValidator:
    """
    Validate a dictionary against a reference schema

    """

    def __init__(self, reference_schema_file, data_dict):
        """
        Initalize validation class and read required
        reference schema

        Parameters:
        -----------
        reference_schema: dict
            reference schema
        data_dict: dict
            dictionary to be validated

        """

        self._logger = logging.getLogger(__name__)

        if reference_schema_file:
            self._reference_schema = gen.collectDataFromYamlOrDict(
                reference_schema_file, None)
        self.data_dict = data_dict

    def validate(self):
        """
        Schema validation

        """
        self._validate_schema(
            self._reference_schema,
            self.data_dict)

    def _validate_data_type(self, schema, key, data_field):
        """
        Validate data type against the expected data type
        from schema

        Raises value error if data types are inconsistent
        """
        self._logger.debug("checking data field %s for %s",
                           key, schema['type'])

        if schema['type'] == 'datetime':
            format_date = "%Y-%m-%d %H:%M:%S"
            try:
                datetime.datetime.strptime(data_field, format_date)
            except (ValueError, TypeError) as error:
                raise ValueError(
                    f'invalid date format. Expected {format_date}; Found {data_field}') from error

        elif schema['type'] == 'email':
            regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            if not re.fullmatch(regex, data_field):
                raise ValueError(
                    f'invalid email format in field {key}: {data_field}')

        elif type(data_field).__name__ != schema['type']:
            raise ValueError(
                'invalid data type for key {}. Expected: {}, Found: {}'.format(
                    key, schema['type'], type(data_field).__name__))

    def _check_if_field_is_optional(self, key, value):
        """
        Check if data field is labeled as not required in
        the reference schema

        Assume that any field which is not label as required
        is optional

        Raises value error if required field is missing

        """

        if isinstance(value, dict) \
                and 'required' in value \
                and value['required']:
            raise ValueError(
                f'required data field {key} not found')

        self._logger.debug("checking optional key %s", key)

    def _validate_schema(
            self,
            ref_schema,
            data_dict):
        """
        Validate schema for data types and required fields

        """

        for key, value in ref_schema.items():
            if data_dict and key in data_dict:
                _this_data = data_dict[key]
            else:
                self._check_if_field_is_optional(key, value)
                continue
            if isinstance(value, dict):
                if 'type' in value:
                    self._validate_data_type(value, key, _this_data)
                else:
                    self._validate_schema(value, _this_data)
