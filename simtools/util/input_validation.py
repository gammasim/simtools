import datetime
import logging

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

        if schema['type'] == 'datetime':
            format = "%Y-%m-%d %H:%M:%S"
            print('AAAA', data_field)
            try:
                datetime.datetime.strptime(data_field, format)
            except:
                raise ValueError(
                    'invalid date format. Expected {}; Found {}'.format(
                        format, data_field))
            self._logger.debug('checking {} for datetime'.format(key))

        # TODO
        self._logger.debug('data type checking to be fixed {}'.format(schema['type']))
        return None

        if not str(type(data_field)) is schema['type']:
            raise ValueError(
                'invalid data type for key {}. Expected: {}, Found: {}'.format(
                    key, schema['type'], type(data_field)))

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
                'required data field {} not found')

        self._logger.debug(
            'checking optional key {}'.format(key))

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
                return None
            if isinstance(value, dict) and 'type' in value:
                self._validate_data_type(
                    value, key, _this_data)
            else:
                self._validate_schema(value, _this_data)
